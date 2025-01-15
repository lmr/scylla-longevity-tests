import logging
import random
import traceback
from functools import cached_property


import yaml
from deepdiff import DeepDiff

from sdcm.sct_events import Severity
from sdcm.sct_events.system import PerftuneResultEvent
from sdcm.utils.version_utils import get_systemd_version


# https://docs.scylladb.com/stable/operating-scylla/admin-tools/perftune.html


PERFTUNE_LOCATION = "/opt/scylladb/scripts/perftune.py"
TEMP_PERFTUNE_YAML_PATH = "/tmp/perftune.yaml"
PERFTUNE_EXPECTED_RESULTS_PATH = "defaults/perftune_results.json"


def get_machine_architecture_type(node):
    result = node.remoter.run("uname -m")
    return result.stdout.strip()


def get_number_of_cpu_cores(node) -> int:
    result = node.remoter.run("grep -c ^processor /proc/cpuinfo")
    cores_num = int(result.stdout)
    return cores_num


class PerftuneExpectedResult:
    def __init__(self, number_of_cpu_cores, nic_name, architecture):
        self.nic_name = nic_name
        self.total_cpu_bits = number_of_cpu_cores
        self.architecture = architecture
        self.expected_compute_bits: int = 0

        match number_of_cpu_cores:
            case 1 | 2 | 4:
                self.expected_irq_bits: int = number_of_cpu_cores
                self.expected_compute_bits: int = number_of_cpu_cores
            case 8:
                self.expected_irq_bits: int = 1
            case 12 | 24 | 32:
                self.expected_irq_bits: int = 2
            case 16:
                if self.architecture == "aarch64":
                    self.expected_irq_bits: int = 1
                else:
                    self.expected_irq_bits: int = 2
            case 48 | 64:
                self.expected_irq_bits: int = 4
            case 80 | 72 | 96 | 128:
                self.expected_irq_bits: int = 8

        if self.expected_compute_bits == 0:
            self.expected_compute_bits: int = number_of_cpu_cores - self.expected_irq_bits

    @staticmethod
    def count_bits(mask_string) -> int:
        return sum([int(m, base=16).bit_count() for m in mask_string.split(',')])

    def get_expected_options_file_contents(self) -> dict:
        return {
            "nic": [self.nic_name],
            "irq_core_auto_detection_ratio": 16,
            "tune": ["net"]
        }


class PerftuneExecutor:
    def __init__(self, node, nic_name):
        self.node = node
        self.nic_name = nic_name

    def get_cpu_mask(self) -> str:
        result = self.node.remoter.run(
            f"{self.node.add_install_prefix(PERFTUNE_LOCATION)} --tune net --nic {self.nic_name} --get-cpu-mask")
        return result.stdout.strip()

    def get_irq_cpu_mask(self) -> str:
        result = self.node.remoter.run(
            f"{self.node.add_install_prefix(PERFTUNE_LOCATION)} --tune net --nic {self.nic_name} --get-irq-cpu-mask")
        return result.stdout.strip()

    def get_default_perftune_config(self) -> dict:
        return yaml.safe_load(self.node.get_perftune_yaml())

    def get_options_file_contents(self, use_temp_file=False, override_irq_cpu_mask="") -> dict:
        cmd = f"{self.node.add_install_prefix(PERFTUNE_LOCATION)} --tune net --nic {self.nic_name} --dump-options-file"
        if use_temp_file:
            cmd += f" --options-file {TEMP_PERFTUNE_YAML_PATH}"
        if override_irq_cpu_mask:
            cmd += f" --irq-cpu-mask {override_irq_cpu_mask}"
        result = self.node.remoter.run(cmd)
        result_as_yaml = yaml.safe_load(result.stdout)
        return result_as_yaml

    def create_temp_perftune_yaml(self, yaml_dict) -> None:
        self.node.remoter.run(f"touch {TEMP_PERFTUNE_YAML_PATH}")
        with self.node._remote_yaml(path=TEMP_PERFTUNE_YAML_PATH, sudo=False) as temp_yaml:  # pylint: disable=protected-access
            temp_yaml.update(yaml_dict)


class PerftuneOutputChecker:  # pylint: disable=too-few-public-methods
    def __init__(self, node):
        self.log = logging.getLogger(self.__class__.__name__)
        self.node = node
        nic_name = node.get_nic_devices()[0]
        self.executor = PerftuneExecutor(node, nic_name)
        self.expected_result = PerftuneExpectedResult(number_of_cpu_cores=get_number_of_cpu_cores(node),
                                                      nic_name=nic_name,
                                                      architecture=get_machine_architecture_type(node))

    @cached_property
    def systemd_version(self):
        systemd_version = 0
        try:
            systemd_version = get_systemd_version(self.node.remoter.run(
                "systemctl --version", ignore_status=True).stdout)
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            self.log.warning("failed to get systemd version:", exc_info=True)
        return systemd_version

    def compare_cpu_mask(self) -> None:
        cpu_mask = self.executor.get_cpu_mask()
        cpu_mask_bits = self.expected_result.count_bits(cpu_mask)
        if cpu_mask_bits != self.expected_result.expected_compute_bits:
            PerftuneResultEvent(
                message=f"Mismatched results when testing the output of the 'get-cpu-mask' command on {self.node}"
                        f"\nActual result: '{cpu_mask}' / {cpu_mask_bits} bits"
                        f"\nExpected output: {self.expected_result.expected_compute_bits} bits",
                severity=Severity.ERROR).publish()

    def compare_irq_cpu_mask(self) -> None:
        irq_cpu_mask = self.executor.get_irq_cpu_mask()
        irq_cpu_bits = self.expected_result.count_bits(irq_cpu_mask)
        if irq_cpu_bits != self.expected_result.expected_irq_bits:
            PerftuneResultEvent(
                message=f"Mismatched results when testing the output of the 'get-irq-cpu-mask' command on "
                        f"{self.node}"
                        f"\nActual result: '{irq_cpu_mask}' / {irq_cpu_bits} bits"
                        f"\nExpected output: {self.expected_result.expected_irq_bits} bits",
                severity=Severity.ERROR).publish()

    def compare_default_option_file(self, option_file_dict) -> None:
        cpu_mask_bits = self.expected_result.count_bits(option_file_dict["cpu_mask"])
        irq_cpu_bits = self.expected_result.count_bits(option_file_dict["irq_cpu_mask"])

        if cpu_mask_bits != self.expected_result.total_cpu_bits:
            PerftuneResultEvent(
                message=f"Mismatched results when testing the output of `cpu_mask` from the 'dump-options-file' command on "
                        f"{self.node}"
                        f"\nActual result: '{option_file_dict['cpu_mask']}' / {cpu_mask_bits} bits"
                        f"\nExpected output: {self.expected_result.total_cpu_bits} bits",
                severity=Severity.ERROR).publish()

        if irq_cpu_bits != self.expected_result.expected_irq_bits:
            PerftuneResultEvent(
                message=f"Mismatched results when testing the output of the 'get-irq-cpu-mask' command on "
                        f"{self.node}"
                        f"\nActual result: '{option_file_dict['irq_cpu_mask']}' / {irq_cpu_bits} bits"
                        f"\nExpected output: {self.expected_result.expected_irq_bits} bits",
                severity=Severity.ERROR).publish()

        if option_file_dict.items() <= self.expected_result.get_expected_options_file_contents().items():
            PerftuneResultEvent(
                message=f"Mismatched results when testing the output of the 'dump-options-file' command on "
                        f"{self.node}"
                        f"\nActual result: '{option_file_dict}'"
                        f"\nExpected output: '{self.expected_result.get_expected_options_file_contents()}'",
                severity=Severity.ERROR).publish()

    def compare_option_file_yaml_with_temp_yaml_copy(self, option_file_dict) -> None:
        temp_perftune_yaml_content_dict = self.executor.get_options_file_contents(use_temp_file=True)
        if diff := DeepDiff(temp_perftune_yaml_content_dict, option_file_dict, ignore_order=True):
            PerftuneResultEvent(
                message=f"Mismatched results when comparing the output of the 'dump-options-file' command to "
                        f"the content of the generated perftune.yaml file on {self.node}"
                        f"\nActual result: '{temp_perftune_yaml_content_dict}'"
                        f"\nExpected output: '{option_file_dict}'"
                        f"\nDiff: {diff}",
                severity=Severity.ERROR).publish()

    @staticmethod
    def _generate_new_irq_cpu_mask_string(mask_int_values) -> str:
        new_masks = []
        for mask_value in mask_int_values:
            random_irq_int_value = random.randint(0, mask_value)
            # The irq_cpu_mask cannot be greater than the cpu_mask, since it is a part of it
            new_mask_string = "{0:x}".format(random_irq_int_value)  # converting back to base 16
            padded_new_mask_string = f"0x{new_mask_string.rjust(8, '0')}"  # Padding
            new_masks.append(padded_new_mask_string)
        return ",".join(new_masks)

    @staticmethod
    def get_mask_int_values(complete_mask_string) -> list:
        split_masks = complete_mask_string.split(",")
        numerical_values = []
        for mask_string in split_masks:
            num_value = int(mask_string, base=16)
            numerical_values.append(num_value)
        return numerical_values

    def compare_option_file_with_overridden_irq_cpu_mask_param(self, option_file_dict) -> None:
        current_cpu_mask = option_file_dict["cpu_mask"]
        cpu_mask_int_values = self.get_mask_int_values(current_cpu_mask)
        random_irq_cpu_mask_string = self._generate_new_irq_cpu_mask_string(cpu_mask_int_values)
        altered_option_file_contents = self.executor.get_options_file_contents(
            override_irq_cpu_mask=random_irq_cpu_mask_string)
        if altered_option_file_contents["irq_cpu_mask"] != random_irq_cpu_mask_string:
            PerftuneResultEvent(
                message=f"Despite overriding the irq_cpu_mask param, its value is was not altered properly in the "
                        f"output of the 'dump-options-file' command on node {self.node}",
                severity=Severity.ERROR).publish()

    def compare_perftune_results(self) -> None:
        PerftuneResultEvent(
            message="Checking the output of perftune.py",
            severity=Severity.NORMAL).publish()
        try:
            self.compare_cpu_mask()
            self.compare_irq_cpu_mask()
            if not self.node.is_nonroot_install:
                # The following checks are not applicable for non-root installations,
                # since the scylla_prepare isn't called in them
                option_file_dict = self.executor.get_default_perftune_config()
                self.compare_default_option_file(option_file_dict)
                self.executor.create_temp_perftune_yaml(yaml_dict=option_file_dict)
                self.compare_option_file_yaml_with_temp_yaml_copy(option_file_dict)
                self.compare_option_file_with_overridden_irq_cpu_mask_param(option_file_dict)
        except Exception as error:  # pylint: disable=broad-except  # noqa: BLE001
            PerftuneResultEvent(
                message=f"Unexpected error when verifying the output of Perftune: {error}",
                severity=Severity.ERROR, trace=traceback.format_exc()).publish()
