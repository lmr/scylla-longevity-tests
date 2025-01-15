import datetime
import json
import logging
import random
from pathlib import Path

from sdcm.paths import SCYLLA_YAML_PATH
from sdcm.utils.version_utils import ComparableScyllaVersion
from sdcm.exceptions import SstablesNotFound


class NonDeletedTombstonesFound(Exception):
    pass


class SstableUtils:
    """
    Provides table details, related to its sstables and tombstones.
    Example: get a list of sstables for this table-name, count its tombstones etc.

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, propagation_delay_in_seconds: int = 0, ks_cf: str = None,
                 db_node: 'BaseNode' = None,  # noqa: F821
                 **kwargs):
        self.db_node = db_node
        self.db_cluster = self.db_node.parent_cluster if self.db_node else None
        self.ks_cf = ks_cf or random.choice(self.db_cluster.get_non_system_ks_cf_list(self.db_cluster.data_nodes[0]))
        self.keyspace, self.table = self.ks_cf.split('.')
        self.propagation_delay_in_seconds = propagation_delay_in_seconds
        self.log = logging.getLogger(self.__class__.__name__)
        self.user = kwargs.get("user", None)
        self.password = kwargs.get("password", None)

    def count_tombstones(self):
        sstables = self.get_sstables()
        tombstones_num = 0
        for sstable in sstables:
            tombstones_num += self.count_sstable_tombstones(sstable=sstable)
        self.log.debug('Got %s tombstones for %s', tombstones_num, self.ks_cf)
        return tombstones_num

    def get_sstables(self, from_minutes_ago: int = 0):
        selected_sstables = []
        ks_cf_path = self.ks_cf.replace('.', '/')
        find_cmd = f"find /var/lib/scylla/data/{ks_cf_path}-*/*-big-Data.db -maxdepth 1 -type f"
        if from_minutes_ago:
            find_cmd += f" -cmin -{from_minutes_ago}"
        sstables_res = self.db_node.remoter.sudo(find_cmd, verbose=True, ignore_status=True)
        if sstables_res.stderr:
            self.log.debug('Failed to get sstables for %s. Error: %s', self.ks_cf, sstables_res.stderr)
        else:
            selected_sstables = sstables_res.stdout.split()

        message = f'filtered by last {from_minutes_ago} minutes' if from_minutes_ago else '(not filtered by time)'
        self.log.debug('Got %s sstables %s', len(selected_sstables), message)
        return selected_sstables

    def check_that_sstables_are_encrypted(self, sstables=None,  # pylint: disable=too-many-branches
                                          expected_bool_value: bool = True) -> list:

        if not sstables:
            sstables = self.get_sstables()
        if isinstance(sstables, str):
            sstables = [sstables]
        sstables_encrypted_mapping = {}

        if not sstables:
            raise SstablesNotFound(f"sstables for '{self.keyspace}.{self.table}' wasn't found")

        dump_cmd = get_sstable_metadata_dump_command(self.db_node, self.keyspace, self.table)
        for sstable in sstables:
            sstables_res = self.db_node.remoter.sudo(
                f"{dump_cmd} {sstable}",
                ignore_status=True, verbose=True)

            self.log.debug("sstables_res.stdout: %s", sstables_res.stdout)
            self.log.debug("sstables_res.stderr: %s", sstables_res.stderr)
            # NOTE: if we have 'stdout' then the data was successfully read and it means there was no encryption
            if sstables_res.stdout:
                self.log.debug("Successfully read the sstable located at '%s'.", sstable)
                if dump_cmd == 'sstabledump':
                    sstables_encrypted_mapping[sstable] = False
                else:
                    scylla_metadata = json.loads(sstables_res.stdout, strict=False)['sstables']
                    sstables_encrypted_mapping[sstable] = all('scylla_encryption_options' in metadata.get('extension_attributes', {})
                                                              for table, metadata in scylla_metadata.items())
            # NOTE: case when sstable exists and it is encrypted:
            #       [shard 0:main] seastar - Exiting on unhandled exception: \
            #       sstables::malformed_sstable_exception (Buffer improperly sized to hold requested data. \
            #       Got: 2. Expected: 4 in sstable \
            #       /var/.../me-3g9s_1eqc_2z60w2ushgma9j7ypu-big-Statistics.db)
            elif "malformed_sstable_exception (Buffer improperly sized to hold requested data" in sstables_res.stderr:
                sstables_encrypted_mapping[sstable] = True
            # NOTE: case when sstable was concurrently deleted:
            #       [shard 0:main] seastar - Exiting on unhandled exception: \
            #       sstables::malformed_sstable_exception \
            #       (/var/.../me-3g9s_1941_4web4236cvthbyawsi-big-TOC.txt: file not found)
            elif (" file not found)" in sstables_res.stderr or "Cannot find file" in sstables_res.stderr
                    or "No such file or directory" in sstables_res.stderr):
                self.log.debug("'%s' sstable doesn't exist anymore. Skipping it.", sstable)
            # NOTE: case when
            #       Could not load SSTable: /var/.../me-3g9w_104a_01xg12j55gjivrwtt5-big-Data.db: \
            #       sstables::malformed_sstable_exception (/var/.../me-3g9w_104a_01xg12j55gjivrwtt5-big-Data.db: \
            #       first and last keys of summary are misordered: \
            #       first={key: pk{00080000000000000003}, token: -578762209316392770} \
            #       > last={key: pk{00080000000000000008}, token: -6917704163689751025})
            elif "first and last keys of summary are misordered" in sstables_res.stderr:
                self.log.warning(
                    "'%s' sstable cannot be loaded with 'first and last keys of summary are misordered' error. "
                    "Not representative. Skipping it.",
                    sstable)
            elif "NullPointerException" in sstables_res.stderr or "ArrayIndexOutOfBoundsException" in sstables_res.stderr:
                # using sstabledump
                sstables_encrypted_mapping[sstable] = True
            # NOTE: all other unexpected cases
            else:
                self.log.warning(
                    "Unexpected error reading sstable located at '%s': %s", sstable, sstables_res.stderr)
                sstables_encrypted_mapping[sstable] = None

        # NOTE: we read sstables in a concurrent environment.
        #       So, we can get failures trying to read some of them when it gets deleted concurrently...
        encryption_success_part, encryption_results = 0.79, list(sstables_encrypted_mapping.values())
        assert encryption_results
        assert (encryption_results.count(expected_bool_value) / len(encryption_results) >= encryption_success_part), (
            "Sstables encryption check failed."
            f" Success part: '{encryption_success_part}'. Expected bool value: '{expected_bool_value}'."
            f" Encryption results: {encryption_results}")

    def count_sstable_tombstones(self, sstable: str) -> int:
        dump_cmd = get_sstable_data_dump_command(node=self.db_node, keyspace=self.keyspace, table=self.table)
        self.db_node.remoter.run(
            f'sudo {dump_cmd}  {sstable} 1>/tmp/sstabledump.json', verbose=False, ignore_status=True)
        tombstones_deletion_info = self.db_node.remoter.run(
            'sudo egrep \'"expired" : true|marked_deleted\' /tmp/sstabledump.json', verbose=False, ignore_status=True)
        if not tombstones_deletion_info:
            self.log.debug('Got no tombstones for sstable: %s', sstable)
            return 0

        num_tombstones = len(tombstones_deletion_info.stdout.splitlines())
        self.log.debug('Got %s tombstones for sstable: %s', num_tombstones, sstable)
        return num_tombstones

    def verify_a_live_normal_node_is_used(self):
        if not self.db_node:
            self.db_node = next(node for node in self.db_cluster.data_nodes if node.db_up())
        elif not self.db_node.db_up():
            self.db_node = next(node for node in self.db_node.parent_cluster.data_nodes if node.db_up())

    def get_table_repair_date(self) -> str | None:
        """
        Search entries of requested table in system.repair_history
        Return last entry found.
        Example returned value: '2022-12-28 11:53:53'
        """
        self.verify_a_live_normal_node_is_used()
        with self.db_cluster.cql_connection_patient(node=self.db_node, connect_timeout=300,
                                                    user=self.user, password=self.password) as session:
            try:
                query = f"SELECT repair_time from system.repair_history WHERE keyspace_name = '{self.keyspace}' " \
                        f"AND table_name = '{self.table}' ALLOW FILTERING;"
                results = session.execute(query)
                output = results.all()
                output_length = len(output)
                if output_length == 0:
                    self.log.debug('No repair history found for %s.%s', self.keyspace, self.table)
                    return None
                self.log.debug('Number of rows in repair_time results: %d', output_length)
                self.log.debug('Last row in repair_time results: %s', output[-1])
                return str(output[-1].repair_time)
            except Exception as exc:  # pylint: disable=broad-except
                self.log.error('Failed to get repair date of %s.%s. Error: %s', self.keyspace, self.table, exc)
                raise

    def get_table_repair_date_and_delta_minutes(self) -> (datetime.datetime, int):

        table_repair_date = datetime.datetime.strptime(self.get_table_repair_date(), '%Y-%m-%d %H:%M:%S')
        now = datetime.datetime.now()
        delta_repair_date_minutes = ((now - table_repair_date).seconds - self.propagation_delay_in_seconds) // 60
        self.log.debug('Found table-repair-date: %s, Ended %s minutes ago',
                       table_repair_date, delta_repair_date_minutes)
        return table_repair_date, delta_repair_date_minutes

    def get_compacted_tombstone_deletion_info(self, sstable: str) -> list:
        tombstones_deletion_info = []
        dump_cmd = get_sstable_data_dump_command(node=self.db_node, keyspace=self.keyspace, table=self.table)
        self.db_node.remoter.run(
            f'sudo {dump_cmd}  {sstable} 1>/tmp/sstabledump.json', verbose=False, ignore_status=True)
        result = self.db_node.remoter.run('sudo grep marked_deleted /tmp/sstabledump.json', verbose=False,
                                          ignore_status=True)
        if result.ok:
            tombstones_deletion_info = result.stdout.splitlines()
        else:
            self.log.warning('Failed to find compacted tombstones in %s: (%s, %s)',
                             sstable, result.stdout, result.stderr)
        self.log.debug('Found %s tombstones for sstable %s', len(tombstones_deletion_info), sstable)
        return tombstones_deletion_info

    def verify_post_repair_sstable_tombstones(self, table_repair_date: datetime.datetime, sstable: str):
        non_deleted_tombstones = []
        tombstones_deletion_info = self.get_compacted_tombstone_deletion_info(sstable=sstable)
        for tombstone_deletion_info in tombstones_deletion_info:
            tombstone_delete_date = self.get_tombstone_date(tombstone_deletion_info=tombstone_deletion_info)
            if tombstone_delete_date < table_repair_date:
                non_deleted_tombstones.append(tombstone_delete_date)
        if non_deleted_tombstones:
            raise NonDeletedTombstonesFound(
                f"Found pre-repair time ({table_repair_date}) tombstones in a post-repair sstable ({sstable}): {non_deleted_tombstones}")

    @staticmethod
    def get_tombstone_date(tombstone_deletion_info: str) -> datetime.datetime:
        """
        Parse a datetime value out of an sstable tombstone dump.
        Example input is: '{ "name" : "name_list", "deletion_info" : { "marked_deleted" : "2023-01-03T18:06:36.559369Z",
         "local_delete_time" : "2023-01-03T18:06:36Z" } },'
        Example Output:  datetime.datetime(2023, 1, 3, 18, 5, 58)
        """
        tombstone_dict = json.loads(tombstone_deletion_info[:-1])
        deletion_date, deletion_time = tombstone_dict['deletion_info']['marked_deleted'].split('T')
        deletion_hour, deletion_minutes, deletion_seconds = deletion_time.split(':')
        deletion_seconds = deletion_seconds.split('.')[0]
        full_deletion_date = f'{deletion_date} {deletion_hour}:{deletion_minutes}:{deletion_seconds}'
        full_deletion_date_datetime = datetime.datetime.strptime(full_deletion_date, '%Y-%m-%d %H:%M:%S')
        return full_deletion_date_datetime


def is_new_sstable_dump_supported(node) -> bool:
    """
    Determines if the new "scylla sstable" dump command is supported based on the Scylla version.

    :param node: A DB node object to provide Scylla version.
    :return: True if the new "scylla sstable" dump is supported, else False.
    """
    if node.is_enterprise:
        return ComparableScyllaVersion(node.scylla_version) >= "2023.1.3"
    return ComparableScyllaVersion(node.scylla_version) >= "5.4.0~rc0"


def _generate_sstable_dump_command(node, command: str, keyspace: str, table: str) -> str:
    """
    Constructs the base command for the "scylla sstable" dump tool.

    :param node: A DB node object to provide Scylla version.
    :param command: Specific subcommand for the "scylla sstable" tool (e.g., "dump-data").
    :param keyspace: Keyspace name.
    :param table: Table name.
    :return: Base command string.
    """
    scylla_conf_dir = Path(node.add_install_prefix(SCYLLA_YAML_PATH)).parent
    return (
        f'SCYLLA_CONF={scylla_conf_dir} '
        f'{node.add_install_prefix("/usr/bin/scylla")} sstable {command} '
        f'--keyspace {keyspace} --table {table} --sstables'
    )


def get_sstable_metadata_dump_command(node, keyspace: str, table: str, debug_log_level: bool = True) -> str:
    """
    Constructs the command for dumping sstable metadata using either "sstabledump" or "scylla sstable" tool.

    :param node: A DB node object to provide Scylla version.
    :param keyspace: Keyspace name.
    :param table: Table name.
    :param debug_log_level: If True, includes debug-level logging in the command.
    :return: Command string for metadata dump.
    """
    if not is_new_sstable_dump_supported(node):
        return 'sstabledump'

    log_level_option = " --logger-log-level scylla-sstable=debug" if debug_log_level else ""
    command = f"dump-scylla-metadata{log_level_option}"

    return _generate_sstable_dump_command(node, command, keyspace, table)


def get_sstable_data_dump_command(node, keyspace: str, table: str) -> str:
    """
    Constructs the command for dumping sstable data using either "sstabledump" or "scylla sstable" tool.

    :param node: A DB node object to provide Scylla version.
    :param keyspace: Keyspace name.
    :param table: Table name.
    :return: Command string for data dump.
    """
    if not is_new_sstable_dump_supported(node):
        return 'sstabledump'
    return _generate_sstable_dump_command(node, "dump-data", keyspace, table)
