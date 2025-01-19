import json

from performance_regression_test import PerformanceRegressionTest
from sdcm.utils import loader_utils
from sdcm.utils.operations_thread import ThreadParams
from test_lib.sla import create_sla_auth


class PerformanceRegressionSLATest(PerformanceRegressionTest, loader_utils.LoaderUtilsMixin):
    FULLSCAN_SERVICE_LEVEL_SHARES = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_level_shares = self.params.get("service_level_shares")
        self.fullscan_role = None
        self.roles = []

    def _get_scan_operation_params(self) -> list[ThreadParams]:
        params = self.params.get("run_fullscan")
        self.log.info('Scan operation params are: %s', params)

        sla_role_name, sla_role_password = None, None
        if fullscan_role := getattr(self, "fullscan_role", None):
            sla_role_name = fullscan_role.name
            sla_role_password = fullscan_role.password
        scan_operation_params_list = []
        for item in params:
            item_dict = json.loads(item)
            scan_operation_params_list.append(ThreadParams(
                db_cluster=self.db_cluster,
                termination_event=self.db_cluster.nemesis_termination_event,
                user=sla_role_name,
                user_password=sla_role_password,
                duration=self.get_duration(item_dict.get("duration")),
                **item_dict
            ))

            self.log.info('Scan operation scan_operation_params_list are: %s', scan_operation_params_list)
        return scan_operation_params_list

    def _configure_workload_prioritization(self, workload_names):
        with self.db_cluster.cql_connection_patient(node=self.db_cluster.nodes[0], user=loader_utils.DEFAULT_USER,
                                                    password=loader_utils.DEFAULT_USER_PASSWORD) as session:
            # Add index (shares position in the self.service_level_shares list) to role and service level names to do
            # it unique and prevent failure when try to create role/SL with same name
            for index, shares in enumerate(self.service_level_shares):
                self.roles.append(create_sla_auth(session=session, shares=shares, index=index))

            if self.params.get("run_fullscan"):
                self.fullscan_role = create_sla_auth(session=session, shares=self.FULLSCAN_SERVICE_LEVEL_SHARES,
                                                     index=0)

        self.add_sla_credentials_to_stress_cmds(workload_names=workload_names,
                                                roles=self.roles, params=self.params,
                                                parent_class_name=self.__class__.__name__)

        if scan_operation_params := self._get_scan_operation_params():
            for scan_param in scan_operation_params:
                self.log.info("Starting fullscan operation thread with the following params: %s", scan_param)
                self.run_fullscan_thread(fullscan_params=scan_param,
                                         thread_name=str(scan_operation_params.index(scan_param)))

    def test_mixed(self):
        self._configure_workload_prioritization(workload_names=["stress_cmd_m"])
        super().test_mixed()

    def test_read(self):
        self._configure_workload_prioritization(workload_names=["stress_cmd_r"])
        super().test_mixed()

    def test_write(self):
        self._configure_workload_prioritization(workload_names=["stress_cmd_w"])
        super().test_mixed()

    def test_latency(self):
        """
        Test steps:
        0. Configure WLP
        1. Prepare cluster with data (reach steady_stet of compactions and ~x10 capacity than RAM.
        with round_robin and list of stress_cmd - the data will load several times faster.
        2. Run WRITE workload with gauss population.
        """
        self._configure_workload_prioritization(workload_names=["stress_cmd_w", "stress_cmd_r", "stress_cmd_m"])
        super().test_latency()
