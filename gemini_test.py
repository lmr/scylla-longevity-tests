#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (c) 2018 ScyllaDB

import time

from sdcm.tester import ClusterTester


class GeminiTest(ClusterTester):

    """
    Test Scylla with gemini - tool for testing data integrity
    https://github.com/scylladb/gemini
    """
    gemini_results = {
        "cmd": ["N/A"],
        "status": "Not Running",
        "results": [],
        'errors': {}
    }

    def test_random_load(self):
        """
        Run gemini tool
        """
        cmd = self.params.get('gemini_cmd')

        self.log.debug('Start gemini benchmark')
        gemini_thread = self.run_gemini(cmd=cmd)
        self.gemini_results["cmd"] = gemini_thread.gemini_commands
        self.gemini_results.update(self.verify_gemini_results(queue=gemini_thread))

        self.verify_results()

    def test_load_random_with_nemesis(self):

        cmd = self.params.get('gemini_cmd')

        self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(),
                                    tester_obj=self)

        self.log.debug('Start gemini benchmark')
        gemini_thread = self.run_gemini(cmd=cmd)
        self.gemini_results["cmd"] = gemini_thread.gemini_commands
        # sleep before run nemesis test_duration * .25
        sleep_before_start = float(self.params.get('test_duration')) * 60 * .1
        self.log.info('Sleep interval {}'.format(sleep_before_start))
        time.sleep(sleep_before_start)

        self.db_cluster.start_nemesis()

        self.gemini_results.update(self.verify_gemini_results(queue=gemini_thread))

        self.db_cluster.stop_nemesis(timeout=1600)

        self.verify_results()

    def test_load_random_with_nemesis_cdc_reader(self):
        cmd = self.params.get('gemini_cmd')
        cdc_stress = self.params.get('stress_cdclog_reader_cmd')
        update_es = self.params.get('store_cdclog_reader_stats_in_es')

        self.db_cluster.add_nemesis(nemesis=self.get_nemesis_class(),
                                    tester_obj=self)

        self.log.debug('Start gemini benchmark')
        gemini_thread = self.run_gemini(cmd=cmd)
        self.gemini_results["cmd"] = gemini_thread.gemini_commands
        # wait gemini create schema and write some data
        self.db_cluster.wait_for_schema_agreement()
        cdc_stress_queue = self.run_cdclog_reader_thread(stress_cmd=cdc_stress,
                                                         keyspace_name="ks1",
                                                         base_table_name="table1")
        # sleep before run nemesis test_duration * .1
        sleep_before_start = float(self.params.get('test_duration')) * 60 * .1
        self.log.info('Sleep interval {}'.format(sleep_before_start))
        time.sleep(sleep_before_start)

        self.db_cluster.start_nemesis()

        self.gemini_results.update(self.verify_gemini_results(queue=gemini_thread))

        cdc_stress_results = self.verify_cdclog_reader_results(cdc_stress_queue, update_es)
        self.log.debug(cdc_stress_results)

        self.db_cluster.stop_nemesis(timeout=1600)

        self.verify_results()

    def test_gemini_and_cdc_reader(self):
        gemini_cmd = self.params.get('gemini_cmd')
        cdc_stress = self.params.get('stress_cdclog_reader_cmd')
        update_es = self.params.get('store_cdclog_reader_stats_in_es')

        gemini_thread = self.run_gemini(cmd=gemini_cmd)
        self.gemini_results["cmd"] = gemini_thread.gemini_commands

        # wait gemini create schema
        time.sleep(10)

        cdc_stress_queue = self.run_cdclog_reader_thread(stress_cmd=cdc_stress,
                                                         keyspace_name="ks1",
                                                         base_table_name="table1")

        self.gemini_results.update(self.verify_gemini_results(queue=gemini_thread))

        cdc_stress_results = self.verify_cdclog_reader_results(cdc_stress_queue, update_es)

        self.log.debug(cdc_stress_results)

        self.verify_results()

    def verify_results(self):
        if self.gemini_results['status'] == 'FAILED':
            self.fail(self.gemini_results['results'])

    def get_email_data(self):
        self.log.info('Prepare data for email')

        email_data = self._get_common_email_data()
        grafana_dataset = self.monitors.get_grafana_screenshot_and_snapshot(self.start_time) if self.monitors else {}

        if self.loaders:
            gemini_version = self.loaders.gemini_version
        else:
            self.log.warning("Failed to get gemini version as loader instance is not created")
            gemini_version = ""

        email_data.update({"gemini_cmd": self.gemini_results["cmd"],
                           "gemini_version": gemini_version,
                           "nemesis_details": self.get_nemesises_stats(),
                           "nemesis_name": self.params.get("nemesis_class_name"),
                           "number_of_oracle_nodes": self.params.get("n_test_oracle_db_nodes"),
                           "oracle_ami_id": self.params.get("ami_id_db_oracle"),
                           "oracle_db_version":
                               self.cs_db_cluster.nodes[0].artifact_scylla_version if self.cs_db_cluster else "N/A",
                           "oracle_instance_type": self.params.get("instance_type_db_oracle"),
                           "results": self.gemini_results["results"],
                           "scylla_ami_id": self.params.get("ami_id_db_scylla"),
                           "status": self.gemini_results["status"],
                           "grafana_screenshots": grafana_dataset.get("screenshots", []),
                           "grafana_snapshots": grafana_dataset.get("snapshots", [])})

        return email_data
