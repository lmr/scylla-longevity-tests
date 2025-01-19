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
# Copyright (c) 2021 ScyllaDB
import unittest
from pathlib import Path

import yaml

from sdcm.provision.scylla_yaml import ServerEncryptionOptions, ClientEncryptionOptions, SeedProvider, ScyllaYaml
from sdcm.provision.scylla_yaml.auxiliaries import RequestSchedulerOptions


class ScyllaYamlTest(unittest.TestCase):

    @staticmethod
    def _run_test(object_type, init_params: dict, expected_without_defaults: dict, expected_with_defaults: dict):
        instance = object_type(**init_params)
        assert instance.dict(exclude_unset=True) == expected_without_defaults
        assert instance.dict(exclude_unset=False) == expected_with_defaults

    def test_server_encryption_options(self):
        self._run_test(
            object_type=ServerEncryptionOptions,
            init_params={
                'certificate': '/tmp/123.crt',
                'keyfile': '/tmp/123.key',
            },
            expected_without_defaults={
                'certificate': '/tmp/123.crt',
                'keyfile': '/tmp/123.key',
            },
            expected_with_defaults={
                'internode_encryption': 'none',
                'certificate': '/tmp/123.crt',
                'keyfile': '/tmp/123.key',
                'truststore': None,
                'priority_string': None,
                'require_client_auth': False
            }
        )

    def test_client_encryption_options(self):
        self._run_test(
            object_type=ClientEncryptionOptions,
            init_params={
                'certificate': '/tmp/123.crt',
                'keyfile': '/tmp/123.key',
                'truststore': '/tmp/123.pem',
                'require_client_auth': True,
            },
            expected_without_defaults={
                'certificate': '/tmp/123.crt',
                'keyfile': '/tmp/123.key',
                'truststore': '/tmp/123.pem',
                'require_client_auth': True
            },
            expected_with_defaults={
                'enabled': False,
                'certificate': '/tmp/123.crt',
                'keyfile': '/tmp/123.key',
                'truststore': '/tmp/123.pem',
                'priority_string': None,
                'require_client_auth': True
            }
        )

    def test_request_scheduler_options(self):
        self._run_test(
            object_type=RequestSchedulerOptions,
            init_params={
                'throttle_limit': 100,
                'default_weight': 1,
            },
            expected_without_defaults={'throttle_limit': 100, 'default_weight': 1},
            expected_with_defaults={'throttle_limit': 100, 'default_weight': 1, 'weights': 1}
        )

    def test_seed_provider(self):
        self._run_test(
            object_type=SeedProvider,
            init_params={
                'class_name': 'org.apache.cassandra.locator.SimpleSeedProvider',
                'parameters': [{'seeds': ['1.1.1.1', '2.2.2.2']}]
            },
            expected_without_defaults={
                'class_name': 'org.apache.cassandra.locator.SimpleSeedProvider',
                'parameters': [{'seeds': ['1.1.1.1', '2.2.2.2']}]},
            expected_with_defaults={
                'class_name': 'org.apache.cassandra.locator.SimpleSeedProvider',
                'parameters': [{'seeds': ['1.1.1.1', '2.2.2.2']}]
            }
        )

    def test_scylla_yaml(self):
        self._run_test(
            object_type=ScyllaYaml,
            init_params={
                'log_to_stdout': True,
                'auto_adjust_flush_quota': False,
                'background_writer_scheduling_quota': 1.0,
                'listen_address': "localhost",
                'prometheus_prefix': 'someprefix',
                'server_encryption_options': ServerEncryptionOptions(
                    internode_encryption='none',
                    certificate='/tmp/123.crt',
                    keyfile='/tmp/123.key',
                ),
                'client_encryption_options': ClientEncryptionOptions(
                    enabled=False,
                    certificate='/tmp/123.crt',
                    keyfile='/tmp/123.key',
                    require_client_auth=True,
                ),
                'seed_provider': [
                    SeedProvider(
                        class_name='org.apache.cassandra.locator.SimpleSeedProvider',
                        parameters=[{'seeds': ['1.1.1.1', '2.2.2.2']}]),
                ],
                'force_schema_commit_log': True,
                'experimental_features': None,
            },
            expected_without_defaults={
                'background_writer_scheduling_quota': 1.0,
                'auto_adjust_flush_quota': False,
                'listen_address': 'localhost',
                'seed_provider': [
                    {
                        'class_name': 'org.apache.cassandra.locator.SimpleSeedProvider',
                        'parameters': [{'seeds': ['1.1.1.1', '2.2.2.2']}]
                    }
                ],
                'server_encryption_options': {
                    'internode_encryption': 'none',
                    'certificate': '/tmp/123.crt',
                    'keyfile': '/tmp/123.key'
                },
                'client_encryption_options': {
                    'enabled': False,
                    'certificate': '/tmp/123.crt',
                    'keyfile': '/tmp/123.key',
                    'require_client_auth': True
                },
                'prometheus_prefix': 'someprefix',
                'log_to_stdout': True,
                'force_schema_commit_log': True,
                'experimental_features': None,
            },
            expected_with_defaults={
                'abort_on_ebadf': None,
                'abort_on_internal_error': None,
                'abort_on_lsa_bad_alloc': None,
                'alternator_address': None,
                'alternator_enforce_authorization': None,
                'alternator_https_port': None,
                'alternator_port': None,
                'alternator_streams_time_window_s': None,
                'alternator_ttl_period_in_seconds': None,
                'alternator_write_isolation': None,
                'api_address': None,
                'api_doc_dir': None,
                'api_port': None,
                'api_ui_dir': None,
                'audit': None,
                'audit_categories': None,
                'audit_keyspaces': None,
                'audit_tables': None,
                'authenticator': None,
                'authenticator_password': None,
                'authenticator_user': None,
                'authorizer': None,
                'auto_adjust_flush_quota': False,
                'auto_bootstrap': None,
                'auto_snapshot': None,
                'background_writer_scheduling_quota': 1.0,
                'batch_size_fail_threshold_in_kb': None,
                'batch_size_warn_threshold_in_kb': None,
                'batchlog_replay_throttle_in_kb': None,
                'broadcast_address': None,
                'broadcast_rpc_address': None,
                'cache_hit_rate_read_balancing': None,
                'cas_contention_timeout_in_ms': None,
                'client_encryption_options': {'certificate': '/tmp/123.crt',
                                              'enabled': False,
                                              'keyfile': '/tmp/123.key',
                                              'priority_string': None,
                                              'require_client_auth': True,
                                              'truststore': None},
                'cluster_name': None,
                'column_index_size_in_kb': None,
                'commit_failure_policy': None,
                'commitlog_directory': None,
                'commitlog_reuse_segments': None,
                'commitlog_segment_size_in_mb': None,
                'commitlog_sync': None,
                'commitlog_sync_batch_window_in_ms': None,
                'commitlog_sync_period_in_ms': None,
                'commitlog_total_space_in_mb': None,
                'commitlog_use_o_dsync': None,
                'compaction_collection_items_count_warning_threshold': None,
                'compaction_enforce_min_threshold': None,
                'compaction_large_cell_warning_threshold_mb': None,
                'compaction_large_partition_warning_threshold_mb': None,
                'compaction_large_row_warning_threshold_mb': None,
                'compaction_preheat_key_cache': None,
                'compaction_rows_count_warning_threshold': None,
                'compaction_static_shares': None,
                'compaction_throughput_mb_per_sec': None,
                'concurrent_compactors': None,
                'concurrent_counter_writes': None,
                'concurrent_reads': None,
                'concurrent_writes': None,
                'consistent_cluster_management': None,
                'consistent_rangemovement': None,
                'counter_cache_keys_to_save': None,
                'counter_cache_save_period': None,
                'counter_cache_size_in_mb': None,
                'counter_write_request_timeout_in_ms': None,
                'cpu_scheduler': None,
                'cross_node_timeout': None,
                'data_file_directories': None,
                'default_log_level': None,
                'defragment_memory_on_idle': None,
                'developer_mode': None,
                'disk_failure_policy': None,
                'dynamic_snitch_badness_threshold': None,
                'dynamic_snitch_reset_interval_in_ms': None,
                'dynamic_snitch_update_interval_in_ms': None,
                'enable_3_1_0_compatibility_mode': None,
                'enable_cache': None,
                'enable_commitlog': None,
                'enable_dangerous_direct_import_of_cassandra_counters': None,
                'enable_deprecated_partitioners': None,
                'enable_in_memory_data_store': None,
                'enable_ipv6_dns_lookup': None,
                'enable_keyspace_column_family_metrics': None,
                'enable_node_aggregated_table_metrics': None,
                'enable_repair_based_node_ops': None,
                'allowed_repair_based_node_ops': None,
                'enable_shard_aware_drivers': None,
                'enable_small_table_optimization_for_rbno': None,
                'enable_sstable_data_integrity_check': None,
                'enable_sstable_key_validation': None,
                'enable_sstables_mc_format': None,
                'enable_sstables_md_format': None,
                'enable_user_defined_functions': None,
                'endpoint_snitch': None,
                'experimental_features': None,
                'fd_initial_value_ms': None,
                'fd_max_interval_ms': None,
                'file_cache_size_in_mb': None,
                'force_gossip_generation': None,
                'force_schema_commit_log': True,
                'hinted_handoff_enabled': None,
                'hinted_handoff_throttle_in_kb': None,
                'hints_directory': None,
                'in_memory_compaction_limit_in_mb': None,
                'incremental_backups': None,
                'index_summary_capacity_in_mb': None,
                'index_summary_resize_interval_in_minutes': None,
                'initial_sstable_loading_concurrency': None,
                'initial_token': None,
                'inter_dc_stream_throughput_outbound_megabits_per_sec': None,
                'inter_dc_tcp_nodelay': None,
                'internode_authenticator': None,
                'internode_compression': None,
                'internode_compression_enable_advanced': None,
                'rpc_dict_training_when': None,
                'rpc_dict_training_min_time_seconds': None,
                'rpc_dict_update_period_seconds': None,
                'rpc_dict_training_min_bytes': None,
                'internode_compression_zstd_max_cpu_fraction': None,
                'internode_recv_buff_size_in_bytes': None,
                'internode_send_buff_size_in_bytes': None,
                'join_ring': None,
                'key_cache_keys_to_save': None,
                'key_cache_save_period': None,
                'key_cache_size_in_mb': None,
                'kmip_hosts': None,
                'kms_hosts': None,
                'large_memory_allocation_warning_threshold': None,
                'ldap_attr_role': None,
                'ldap_bind_dn': None,
                'ldap_bind_passwd': None,
                'ldap_url_template': None,
                'listen_address': 'localhost',
                'listen_interface': None,
                'listen_interface_prefer_ipv6': None,
                'listen_on_broadcast_address': None,
                'load_balance': None,
                'load_ring_state': None,
                'log_to_stdout': True,
                'log_to_syslog': None,
                'logger_log_level': None,
                'lsa_reclamation_step': None,
                'max_clustering_key_restrictions_per_query': None,
                'max_hint_window_in_ms': None,
                'max_hints_delivery_threads': None,
                'max_memory_for_unlimited_query': None,
                'max_partition_key_restrictions_per_query': None,
                'memory_allocator': None,
                'memtable_allocation_type': None,
                'memtable_cleanup_threshold': None,
                'memtable_flush_queue_size': None,
                'memtable_flush_static_shares': None,
                'memtable_flush_writers': None,
                'memtable_heap_space_in_mb': None,
                'memtable_offheap_space_in_mb': None,
                'memtable_total_space_in_mb': None,
                'murmur3_partitioner_ignore_msb_bits': None,
                'native_shard_aware_transport_port': None,
                'native_shard_aware_transport_port_ssl': None,
                'native_transport_max_frame_size_in_mb': None,
                'native_transport_max_threads': None,
                'native_transport_port': None,
                'native_transport_port_ssl': None,
                'num_tokens': None,
                'override_decommission': None,
                'partitioner': None,
                'permissions_cache_max_entries': None,
                'permissions_update_interval_in_ms': None,
                'permissions_validity_in_ms': None,
                'phi_convict_threshold': None,
                'preheat_kernel_page_cache': None,
                'prometheus_address': None,
                'prometheus_port': None,
                'prometheus_prefix': 'someprefix',
                'range_request_timeout_in_ms': None,
                'read_request_timeout_in_ms': None,
                'redis_database_count': None,
                'redis_keyspace_replication_strategy': None,
                'redis_port': None,
                'redis_read_consistency_level': None,
                'redis_ssl_port': None,
                'redis_write_consistency_level': None,
                'reduce_cache_capacity_to': None,
                'reduce_cache_sizes_at': None,
                'replace_address': None,
                'replace_address_first_boot': None,
                'replace_node': None,
                'replace_node_first_boot': None,
                'replace_token': None,
                'request_scheduler': None,
                'request_scheduler_id': None,
                'request_scheduler_options': None,
                'request_timeout_in_ms': None,
                'ring_delay_ms': None,
                'role_manager': None,
                'row_cache_keys_to_save': None,
                'row_cache_save_period': None,
                'row_cache_size_in_mb': None,
                'rpc_address': None,
                'rpc_interface': None,
                'rpc_interface_prefer_ipv6': None,
                'rpc_keepalive': None,
                'rpc_max_threads': None,
                'rpc_min_threads': None,
                'rpc_port': None,
                'rpc_recv_buff_size_in_bytes': None,
                'rpc_send_buff_size_in_bytes': None,
                'rpc_server_type': None,
                'saslauthd_socket_path': None,
                'saved_caches_directory': None,
                'seed_provider': [{'class_name': 'org.apache.cassandra.locator.SimpleSeedProvider',
                                   'parameters': [{'seeds': ['1.1.1.1', '2.2.2.2']}]}],
                'server_encryption_options': {'certificate': '/tmp/123.crt',
                                              'internode_encryption': 'none',
                                              'keyfile': '/tmp/123.key',
                                              'priority_string': None,
                                              'require_client_auth': False,
                                              'truststore': None},
                'shadow_round_ms': None,
                'shutdown_announce_in_ms': None,
                'skip_wait_for_gossip_to_settle': None,
                'snapshot_before_compaction': None,
                'ssl_storage_port': None,
                'sstable_preemptive_open_interval_in_mb': None,
                'sstable_summary_ratio': None,
                'start_native_transport': None,
                'start_rpc': None,
                'storage_port': None,
                'stream_io_throughput_mb_per_sec': None,
                'stream_throughput_outbound_megabits_per_sec': None,
                'streaming_socket_timeout_in_ms': None,
                'system_info_encryption': None,
                'system_key_directory': None,
                'tablets_initial_scale_factor': None,
                'thrift_framed_transport_size_in_mb': None,
                'thrift_max_message_length_in_mb': None,
                'tombstone_failure_threshold': None,
                'tombstone_warn_threshold': None,
                'trickle_fsync': None,
                'trickle_fsync_interval_in_kb': None,
                'truncate_request_timeout_in_ms': None,
                'user_defined_function_allocation_limit_bytes': None,
                'user_defined_function_contiguous_allocation_limit_bytes': None,
                'user_defined_function_time_limit_ms': None,
                'user_info_encryption': None,
                'view_building': None,
                'view_hints_directory': None,
                'virtual_dirty_soft_limit': None,
                'volatile_system_keyspace_for_testing': None,
                'workdir': None,
                'write_request_timeout_in_ms': None,
                'enable_tablets': None,
                'force_gossip_topology_changes': None,
                'reader_concurrency_semaphore_cpu_concurrency': None,
            }
        )

    @staticmethod
    def test_update_with_scylla_yaml_object():
        yaml1 = ScyllaYaml(cluster_name='cluster1', redis_keyspace_replication_strategy='NetworkTopologyStrategy')
        yaml2 = ScyllaYaml(
            redis_keyspace_replication_strategy='SimpleStrategy',
            client_encryption_options=ClientEncryptionOptions(
                enabled=True,
                certificate='/tmp/123.crt',
                keyfile='/tmp/123.key',
                truststore='/tmp/trust.pem',
            ),
            server_encryption_options=ServerEncryptionOptions(
                internode_encryption='all',
                certificate='/tmp/123.crt',
                keyfile='/tmp/123.key',
                truststore='/tmp/trust.pem',
            )
        )
        yaml3 = ScyllaYaml(client_encryption_options=ClientEncryptionOptions())
        yaml1.update(yaml2, yaml3)
        assert yaml1 == ScyllaYaml(
            cluster_name='cluster1',
            redis_keyspace_replication_strategy='SimpleStrategy',
            server_encryption_options=ServerEncryptionOptions(internode_encryption='all',
                                                              certificate='/tmp/123.crt', keyfile='/tmp/123.key',
                                                              truststore='/tmp/trust.pem'),
            client_encryption_options=ClientEncryptionOptions()
        )

    @staticmethod
    def test_update_with_dict_object():
        yaml1 = ScyllaYaml(cluster_name='cluster1', redis_keyspace_replication_strategy='NetworkTopologyStrategy')
        test_config_file = Path(__file__).parent / 'test_data' / 'scylla_yaml_update.yaml'
        with open(test_config_file, encoding="utf-8") as test_file:
            test_config_file_yaml = yaml.safe_load(test_file)
            append_scylla_args_dict = test_config_file_yaml.get("append_scylla_yaml", {})
        yaml1.update(append_scylla_args_dict)
        assert yaml1.enable_sstables_mc_format == append_scylla_args_dict["enable_sstables_mc_format"]
        assert yaml1.enable_sstables_md_format == append_scylla_args_dict["enable_sstables_md_format"]

        assert yaml1.force_schema_commit_log == append_scylla_args_dict["force_schema_commit_log"]  # pylint: disable=no-member

    @staticmethod
    def test_copy():
        original = ScyllaYaml(
            redis_keyspace_replication_strategy='SimpleStrategy',
            client_encryption_options=ClientEncryptionOptions(
                enabled=True,
                certificate='/tmp/123.crt',
                keyfile='/tmp/123.key',
                truststore='/tmp/trust.pem',
            ),
            server_encryption_options=ServerEncryptionOptions(
                internode_encryption='all',
                certificate='/tmp/123.crt',
                keyfile='/tmp/123.key',
                truststore='/tmp/trust.pem',
            )
        )
        copy_instance = original.copy()
        assert copy_instance == original
        assert copy_instance.dict(exclude_unset=True, exclude_defaults=True) == original.dict(
            exclude_unset=True, exclude_defaults=True)
        copy_instance.client_encryption_options.enabled = False
        assert copy_instance.client_encryption_options.enabled is False
        assert original.client_encryption_options.enabled is True
        assert copy_instance != original
        assert copy_instance.dict(exclude_unset=True, exclude_defaults=True) != original.dict(
            exclude_unset=True, exclude_defaults=True)

        copy_instance = original.copy()
        copy_instance.client_encryption_options = None
        assert copy_instance != original
        assert copy_instance.dict(exclude_unset=True, exclude_defaults=True) != original.dict(
            exclude_unset=True, exclude_defaults=True)
