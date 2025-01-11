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
from difflib import unified_diff
from typing import List, Literal, Union

import logging
import yaml
from pydantic import validator, BaseModel, ConfigDict  # pylint: disable=no-name-in-module

from sdcm.provision.scylla_yaml.auxiliaries import RequestSchedulerOptions, EndPointSnitchType, SeedProvider, \
    ServerEncryptionOptions, ClientEncryptionOptions


logger = logging.getLogger(__name__)


class ScyllaYaml(BaseModel):  # pylint: disable=too-few-public-methods,too-many-instance-attributes

    model_config = ConfigDict(arbitrary_types_allowed=True)

    broadcast_address: str = None  # ""
    api_port: int = None  # 10000
    api_address: str = None  # ""
    ssl_storage_port: int = None  # 7001
    background_writer_scheduling_quota: float = None  # 1.0
    auto_adjust_flush_quota: bool = None  # False
    memtable_flush_static_shares: int = None  # 0
    compaction_static_shares: int = None  # 0
    compaction_enforce_min_threshold: bool = None  # False
    cluster_name: str = None  # ""
    listen_address: str = None  # "localhost"
    listen_interface: str = None  # "eth0"
    listen_interface_prefer_ipv6: bool = None  # False
    commitlog_directory: str = None  # ""
    data_file_directories: List[str] = None  # None
    hints_directory: str = None  # ""
    view_hints_directory: str = None  # ""
    saved_caches_directory: str = None  # ""
    commit_failure_policy: Literal["die", "stop", "stop_commit", "ignore"] = None  # "stop"
    disk_failure_policy: Literal["die", "stop_paranoid", "stop", "best_effort", "ignore"] = None  # "stop"
    endpoint_snitch: EndPointSnitchType = None

    # pylint: disable=no-self-argument,no-self-use
    @validator("endpoint_snitch", pre=True, always=True)
    def set_endpoint_snitch(cls, endpoint_snitch: str):
        if endpoint_snitch is None:
            return endpoint_snitch
        if endpoint_snitch.startswith('org.apache.cassandra.locator.'):
            return endpoint_snitch
        return 'org.apache.cassandra.locator.' + endpoint_snitch

    rpc_address: str = None  # 'localhost'
    rpc_interface: str = None  # "eth1"
    rpc_interface_prefer_ipv6: bool = None  # False
    # [SeedProvider(class_name='org.apache.cassandra.locator.SimpleSeedProvider')]
    seed_provider: List[SeedProvider] = None
    force_schema_commit_log: bool = None  # False
    consistent_cluster_management: bool = None  # False
    compaction_throughput_mb_per_sec: int = None  # 0
    compaction_large_partition_warning_threshold_mb: int = None  # 1000
    compaction_large_row_warning_threshold_mb: int = None  # 10
    compaction_large_cell_warning_threshold_mb: int = None  # 1
    compaction_rows_count_warning_threshold: int = None  # 100000
    memtable_total_space_in_mb: int = None  # 0
    concurrent_reads: int = None  # 32
    concurrent_writes: int = None  # 32
    concurrent_counter_writes: int = None  # 32
    incremental_backups: bool = None  # False
    snapshot_before_compaction: bool = None  # False
    phi_convict_threshold: int = None  # 8
    commitlog_sync: Literal['periodic', 'batch'] = None  # 'periodic'
    commitlog_segment_size_in_mb: int = None  # 64
    commitlog_sync_period_in_ms: int = None  # 10000
    commitlog_sync_batch_window_in_ms: int = None  # 10000
    commitlog_total_space_in_mb: int = None  # -1
    commitlog_reuse_segments: bool = None  # True
    commitlog_use_o_dsync: bool = None  # True
    compaction_preheat_key_cache: bool = None  # True
    concurrent_compactors: int = None  # 0
    in_memory_compaction_limit_in_mb: int = None  # 64
    preheat_kernel_page_cache: bool = None  # False
    sstable_preemptive_open_interval_in_mb: int = None  # 50
    defragment_memory_on_idle: bool = None  # False
    memtable_allocation_type: Literal[
        'heap_buffers',
        'offheap_buffers',
        'offheap_objects'
    ] = None  # "heap_buffers"
    memtable_cleanup_threshold: float = None  # .11
    file_cache_size_in_mb: int = None  # 512
    memtable_flush_queue_size: int = None  # 4
    memtable_flush_writers: int = None  # 1
    memtable_heap_space_in_mb: int = None  # 0
    memtable_offheap_space_in_mb: int = None  # 0
    column_index_size_in_kb: int = None  # 64
    index_summary_capacity_in_mb: int = None  # 0
    index_summary_resize_interval_in_minutes: int = None  # 60
    reduce_cache_capacity_to: float = None  # .6
    reduce_cache_sizes_at: float = None  # .85
    stream_throughput_outbound_megabits_per_sec: int = None  # 400
    inter_dc_stream_throughput_outbound_megabits_per_sec: int = None  # 0
    trickle_fsync: bool = None  # False
    trickle_fsync_interval_in_kb: int = None  # 10240
    auto_bootstrap: bool = None  # True
    batch_size_warn_threshold_in_kb: int = None  # 5
    batch_size_fail_threshold_in_kb: int = None  # 50
    listen_on_broadcast_address: bool = None  # False
    initial_token: int = None  # None
    num_tokens: int = None  # 1
    # 'org.apache.cassandra.dht.Murmur3Partitioner'
    partitioner: Literal['org.apache.cassandra.dht.Murmur3Partitioner'] = None
    storage_port: int = None  # 7000
    auto_snapshot: bool = None  # True
    key_cache_keys_to_save: int = None  # 0
    key_cache_save_period: int = None  # 14400
    key_cache_size_in_mb: int = None  # 100
    row_cache_keys_to_save: int = None  # 0
    row_cache_size_in_mb: int = None  # 0
    row_cache_save_period: int = None  # 0
    memory_allocator: Literal[
        "NativeAllocator",
        "JEMallocAllocator"
    ] = None  # "NativeAllocator"
    counter_cache_size_in_mb: int = None  # 0
    counter_cache_save_period: int = None  # 7200
    counter_cache_keys_to_save: int = None  # 0
    tombstone_warn_threshold: int = None  # 1000
    tombstone_failure_threshold: int = None  # 100000
    range_request_timeout_in_ms: int = None  # 10000
    read_request_timeout_in_ms: int = None  # 5000
    counter_write_request_timeout_in_ms: int = None  # 5000
    cas_contention_timeout_in_ms: int = None  # 1000
    truncate_request_timeout_in_ms: int = None  # 60000
    write_request_timeout_in_ms: int = None  # 2000
    request_timeout_in_ms: int = None  # 10000
    cross_node_timeout: bool = None  # False
    internode_send_buff_size_in_bytes: int = None  # 0
    internode_recv_buff_size_in_bytes: int = None  # 0
    internode_compression: Literal['none', 'all', 'dc'] = None  # "none"
    internode_compression_enable_advanced: bool = None
    rpc_dict_training_when: Literal['always', 'never', 'when_leader'] = None
    rpc_dict_training_min_time_seconds: int = None
    rpc_dict_update_period_seconds: int = None
    rpc_dict_training_min_bytes: int = None
    internode_compression_zstd_max_cpu_fraction: float = None
    inter_dc_tcp_nodelay: bool = None  # False
    streaming_socket_timeout_in_ms: int = None  # 0
    start_native_transport: bool = None  # True
    native_transport_port: int = None  # 9042
    native_transport_port_ssl: int = None  # 9142
    native_transport_max_threads: int = None  # 128
    native_transport_max_frame_size_in_mb: int = None  # 256
    native_shard_aware_transport_port: int = None  # 19042
    native_shard_aware_transport_port_ssl: int = None  # 19142
    broadcast_rpc_address: str = None  # None
    rpc_port: int = None  # 9160
    start_rpc: bool = None  # True
    rpc_keepalive: bool = None  # True
    rpc_max_threads: int = None  # 0
    rpc_min_threads: int = None  # 16
    rpc_recv_buff_size_in_bytes: int = None  # 0
    rpc_send_buff_size_in_bytes: int = None  # 0
    rpc_server_type: Literal['sync', 'hsh'] = None  # "sync"
    cache_hit_rate_read_balancing: bool = None  # True
    dynamic_snitch_badness_threshold: int = None  # 0
    dynamic_snitch_reset_interval_in_ms: int = None  # 60000
    dynamic_snitch_update_interval_in_ms: int = None  # 100
    hinted_handoff_enabled: Literal[
        'enabled', 'true', 'True', '1', True,
        'disabled', 'false', 'False', '0', False,
    ] = None
    hinted_handoff_throttle_in_kb: int = None  # 1024
    max_hint_window_in_ms: int = None  # 10800000
    max_hints_delivery_threads: int = None  # 2
    batchlog_replay_throttle_in_kb: int = None  # 1024
    request_scheduler: Literal[
        'org.apache.cassandra.scheduler.NoScheduler',
        'org.apache.cassandra.scheduler.RoundRobinScheduler'
    ] = None
    stream_io_throughput_mb_per_sec: int = None  # 0

    # pylint: disable=no-self-argument,no-self-use
    @validator("request_scheduler", pre=True, always=True)
    def set_request_scheduler(cls, request_scheduler: str):
        if request_scheduler is None:
            return request_scheduler
        if request_scheduler.startswith('org.apache.cassandra.scheduler.'):
            return request_scheduler
        return 'org.apache.cassandra.scheduler.' + request_scheduler

    request_scheduler_id: str = None  # None
    request_scheduler_options: RequestSchedulerOptions = None  # None
    thrift_framed_transport_size_in_mb: int = None  # 15
    thrift_max_message_length_in_mb: int = None  # 16
    authenticator: Literal[
        "org.apache.cassandra.auth.PasswordAuthenticator",
        "org.apache.cassandra.auth.AllowAllAuthenticator",
        "com.scylladb.auth.TransitionalAuthenticator",
        "com.scylladb.auth.SaslauthdAuthenticator"
    ] = None  # "org.apache.cassandra.auth.AllowAllAuthenticator"

    # pylint: disable=no-self-argument,no-self-use
    @validator("authenticator", pre=True, always=True)
    def set_authenticator(cls, authenticator: str):
        if authenticator is None:
            return authenticator
        if authenticator.startswith(('org.apache.cassandra.auth.', 'com.scylladb.auth.')):
            return authenticator
        if authenticator in ['PasswordAuthenticator', 'AllowAllAuthenticator']:
            return 'org.apache.cassandra.auth.' + authenticator
        if authenticator in ['TransitionalAuthenticator', 'SaslauthdAuthenticator']:
            return 'com.scylladb.auth.' + authenticator
        return authenticator

    internode_authenticator: Literal["enabled", "disabled"] = None  # "disabled"
    authorizer: Literal[
        "org.apache.cassandra.auth.AllowAllAuthorizer",
        "org.apache.cassandra.auth.CassandraAuthorizer",
        "com.scylladb.auth.TransitionalAuthorizer",
        "com.scylladb.auth.SaslauthdAuthorizer"
    ] = None  # "org.apache.cassandra.auth.AllowAllAuthorizer"

    # pylint: disable=no-self-argument,no-self-use
    @validator("authorizer", pre=True, always=True)
    def set_authorizer(cls, authorizer: str):
        if authorizer is None:
            return authorizer
        if authorizer.startswith(('org.apache.cassandra.auth.', 'com.scylladb.auth.')):
            return authorizer
        if authorizer in ['AllowAllAuthorizer', 'CassandraAuthorizer']:
            return 'org.apache.cassandra.auth.' + authorizer
        if authorizer in ['TransitionalAuthorizer', 'SaslauthdAuthorizer']:
            return 'com.scylladb.auth.' + authorizer
        return authorizer

    role_manager: str = None  # "org.apache.cassandra.auth.CassandraRoleManager"
    permissions_validity_in_ms: int = None  # 10000
    permissions_update_interval_in_ms: int = None  # 2000
    permissions_cache_max_entries: int = None  # 1000
    server_encryption_options: ServerEncryptionOptions = None
    client_encryption_options: ClientEncryptionOptions = None
    enable_in_memory_data_store: bool = None  # False
    enable_cache: bool = None  # True
    enable_commitlog: bool = None  # True
    volatile_system_keyspace_for_testing: bool = None  # False
    api_ui_dir: str = None  # "swagger-ui/dist/"
    api_doc_dir: str = None  # "api/api-doc/"
    load_balance: str = None  # "none"
    consistent_rangemovement: bool = None  # True
    join_ring: bool = None  # True
    load_ring_state: bool = None  # True
    replace_node: str = None  # ""
    replace_token: str = None  # ""
    replace_address: str = None  # ""
    replace_address_first_boot: str = None  # ""
    replace_node_first_boot: str = None  # ""
    override_decommission: bool = None  # False
    enable_repair_based_node_ops: bool = None  # True
    # NOTE: example for disabling RBNO for 'bootstrap' and 'decommission' operations:
    #       allowed_repair_based_node_ops: "replace,removenode,rebuild"
    allowed_repair_based_node_ops: str = None
    enable_small_table_optimization_for_rbno: bool = None  # False
    ring_delay_ms: int = None  # 30 * 1000
    shadow_round_ms: int = None  # 300 * 1000
    fd_max_interval_ms: int = None  # 2 * 1000
    fd_initial_value_ms: int = None  # 2 * 1000
    shutdown_announce_in_ms: int = None  # 2 * 1000
    developer_mode: bool = None  # False
    skip_wait_for_gossip_to_settle: int = None  # -1
    force_gossip_generation: int = None  # -1
    experimental_features: list[str] = None  # []
    tablets_initial_scale_factor: int = None
    lsa_reclamation_step: int = None  # 1
    prometheus_port: int = None  # 9180
    prometheus_address: str = None  # "localhost"
    prometheus_prefix: str = None  # "scylla"
    abort_on_lsa_bad_alloc: bool = None  # False
    murmur3_partitioner_ignore_msb_bits: int = None  # 12
    virtual_dirty_soft_limit: float = None  # 0.6
    sstable_summary_ratio: float = None  # 0.0005
    large_memory_allocation_warning_threshold: int = None  # 2 ** 20
    enable_deprecated_partitioners: bool = None  # False
    enable_keyspace_column_family_metrics: bool = None  # False
    enable_sstable_data_integrity_check: bool = None  # False
    enable_sstable_key_validation: bool = None  # None
    cpu_scheduler: bool = None  # True
    view_building: bool = None  # True
    enable_sstables_mc_format: bool = None  # True
    enable_sstables_md_format: bool = None  # False
    enable_dangerous_direct_import_of_cassandra_counters: bool = None  # False
    enable_shard_aware_drivers: bool = None  # True
    enable_ipv6_dns_lookup: bool = None  # False
    abort_on_internal_error: bool = None  # False
    max_partition_key_restrictions_per_query: int = None  # 100
    max_clustering_key_restrictions_per_query: int = None  # 100
    max_memory_for_unlimited_query: int = None  # 1048576
    initial_sstable_loading_concurrency: str = None  # '4u'
    enable_3_1_0_compatibility_mode: bool = None  # False
    enable_user_defined_functions: bool = None  # False
    user_defined_function_time_limit_ms: int = None  # 10
    user_defined_function_allocation_limit_bytes: int = None  # 1024 * 1024
    user_defined_function_contiguous_allocation_limit_bytes: int = None  # 1024 * 1024
    alternator_port: int = None  # 0
    alternator_https_port: int = None  # 0
    alternator_address: str = None  # "0.0.0.0"
    alternator_enforce_authorization: bool = None  # False
    alternator_write_isolation: Literal["unsafe_rmw", "only_rmw_uses_lwt",
                                        "forbid_rmw", "always_use_lwt"] = None  # None
    alternator_streams_time_window_s: int = None  # 10
    alternator_ttl_period_in_seconds: int = None  # None
    abort_on_ebadf: bool = None  # True
    redis_port: int = None  # 0
    redis_ssl_port: int = None  # 0
    redis_read_consistency_level: str = None  # "LOCAL_QUORUM"
    redis_write_consistency_level: str = None  # "LOCAL_QUORUM"
    redis_database_count: int = None  # 16
    redis_keyspace_replication_strategy: Literal['SimpleStrategy', 'NetworkTopologyStrategy'] = None  # 'SimpleStrategy'
    default_log_level: str = None  # None
    logger_log_level: dict = None  # None
    log_to_stdout: bool = None  # None
    log_to_syslog: bool = None  # None
    authenticator_user: str = None  # None
    authenticator_password: str = None  # None
    workdir: str = None  # None

    ldap_attr_role: str = None  # None
    ldap_bind_dn: str = None  # None
    ldap_bind_passwd: str = None  # None
    ldap_url_template: str = None  # None
    saslauthd_socket_path: str = None  # None

    system_key_directory: str = None  # None
    system_info_encryption: dict = None  # None
    user_info_encryption: dict = None  # None
    kmip_hosts: dict = None  # None
    kms_hosts: dict = None  # None

    audit: str = None  # None
    audit_categories: str = None  # None
    audit_tables: str = None  # None
    audit_keyspaces: str = None  # None

    compaction_collection_items_count_warning_threshold: int = None  # None

    enable_tablets: bool = None  # False, but default scylla.yaml for some versions (e.g. 6.0) override it to True
    force_gossip_topology_changes: bool = None  # False

    reader_concurrency_semaphore_cpu_concurrency: int = None

    def dict(  # pylint: disable=arguments-differ
        self,
        *,
        include: Union['MappingIntStrAny', 'AbstractSetIntStr'] = None,  # noqa: F821
        exclude: Union['MappingIntStrAny', 'AbstractSetIntStr'] = None,  # noqa: F821
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        explicit: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,  # noqa: F821
    ) -> 'DictStrAny':  # noqa: F821
        to_dict = super().dict(
            include=include, exclude=exclude, by_alias=by_alias, skip_defaults=skip_defaults,
            exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)
        if explicit:
            for required_attrs in explicit:
                to_dict[required_attrs] = getattr(self, required_attrs)
        return to_dict

    def _update_dict(self, obj: dict, fields_data: dict):
        for attr_name, attr_value in obj.items():
            attr_info = fields_data.get(attr_name, None)
            if attr_info is None:
                logger.warning("Provided unknown attribute `%s`", attr_name)
            if attr_info and hasattr(attr_info.type_, "__attrs_attrs__"):
                if attr_value is not None:
                    if not isinstance(attr_value, dict):
                        raise ValueError("Unexpected data `%s` in attribute `%s`" % (
                            type(attr_value), attr_name))
                    attr_value = attr_info.type(**attr_value)  # noqa: PLW2901
            setattr(self, attr_name, attr_value)

    def update(self, *objects: Union['ScyllaYaml', dict]):
        """
        Do the same as dict.update, with one exception.
        It ignores whatever key if it's value equal to default
        This comes from module `attr` and probably could be tackled there
        """
        fields_data = self.__fields__
        for obj in objects:
            if isinstance(obj, ScyllaYaml):
                for attr_name, attr_value in obj.__dict__.items():
                    attr_type = fields_data.get(attr_name)
                    if (attr_type and attr_value is not None) and attr_value != getattr(self, attr_name, None):
                        setattr(self, attr_name, attr_value)
            elif isinstance(obj, dict):
                self._update_dict(obj, fields_data=fields_data)
            else:
                raise ValueError("Only dict or ScyllaYaml is accepted")
        return self

    def diff(self, other: 'ScyllaYaml') -> str:
        self_str = yaml.safe_dump(self.dict(
            exclude_defaults=True, exclude_unset=True, exclude_none=True)).splitlines(keepends=True)
        other_str = yaml.safe_dump(other.dict(
            exclude_defaults=True, exclude_unset=True, exclude_none=True)).splitlines(keepends=True)
        return "".join(unified_diff(self_str, other_str))

    def __copy__(self):
        return self.__class__(**self.dict(exclude_defaults=True, exclude_unset=True))

    copy = __copy__
