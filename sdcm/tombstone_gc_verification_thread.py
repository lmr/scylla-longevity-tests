import datetime
import logging
import threading
import time

from sdcm import wait
from sdcm.cluster import BaseScyllaCluster, BaseCluster
from sdcm.remote import LocalCmdRunner
from sdcm.sct_events import Severity
from sdcm.sct_events.database import TombstoneGcVerificationEvent
from sdcm.utils.sstable.sstable_utils import SstableUtils

LOCAL_CMD_RUNNER = LocalCmdRunner()
ERROR_SUBSTRINGS = ("timed out", "timeout")


# pylint: disable=too-many-instance-attributes
class TombstoneGcVerificationThread:

    # pylint: disable=too-many-arguments

    def __init__(self, db_cluster: [BaseScyllaCluster, BaseCluster], duration: int, interval: int,
                 termination_event: threading.Event, **kwargs):

        self._sstable_utils = SstableUtils(**kwargs)
        self.duration = duration
        self.interval = interval
        self.termination_event = termination_event
        self._thread = threading.Thread(daemon=True, name=self.__class__.__name__, target=self.run)
        self.db_cluster: [BaseScyllaCluster, BaseCluster] = db_cluster
        node = next(node for node in self.db_cluster.data_nodes if node.db_up())
        self._sstable_utils = SstableUtils(db_node=node, **kwargs)
        self.log = logging.getLogger(self.__class__.__name__)

    def _wait_until_user_table_exists(self, db_node, table_name: str = 'random', timeout_min: int = 20):
        text = f'Waiting until {table_name} user table exists'
        if table_name.lower() == 'random':
            wait.wait_for(func=lambda: len(self.db_cluster.get_non_system_ks_cf_list(db_node)) > 0, step=60,
                          text=text, timeout=60 * timeout_min, throw_exc=True)
        else:
            wait.wait_for(func=lambda: table_name in (self.db_cluster.get_non_system_ks_cf_list(db_node)), step=60,
                          text=text, timeout=60 * timeout_min, throw_exc=True)

    def tombstone_gc_verification(self, max_sstable_num: int = 50):
        table_repair_date = self._sstable_utils.get_table_repair_date()  # Example: 2022-12-28 11:53:53
        if not table_repair_date:
            return
        table_repair_date = datetime.datetime.strptime(table_repair_date, '%Y-%m-%d %H:%M:%S')
        delta_repair_date_minutes = (int((
            datetime.datetime.now() - table_repair_date).total_seconds()) - self._sstable_utils.propagation_delay_in_seconds) // 60
        self.log.debug(
            "Calculated delta_repair_date_minutes: %d, table_repair_date: %s, propagation_delay: %d",
            delta_repair_date_minutes,
            table_repair_date,
            self._sstable_utils.propagation_delay_in_seconds,
        )

        if delta_repair_date_minutes <= 0:
            self.log.debug('Table %s repair date is smaller than propagation delay, aborting.',
                           self._sstable_utils.ks_cf)
            return
        sstables = self._sstable_utils.get_sstables(from_minutes_ago=delta_repair_date_minutes)
        if not sstables:
            self.log.warning('No sstable with a creation time of last %s minutes found, aborting.',
                             delta_repair_date_minutes)
            return
        self.log.debug('Starting sstabledump to verify correctness of tombstones for %s sstables',
                       len(sstables))
        if max_sstable_num < len(sstables):
            sstables = sstables[:max_sstable_num]
        for sstable in sstables:
            self._sstable_utils.verify_post_repair_sstable_tombstones(
                table_repair_date=table_repair_date, sstable=sstable)

    def _run_tombstone_gc_verification(self):
        db_node = self._sstable_utils.db_node
        self._wait_until_user_table_exists(db_node=db_node, table_name=self._sstable_utils.ks_cf)
        with TombstoneGcVerificationEvent(node=db_node.name, ks_cf=self._sstable_utils.ks_cf, message="") as tombstone_event:
            if self.termination_event.is_set():
                return

            try:
                self.tombstone_gc_verification()
                tombstone_event.message = "Tombstone GC verification ended successfully"
            except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
                msg = str(exc)
                msg = f"{msg} while running Nemesis: {db_node.running_nemesis}" if db_node.running_nemesis else msg
                tombstone_event.message = msg

                if db_node.running_nemesis or any(s in msg.lower() for s in ERROR_SUBSTRINGS):
                    tombstone_event.severity = Severity.WARNING
                else:
                    tombstone_event.severity = Severity.ERROR

    def run(self):
        end_time = time.time() + self.duration
        while time.time() < end_time and not self.termination_event.is_set():
            self._sstable_utils.verify_a_live_normal_node_is_used()
            self._run_tombstone_gc_verification()
            self.log.debug('Executed %s', TombstoneGcVerificationEvent.__name__)
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        return self._thread.join(timeout)
