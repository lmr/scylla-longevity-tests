from __future__ import annotations
from threading import RLock
from contextlib import contextmanager, ExitStack
import random
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sdcm.cluster import BaseNode

# this lock must use when you are checking id nod free or not(when calling  node.lock.locked())
# so you can mace sure that of node free, only ou you about in and at the acquire() call node still will be free
# for acquire() node that already locked NEMESIS_TARGET_SELECTION_LOCK should not be used
NEMESIS_TARGET_SELECTION_LOCK = RLock()

# this lock need to support current nemesis logic in "set_target_node "when nemesis call release any lock rather than releasing only own locks
NEMESIS_TARGET_RELEASE_LOCK = RLock()


# the following 2 Exception just to avoid using wide AssertionError
class CantAcquireLockException(Exception):
    pass


class CantReleaseLockException(Exception):
    pass


class AlreadyAcquireLockException(Exception):
    pass


def unset_running_nemesis(node, nemesis=None, raise_on_nemesis_mismatching=True):
    # unlock node
    with NEMESIS_TARGET_RELEASE_LOCK:
        if node is not None and node.lock.locked():
            if nemesis and nemesis != node.running_nemesis and raise_on_nemesis_mismatching:
                raise CantReleaseLockException(
                    f"node locked by another nemesis, expected {nemesis}, got {node.running_nemesis}")
            node.lock.release()
            node.running_nemesis = None


def set_running_nemesis(node, nemesis, timeout=30, raise_on_nemesis_already_acquire_lock=True):
    # use NEMESIS_TARGET_SELECTION_LOCK in case of locking free node
    # if nade are lock, release NEMESIS_TARGET_SELECTION_LOCK and wait for lock by acquire
    with NEMESIS_TARGET_SELECTION_LOCK:
        if not node.lock.locked():
            #node is free
            assert node.lock.acquire(timeout=5)
            node.running_nemesis = nemesis
        elif node.running_nemesis != nemesis:
            # node locked by another nemesis
            if not node.lock.acquire(timeout=timeout):
                raise CantAcquireLockException(f"cant lock node within given timeout: {timeout}")
            node.running_nemesis = nemesis
        elif raise_on_nemesis_already_acquire_lock:
            # node already locked by this nemesis
            raise AlreadyAcquireLockException(f"lock already acquired by nemesis name '{nemesis}'")


@contextmanager
def run_nemesis(node_list: list[BaseNode], nemesis_label: str):
    """
    pick a free node out of a `node_list`, and lock it
    for the duration of this context
    """
    with NEMESIS_TARGET_SELECTION_LOCK:
        free_nodes = [node for node in node_list if not node.lock.locked()]
        assert free_nodes, f"couldn't find nodes for running:`{nemesis_label}`, are all nodes running nemesis ?"
        node = random.choice(free_nodes)
        set_running_nemesis(node, nemesis_label)
    try:
        yield node
    finally:
        unset_running_nemesis(node, nemesis_label)


@contextmanager
def lock_node(node: BaseNode, nemesis_label: str, timeout=30):
    try:
        set_running_nemesis(node, nemesis_label, timeout)
        try:
            yield
        finally:
            unset_running_nemesis(node, nemesis_label)
    except AlreadyAcquireLockException:
        # lock already Acquire somewhere else, do not acquire/release
        yield
    except CantAcquireLockException:
        unset_running_nemesis(node, nemesis_label, False)
        raise


@contextmanager
def lock_nodes(nodelist, nemesis_label: str, timeout=30):
    with ExitStack() as stack:
        [stack.enter_context(lock_node(x, nemesis_label, timeout)) for x in nodelist]
        yield
