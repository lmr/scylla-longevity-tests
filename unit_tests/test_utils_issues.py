import json

import pytest

from sdcm.utils.issues import SkipPerIssues

pytestmark = pytest.mark.integration


def test_no_existing_issue(events, params):
    params.artifact_scylla_version = '2023.1.3'
    with events.wait_for_n_events(events.get_events_logger(), count=1, timeout=10):
        assert not SkipPerIssues(["ABC"], params)

    events_list = []
    with events.get_raw_events_log().open() as events_file:
        for line in events_file.readlines():
            events_list.append(json.loads(line))

    assert events_list[-1]['message'] == "couldn't parse issue: ABC"
    assert events_list[-1]['severity'] == 'WARNING'


def test_open_issues(params):
    params.artifact_scylla_version = '2023.1.3'
    assert SkipPerIssues(["https://github.com/scylladb/qa-tasks/issues/1613",
                          "scylladb/qa-tasks#1613", ], params)


def test_closed_issues(params):
    params.artifact_scylla_version = '2023.1.3'
    assert not SkipPerIssues(
        ["https://github.com/scylladb/qa-tasks/issues/1614",
         "scylladb/qa-tasks#1614"], params)


def test_closed_issues_with_skip_tag(params):
    params.artifact_scylla_version = '2023.1.3'
    assert SkipPerIssues(
        ["https://github.com/scylladb/qa-tasks/issues/1615",
         "scylladb/qa-tasks#1615"], params)


def test_no_version(params):
    params.artifact_scylla_version = None
    assert SkipPerIssues(
        ["https://github.com/scylladb/qa-tasks/issues/1613",
         "scylladb/qa-tasks#1613"], params)


def test_opened_issue_with_closed_issue(params):
    params.artifact_scylla_version = '2023.1.3'
    assert SkipPerIssues(
        ["scylladb/qa-tasks#1613", "scylladb/qa-tasks#1614"], params)

    assert SkipPerIssues(
        ["scylladb/qa-tasks#1614", "scylladb/qa-tasks#1613"], params)


def test_closed_issue_with_tag_not_matching_version(params):
    params.artifact_scylla_version = '2019.1.3'
    assert SkipPerIssues(
        ["scylladb/qa-tasks#1613", "scylladb/qa-tasks#1615"], params)

    assert SkipPerIssues(
        ["scylladb/qa-tasks#1615", "scylladb/qa-tasks#1613"], params)


def test_opened_issue_tag_not_matching_version_tag(params):
    params.artifact_scylla_version = '2019.1.3'

    assert not SkipPerIssues(
        ["scylladb/qa-tasks#1615"], params)


def test_closed_pull_request(params):
    params.artifact_scylla_version = '2019.1.3'

    assert not SkipPerIssues(
        ["scylladb/scylla-cluster-tests#7832"], params)
