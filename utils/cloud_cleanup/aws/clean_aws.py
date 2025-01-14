#!/usr/bin/env python

import argparse
import datetime
import os
import time
import logging
import sys
import boto3
import pytz

from utils.cloud_cleanup import update_argus_resource_status

DRY_RUN = False
VERBOSE = False
TRACE = False
WAIT = 60
DEFAULT_KEEP_HOURS = 14


def is_running(instance):
    return instance.state['Name'] == 'running'


def is_stopped(instance):
    return instance.state['Name'] == 'stopped'


def is_terminated(instance):
    return instance.state['Name'] == 'terminated'


def debug(msg):
    if VERBOSE:
        print(msg)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def since_now(since: datetime.datetime):
    return datetime.datetime.now(tz=pytz.utc) - since


def since_now_hours(since: datetime.datetime):
    return abs(since_now(since)).total_seconds() / 3600.0


def keep_alive_instance_duration(instance, duration):
    return duration > 0 and since_now_hours(instance.launch_time) < duration


def keep_alive_tag_val(instance):
    return get_tag_value(instance, 'keep')


def keep_alive_action_tag_val(instance):
    return get_tag_value(instance, 'keep_action')


def get_tag_value(instance, key):
    if instance.tags is not None:
        for tag in instance.tags:
            if tag['Key'].lower() == key.lower():
                return tag['Value']
    return ""


def keep_alive_tag_int(instance):
    keep = keep_alive_tag_val(instance)
    if keep.isdigit():
        return int(keep)
    return 0


def should_terminate(instance):
    action = keep_alive_action_tag_val(instance)
    return TERMINATE if action == '' else action == 'terminate'

# check tags for keep:alive


def keep_alive_instance(instance):
    return keep_alive_tag_val(instance).lower() == 'alive'


# check tags for keep:num hours since instance creation
def keep_alive_instance_launch_time(instance):
    keep = keep_alive_tag_int(instance)
    return keep_alive_instance_duration(instance, keep)


def stop_instance(instance):
    try:
        test_id = get_tag_value(instance, 'TestId')
        name = get_tag_value(instance, 'Name')
        if not DRY_RUN:
            instance.create_tags(Tags=[
                {
                    'Key': 'keep_alive_action',
                    'Value': 'stop'
                }
            ])
            instance.stop()
            update_argus_resource_status(test_id, name, 'terminate')
    except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
        eprint("stop instance %s error: %s" % (instance.id, str(exc)))


def remove_protection(instance):
    try:

        if not DRY_RUN:
            instance.modify_attribute(
                DisableApiTermination={
                    'Value': False
                })
            print_instance(instance, "Disabling API Termination protection")
    except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
        eprint("DisableApiTermination protection %s error: %s" % (instance.id, str(exc)))


def terminate_instance(instance):
    try:
        test_id = get_tag_value(instance, 'TestId')
        name = get_tag_value(instance, 'Name')
        if not DRY_RUN:
            instance.create_tags(Tags=[
                {
                    'Key': 'keep_alive_action',
                    'Value': 'terminate'
                }
            ])
            instance.terminate()
            update_argus_resource_status(test_id, name, 'terminate')
    except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
        eprint("terminate instance %s error: %s" % (instance.id, str(exc)))


def print_instance(instance, msg):
    print("instance %s type %s launched at %s keys %s %s" % (
        instance.id, instance.instance_type,
        instance.launch_time, instance.key_name,
        msg))


def regions_names():
    session = boto3.Session()
    default_region = session.region_name
    if not default_region:
        default_region = "eu-central-1"
    client = session.client('ec2', region_name=default_region)
    return [region['RegionName'] for region in client.describe_regions()['Regions']]


def scan_region_instances(region_name, duration):
    ec2 = boto3.resource('ec2', region_name=region_name)
    instances = ec2.instances.filter(Filters=[{
        'Name': 'instance-state-name',
        'Values': ['running']
    }])

    running, keep_alive, expiring, expired = 0, 0, 0, []

    for instance in instances:
        debug("checking instance %s, type %s, tags %s, placement %s, launch_time %s" % (
            instance.id, instance.instance_type, instance.tags, instance.placement, instance.launch_time))

        kalive = keep_alive_instance(instance)  # keep:alive
        ka_lt = keep_alive_instance_launch_time(instance)  # keep:X_HOURS
        ka_min = keep_alive_instance_duration(instance, duration)

        if kalive or ka_lt or ka_min:
            if kalive or ka_lt:
                keep_alive = keep_alive + 1
                if VERBOSE:
                    print_instance(instance, "keep alive")
            if ka_lt:
                expiring = expiring + 1
            running = running + 1
            debug("will not be stopped")
        else:
            expired.append(instance)

    return running, keep_alive, expiring, expired


def clean_instances(region_name, duration):
    print("cleaning region %s instances" % region_name)
    running, keep_alive, expiring, expired = scan_region_instances(region_name, duration)
    terminated = 0
    # stop instances
    for instance in expired:
        if is_running(instance):
            if should_terminate(instance):
                print_instance(instance, "terminating...")
                terminated = terminated + 1
                remove_protection(instance)
                terminate_instance(instance)
            else:
                print_instance(instance, "stopping...")
                remove_protection(instance)
                stop_instance(instance)

    # blind wait stop timeout
    if len(expired) > 0 and not DRY_RUN:
        print("region %s: waiting %d instances to stop" % (region_name, len(expired)))
        time.sleep(WAIT)

    # check instances are stopped. if they are not - terminate them
    for instance in expired:
        instance.reload()
        if not is_stopped(instance) and not is_terminated(instance):
            print_instance(instance, "terminating...")
            terminate_instance(instance)

    print("region %s: stopped %d, terminated %d, kept alive %d and expiring %d, running %d instances" % (
        region_name, len(expired) - terminated, terminated, keep_alive, expiring, running))


def keep_alive_volume(volume):
    if volume.create_time < datetime.datetime.now(tz=pytz.utc) + datetime.timedelta(hours=1):
        # skipping if created recently and might miss tags yet
        return True
    # checking tags
    if volume.tags is None:
        return False
    for tag in volume.tags:
        if tag['Key'] == 'keep' and tag['Value'] == 'alive':
            return True
    return False


def delete_volume(volume):
    try:
        print_volume(volume, "deleting")
        if not DRY_RUN:
            volume.delete()
    except Exception:  # pylint: disable=broad-except  # noqa: BLE001
        pass


def print_volume(volume, msg):
    print("volume %s %s" % (volume.id, msg))


def clean_volumes(region_name):
    print("cleaning region %s volumes" % region_name)
    ec2 = boto3.resource('ec2', region_name=region_name)

    volumes = ec2.volumes.all()

    count_kept_volume = 0
    count_deleted_volume = 0
    for volume in volumes:
        debug("checking volume %s %s %s %s %s" %
              (volume.id, volume.volume_id, volume.state, volume.tags, volume.create_time))
        keep_alive = keep_alive_volume(volume)
        if keep_alive or volume.state == "in-use":
            count_kept_volume = count_kept_volume + 1
            if VERBOSE:
                print_volume(volume, "kept")
        else:
            count_deleted_volume = count_deleted_volume + 1
            delete_volume(volume)
            if VERBOSE:
                print_volume(volume, "deleted")

    print("region %s deleted %d kept %d" % (region_name, count_deleted_volume, count_kept_volume))


def keep_alive_address(eip_dict):
    # checking tags
    if "Tags" not in eip_dict:
        return False
    for tag in eip_dict["Tags"]:
        if tag['Key'] == 'keep' and tag['Value'] == 'alive':
            return True
    return False


def release_address(eip_dict, client):
    try:
        print_adress(eip_dict, "deleting")
        if not DRY_RUN:
            if "AllocationId" in eip_dict:
                client.release_address(AllocationId=eip_dict['AllocationId'])
            elif "PublicIp" in eip_dict:
                client.release_address(PublicIp=eip_dict['PublicIp'])
    except Exception as exc:  # pylint: disable=broad-except  # noqa: BLE001
        print(exc)


def print_adress(eip_dict, msg):
    if "AllocationId" in eip_dict:
        print("Address ID %s %s" % (eip_dict["AllocationId"], msg))
    elif "PublicIp" in eip_dict:
        print("Address IP %s %s" % (eip_dict["PublicIp"], msg))


def clean_ips(region_name):
    print("cleaning region %s IP's" % region_name)
    client = boto3.client('ec2', region_name=region_name)
    addresses_dict = client.describe_addresses()
    deleted_addresses = 0
    kept_addresses = 0
    for eip_dict in addresses_dict['Addresses']:
        debug(eip_dict)
        if "NetworkInterfaceId" not in eip_dict and not keep_alive_address(eip_dict):
            deleted_addresses = deleted_addresses + 1
            release_address(eip_dict, client)
        else:
            kept_addresses = kept_addresses + 1
    print("region %s deleted %d ip addresses kept %s" % (region_name, deleted_addresses, kept_addresses))


def print_dedicate_host(host: dict, msg: str):
    print("dedicate host %s %s" % (host.get('HostId'), msg))


def keep_alive_host(host: dict):
    if datetime.datetime.now(tz=pytz.utc) - host.get('AllocationTime') < datetime.timedelta(hours=1):
        # skipping if created recently and might miss tags yet
        return True
    # checking tags
    if host.get('Tags') is None:
        return False
    for tag in host.get('Tags'):
        if tag['Key'] == 'keep' and tag['Value'] == 'alive':
            return True
    return False


def clean_dedicate_hosts(region_name):
    print("cleaning region %s dedicate_hosts" % region_name)
    ec2 = boto3.client('ec2', region_name=region_name)

    def delete_host(_host: dict):
        try:
            print_dedicate_host(_host, "deleting")
            if not DRY_RUN:
                ec2.release_hosts(HostIds=[_host.get['HostId'], ])
        except Exception:  # pylint: disable=broad-except  # noqa: BLE001
            pass

    count_kept_hosts = 0
    count_deleted_hosts = 0
    response = ec2.describe_hosts(Filters=[
        {
            'Name': 'state',
            'Values': ['available']
        }
    ]
    )

    for host in response['Hosts']:
        debug("checking host %s %s %s" %
              (host.get('HostId'), host.get('Instances'), host.get('AllocationTime')))
        keep_alive = keep_alive_host(host)

        if keep_alive or host.get('Instances'):
            count_kept_hosts += 1
            if VERBOSE:
                print_dedicate_host(host, "kept")
        else:
            count_deleted_hosts += + 1
            delete_host(host)
            if VERBOSE:
                print_dedicate_host(host, "deleted")

    print("region %s deleted %d kept %d" % (region_name, count_deleted_hosts, count_kept_hosts))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser('ec2_stop')
    arg_parser.add_argument("--duration", type=int,
                            help="duration to keep non-tagged instances running in hours",
                            default=os.environ.get('DURATION', DEFAULT_KEEP_HOURS))
    arg_parser.add_argument("--verbose", action="store_true",
                            help="print processing instances details",
                            default=os.environ.get('VERBOSE', False))
    arg_parser.add_argument("--trace", action="store_true",
                            help="trace every AWS call",
                            default=os.environ.get('TRACE', False))
    arg_parser.add_argument("--wait", type=int,
                            help="blind wait for instances to stop timeout",
                            default=os.environ.get('WAIT', 60))
    arg_parser.add_argument("--dry-run", action="store_true",
                            help="do not stop or terminate anything",
                            default=os.environ.get('DRY_RUN', False))
    arg_parser.add_argument("--default-action",
                            help="The default action when stopping an image (stop/terminate)",
                            default=os.environ.get('DEFAULT_ACTION', "terminate"))

    arguments = arg_parser.parse_args()

    VERBOSE = bool(arguments.verbose)
    WAIT = int(arguments.wait)
    DRY_RUN = bool(arguments.dry_run)
    TRACE = bool(arguments.trace)
    TERMINATE = arguments.default_action == 'terminate'

    if DRY_RUN:
        print("dry mode on")

    if TRACE:
        boto3.set_stream_logger(name='botocore')
        logging.getLogger('botocore').setLevel(logging.DEBUG)

    for region in regions_names():
        clean_instances(region, arguments.duration)
        clean_volumes(region)
        clean_ips(region)
        clean_dedicate_hosts(region)
