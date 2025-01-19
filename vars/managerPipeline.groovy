#!groovy

boolean jobEnabled (String jobName) {
	echo "Checking if Job $jobName exists / enabled"
	try {
		if (Jenkins.instance.getItemByFullName(jobName).isBuildable()) {
			echo "Job $jobName is enabled"
			return true
		} else {
			echo "Job $jobName is disabled, Skipping"
			return false
		}
	} catch (error) {
		echo "Error: General error |$error| while checking if job |$jobName| enabled (job does not exist)"
		return false
	}
}

def triggerJob(String jobToTrigger, def parameterList = [], boolean propagate = false, boolean wait = false) {
    if (jobEnabled(jobToTrigger)) {
        echo "Triggering '$jobToTrigger'"
        try {
            jobResults=build job: jobToTrigger,
                parameters: parameterList,
                propagate: propagate,  // if true, the triggering test will fail/pass based on the status of the triggered/downstream job/s
                wait: wait  // if true, the triggering job will not end until the triggered/downstream job/s will end
        } catch(Exception ex) {
            echo "Could not trigger jon $jobToTrigger due to"
            println(ex.toString())
        }
    }
}


def completed_stages = [:]
def (testDuration, testRunTimeout, runnerTimeout, collectLogsTimeout, resourceCleanupTimeout) = [0,0,0,0,0]

def call(Map pipelineParams) {

    def builder = getJenkinsLabels(params.backend, params.region, params.gce_datacenter, params.azure_region_name)

    pipeline {
        agent {
            label {
                label builder.label
            }
        }
        environment {
            AWS_ACCESS_KEY_ID     = credentials('qa-aws-secret-key-id')
            AWS_SECRET_ACCESS_KEY = credentials('qa-aws-secret-access-key')
            SCT_TEST_ID = UUID.randomUUID().toString()
        }
        parameters {
            string(defaultValue: "${pipelineParams.get('backup_bucket_backend', '')}",
               description: 's3|gcs|azure or empty',
               name: 'backup_bucket_backend')
            string(defaultValue: "${pipelineParams.get('backup_bucket_location', '')}",
               description: """Backup bucket name, if empty - the default 'manager-backup-tests-us-east-1' bucket is used.
                               'manager-backup-tests-permanent-snapshots-us-east-1' bucket can be used to store backups permanently.""",
               name: 'backup_bucket_location')
            string(defaultValue: "${pipelineParams.get('backend', 'aws')}",
               description: 'aws|gce',
               name: 'backend')
            string(defaultValue: "${pipelineParams.get('region', 'eu-west-1')}",
               description: 'Supported: us-east-1 | eu-west-1 | eu-west-2 | eu-north-1 | eu-central-1 | us-west-2 | random (randomly select region)',
               name: 'region')
            string(defaultValue: "${pipelineParams.get('gce_datacenter', 'us-east1')}",
                   description: 'GCE datacenter',
                   name: 'gce_datacenter')
            string(defaultValue: "${pipelineParams.get('azure_region_name', 'eastus')}",
                   description: 'Azure location',
                   name: 'azure_region_name')
            string(defaultValue: "a",
               description: 'Availability zone',
               name: 'availability_zone')


            string(defaultValue: '', description: '', name: 'scylla_ami_id')
            string(defaultValue: "${pipelineParams.get('scylla_version', '2024.2')}", description: '', name: 'scylla_version')
            // When branching to manager version branch, set scylla_version to the latest release
            string(defaultValue: '', description: '', name: 'scylla_repo')
            string(defaultValue: "${pipelineParams.get('gce_image_db', '')}",
                   description: "gce image of scylla (since scylla_version doesn't work with gce)",
                   name: 'gce_image_db')  // TODO: remove setting once hydra is able to discover scylla images in gce from scylla_version
            string(defaultValue: "${pipelineParams.get('azure_image_db', '')}",
                   description: '',
                   name: 'azure_image_db')
            string(defaultValue: "${pipelineParams.get('provision_type', 'spot')}",
                   description: 'spot|on_demand|spot_fleet',
                   name: 'provision_type')
            string(defaultValue: "${pipelineParams.get('instance_provision_fallback_on_demand', 'false')}",
                   description: 'true|false',
                   name: 'instance_provision_fallback_on_demand')

            string(defaultValue: "${pipelineParams.get('post_behavior_db_nodes', 'keep-on-failure')}",
                   description: 'keep|keep-on-failure|destroy',
                   name: 'post_behavior_db_nodes')
            string(defaultValue: "${pipelineParams.get('post_behavior_loader_nodes', 'destroy')}",
                   description: 'keep|keep-on-failure|destroy',
                   name: 'post_behavior_loader_nodes')
            string(defaultValue: "${pipelineParams.get('post_behavior_monitor_nodes', 'keep-on-failure')}",
                   description: 'keep|keep-on-failure|destroy',
                   name: 'post_behavior_monitor_nodes')

            string(defaultValue: "${pipelineParams.get('ip_ssh_connections', 'private')}",
                   description: 'private|public|ipv6',
                   name: 'ip_ssh_connections')

            string(defaultValue: "${pipelineParams.get('manager_version', 'master_latest')}",
                   description: 'master_latest|3.4|3.3',
                   name: 'manager_version')

            string(defaultValue: "${pipelineParams.get('scylla_mgmt_address', '')}",
                   description: 'If empty - the default manager version will be taken',
                   name: 'scylla_mgmt_address')

            string(defaultValue: "${pipelineParams.get('scylla_mgmt_agent_address', '')}",
                   description: 'manager agent repo',
                   name: 'scylla_mgmt_agent_address')

            string(defaultValue: "${pipelineParams.get('scylla_mgmt_pkg', '')}",
                   description: 'Url to the scylla manager packages',
                   name: 'scylla_mgmt_pkg')

            string(defaultValue: "${pipelineParams.get('target_manager_version', '')}",
                   description: 'master_latest|3.4|3.3. Only for upgrade test',
                   name: 'target_manager_version')

            string(defaultValue: "${pipelineParams.get('target_scylla_mgmt_server_address', '')}",
                   description: 'Link to the repository of the manager that will be used as a target of the manager server in the manager upgrade test',
                   name: 'target_scylla_mgmt_server_address')

            string(defaultValue: "${pipelineParams.get('target_scylla_mgmt_agent_address', '')}",
                   description: 'Link to the repository of the manager that will be used as a target of the manager agents in the manager upgrade test',
                   name: 'target_scylla_mgmt_agent_address')

            string(defaultValue: "'qa@scylladb.com','mgmt@scylladb.com'",
                   description: 'email recipients of email report',
                   name: 'email_recipients')

            string(defaultValue: "${pipelineParams.get('test_config', '')}",
                   description: 'Test configuration file',
                   name: 'test_config')

            string(defaultValue: "${pipelineParams.get('test_name', '')}",
                   description: 'Name of the test to run',
                   name: 'test_name')

            string(defaultValue: "${pipelineParams.get('mgmt_reuse_backup_snapshot_name', '')}",
                   description: """Name of backup snapshot to use in Manager restore benchmark test, for example, 500gb_2t_ics.
                                   The name provides the info about dataset size (500gb), number of tables (2) and compaction (ICS).
                                   All snapshots are defined in the 'https://github.com/scylladb/scylla-cluster-tests/blob/master/defaults/manager_restore_benchmark_snapshots.yaml'""",
                   name: 'mgmt_reuse_backup_snapshot_name')

            string(defaultValue: "${pipelineParams.get('mgmt_agent_backup_config', '')}",
                   description: """Backup general configuration for the agent (scylla-manager-agent.yaml):
                                   checkers, transfers, low_level_retries.
                                   For example, {'checkers': 100, 'transfers': 2, 'low_level_retries': 20}""",
                   name: 'mgmt_agent_backup_config')

            string(defaultValue: "${pipelineParams.get('keyspace_num', '')}",
                   description: 'Number of keyspaces to create. If > 1, cassandra-stress inserts the data into each keyspace',
                   name: 'keyspace_num')

            string(defaultValue: "${pipelineParams.get('downstream_jobs_to_run', '')}",
                   description: 'Comma separated list of downstream jobs to run when the job passes',
                   name: 'downstream_jobs_to_run')
            string(defaultValue: '',
                   description: 'Actual user requesting job start, for automated job builds (e.g. through Argus)',
                   name: 'requested_by_user')
            text(defaultValue: "${pipelineParams.get('extra_environment_variables', '')}",
                 description: (
                     """Extra environment variables to be set in the test environment, uses the java Properties File Format.
                        Example:
                        SCT_MGMT_RESTORE_EXTRA_PARAMS=--batch-size 50 --parallel 0
                        SCT_N_DB_NODES=6
                        SCT_MGMT_SKIP_POST_RESTORE_STRESS_READ=true
                        SCT_MGMT_SNAPSHOTS_PREPARER_PARAMS={'compaction': 'SizeTieredCompactionStrategy', 'rf': 3}"""
                     ),
                 name: 'extra_environment_variables')
        }
        options {
            timestamps()
            disableConcurrentBuilds()
            buildDiscarder(logRotator(numToKeepStr: '20'))
        }
        stages {
            stage('Checkout') {
                options {
                    timeout(time: 5, unit: 'MINUTES')
                }
                steps {
                    script {
                        loadEnvFromString(params.extra_environment_variables)
                        completed_stages = [:]
                    }
                    dir('scylla-cluster-tests') {
                        checkout scm
                        checkoutQaInternal(params)
                    }
                    dockerLogin(params)
               }
            }
            stage('Create Argus Test Run') {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    timeout(time: 5, unit: 'MINUTES') {
                                        createArgusTestRun(params)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            stage('Get test duration') {
                options {
                    timeout(time: 10, unit: 'MINUTES')
                }
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    (testDuration, testRunTimeout, runnerTimeout, collectLogsTimeout, resourceCleanupTimeout) = getJobTimeouts(params, builder.region)
                                }
                            }
                        }
                    }
                }
            }
            stage('Create SCT Runner') {
                options {
                    timeout(time: 5, unit: 'MINUTES')
                }
                steps {
                    script {
                        wrap([$class: 'BuildUser']) {
                            dir('scylla-cluster-tests') {
                                createSctRunner(params, runnerTimeout , builder.region)
                            }
                        }
                    }
                }
            }
            stage('Provision Resources') {
                steps {
                    catchError() {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    timeout(time: 30, unit: 'MINUTES') {
                                        if (params.backend == 'aws' || params.backend == 'azure') {
                                            provisionResources(params, builder.region)
                                        } else if (params.backend.contains('docker')) {
                                            sh """
                                                echo 'Tests are to be executed on Docker backend in SCT-Runner. No additional resources to be provisioned.'
                                            """
                                        } else {
                                            sh """
                                                echo 'Skipping because non-AWS/Azure backends are not supported'
                                            """
                                        }
                                        completed_stages['provision_resources'] = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
            stage('Run SCT Test') {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                timeout(time: testRunTimeout, unit: 'MINUTES') {
                                    dir('scylla-cluster-tests') {

                                        // handle params which can be a json list
                                        def region = initAwsRegionParam(params.region, builder.region)
                                        def datacenter = groovy.json.JsonOutput.toJson(params.gce_datacenter)
                                        def test_config = groovy.json.JsonOutput.toJson(params.test_config)
                                        def cloud_provider = params.backend.trim().toLowerCase()

                                        sh """#!/bin/bash
                                        set -xe
                                        env
                                        rm -fv ./latest

                                        export SCT_CLUSTER_BACKEND="${params.backend}"
                                        export SCT_REGION_NAME=${region}
                                        export SCT_GCE_DATACENTER=${datacenter}
                                        if [[ -n "${params.azure_region_name ? params.azure_region_name : ''}" ]] ; then
                                            export SCT_AZURE_REGION_NAME=${params.azure_region_name}
                                        fi
                                        export SCT_CONFIG_FILES=${test_config}
                                        export SCT_COLLECT_LOGS=false

                                        if [[ -n "${params.backup_bucket_backend}" ]] ; then
                                            export SCT_BACKUP_BUCKET_BACKEND="${params.backup_bucket_backend}"
                                        fi

                                        if [[ -n "${params.backup_bucket_location}" ]] ; then
                                            export SCT_BACKUP_BUCKET_LOCATION="${params.backup_bucket_location}"
                                        fi

                                        if [[ ! -z "${params.scylla_ami_id}" ]] ; then
                                            export SCT_AMI_ID_DB_SCYLLA="${params.scylla_ami_id}"
                                        elif [[ ! -z "${params.scylla_version}" ]] ; then
                                            export SCT_SCYLLA_VERSION="${params.scylla_version}"
                                        elif [[ ! -z "${params.gce_image_db}" ]] ; then
                                            export SCT_GCE_IMAGE_DB="${params.gce_image_db}"  #TODO: remove it once scylla_version supports gce image detection
                                        elif [[ ! -z "${params.azure_image_db}" ]] ; then
                                            export SCT_AZURE_IMAGE_DB="${params.azure_image_db}"  #TODO: remove it once scylla_version supports azure image detection
                                        elif [[ ! -z "${params.scylla_repo}" ]] ; then
                                            export SCT_SCYLLA_REPO="${params.scylla_repo}"
                                        else
                                            echo "need to choose one of SCT_AMI_ID_DB_SCYLLA | SCT_GCE_IMAGE_DB | SCT_SCYLLA_VERSION | SCT_SCYLLA_REPO | SCT_AZURE_IMAGE_DB"
                                            exit 1
                                        fi
                                        if [[ -n "${params.availability_zone ? params.availability_zone : ''}" ]] ; then
                                            export SCT_AVAILABILITY_ZONE="${params.availability_zone}"
                                        fi
                                        export SCT_POST_BEHAVIOR_DB_NODES="${params.post_behavior_db_nodes}"
                                        export SCT_POST_BEHAVIOR_LOADER_NODES="${params.post_behavior_loader_nodes}"
                                        export SCT_POST_BEHAVIOR_MONITOR_NODES="${params.post_behavior_monitor_nodes}"
                                        export SCT_INSTANCE_PROVISION="${params.provision_type}"
                                        export SCT_AMI_ID_DB_SCYLLA_DESC=\$(echo \$GIT_BRANCH | sed -E 's+(origin/|origin/branch-)++')
                                        export SCT_AMI_ID_DB_SCYLLA_DESC=\$(echo \$SCT_AMI_ID_DB_SCYLLA_DESC | tr ._ - | cut -c1-8 )

                                        export SCT_IP_SSH_CONNECTIONS="${params.ip_ssh_connections}"

                                        if [[ ! -z "${params.scylla_mgmt_address}" ]] ; then
                                            export SCT_SCYLLA_MGMT_ADDRESS="${params.scylla_mgmt_address}"
                                        fi

                                        if [[ ! -z "${params.manager_version}" ]] ; then
                                            export SCT_MANAGER_VERSION="${params.manager_version}"
                                        fi

                                        if [[ ! -z "${params.target_manager_version}" ]] ; then
                                            export SCT_TARGET_MANAGER_VERSION="${params.target_manager_version}"
                                        fi

                                        if [[ ! -z "${params.target_scylla_mgmt_server_address}" ]] ; then
                                            export SCT_TARGET_SCYLLA_MGMT_SERVER_ADDRESS="${params.target_scylla_mgmt_server_address}"
                                        fi

                                        if [[ ! -z "${params.target_scylla_mgmt_agent_address}" ]] ; then
                                            export SCT_TARGET_SCYLLA_MGMT_AGENT_ADDRESS="${params.target_scylla_mgmt_agent_address}"
                                        fi

                                        if [[ ! -z "${params.scylla_mgmt_agent_address}" ]] ; then
                                            export SCT_SCYLLA_MGMT_AGENT_ADDRESS="${params.scylla_mgmt_agent_address}"
                                        fi

                                        if [[ ! -z "${params.scylla_mgmt_pkg}" ]] ; then
                                            export SCT_SCYLLA_MGMT_PKG="${params.scylla_mgmt_pkg}"
                                        fi

                                        if [[ ! -z "${params.mgmt_reuse_backup_snapshot_name}" ]] ; then
                                            export SCT_MGMT_REUSE_BACKUP_SNAPSHOT_NAME="${params.mgmt_reuse_backup_snapshot_name}"
                                        fi

                                        if [[ ! -z "${params.mgmt_agent_backup_config}" ]] ; then
                                            export SCT_MGMT_AGENT_BACKUP_CONFIG="${params.mgmt_agent_backup_config}"
                                        fi

                                        if [[ ! -z "${params.keyspace_num}" ]] ; then
                                            export SCT_KEYSPACE_NUM="${params.keyspace_num}"
                                        fi

                                        echo "start test ......."
                                        RUNNER_IP=\$(cat sct_runner_ip||echo "")
                                        if [[ -n "\${RUNNER_IP}" ]] ; then
                                            ./docker/env/hydra.sh --execute-on-runner \${RUNNER_IP} run-test ${params.test_name} --backend ${params.backend}
                                        else
                                            ./docker/env/hydra.sh run-test ${params.test_name} --backend ${params.backend}  --logdir "`pwd`"
                                        fi
                                        echo "end test ....."
                                        """
                                        completed_stages['run_tests'] = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
            stage('Running Downstream Jobs') {  // Specifically placed after test stage, since downstream jobs should still be triggered when stages like collect logs fail.
                options {
                    timeout(time: 5, unit: 'MINUTES')
                }
                steps {
                    script {
                        if (currentBuild.currentResult == 'SUCCESS') {
                            jobNamesToTrigger = params.downstream_jobs_to_run.split(',')
                            currentJobDirectoryPath = JOB_NAME.substring(0, JOB_NAME.lastIndexOf('/'))
                            for (downstreamJobName in jobNamesToTrigger) {
                                fullJobPath = currentJobDirectoryPath + '/' + downstreamJobName.trim()
                                def repoParams = []
                                if (downstreamJobName.contains("upgrade")) {
                                    repoParams = [
                                        [$class: 'StringParameterValue', name: 'target_scylla_mgmt_server_address', value: params.scylla_mgmt_address],
                                        [$class: 'StringParameterValue', name: 'target_scylla_mgmt_agent_address', value: params.scylla_mgmt_agent_address],
                                        [$class: 'StringParameterValue', name: 'TARGET_MANAGER_VERSION', value: params.manager_version],
                                        [$class: 'StringParameterValue', name: 'provision_type', value: params.provision_type]
                                    ]
                                } else {
                                    repoParams = [
                                        [$class: 'StringParameterValue', name: 'scylla_mgmt_address', value: params.scylla_mgmt_address],
                                        [$class: 'StringParameterValue', name: 'scylla_mgmt_agent_address', value: params.scylla_mgmt_agent_address],
                                        [$class: 'StringParameterValue', name: 'manager_version', value: params.manager_version],
                                        [$class: 'StringParameterValue', name: 'provision_type', value: params.provision_type]
                                    ]
                                }
                                triggerJob(fullJobPath, repoParams)
                            }
                        } else {
                            echo "Job failed. Will not run downstream jobs."
                        }
                    }
                }
            }
            stage("Collect log data") {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    timeout(time: collectLogsTimeout, unit: 'MINUTES') {
                                        runCollectLogs(params, builder.region)
                                        completed_stages['collect_logs'] = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
            stage('Clean resources') {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    timeout(time: resourceCleanupTimeout, unit: 'MINUTES') {
                                        runCleanupResource(params, builder.region)
                                        completed_stages['clean_resources'] = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
            stage("Send email with result") {
                options {
                    timeout(time: 10, unit: 'MINUTES')
                }
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    runSendEmail(params, currentBuild)
                                    completed_stages['send_email'] = true
                                }
                            }
                        }
                    }
                }
            }
            stage('Clean SCT Runners') {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    cleanSctRunners(params, currentBuild)
                                    completed_stages['clean_sct_runner'] = true
                                }
                            }
                        }
                    }
                }
            }
            stage('Finish Argus Test Run') {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    timeout(time: 5, unit: 'MINUTES') {
                                        finishArgusTestRun(params, currentBuild)
                                        completed_stages['report_to_argus'] = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        post {
            always {
                script {
                    def provision_resources = completed_stages['provision_resources']
                    def run_tests = completed_stages['run_tests']
                    def collect_logs = completed_stages['collect_logs']
                    def clean_resources = completed_stages['clean_resources']
                    def send_email = completed_stages['send_email']
                    def clean_sct_runner = completed_stages['clean_sct_runner']
                    sh """
                        echo "'provision_resources' stage is completed: $provision_resources"
                        echo "'run_tests' stage is completed: $run_tests"
                        echo "'collect_logs' stage is completed: $collect_logs"
                        echo "'clean_resources' stage is completed: $clean_resources"
                        echo "'send_email' stage is completed: $send_email"
                        echo "'clean_sct_runner' stage is completed: $clean_sct_runner"
                    """
                    if (!completed_stages['clean_resources']) {
                        catchError {
                            script {
                                wrap([$class: 'BuildUser']) {
                                    dir('scylla-cluster-tests') {
                                        timeout(time: resourceCleanupTimeout, unit: 'MINUTES') {
                                            runCleanupResource(params, builder.region)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (!completed_stages['send_email']) {
                        catchError {
                            script {
                                wrap([$class: 'BuildUser']) {
                                    dir('scylla-cluster-tests') {
                                        timeout(time: 10, unit: 'MINUTES') {
                                            runSendEmail(params, currentBuild)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (!completed_stages['report_to_argus']) {
                        catchError {
                            script {
                                wrap([$class: 'BuildUser']) {
                                    dir('scylla-cluster-tests') {
                                        timeout(time: 5, unit: 'MINUTES') {
                                            finishArgusTestRun(params, currentBuild)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
