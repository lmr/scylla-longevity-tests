#!groovy

def completed_stages = [:]


def call(Map pipelineParams) {

    def builder = getJenkinsLabels(params.backend, params.aws_region)

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
            string(defaultValue: "${pipelineParams.get('backend', 'aws')}",
               description: 'aws|gce',
               name: 'backend')

            string(defaultValue: "${pipelineParams.get('aws_region', 'eu-west-1')}",
               description: 'Supported: us-east-1|eu-west-1|eu-west-2|eu-north-1|random (randomly select region)',
               name: 'aws_region')
            string(defaultValue: "a",
               description: 'Availability zone',
               name: 'availability_zone')

            string(defaultValue: '', description: '', name: 'scylla_ami_id')
            string(defaultValue: '', description: '', name: 'scylla_version')
            string(defaultValue: '', description: '', name: 'scylla_repo')
            string(defaultValue: "${pipelineParams.get('provision_type', 'spot_low_price')}",
                   description: 'spot_low_price|on_demand|spot_fleet|spot_duration',
                   name: 'provision_type')

            string(defaultValue: "${pipelineParams.get('post_behavior_db_nodes', 'keep-on-failure')}",
                   description: 'keep|keep-on-failure|destroy',
                   name: 'post_behavior_db_nodes')
            string(defaultValue: "${pipelineParams.get('post_behavior_loader_nodes', 'destroy')}",
                   description: 'keep|keep-on-failure|destroy',
                   name: 'post_behavior_loader_nodes')
            string(defaultValue: "${pipelineParams.get('post_behavior_monitor_nodes', 'keep-on-failure')}",
                   description: 'keep|keep-on-failure|destroy',
                   name: 'post_behavior_monitor_nodes')

            string(defaultValue: "${pipelineParams.get('tag_ami_with_result', 'false')}",
                   description: 'true|false',
                   name: 'tag_ami_with_result')

            string(defaultValue: "${pipelineParams.get('ip_ssh_connections', 'private')}",
                   description: 'private|public|ipv6',
                   name: 'ip_ssh_connections')

            string(defaultValue: '',
                   description: 'If empty - the default manager version will be taken',
                   name: 'scylla_mgmt_repo')

            string(defaultValue: "${pipelineParams.get('email_recipients', 'qa@scylladb.com')}",
                   description: 'email recipients of email report',
                   name: 'email_recipients')

            string(defaultValue: "${pipelineParams.get('test_config', '')}",
                   description: 'Test configuration file',
                   name: 'test_config')

            string(defaultValue: "${pipelineParams.get('test_name', '')}",
                   description: 'Name of the test to run',
                   name: 'test_name')
        }
        options {
            timestamps()
            disableConcurrentBuilds()
            timeout(pipelineParams.timeout)
            buildDiscarder(logRotator(numToKeepStr: '20'))
        }
        stages {
            stage('Checkout') {
                steps {
                    script {
                        completed_stages = [:]
                    }
                    dir('scylla-cluster-tests') {
                        checkout scm

                        dir("scylla-qa-internal") {
                            git(url: 'git@github.com:scylladb/scylla-qa-internal.git',
                                credentialsId:'b8a774da-0e46-4c91-9f74-09caebaea261',
                                branch: 'master')
                        }
                    }
                }
            }
            stage('Create SCT Runner') {
                steps {
                    catchError(stageResult: 'FAILURE') {
                        script {
                            wrap([$class: 'BuildUser']) {
                                dir('scylla-cluster-tests') {
                                    createSctRunner(params, pipelineParams.timeout.time , builder.region)
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
                                dir('scylla-cluster-tests') {
                                    runSctTest(params, builder.region)
                                }
                            }
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
                                    runCollectLogs(params, builder.region)
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
                                    runCleanupResource(params, builder.region)
                                    completed_stages['clean_resources'] = true
                                }
                            }
                        }
                    }
                }
            }
            stage("Send email with result") {
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
        }
        post {
            always {
                script {
                    def collect_logs = completed_stages['collect_logs']
                    def clean_resources = completed_stages['clean_resources']
                    def send_email = completed_stages['send_email']
                    sh """
                        echo "$collect_logs"
                        echo "$clean_resources"
                        echo "$send_email"
                    """
                    if (!completed_stages['clean_resources']) {
                        catchError {
                            script {
                                wrap([$class: 'BuildUser']) {
                                    dir('scylla-cluster-tests') {
                                    runCleanupResource(params, builder.region)
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
                                    runSendEmail(params, currentBuild)
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
