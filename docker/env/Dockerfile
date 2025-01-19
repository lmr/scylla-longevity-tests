ARG PYTHON_IMAGE_TAG=3.10.12-slim-bullseye

FROM python:$PYTHON_IMAGE_TAG AS apt_base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# Add 3rd-party APT repositories.
FROM apt_base AS apt_repos
RUN apt-get install -y --no-install-recommends gnupg2 apt-transport-https software-properties-common curl
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
RUN echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Download, build and install Python packages.
FROM apt_base AS python_packages
ENV PIP_NO_CACHE_DIR=1
RUN apt-get install -y --no-install-recommends build-essential cmake libssl-dev zlib1g-dev libffi-dev
ADD requirements.txt .
RUN pip install uv==0.2.24
RUN uv pip install --system -r requirements.txt

FROM python:$PYTHON_IMAGE_TAG
ARG KUBECTL_VERSION=1.27.3
ARG EKSCTL_VERSION=0.165.0
ARG HELM_VERSION=3.12.2
ENV PYTHONWARNINGS="ignore:unclosed ignore::SyntaxWarning" \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=apt_repos /etc/apt/trusted.gpg /etc/apt/sources.list /etc/apt/
COPY --from=apt_repos /usr/share/keyrings/cloud.google.gpg /usr/share/keyrings/cloud.google.gpg
COPY --from=apt_repos /etc/apt/sources.list.d/google-cloud-sdk.list /etc/apt/sources.list.d/google-cloud-sdk.list
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        google-cloud-sdk \
        google-cloud-sdk-gke-gcloud-auth-plugin \
        binutils \
        curl \
        gettext \
        git \
        iproute2 \
        iptables \
        libnss-myhostname \
        openssh-client \
        rsync \
        sudo \
        unzip \
        zstd \
        wget \
        psmisc \
        procps \
        docker-ce-cli \
        docker-compose-plugin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /usr/local/bin/kubectl https://dl.k8s.io/release/v$KUBECTL_VERSION/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl
RUN curl --silent --location "https://github.com/eksctl-io/eksctl/releases/download/v$EKSCTL_VERSION/eksctl_Linux_amd64.tar.gz" | tar xz -C /tmp && \
    mv /tmp/eksctl /usr/local/bin
RUN curl --silent --location  "https://get.helm.sh/helm-v$HELM_VERSION-linux-amd64.tar.gz" | tar xz -C /tmp && mv /tmp/linux-amd64/helm /usr/local/bin
COPY --from=python_packages /usr/local /usr/local
