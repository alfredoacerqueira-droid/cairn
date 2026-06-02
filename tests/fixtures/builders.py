"""Fixture builders for synthetic but realistic repos.

Each builder writes a hermetic repo to disk (no network, no Ollama) and returns
the repo root Path. Each repo is git-initialized with one commit, since cairn
uses git commit hashes for caching/freshness detection.
"""

import subprocess
from pathlib import Path


def _init_git_repo(repo_root: Path) -> None:
    """Initialize a git repo at repo_root with one commit."""
    subprocess.run(
        ["git", "init"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    # Create a minimal initial commit so git has history
    (repo_root / ".gitkeep").write_text("")
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def make_helm_repo(base: Path) -> Path:
    """Build a Helm chart repo (YAML + shell + JSON mix).

    Simulates a typical Helm chart structure with:
    - Chart.yaml and values.yaml (config)
    - templates/deployment.yaml and templates/service.yaml (k8s YAML)
    - scripts/deploy.sh (helper script)
    - config.json (additional config)

    This will be detected as the 'iac' profile (no embeddings, structural retrieval).

    Args:
        base: Parent directory for the repo.

    Returns:
        Path to the repo root.
    """
    repo_root = base / "helm-repo"
    repo_root.mkdir(exist_ok=True)

    # Chart.yaml
    (repo_root / "Chart.yaml").write_text("""apiVersion: v2
name: my-app
version: 0.1.0
description: Example Helm chart
type: application
""")

    # values.yaml
    (repo_root / "values.yaml").write_text("""replicaCount: 3
image:
  repository: my-app
  tag: latest
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 80
""")

    # templates/deployment.yaml
    templates = repo_root / "templates"
    templates.mkdir(exist_ok=True)
    (templates / "deployment.yaml").write_text("""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    app: my-app
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - containerPort: 8080
""")

    # templates/service.yaml
    (templates / "service.yaml").write_text("""apiVersion: v1
kind: Service
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    app: my-app
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: 8080
      protocol: TCP
  selector:
    app: my-app
""")

    # scripts/deploy.sh
    scripts = repo_root / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "deploy.sh").write_text("""#!/bin/bash
set -e
NAMESPACE=${NAMESPACE:-default}
RELEASE_NAME=${RELEASE_NAME:-my-app}
echo "Deploying $RELEASE_NAME to $NAMESPACE"
helm upgrade --install "$RELEASE_NAME" . --namespace "$NAMESPACE"
""")

    # config.json
    (repo_root / "config.json").write_text('{"app": "my-app", "version": "0.1.0", "replicas": 3}\n')

    _init_git_repo(repo_root)
    return repo_root


def make_terraform_repo(base: Path) -> Path:
    """Build a Terraform module repo.

    Simulates a typical Terraform module with:
    - main.tf (resource definitions)
    - variables.tf (input variables)
    - outputs.tf (outputs)
    - terraform.tfvars (variable values)
    - environments/*.tfvars (per-environment overrides)

    Detected as 'iac' profile (no embeddings, structural retrieval).

    Args:
        base: Parent directory for the repo.

    Returns:
        Path to the repo root.
    """
    repo_root = base / "terraform-repo"
    repo_root.mkdir(exist_ok=True)

    # main.tf
    (repo_root / "main.tf").write_text("""terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true

  tags = {
    Name = var.project_name
  }
}

resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.subnet_cidr
  availability_zone = var.availability_zone

  tags = {
    Name = "${var.project_name}-private"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}
""")

    # variables.tf
    (repo_root / "variables.tf").write_text("""variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_cidr" {
  description = "Subnet CIDR block"
  type        = string
  default     = "10.0.1.0/24"
}

variable "availability_zone" {
  description = "Availability zone"
  type        = string
  default     = "us-east-1a"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "my-project"
}
""")

    # outputs.tf
    (repo_root / "outputs.tf").write_text("""output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "Subnet ID"
  value       = aws_subnet.private.id
}

output "igw_id" {
  description = "Internet Gateway ID"
  value       = aws_internet_gateway.main.id
}
""")

    # terraform.tfvars
    (repo_root / "terraform.tfvars").write_text("""aws_region     = "us-east-1"
vpc_cidr       = "10.0.0.0/16"
subnet_cidr    = "10.0.1.0/24"
availability_zone = "us-east-1a"
project_name   = "my-project"
""")

    # environments/staging.tfvars
    environments = repo_root / "environments"
    environments.mkdir(exist_ok=True)
    (environments / "staging.tfvars").write_text("""vpc_cidr    = "10.1.0.0/16"
subnet_cidr = "10.1.1.0/24"
project_name = "my-project-staging"
""")

    # environments/production.tfvars
    (environments / "production.tfvars").write_text("""vpc_cidr    = "10.2.0.0/16"
subnet_cidr = "10.2.1.0/24"
project_name = "my-project-prod"
""")

    # backend.hcl
    (repo_root / "backend.hcl").write_text("""bucket         = "my-terraform-state"
key            = "prod/terraform.tfstate"
region         = "us-east-1"
dynamodb_table = "terraform-locks"
""")

    _init_git_repo(repo_root)
    return repo_root


def make_k8s_repo(base: Path, with_pathological: bool = False) -> Path:
    """Build a Kubernetes manifests repo.

    Simulates a typical K8s repo with:
    - manifests/ containing multi-doc YAML files
    - kustomize/ with overlays
    - CRDs/

    Optionally (with_pathological=True), adds:
    - A ~60KB deeply-nested Deployment YAML with huge env: list
    - A large CRD YAML generating thousands of documents

    Args:
        base: Parent directory for the repo.
        with_pathological: If True, add large/pathological YAML files.

    Returns:
        Path to the repo root.
    """
    repo_root = base / "k8s-repo"
    repo_root.mkdir(exist_ok=True)

    manifests = repo_root / "manifests"
    manifests.mkdir(exist_ok=True)

    # manifests/namespace.yaml
    (manifests / "namespace.yaml").write_text("""apiVersion: v1
kind: Namespace
metadata:
  name: default
""")

    # manifests/configmap.yaml
    (manifests / "configmap.yaml").write_text("""apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: default
data:
  app.properties: |
    server.port=8080
    server.servlet.context-path=/api
""")

    # manifests/service.yaml
    (manifests / "service.yaml").write_text("""apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: default
  labels:
    app: my-app
spec:
  type: ClusterIP
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
  selector:
    app: my-app
""")

    # manifests/deployment.yaml (simple version)
    (manifests / "deployment.yaml").write_text("""apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: LOG_LEVEL
          value: "INFO"
""")

    # kustomize/ structure
    kustomize = repo_root / "kustomize"
    kustomize.mkdir(exist_ok=True)
    (kustomize / "kustomization.yaml").write_text("""apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: default

resources:
  - ../manifests/namespace.yaml
  - ../manifests/configmap.yaml
  - ../manifests/service.yaml
  - ../manifests/deployment.yaml
""")

    if with_pathological:
        # Generate a large, deeply-nested Deployment with huge env list
        large_env_list = "\n".join(
            f'        - name: VAR_{i:04d}\n          value: "value_{i}"' for i in range(1000)
        )
        large_deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: large-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: large-app
  template:
    metadata:
      labels:
        app: large-app
    spec:
      containers:
      - name: app
        image: large-app:latest
        env:
{large_env_list}
"""
        (manifests / "large-deployment.yaml").write_text(large_deployment)

        # Generate a large CRD-style YAML with many documents
        crd_content = ""
        for i in range(500):
            crd_content += f"""---
apiVersion: example.com/v1
kind: CustomResource
metadata:
  name: resource-{i:04d}
  namespace: default
spec:
  config:
    setting: value_{i}
"""
        (manifests / "large-crd.yaml").write_text(crd_content)

    _init_git_repo(repo_root)
    return repo_root


def make_python_repo(base: Path) -> Path:
    """Build a small Python package for smoke tests.

    A minimal Python package with:
    - __init__.py
    - module.py (with functions/classes)
    - utils.py

    Args:
        base: Parent directory for the repo.

    Returns:
        Path to the repo root.
    """
    repo_root = base / "python-repo"
    repo_root.mkdir(exist_ok=True)

    # __init__.py
    (repo_root / "__init__.py").write_text('"""My app package."""\n__version__ = "0.1.0"\n')

    # module.py
    (repo_root / "module.py").write_text("""\"\"\"Main module with functions and classes.\"\"\"


class MyClass:
    \"\"\"A simple class.\"\"\"

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        \"\"\"Return the name.\"\"\"
        return self.name

    def set_name(self, name: str) -> None:
        \"\"\"Set the name.\"\"\"
        self.name = name


def my_function(x: int, y: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return x + y


def process_data(data: list[str]) -> dict[str, int]:
    \"\"\"Process a list of strings.\"\"\"
    result = {}
    for item in data:
        result[item] = len(item)
    return result
""")

    # utils.py
    (repo_root / "utils.py").write_text("""\"\"\"Utility functions.\"\"\"


def stringify(obj):
    \"\"\"Convert object to string.\"\"\"
    return str(obj)


def validate_input(value):
    \"\"\"Validate input value.\"\"\"
    if value is None:
        raise ValueError("Value cannot be None")
    return value
""")

    _init_git_repo(repo_root)
    return repo_root


def make_workspace(base: Path) -> Path:
    """Build a workspace containing multiple repos (helm, terraform, k8s).

    This simulates a workspace where a user has multiple IaC projects as
    sibling subdirectories (e.g., an EKS migration workspace).

    Args:
        base: Parent directory for the workspace.

    Returns:
        Path to the workspace root.
    """
    workspace_root = base / "workspace"
    workspace_root.mkdir(exist_ok=True)

    make_helm_repo(workspace_root)
    make_terraform_repo(workspace_root)
    make_k8s_repo(workspace_root)

    # Initialize git at workspace level (not required, but realistic)
    _init_git_repo(workspace_root)

    return workspace_root
