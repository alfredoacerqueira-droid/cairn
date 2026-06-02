"""Unit tests for new tree-sitter parsing (HCL/YAML/C#/bash)."""

from pipeline.ast_parser import ASTParser


class TestHCLParsing:
    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_terraform_resource(self):
        code = """
locals {
  kubernetes_network_config = try(aws_eks_cluster.this[0].kubernetes_network_config[0], {})
}

resource "aws_eks_cluster" "main" {
  name = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
}

resource "aws_eks_node_group" "workers" {
  cluster_name = aws_eks_cluster.main.name
  node_group_name = "worker-group"
  node_role_arn = aws_iam_role.nodes.arn
}

variable "cluster_name" {
  type = string
  default = "my-cluster"
}

module "vpc" {
  source = "./modules/vpc"
  cidr = "10.0.0.0/16"
}
"""
        result = self.parser.parse_string(code, "test.tf", "hcl")

        # Should extract blocks
        assert len(result.functions) > 0
        names = [f.name for f in result.functions]

        # Check for expected block types
        assert any(
            "resource" in n and "aws_eks_cluster" in n for n in names
        ), f"Missing resource block, got: {names}"
        assert any(
            "resource" in n and "aws_eks_node_group" in n for n in names
        ), f"Missing node group block, got: {names}"
        assert any(
            "variable" in n and "cluster_name" in n for n in names
        ), f"Missing variable block, got: {names}"
        assert any(
            "module" in n and "vpc" in n for n in names
        ), f"Missing module block, got: {names}"

    def test_parse_hcl_block_naming(self):
        code = """resource "aws_s3_bucket" "data_bucket" {
  bucket = "my-data-bucket"
}

variable "environment" {
  type = string
}
"""
        result = self.parser.parse_string(code, "test.tf", "hcl")

        assert len(result.functions) >= 2
        # Look for dotted names
        assert any("resource.aws_s3_bucket" in f.name for f in result.functions)


class TestYAMLParsing:
    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_kubernetes_manifest(self):
        code = """apiVersion: v1
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
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
"""
        result = self.parser.parse_string(code, "app.yaml", "yaml")

        # Should extract documents
        assert len(result.functions) >= 2
        names = [f.name for f in result.functions]

        # Check that we have documents with kind and name
        assert any(
            "Deployment" in str(n) and "my-app" in str(n) for n in names
        ), f"Missing Deployment doc, got: {names}"
        assert any(
            "Service" in str(n) and "my-app-service" in str(n) for n in names
        ), f"Missing Service doc, got: {names}"

    def test_parse_yaml_single_document(self):
        code = """kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    key: value
"""
        result = self.parser.parse_string(code, "config.yaml", "yaml")

        assert len(result.functions) >= 1
        # Should have extracted a document block
        assert any(
            "ConfigMap" in str(f.name) and "app-config" in str(f.name) for f in result.functions
        )


class TestCSharpParsing:
    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_interface_and_class(self):
        code = """namespace MediatR;

public interface IMediator : ISender, IPublisher
{
    Task<TResponse> Send<TResponse>(IRequest<TResponse> request);
    Task Send<TRequest>(TRequest request) where TRequest : IRequest;
}

public class Mediator : IMediator
{
    public Mediator(IServiceProvider serviceProvider)
    {
    }

    public Task<TResponse> Send<TResponse>(IRequest<TResponse> request)
    {
        return SendImpl(request);
    }

    public Task Send<TRequest>(TRequest request) where TRequest : IRequest
    {
        return SendImpl(request);
    }

    private Task SendImpl<T>(T request)
    {
        return Task.CompletedTask;
    }
}
"""
        result = self.parser.parse_string(code, "Mediator.cs", "csharp")

        # Should extract interface and class
        assert len(result.classes) >= 2, f"Expected >= 2 classes, got {len(result.classes)}"
        class_names = [c.name for c in result.classes]

        # Check for interface IMediator
        assert "IMediator" in class_names, f"Missing IMediator interface, got: {class_names}"
        # Check for class Mediator
        assert "Mediator" in class_names, f"Missing Mediator class, got: {class_names}"

        # Check that Mediator has methods
        mediator_class = next(c for c in result.classes if c.name == "Mediator")
        assert (
            len(mediator_class.methods) >= 2
        ), f"Expected Mediator to have methods, got {len(mediator_class.methods)}"
        method_names = [m.name for m in mediator_class.methods]
        assert "Send" in method_names or any(
            "Send" in m for m in method_names
        ), f"Missing Send method, got: {method_names}"

    def test_parse_class_with_constructor_and_methods(self):
        code = """public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }

    public int Subtract(int a, int b)
    {
        return a - b;
    }

    public int Multiply(int a, int b)
    {
        return a * b;
    }
}
"""
        result = self.parser.parse_string(code, "calc.cs", "csharp")

        assert len(result.classes) >= 1
        assert result.classes[0].name == "Calculator"
        assert len(result.classes[0].methods) >= 3
        method_names = [m.name for m in result.classes[0].methods]
        assert "Add" in method_names
        assert "Subtract" in method_names
        assert "Multiply" in method_names

    def test_bom_prefixed_file_names_not_mangled(self):
        """A UTF-8 BOM (common in Windows/C# files) must not shift name offsets.

        Regression: tree-sitter byte offsets were used to slice the str, so a
        3-byte/1-char BOM dropped the first 2 chars of every identifier
        (ISender -> 'ender', ServiceRegistrar -> 'rviceRegistrar').
        """
        code = "\ufeff" + (
            "public interface ISender\n"
            "{\n"
            "    Task<TResponse> Send<TResponse>(IRequest<TResponse> request);\n"
            "}\n"
            "\n"
            "public static class ServiceRegistrar\n"
            "{\n"
            "    public static void RegisterServices(IServiceCollection services)\n"
            "    {\n"
            "    }\n"
            "}\n"
        )
        result = self.parser.parse_string(code, "ISender.cs", "csharp")
        class_names = [c.name for c in result.classes]
        assert "ISender" in class_names, f"BOM mangled interface name: {class_names}"
        assert "ServiceRegistrar" in class_names, f"BOM mangled class name: {class_names}"
        # No name should carry a leading BOM either.
        for name in class_names:
            assert not name.startswith("\ufeff"), f"name retains BOM: {name!r}"
        registrar = next(c for c in result.classes if c.name == "ServiceRegistrar")
        assert "RegisterServices" in [m.name for m in registrar.methods]


class TestBashParsing:
    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_bash_functions(self):
        code = """#!/bin/bash

deploy() {
    echo "Deploying application"
    docker build -t myapp:latest .
    docker push myapp:latest
}

rollback() {
    echo "Rolling back"
    helm rollback my-release
}

validate_config() {
    echo "Validating configuration"
    if [ -f config.yaml ]; then
        echo "Config found"
    fi
}
"""
        result = self.parser.parse_string(code, "deploy.sh", "bash")

        # Should extract functions
        assert len(result.functions) >= 3, f"Expected >= 3 functions, got {len(result.functions)}"
        func_names = [f.name for f in result.functions]

        assert "deploy" in func_names, f"Missing deploy function, got: {func_names}"
        assert "rollback" in func_names, f"Missing rollback function, got: {func_names}"
        assert (
            "validate_config" in func_names
        ), f"Missing validate_config function, got: {func_names}"

    def test_parse_bash_function_variations(self):
        code = """function hello() {
    echo "Hello"
}

world() {
    echo "World"
}

say_goodbye() {
    echo "Goodbye"
}
"""
        result = self.parser.parse_string(code, "script.sh", "bash")

        assert len(result.functions) >= 3
        func_names = [f.name for f in result.functions]
        assert "hello" in func_names
        assert "world" in func_names
        assert "say_goodbye" in func_names


class TestFallbackBehavior:
    def setup_method(self):
        self.parser = ASTParser()

    def test_broken_hcl_falls_back_to_regex(self):
        # Malformed HCL that will cause tree-sitter to fail gracefully
        code = "totally invalid hcl syntax {{{"
        result = self.parser.parse_string(code, "broken.tf", "hcl")

        # Should not crash; should return empty or regex parse result
        assert isinstance(result.functions, list)

    def test_broken_csharp_falls_back_to_regex(self):
        code = "completely broken csharp {{{ oops"
        result = self.parser.parse_string(code, "broken.cs", "csharp")

        assert isinstance(result.classes, list)

    def test_empty_file_parsing(self):
        for lang in ("hcl", "yaml", "csharp", "bash"):
            result = self.parser.parse_string("", f"empty.{lang}", lang)
            assert len(result.functions) == 0
            assert len(result.classes) == 0
