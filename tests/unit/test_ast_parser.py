"""Unit tests for AST parser."""

from pipeline.ast_parser import ASTParser


class TestASTParser:
    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_simple_function(self):
        code = "def hello():\n    return 'world'"
        result = self.parser.parse_string(code)

        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"
        assert result.functions[0].line_start == 1
        assert result.functions[0].line_end == 2

    def test_parse_class_with_methods(self):
        code = """class AuthMiddleware:
    def authenticate(self, request):
        return True

    def authorize(self, user):
        return user.is_admin"""
        result = self.parser.parse_string(code)

        assert len(result.classes) == 1
        assert result.classes[0].name == "AuthMiddleware"
        assert result.classes[0].line_start == 1

        methods = result.classes[0].methods
        assert len(methods) == 2
        assert methods[0].name == "authenticate"
        assert methods[1].name == "authorize"

    def test_parse_multiple_functions(self):
        code = """def login():
    pass

def logout():
    pass

def validate():
    pass"""
        result = self.parser.parse_string(code)

        assert len(result.functions) == 3
        assert result.functions[0].name == "login"
        assert result.functions[1].name == "logout"
        assert result.functions[2].name == "validate"

    def test_parse_empty_file(self):
        result = self.parser.parse_string("")
        assert len(result.functions) == 0
        assert len(result.classes) == 0

    def test_to_dict(self):
        code = "def hello():\n    pass"
        result = self.parser.parse_string(code)

        d = result.to_dict()
        assert d["filepath"] == "<string>"
        assert len(d["functions"]) == 1
        assert d["functions"][0]["name"] == "hello"
        assert d["functions"][0]["line_start"] == 1

    def test_line_numbers_accurate(self):
        code = """# Top comment

def first():
    pass

def second():
    pass"""
        result = self.parser.parse_string(code)

        assert result.functions[0].name == "first"
        assert result.functions[0].line_start == 3
        assert result.functions[1].name == "second"
        assert result.functions[1].line_start == 6

    def test_code_extraction(self):
        code = "def add(a, b):\n    return a + b"
        result = self.parser.parse_string(code)

        assert result.functions[0].code == "def add(a, b):\n    return a + b"

    def test_class_with_decorated_methods(self):
        code = """class MyClass:
    @staticmethod
    def static_method():
        pass

    def regular_method(self):
        pass"""
        result = self.parser.parse_string(code)

        assert len(result.classes) == 1
        methods = result.classes[0].methods
        assert len(methods) == 2
        assert methods[0].name == "static_method"

    def test_nested_functions(self):
        code = """def outer():
    def inner():
        pass
    return inner"""
        result = self.parser.parse_string(code)

        assert len(result.functions) == 2


class TestASTParserPerformance:
    def setup_method(self):
        self.parser = ASTParser()

    def test_small_file_performance(self):
        import time

        # Generate ~100 lines of code
        code = "\n".join([f"def func{i}():\n    pass" for i in range(50)])

        start = time.perf_counter()
        result = self.parser.parse_string(code)
        elapsed = time.perf_counter() - start

        assert len(result.functions) == 50
        assert elapsed * 1000 < 500  # Should be well under 500ms


class TestYAMLPerKeyExtraction:
    """Tests for YAML per-top-level-key extraction (CHANGE C)."""

    def setup_method(self):
        self.parser = ASTParser()

    def test_helm_values_yaml_multiple_top_level_keys(self):
        """Test: helm-style values.yaml yields one block per top-level key."""
        code = """replicaCount: 3
image:
  repository: myapp
  tag: "1.0"
resources:
  limits:
    memory: "256Mi"
  requests:
    memory: "128Mi"
kubeseal:
  enabled: true
  version: "0.24"
"""
        result = self.parser.parse_string(code, filepath="values.yaml", lang="yaml")

        # Should have 4 top-level keys: replicaCount, image, resources, kubeseal
        assert len(result.functions) == 4, f"Expected 4 blocks, got {len(result.functions)}"

        names = {f.name for f in result.functions}
        assert "replicaCount" in names
        assert "image" in names
        assert "resources" in names
        assert "kubeseal" in names

        # Verify line ranges are distinct and correct
        for func in result.functions:
            assert func.line_start > 0
            assert func.line_end >= func.line_start

    def test_multi_doc_k8s_manifest_per_resource_preserved(self):
        """Test: multi-doc k8s manifest (with kind:) yields one block per resource."""
        code = """apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
data:
  key: value
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
"""
        result = self.parser.parse_string(code, filepath="manifest.yaml", lang="yaml")

        # Should have 2 blocks: one per resource (both have 'kind')
        assert len(result.functions) == 2, f"Expected 2 blocks, got {len(result.functions)}"

        names = {f.name for f in result.functions}
        # Names should follow kind.name format (or just kind if no name)
        assert any("ConfigMap" in n or "config" in n.lower() for n in names)
        assert any("Deployment" in n or "app" in n.lower() for n in names)

    def test_malformed_yaml_falls_back_to_document_level(self):
        """Test: malformed YAML falls back to single document-level block (no crash)."""
        code = """key1: value1
  invalid indentation: here
key2: value2
  another bad line:
"""
        # Should not crash, even though YAML is malformed
        result = self.parser.parse_string(code, filepath="bad.yaml", lang="yaml")

        # Should fall back to document-level block
        assert len(result.functions) > 0
        # At least one block should exist (fallback)
        assert all(len(f.code) > 0 for f in result.functions)

    def test_empty_yaml_document(self):
        """Test: empty YAML document yields no blocks (or one fallback)."""
        code = ""
        result = self.parser.parse_string(code, filepath="empty.yaml", lang="yaml")

        # Empty document: should yield 0 blocks
        assert len(result.functions) == 0

    def test_yaml_list_document_fallback(self):
        """Test: YAML document with top-level list (not dict) falls back."""
        code = """- name: item1
  value: 1
- name: item2
  value: 2
"""
        result = self.parser.parse_string(code, filepath="list.yaml", lang="yaml")

        # Top-level list (not dict): should fall back to document-level block
        assert len(result.functions) >= 1
        # Should have the full document as a fallback block
        if result.functions:
            assert "item1" in result.functions[0].code

    def test_helm_values_yaml_line_ranges_correct(self):
        """Test: helm values.yaml blocks have correct line_start/line_end."""
        code = """replicaCount: 3

image:
  tag: "1.0"

resources:
  limits: 256Mi
"""
        result = self.parser.parse_string(code, filepath="values.yaml", lang="yaml")

        # Should have 3 top-level keys
        blocks_by_name = {f.name: f for f in result.functions}
        assert len(blocks_by_name) == 3

        # Check line_start for each
        # replicaCount should start at line 1
        assert blocks_by_name["replicaCount"].line_start == 1
        # image should start at line 3
        assert blocks_by_name["image"].line_start == 3
        # resources should start at line 6
        assert blocks_by_name["resources"].line_start == 6

    def test_yaml_complex_nested_structure(self):
        """Test: deeply nested YAML dict yields one block per top-level key."""
        code = """app:
  name: myapp
  version: 1.0
  services:
    api:
      port: 8080
    db:
      port: 5432

config:
  timeout: 30
  retries: 3
"""
        result = self.parser.parse_string(code, filepath="config.yaml", lang="yaml")

        # Should have 2 top-level keys: app and config
        assert len(result.functions) == 2
        names = {f.name for f in result.functions}
        assert "app" in names
        assert "config" in names

        # Each block should contain all nested content under that key
        app_block = next(f for f in result.functions if f.name == "app")
        assert "myapp" in app_block.code
        assert "api:" in app_block.code

    def test_yaml_with_kind_overrides_per_key_logic(self):
        """Test: if YAML has 'kind:', it uses per-resource, not per-key."""
        code = """apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  ports:
    - port: 80
"""
        result = self.parser.parse_string(code, filepath="service.yaml", lang="yaml")

        # Should have 1 block (per-resource logic for 'kind')
        assert len(result.functions) == 1
        assert "Service" in result.functions[0].name or "service" in result.functions[0].name
