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
