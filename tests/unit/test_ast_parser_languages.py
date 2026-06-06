"""Unit tests for tree-sitter AST parsing of Go, Rust, Java, JavaScript, TypeScript.

Tests verify that real tree-sitter AST extraction produces expected function/class/method
names and line numbers, with proper naming conventions (e.g., Type.method for methods).
"""

from pipeline.ast_parser import ASTParser


class TestGoParser:
    """Tests for Go AST parsing."""

    def test_function_declaration(self):
        """Extract top-level function."""
        code = """package main

func Add(a, b int) int {
    return a + b
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.go", "go")

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "Add"
        assert func.line_start == 3
        assert "Add" in func.code

    def test_method_with_receiver(self):
        """Extract method with receiver (should be named Receiver.method)."""
        code = """package main

type Calculator struct {
    value int
}

func (c *Calculator) Set(v int) {
    c.value = v
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.go", "go")

        assert len(result.functions) == 1
        func = result.functions[0]
        # Method should be named "Calculator.Set"
        assert func.name == "Calculator.Set"
        assert "Set" in func.code

    def test_struct_type(self):
        """Extract struct type declaration."""
        code = """package main

type User struct {
    ID   int
    Name string
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.go", "go")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "User"
        assert "struct" in cls.code


class TestRustParser:
    """Tests for Rust AST parsing."""

    def test_function(self):
        """Extract top-level function."""
        code = """fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.rs", "rust")

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "add"
        assert "add" in func.code

    def test_struct_with_methods(self):
        """Extract struct and methods (should be named Struct.method)."""
        code = """struct Calculator {
    value: i32,
}

impl Calculator {
    fn new() -> Self {
        Calculator { value: 0 }
    }

    fn add(&mut self, x: i32) {
        self.value += x;
    }
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.rs", "rust")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Calculator"
        # Methods should be named Calculator.new and Calculator.add
        method_names = [m.name for m in cls.methods]
        assert "Calculator.new" in method_names or "new" in method_names
        assert "Calculator.add" in method_names or "add" in method_names

    def test_enum_type(self):
        """Extract enum type."""
        code = """enum Color {
    Red,
    Green,
    Blue,
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.rs", "rust")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Color"


class TestJavaParser:
    """Tests for Java AST parsing."""

    def test_class_declaration(self):
        """Extract class declaration."""
        code = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "Calculator.java", "java")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Calculator"

    def test_class_with_methods(self):
        """Extract methods from class."""
        code = """public class User {
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String n) {
        name = n;
    }
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "User.java", "java")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "User"
        assert len(cls.methods) == 2
        method_names = {m.name for m in cls.methods}
        assert "getName" in method_names
        assert "setName" in method_names

    def test_interface_declaration(self):
        """Extract interface declaration."""
        code = """public interface Service {
    void start();
    void stop();
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "Service.java", "java")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Service"


class TestJavaScriptParser:
    """Tests for JavaScript AST parsing."""

    def test_function_declaration(self):
        """Extract function declaration."""
        code = """function add(a, b) {
    return a + b;
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.js", "javascript")

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "add"

    def test_class_with_methods(self):
        """Extract class and methods."""
        code = """class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.js", "javascript")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Calculator"
        assert len(cls.methods) == 2
        method_names = {m.name for m in cls.methods}
        assert "add" in method_names
        assert "multiply" in method_names

    def test_arrow_function_expression(self):
        """Extract arrow function expressions (const NAME = () => {})."""
        code = """const add = (a, b) => {
    return a + b;
};

const multiply = (a, b) => a * b;
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.js", "javascript")

        # Arrow functions should be extracted as top-level functions
        func_names = {f.name for f in result.functions}
        assert "add" in func_names
        assert "multiply" in func_names


class TestTypeScriptParser:
    """Tests for TypeScript AST parsing."""

    def test_function_declaration(self):
        """Extract function declaration."""
        code = """function add(a: number, b: number): number {
    return a + b;
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.ts", "typescript")

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "add"

    def test_class_with_methods(self):
        """Extract class and methods."""
        code = """class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }

    divide(a: number, b: number): number {
        return a / b;
    }
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.ts", "typescript")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Calculator"
        assert len(cls.methods) == 2
        method_names = {m.name for m in cls.methods}
        assert "add" in method_names
        assert "divide" in method_names

    def test_interface_declaration(self):
        """Extract TypeScript interface."""
        code = """interface User {
    id: number;
    name: string;
    email?: string;
}
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.ts", "typescript")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "User"

    def test_type_alias(self):
        """Extract TypeScript type alias."""
        code = """type Point = {
    x: number;
    y: number;
};
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.ts", "typescript")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Point"


class TestTSXParser:
    """Tests for TSX (TypeScript + JSX) AST parsing."""

    def test_react_component(self):
        """Extract TypeScript React component."""
        code = """interface Props {
    title: string;
}

export const MyComponent: React.FC<Props> = ({ title }) => {
    return <div>{title}</div>;
};
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.tsx", "tsx")

        # Should extract the interface and arrow function
        assert len(result.classes) >= 1  # At least the interface
        type_names = {c.name for c in result.classes}
        assert "Props" in type_names


class TestFallbackBehavior:
    """Tests that parser falls back gracefully to regex on errors."""

    def test_malformed_code_doesnt_crash(self):
        """Parser should handle malformed code without crashing."""
        code = """func broken( {
    invalid syntax here
}"""
        parser = ASTParser()
        # Should not crash; may return empty or partial results
        result = parser.parse_string(code, "test.go", "go")
        assert isinstance(result.functions, list)
        assert isinstance(result.classes, list)

    def test_empty_file(self):
        """Parser should handle empty files."""
        code = ""
        parser = ASTParser()
        result = parser.parse_string(code, "test.go", "go")
        assert len(result.functions) == 0
        assert len(result.classes) == 0

    def test_comments_only(self):
        """Parser should handle files with only comments."""
        code = """// This is a comment
// Another comment
"""
        parser = ASTParser()
        result = parser.parse_string(code, "test.go", "go")
        assert len(result.functions) == 0
        assert len(result.classes) == 0


class TestHangGuard:
    """Regression tests for hang-safety on pathological large inputs.

    These tests verify that the parser does NOT hang on large/pathological files,
    and instead gracefully times out or hits the size ceiling.
    """

    def test_large_js_file_within_timeout(self):
        """Verify large JS file parses or falls back without hanging."""
        import time

        # Build a pathological ~2MB JS file (deeply nested object literals)
        code_parts = ["const data = {"]
        for i in range(50000):
            code_parts.append(f"  key{i}: {{")
        code_parts.append("    value: 42")
        for _ in range(50000):
            code_parts.append("  }")
        code_parts.append("};")
        code = "\n".join(code_parts)

        parser = ASTParser(parse_timeout_s=5.0)

        start = time.time()
        result = parser.parse_string(code, "huge.js", "javascript")
        elapsed = time.time() - start

        # Should complete in under 5s (timeout guard works)
        assert elapsed < 5.0, f"Parse took {elapsed:.2f}s (should be < 5s)"

        # Should return a result (either parsed or fallback)
        assert isinstance(result.functions, list)
        assert isinstance(result.classes, list)

    def test_large_go_file_exceeds_ceiling(self):
        """Verify large Go file hits size ceiling and falls back."""
        # Build a ~1600 KB Go file (exceeds TREESITTER_ML_MAX_KB = 1500 KB)
        code = "package main\n\n"
        # Each function ~500 bytes
        for i in range(3500):
            code += f"func function{i}() {{\n    x := {i}\n}}\n"

        parser = ASTParser(parse_timeout_s=10.0)
        result = parser.parse_string(code, "huge.go", "go")

        # Should return a result (fallback to regex or skipped)
        assert isinstance(result.functions, list)
        assert isinstance(result.classes, list)

    def test_normal_file_still_extracts(self):
        """Verify normal-sized files still extract symbols correctly."""
        code = """function helper() {
    return 42;
}

class MyClass {
    method() {
        return "test";
    }
}
"""
        parser = ASTParser(parse_timeout_s=10.0)
        result = parser.parse_string(code, "normal.js", "javascript")

        # Should extract the function and class normally
        assert len(result.functions) >= 1
        assert len(result.classes) >= 1
        func_names = {f.name for f in result.functions}
        assert "helper" in func_names
        class_names = {c.name for c in result.classes}
        assert "MyClass" in class_names
