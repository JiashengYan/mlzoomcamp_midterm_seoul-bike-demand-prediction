"""Check function-comment structure; readable meaning still needs human review."""
import ast
import re
from pathlib import Path


def test_function_documentation():
    """Require purpose, inputs, and outputs for every project Python function.

    Inputs:
        No arguments. Reads Python source files below the project root, excluding
            virtual environments, version-control metadata, and generated caches.

    Outputs:
        None. Assertion failure names each function with missing or malformed
        documentation, an overlong docstring line, or a parameter without its own Inputs
        line. This does not judge whether the explanation is understandable.
    """
    root = Path(__file__).resolve().parents[1]
    ignored = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    failures = []
    for path in sorted(root.rglob("*.py")):
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            label = f"{path.relative_to(root)}:{node.lineno} {node.name}"
            doc = ast.get_docstring(node) or ""
            paragraphs = doc.split("\n\n")
            if (len(paragraphs) != 3 or not paragraphs[0].strip()
                    or "\n" in paragraphs[0]
                    or not paragraphs[1].startswith("Inputs:\n")
                    or not paragraphs[2].startswith("Outputs:\n")):
                failures.append(label + ": expected purpose, Inputs, Outputs")
                continue
            doc_node = node.body[0]
            doc_lines = source.splitlines()[doc_node.lineno - 1:doc_node.end_lineno]
            if any(len(line) > 88 for line in doc_lines):
                failures.append(label + ": docstring line exceeds 88 characters")
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
            for arg in args:
                if arg.arg in {"self", "cls"}:
                    continue
                parameter_line = rf"(?m)^    {re.escape(arg.arg)}:"
                if not re.search(parameter_line, paragraphs[1]):
                    failures.append(label + f": give input {arg.arg} its own line")
    assert not failures, "\n".join(failures)
