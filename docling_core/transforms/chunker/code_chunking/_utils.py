"""Utility functions and classes for code language detection and processing."""

from typing import List, Optional

import tree_sitter_c as ts_c
import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_js
import tree_sitter_python as ts_python
import tree_sitter_typescript as ts_ts
from tree_sitter import Language as Lang
from tree_sitter import Node, Tree

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc.labels import CodeLanguageLabel


def _get_file_extensions(language: CodeLanguageLabel) -> List[str]:
    """Get the file extensions associated with a language."""
    extensions_map = {
        CodeLanguageLabel.PYTHON: [".py"],
        CodeLanguageLabel.TYPESCRIPT: [".ts", ".tsx", ".cts", ".mts", ".d.ts"],
        CodeLanguageLabel.JAVA: [".java"],
        CodeLanguageLabel.JAVASCRIPT: [".js", ".jsx", ".cjs", ".mjs"],
        CodeLanguageLabel.C: [".c"],
    }
    return extensions_map.get(language, [])


def _get_tree_sitter_language(language: CodeLanguageLabel):
    """Get the tree-sitter language object for a language."""
    language_map = {
        CodeLanguageLabel.PYTHON: lambda: Lang(ts_python.language()),
        CodeLanguageLabel.TYPESCRIPT: lambda: Lang(ts_ts.language_typescript()),
        CodeLanguageLabel.JAVA: lambda: Lang(ts_java.language()),
        CodeLanguageLabel.JAVASCRIPT: lambda: Lang(ts_js.language()),
        CodeLanguageLabel.C: lambda: Lang(ts_c.language()),
    }
    factory = language_map.get(language)
    return factory() if factory else None


def _get_import_query(language: CodeLanguageLabel) -> Optional[str]:
    """Get the tree-sitter query string for finding imports in this language."""
    if language == CodeLanguageLabel.PYTHON:
        return """
            (import_statement) @import
            (import_from_statement) @import
            (future_import_statement) @import
            """
    elif language in (CodeLanguageLabel.TYPESCRIPT, CodeLanguageLabel.JAVASCRIPT):
        return """
            (import_statement) @import_full

            (lexical_declaration
            (variable_declarator
                name: (identifier)
                value: (call_expression
                function: (identifier) @require_function
                arguments: (arguments
                    (string (string_fragment))
                )
                (#eq? @require_function "require")
                )
            )
            ) @import_full

            (lexical_declaration
            (variable_declarator
                name: (identifier)
                value: (await_expression
                (call_expression
                    function: (import)
                    arguments: (arguments
                    (string (string_fragment))
                    )
                )
                )
            )
            ) @import_full
            """
    else:
        return None


def _get_function_name(language: CodeLanguageLabel, node: Node) -> Optional[str]:
    """Extract the function name from a function node."""
    if language == CodeLanguageLabel.C:
        declarator = node.child_by_field_name("declarator")
        if declarator:
            inner_declarator = declarator.child_by_field_name("declarator")
            if inner_declarator and inner_declarator.text:
                return inner_declarator.text.decode("utf8")
        return None
    else:
        name_node = node.child_by_field_name("name")
        if name_node and name_node.text:
            return name_node.text.decode("utf8")
        return None


def _is_collectable_function(
    language: CodeLanguageLabel, node: Node, constructor_name: str
) -> bool:
    """Check if a function should be collected for chunking."""
    if language == CodeLanguageLabel.C:
        return True
    else:
        name = _get_function_name(language, node)
        if not name:
            return False
        return name != constructor_name


def _get_default_tokenizer() -> "BaseTokenizer":
    """Get the default tokenizer instance."""
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    return HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _has_child(node: Node, child_name: str) -> bool:
    """Check if a node has a child with the specified name."""
    return bool(node and node.child_by_field_name(child_name))


def _get_children(node: Node, child_types: List[str]) -> List[Node]:
    """Get all children of a node that match the specified types."""
    if not node.children:
        return []

    return [child for child in node.children if child.type in child_types]


def _to_str(node: Node) -> str:
    """Convert a tree-sitter node to a string."""
    if not node or not node.text:
        return ""
    text = node.text.decode()
    indent = node.start_point.column
    return f"{' ' * indent}{text}".rstrip()


def _query_tree(language, tree: Tree, query: str):
    """Query a tree-sitter tree with the given query string."""
    if not language:
        return []
    q = language.query(query)
    return q.captures(tree.root_node)
