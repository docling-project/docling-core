from enum import Enum
from typing import List, Optional

import tree_sitter_c as ts_c
import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_js
import tree_sitter_python as ts_python
import tree_sitter_typescript as ts_ts
from tree_sitter import Language as Lang
from tree_sitter import Node, Tree

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    C = "c"

    def file_extensions(self) -> List[str]:
        if self == Language.PYTHON:
            return [".py"]
        elif self == Language.TYPESCRIPT:
            return [".ts", ".tsx", ".cts", ".mts", ".d.ts"]
        elif self == Language.JAVA:
            return [".java"]
        elif self == Language.JAVASCRIPT:
            return [".js", ".jsx", ".cjs", ".mjs"]
        elif self == Language.C:
            return [".c"]
        else:
            return []

    def get_tree_sitter_language(self):
        if self == Language.PYTHON:
            return Lang(ts_python.language())
        elif self == Language.TYPESCRIPT:
            return Lang(ts_ts.language_typescript())
        elif self == Language.JAVA:
            return Lang(ts_java.language())
        elif self == Language.JAVASCRIPT:
            return Lang(ts_js.language())
        elif self == Language.C:
            return Lang(ts_c.language())
        else:
            return None

    def get_import_query(self) -> Optional[str]:
        if self == Language.PYTHON:
            return """
                (import_statement) @import
                (import_from_statement) @import
                (future_import_statement) @import
                """
        elif self in (Language.TYPESCRIPT, Language.JAVASCRIPT):
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

    def get_function_name(self, node: Node) -> Optional[str]:
        if self == Language.C:
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

    def is_collectable_function(self, node: Node, constructor_name: str) -> bool:
        if self == Language.C:
            return True
        else:
            name = self.get_function_name(node)
            if not name:
                return False

            return name != constructor_name


def _get_default_tokenizer() -> "BaseTokenizer":
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    return HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def has_child(node: Node, child_name: str) -> bool:
    return bool(node and node.child_by_field_name(child_name))


def get_children(node: Node, child_types: List[str]) -> List[Node]:
    if not node.children:
        return []

    return [child for child in node.children if child.type in child_types]


def to_str(node: Node) -> str:
    if not node or not node.text:
        return ""
    text = node.text.decode()
    indent = node.start_point.column
    return f"{' ' * indent}{text}".rstrip()


def query_tree(language, tree: Tree, query: str):
    """Query a tree-sitter tree with the given query string."""
    if not language:
        return []
    q = language.query(query)
    return q.captures(tree.root_node)
