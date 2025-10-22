from typing import Any, Dict, List, Tuple

from pydantic import Field
from tree_sitter import Node, Tree
from typing_extensions import override

from docling_core.transforms.chunker.base_code_chunker import _CodeChunker
from docling_core.transforms.chunker.code_chunk_utils.utils import (
    Language,
    _get_default_tokenizer,
    get_children,
    has_child,
    query_tree,
    to_str,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer


class _PythonFunctionChunker(_CodeChunker):

    language: Language = Language.PYTHON
    ts_language: Any = Field(default=None)
    parser: Any = Field(default=None)
    function_definition_types: List[str] = ["function_definition"]
    class_definition_types: List[str] = ["class_definition"]
    constructor_name: str = "__init__"
    decorator_type: str = "decorated_definition"
    expression_types: List[str] = ["expression_statement"]
    chunk_prefix: str = "\n\t"
    chunk_suffix: str = ""
    function_body: str = "block"
    tokenizer: BaseTokenizer = Field(default_factory=_get_default_tokenizer)
    min_chunk_size: int = 300
    max_tokens: int = 50
    docs_types: List[str] = ["body", "comment"]
    dotted_name: str = "dotted_name"
    aliased_import: str = "aliased_import"

    def __init__(self, **data):
        super().__init__(**data)

    @override
    def _get_docstring(self, node: Node) -> str:
        body_node = node.child_by_field_name(self.function_body)
        if not body_node or not body_node.named_children:
            return ""

        docstring_node = next(
            (child for child in body_node.named_children if self._is_docstring(child)),
            None,
        )

        if docstring_node and docstring_node.named_children:
            text = docstring_node.named_children[0].text
            return text.decode(self.utf8_encoding) if text else ""
        return ""

    @override
    def _get_imports(self, tree: Tree) -> Dict[str, Node]:
        """Get imports for Python."""
        import_query = self.language.get_import_query()
        if not import_query:
            return {}
        import_query_results = query_tree(self.ts_language, tree, import_query)
        imports = {}

        if import_query_results:
            nodes = [node for node in import_query_results["import"]]
            nodes.sort(key=lambda node: node.start_point)
            for node in nodes:
                import_names = []
                aliases = node.named_children
                for child in aliases:
                    if child.type == self.dotted_name:
                        import_names.append(child.text.decode(self.utf8_encoding))
                    elif child.type == self.aliased_import:
                        original = child.child(0).text.decode(self.utf8_encoding)
                        alias = child.child(2).text.decode(self.utf8_encoding)
                        import_names.append(alias)
                        import_names.append(original)
                for name in import_names:
                    imports[name] = node
        return imports

    def _get_module_variables(self, tree: Tree) -> Dict[str, Node]:
        """Get module-level variable assignments for Python."""
        variables = {}
        for child in tree.root_node.children:
            if child.type in self.expression_types and child.named_children:
                expr = child.named_children[0]
                if expr.type == "assignment":
                    if (
                        expr.named_children
                        and expr.named_children[0].type in self.identifiers
                    ):
                        text = expr.named_children[0].text
                        var_name = text.decode(self.utf8_encoding) if text else ""
                        extended_node = self._get_variable_with_comments(
                            child, tree.root_node
                        )
                        variables[var_name] = extended_node
        return variables

    @override
    def _get_variable_with_comments(self, var_node: Node, root_node: Node) -> Node:
        """Get variable node including any preceding comments."""
        return var_node

    @override
    def _find_used_variables(self, function_node: Node) -> set:
        """Find variable names used within a function."""
        used_vars = set()

        def collect_identifiers(node, depth=0):
            "  " * depth
            if node.type in self.identifiers:
                var_name = node.text.decode(self.utf8_encoding)
                is_local = self._is_local_assignment(node)
                if not is_local:
                    used_vars.add(var_name)
            for child in node.children:
                collect_identifiers(child, depth + 1)

        body_node = function_node.child_by_field_name("block")
        if not body_node:
            body_node = function_node.child_by_field_name("body")
        if not body_node:
            for child in function_node.children:
                if child.type in ["block", "suite", "compound_statement"]:
                    body_node = child
                    break

        if body_node:
            collect_identifiers(body_node)
        else:
            collect_identifiers(function_node)

        return used_vars

    def _is_local_assignment(self, identifier_node: Node) -> bool:
        """Check if an identifier is part of a local assignment (not a reference)."""
        current = identifier_node.parent
        while current:
            if current.type == "assignment":
                if (
                    current.named_children
                    and current.named_children[0] == identifier_node
                ):
                    return True
            current = current.parent
        return False


class _TypeScriptFunctionChunker(_CodeChunker):
    language: Language = Language.TYPESCRIPT
    ts_language: Any = Field(default=None)
    parser: Any = Field(default=None)
    function_definition_types: List[str] = [
        "function_declaration",
        "arrow_function",
        "method_definition",
        "function_expression",
        "generator_function",
        "generator_function_declaration",
        "export_statement",
    ]
    class_definition_types: List[str] = ["class_declaration"]
    constructor_name: str = "constructor"
    decorator_type: str = "decorator"
    function_body: str = "block"
    expression_types: List[str] = ["expression_statement"]
    tokenizer: BaseTokenizer = Field(default_factory=_get_default_tokenizer)
    min_chunk_size: int = 300
    max_tokens: int = 5000
    chunk_prefix: str = " {"
    chunk_suffix: str = "\n}"
    docs_types: List[str] = ["comment"]
    import_clause: str = "import_clause"
    named_imports: str = "named_imports"
    import_specifier: str = "import_specifier"
    namespace_import: str = "namespace_import"
    variable_declarator: str = "variable_declarator"

    def __init__(self, **data):
        super().__init__(**data)

    @override
    def _is_docstring(self, node: Node) -> bool:
        return node.type in self.docs_types

    @override
    def _get_imports(self, tree: Tree) -> Dict[str, Node]:
        import_query = self.language.get_import_query()
        if not import_query:
            return {}
        import_query_results = query_tree(self.ts_language, tree, import_query)
        imports = {}
        for import_node in import_query_results.get("import_full", []):
            identifiers = []
            for child in import_node.children:
                if child.type == self.import_clause:
                    default_name = child.child_by_field_name(self.name_field)
                    if default_name:
                        identifiers.append(default_name.text.decode("utf8"))
                    for sub_child in child.children:
                        if sub_child.type == self.named_imports:
                            for spec in sub_child.children:
                                if spec.type == self.import_specifier:
                                    name_node = spec.child_by_field_name(
                                        self.name_field
                                    )
                                    if name_node:
                                        identifiers.append(
                                            name_node.text.decode("utf8")
                                        )
                        elif sub_child.type in self.identifiers:
                            identifiers.append(sub_child.text.decode("utf8"))
                        elif sub_child.type == self.namespace_import:
                            for ns_child in sub_child.children:
                                if ns_child.type in self.identifiers:
                                    identifiers.append(ns_child.text.decode("utf8"))
                elif child.type == self.variable_declarator:
                    identifier = child.child_by_field_name(self.name_field)
                    if identifier:
                        identifiers.append(identifier.text.decode("utf8"))
            for identifier_val in identifiers:
                imports[identifier_val] = import_node
        return imports

    def _get_module_variables(self, tree: Tree) -> Dict[str, Node]:
        """TypeScript/JavaScript don't have module-level variables like Python or C macros."""
        return {}


class _JavaScriptFunctionChunker(_TypeScriptFunctionChunker):
    def __init__(self, **data):
        super().__init__(language=Language.JAVASCRIPT)


class _CFunctionChunker(_CodeChunker):
    language: Language = Language.C
    ts_language: Any = Field(default=None)
    parser: Any = Field(default=None)
    function_definition_types: List[str] = ["function_definition"]
    class_definition_types: List[str] = [""]
    constructor_name: str = ""
    decorator_type: str = ""
    function_body: str = "compound_statement"
    tokenizer: BaseTokenizer = Field(default_factory=_get_default_tokenizer)
    min_chunk_size: int = 300
    max_tokens: int = 5000
    chunk_prefix: str = " {"
    chunk_suffix: str = "\n}"
    expression_types: List[str] = []
    docs_types: List[str] = ["comment", "block_comment"]
    structs: List[str] = ["struct_specifier", "preproc_def", "preproc_function_def"]
    declaration: str = "declaration"
    declarator: str = "declarator"
    function_declaration: List[str] = ["type_definition", "function_declaration"]
    type_field: str = "type"
    identifiers: List[str] = ["identifier"]

    def __init__(self, **data):
        super().__init__(**data)

    @override
    def _is_docstring(self, node: Node) -> bool:
        return node.type in self.docs_types

    @override
    def _get_docstring(self, node: Node) -> str:
        docstring = ""
        if node.prev_named_sibling and node.prev_named_sibling.type in self.docs_types:
            while (
                node.prev_named_sibling
                and node.prev_named_sibling.type in self.docs_types
            ):
                text = node.prev_named_sibling.text
                if text:
                    docstring += text.decode(self.utf8_encoding)
                node = node.prev_named_sibling
            return docstring
        return ""

    @override
    def _is_constructor(self, node: Node) -> bool:
        return False

    def _get_imports(self, tree: Tree) -> Dict[str, Node]:
        structs = {}

        def _clean_name(name_text: str) -> str:
            for char in ["[", "("]:
                if char in name_text:
                    name_text = name_text.split(char)[0]
            return name_text.strip()

        def _structs(node):
            if node.type in self.structs and node.child_by_field_name(self.name_field):
                name = node.child_by_field_name(self.name_field)
                clean_name = _clean_name(name.text.decode("utf8"))
                if clean_name:
                    structs[clean_name] = node
            elif node.type in [self.declaration]:
                if has_child(
                    node.child_by_field_name(self.declarator), self.declarator
                ):
                    name = node.child_by_field_name(
                        self.declarator
                    ).child_by_field_name(self.declarator)
                else:
                    name = node.child_by_field_name(self.declarator)
                if name:
                    clean_name = _clean_name(name.text.decode("utf8"))
                    if clean_name:
                        structs[clean_name] = node
            elif node.type in self.function_declaration:
                if has_child(
                    node.child_by_field_name(self.type_field), self.name_field
                ):
                    name = node.child_by_field_name(
                        self.type_field
                    ).child_by_field_name(self.name_field)
                else:
                    name = node.child_by_field_name(self.type_field)
                if name:
                    clean_name = _clean_name(name.text.decode("utf8"))
                    if clean_name:
                        structs[clean_name] = node
            if node.type not in ["compound_statement", "block"]:
                for child in node.children:
                    _structs(child)

        for child in tree.root_node.children:
            _structs(child)

        return {**structs}

    def _get_module_variables(self, tree: Tree) -> Dict[str, Node]:
        """Get module-level #define macros for C."""
        macros = {}
        for child in tree.root_node.children:
            if child.type == "preproc_def":
                macro_name = self._extract_macro_name(child)
                if macro_name:
                    extended_node = self._get_macro_with_comments(child, tree.root_node)
                    macros[macro_name] = extended_node
        return macros

    def _extract_macro_name(self, define_node: Node) -> str:
        """Extract the macro name from a #define node."""
        for child in define_node.children:
            if child.type in self.identifiers:
                text = child.text
                return text.decode(self.utf8_encoding) if text else ""
        return ""

    def _get_macro_with_comments(self, macro_node: Node, root_node: Node) -> Node:
        """Get macro node including any preceding comments."""
        return macro_node

    @override
    def _find_used_variables(self, function_node: Node) -> set:
        """Find macro names used within a function."""
        used_macros = set()

        def collect_identifiers(node, depth=0):
            "  " * depth
            if node.type in self.identifiers:
                macro_name = node.text.decode(self.utf8_encoding)
                used_macros.add(macro_name)
            for child in node.children:
                collect_identifiers(child, depth + 1)

        body_node = function_node.child_by_field_name(self.function_body)
        if not body_node:
            body_node = function_node.child_by_field_name("body")
        if not body_node:
            for child in function_node.children:
                if child.type in ["compound_statement", "block"]:
                    body_node = child
                    break

        if body_node:
            collect_identifiers(body_node)
        else:
            collect_identifiers(function_node)

        return used_macros


class _JavaFunctionChunker(_CodeChunker):

    language: Language = Language.JAVA
    ts_language: Any = Field(default=None)
    parser: Any = Field(default=None)
    method_declaration: str = "method_declaration"
    function_definition_types: List[str] = [
        method_declaration,
        "constructor_declaration",
        "static_initializer",
    ]
    class_definition_types: List[str] = ["class_declaration", "interface_declaration"]
    constructor_name: str = "<init>"
    decorator_type: str = "annotation"
    function_body: str = "block"
    expression_types: List[str] = []
    tokenizer: BaseTokenizer = Field(default_factory=_get_default_tokenizer)
    min_chunk_size: int = 300
    max_tokens: int = 5000
    chunk_prefix: str = " {"
    chunk_suffix: str = "\n}"
    docs_types: List[str] = ["block_comment", "comment"]
    package_declaration: str = "package_declaration"
    import_declaration: str = "import_declaration"
    class_declaration: str = "class_declaration"
    record_declaration: str = "record_declaration"
    enum_declaration: str = "enum_declaration"
    interface_declaration: str = "interface_declaration"
    field_declaration: str = "field_declaration"
    static_initializer: str = "static_initializer"
    constructor_declaration: str = "constructor_declaration"
    compact_constructor_declaration: str = "compact_constructor_declaration"
    enum_constant: str = "enum_constant"
    enum_body_declarations: str = "enum_body_declarations"
    constant_declaration: str = "constant_declaration"

    enum_inner_types: List[str] = [
        field_declaration,
        method_declaration,
        function_body,
        constructor_declaration,
        compact_constructor_declaration,
    ]
    class_header_inner_types: List[str] = [
        field_declaration,
        static_initializer,
        function_body,
    ]
    object_declarations: List[str] = [
        class_declaration,
        record_declaration,
        enum_declaration,
        interface_declaration,
    ]

    def __init__(self, **data):
        super().__init__(**data)

    @override
    def _file_prefix(self, root_node: Node) -> Tuple[str, List[Tuple[int, int]]]:
        used_ranges = []
        prefix = ""
        for child in root_node.children:
            if child.type == self.package_declaration:
                prefix = to_str(child).strip() + "\n"
        package_nodes = get_children(root_node, [self.package_declaration])
        for package_node in package_nodes:
            used_ranges.append((package_node.start_byte, package_node.end_byte))
        return prefix, used_ranges

    @override
    def _get_imports(self, tree: Tree) -> Dict[str, Node]:
        import_nodes = get_children(tree.root_node, [self.import_declaration])
        import_dict = {}
        for import_node in import_nodes:
            last_child = import_node.children[-2].children[-1]
            import_name = to_str(last_child).strip()
            if import_name == "*":
                import_name = to_str(import_node)
            import_dict[import_name] = import_node
        return import_dict

    @override
    def _build_additional_context(
        self, function_node: Node, root_node: Node
    ) -> Tuple[str, str]:
        context: List[str] = []
        context_no_doc: List[str] = []
        while function_node.parent is not None:
            if function_node.type in self.object_declarations:
                with_doc, without_doc = self._build_java_object_context(
                    function_node, root_node
                )
                context.insert(0, with_doc)
                context_no_doc.insert(0, without_doc)
            function_node = function_node.parent
        with_doc = "".join(context).rstrip()
        without_doc = "".join(context_no_doc).rstrip()
        return (
            with_doc + ("" if with_doc else ""),
            without_doc + ("" if without_doc else ""),
        )

    def _build_java_object_context(
        self, obj_node: Node, root_node: Node
    ) -> Tuple[str, str]:
        """Build context for Java objects (classes, enums, interfaces)."""
        obj_type = obj_node.type

        if obj_type in (self.class_declaration, self.record_declaration):
            return self._build_java_class_like_context(obj_node, root_node, "class")
        elif obj_type == self.enum_declaration:
            return self._build_java_class_like_context(obj_node, root_node, "enum")
        elif obj_type == self.interface_declaration:
            return self._build_java_class_like_context(obj_node, root_node, "interface")

        return ("", "")

    def _build_java_class_like_context(
        self, node: Node, root_node: Node, context_type: str
    ) -> Tuple[str, str]:
        """Unified context building for Java classes, enums, and interfaces."""
        body = node.child_by_field_name(self.class_body_field)
        if not body:
            text = to_str(node)
            return (text, text)

        header = self._get_function_signature(node, root_node)
        doc = self._get_docstring(node)
        header_with_doc = (
            f"{header}{' ' * (node.start_point.column + 4)}{doc}" if doc else header
        )

        inner_parts = []

        if context_type == "enum":
            constants = [
                to_str(child)
                for child in body.children
                if child.type == self.enum_constant
            ]
            const_block = (",".join(constants) + ";") if constants else ""
            inner_parts.append(const_block)

            decl = next(
                (
                    child
                    for child in body.children
                    if child.type == self.enum_body_declarations
                ),
                None,
            )
            if decl:
                decl_parts = [
                    to_str(child)
                    for child in decl.children
                    if child.type in self.enum_inner_types
                ]
                inner_parts.append("".join(decl_parts))

        elif context_type == "interface":
            constants = [
                to_str(child)
                for child in body.children
                if child.type == self.constant_declaration
            ]
            methods = [
                to_str(child)
                for child in body.children
                if child.type in self.function_definition_types
            ]
            inner_parts.extend(["".join(constants), "".join(methods)])

        else:
            parts = [
                to_str(child)
                for child in body.children
                if child.type in self.class_header_inner_types
            ]
            inner_parts.extend(parts)

        ctor = self._find_constructor(body)
        if ctor:
            inner_parts.append(self._build_node_with_decorators(ctor))

        inner = "".join(part for part in inner_parts if part.strip())
        close = (" " * node.start_point.column) + "}"

        with_doc = (
            "\n\n".join(x for x in [header_with_doc, inner] if x).rstrip() + close
        )
        without_doc = "\n\n".join(x for x in [header, inner] if x).rstrip() + close

        return with_doc, without_doc

    def _get_function_signature(self, node: Node, root_node: Node) -> str:
        indent = node.start_point.column
        body_node = node.child_by_field_name(self.class_body_field)
        if not body_node:
            return to_str(node)
        text = root_node.text
        if text:
            sig = text[node.start_byte : body_node.start_byte].decode().rstrip()
        else:
            sig = ""
        return (" " * indent) + sig + " {"

    def _get_class_member_ranges(self, current_node: Node) -> List[Tuple[int, int]]:
        used_ranges = []

        parent = current_node.parent
        if parent:
            field_nodes = get_children(parent, [self.field_declaration])
            for field_node in field_nodes:
                used_ranges.append((field_node.start_byte, field_node.end_byte))

            constant_nodes = get_children(parent, [self.constant_declaration])
            for constant_node in constant_nodes:
                used_ranges.append((constant_node.start_byte, constant_node.end_byte))

        return used_ranges

    def _get_module_variables(self, tree: Tree) -> Dict[str, Node]:
        """Java doesn't have module-level variables like Python or C macros."""
        return {}

    def _build_node_with_decorators(self, node: Node) -> str:
        """Build a node including any decorators/annotations."""
        if node.parent and node.parent.type == self.decorator_type:
            return to_str(node.parent)
        return to_str(node)
