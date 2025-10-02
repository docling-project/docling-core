from typing import Any, Dict, Iterator, List, Optional, Tuple

from tree_sitter import Node, Parser, Tree

from docling_core.transforms.chunker import BaseChunker
from docling_core.transforms.chunker.code_chunk_utils.chunk_utils import (
    ChunkBuilder,
    ChunkSizeProcessor,
    RangeTracker,
)
from docling_core.transforms.chunker.code_chunk_utils.types import CodeChunk
from docling_core.transforms.chunker.code_chunk_utils.utils import (
    Language,
    get_children,
    to_str,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.labels import DocItemLabel


class CodeChunker(BaseChunker):
    """Data model for code chunker."""

    language: Language
    ts_language: Any
    parser: Any
    function_body: str
    constructor_name: str
    decorator_type: str
    class_definition_types: List[str]
    docs_types: List[str]
    expression_types: List[str]
    chunk_prefix: str
    chunk_suffix: str
    function_definition_types: List[str]
    tokenizer: BaseTokenizer
    min_chunk_size: int
    max_tokens: int
    class_body_field: str = "body"
    utf8_encoding: str = "utf-8"
    name_field: str = "name"
    expression_statement: str = "expression_statement"
    string_field: str = "string"
    identifiers: List[str] = ["identifier", "type_identifier"]
    definition_field: str = "definition"
    copyright_words: List[str] = [
        "copyright",
        "license",
        "licensed under",
        "all rights reserved",
    ]

    def __init__(self, **data):
        super().__init__(**data)
        if self.ts_language is None:
            self.ts_language = self.language.get_tree_sitter_language()
        if self.parser is None:
            self.parser = Parser(self.ts_language)

    @property
    def max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        return self.tokenizer.get_max_tokens()

    def parse_code(self, code: str) -> Tree:
        """Get tree sitter parser"""
        return self.parser.parse(bytes(code, self.utf8_encoding))

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[CodeChunk]:
        """Chunk the provided code by methods."""
        if not dl_doc.texts:
            return

        code_blocks = [t.text for t in dl_doc.texts if t.label == DocItemLabel.CODE]
        if not code_blocks:
            return

        for code in code_blocks:
            tree = self.parse_code(code)
            import_nodes = self._get_imports(tree)
            module_variables = self._get_module_variables(tree)
            range_tracker = RangeTracker()
            chunk_builder = ChunkBuilder(dl_doc.origin) if dl_doc.origin else None
            size_processor = ChunkSizeProcessor(
                self.tokenizer, self.max_tokens, self.min_chunk_size, chunker=self
            )

            self._mark_copyright_comments(tree.root_node, range_tracker)

            all_chunks = []

            functions = self._get_all_functions(tree.root_node, "")
            for node in functions:
                for chunk, chunk_used_ranges in self._yield_function_chunks_with_ranges(
                    node, tree.root_node, import_nodes, chunk_builder, module_variables
                ):
                    range_tracker.extend(chunk_used_ranges)
                    all_chunks.append((chunk, chunk_used_ranges))

            if module_variables:
                self._track_constructor_variables(
                    tree.root_node, module_variables, range_tracker
                )

            empty_classes = self._get_classes_no_methods(tree.root_node, "")
            for node in empty_classes:
                if chunk_builder:
                    for chunk, chunk_used_ranges in self._yield_class_chunk_with_ranges(
                        node, import_nodes, chunk_builder
                    ):
                        range_tracker.extend(chunk_used_ranges)
                        all_chunks.append((chunk, chunk_used_ranges))

            if chunk_builder:
                for chunk in chunk_builder.process_orphan_chunks(
                    range_tracker.get_used_ranges(), dl_doc
                ):
                    all_chunks.append((chunk, []))

            for chunk, _ in size_processor.process_chunks(all_chunks):
                yield chunk

    def _mark_copyright_comments(
        self, root_node: Node, range_tracker: RangeTracker
    ) -> None:
        """Mark copyright comments as used."""
        comment_nodes = get_children(root_node, self.docs_types)
        for node in comment_nodes:
            comment_text = to_str(node).lower()
            if any(keyword in comment_text for keyword in self.copyright_words):
                range_tracker.mark_node_used(node)

    def _yield_function_chunks_with_ranges(
        self,
        node: Node,
        root_node: Node,
        import_nodes: Dict[str, Node],
        chunk_builder: Optional[ChunkBuilder],
        module_variables: Optional[Dict[str, Node]] = None,
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:

        docstring = self._get_docstring(node)
        additional_context, additional_context_no_docstring = (
            self._build_additional_context(node, root_node)
        )
        imports = self._build_imports(
            import_nodes, node, additional_context_no_docstring
        )
        function_line_start, _ = node.start_point
        function_line_end, _ = node.end_point
        signature_line_end, _ = self._get_function_signature_end(node)
        function_name = self.language.get_function_name(node) or "unknown_function"
        prefix, prefix_range = self._file_prefix(root_node)

        used_ranges = []
        used_ranges.append((node.start_byte, node.end_byte))

        if imports:
            used_imports = self._find_used_imports_in_function(
                import_nodes, node, additional_context_no_docstring, module_variables
            )
            for import_name in sorted(used_imports):
                if import_name in import_nodes:
                    import_node = import_nodes[import_name]
                    import_ranges = self._get_import_ranges_with_comments(import_node)
                    used_ranges.extend(import_ranges)

        if prefix:
            used_ranges.extend(prefix_range)

        if additional_context:
            current_node = node
            while current_node.parent:
                if current_node.parent.type in self.class_definition_types:
                    used_ranges.append(
                        (current_node.parent.start_byte, current_node.parent.end_byte)
                    )
                    used_ranges.extend(
                        self._get_class_member_ranges(current_node.parent)
                    )
                    break
                current_node = current_node.parent

        module_variable_definitions = ""
        if module_variables:
            used_variables = self._find_used_variables(node)
            for var_name in sorted(used_variables):
                if var_name in module_variables:
                    var_def_node = module_variables[var_name]
                    var_ranges = self._get_variable_ranges_with_comments(var_def_node)
                    used_ranges.extend(var_ranges)
                    var_node = self._get_variable_with_comments(var_def_node, root_node)
                    var_text = to_str(var_node)
                    module_variable_definitions += var_text + "\n"

        function_content = self._build_function(node)
        function_no_docstring = (
            function_content.replace(docstring, "") if docstring else function_content
        )

        base_content = f"{prefix}{imports}{module_variable_definitions}{additional_context_no_docstring}{function_no_docstring}"

        if chunk_builder:
            yield chunk_builder.build_function_chunk(
                base_content,
                function_name,
                docstring,
                function_line_start,
                function_line_end,
                signature_line_end,
            ), used_ranges

    def _yield_class_chunk_with_ranges(
        self, node: Node, import_nodes: Dict[str, Node], chunk_builder: ChunkBuilder
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:
        docstring = self._get_docstring(node)
        function_content = self._build_class_with_comments(node)
        imports = self._build_imports(import_nodes, node, function_content)
        function_line_start, _ = node.start_point
        function_line_end, _ = node.end_point
        class_name = self.language.get_function_name(node) or "unknown_class"

        root_node = node
        while root_node.parent:
            root_node = root_node.parent
        prefix, prefix_range = self._file_prefix(root_node)

        used_ranges = []
        class_ranges = self._get_class_ranges_with_comments(node)
        used_ranges.extend(class_ranges)

        if imports:
            used_imports = self._find_used_imports_in_function(
                import_nodes, node, function_content, None
            )
            for import_name in sorted(used_imports):
                if import_name in import_nodes:
                    import_node = import_nodes[import_name]
                    import_ranges = self._get_import_ranges_with_comments(import_node)
                    used_ranges.extend(import_ranges)

        if prefix:
            used_ranges.extend(prefix_range)

        function_no_docstring = (
            function_content.replace(docstring, "") if docstring else function_content
        )
        content_no_docstring = f"{prefix}{imports}{function_no_docstring}"

        if chunk_builder:
            yield chunk_builder.build_class_chunk(
                content_no_docstring,
                class_name,
                docstring,
                function_line_start,
                function_line_end,
            ), used_ranges

    def _file_prefix(self, root_node: Node) -> Tuple[str, List]:
        return "", []

    def _get_function_body(self, node: Node) -> Optional[Node]:
        return next(
            (child for child in node.children if child.type == self.function_body), None
        )

    def _get_docstring(self, node: Node) -> str:
        if node.prev_named_sibling and node.prev_named_sibling.type in self.docs_types:
            text = node.prev_named_sibling.text
            return text.decode(self.utf8_encoding) if text else ""
        return ""

    def _get_all_functions(self, node: Node, parent_type: str) -> List[Node]:
        """Get all functions in the file."""
        if not node or parent_type in self.function_definition_types:
            return []

        nodes = []

        if node.type in self.function_definition_types:
            if self.language.is_collectable_function(node, self.constructor_name):
                nodes.append(node)
            elif self._is_constructor(node):
                if self._is_only_function_in_class(node):
                    nodes.append(node)

        for child in node.children:
            nodes.extend(self._get_all_functions(child, node.type))

        return nodes

    def _get_classes_no_methods(self, node: Node, parent_type: str) -> List[Node]:
        """Get classes and interfaces without methods."""

        def has_methods(class_node: Node) -> bool:
            return any(
                child.type in self.function_definition_types
                or any(
                    grandchild.type in self.function_definition_types
                    for grandchild in child.children
                )
                for child in class_node.children
            )

        if not node or parent_type in self.class_definition_types:
            return []

        nodes = []
        if node.type in self.class_definition_types and not has_methods(node):
            nodes.append(node)

        for child in node.children:
            nodes.extend(self._get_classes_no_methods(child, node.type))

        return nodes

    def _get_class_member_ranges(self, class_node: Node) -> List[Tuple[int, int]]:
        return []

    def _get_module_variables(self, tree: Tree) -> Dict[str, Node]:
        """Get module-level variables/macros. Must be implemented by language-specific chunkers."""
        raise NotImplementedError

    def _find_used_variables(self, function_node: Node) -> set:
        """Find variable/macro names used within a function. Default implementation returns empty set."""
        return set()

    def _get_variable_with_comments(self, var_node: Node, root_node: Node) -> Node:
        """Get variable node including any preceding comments. Default implementation returns the node as-is."""
        return var_node

    def _get_function_signature_end(self, node: Node) -> Tuple[int, int]:
        body_node = self._get_function_body(node)
        return body_node.start_point if body_node else node.end_point

    def _build_function(self, function_node: Node) -> str:
        if function_node.parent and function_node.parent.type == self.decorator_type:
            function_node = function_node.parent
        return to_str(function_node)

    def _build_class_with_comments(self, class_node: Node) -> str:
        """Build class content including any preceding comments and docstrings."""
        current = class_node.prev_sibling
        comment_parts: List[str] = []

        while current and current.type in self.docs_types:
            current_end_line = current.end_point[0]
            class_start_line = class_node.start_point[0]

            if current_end_line <= class_start_line:
                comment_parts.insert(0, to_str(current))
                current = current.prev_sibling
            else:
                break

        if comment_parts:
            result = "".join(comment_parts) + "\n" + to_str(class_node)
            return result
        else:
            return to_str(class_node)

    def _build_imports(
        self,
        imports: Dict[str, Node],
        function_node: Node,
        additional_context: str = "",
    ) -> str:
        used, set_imports = set(), set()

        def find_used_imports(node):
            if (
                node.type in self.identifiers
                and node.text.decode(self.utf8_encoding) in imports
            ):
                used.add(node.text.decode(self.utf8_encoding))
            for child in node.children:
                find_used_imports(child)

        find_used_imports(function_node)

        if additional_context:
            for import_name in imports.keys():
                if import_name in additional_context:
                    used.add(import_name)

        for import_name, import_node in imports.items():
            if "*" in import_name:
                import_text = self._get_import_with_comments(import_node)
                set_imports.add(import_text)

        for u in used:
            import_text = self._get_import_with_comments(imports[u])
            set_imports.add(import_text)

        return "\n".join(sorted(set_imports)) + "\n"

    def _find_used_imports_in_function(
        self,
        imports: Dict[str, Node],
        function_node: Node,
        additional_context: str = "",
        module_variables: Optional[Dict[str, Node]] = None,
    ) -> set:
        """Find which imports are used in a function and its additional context."""
        used = set()

        def find_used_imports(node):
            if (
                node.type in self.identifiers
                and node.text.decode(self.utf8_encoding) in imports
            ):
                used.add(node.text.decode(self.utf8_encoding))
            for child in node.children:
                find_used_imports(child)

        find_used_imports(function_node)

        if additional_context:
            for import_name in imports.keys():
                if import_name in additional_context:
                    used.add(import_name)

        if module_variables:
            used_variables = self._find_used_variables(function_node)

            for var_name in used_variables:
                if var_name in module_variables:
                    var_def_node = module_variables[var_name]
                    find_used_imports(var_def_node)

        for import_name in imports.keys():
            if "*" in import_name:
                used.add(import_name)

        return used

    def _get_node_with_comments(self, node: Node) -> str:
        """Get node text including any preceding comments."""

        current = node.prev_sibling
        comment_parts: List[str] = []

        while current and current.type in self.docs_types:
            current_end_line = current.end_point[0]
            node_start_line = node.start_point[0]

            if current_end_line <= node_start_line:
                comment_parts.insert(0, to_str(current))
                current = current.prev_sibling
            else:
                break

        if comment_parts:
            result = "".join(comment_parts) + "\n" + to_str(node)
            return result
        else:
            return to_str(node)

    def _get_import_with_comments(self, import_node: Node) -> str:
        """Get import text including any preceding comments."""
        return self._get_node_with_comments(import_node)

    def _get_node_ranges_with_comments(self, node: Node) -> List[Tuple[int, int]]:
        """Get node ranges including any preceding comments."""
        ranges = []

        current = node.prev_sibling

        while current and current.type in self.docs_types:
            current_end_line = current.end_point[0]
            node_start_line = node.start_point[0]

            if current_end_line <= node_start_line:
                ranges.append((current.start_byte, current.end_byte))
                current = current.prev_sibling
            else:
                break

        ranges.append((node.start_byte, node.end_byte))

        return ranges

    def _get_variable_ranges_with_comments(
        self, var_node: Node
    ) -> List[Tuple[int, int]]:
        """Get variable ranges including any preceding comments."""
        return self._get_node_ranges_with_comments(var_node)

    def _get_import_ranges_with_comments(
        self, import_node: Node
    ) -> List[Tuple[int, int]]:
        """Get import ranges including any preceding comments."""
        return self._get_node_ranges_with_comments(import_node)

    def _get_class_ranges_with_comments(
        self, class_node: Node
    ) -> List[Tuple[int, int]]:
        """Get class ranges including any preceding comments and docstrings."""
        return self._get_node_ranges_with_comments(class_node)

    def _build_additional_context(
        self, function_node: Node, root_node: Node
    ) -> Tuple[str, str]:
        context = ""
        context_no_docstring = ""
        node = function_node

        while node.parent:
            if node.type in self.class_definition_types:
                with_doc, without_doc = self._build_class_context(node, root_node)
                context = f"{with_doc}\n{context}"
                context_no_docstring = f"{without_doc}\n{context_no_docstring}"
            node = node.parent

        return context, context_no_docstring

    def _is_docstring(self, node: Node) -> bool:
        """Determines if a node is a docstring"""
        return bool(
            node.type == self.expression_statement
            and node.named_children
            and node.named_children[0].type == self.string_field
        )

    def _get_imports(self, tree: Tree) -> Dict[str, Node]:
        """Get imports from the AST. Must be implemented by language-specific chunkers."""
        raise NotImplementedError

    def _build_class_context(
        self, class_node: Node, root_node: Node
    ) -> Tuple[str, str]:
        class_indent = class_node.start_point.column
        start_byte = class_node.start_byte

        if class_node.parent and class_node.parent.type == self.decorator_type:
            start_byte = class_node.parent.start_byte
            class_indent = class_node.parent.start_point.column

        body_node = class_node.child_by_field_name(self.class_body_field)

        if not body_node:
            return ("", "")

        text = root_node.text
        if text:
            header_text = text[start_byte : body_node.start_byte].decode().rstrip()
        else:
            header_text = ""
        header = f"{' ' * class_indent}{header_text}\n"
        docstring = self._get_docstring(class_node)
        header_with_docstring = (
            f"{header}{' ' * (class_indent + 4)}{docstring}\n" if docstring else header
        )

        fields = [
            to_str(child)
            for child in body_node.children
            if child.type in self.expression_types and not self._is_docstring(child)
        ]
        fields_text = "\n".join(fields)
        constructor_node = self._find_constructor(body_node)
        if constructor_node:
            constructor_doc = self._get_docstring(constructor_node)
            constructor_text = self._build_function(constructor_node)
            constructor_text_no_doc = (
                constructor_text.replace(constructor_doc, "")
                if constructor_doc
                else constructor_text
            )
        else:
            constructor_text = constructor_text_no_doc = ""

        with_doc = f"{header_with_docstring}\n{fields_text}\n{constructor_text}".strip()
        without_doc = f"{header}\n{fields_text}\n{constructor_text_no_doc}".strip()

        return with_doc, without_doc

    def _find_constructor(self, body: Node) -> Optional[Node]:
        for child in body.children:
            definition_field = child.child_by_field_name(self.definition_field)
            if self._is_constructor(child) or (
                child.type == self.decorator_type
                and definition_field
                and self._is_constructor(definition_field)
            ):
                return child
        return None

    def _is_constructor(self, node: Node) -> bool:
        if node is None:
            return False

        child = node.child_by_field_name(self.name_field)
        if child is None:
            return False

        name_field = node.child_by_field_name(self.name_field)
        if not name_field or not name_field.text:
            return False
        return (
            node.type in self.function_definition_types
            and name_field.text.decode(self.utf8_encoding) == self.constructor_name
        )

    def _is_only_function_in_class(self, constructor_node: Node) -> bool:
        """Check if a constructor is the only function in its class."""
        class_node = constructor_node.parent
        while class_node and class_node.type not in self.class_definition_types:
            class_node = class_node.parent

        if not class_node:
            return False

        body_node = class_node.child_by_field_name(self.class_body_field)
        if not body_node:
            return False

        function_count = 0
        for child in body_node.children:
            if (
                child.type in self.function_definition_types
                and child != constructor_node
            ):
                function_count += 1

        return function_count == 0

    def _track_constructor_variables(
        self, node: Node, module_variables: Dict[str, Node], range_tracker: RangeTracker
    ) -> None:
        """Track variables used in constructor functions that aren't being chunked separately."""
        if node.type in self.function_definition_types and self._is_constructor(node):
            if not self._is_only_function_in_class(node):
                used_variables = self._find_used_variables(node)
                for var_name in used_variables:
                    if var_name in module_variables:
                        var_def_node = module_variables[var_name]
                        range_tracker.mark_node_used(var_def_node)

        for child in node.children:
            self._track_constructor_variables(child, module_variables, range_tracker)
