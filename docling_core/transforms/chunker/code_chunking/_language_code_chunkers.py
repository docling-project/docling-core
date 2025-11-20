"""Language-specific code chunker implementations."""

import hashlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import Field
from tree_sitter import Node, Parser, Tree
from typing_extensions import override

from docling_core.transforms.chunker import (
    BaseChunker,
    CodeChunk,
    CodeChunkType,
    CodeDocMeta,
)
from docling_core.transforms.chunker.code_chunking._utils import (
    _get_children,
    _get_default_tokenizer,
    _get_function_name,
    _get_import_query,
    _get_tree_sitter_language,
    _has_child,
    _is_collectable_function,
    _query_tree,
    _to_str,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import CodeItem, DocumentOrigin
from docling_core.types.doc.labels import CodeLanguageLabel


def _new_hash(code: str) -> int:
    """Generate SHA256 hash for code."""
    return int(hashlib.sha1(bytes(code, "utf-8")).hexdigest(), 16)


class _RangeTracker:
    """Handles tracking and management of used byte ranges in code."""

    def __init__(self) -> None:
        """Initialize the range tracker with an empty list of used ranges."""
        self.used_ranges: List[Tuple[int, int]] = []

    def mark_used(self, start_byte: int, end_byte: int) -> None:
        """Mark a range as used."""
        self.used_ranges.append((start_byte, end_byte))

    def mark_node_used(self, node: Node) -> None:
        """Mark a node's range as used."""
        self.mark_used(node.start_byte, node.end_byte)

    def merge_ranges(self) -> List[Tuple[int, int]]:
        """Merge overlapping ranges and return sorted list."""
        if not self.used_ranges:
            return []

        sorted_ranges = sorted(self.used_ranges)
        merged: List[Tuple[int, int]] = []

        for start, end in sorted_ranges:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        return merged

    def find_gaps(self, total_length: int) -> List[Tuple[int, int]]:
        """Find gaps between used ranges."""
        merged = self.merge_ranges()
        gaps = []
        last_end = 0

        for start, end in merged:
            if last_end < start:
                gaps.append((last_end, start))
            last_end = end

        if last_end < total_length:
            gaps.append((last_end, total_length))

        return gaps

    def get_used_ranges(self) -> List[Tuple[int, int]]:
        """Get all used ranges."""
        return self.used_ranges.copy()

    def clear(self) -> None:
        """Clear all used ranges."""
        self.used_ranges.clear()

    def extend(self, ranges: List[Tuple[int, int]]) -> None:
        """Add multiple ranges at once."""
        self.used_ranges.extend(ranges)


class _ChunkMetadataBuilder:
    """Builds metadata for code chunks."""

    def __init__(self, origin: Optional[DocumentOrigin] = None):
        """Initialize the metadata builder with document origin."""
        self.origin = origin

    def build_function_metadata(
        self,
        *,
        item: CodeItem,
        function_name: str,
        docstring: str,
        content: str,
        start_line: int,
        end_line: int,
        signature_end_line: int,
    ) -> CodeDocMeta:
        """Build metadata for function chunks."""
        return CodeDocMeta(
            doc_items=[item],
            part_name=function_name,
            docstring=docstring,
            sha256=_new_hash(content),
            start_line=start_line,
            end_line=end_line,
            end_line_signature=signature_end_line,
            origin=self.origin,
            chunk_type=CodeChunkType.FUNCTION,
        )

    def build_class_metadata(
        self,
        *,
        item: CodeItem,
        class_name: str,
        docstring: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> CodeDocMeta:
        """Build metadata for class chunks."""
        return CodeDocMeta(
            doc_items=[item],
            part_name=class_name,
            docstring=docstring,
            sha256=_new_hash(content),
            start_line=start_line,
            end_line=end_line,
            end_line_signature=end_line,
            origin=self.origin,
            chunk_type=CodeChunkType.CLASS,
        )

    def build_preamble_metadata(
        self, *, item: CodeItem, content: str, start_line: int, end_line: int
    ) -> CodeDocMeta:
        """Build metadata for preamble chunks."""
        return CodeDocMeta(
            doc_items=[item],
            sha256=_new_hash(content),
            start_line=start_line,
            end_line=end_line,
            origin=self.origin,
            chunk_type=CodeChunkType.PREAMBLE,
        )

    def calculate_line_numbers(
        self, code: str, start_byte: int, end_byte: int
    ) -> Tuple[int, int]:
        """Calculate line numbers from byte positions."""
        start_line = code[:start_byte].count("\n") + 1
        if end_byte > 0 and end_byte <= len(code):
            end_line = code[:end_byte].count("\n") + 1
            if end_byte < len(code) and code[end_byte - 1] == "\n":
                end_line -= 1
        else:
            end_line = start_line
        return start_line, end_line


class _ChunkBuilder:
    """Builds code chunks from nodes and content."""

    def __init__(self, *, item: CodeItem, origin: Optional[DocumentOrigin] = None):
        """Initialize the chunk builder with document origin."""
        self.metadata_builder = _ChunkMetadataBuilder(origin)
        self.item = item

    def build_function_chunk(
        self,
        content: str,
        function_name: str,
        docstring: str,
        start_line: int,
        end_line: int,
        signature_end_line: int,
    ) -> CodeChunk:
        """Build a function chunk."""
        metadata = self.metadata_builder.build_function_metadata(
            item=self.item,
            function_name=function_name,
            docstring=docstring,
            content=content,
            start_line=start_line,
            end_line=end_line,
            signature_end_line=signature_end_line,
        )
        return CodeChunk(text=content, meta=metadata)

    def build_class_chunk(
        self,
        content: str,
        class_name: str,
        docstring: str,
        start_line: int,
        end_line: int,
    ) -> CodeChunk:
        """Build a class chunk."""
        metadata = self.metadata_builder.build_class_metadata(
            item=self.item,
            class_name=class_name,
            docstring=docstring,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )
        return CodeChunk(text=content, meta=metadata, doc_items=[self.item])

    def build_preamble_chunk(
        self, content: str, start_line: int, end_line: int
    ) -> CodeChunk:
        """Build a preamble chunk."""
        metadata = self.metadata_builder.build_preamble_metadata(
            item=self.item,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )
        return CodeChunk(text=content, meta=metadata, doc_items=[self.item])

    def process_orphan_chunks(
        self, used_ranges: List[Tuple[int, int]], dl_doc
    ) -> Iterator[CodeChunk]:
        """Process orphan chunks (preamble) from unused code ranges."""
        from docling_core.types.doc.labels import DocItemLabel

        code = next(
            (t.text for t in dl_doc.texts if t.label == DocItemLabel.CODE), None
        )
        if not code:
            return

        range_tracker = _RangeTracker()
        range_tracker.extend(used_ranges)

        gaps = range_tracker.find_gaps(len(code))
        orphan_pieces = []
        for start_byte, end_byte in gaps:
            orphan_text = code[start_byte:end_byte].strip()
            if orphan_text:
                orphan_pieces.append((orphan_text, start_byte, end_byte))

        if orphan_pieces:
            merged_content = "\n\n".join(piece[0] for piece in orphan_pieces)
            first_start_byte = orphan_pieces[0][1]
            last_end_byte = orphan_pieces[-1][2]

            start_line, end_line = self.metadata_builder.calculate_line_numbers(
                code, first_start_byte, last_end_byte
            )
            yield self.build_preamble_chunk(merged_content, start_line, end_line)


class _ChunkSizeProcessor:
    """Processes chunks to split large ones into smaller pieces."""

    def __init__(
        self, tokenizer, max_tokens: int, min_chunk_size: int = 300, chunker=None
    ):
        """Initialize the chunk size processor with tokenizer and size constraints."""
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.min_chunk_size = min_chunk_size
        self.chunker = chunker

    def process_chunks(
        self, chunks_and_ranges: List[Tuple[CodeChunk, List[Tuple[int, int]]]]
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:
        """Process chunks and split large ones if needed."""
        for chunk, ranges in chunks_and_ranges:
            token_count = self.tokenizer.count_tokens(chunk.text)

            if token_count <= self.max_tokens:
                yield chunk, ranges
            else:
                yield from self._split_large_chunk(chunk, ranges)

    def _split_large_chunk(
        self, chunk: CodeChunk, ranges: List[Tuple[int, int]]
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:
        """Split a large chunk into smaller pieces."""
        if chunk.meta.chunk_type.value in {
            CodeChunkType.FUNCTION.value,
            CodeChunkType.METHOD.value,
        }:
            yield from self._split_function_chunk(chunk, ranges)
        else:
            yield from self._split_generic_chunk(chunk, ranges)

    def _split_function_chunk(
        self, chunk: CodeChunk, ranges: List[Tuple[int, int]]
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:
        """Split a large function chunk using the original sophisticated logic."""
        lines = chunk.text.split("\n")
        if not lines:
            yield chunk, ranges
            return

        signature_line = ""
        body_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip():
                signature_line = line
                body_start_idx = i + 1
                break

        if not signature_line:
            yield chunk, ranges
            return

        body_lines = lines[body_start_idx:]
        if not body_lines:
            yield chunk, ranges
            return

        if body_lines and body_lines[-1].strip() == "}":
            body_lines = body_lines[:-1]

        chunks = []
        current_chunk = [f"{signature_line}{self._get_chunk_prefix()}"]
        current_size = 0

        for line in body_lines:
            line_tokens = self.tokenizer.count_tokens(line)

            if current_size + line_tokens > self.max_tokens and len(current_chunk) > 1:
                chunks.append("".join(current_chunk) + f"{self._get_chunk_suffix()}")
                current_chunk = [f"{signature_line}{self._get_chunk_prefix()}"]
                current_size = 0

            current_chunk.append(line)
            current_size += line_tokens

        if current_chunk:
            chunks.append("".join(current_chunk) + f"{self._get_chunk_suffix()}")

        if len(chunks) > 1:
            last_chunk = chunks.pop()
            last_chunk_tokens = self.tokenizer.count_tokens(last_chunk)
            if last_chunk_tokens < self.min_chunk_size:
                chunks[-1] = (
                    chunks[-1].rstrip(self._get_chunk_suffix())
                    + "\n"
                    + last_chunk.lstrip(signature_line + f"{self._get_chunk_prefix()}")
                )
            else:
                chunks.append(last_chunk)

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            new_meta = chunk.meta.model_copy()
            new_meta.part_name = (
                f"{chunk.meta.part_name}_part_{i + 1}"
                if len(chunks) > 1
                else chunk.meta.part_name
            )

            sub_chunk = CodeChunk(text=chunk_text, meta=new_meta)
            yield sub_chunk, ranges

    def _get_chunk_prefix(self) -> str:
        """Get the chunk prefix for function splitting."""
        if self.chunker and hasattr(self.chunker, "chunk_prefix"):
            return self.chunker.chunk_prefix
        return " {\n"

    def _get_chunk_suffix(self) -> str:
        """Get the chunk suffix for function splitting."""
        if self.chunker and hasattr(self.chunker, "chunk_suffix"):
            return self.chunker.chunk_suffix
        return "\n}"

    def _split_generic_chunk(
        self, chunk: CodeChunk, ranges: List[Tuple[int, int]]
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:
        """Split a generic chunk by lines."""
        lines = chunk.text.split("\n")
        current_chunk_lines: List[str] = []
        current_size = 0
        chunk_number = 1

        for line in lines:
            line_tokens = self.tokenizer.count_tokens(line)

            if current_size + line_tokens > self.max_tokens and current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                if self.tokenizer.count_tokens(chunk_text) >= self.min_chunk_size:
                    yield self._create_split_chunk(
                        chunk, chunk_text, chunk_number
                    ), ranges
                    chunk_number += 1

                current_chunk_lines = [line]
                current_size = line_tokens
            else:
                current_chunk_lines.append(line)
                current_size += line_tokens

        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            if self.tokenizer.count_tokens(chunk_text) >= self.min_chunk_size:
                yield self._create_split_chunk(chunk, chunk_text, chunk_number), ranges

    def _create_split_chunk(
        self, original_chunk: CodeChunk, text: str, chunk_number: int
    ) -> CodeChunk:
        """Create a new chunk from split text."""
        new_meta = original_chunk.meta.model_copy()
        new_meta.part_name = f"{original_chunk.meta.part_name}_part_{chunk_number}"

        return CodeChunk(text=text, meta=new_meta)


class _CodeChunker(BaseChunker):
    """Data model for code chunker."""

    language: CodeLanguageLabel
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
            self.ts_language = _get_tree_sitter_language(self.language)
        if self.parser is None:
            self.parser = Parser(self.ts_language)

    def parse_code(self, code: str) -> Tree:
        """Get tree sitter parser."""
        return self.parser.parse(bytes(code, self.utf8_encoding))

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[CodeChunk]:
        """Chunk the provided code by methods."""
        for item, _ in dl_doc.iterate_items():
            if isinstance(item, CodeItem):
                code = item.text
                tree = self.parse_code(code)
                import_nodes = self._get_imports(tree)
                module_variables = self._get_module_variables(tree)
                range_tracker = _RangeTracker()
                chunk_builder = _ChunkBuilder(item=item, origin=dl_doc.origin)
                size_processor = _ChunkSizeProcessor(
                    self.tokenizer, self.max_tokens, self.min_chunk_size, chunker=self
                )

                self._mark_copyright_comments(tree.root_node, range_tracker)

                all_chunks = []

                functions = self._get_all_functions(tree.root_node, "")
                for node in functions:
                    for (
                        chunk,
                        chunk_used_ranges,
                    ) in self._yield_function_chunks_with_ranges(
                        node,
                        tree.root_node,
                        import_nodes,
                        chunk_builder,
                        module_variables,
                    ):
                        range_tracker.extend(chunk_used_ranges)
                        all_chunks.append((chunk, chunk_used_ranges))

                if module_variables:
                    self._track_constructor_variables(
                        tree.root_node, module_variables, range_tracker
                    )

                empty_classes = self._get_classes_no_methods(tree.root_node, "")
                for node in empty_classes:
                    for (
                        chunk,
                        chunk_used_ranges,
                    ) in self._yield_class_chunk_with_ranges(
                        node, import_nodes, chunk_builder
                    ):
                        range_tracker.extend(chunk_used_ranges)
                        all_chunks.append((chunk, chunk_used_ranges))

                for chunk in chunk_builder.process_orphan_chunks(
                    range_tracker.get_used_ranges(), dl_doc
                ):
                    all_chunks.append((chunk, []))

                for chunk, _ in size_processor.process_chunks(all_chunks):
                    yield chunk

    def _mark_copyright_comments(
        self, root_node: Node, range_tracker: _RangeTracker
    ) -> None:
        """Mark copyright comments as used."""
        comment_nodes = _get_children(root_node, self.docs_types)
        for node in comment_nodes:
            comment_text = _to_str(node).lower()
            if any(keyword in comment_text for keyword in self.copyright_words):
                range_tracker.mark_node_used(node)

    def _yield_function_chunks_with_ranges(
        self,
        node: Node,
        root_node: Node,
        import_nodes: Dict[str, Node],
        chunk_builder: _ChunkBuilder,
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
        function_name = _get_function_name(self.language, node) or "unknown_function"
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
                    var_text = _to_str(var_node)
                    module_variable_definitions += var_text + "\n"

        function_content = self._build_function(node)
        function_no_docstring = (
            function_content.replace(docstring, "") if docstring else function_content
        )

        base_content = (
            f"{prefix}{imports}{module_variable_definitions}"
            f"{additional_context_no_docstring}{function_no_docstring}"
        )

        yield chunk_builder.build_function_chunk(
            base_content,
            function_name,
            docstring,
            function_line_start,
            function_line_end,
            signature_line_end,
        ), used_ranges

    def _yield_class_chunk_with_ranges(
        self, node: Node, import_nodes: Dict[str, Node], chunk_builder: _ChunkBuilder
    ) -> Iterator[Tuple[CodeChunk, List[Tuple[int, int]]]]:
        docstring = self._get_docstring(node)
        function_content = self._build_class_with_comments(node)
        imports = self._build_imports(import_nodes, node, function_content)
        function_line_start, _ = node.start_point
        function_line_end, _ = node.end_point
        class_name = _get_function_name(self.language, node) or "unknown_class"

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
            if _is_collectable_function(self.language, node, self.constructor_name):
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
        return _to_str(function_node)

    def _build_class_with_comments(self, class_node: Node) -> str:
        """Build class content including any preceding comments and docstrings."""
        current = class_node.prev_sibling
        comment_parts: List[str] = []

        while current and current.type in self.docs_types:
            current_end_line = current.end_point[0]
            class_start_line = class_node.start_point[0]

            if current_end_line <= class_start_line:
                comment_parts.insert(0, _to_str(current))
                current = current.prev_sibling
            else:
                break

        if comment_parts:
            result = "".join(comment_parts) + "\n" + _to_str(class_node)
            return result
        else:
            return _to_str(class_node)

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
                comment_parts.insert(0, _to_str(current))
                current = current.prev_sibling
            else:
                break

        if comment_parts:
            result = "".join(comment_parts) + "\n" + _to_str(node)
            return result
        else:
            return _to_str(node)

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
        """Determines if a node is a docstring."""
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
            _to_str(child)
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
        self,
        node: Node,
        module_variables: Dict[str, Node],
        range_tracker: _RangeTracker,
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


class _PythonFunctionChunker(_CodeChunker):

    language: CodeLanguageLabel = CodeLanguageLabel.PYTHON
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
    max_tokens: int = 5000
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
        import_query = _get_import_query(self.language)
        if not import_query:
            return {}
        import_query_results = _query_tree(self.ts_language, tree, import_query)
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
            """Collect identifiers from node."""
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
    language: CodeLanguageLabel = CodeLanguageLabel.TYPESCRIPT
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
        import_query = _get_import_query(self.language)
        if not import_query:
            return {}
        import_query_results = _query_tree(self.ts_language, tree, import_query)
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
        super().__init__(language=CodeLanguageLabel.JAVASCRIPT)


class _CFunctionChunker(_CodeChunker):
    language: CodeLanguageLabel = CodeLanguageLabel.C
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
                if _has_child(
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
                if _has_child(
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
            """Collect identifiers from node."""
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

    language: CodeLanguageLabel = CodeLanguageLabel.JAVA
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
                prefix = _to_str(child).strip() + "\n"
        package_nodes = _get_children(root_node, [self.package_declaration])
        for package_node in package_nodes:
            used_ranges.append((package_node.start_byte, package_node.end_byte))
        return prefix, used_ranges

    @override
    def _get_imports(self, tree: Tree) -> Dict[str, Node]:
        import_nodes = _get_children(tree.root_node, [self.import_declaration])
        import_dict = {}
        for import_node in import_nodes:
            last_child = import_node.children[-2].children[-1]
            import_name = _to_str(last_child).strip()
            if import_name == "*":
                import_name = _to_str(import_node)
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
            text = _to_str(node)
            return (text, text)

        header = self._get_function_signature(node, root_node)
        doc = self._get_docstring(node)
        header_with_doc = (
            f"{header}{' ' * (node.start_point.column + 4)}{doc}" if doc else header
        )

        inner_parts = []

        if context_type == "enum":
            constants = [
                _to_str(child)
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
                    _to_str(child)
                    for child in decl.children
                    if child.type in self.enum_inner_types
                ]
                inner_parts.append("".join(decl_parts))

        elif context_type == "interface":
            constants = [
                _to_str(child)
                for child in body.children
                if child.type == self.constant_declaration
            ]
            methods = [
                _to_str(child)
                for child in body.children
                if child.type in self.function_definition_types
            ]
            inner_parts.extend(["".join(constants), "".join(methods)])

        else:
            parts = [
                _to_str(child)
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
            return _to_str(node)
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
            field_nodes = _get_children(parent, [self.field_declaration])
            for field_node in field_nodes:
                used_ranges.append((field_node.start_byte, field_node.end_byte))

            constant_nodes = _get_children(parent, [self.constant_declaration])
            for constant_node in constant_nodes:
                used_ranges.append((constant_node.start_byte, constant_node.end_byte))

        return used_ranges

    def _get_module_variables(self, tree: Tree) -> Dict[str, Node]:
        """Java doesn't have module-level variables like Python or C macros."""
        return {}

    def _build_node_with_decorators(self, node: Node) -> str:
        """Build a node including any decorators/annotations."""
        if node.parent and node.parent.type == self.decorator_type:
            return _to_str(node.parent)
        return _to_str(node)
