import hashlib
from typing import Iterator, List, Tuple

from tree_sitter import Node

from docling_core.transforms.chunker.code_chunk_utils.types import (
    ChunkType,
    CodeChunk,
    CodeDocMeta,
)
from docling_core.types.doc.document import DocumentOrigin


def new_hash(code: str) -> int:
    """Generate SHA256 hash for code."""
    return int(hashlib.sha1(bytes(code, "utf-8")).hexdigest(), 16)


class RangeTracker:
    """Handles tracking and management of used byte ranges in code."""

    def __init__(self):
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


class ChunkMetadataBuilder:
    """Builds metadata for code chunks."""

    def __init__(self, origin: DocumentOrigin):
        self.origin = origin

    def build_function_metadata(
        self,
        function_name: str,
        docstring: str,
        content: str,
        start_line: int,
        end_line: int,
        signature_end_line: int,
    ) -> CodeDocMeta:
        """Build metadata for function chunks."""
        return CodeDocMeta(
            part_name=function_name,
            docstring=docstring,
            sha256=new_hash(content),
            start_line=start_line,
            end_line=end_line,
            end_line_signature=signature_end_line,
            origin=self.origin,
            chunk_type=ChunkType.FUNCTION,
        )

    def build_class_metadata(
        self,
        class_name: str,
        docstring: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> CodeDocMeta:
        """Build metadata for class chunks."""
        return CodeDocMeta(
            part_name=class_name,
            docstring=docstring,
            sha256=new_hash(content),
            start_line=start_line,
            end_line=end_line,
            end_line_signature=end_line,
            origin=self.origin,
            chunk_type=ChunkType.CLASS,
        )

    def build_preamble_metadata(
        self, content: str, start_line: int, end_line: int
    ) -> CodeDocMeta:
        """Build metadata for preamble chunks."""
        return CodeDocMeta(
            sha256=new_hash(content),
            start_line=start_line,
            end_line=end_line,
            origin=self.origin,
            chunk_type=ChunkType.PREAMBLE,
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


class ChunkBuilder:
    """Builds code chunks from nodes and content."""

    def __init__(self, origin: DocumentOrigin):
        self.metadata_builder = ChunkMetadataBuilder(origin)

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
            function_name, docstring, content, start_line, end_line, signature_end_line
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
            class_name, docstring, content, start_line, end_line
        )
        return CodeChunk(text=content, meta=metadata)

    def build_preamble_chunk(
        self, content: str, start_line: int, end_line: int
    ) -> CodeChunk:
        """Build a preamble chunk."""
        metadata = self.metadata_builder.build_preamble_metadata(
            content, start_line, end_line
        )
        return CodeChunk(text=content, meta=metadata)

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

        range_tracker = RangeTracker()
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


class ChunkSizeProcessor:
    """Processes chunks to split large ones into smaller pieces."""

    def __init__(
        self, tokenizer, max_tokens: int, min_chunk_size: int = 300, chunker=None
    ):
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
        if chunk.meta.chunk_type in ["function", "method"]:
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
                f"{chunk.meta.part_name}_part_{i+1}"
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
