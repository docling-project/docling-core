from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    GroupItem,
    GroupLabel,
    DocItem,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    RefItem,
    PictureItem,
)

class MarkdownSummaryParams(CommonParams):
    """Markdown-specific serialization parameters."""

    use_markdown_headers: bool = False
    
class MarkdownSummarySerializer(DocSerializer):
    """Markdown-specific document summary serializer."""

        params: MarkdownParams = MarkdownParams()

    @override
    def serialize_bold(self, text: str, **kwargs: Any):
        """Apply Markdown-specific bold serialization."""
        return f"**{text}**"

    @override
    def serialize_italic(self, text: str, **kwargs: Any):
        """Apply Markdown-specific italic serialization."""
        return f"*{text}*"

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any):
        """Apply Markdown-specific strikethrough serialization."""
        return f"~~{text}~~"

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ):
        """Apply Markdown-specific hyperlink serialization."""
        return f"[{text}]({str(hyperlink)})"

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts."""
        text_res = "\n\n".join([p.text for p in parts if p.text])

        return create_ser_result(text=text_res, span_source=parts)
    
    def _create_document_outline(self, doc: DoclingDocument) -> str:
        label_counter: dict[DocItemLabel, int] = {
            DocItemLabel.TABLE: 0,
            DocItemLabel.PICTURE: 0,
            DocItemLabel.TEXT: 0,
        }

        lines = []
        for item, level in doc.iterate_items(with_groups=True):
            if isinstance(item, TitleItem):
                lines.append(f"title (reference={item.self_ref}): {item.text}")
                
            elif isinstance(item, SectionHeaderItem):
                lines.append(
                    f"section-header (level={item.level}, reference={item.self_ref}): {item.text}"
                )

            elif isinstance(item, ListItem):
                continue
            
            elif isinstance(item, TextItem):
                lines.append(f"{item.label} (reference={item.self_ref})")
                
            elif isinstance(item, TableItem):
                label_counter[item.label] += 1
                lines.append(
                    f"{item.label} {label_counter[item.label]} (reference={item.self_ref})"
                )
                
            elif isinstance(item, PictureItem):
                label_counter[item.label] += 1
                lines.append(
                    f"{item.label} {label_counter[item.label]} (reference={item.self_ref})"
                )

        outline = "\n\n".join(lines)

        return outline

