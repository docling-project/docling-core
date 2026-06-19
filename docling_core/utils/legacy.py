"""Legacy compatibility hack for docling<2.103.

This does not recreate the removed legacy document model. It only keeps old
docling code from crashing when it accesses the deprecated converter hook.
"""


class _NullLegacyDocument:
    # Legacy callers may iterate this and call the methods below. Keep the
    # object inert so old package code can limp along without real legacy data.
    main_text: list[object] = []

    def _resolve_ref(self, item):
        return item

    def export_to_markdown(self, *args, **kwargs):
        return ""

    def export_to_document_tokens(self, *args, **kwargs):
        return ""


def docling_document_to_legacy(doc):
    # Compatibility shim only: older docling expects this symbol to exist.
    return _NullLegacyDocument()
