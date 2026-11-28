# Optional group provenance prototype

Layout provenance belongs to `NodeItem`, so `DocItem`, `GroupItem`, `ListGroup`,
and `InlineGroup` share the existing `list[ProvenanceItem]` representation.
An omitted value defaults to an empty list. Empty group provenance is omitted
from JSON; content items retain their existing serialization of empty provenance.
Groups use `charspan=(0, 0)` because they have no own text buffer.

The group creation methods accept an optional keyword argument `prov`, following
the content creation methods. Multiple locations can be supplied through the
model's `prov` list. Document validation applies the existing page-boundary
clamping to group locations, including root groups.

Provenance is explicit and owned by the node on which it is stored. Children do
not inherit a group's bounding box or page membership. Groups remain structural
containers during traversal and filtering, with the existing behavior for their
children. Page filtering removes group provenance for discarded pages;
concatenation and reference normalization shift group page numbers alongside
content page numbers. Concatenation combines the source body roots' provenance
on the resulting body root.

Group provenance changes neither content rendering nor semantic chunking.
Serialization spans and chunk `doc_items` continue to refer to content items.

The JSON schema version is 1.11.0. The 1.11-to-1.10 downgrade projector removes
group provenance, including body and furniture roots, while preserving content
provenance and hierarchy. Old JSON documents load with empty group provenance.

DocLang serialization/deserialization is outside this prototype. Existing
DocLang/DCLX paths do not preserve the new group geometry; their behavior needs
a separate analysis before this feature is ready for general use.
