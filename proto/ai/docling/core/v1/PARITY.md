# DoclingDocument Parity Contract

This document explains how `DoclingDocument` parity is maintained between:

- **Pydantic model** (`docling_core.types.doc.document.DoclingDocument`) as semantic source of truth.
- **Protobuf IDL** (`docling_document.proto`) as wire contract source of truth.

## Invariants

1. New/updated Pydantic fields must be represented in protobuf with equivalent meaning.
2. Enums in Pydantic should map to protobuf enums (not downgraded to string fields).
3. gRPC payloads use protobuf as the primary document transport.
4. Any intentional difference must be documented and validated.

## Intentional Differences (Keep Small)


| Pattern                                                                                                             | Type                          | Reason                                                                                                                                                                                                                                                                                                                                            | Status      |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `*_raw` companion strings (e.g. `label_raw`, `code_language_raw`, `language_raw`, `coord_origin_raw`, `script_raw`) | proto-only fallback field     | Preserves unrecognized values without breaking clients. The string carries the original source value when the enum tag is `*_UNSPECIFIED`.                                                                                                                                                                                                        | intentional |
| `TableData.grid` (and `*.chart_data.grid`)                                                                          | computed-field surfaced       | Pydantic `TableData.grid` is a `@computed_field` — derived from `table_cells` + cell row/col offsets. It IS in the Pydantic JSON dump, so proto surfaces it for parity. Not a divergence; the validator allowlists it because it doesn't appear in `model_fields`.                                                                                | intentional |
| `TrackSource.kind` (Pydantic-side only)                                                                             | discriminator absorbed        | Pydantic discriminated unions need a per-variant `kind: Literal[...]` field. Proto encodes the same information in the parent oneof tag (`SourceType.source.track`), so the per-variant string field is redundant on the wire.                                                                                                                    | intentional |
| `CodeItem` proto inlines `TextItemBase` fields instead of using `base = 1`                                          | inheritance-without-shadowing | All other text variants wrap a `TextItemBase base = 1` for shared fields. CodeItem is the only Pydantic text variant that overrides `meta` (FloatingMeta vs BaseMeta). Wrapping would surface two `meta` slots on the wire (`base.meta` AND `meta`) with no schema rule for which to populate. Inlining keeps a single, unambiguous `meta` field. | intentional |


Only fields matching a pattern listed here are allowed to differ intentionally. New `*_raw` patterns must be added to this table *and* registered in `_RAW_FALLBACK_SUFFIXES` in `docling_serve/grpc/schema_validator.py`. New discriminator-only fields must be registered in `_PYDANTIC_ONLY_DISCRIMINATORS` in the same file.

### `*_raw` Discriminator Contract

For every enum field that has a `*_raw` companion, the pair forms a single
two-field discriminator. There is no third sentinel enum value
(deliberately — see "Why no `*_UNKNOWN` sentinel" below). Consumers
distinguish the three valid states by inspecting both fields:


| Enum tag            | `*_raw` value    | Meaning                                              | Producer must                                                 |
| ------------------- | ---------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| `*_UNSPECIFIED` (0) | `""` (empty)     | Field was not set on the source.                     | Leave both unset.                                             |
| `*_UNSPECIFIED` (0) | non-empty string | Source had a value the converter does not recognize. | Set tag to 0 *and* populate `*_raw` with the original string. |
| any value `> 0`     | `""` (empty)     | Recognized value.                                    | Set tag only; do not populate `*_raw`.                        |


Producers must never set both a non-zero enum tag *and* a non-empty
`*_raw`. Consumers should treat that combination as a producer bug; if
they need to be defensive, the enum tag wins.

### Why no `*_UNKNOWN` Sentinel

We considered adding a named `*_UNKNOWN` enum value to mark the
"received but not recognized" case explicitly. We deliberately did not.
Rationale:

- Protobuf does not have this idiom. `*_UNSPECIFIED` at tag 0 plus a
companion fallback string is the conventional pattern (used by
Envoy, the gRPC ecosystem, and others). A second sentinel would be
a project-specific convention every client has to learn.
- It would expand every generated language's enum surface
(`DocItemLabel.UNKNOWN`, `CodeLanguageLabel.UNKNOWN`, …), forcing
every exhaustive switch in client code to add a new case for an
abstract concept rather than a real new value.
- It would create a request-side foot-gun: a client could send
`*_UNKNOWN` as input, which has no defined meaning.
- The two-field discriminator above already gives consumers complete,
unambiguous information. The sentinel only saves typing, not
semantics.

Forward compatibility of *new* enum values added by upstream is handled
by the `*_raw` companion when the producer is older than the source
schema, and by protobuf's runtime "unknown enum value" handling
(Python's `UnknownFieldSet`, Go's raw-int passthrough, etc.) when the
consumer is older than the producer. The `*_raw` companion is not a
substitute for either; it is the human-readable carry-along that makes
the unrecognized case observable in logs and clients.

## Enforcement

- Conversion logic: `docling_core/utils/conversion.py`
- Startup validation (serve): `docling_serve/grpc/schema_validator.py`
- Tests:
  - `docling-core/test/test_proto_conversion.py`
  - `docling-serve/tests/test_schema_validator.py`

## Developer Workflow

When changing the model or IDL:

1. Update protobuf + converter for logical parity.
2. Regenerate stubs with `scripts/gen_proto.py`.
3. Update validator rules only for intentional, documented exceptions.
4. Update tests in `test/test_proto_conversion.py`.
5. Keep this file current if an intentional difference is added/removed.

## Keeping In Sync With Upstream `main`

The full procedure for bringing both `docling-core` and `docling-serve`
gRPC branches up to date with their respective `main` branches —
including how to detect and fix Pydantic↔proto drift mechanically — is
documented in the serve repo at `docs/grpc/upstream_sync_procedure.md`.

The short version for this repo:

1. `git fetch upstream && git merge upstream/main` on the gRPC branch.
2. `uv run python scripts/gen_proto.py` to regenerate stubs.
3. `uv run pytest test/test_proto_conversion.py -q` — must stay green.
4. Run the serve-side schema validator (see the serve procedure doc).
  That is what surfaces new Pydantic fields requiring a proto mirror.
5. For each new Pydantic field the validator reports, add a proto
  field, a converter helper, and a test, following the patterns
   already in `conversion.py` and `test_proto_conversion.py`.
6. For each new enum value, mirror it into the proto enum and the
  conversion map. Old proto enum tags are append-only — never renumber
   or remove tags, even if the upstream Pydantic enum is renamed.