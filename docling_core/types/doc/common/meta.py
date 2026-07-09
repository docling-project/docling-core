"""Metadata models attached to document nodes."""

from enum import Enum
from typing import Annotated, Any, Final, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from docling_core.types.base import UniqueList
from docling_core.types.doc.base import PydanticSerCtxKey, round_pydantic_float
from docling_core.types.doc.common.scalars import CharSpan
from docling_core.types.doc.labels import CodeLanguageLabel, HumanLanguageLabel
from docling_core.utils.validators import ensure_unique_list


class _ExtraAllowingModel(BaseModel):
    """Base model allowing extra fields."""

    model_config = ConfigDict(extra="allow")

    def get_custom_part(self) -> dict[str, Any]:
        """Get the extra fields as a dictionary."""
        return self.__pydantic_extra__ or {}

    def _copy_without_extra(self) -> Self:
        """Create a copy without the extra fields."""
        return self.model_validate(self.model_dump(exclude=set(self.get_custom_part())))

    def _check_custom_field_format(self, key: str) -> None:
        parts = key.split(MetaUtils._META_FIELD_NAMESPACE_DELIMITER, maxsplit=1)
        if len(parts) != 2 or (not parts[0]) or (not parts[1]):
            raise ValueError(
                f"Custom meta field name must be in format 'namespace__field_name' (e.g. 'my_corp__max_size'): {key}"
            )

    @model_validator(mode="after")
    def _validate_field_names(self) -> Self:
        extra_dict = self.get_custom_part()
        for key in self.model_dump():
            if key in extra_dict:
                self._check_custom_field_format(key=key)
            elif MetaUtils._META_FIELD_NAMESPACE_DELIMITER in key:
                raise ValueError(f"Standard meta field name must not contain '__': {key}")

        return self

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self.get_custom_part():
            self._check_custom_field_format(key=name)

    def set_custom_field(self, namespace: str, name: str, value: Any) -> str:
        """Set a custom field and return the key."""
        key = MetaUtils.create_meta_field_name(namespace=namespace, name=name)
        setattr(self, key, value)
        return key


class BasePrediction(_ExtraAllowingModel):
    """Prediction field."""

    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="The confidence of the prediction.",
        examples=[0.9, 0.42],
    )
    created_by: Optional[str] = Field(
        default=None,
        description="The origin of the prediction.",
        examples=["ibm-granite/granite-docling-258M"],
    )

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)


class SummaryMetaField(BasePrediction):
    """Summary data."""

    text: str


class LanguageMetaField(BasePrediction):
    """Detected human language."""

    code: HumanLanguageLabel


class MetaFieldName(str, Enum):
    """Standard meta field names attached to document nodes.

    Note:
        These enum members must be kept in sync with the fields of the `BaseMeta` class or its subclasses.
    """

    SUMMARY = "summary"
    """A condensed natural-language summary of the content rooted at this node (e.g. a paragraph summary or section abstract)."""

    LANGUAGE = "language"
    """The detected human language of the node content, expressed as a BCP 47 code (e.g. ``"en"``, ``"de"``)."""

    ENTITIES = "entities"
    """Named entities extracted from the node text, such as persons, organisations, and locations."""

    KEYWORDS = "keywords"
    """Salient terms or short keyphrases that characterise the node content. Values are order-preserving and unique."""

    TOPICS = "topics"
    """Higher-level subject categories or thematic labels inferred for the node. Values are order-preserving and unique."""

    DESCRIPTION = "description"
    """A free-text description of the node, typically used for non-textual items such as figures and images."""

    CLASSIFICATION = "classification"
    """A classification label or category assigned to the node content (e.g. picture type, document genre)."""

    MOLECULE = "molecule"
    """Structured chemical / molecule data associated with the node."""

    TABULAR_CHART = "tabular_chart"
    """Tabular data extracted from a chart element."""


class EntityMention(BasePrediction):
    """Entity mention extracted from text."""

    text: Annotated[
        str,
        Field(description="Normalized text of the entity mention."),
    ]
    orig: Annotated[
        Optional[str],
        Field(
            description=(
                "Exact source text extracted from the original charspan, "
                "analogous to TextItem.orig. This may differ from 'text' when the "
                "mention has been normalized."
            )
        ),
    ] = None
    label: Annotated[
        Optional[str],
        Field(description="Entity type or category."),
    ] = None
    charspan: Annotated[
        Optional[CharSpan],
        Field(description="Character span (0-indexed) of the entity mention in the source text."),
    ] = None


class EntitiesMetaField(_ExtraAllowingModel):
    """Container for extracted entity mentions."""

    mentions: Annotated[list[EntityMention], Field(min_length=1)]


class KeywordsMetaField(_ExtraAllowingModel):
    """Container for a list of unique keywords / keyphrases."""

    values: Annotated[UniqueList[str], BeforeValidator(ensure_unique_list), Field(min_length=1)]


class TopicsMetaField(_ExtraAllowingModel):
    """Container for a list of unique topics / subjects."""

    values: Annotated[UniqueList[str], BeforeValidator(ensure_unique_list), Field(min_length=1)]


class BaseMeta(_ExtraAllowingModel):
    """Base class for metadata."""

    summary: Annotated[
        Optional[SummaryMetaField],
        Field(
            description="A condensed natural-language summary of the content rooted at this node.",
            examples=[{"text": "A short company/location statement."}],
        ),
    ] = None
    language: Annotated[
        Optional[LanguageMetaField],
        Field(
            description="The detected human language of the node content, expressed as a BCP 47 code.",
            examples=[{"code": "en"}],
        ),
    ] = None
    entities: Annotated[
        Optional[EntitiesMetaField],
        Field(
            description=(
                "Named entities extracted from the node text (persons, organisations, locations, etc.). "
                "Each mention carries the entity text, an optional type label, and an optional character span."
            ),
            examples=[{"mentions": [{"text": "IBM", "label": "ORG", "charspan": [0, 3]}]}],
        ),
    ] = None
    keywords: Annotated[
        Optional[KeywordsMetaField],
        Field(
            description=(
                "Salient terms or short keyphrases that characterise the node content. "
                "Keywords are more specific than topics and typically correspond to individual words or "
                "short multi-word expressions found in or closely related to the text. "
                "Values are order-preserving and deduplicated."
            ),
            examples=[{"values": ["transformer", "attention mechanism", "BERT"]}],
        ),
    ] = None
    topics: Annotated[
        Optional[TopicsMetaField],
        Field(
            description=(
                "Higher-level subject categories or thematic labels inferred for the node content. "
                "Topics are broader than keywords and describe the domain or theme rather than specific terms "
                "(e.g., 'machine learning' rather than 'gradient descent'). "
                "Values are order-preserving and deduplicated."
            ),
            examples=[{"values": ["natural language processing", "computer vision"]}],
        ),
    ] = None

    def has_content(self) -> bool:
        """Return True if this metadata contains any meaningful content."""
        return any(self._value_has_content(value) for value in self.model_dump(exclude_none=True).values())

    @staticmethod
    def _value_has_content(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, list):
            return any(BaseMeta._value_has_content(v) for v in value)
        if isinstance(value, dict):
            return any(BaseMeta._value_has_content(v) for v in value.values())
        if isinstance(value, BaseModel):
            return any(BaseMeta._value_has_content(v) for v in value.model_dump(exclude_none=True).values())
        return True


class DescriptionMetaField(BasePrediction):
    """Description metadata field."""

    text: str


class FloatingMeta(BaseMeta):
    """Metadata model for floating."""

    description: Optional[DescriptionMetaField] = None


class CodeMetaField(BasePrediction):
    """Code representation for the respective item."""

    text: str  # the actual code
    language: Optional[CodeLanguageLabel] = None


class MetaUtils:
    """Metadata-related utilities."""

    _META_FIELD_NAMESPACE_DELIMITER: Final = "__"
    _META_FIELD_LEGACY_NAMESPACE: Final = "docling_legacy"

    @classmethod
    def create_meta_field_name(
        cls,
        *,
        namespace: str,
        name: str,
    ) -> str:
        """Create a meta field name."""
        return f"{namespace}{cls._META_FIELD_NAMESPACE_DELIMITER}{name}"

    @classmethod
    def _create_migrated_meta_field_name(
        cls,
        *,
        name: str,
    ) -> str:
        return cls.create_meta_field_name(namespace=cls._META_FIELD_LEGACY_NAMESPACE, name=name)
