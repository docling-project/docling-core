"""Package for models defined by the Document type."""

from docling_core.types.doc.base import (
    BoundingBox,
    CoordOrigin,
    ImageRefMode,
    PydanticSerCtxKey,
    Size,
)
from docling_core.types.doc.common.annotations import (
    BaseAnnotation,
    DescriptionAnnotation,
    MiscAnnotation,
)
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.common.formatting import (
    Formatting,
    Script,
)
from docling_core.types.doc.common.meta import (
    BaseMeta,
    BasePrediction,
    CodeMetaField,
    DescriptionMetaField,
    EntitiesMetaField,
    EntityMention,
    FloatingMeta,
    KeywordsMetaField,
    LanguageMetaField,
    MetaFieldName,
    MetaUtils,
    MoleculeMetaField,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    SummaryMetaField,
    TabularChartMetaField,
    TopicsMetaField,
)
from docling_core.types.doc.common.origin import (
    BaseSource,
    DocumentOrigin,
    SourceType,
    TrackSource,
)
from docling_core.types.doc.common.page_item import PageItem
from docling_core.types.doc.common.reference import (
    FineRef,
    ImageRef,
    ProvenanceItem,
    RefItem,
)
from docling_core.types.doc.common.scalars import (
    CharSpan,
    LevelNumber,
    Uint64,
)
from docling_core.types.doc.doctags import (
    DocTagsDocument,
    DocTagsPage,
)
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.code import CodeItem
from docling_core.types.doc.items.content import ContentItem
from docling_core.types.doc.items.form import (
    FieldHeadingItem,
    FieldItem,
    FieldRegionItem,
    FieldValueItem,
)
from docling_core.types.doc.items.group import (
    GroupItem,
    InlineGroup,
    ListGroup,
    OrderedList,
    UnorderedList,
)
from docling_core.types.doc.items.key_value import (
    FormItem,
    GraphCell,
    GraphData,
    GraphLink,
    KeyValueItem,
)
from docling_core.types.doc.items.node import (
    DocItem,
    FloatingItem,
    NodeItem,
)
from docling_core.types.doc.items.picture.charts import (
    ChartBar,
    ChartLine,
    ChartPoint,
    ChartSlice,
    ChartStackedBar,
    PictureBarChartData,
    PictureChartData,
    PictureLineChartData,
    PicturePieChartData,
    PictureScatterChartData,
    PictureStackedBarChartData,
    PictureTabularChartData,
)
from docling_core.types.doc.items.picture.classification import (
    PictureClassificationClass,
    PictureClassificationData,
)
from docling_core.types.doc.items.picture.molecule import PictureMoleculeData
from docling_core.types.doc.items.picture.picture import (
    BasePictureData,
    PictureDataType,
    PictureDescriptionData,
    PictureItem,
    PictureMiscData,
)
from docling_core.types.doc.items.table.table import (
    TableAnnotationType,
    TableItem,
)
from docling_core.types.doc.items.table.table_data import (
    AnyTableCell,
    Orientation,
    RichTableCell,
    TableCell,
    TableData,
)
from docling_core.types.doc.items.text import (
    FormulaItem,
    ListItem,
    SectionHeaderItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    GraphCellLabel,
    GraphLinkLabel,
    GroupLabel,
    HumanLanguageLabel,
    PictureClassificationLabel,
    TableCellLabel,
)
from docling_core.types.doc.page import (
    BitmapResource,
    BoundingRectangle,
    ColorChannelValue,
    ColorMixin,
    ColorRGBA,
    Coord2D,
    OrderedElement,
    PageGeometry,
    PageNumber,
    ParsedPdfDocument,
    PdfCellRenderingMode,
    PdfHyperlink,
    PdfLine,
    PdfMetaData,
    PdfPageBoundaryType,
    PdfPageGeometry,
    PdfShape,
    PdfTableOfContents,
    PdfTextCell,
    PdfWidget,
    SegmentedPage,
    SegmentedPdfPage,
    TextCell,
    TextCellUnit,
    TextDirection,
)
from docling_core.types.doc.tokens import (
    DocumentToken,
    TableToken,
)
from docling_core.types.doc.webvtt import (
    WebVTTCueBlock,
    WebVTTCueBoldSpan,
    WebVTTCueClassSpan,
    WebVTTCueComponent,
    WebVTTCueComponentBase,
    WebVTTCueComponentWithTerminator,
    WebVTTCueIdentifier,
    WebVTTCueInternalText,
    WebVTTCueItalicSpan,
    WebVTTCueLanguageSpan,
    WebVTTCueLanguageSpanStartTag,
    WebVTTCueSpanStartTag,
    WebVTTCueSpanStartTagAnnotated,
    WebVTTCueTextSpan,
    WebVTTCueTimings,
    WebVTTCueUnderlineSpan,
    WebVTTCueVoiceSpan,
    WebVTTFile,
    WebVTTLineTerminator,
    WebVTTTimestamp,
)
