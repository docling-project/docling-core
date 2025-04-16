"""Define base classes for visualization."""

from abc import ABC, abstractmethod
from typing import Optional

from PIL.Image import Image
from pydantic import BaseModel

from docling_core.types.doc import DoclingDocument


class BaseVisualizer(BaseModel, ABC):
    """Visualize base class."""

    @abstractmethod
    def get_visualization(
        self,
        *,
        doc: DoclingDocument,
        **kwargs,
    ) -> dict[Optional[int], Image]:
        """Get visualization of the document as images by page."""
        raise NotImplementedError()
