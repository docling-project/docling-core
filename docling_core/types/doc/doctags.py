"""DocTags-format page/document container models."""

import typing
from pathlib import Path
from typing import Optional, Union

from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict

from docling_core.types.doc.tokens import DocumentToken


class DocTagsPage(BaseModel):
    """DocTagsPage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: str
    image: Optional[PILImage.Image] = None


class DocTagsDocument(BaseModel):
    """DocTagsDocument."""

    pages: list[DocTagsPage] = []

    @classmethod
    def from_doctags_and_image_pairs(
        cls,
        doctags: typing.Sequence[Union[Path, str]],
        images: Optional[list[Union[Path, PILImage.Image]]],
    ):
        """from_doctags_and_image_pairs."""
        if images is not None and len(doctags) != len(images):
            raise ValueError("Number of page doctags must be equal to page images!")
        doctags_doc = cls()

        pages = []

        for ix, dt in enumerate(doctags):
            if isinstance(dt, Path):
                with dt.open("r") as fp:
                    dt = fp.read()
            elif isinstance(dt, str):
                pass

            img = None
            if images is not None:
                img = images[ix]

                if isinstance(img, Path):
                    img = PILImage.open(img)
                elif isinstance(img, PILImage.Image):
                    pass

            page = DocTagsPage(tokens=dt, image=img)
            pages.append(page)

        doctags_doc.pages = pages
        return doctags_doc

    @classmethod
    def from_multipage_doctags_and_images(
        cls,
        doctags: Union[Path, str],
        images: Optional[list[Union[Path, PILImage.Image]]],
    ):
        """From doctags with `<page_break>` and corresponding list of page images."""
        if isinstance(doctags, Path):
            with doctags.open("r") as fp:
                doctags = fp.read()
        dt_list = (
            doctags.removeprefix(f"<{DocumentToken.DOCUMENT.value}>")
            .removesuffix(f"</{DocumentToken.DOCUMENT.value}>")
            .split(f"<{DocumentToken.PAGE_BREAK.value}>")
        )
        dt_list = [el.strip() for el in dt_list]

        return cls.from_doctags_and_image_pairs(dt_list, images)
