#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define models for labeling Q&A pairs."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from docling_core.search.mapping import es_field

QAScopeLabel = Literal["corpus", "document", "out_of_scope"]
QAAlignmentLabel = Literal["aligned", "tangential", "misaligned"]
QACorrectnessLabel = Literal["entailed", "not_entailed"]
QACompletenessLabel = Literal["complete", "incomplete"]
QAInformationLabel = Literal[
    "fact_single",
    "fact_multi",
    "summary",
    "reasoning",
    "choice",
    "procedure",
    "opinion",
    "feedback",
]


class QALabelling(BaseModel, extra="forbid"):
    """Subclass to classify QA pair."""

    scope: Optional[QAScopeLabel] = Field(
        default=None,
        description="""Enumeration of QA scope types based on question only.
            - Corpus: question is asked on the entire corpus
                > Example: "What is the operating temperature of device X?"
            - Document: need to know the precise document before answering the question
                > Example: "What is its operating temperature?"
            - Out of scope: question is out of scope for the system
                > Example: "What is the volume of moon?" """,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    alignment: Optional[QAAlignmentLabel] = Field(
        default=None,
        description="""Enumeration of QA alignment types based on question-context pair.
            Given the following context: "Device X works between 2 and 20 degrees C"
            A question can be:
            - Aligned: the context has information that the question seeks
                > Example: "Can device X work at 10 degrees?"
            - Tangential: the context does not have the information directly
                            but the question is related to the context
                > Example: "Is device X safe?"
            - Misaligned: the question has nothing to do with the context
                > Example: "Why is device Y not working?" """,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    correctness: Optional[QACorrectnessLabel] = Field(
        default=None,
        description="""Enumeration of QA correctness types based on
            question-answer-context triplet.
            Given the following context: "Device X works between 2 and 20 degrees C"
            and the following question: "Can device X work at 10 degrees?"
            An answer can be:
            - Entailed: answer is entailed to both question and context
                > Example: "Yes, as it works between 2 and 20 degrees."
            - Not entailed: answer is not entailed to either question or context
                > Example: "Yes, device X can work at any temperature." """,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    completeness: Optional[QACompletenessLabel] = Field(
        default=None,
        description="""Enumeration of QA completeness types based on
            question-answer-context triplet.
            Given the following context: "A, B, C, and D met on Friday."
            and the following question: "Who was in the meeting?"
            An answer can be:
            - Complete: Answer contains all relevant information requested by a
                question that can be extracted from the associated ground-truth context
                > Example: "A, B, C, and D."
            - Incomplete: Answer does not contain the entire relevant information in
                the context
                > Example: "B and D" """,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    information: Optional[QAInformationLabel] = Field(
        default=None,
        description="""Enumeration of QA nature of information types based on question
            only.
            - Single fact: Answer should be a short phrase containing a numerical or
                textual fact
                > Example: "What is the boiling point of water?"
            - Multiple fact: Answer is a list of two or more facts (not necessarily in
                list format)
                > Example: "What is the minimum and maximum age of people working at
                IBM?"
            - Summary: Answer summarises a part of the context without any modification.
                > Example: "Briefly describe the temperature requirements for this
                device in a table"
            - Reasoning: Answer requires inferring information from the context that
                can be inferred but is not explicitly stated (e.g., operating
                temperature is given and the question asks if the device can operate at
                a particular temperature)
                > Example: "Why can I not operate this device under water?"
            - Multiple choice: Question provides a few choices implicitly or explicitly
                and the answer must be one of these choices. Includes yes/no questions
                > Example: "If I operate this device at 10 degrees, will it be in the
                green range or red?"
            - Procedure: Answer outlines the steps to do something. As opposed to a
                summary, the order of information matters here
                > Example: "How can I access part X of device Y?"
            - Opinion: The context provides several viewpoints and the question
                requests the opinion of the chatbot
                > Example: "Is device X better than Y?"
            - Feedback: The question is actually a feedback on the preceding generation
                within a session
                > Example: "Your summary was inadequate" """,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
