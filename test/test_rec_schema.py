#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module data_types.cxs."""
import glob
import unittest
from typing import Literal

import pytest
from pydantic import ValidationError

from docling_core.types.rec.attribute import Attribute
from docling_core.types.rec.predicate import Predicate
from docling_core.types.rec.record import Record
from docling_core.types.rec.statement import Statement
from docling_core.types.rec.subject import Subject


class TestCxsModel(unittest.TestCase):
    def test_predicates(self):
        """Validate data with Predicate schema."""
        for filename in glob.glob("test/data/rec/predicate-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                Predicate.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}", e.json())
                raise

    def test_predicates_wrong(self):
        filename = "test/data/rec/error-predicate-01.json"
        with (
            pytest.raises(ValidationError, match="invalid latitude"),
            open(filename, encoding="utf-8") as file_obj,
        ):
            file_json = file_obj.read()
            Predicate.model_validate_json(file_json)

        filename = "test/data/rec/error-predicate-02.json"
        with (
            pytest.raises(ValidationError, match="geopoint_value.conf"),
            open(filename, encoding="utf-8") as file_obj,
        ):
            file_json = file_obj.read()
            Predicate.model_validate_json(file_json)

    def test_attributes(self):
        """Validate data with Attribute schema."""
        for filename in glob.glob("test/data/rec/attribute-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                Attribute.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}:\n{e.json()}")
                raise

    def test_attributes_wrong(self):
        """Validate data with Attribute schema."""
        for filename in glob.glob("test/data/rec/error-attribute-*.json"):
            with (
                pytest.raises(ValidationError),
                open(filename, encoding="utf-8") as file_obj,
            ):
                file_json = file_obj.read()
                Attribute.model_validate_json(file_json)

    def test_subjects(self):
        """Validate data with Subject schema."""
        for filename in glob.glob("test/data/rec/subject-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                Subject.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}:\n{e.json()}")
                raise

    def test_subjects2(self):
        """Validate data with Subject schema."""
        # IdentifierTypeT, SubjectTypeT, SubjectNameTypeT
        subject = Subject[
            Literal["db"], Literal["material"], Literal["chemical_name", "sum_formula"]
        ]
        for filename in glob.glob("test/data/rec/subject-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                subject.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}:\n{e.json()}")
                raise

    def test_subjects_wrong(self):
        """Validate data with Subject schema."""
        # IdentifierTypeT, SubjectTypeT, SubjectNameTypeT
        subject = Subject[
            Literal["db_"], Literal["material"], Literal["chemical_name", "sum_formula"]
        ]
        for filename in glob.glob("test/data/rec/subject-*.json"):
            with (
                self.assertRaises(ValidationError),
                open(filename, encoding="utf-8") as file_obj,
            ):
                file_json = file_obj.read()
                subject.model_validate_json(file_json)
        subject = Subject[
            Literal["db"], Literal["material_"], Literal["chemical_name", "sum_formula"]
        ]
        for filename in glob.glob("test/data/rec/subject-*.json"):
            with (
                self.assertRaises(ValidationError),
                open(filename, encoding="utf-8") as file_obj,
            ):
                file_json = file_obj.read()
                subject.model_validate_json(file_json)
        subject = Subject[
            Literal["db"],
            Literal["material"],
            Literal["chemical_name_", "sum_formula_"],
        ]
        for filename in glob.glob("test/data/rec/subject-*.json"):
            with (
                self.assertRaises(ValidationError),
                open(filename, encoding="utf-8") as file_obj,
            ):
                file_json = file_obj.read()
                subject.model_validate_json(file_json)

    def test_statements(self):
        """Validate data with Statement schema."""
        for filename in glob.glob("test/data/rec/statement-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                Statement.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}:\n{e.json()}")
                raise

    def test_records(self):
        """Validate data with Record schema."""
        for filename in glob.glob("test/data/rec/record-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                Record.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}:\n{e.json()}")
                raise

    def test_records_2(self):
        """Validate data with Record schema by passing type parameters."""
        record = Record[
            Literal["db"],  # IdentifierTypeT,
            Literal["property-value"],  # PredicateValueTypeT
            Literal["Tc", "pressure"],  # PredicateKeyNameT
            Literal["property"],  # PredicateKeyTypeT
            Literal["sentence"],  # ProvenanceTypeT
            Literal["material"],  # SubjectTypeT
            Literal["chemical_name", "sum_formula"],  # SubjectNameTypeT
            Literal["DB", "Chemicals", "ChemDatabase"],  # CollectionNameTypeT
        ]
        for filename in glob.glob("test/data/rec/record-01.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                record.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}:\n{e.json()}")
                raise

    def test_records_3(self):
        """Validate data with Record schema by passing complex type parameters."""
        record = Record[
            Literal["arxivid", "cid", "cod", "doi", "db", "ent_id"],  # IdentifierTypeT
            Literal["property-value"],  # PredicateValueTypeT
            Literal[
                "space group",
                "cell symmetry",
                "cell length a",
                "cell length b",
                "cell length c",
                "cell angle alpha",
                "cell angle beta",
                "cell angle gamma",
                "molecular weight",
                "melting point",
                "boiling point",
                "density",
                "solubility",
                "temperature",
                "solvent",
                "km_value",
                "turnover_number",
                "ph_optimum",
                "temperature_optimum",
                "material-shape",
                "molecular",
                "material-state",
                "triangular lattice",
                "magnetic",
                "hexagonal",
                "multi layer",
                "pressure",
            ],  # PredicateKeyNameT
            Literal["property"],  # PredicateKeyTypeT
            Literal["url", "sentence"],  # ProvenanceTypeT
            Literal["material"],  # SubjectTypeT
            Literal[
                "chemical_name",
                "iupac_name",
                "sum_formula",
                "protein_name",
                "organism_name",
                "taxon",
                "enzyme_class",
            ],  # SubjectNameTypeT,
            Literal["DB", "Chemicals", "ChemDatabase"],  # CollectionNameTypeT
        ]
        try:
            filename = "test/data/rec/record-04.json"
            with open(filename, encoding="utf-8") as file_obj:
                file_json = file_obj.read()
            record.model_validate_json(file_json)
        except ValidationError as e:
            print(f"Validation error in file {filename}:\n{e.json()}")
            raise

    def test_records_wrong(self):
        """Validate data with Record schema."""
        record = Record[
            Literal["db"],  # IdentifierTypeT,
            Literal["property-value"],  # PredicateValueTypeT
            Literal["Tc", "pressure"],  # PredicateKeyNameT
            Literal["property"],  # PredicateKeyTypeT
            Literal["database"],  # ProvenanceTypeT
            Literal["material"],  # SubjectTypeT
            Literal["chemical_name", "sum_formula"],  # SubjectNameTypeT
            Literal["DB", "Chemicals", "ChemDatabase"],  # CollectionNameTypeT
        ]
        for filename in glob.glob("test/data/rec/record-01.json"):
            with (
                self.assertRaises(ValidationError),
                open(filename, encoding="utf-8") as file_obj,
            ):
                file_json = file_obj.read()
                record.model_validate_json(file_json)
        record = Record[
            Literal["db"],  # IdentifierTypeT,
            Literal["property-value_"],  # PredicateValueTypeT
            Literal["Tc", "pressure"],  # PredicateKeyNameT
            Literal["property"],  # PredicateKeyTypeT
            Literal["sentence"],  # ProvenanceTypeT
            Literal["material"],  # SubjectTypeT
            Literal["chemical_name", "sum_formula"],  # SubjectNameTypeT
            Literal["DB", "Chemicals", "ChemDatabase"],  # CollectionNameTypeT
        ]
        for filename in glob.glob("test/data/rec/record-01.json"):
            with (
                self.assertRaises(ValidationError),
                open(filename, encoding="utf-8") as file_obj,
            ):
                file_json = file_obj.read()
                record.model_validate_json(file_json)
