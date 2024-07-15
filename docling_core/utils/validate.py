#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Validation of Document-related files against their data schemas."""
import argparse
import json
import logging

from docling_core.utils.validators import (
    validate_ann_schema,
    validate_ocr_schema,
    validate_raw_schema,
)

logger = logging.getLogger("docling-core")


def parse_arguments():
    """Parse the arguments from the command line."""
    argparser = argparse.ArgumentParser(description="validate example-file with schema")

    argparser.add_argument(
        "-f", "--format", required=True, help="format of the file [RAW, ANN, OCR]"
    )

    argparser.add_argument(
        "-i", "--input-file", required=True, help="JSON filename to be validated"
    )

    pargs = argparser.parse_args()

    return pargs.format, pargs.input_file


def run():
    """Run the validation of a file containing a Document."""
    file_format, input_file = parse_arguments()

    with open(input_file, "r") as fd:
        file_ = json.load(fd)

    result = (False, "Empty result")

    if file_format == "RAW":
        result = validate_raw_schema(file_)

    elif file_format == "ANN":
        result = validate_ann_schema(file_)

    elif file_format == "OCR":
        result = validate_ocr_schema(file_)

    else:
        logger.error("format of the file needs to `RAW`, `ANN` or `OCR`")

    if result[0]:
        logger.info("Done!")
    else:
        logger.error("invalid schema: {}".format(result[1]))


def main():
    """Set up the environment and run the validation of a Document."""
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    # logger.addHandler(ch)

    logging.basicConfig(handlers=[ch])
    run()


if __name__ == "__main__":
    main()
