import fnmatch
import glob
import os
from typing import List

import git

from docling_core.types.doc import DoclingDocument, DocumentOrigin
from docling_core.types.doc.labels import CodeLanguageLabel
from docling_core.utils.legacy import _create_hash

language_to_extension = {
    "Python": ".py",
    "Java": ".java",
    "C": ".c",
    "TypeScript": ".ts",
    "JavaScript": ".js",
}

language_to_enum = {
    "Python": CodeLanguageLabel.PYTHON,
    "Java": CodeLanguageLabel.JAVA,
    "C": CodeLanguageLabel.C,
    "TypeScript": CodeLanguageLabel.TYPESCRIPT,
    "JavaScript": CodeLanguageLabel.JAVASCRIPT,
}


def get_latest_commit_id(file_dir: str) -> str:
    """
    Returns the hexadecimal SHA-1 ID of the latest commit in the given Git repository directory.

    Parameters:
    file_dir (str): The path to the Git repository directory.

    Returns:
    str: The hexadecimal SHA-1 ID of the latest commit, or an empty string if an error occurs.
    """
    try:
        repo = git.Repo(file_dir, search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception:
        return ""


def load_ignore_patterns(ignore_file_path: str) -> list:
    """
    Load ignore patterns from a file.

    This function reads a file containing ignore patterns (one per line) and returns a list of patterns,
    excluding empty lines and lines starting with '#'. If the file does not exist, it returns an empty list.

    Args:
        ignore_file_path (str): The path to the ignore file.

    Returns:
        list: A list of ignore patterns.
    """
    if not os.path.exists(ignore_file_path):
        return []
    with open(ignore_file_path, "r", encoding="utf-8") as file:
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


def is_ignored(file_path: str, ignore_patterns: List[str]) -> bool:
    """
    Check if a file path matches any of the given ignore patterns.

    This function takes a file path and a list of ignore patterns, and returns True if the file path matches any of the patterns,
    indicating that the file should be ignored. Otherwise, it returns False.

    Args:
        file_path (str): The path of the file to check.
        ignore_patterns (list of str): A list of patterns to check against the file path.

    Returns:
        bool: True if the file path matches any ignore pattern, False otherwise.
    """
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def create_ds(
    file_dir: str, repo_url: str, commit_id: str = None
) -> List[DoclingDocument]:
    """
    Build DoclingDocument objects from a local checkout, one per code file.
    Deterministic ordering and hashes for use in tests.

    Args:
        file_dir: Directory containing the repository
        repo_url: URL of the repository
        commit_id: Specific commit ID to use (defaults to "main" for deterministic testing)
    """
    documents: List[DoclingDocument] = []
    if commit_id is None:
        commit_id = get_latest_commit_id(file_dir)
    ignore_file = os.path.join(file_dir, ".ragignore")
    ignore_patterns = load_ignore_patterns(ignore_file)

    for language, extension in language_to_extension.items():
        files = [
            f
            for f in sorted(glob.glob(f"{file_dir}/**/*{extension}", recursive=True))
            if not is_ignored(f, ignore_patterns)
        ]
        files.sort()
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                file_relative = os.path.relpath(file_path, start=file_dir).replace(
                    "\\", "/"
                )

                origin = DocumentOrigin(
                    filename=file_relative,
                    uri=(
                        f"{repo_url}/blob/{commit_id}/{file_relative}"
                        if commit_id
                        else f"{repo_url}/{file_relative}"
                    ),
                    mimetype="text/plain",
                    binary_hash=_create_hash(file_content),
                )

                doc = DoclingDocument(name=file_relative, origin=origin)
                doc.add_code(
                    text=file_content, code_language=language_to_enum[language]
                )
                documents.append(doc)
            except Exception:
                continue

    return documents
