import os
import json
import glob
import argparse
from pathlib import Path
from typing import Sequence, Dict, Any, Optional
from collections import Counter
from io import BytesIO

from PIL import Image as PILImage

from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from docling_core.types.doc import DoclingDocument, ImageRef
from docling_core.types.doc.base import ImageRefMode
from docling_core.experimental.idoctags import (
    IDocTagsSerializationMode,
    IDocTagsParams,
    IDocTagsVocabulary,
    IDocTagsDocSerializer,
)

import matplotlib.pyplot as plt
import numpy as np

# In order to download **before** the datasets library, run
# 
# HF_HUB_DISABLE_XET=1 hf download --repo-type dataset "docling-project/canva-set-a"
#

def update_tokenizer(tokenizer: PreTrainedTokenizerBase, verbose: bool = False) -> PreTrainedTokenizerBase:
    """Extend tokenizer with IDocTags special tokens.

    Parameters
    - tokenizer: base tokenizer to extend
    - verbose: print added tokens if True
    """
    special_tokens = IDocTagsVocabulary.get_special_tokens()
    if verbose:
        for i, tok in enumerate(special_tokens):
            print(i, "\t", tok)
    tokenizer.add_tokens(special_tokens)
    if verbose:
        print(f"New vocab size: {tokenizer.vocab_size}")
    return tokenizer

def run_dump(cfg: Dict[str, Any]) -> int:
    """Dump/serialize documents from a dataset to IDocTags strings/files.

    Config keys (with defaults):
    - dataset_name: str ("docling-project/doclaynet-set-a")
    - dataset_subset: str ("pdf_train")
    - dataset_split: str ("train")
    - output_dir: str ("./scratch/idoctags")
    - failed_dir: str ("./scratch/idoctags_failed") — where to dump HTML+JSON when serialization fails
    - write_outputs: bool (True) — write serialized outputs if True
    - variants: list of serialization variants; each item:
        {"add_content": bool, "mode": "LLM_FRIENDLY"|"HUMAN_FRIENDLY", "suffix": str}
      Defaults to three variants mirroring prior behavior.
    """
    dataset_name = cfg.get("dataset_name", "docling-project/doclaynet-set-a")
    dataset_subset = cfg.get("dataset_subset", "train")
    dataset_split = cfg.get("dataset_split", "train")
    output_dir = Path(cfg.get("output_dir", "./scratch/idoctags"))
    failed_dir = Path(cfg.get("failed_dir", "./scratch/idoctags_failed"))
    pngs_dir = Path(cfg.get("pngs_dir", "./scratch/pngs_dir"))
    write_outputs: bool = bool(cfg.get("write_outputs", True))

    default_variants = [
        {"add_content": False, "mode": "LLM_FRIENDLY", "suffix": "_without"},
        {"add_content": True, "mode": "LLM_FRIENDLY", "suffix": "_with"},
        {"add_content": True, "mode": "HUMAN_FRIENDLY", "suffix": "_with_h"},
    ]
    variants = cfg.get("variants", default_variants)

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create failed dir proactively; also created again on demand in the exception path
        failed_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, dataset_subset)
    split = ds[dataset_split]
    total = len(split)

    errors: list[str] = []
    ok = 0

    for idx, row in tqdm(enumerate(split), total=total, ncols=128):
        text = row.get("GroundTruthDocument", "")
        try:
            doc = DoclingDocument.model_validate_json(text)
            page_images = [
                __ for __ in row["GroundTruthPageImages"]
            ]
            # page_images[0].show()

        except Exception as exc:
            errors.append(
                f"Parse error: {exc} for {dataset_name}/{dataset_subset}/{dataset_split} idx={idx}"
            )
            continue

        for i, __ in enumerate(page_images, start=1):            
            doc.pages[i].image = ImageRef.from_pil(__, dpi=140)
            # png_path = pngs_dir / f"{idx}_{i}.png"
            # __.save(png_path)
            
        try:
            for var in variants:
                params = IDocTagsParams()
                params.add_content = bool(var.get("add_content", True))
                mode_str = str(var.get("mode", "LLM_FRIENDLY"))
                params.mode = (
                    IDocTagsSerializationMode.LLM_FRIENDLY
                    if mode_str == "LLM_FRIENDLY"
                    else IDocTagsSerializationMode.HUMAN_FRIENDLY
                )

                iser = IDocTagsDocSerializer(doc=doc, params=params)
                serialized = iser.serialize().text

                if write_outputs:
                    suffix = var.get("suffix", "")
                    out_path = output_dir / f"{idx}{suffix}.idoctags"
                    with open(out_path, "w", encoding="utf-8") as fw:
                        fw.write(serialized)
            ok += 1
        except Exception as exc:
            print(f"exc: {exc}")
            # page_images[0].show()
            
            failed_dir.mkdir(parents=True, exist_ok=True)
            # JSON dump of the parsed DoclingDocument
            json_path = failed_dir / f"{idx}.json"
            print(f"\n\n -> writing {json_path}")
            with open(json_path, "w", encoding="utf-8") as fj:
                fj.write(doc.model_dump_json(indent=2))

            try:
                # Split-page HTML with layout/images; avoid single-column variant
                html_path = failed_dir / f"{idx}_split.html"
                print(f"\n\n -> writing {html_path}")
                doc.save_as_html(
                    filename=html_path,
                    image_mode=ImageRefMode.EMBEDDED,
                    split_page_view=True,
                    include_annotations=True,
                )
            except Exception as exc2:
                print(f"exception: {exc2}")
            
    print(f"Serialized OK: {ok} / {total}")
    if errors:
        print("Errors:")
        for e in errors:
            print(" -", e)
    return 0 if ok > 0 and not errors else (0 if ok == total else 1)


def run_analyse(cfg: Dict[str, Any]) -> int:
    """Analyse token lengths and special-token usage from IDocTags files.

    Config keys (with defaults):
    - tokenizer_name: str ("ibm-granite/granite-docling-258M")
    - input_glob: str ("./scratch/idoctags/*_with_h.idoctags")
    - pair_replace: dict with {"from": "_h", "to": ""} to locate paired files (optional)
    - show_plots: bool (True)
    - verbose: bool (False)
    """
    tokenizer_name = cfg.get("tokenizer_name", "ibm-granite/granite-docling-258M")
    input_glob_pat = cfg.get("input_glob", "./scratch/idoctags/*_with_h.idoctags")
    pair_replace = cfg.get("pair_replace", {"from": "_h", "to": ""})
    show_plots = bool(cfg.get("show_plots", True))
    verbose = bool(cfg.get("verbose", False))

    base_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
    ext_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
    ext_tokenizer = update_tokenizer(tokenizer=ext_tokenizer, verbose=verbose)

    filenames = sorted(glob.glob(input_glob_pat))
    if not filenames:
        print(f"No files matched pattern: {input_glob_pat}")
        return 1

    base_lengths: list[int] = []
    ext_lengths: list[int] = []
    special_counts: Counter[int] = Counter()

    # Map special tokens to IDs in the extended tokenizer
    special_tokens = IDocTagsVocabulary.get_special_tokens()
    special_token_ids = {ext_tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens}

    for filename in tqdm(filenames, desc="Analyse", ncols=128):
        with open(filename, "r", encoding="utf-8") as fr:
            text_h = fr.read()

        paired_name: Optional[str] = None
        if isinstance(pair_replace, dict) and pair_replace.get("from") is not None:
            paired_name = filename.replace(str(pair_replace.get("from")), str(pair_replace.get("to", "")))

        text_plain = None
        if paired_name and os.path.exists(paired_name):
            with open(paired_name, "r", encoding="utf-8") as fr:
                text_plain = fr.read()

        # Tokenize and collect lengths
        base_tokens_h = base_tokenizer.encode(text_h, add_special_tokens=True)
        ext_tokens_h = ext_tokenizer.encode(text_h, add_special_tokens=True)
        base_lengths.append(len(base_tokens_h))
        ext_lengths.append(len(ext_tokens_h))

        # Special token usage on the extended-tokenized sequence of the HF/H text
        for tid in ext_tokens_h:
            if tid in special_token_ids:
                special_counts[tid] += 1

        # Optionally also include the paired non-human-friendly text lengths
        if text_plain is not None:
            _ = base_tokenizer.encode(text_plain, add_special_tokens=True)
            _e = ext_tokenizer.encode(text_plain, add_special_tokens=True)
            base_lengths.append(len(_))
            ext_lengths.append(len(_e))
            for tid in _e:
                if tid in special_token_ids:
                    special_counts[tid] += 1

    # Report summary
    print(f"Files analysed: {len(filenames)}")
    print(f"Sequences tokenized (including pairs when available): {len(ext_lengths)}")
    if ext_lengths:
        arr = np.asarray(ext_lengths)
        print(
            f"Extended tokenizer lengths — min: {arr.min()}, p50: {np.median(arr)}, p95: {np.percentile(arr,95)}, max: {arr.max()}, mean: {arr.mean():.1f}"
        )

    # Map special IDs back to tokens and show top-k
    if special_counts:
        id_to_token = {i: ext_tokenizer.convert_ids_to_tokens([i])[0] for i in special_counts.keys()}
        total_special = sum(special_counts.values())
        print("Special token usage (top 30):")
        for tid, cnt in special_counts.most_common(30):
            tok = id_to_token.get(tid, str(tid))
            pct = 100.0 * cnt / total_special if total_special else 0.0
            print(f" - {tok}: {cnt} ({pct:.2f}%)")

        if show_plots:
            # Bar chart of top-N special tokens
            items = special_counts.most_common(30)
            labels = [id_to_token.get(tid, str(tid)) for tid, _ in items]
            values = [v for _, v in items]
            plt.figure(figsize=(12, 5))
            plt.bar(labels, values, color="#4C78A8")
            plt.xticks(rotation=90)
            plt.title("Top Special Tokens (by frequency)")
            plt.tight_layout()
            plt.show()

    # Visualizations for lengths
    if show_plots and base_lengths and ext_lengths:
        plot_token_histograms(base_lengths, ext_lengths)
        plot_token_scatter_with_regression(base_lengths, ext_lengths)

    return 0


def plot_token_histograms(
    original_num_tokens: Sequence[int] | np.ndarray,
    optimal_num_tokens: Sequence[int] | np.ndarray,
    *,
    bins: int = 50,
    density: bool = False,
) -> None:
    """Plot overlapping histograms for original and optimal token counts.

    Parameters
    - original_num_tokens: sequence of counts before optimization
    - optimal_num_tokens: sequence of counts after optimization
    - bins: number of histogram bins
    - density: normalize histograms if True
    """
    x = np.asarray(original_num_tokens)
    y = np.asarray(optimal_num_tokens)

    plt.figure(figsize=(10, 6))
    plt.hist(
        x,
        bins=bins,
        alpha=0.5,
        label="Original",
        color="#4C78A8",
        edgecolor="black",
        density=density,
    )
    plt.hist(
        y,
        bins=bins,
        alpha=0.5,
        label="Optimal",
        color="#F58518",
        edgecolor="black",
        density=density,
    )
    plt.xlabel("Number of tokens")
    plt.ylabel("Density" if density else "Frequency")
    plt.title("Token Count Distributions")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_token_scatter_with_regression(
    original_num_tokens: Sequence[int] | np.ndarray,
    optimal_num_tokens: Sequence[int] | np.ndarray,
) -> None:
    """Scatter plot of original vs optimal tokens with a linear regression line.

    Parameters
    - original_num_tokens: x-values
    - optimal_num_tokens: y-values
    """
    x = np.asarray(original_num_tokens, dtype=float)
    y = np.asarray(optimal_num_tokens, dtype=float)

    if x.size == 0 or y.size == 0:
        print("No data to plot.")
        return

    # Linear regression with numpy polyfit (degree 1)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    # Compute R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Prepare line for full x-range
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.6, color="#4C78A8", label="Documents")
    plt.plot(x_line, y_line, color="#E45756", linewidth=2,
             label=f"y = {slope:.3f}x + {intercept:.3f} (R²={r2:.3f})")
    plt.xlabel("Original tokens")
    plt.ylabel("Optimal tokens")
    plt.title("Original vs Optimal Tokens with Linear Fit")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def default_config(mode: str) -> Dict[str, Any]:
    if mode == "dump":
        return {
            "dataset_name": "docling-project/doclaynet-set-a",
            "dataset_subset": "pdf_train",
            "dataset_split": "train",
            "output_dir": "./scratch/idoctags",
            "failed_dir": "./scratch/idoctags_failed",
            "write_outputs": True,
            "variants": [
                {"add_content": False, "mode": "LLM_FRIENDLY", "suffix": "_without"},
                {"add_content": True, "mode": "LLM_FRIENDLY", "suffix": "_with"},
                {"add_content": True, "mode": "HUMAN_FRIENDLY", "suffix": "_with_h"},
            ],
        }
    elif mode == "analyse":
        return {
            "tokenizer_name": "ibm-granite/granite-docling-258M",
            "input_glob": "./scratch/idoctags/*_with_h.idoctags",
            "pair_replace": {"from": "_h", "to": ""},
            "show_plots": True,
            "verbose": False,
        }
    else:
        raise ValueError(f"Unknown mode for default config: {mode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert and analyse IDocTags data")
    parser.add_argument(
        "--mode",
        choices=["dump", "analyse"],
        required=True,
        help="Mode: dump dataset to idoctags or analyse token stats",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config. If omitted, a default config is created for the chosen mode.",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Only write the default config for the selected mode and exit.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path: Optional[Path] = args.config
    cfg: Dict[str, Any]

    if cfg_path is None:
        cfg = default_config(args.mode)
        out_name = Path(f"idoctags_{args.mode}_config.json")
        with open(out_name, "w", encoding="utf-8") as fw:
            json.dump(cfg, fw, indent=2)
        print(f"Wrote default config to: {out_name.resolve()}")
        if args.write_default_config:
            return 0
        # proceed using the freshly created config in-memory
    else:
        with open(cfg_path, "r", encoding="utf-8") as fr:
            cfg = json.load(fr)

    if args.mode == "dump":
        return run_dump(cfg)
    elif args.mode == "analyse":
        return run_analyse(cfg)
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
