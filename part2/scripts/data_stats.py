#!/usr/bin/env python3
"""
Compute dataset statistics before/after preprocessing and emit LaTeX tables.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
import nltk
from nltk.tokenize import word_tokenize
from transformers import T5TokenizerFast

DATA_FIELDS = (
    ("num_examples", "Number of examples"),
    ("mean_nl_len", "Mean sentence length"),
    ("mean_sql_len", "Mean SQL query length"),
    ("nl_vocab", "Vocabulary size (natural language)"),
    ("sql_vocab", "Vocabulary size (SQL)"),
)

AFTER_FIELDS = (
    ("mean_encoder_len", "Mean encoder length"),
    ("mean_decoder_len", "Mean decoder input length"),
    ("mean_target_len", "Mean decoder target length"),
    ("encoder_vocab", "Encoder vocabulary size"),
    ("decoder_vocab", "Decoder vocabulary size"),
)

def read_lines(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f]

def compute_before(split: str, data_dir: Path) -> dict[str, float]:
    nl = read_lines(data_dir / f"{split}.nl")
    sql = read_lines(data_dir / f"{split}.sql")
    assert len(nl) == len(sql), f"{split}: mismatched NL/SQL counts"
    nl_tokens = [word_tokenize(line) for line in nl]
    sql_tokens = [line.split() for line in sql]
    return {
        "num_examples": len(nl),
        "mean_nl_len": mean(len(toks) for toks in nl_tokens),
        "mean_sql_len": mean(len(toks) for toks in sql_tokens),
        "nl_vocab": len({tok.lower() for seq in nl_tokens for tok in seq}),
        "sql_vocab": len({tok.lower() for seq in sql_tokens for tok in seq}),
    }

def compute_after(split: str, data_dir: Path, tokenizer: T5TokenizerFast) -> dict[str, float]:
    nl = read_lines(data_dir / f"{split}.nl")
    sql = read_lines(data_dir / f"{split}.sql")
    encodings = tokenizer(
        nl, padding=False, truncation=True, max_length=512, return_attention_mask=False
    )["input_ids"]
    dec_inputs = tokenizer(
        [f"<extra_id_0>{q}" for q in sql],
        padding=False,
        truncation=True,
        max_length=512,
        return_attention_mask=False,
    )["input_ids"]
    extra_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    dec_targets = []
    for seq in dec_inputs:
        tail = seq[1:] + [eos] if seq and seq[0] == extra_id else seq + [eos]
        if len(tail) > len(seq):
            tail = tail[:len(seq)]
        elif len(tail) < len(seq):
            tail += [pad] * (len(seq) - len(tail))
        dec_targets.append(tail)
    return {
        "mean_encoder_len": mean(len(seq) for seq in encodings),
        "mean_decoder_len": mean(len(seq) for seq in dec_inputs),
        "mean_target_len": mean(len(seq) for seq in dec_targets),
        "encoder_vocab": len({tok for seq in encodings for tok in seq}),
        "decoder_vocab": len({tok for seq in dec_inputs for tok in seq}),
    }

def fmt(value: float) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)

def latex_table(before: dict, after: dict) -> str:
    lines = [r"\begin{table}[h!]", r"\centering", r"\begin{tabular}{lcc}", r"\toprule",
             r"Statistics Name & Train & Dev \\", r"\midrule"]
    for key, label in DATA_FIELDS:
        lines.append(f"{label} & {fmt(before['train'][key])} & {fmt(before['dev'][key])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Data statistics before any pre-processing.}",
              r"\label{tab:data_stats_before}", r"\end{table}", ""]
    lines += [r"\begin{table}[h!]", r"\centering", r"\begin{tabular}{lcc}", r"\toprule",
              r"Statistics Name & Train & Dev \\", r"\midrule",
              r"\multicolumn{3}{l}{\textbf{T5 fine-tuned model}} \\", r"\midrule"]
    for key, label in AFTER_FIELDS:
        lines.append(f"{label} & {fmt(after['train'][key])} & {fmt(after['dev'][key])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Data statistics after pre-processing.}",
              r"\label{tab:data_stats_after}", r"\end{table}"]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="part2/data", type=Path)
    parser.add_argument("--output", default="part2/data_stats.tex", type=Path)
    args = parser.parse_args()

    nltk.download("punkt", quiet=True)
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    before = {split: compute_before(split, args.data_dir) for split in ("train", "dev")}
    after = {split: compute_after(split, args.data_dir, tokenizer) for split in ("train", "dev")}

    args.output.write_text(latex_table(before, after))
    print(f"Wrote LaTeX tables to {args.output}")

if __name__ == "__main__":
    main()