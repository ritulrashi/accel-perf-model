#!/usr/bin/env python3
"""Rule 1 guard: every number quoted in a hand-written results document must
appear in a generated results file.

    scripts/check_results_numbers.py RESULTS.md GENERATED_FILE [GENERATED_FILE ...]

Checked tokens are the ones a reader would take as measurements:
  * integers of 3+ digits, with or without thousands separators (5016, 2,778,141)
  * decimals (0.7320, 93.69, 98.3)
  * percentages are checked through their number (73.2 in "73.2 %")
Small integers (grid sizes, "4x16", "3 configs") are parameters and are not
checked. A decimal may be quoted with fewer digits than the generated value
only by truncation (never rounding): "0.73" is accepted for "0.7320" but
"0.74" is not. A percentage p is accepted if p/100 truncates a generated
fraction (73.2 % from 0.7320) or appears itself.

Generated text files are scanned for the same token shapes; CSV files are
read with the csv module and each field taken whole.
"""
import csv
import re
import sys

NUM = re.compile(r"(?<![\w.§#/-])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})(?![\w.]?\d)")
PCT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?%")


def strip_refs(text):
    text = re.sub(r"§\d+(?:\.\d+)*", "", text)                  # section references
    text = re.sub(r"(?m)^(#+\s+)\d+(?:\.\d+)*", r"\1", text)    # numbered headings
    # Commit hashes (hex with at least one letter, so all-digit measurements stay).
    text = re.sub(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b", "", text)
    return text


def generated_values(paths):
    vals = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            if p.endswith(".csv"):
                for row in csv.reader(f):
                    vals.update(v.strip() for v in row)
            else:
                for tok in NUM.findall(f.read()):
                    vals.add(tok.replace(",", ""))
    return {v for v in vals if re.fullmatch(r"\d+(?:\.\d+)?", v)}


def truncates(quoted, generated):
    """quoted '0.73' accepted for generated '0.7320' (same integer part, prefix digits)."""
    return any(g.startswith(quoted) and "." in g and g.split(".")[0] == quoted.split(".")[0]
               for g in generated)


def main():
    doc, gens = sys.argv[1], sys.argv[2:]
    with open(doc, encoding="utf-8") as f:
        text = strip_refs(f.read())
    generated = generated_values(gens)
    fractions = {g for g in generated if g.startswith("0.")}

    missing, checked = [], set()
    pct_spans = []
    for m in PCT.finditer(text):
        pct_spans.append(m.span(1))
        tok = m.group(1)
        checked.add(tok + "%")
        if tok in generated:
            continue
        # p % accepted if p/100 truncates a generated fraction (73.2 % <- 0.7320)
        whole, _, frac = tok.partition(".")
        as_fraction = "0." + whole.zfill(2) + frac
        if any(g.startswith(as_fraction) for g in fractions):
            continue
        missing.append(tok + " %")
    for m in NUM.finditer(text):
        if any(a <= m.start(1) < b for a, b in pct_spans):
            continue
        tok = m.group(1).replace(",", "")
        checked.add(tok)
        if tok in generated or ("." in tok and truncates(tok, generated)):
            continue
        missing.append(m.group(1))
    if missing:
        print(f"check-results-numbers: {len(missing)} of {len(checked)} numbers in {doc} not found in generated files:")
        for t in sorted(set(missing)):
            print("  " + t)
        return 1
    print(f"check-results-numbers: all {len(checked)} numbers in {doc} appear in the generated results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
