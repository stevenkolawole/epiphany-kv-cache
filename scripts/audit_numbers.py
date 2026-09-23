#!/usr/bin/env python3
"""House rule H1 (reports/BIBLE.md): result numbers live in tables, figures and captions, not in
running prose. This audit lists every number that appears in prose in the main-text sections,
so a writer can move or justify each one. Captions, table bodies, math, labels, citations and
comments are skipped; so are numbers that name things (layer indices, budgets K, years, section
numbers, token counts that define the setup) when they appear in an allow-list pattern.

  python scripts/audit_numbers.py ../EpiKV-overleaf/sections/{abstract,intro,related_work,methods,results,discussion}.tex
  python scripts/audit_numbers.py --all ../EpiKV-overleaf/sections/*.tex   # appendix too

Exit status is the number of flagged lines, so it can gate a commit.
"""
import argparse
import re
import sys

ALLOW = [
    r"\$[^$]*\$",                              # inline math
    r"\\(ref|eqref|cite[pt]?|citealp|label|S)\{[^}]*\}",
    r"\\ref\{[^}]*\}",
    r"\b(19|20)\d\d[ab]?\b",                   # years
    r"\b(Llama|Qwen)-?\d+B\b|-7B|-8B",          # model names
    r"\b(MATH|AIME|GSM)-?\d+K?\b|AIME \d{4}",   # dataset names
    r"\bK\s*[=∈]\s*\{?[\d, ]+\}?",              # budgets
    r"\b(layers?|layer)\s+\d+(--\d+)?\b|\(\d+--\d+\)",   # layer indices
    r"\b\d+-token\b|\b\d+(,\d{3})?[- ]tokens?\b",        # setup lengths
    r"\b(rolling|window)[- ]?\d+\b|w\{=\}\d+|\\tau\{=\}\d+",
    r"\bA100s?\b|\b80\\?,?\\?GB\b|H100",
    r"\\vspace\*?\{[^}]*\}|\\begin\{enumerate\}\[[^\]]*\]|\\setlength\{[^}]*\}\{[^}]*\}",
    r"\bn\{=\}\d+\b|\bn=\d+\b",
    r"\b(FA2|FlashAttention-2|transformers|torch|vLLM|R1)\s*[\d.]+\b",
    r"\b\d+(\.\d+)?\s*(x|\\times)\b",
    r"top-\$?K",
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",   # words are fine
    r"\bL2\b|\bH2O\b|\b\d+D\b|10\^\{?[-\d]+\}?|\\S\d",
]


def strip(line):
    s = line
    s = re.sub(r"(?<!\\)%.*$", "", s)             # comments
    for pat in ALLOW:
        s = re.sub(pat, " ", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--all", action="store_true", help="include appendix-style files (same rules)")
    a = ap.parse_args()
    flagged = 0
    for f in a.files:
        in_table = in_caption = 0
        depth = 0
        for ln, raw in enumerate(open(f), 1):
            line = raw.rstrip("\n")
            if re.search(r"\\begin\{(table|tabular|figure|minipage|equation|align|center)", line):
                in_table += 1
            if re.search(r"\\end\{(table|tabular|figure|minipage|equation|align|center)", line):
                in_table = max(0, in_table - 1)
                continue
            if "\\caption{" in line or "\\captionof{" in line:
                in_caption = 1
                depth = line.count("{") - line.count("}")
                continue
            if in_caption:
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    in_caption = 0
                continue
            if in_table or line.lstrip().startswith("%") or "\\tablerows" in line or "\\input{" in line:
                continue
            s = strip(line)
            nums = re.findall(r"(?<![A-Za-z\\{])\d+(?:[.,]\d+)?(?:\\%|%)?", s)
            nums = [n for n in nums if not re.fullmatch(r"\d", n) or n in ("0",)]   # single digits are usually list/section items
            if nums:
                flagged += 1
                print(f"{f}:{ln}: {nums}  |  {line.strip()[:110]}")
    print(f"\n{flagged} prose lines carry numbers", file=sys.stderr)
    sys.exit(min(flagged, 255))


if __name__ == "__main__":
    main()
