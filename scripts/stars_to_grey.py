"""Rewrite a table-rows file so that a starred cell (significantly below EpiKV, exact McNemar) is set in grey with
\\sigbelow{} and carries no star. Reads the rows on stdin, writes them on stdout.

  <generator> | stars_to_grey.py > tables/<rows>.tex
"""
import re
import sys

for line in sys.stdin:
    sys.stdout.write(re.sub(r"(\\textbf\{[\d.]+\}|[\d.]+)\$\^\{\*+\}\$", r"\\sigbelow{\1}", line))
