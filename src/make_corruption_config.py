"""Task 3.1 output: write the corruption benchmark configuration to
docs/task3_1_corruption_levels.csv (corruption family x severity level x parameter).
"""
import csv

from config import DOCS_DIR
from corruptions import corruption_config_rows

OUT = DOCS_DIR / "task3_1_corruption_levels.csv"

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["corruption_type", "severity_level_1_mildest_5_strongest",
                "parameter_value", "parameter_description"])
    for ctype, level, param, desc in corruption_config_rows():
        w.writerow([ctype, level, param, desc])

print(f"saved -> {OUT}")
