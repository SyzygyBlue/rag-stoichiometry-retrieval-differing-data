#!/usr/bin/env python3
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAPPINGS_DIR = os.path.join(BASE_DIR, "mappings")
PROBLEM_PATH = os.path.join(MAPPINGS_DIR, "problem_mapping.json")
OUT_JSON = os.path.join(MAPPINGS_DIR, "unresolved_alias_top30.json")
OUT_TSV = os.path.join(MAPPINGS_DIR, "unresolved_alias_top30.tsv")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(topn: int = 30):
    data = load_json(PROBLEM_PATH)
    unresolved = data.get("unresolved", [])

    counts = Counter()
    samples = defaultdict(list)
    cats = Counter()

    for rec in unresolved:
        alias = rec.get("alias")
        if not alias or alias == "<file_error>":
            continue
        counts[alias] += 1
        sc = rec.get("scene")
        if sc and len(samples[alias]) < 5:
            samples[alias].append(sc)
        cat = rec.get("category_hint") or "unknown"
        cats[(alias, cat)] += 1

    top = counts.most_common(topn)

    out = []
    for alias, cnt in top:
        cat_counts = {k[1]: v for k, v in cats.items() if k[0] == alias}
        cat_sorted = sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_cat = cat_sorted[0][0] if cat_sorted else "unknown"
        out.append({
            "alias": alias,
            "count": cnt,
            "category_hint": top_cat,
            "sample_scenes": samples.get(alias, [])
        })

    os.makedirs(MAPPINGS_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "source_updated_at": data.get("updated_at"),
            "total_unresolved": len(unresolved),
            "top": out
        }, f, ensure_ascii=False, indent=2)

    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("alias\tcount\tcategory_hint\tsample_scenes\n")
        for row in out:
            f.write(f"{row['alias']}\t{row['count']}\t{row['category_hint']}\t{', '.join(row['sample_scenes'])}\n")


if __name__ == "__main__":
    main()
