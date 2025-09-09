#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PM_PATH = ROOT / 'mappings' / 'problem_mapping.json'
OUT_PATH = ROOT / 'mappings' / 'top_twenty.json'

ALLOWED = {'item', 'character'}

def main():
    with PM_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    unresolved = data.get('unresolved', [])

    # group by alias, tracking counts, modal category, and sample scenes
    buckets = {}
    for it in unresolved:
        cat = it.get('category_hint')
        if cat not in ALLOWED:
            continue
        alias = it.get('alias') or ''
        if not alias:
            continue
        b = buckets.setdefault(alias, {
            'alias': alias,
            'count': 0,
            'category_counts': Counter(),
        })
        b['count'] += 1
        b['category_counts'][cat] += 1

    # build list with modal category
    rows = []
    for alias, b in buckets.items():
        modal_cat = None
        if b['category_counts']:
            modal_cat = b['category_counts'].most_common(1)[0][0]
        rows.append({
            'alias': alias,
            'category_hint': modal_cat,
            'count': b['count'],
            'resolved_to': ''
        })

    rows.sort(key=lambda x: (-x['count'], x['alias']))
    top = rows[:20]

    with OUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
