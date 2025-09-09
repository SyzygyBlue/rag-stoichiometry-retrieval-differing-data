#!/usr/bin/env python3
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any


def load_problem_mapping(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def summarize_unresolved(data: Dict[str, Any], top_n: int = 20) -> List[Dict[str, Any]]:
    items = data.get('unresolved', [])

    # Count frequency per alias
    alias_counts = Counter(it.get('alias', '') for it in items)

    # Pre-aggregate by alias
    by_alias: Dict[str, Dict[str, Any]] = {}
    for it in items:
        alias = it.get('alias', '')
        if not alias:
            continue
        bucket = by_alias.setdefault(alias, {
            'alias': alias,
            'count': 0,
            'category_counts': Counter(),
            'scene_counts': Counter(),
            'candidates': [],  # list of dicts with entity_id, name, type, score
        })
        bucket['count'] += 1
        if it.get('category_hint'):
            bucket['category_counts'][it['category_hint']] += 1
        if it.get('scene'):
            bucket['scene_counts'][it['scene']] += 1
        for c in it.get('top_candidates', []) or []:
            # Keep minimal candidate info
            bucket['candidates'].append({
                'entity_id': c.get('entity_id'),
                'entity_type': c.get('entity_type'),
                'name': c.get('name'),
                'score': float(c.get('score') or 0.0),
            })

    # Build summary list
    summary = []
    for alias, info in by_alias.items():
        # Determine modal category
        category_hint = None
        if info['category_counts']:
            category_hint = info['category_counts'].most_common(1)[0][0]
        # Top scenes
        sample_scenes = [s for s, _ in info['scene_counts'].most_common(5)]
        # Aggregate candidate scores by entity_id (max score seen)
        cand_scores: Dict[str, Dict[str, Any]] = {}
        for c in info['candidates']:
            eid = c.get('entity_id')
            if not eid:
                continue
            entry = cand_scores.setdefault(eid, {
                'entity_id': eid,
                'entity_type': c.get('entity_type'),
                'name': c.get('name'),
                'best_score': -1.0,
            })
            if c['score'] > entry['best_score']:
                entry['best_score'] = c['score']
        # Sort candidates by best_score desc and take top 5
        top_candidates = sorted(cand_scores.values(), key=lambda x: x['best_score'], reverse=True)[:5]

        summary.append({
            'alias': alias,
            'count': info['count'],
            'category_hint': category_hint,
            'sample_scenes': sample_scenes,
            'top_candidates': top_candidates,
        })

    # Sort by count desc then alias asc
    summary.sort(key=lambda x: (-x['count'], x['alias']))
    return summary[:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='mappings/problem_mapping.json')
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    data = load_problem_mapping(args.path)
    out = summarize_unresolved(data, top_n=args.top)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
