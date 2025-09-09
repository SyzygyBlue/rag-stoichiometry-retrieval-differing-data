#!/usr/bin/env python3
import os
import re
import json
import glob
import difflib
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAG_DIR = BASE_DIR
ENTITIES_PATH = os.path.join(RAG_DIR, "entities", "entities.json")
CHAR_DIR = os.path.join(RAG_DIR, "character")
MAPPINGS_DIR = os.path.join(RAG_DIR, "mappings")
ALIAS_MAP_PATH = os.path.join(MAPPINGS_DIR, "entity_aliases.json")
PROBLEM_PATH = os.path.join(MAPPINGS_DIR, "problem_mapping.json")
LOG_PATH = os.path.join(MAPPINGS_DIR, "resolve_run.log")
TOP_TWENTY_PATH = os.path.join(MAPPINGS_DIR, "top_twenty.json")

SCENE_EXCLUDES = {"entities", "character", "indexes", "mappings", "scripts"}
SCENE_GLOB = os.path.join(RAG_DIR, "*.json")

# Note: Removed unsupported Unicode property class usage (\p{P}, \p{S}).
# Punctuation and symbol stripping is handled in norm_text() below.


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # basic normalization: lowercase, strip punctuation, collapse whitespace
    t = s.lower()
    # remove common punctuation non-regex-safe
    t = re.sub(r"[\.,;:'\"`~!@#$%^&*()\-_=+\[\]{}|\\/<>?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log(msg: str):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(f"{datetime.now().isoformat()} | {msg}\n")
    except Exception:
        # best-effort logging; ignore failures
        pass


def build_entity_indices(entities_path, char_dir):
    data = load_json(entities_path)
    root = data.get("entities", data)  # support either wrapped or flat

    type_keys = [
        ("items", "item"),
        ("technologies", "technology"),
        ("locations", "location"),
        ("organizations", "organization"),
        ("factions", "faction"),
    ]

    alias_by_type = {t: defaultdict(set) for _, t in type_keys}
    name_by_id = {}
    type_by_id = {}

    for key, tname in type_keys:
        arr = root.get(key, [])
        for obj in arr:
            eid = obj.get("id") or obj.get("name")
            if not eid:
                continue
            name = obj.get("name", eid)
            type_by_id[eid] = tname
            name_by_id[eid] = name
            aliases = set()
            aliases.add(eid)
            aliases.add(name)
            for a in obj.get("aliases", []) or []:
                aliases.add(a)
            # include aka_in_text if present
            for a in obj.get("aka_in_text", []) or []:
                aliases.add(a)
            for a in aliases:
                alias_by_type[tname][norm_text(a)].add(eid)

    # characters from character/*.json
    char_alias = defaultdict(set)
    for path in glob.glob(os.path.join(char_dir, "*.json")):
        try:
            c = load_json(path)
        except Exception:
            continue
        ident = c.get("identity", {})
        cid = ident.get("id") or os.path.splitext(os.path.basename(path))[0]
        display = ident.get("display_name") or cid
        aliases = set([cid, display])
        for a in ident.get("aliases", []) or []:
            aliases.add(a)
        for a in aliases:
            char_alias[norm_text(a)].add(cid)
        name_by_id[cid] = display
        type_by_id[cid] = "character"

    return alias_by_type, char_alias, name_by_id, type_by_id


def load_top_twenty_map(top_twenty_path: str, char_dir: str):
    """Load override alias->entity_id mapping from top_twenty.json.

    The file contains records with fields including:
      - alias (string)
      - resolved_to: either an entity id string (e.g., "two-seater-buggy") or a path to a character JSON.

    Returns a dict mapping normalized alias -> entity_id.
    """
    if not os.path.exists(top_twenty_path):
        return {}
    try:
        data = load_json(top_twenty_path)
    except Exception:
        return {}

    out = {}
    for rec in data or []:
        try:
            alias = rec.get("alias")
            resolved = rec.get("resolved_to")
            if not alias or not resolved:
                continue
            an = norm_text(alias)
            eid = None
            if isinstance(resolved, str) and resolved.lower().endswith(".json"):
                # Likely a character path; derive id from JSON identity or filename.
                try:
                    c = load_json(resolved)
                    ident = c.get("identity", {})
                    eid = ident.get("id") or os.path.splitext(os.path.basename(resolved))[0]
                except Exception:
                    eid = os.path.splitext(os.path.basename(resolved))[0]
            else:
                eid = resolved
            if eid:
                out[an] = eid
        except Exception:
            # skip malformed record
            continue
    return out


def collect_scene_files(rag_dir):
    # top-level json files are scenes (e.g., 00-prologue.json, 03-chapter02-scene02.json)
    files = [p for p in glob.glob(SCENE_GLOB) if os.path.isfile(p)]
    return files


def extract_mentions(scene_json):
    out = []  # list of (category_hint, text)

    # characters
    for k in ["characters", "cast", "people"]:
        arr = scene_json.get(k) or []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    out.append(("character", it.get("id") or it.get("display_name") or it.get("name") or it.get("alias")))
                elif isinstance(it, str):
                    out.append(("character", it))

    # items & tech
    for k in ["items_and_tech", "items", "tech", "technologies"]:
        arr = scene_json.get(k) or []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    # prefer id, else name
                    txt = it.get("id") or it.get("name") or it.get("alias")
                    cat = it.get("category") or it.get("type") or "item"
                    out.append(("technology" if "tech" in (cat or "").lower() else "item", txt))
                elif isinstance(it, str):
                    out.append(("item", it))

    # locations
    for k in ["locations", "places"]:
        arr = scene_json.get(k) or []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    out.append(("location", it.get("id") or it.get("name") or it.get("alias")))
                elif isinstance(it, str):
                    out.append(("location", it))

    # factions / orgs
    for k in ["factions_orgs", "factions", "organizations", "orgs"]:
        arr = scene_json.get(k) or []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    out.append(("organization", it.get("id") or it.get("name") or it.get("alias")))
                elif isinstance(it, str):
                    out.append(("organization", it))

    # narrative beats entities
    for k in ["narrative_beats", "beats"]:
        arr = scene_json.get(k) or []
        if isinstance(arr, list):
            for beat in arr:
                if isinstance(beat, dict):
                    ents = beat.get("entities_involved") or []
                    for it in ents:
                        if isinstance(it, dict):
                            out.append(("unknown", it.get("id") or it.get("name") or it.get("alias")))
                        elif isinstance(it, str):
                            out.append(("unknown", it))

    # filter nulls
    return [(cat, t) for (cat, t) in out if isinstance(t, str) and t.strip()]


def choose_type_hint(cat):
    c = (cat or "").lower()
    if c.startswith("char"):
        return "character"
    if c.startswith("loc") or c == "place":
        return "location"
    if c.startswith("faction") or c.startswith("org"):
        return "organization"
    if c.startswith("tech"):
        return "technology"
    if c.startswith("item"):
        return "item"
    return "unknown"


def fuzzy_candidates(query, alias_map_all, name_by_id, type_by_id, topn=5):
    # alias_map_all: dict alias(normalized)->set(ids)
    keys = list(alias_map_all.keys())
    qn = norm_text(query)
    # use difflib for similarity
    scores = [(k, difflib.SequenceMatcher(None, qn, k).ratio()) for k in keys]
    scores.sort(key=lambda x: x[1], reverse=True)
    out = []
    for k, sc in scores[:max(topn, 10)]:
        for eid in alias_map_all[k]:
            out.append({
                "entity_id": eid,
                "entity_type": type_by_id.get(eid, "unknown"),
                "score": round(float(sc), 4),
                "name": name_by_id.get(eid, eid)
            })
        if len(out) >= topn:
            break
    # unique by (entity_id)
    uniq = {}
    for c in out:
        uniq[c["entity_id"]] = c
    out = list(uniq.values())
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topn]


def resolve_mentions(scene_path, alias_by_type, char_alias, name_by_id, type_by_id,
                     existing_rules, unresolved, ambiguous, top20_map,
                     multi_threshold=0.86, single_threshold=0.9):
    log(f"processing scene: {os.path.basename(scene_path)}")
    scene = load_json(scene_path)
    scene_id = os.path.splitext(os.path.basename(scene_path))[0]

    mentions = extract_mentions(scene)

    # build global alias map
    alias_all = defaultdict(set)
    for t, amap in alias_by_type.items():
        for a, ids in amap.items():
            for i in ids:
                alias_all[a].add(i)
    for a, ids in char_alias.items():
        for i in ids:
            alias_all[a].add(i)

    new_rules = []

    for cat, text in mentions:
        if not text:
            continue
        alias_raw = text
        alias_norm = norm_text(alias_raw)
        cat_hint = choose_type_hint(cat)

        # 0) explicit override via top_twenty.json
        if alias_norm in top20_map:
            eid = top20_map[alias_norm]
            etypes = [type_by_id.get(eid, "unknown")]
            canon = [name_by_id.get(eid, eid)]
            rule = {
                "alias": alias_raw,
                "entity_ids": [eid],
                "entity_types": etypes,
                "canonical_phrases": canon,
                "context": {"scenes": [scene_id]},
                "confidence": 1.0
            }
            new_rules.append(rule)
            # no further processing needed for this mention
            continue

        # 1) exact id match in any type
        if alias_raw in type_by_id:
            new_rules.append({
                "alias": alias_raw,
                "entity_ids": [alias_raw],
                "entity_types": [type_by_id.get(alias_raw, "unknown")],
                "canonical_phrases": [name_by_id.get(alias_raw, alias_raw)],
                "context": {"scenes": [scene_id]},
                "confidence": 1.0
            })
            continue

        # 2) exact alias match by type hint
        chosen = set()
        if cat_hint == "character":
            ids = char_alias.get(alias_norm)
            if ids:
                chosen |= set(ids)
        else:
            amap = alias_by_type.get(cat_hint)
            if amap is not None:
                ids = amap.get(alias_norm)
                if ids:
                    chosen |= set(ids)

        # 3) if nothing, try global alias match
        if not chosen:
            ids = alias_all.get(alias_norm)
            if ids:
                chosen |= set(ids)

        # 4) multi-term heuristic: allow up to 2 matches for compound aliases
        entity_ids = []
        if chosen:
            entity_ids = list(chosen)
        else:
            # fuzzy across all
            cands = fuzzy_candidates(alias_raw, alias_all, name_by_id, type_by_id, topn=5)
            # take top-1 if strong, else top-2 if two strong parts
            if cands and cands[0]["score"] >= single_threshold:
                entity_ids = [cands[0]["entity_id"]]
            else:
                strong = [c for c in cands if c["score"] >= multi_threshold]
                # ensure different entities
                strong_ids = []
                for c in strong:
                    if c["entity_id"] not in strong_ids:
                        strong_ids.append(c["entity_id"])
                if len(strong_ids) >= 2:
                    entity_ids = strong_ids[:2]
                else:
                    # record as unresolved with candidates
                    unresolved.append({
                        "alias": alias_raw,
                        "category_hint": cat_hint,
                        "scene": scene_id,
                        "top_candidates": cands,
                        "notes": "no candidate above thresholds"
                    })
                    continue

        # 5) build rule
        etypes = [type_by_id.get(eid, "unknown") for eid in entity_ids]
        canon = [name_by_id.get(eid, eid) for eid in entity_ids]
        conf = 0.95 if len(entity_ids) == 1 else 0.9
        rule = {
            "alias": alias_raw,
            "entity_ids": entity_ids,
            "entity_types": etypes,
            "canonical_phrases": canon,
            "context": {"scenes": [scene_id]},
            "confidence": conf
        }
        new_rules.append(rule)

    # merge with existing by (alias, scenes)
    for r in new_rules:
        existing_rules.append(r)


def main():
    log("resolver start")
    alias_by_type, char_alias, name_by_id, type_by_id = build_entity_indices(ENTITIES_PATH, CHAR_DIR)
    top20_map = load_top_twenty_map(TOP_TWENTY_PATH, CHAR_DIR)
    # log count of overrides to aid debugging/verification
    try:
        log(f"loaded {len(top20_map)} top_twenty overrides")
    except Exception:
        pass

    # load existing mapping files if present
    if os.path.exists(ALIAS_MAP_PATH):
        alias_map = load_json(ALIAS_MAP_PATH)
    else:
        alias_map = {"version": 1, "updated_at": None, "normalization": {}, "rules": []}

    if os.path.exists(PROBLEM_PATH):
        problems = load_json(PROBLEM_PATH)
    else:
        problems = {"version": 1, "updated_at": None, "unresolved": [], "ambiguous": []}

    # IMPORTANT: start fresh each run to avoid accumulating stale/duplicate entries
    # Previous behavior appended to existing files, inflating counts and preserving
    # unresolved aliases that may now be resolved (e.g., top_twenty overrides).
    rules = []
    unresolved = []
    ambiguous = []

    scene_files = collect_scene_files(RAG_DIR)
    log(f"found {len(scene_files)} scene files")

    for spath in sorted(scene_files):
        try:
            resolve_mentions(spath, alias_by_type, char_alias, name_by_id, type_by_id,
                            rules, unresolved, ambiguous, top20_map)
        except Exception as e:
            log(f"error parsing scene {os.path.basename(spath)}: {e}")
            unresolved.append({
                "alias": "<file_error>",
                "category_hint": "unknown",
                "scene": os.path.splitext(os.path.basename(spath))[0],
                "top_candidates": [],
                "notes": f"error parsing scene: {e}"
            })

    alias_map["rules"] = rules
    alias_map["updated_at"] = datetime.now().isoformat()
    problems["unresolved"] = unresolved
    problems["ambiguous"] = ambiguous
    problems["updated_at"] = datetime.now().isoformat()

    save_json(ALIAS_MAP_PATH, alias_map)
    save_json(PROBLEM_PATH, problems)

    msg1 = f"Wrote {ALIAS_MAP_PATH} with {len(rules)} rules"
    msg2 = f"Wrote {PROBLEM_PATH} with unresolved={len(unresolved)} ambiguous={len(ambiguous)}"
    print(msg1)
    print(msg2)
    log(msg1)
    log(msg2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"fatal error: {e}")
        raise
