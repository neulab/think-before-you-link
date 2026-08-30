import json
import re
from typing import Dict, List, Optional


def get_score(pred: str, target: str, aliases: Optional[List[str]] = None,
              use_target_in_pred: bool = False, use_aliases: bool = False) -> bool:
    """Compare predicted Wikipedia title with target."""
    is_correct, _, _, _ = get_score_detailed(
        pred, target, aliases, use_target_in_pred, use_aliases)
    return is_correct


def get_score_detailed(
    pred: str, target: str, aliases: Optional[List[str]] = None,
    use_target_in_pred: bool = False, use_aliases: bool = False,
) -> tuple:
    """Compare predicted Wikipedia title with target.

    Returns (is_correct, match_type, pred_normalized, target_normalized).
    match_type is one of: "exact", "exact_disambig_stripped", "target_in_pred",
    "target_in_pred_disambig", "alias_exact", "alias_disambig", "alias_in_pred",
    "alias_in_pred_disambig", "no_match", "empty_pred", "empty_target".
    """
    pred_clean = pred.lower().strip()
    target_clean = target.lower().strip()

    if not pred_clean:
        return False, "empty_pred", pred_clean, target_clean

    if not target_clean:
        return False, "empty_target", pred_clean, target_clean

    # Exact match
    if pred == target or pred_clean == target_clean:
        return True, "exact", pred_clean, target_clean

    # Target in prediction (if enabled)
    if use_target_in_pred and (target in pred or target_clean in pred_clean):
        return True, "target_in_pred", pred_clean, target_clean

    # Remove disambiguation parentheses and compare main titles
    pred_main = re.sub(r'\s*\(.*?\)', '', pred_clean).strip()
    target_main = re.sub(r'\s*\(.*?\)', '', target_clean).strip()

    if pred_main == target_main:
        return True, "exact_disambig_stripped", pred_clean, target_clean

    if use_target_in_pred and target_main in pred_main:
        return True, "target_in_pred_disambig", pred_clean, target_clean

    # Check aliases (if enabled and available)
    if aliases and use_aliases:
        for alias in aliases:
            alias_clean = alias.lower().strip()
            alias_main = re.sub(r'\s*\(.*?\)', '', alias_clean).strip()

            if pred_clean == alias_clean or pred == alias:
                return True, "alias_exact", pred_clean, target_clean

            if pred_main == alias_main:
                return True, "alias_disambig", pred_clean, target_clean

            if use_target_in_pred and alias_clean in pred_clean:
                return True, "alias_in_pred", pred_clean, target_clean

            if use_target_in_pred and alias_main in pred_main:
                return True, "alias_in_pred_disambig", pred_clean, target_clean

    return False, "no_match", pred_clean, target_clean


def load_entity_links(filepath: str) -> Dict:
    """Load entity link information from JSON file."""
    if not filepath:
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            entities = json.load(f)

        entity_links_dict = {}
        for entity in entities:
            qid = entity.get("Wikidata ID")
            if qid:
                entity_links_dict[qid] = {
                    "wikidata_incoming_links": entity.get("wikidata_incoming_links", 0),
                }

        print(f"Loaded link information for {len(entity_links_dict)} entities")
        return entity_links_dict
    except Exception as e:
        print(f"Warning: Could not load entity links: {e}")
        return {}


def load_aliases_cache(filepath: str) -> Dict:
    """Load aliases cache from JSON file."""
    if not filepath:
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            aliases_cache = json.load(f)
        print(f"Loaded aliases for {len(aliases_cache)} entities")
        return aliases_cache
    except Exception as e:
        print(f"Warning: Could not load aliases cache: {e}")
        return {}


def filter_nil_samples(data: List[Dict]) -> List[Dict]:
    """Filter out samples where English Wikipedia Title is NIL."""
    filtered = [
        sample for sample in data
        if sample.get("English Wikipedia Title", "").strip().upper() != "NIL"
    ]
    print(f"  Filtered out {len(data) - len(filtered)} NIL samples")
    return filtered
