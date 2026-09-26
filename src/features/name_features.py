import pandas as pd
import difflib

def calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Returns normalized edit similarity using difflib.
    """
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    return difflib.SequenceMatcher(None, str(str1), str(str2)).ratio()

def calculate_token_jaccard(str1: str, str2: str) -> float:
    """
    Returns Jaccard similarity of string tokens.
    """
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    
    set1 = set(str(str1).split())
    set2 = set(str(str2).split())
    
    if not set1 and not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0
