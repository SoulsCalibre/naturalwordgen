from typing import List

def jaccard_similarity(seq1: List, seq2: List) -> float:
    """
    Calculate the Jaccard similarity coefficient between two sequences.
    """
    set1, set2 = set(seq1), set(seq2)
    return len(set1 & set2) / len(set1 | set2)

def overlap_coefficient(seq1: List, seq2: List) -> float:
    """
    Calculate the overlap coefficient between two sequences.
    """
    set1, set2 = set(seq1), set(seq2)
    return len(set1 & set2) / min(len(set1), len(set2))

def edit_distance(seq1: List, seq2: List) -> int:
    """
    Calculate the edit distance between two sequences.
    """
    if len(seq1) < len(seq2):
        return edit_distance(seq2, seq1)

    if not seq2:
        return len(seq1)

    previous_row = range(len(seq2) + 1)
    for i, elem1 in enumerate(seq1):
        current_row = [i + 1]
        for j, elem2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (elem1 != elem2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def dice_coefficient(seq1: List, seq2: List) -> float:
    """
    Calculate the Dice coefficient between two sequences.
    """
    set1, set2 = set(seq1), set(seq2)
    return 2 * len(set1 & set2) / (len(set1) + len(set2))
