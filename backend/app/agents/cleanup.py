"""Post-processing helpers for LLM output.

The grounding rule tells every agent to cite `[S#]` markers inline, and models
over-apply it: they append markers to structured *name* fields too, producing
entities like "MedQA [S1]". Those pollute the knowledge graph and split
aggregation buckets ("PubMed" vs "PubMed abstracts [S1]"), so names are cleaned
defensively here in addition to being forbidden in the prompts.
"""

from __future__ import annotations

import re

MARKER_RE = re.compile(r"\s*[\[\(]\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*[\]\)]", re.IGNORECASE)
TRAILING_PUNCT = " \t\n.,;:—-–"


def strip_markers(value: str | None) -> str:
    """Remove [S1], (S2), [S1, S3] style citation markers from a string."""
    if not value:
        return ""
    cleaned = MARKER_RE.sub("", str(value))
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(TRAILING_PUNCT)


# The closing bracket is optional: earlier extraction stripped it, so names
# already stored as "Convolutional Neural Network (CNN" must still resolve.
ACRONYM_RE = re.compile(r"^(.{4,}?)\s*\(([A-Za-z0-9][A-Za-z0-9\-/]{1,12}?)s?\)?\s*$")


def clean_name(value: str | None) -> str:
    """Clean a structured entity name: no markers, no stray brackets, no padding.

    Brackets are only stripped when they wrap the whole string ("[BERT]"). An
    unconditional strip of a trailing bracket mangles the common
    "Convolutional Neural Network (CNN)" form into "...Network (CNN", which
    then fails to match the same concept written any other way.
    """
    cleaned = strip_markers(value)
    if not cleaned:
        return ""

    for opener, closer in (("[", "]"), ("(", ")")):
        if cleaned.startswith(opener) and cleaned.endswith(closer):
            inner = cleaned[1:-1]
            # Only unwrap when the brackets really are a pair around the whole
            # string, not "(a) and (b)".
            if opener not in inner and closer not in inner:
                cleaned = inner
                break

    # A trailing opener or an unmatched closer is leftover noise.
    if cleaned.count("(") > cleaned.count(")"):
        cleaned = cleaned.rstrip("(").rstrip()
    elif cleaned.count(")") > cleaned.count("("):
        cleaned = cleaned.rstrip(")").rstrip()

    return cleaned.strip(TRAILING_PUNCT)


def expand_acronym(name: str | None) -> tuple[str, str] | None:
    """Split "Convolutional Neural Network (CNN)" into its long and short forms.

    Papers introduce a term once in full and then use the acronym, so the same
    concept arrives under two names and becomes two unconnected graph nodes.
    Recognising the pair is what lets them merge.
    """
    match = ACRONYM_RE.match(clean_name(name))
    if not match:
        return None
    long_form, acronym = match.group(1).strip(), match.group(2).strip()
    if not long_form or not acronym or long_form.lower() == acronym.lower():
        return None
    return long_form, acronym


def name_variants(name: str | None) -> list[str]:
    """Every surface form an entity might legitimately be written as."""
    cleaned = clean_name(name)
    if not cleaned:
        return []
    pair = expand_acronym(cleaned)
    if pair is None:
        return [cleaned]
    long_form, acronym = pair
    return [cleaned, long_form, acronym]
