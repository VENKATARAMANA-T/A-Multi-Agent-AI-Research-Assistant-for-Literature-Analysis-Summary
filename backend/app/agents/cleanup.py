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


def clean_name(value: str | None) -> str:
    """Clean a structured entity name: no markers, no stray brackets, no padding."""
    cleaned = strip_markers(value)
    cleaned = re.sub(r"^[\[\(]|[\]\)]$", "", cleaned).strip(TRAILING_PUNCT)
    return cleaned
