"""Citation formatting — BibTeX, APA, IEEE, MLA.

Built from the metadata already extracted at ingestion, so exporting costs
nothing and works without an API key.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

STYLES = ("bibtex", "apa", "ieee", "mla", "ris")

# Characters LaTeX treats specially; a raw "&" or "%" breaks a .bib file.
LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(char, char) for char in str(text or ""))


def split_name(name: str) -> tuple[str, str]:
    """Return (family, given) for a name written either way round."""
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip().strip(",")
    if not cleaned:
        return "", ""
    if "," in cleaned:
        family, _, given = cleaned.partition(",")
        return family.strip(), given.strip()
    parts = cleaned.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    return parts[-1], " ".join(parts[:-1])


def initials(given: str) -> str:
    return " ".join(f"{part[0]}." for part in given.split() if part)


def citation_key(paper: Any) -> str:
    """A stable, LaTeX-safe BibTeX key: surnameYearFirstword."""
    authors = paper.authors or []
    family = split_name(authors[0])[0] if authors else "unknown"
    family = unicodedata.normalize("NFKD", family).encode("ascii", "ignore").decode()
    family = re.sub(r"[^A-Za-z]", "", family).lower() or "unknown"

    # "n.d." would put dots in the key, and a BibTeX key containing a period
    # breaks \cite in LaTeX.
    year = str(paper.year) if paper.year else "nodate"
    title_word = ""
    for word in re.findall(r"[A-Za-z]{4,}", paper.title or ""):
        if word.lower() not in {"the", "and", "for", "with", "from", "into", "using"}:
            title_word = word.lower()
            break
    return f"{family}{year}{title_word}"


def to_bibtex(paper: Any) -> str:
    fields: list[tuple[str, str]] = []
    authors = paper.authors or []
    if authors:
        joined = " and ".join(
            f"{family}, {given}".strip().strip(",")
            for family, given in (split_name(a) for a in authors)
        )
        fields.append(("author", joined))
    if paper.title:
        fields.append(("title", paper.title))
    if paper.year:
        fields.append(("year", str(paper.year)))
    if paper.venue:
        fields.append(("journal", paper.venue))
    if paper.doi:
        fields.append(("doi", paper.doi))
    if paper.keywords:
        fields.append(("keywords", ", ".join(paper.keywords)))

    entry_type = "article" if paper.venue else "misc"
    body = ",\n".join(f"  {name} = {{{escape_latex(value)}}}" for name, value in fields)
    return f"@{entry_type}{{{citation_key(paper)},\n{body}\n}}"


def to_apa(paper: Any) -> str:
    authors = paper.authors or []
    names = [f"{family}, {initials(given)}".strip().strip(",") for family, given in
             (split_name(a) for a in authors)]

    if not names:
        author_text = "Unknown author"
    elif len(names) == 1:
        author_text = names[0]
    elif len(names) <= 20:
        author_text = ", ".join(names[:-1]) + f", & {names[-1]}"
    else:
        author_text = ", ".join(names[:19]) + ", ... " + names[-1]

    year = paper.year or "n.d."
    parts = [f"{author_text} ({year}). {paper.title or 'Untitled'}."]
    if paper.venue:
        parts.append(f"*{paper.venue}*.")
    if paper.doi:
        parts.append(f"https://doi.org/{paper.doi}")
    return " ".join(parts)


def to_ieee(paper: Any, number: int | None = None) -> str:
    authors = paper.authors or []
    names = [f"{initials(given)} {family}".strip() for family, given in
             (split_name(a) for a in authors)]

    if not names:
        author_text = "Unknown author"
    elif len(names) <= 6:
        author_text = ", ".join(names[:-1]) + (f", and {names[-1]}" if len(names) > 1 else names[0] if len(names) == 1 else "")
        if len(names) == 1:
            author_text = names[0]
    else:
        author_text = f"{names[0]} et al."

    parts = [f'{author_text}, "{paper.title or "Untitled"},"']
    if paper.venue:
        parts.append(f"*{paper.venue}*,")
    parts.append(f"{paper.year or 'n.d.'}.")
    if paper.doi:
        parts.append(f"doi: {paper.doi}.")

    prefix = f"[{number}] " if number else ""
    return prefix + " ".join(parts)


def to_mla(paper: Any) -> str:
    authors = paper.authors or []
    if not authors:
        author_text = "Unknown author"
    else:
        family, given = split_name(authors[0])
        author_text = f"{family}, {given}".strip().strip(",")
        if len(authors) > 1:
            author_text += ", et al"

    parts = [f'{author_text}. "{paper.title or "Untitled"}."']
    if paper.venue:
        parts.append(f"*{paper.venue}*,")
    parts.append(f"{paper.year or 'n.d.'}.")
    if paper.doi:
        parts.append(f"https://doi.org/{paper.doi}.")
    return " ".join(parts)


def to_ris(paper: Any) -> str:
    lines = ["TY  - JOUR"]
    for author in paper.authors or []:
        family, given = split_name(author)
        lines.append(f"AU  - {family}, {given}".rstrip(", "))
    if paper.title:
        lines.append(f"TI  - {paper.title}")
    if paper.year:
        lines.append(f"PY  - {paper.year}")
    if paper.venue:
        lines.append(f"JO  - {paper.venue}")
    if paper.doi:
        lines.append(f"DO  - {paper.doi}")
    if paper.abstract:
        lines.append(f"AB  - {re.sub(r'\\s+', ' ', paper.abstract)[:2000]}")
    for keyword in paper.keywords or []:
        lines.append(f"KW  - {keyword}")
    lines.append("ER  - ")
    return "\n".join(lines)


FORMATTERS = {
    "bibtex": to_bibtex,
    "apa": to_apa,
    "ieee": to_ieee,
    "mla": to_mla,
    "ris": to_ris,
}


def format_citation(paper: Any, style: str = "bibtex") -> str:
    formatter = FORMATTERS.get(style.lower())
    if formatter is None:
        raise ValueError(f"Unknown citation style '{style}'. Choose from: {', '.join(STYLES)}")
    return formatter(paper)


def format_many(papers: list[Any], style: str = "bibtex") -> str:
    style = style.lower()
    if style == "ieee":
        # IEEE numbers its references, so they only make sense as a set.
        return "\n".join(to_ieee(paper, index) for index, paper in enumerate(papers, start=1))
    separator = "\n\n" if style in ("bibtex", "ris") else "\n\n"
    return separator.join(format_citation(paper, style) for paper in papers)


def media_type(style: str) -> str:
    return {
        "bibtex": "application/x-bibtex",
        "ris": "application/x-research-info-systems",
    }.get(style.lower(), "text/plain; charset=utf-8")


def file_extension(style: str) -> str:
    return {"bibtex": "bib", "ris": "ris"}.get(style.lower(), "txt")
