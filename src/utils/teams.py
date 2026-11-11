"""Canonical team-name handling shared by ingestion and inference.

Football-Data.co.uk uses short ASCII club names (``Ath Madrid``, ``Betis``,
``Sociedad``). Users and fixture files usually type full names. Normalising
both sides through the same alias table prevents duplicate teams such as
"Atletico Madrid" and "Ath Madrid" from entering the dataset.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# Common spellings -> Football-Data canonical spelling.
TEAM_ALIASES: dict[str, str] = {
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "atletico": "Ath Madrid",
    "club atletico de madrid": "Ath Madrid",
    "atletico madrid cf": "Ath Madrid",
    "ath madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "athletic": "Ath Bilbao",
    "athletic club bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "sociedad": "Sociedad",
    "rayo vallecano": "Vallecano",
    "rayo": "Vallecano",
    "espanyol": "Espanol",
    "rcd espanyol": "Espanol",
    "rcd espanyol de barcelona": "Espanol",
    "espanyol barcelona": "Espanol",
    "celta vigo": "Celta",
    "celta de vigo": "Celta",
    "rc celta": "Celta",
    "rc celta de vigo": "Celta",
    "real club celta de vigo": "Celta",
    "real betis": "Betis",
    "betis": "Betis",
    "real betis balompie": "Betis",
    "deportivo alaves": "Alaves",
    "alaves": "Alaves",
    "real valladolid": "Valladolid",
    "valladolid": "Valladolid",
    "las palmas": "Las Palmas",
    "ud las palmas": "Las Palmas",
    "real oviedo": "Oviedo",
    "oviedo": "Oviedo",
    "fc barcelona": "Barcelona",
    "barcelona": "Barcelona",
    "real madrid cf": "Real Madrid",
    "real madrid": "Real Madrid",
    "sevilla fc": "Sevilla",
    "sevilla": "Sevilla",
    "villarreal cf": "Villarreal",
    "villarreal": "Villarreal",
    "valencia cf": "Valencia",
    "valencia": "Valencia",
    "getafe cf": "Getafe",
    "getafe": "Getafe",
    "girona fc": "Girona",
    "girona": "Girona",
    "ca osasuna": "Osasuna",
    "osasuna": "Osasuna",
    "rcd mallorca": "Mallorca",
    "mallorca": "Mallorca",
    "cadiz cf": "Cadiz",
    "cadiz": "Cadiz",
    "elche cf": "Elche",
    "elche": "Elche",
    "leganes": "Leganes",
    "granada cf": "Granada",
    "granada": "Granada",
    "ud almeria": "Almeria",
    "almeria": "Almeria",
    "sd huesca": "Huesca",
    "huesca": "Huesca",
    "eibar": "Eibar",
    "levante": "Levante",
    "levante ud": "Levante",
    "levante union deportiva": "Levante",
    "malaga": "Malaga",
}


def normalize_text(name: str) -> str:
    """Casefold, strip accents and collapse whitespace."""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", folded).strip().lower()


def canonical_team(name: str) -> str:
    """Map a club name to its canonical dataset spelling.

    Unknown names are returned stripped (never dropped or guessed), so data
    ingestion cannot silently rename a club it does not recognise.
    """
    cleaned = re.sub(r"\s+", " ", str(name)).strip()
    if not cleaned:
        return cleaned
    return TEAM_ALIASES.get(normalize_text(cleaned), cleaned)


def team_alias_table() -> dict[str, str]:
    """Expose a copy of the alias table for documentation/tests."""
    return dict(TEAM_ALIASES)


class TeamNotFoundError(ValueError):
    """Raised when a fixture team cannot be resolved to a dataset team."""


def resolve_team(query: str, available: list[str]) -> str:
    """Resolve a user-typed club name to its canonical dataset spelling.

    Order: exact (accent/case-insensitive) -> canonical alias -> unique
    substring match -> fuzzy suggestions. Unknown names raise with a helpful
    message; nothing is silently guessed.
    """
    cleaned = str(query).strip()
    q = normalize_text(cleaned)
    if not q:
        raise TeamNotFoundError("Empty team name.")

    by_norm = {normalize_text(team): team for team in available}
    if q in by_norm:
        return by_norm[q]

    canonical = canonical_team(cleaned)
    if normalize_text(canonical) in by_norm:
        return by_norm[normalize_text(canonical)]

    contains = [orig for norm, orig in by_norm.items() if q in norm or norm in q]
    if len(contains) == 1:
        return contains[0]

    suggestions = difflib.get_close_matches(q, list(by_norm), n=3, cutoff=0.5)
    pretty = [by_norm[s] for s in suggestions]
    hint = f" Did you mean: {', '.join(pretty)}?" if pretty else ""
    available_list = ", ".join(sorted(available))
    raise TeamNotFoundError(
        f"Unknown team '{query}'.{hint} Available teams: {available_list}"
    )
