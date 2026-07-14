"""Petits outils communs aux scripts de lancement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def trouver_racine_projet() -> Path:
    """Retourne le dossier qui contient README.md et src."""

    courant = Path(__file__).resolve()
    for dossier in [courant.parent, *courant.parents]:
        if (dossier / "README.md").exists() and (dossier / "src").exists():
            return dossier
    return courant.parents[1]


def preparer_imports_projet() -> Path:
    """Ajoute la racine du projet a sys.path et la retourne."""

    racine = trouver_racine_projet()
    if str(racine) not in sys.path:
        sys.path.insert(0, str(racine))
    return racine


def entier_optionnel(valeur: str) -> int | None:
    """Convertit un argument en entier, avec none/all/full pour aucune limite."""

    if valeur.lower() in {"none", "all", "full"}:
        return None
    return int(valeur)


def liste_virgules(valeur: str) -> list[str]:
    """Transforme 'a,b,c' en ['a', 'b', 'c'] sans elements vides."""

    return [element.strip() for element in valeur.split(",") if element.strip()]


def liste_virgules_validee(valeur: str, valeurs_valides: set[str], nom: str) -> list[str]:
    """Lit une liste separee par des virgules et valide chaque element."""

    elements = liste_virgules(valeur)
    if not elements:
        raise argparse.ArgumentTypeError(f"At least one {nom} is required")

    inconnus = [element for element in elements if element not in valeurs_valides]
    if inconnus:
        valeurs = ", ".join(sorted(valeurs_valides))
        raise argparse.ArgumentTypeError(f"Unknown {nom}(s): {', '.join(inconnus)}. Valid values: {valeurs}")

    return elements


def filtre_virgules_ou_aucun(valeur: str) -> set[str] | None:
    """Retourne un set depuis une liste a virgules, ou None si elle est vide."""

    elements = liste_virgules(valeur)
    return set(elements) if elements else None
