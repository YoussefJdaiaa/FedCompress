

from __future__ import annotations

import argparse

from common_cli import preparer_imports_projet


PROJECT_ROOT = preparer_imports_projet()


def build_parser() -> argparse.ArgumentParser:
    """Cree le parseur des arguments terminal.

    Entree:
        Aucune entree.

    Sortie:
        Parseur configure.
    """

    parser = argparse.ArgumentParser(description="Inspecte le dataset Fashion-MNIST.")
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Sauvegarde une grille PNG dans figures/fashion_mnist_samples.png.",
    )
    parser.add_argument("--num-images", type=int, default=10)
    return parser


def main() -> None:
    """Point d'entree du script terminal.

    Entree:
        Arguments tapes dans le terminal.

    Sortie:
        Aucune sortie Python. Affiche les infos dataset et peut sauvegarder un PNG.
    """

    # Etape 1: lire les options simples du terminal.
    parser = build_parser()
    args = parser.parse_args()

    # Etape 2: importer le code dataset apres le parsing des arguments.
    from src.dataset_tools import (
        afficher_resume_dataset,
        charger_fashion_mnist,
        resumer_dataset,
        sauvegarder_grille_exemples,
    )

    # Etape 3: charger le dataset dans le meme dossier que les experiences.
    dossier_donnees = PROJECT_ROOT / "data"
    donnees_train, donnees_test = charger_fashion_mnist(dossier_donnees)

    # Etape 4: afficher les tailles des splits et la repartition des classes.
    resume = resumer_dataset(donnees_train, donnees_test)
    afficher_resume_dataset(resume)

    # Etape 5: sauvegarder si demande une grille d'images exemple.
    if args.save_grid:
        chemin_sortie = PROJECT_ROOT / "figures" / "fashion_mnist_samples.png"
        chemin_sauvegarde = sauvegarder_grille_exemples(
            donnees_train,
            chemin_sortie,
            nombre_images=args.num_images,
        )
        print("\nSaved sample grid:", chemin_sauvegarde)


if __name__ == "__main__":
    main()
