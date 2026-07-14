

from __future__ import annotations

import argparse

from common_cli import entier_optionnel, liste_virgules_validee, preparer_imports_projet


PROJECT_ROOT = preparer_imports_projet()


CLES_MODELES = {"resnet18", "convnext_tiny", "vit_base"}


def parse_models(value: str) -> list[str]:
    if value == "all":
        return ["resnet18", "convnext_tiny", "vit_base"]
    return liste_virgules_validee(value, CLES_MODELES, "model")


def build_parser() -> argparse.ArgumentParser:
    """Cree le parseur des arguments terminal.

    Entree:
        Aucune entree.

    Sortie:
        Parseur configure.
    """

    parser = argparse.ArgumentParser(description="Compare des modeles Hugging Face sur Fashion-MNIST.")
    parser.add_argument("--models", type=parse_models, default=["resnet18", "convnext_tiny"])
    parser.add_argument("--train-subset-size", type=entier_optionnel, default=None)
    parser.add_argument("--test-subset-size", type=entier_optionnel, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Entraine tout le modele au lieu de seulement la tete de classification.",
    )
    parser.set_defaults(freeze_backbone=True)
    return parser


def print_config(args: argparse.Namespace) -> None:
    """Affiche les parametres du benchmark.

    Entree:
        args: arguments lus depuis le terminal.

    Sortie:
        Aucune sortie Python. Le texte est affiche dans le terminal.
    """

    print("\n" + "=" * 80)
    print("FedCompress - Hugging Face model benchmark")
    print("=" * 80)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Models              : {', '.join(args.models)}")
    print(f"Train subset size   : {args.train_subset_size}")
    print(f"Test subset size    : {args.test_subset_size}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Epochs              : {args.epochs}")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Freeze backbone     : {args.freeze_backbone}")
    print(f"Seed                : {args.seed}")
    print("=" * 80 + "\n")


def print_ranking(classement: list[dict[str, object]]) -> None:
    """Affiche un classement compact des modeles.

    Entree:
        classement: lignes classees renvoyees par le benchmark.

    Sortie:
        Aucune sortie Python.
    """

    if not classement:
        print("No ranking generated")
        return

    print("\nRanking")
    print("-" * 80)
    header = f"{'rank':>4} {'model':<18} {'acc':>8} {'comm32_mb':>12} {'time_s':>10} {'score':>8}"
    print(header)
    print("-" * len(header))
    for ligne in classement:
        print(
            f"{int(ligne['rank']):>4} "
            f"{str(ligne['name']):<18} "
            f"{float(ligne['test_accuracy']) * 100:>7.2f}% "
            f"{float(ligne['float32_mb']):>12.2f} "
            f"{float(ligne['elapsed_seconds']):>10.1f} "
            f"{float(ligne['final_score']):>8.3f}"
        )
    print("-" * 80)


def main() -> None:
    """Point d'entree du script terminal.

    Entree:
        Arguments tapes dans le terminal.

    Sortie:
        Aucune sortie Python. Les XLSX du benchmark sont sauvegardes dans ``results``.
    """

    # Etape 1: lire les modeles et parametres a comparer.
    parser = build_parser()
    args = parser.parse_args()

    # Etape 2: afficher un resume avant de charger des modeles.
    print_config(args)

    # Etape 3: importer le code Transformers/PyTorch seulement apres le parsing.
    from src.hf_benchmark import BenchmarkConfig, run_hf_benchmark

    # Etape 4: mettre tous les parametres dans un objet de configuration .
    config = BenchmarkConfig(
        cles_modeles=args.models,
        taille_sous_ensemble_train=args.train_subset_size,
        taille_sous_ensemble_test=args.test_subset_size,
        taille_batch=args.batch_size,
        nombre_epochs=args.epochs,
        taux_apprentissage=args.learning_rate,
        geler_backbone=args.freeze_backbone,
        graine=args.seed,
    )

    # Etape 5: lancer le benchmark puis afficher le classement final.
    _, classement = run_hf_benchmark(config)
    print_ranking(classement)


if __name__ == "__main__":
    main()
