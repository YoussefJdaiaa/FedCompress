

from __future__ import annotations

import argparse

from common_cli import entier_optionnel, filtre_virgules_ou_aucun, preparer_imports_projet


PROJECT_ROOT = preparer_imports_projet()

EXPERIENCES_STANDARD = [
    {
        "name": "iid_baseline",
        "partition": "iid",
        "scenarios": ["none"],
        "rounds": 20,
    },
    {
        "name": "iid_float16",
        "partition": "iid",
        "scenarios": ["none", "float16"],
        "rounds": 20,
    },
    {
        "name": "iid_sparsification",
        "partition": "iid",
        "scenarios": ["none", "topk_50", "topk_20", "topk_10"],
        "rounds": 20,
    },
    {
        "name": "non_iid",
        "partition": "non_iid",
        "scenarios": ["none", "float16", "topk_20", "topk_10"],
        "rounds": 20,
    },
    {
        "name": "non_iid_20rounds",
        "partition": "non_iid",
        "scenarios": ["none", "float16", "topk_20", "topk_10"],
        "rounds": 20,
    },
]


def selected_experiments(args: argparse.Namespace) -> list[dict[str, object]]:
    """Selectionne les experiences demandees par l'utilisateur.

    Entree:
        args: arguments lus depuis le terminal.

    Sortie:
        Liste de dictionnaires d'experiences.
    """

    seulement = filtre_virgules_ou_aucun(args.only)
    experiences_selectionnees = []
    for experience in EXPERIENCES_STANDARD:
        if args.skip_non_iid and str(experience["partition"]) == "non_iid":
            continue
        if seulement and str(experience["name"]) not in seulement:
            continue
        experiences_selectionnees.append(experience)
    return experiences_selectionnees


def build_parser() -> argparse.ArgumentParser:
    """Cree le parseur des arguments terminal.

    Entree:
        Aucune entree.

    Sortie:
        Parseur configure.
    """

    parser = argparse.ArgumentParser(description="Lance la suite standard FedCompress avec le petit CNN.")
    parser.add_argument("--only", default="", help="Noms d'experiences a lancer, separes par des virgules.")
    parser.add_argument("--skip-non-iid", action="store_true", help="Ignore les experiences non-IID.")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-train-examples", type=entier_optionnel, default=None)
    parser.add_argument("--max-test-examples", type=entier_optionnel, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Affiche le plan sans lancer l'entrainement.")
    return parser


def print_plan(args: argparse.Namespace, experiences: list[dict[str, object]]) -> None:
    """Affiche le plan de la suite d'experiences.

    Entree:
        args: arguments lus depuis le terminal.
        experiences: experiences selectionnees.

    Sortie:
        Aucune sortie Python. Le plan est affiche dans le terminal.
    """

    print("\n" + "=" * 80)
    print("FedCompress - standard small CNN suite")
    print("=" * 80)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Experiments         : {', '.join(str(experience['name']) for experience in experiences)}")
    print(f"Clients             : {args.clients}")
    print(f"Local epochs        : {args.local_epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Max train examples  : {args.max_train_examples}")
    print(f"Max test examples   : {args.max_test_examples}")
    print(f"Seed                : {args.seed}")
    print(f"Dry run             : {args.dry_run}")
    print("=" * 80)
    for experience in experiences:
        print(
            f"- {str(experience['name']):<22} "
            f"partition={str(experience['partition']):<7} "
            f"rounds={experience['rounds']:<2} "
            f"scenarios={','.join(experience['scenarios'])}"
        )
    print("=" * 80 + "\n")


def main() -> None:
    """Point d'entree du script terminal.

    Entree:
        Arguments tapes dans le terminal.

    Sortie:
        Aucune sortie Python. Chaque experience ecrit des XLSX dans ``results``.
    """

    # Etape 1: lire les options avant d'importer le code d'entrainement.
    parser = build_parser()
    args = parser.parse_args()

    # Etape 2: choisir les experiences standard a lancer.
    experiences = selected_experiments(args)
    if not experiences:
        raise SystemExit("No experiment selected")

    # Etape 3: afficher un plan clair pour verifier avant le lancement.
    print_plan(args, experiences)
    if args.dry_run:
        return

    # Etape 4: importer le code d'entrainement seulement si on lance vraiment.
    from src.fedcompress_cnn import CnnExperimentConfig, run_cnn_fedavg_experiment

    # Etape 5: lancer les experiences une par une. Cela garde les sorties
    # lisibles et facilite le suivi d'un long calcul.
    for experience in experiences:
        print("\n" + "#" * 80)
        print(f"Running experiment={experience['name']}")
        print("#" * 80)
        config = CnnExperimentConfig(
            nom_experience=str(experience["name"]),
            scenarios=list(experience["scenarios"]),
            repartition=str(experience["partition"]),
            nombre_clients=args.clients,
            nombre_rounds=int(experience["rounds"]),
            epochs_locales=args.local_epochs,
            taille_batch=args.batch_size,
            taux_apprentissage=args.learning_rate,
            max_exemples_train=args.max_train_examples,
            max_exemples_test=args.max_test_examples,
            graine=args.seed,
        )
        run_cnn_fedavg_experiment(config)


if __name__ == "__main__":
    main()
