

from __future__ import annotations

import argparse

from common_cli import entier_optionnel, liste_virgules_validee, preparer_imports_projet


PROJECT_ROOT = preparer_imports_projet()
SCENARIOS_VALIDES = {"none", "float16", "topk_50", "topk_20", "topk_10"}


def parse_scenarios(value: str) -> list[str]:
    return liste_virgules_validee(value, SCENARIOS_VALIDES, "scenario")


def build_parser() -> argparse.ArgumentParser:
    """Cree le parseur des arguments terminal.

    Entree:
        Aucune entree.

    Sortie:
        Objet ``argparse.ArgumentParser`` configure.
    """

    parser = argparse.ArgumentParser(description="Lance une experience FedCompress avec le petit CNN.")
    parser.add_argument("--experiment", required=True, help="Nom de l'experience utilise pour les fichiers de sortie.")
    parser.add_argument("--partition", choices=["iid", "non_iid"], default="iid")
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=["none"],
        help="Scenarios separes par des virgules : none,float16,topk_50,topk_20,topk_10.",
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-train-examples", type=entier_optionnel, default=None)
    parser.add_argument("--max-test-examples", type=entier_optionnel, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Affiche les courbes matplotlib apres l'entrainement.")
    return parser


def print_config(args: argparse.Namespace) -> None:
    """Affiche les parametres avant le debut de l'entrainement.

    Entree:
        args: arguments lus depuis le terminal.

    Sortie:
        Aucune sortie Python. Le texte est affiche dans le terminal.
    """

    print("\n" + "=" * 80)
    print("FedCompress - small CNN experiment")
    print("=" * 80)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Experiment name     : {args.experiment}")
    print(f"Partition           : {args.partition}")
    print(f"Scenarios           : {', '.join(args.scenarios)}")
    print(f"Clients             : {args.clients}")
    print(f"Rounds              : {args.rounds}")
    print(f"Local epochs        : {args.local_epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Max train examples  : {args.max_train_examples}")
    print(f"Max test examples   : {args.max_test_examples}")
    print(f"Seed                : {args.seed}")
    print("=" * 80 + "\n")


def print_summary(summary: list[dict[str, object]]) -> None:
    """Affiche un petit tableau final des resultats.

    Entree:
        summary: lignes resumees renvoyees par l'experience.

    Sortie:
        Aucune sortie Python. Le tableau est affiche dans le terminal.
    """

    if not summary:
        print("No summary generated")
        return

    print("\nFinal summary")
    print("-" * 80)
    header = f"{'scenario':<12} {'accuracy':>10} {'loss':>10} {'comm_mb':>10} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{str(row['scenario']):<12} "
            f"{float(row['final_accuracy']) * 100:>9.2f}% "
            f"{float(row['final_loss']):>10.4f} "
            f"{float(row['communication_total_mb']):>10.3f} "
            f"{float(row['total_seconds']):>10.1f}"
        )
    print("-" * 80)


def main() -> None:
    """Point d'entree du script terminal.

    Entree:
        Arguments tapes dans le terminal.

    Sortie:
        Aucune sortie Python. Les resultats sont affiches et sauvegardes en XLSX.
    """

    # Etape 1: lire les parametres tapes par l'utilisateur dans le terminal.
    parser = build_parser()
    args = parser.parse_args()

    # Etape 2: afficher la configuration avant de lancer un calcul couteux.
    print_config(args)

    # Etape 3: importer le code PyTorch seulement apres le parsing. Cela rend
    # les commandes simples comme --help plus rapides.
    from src.fedcompress_cnn import CnnExperimentConfig, plot_results, run_cnn_fedavg_experiment

    # Etape 4: construire un objet de configuration clair. Tous les parametres
    # importants de l'experience sont regroupes au meme endroit.
    config = CnnExperimentConfig(
        nom_experience=args.experiment,
        scenarios=args.scenarios,
        repartition=args.partition,
        nombre_clients=args.clients,
        nombre_rounds=args.rounds,
        epochs_locales=args.local_epochs,
        taille_batch=args.batch_size,
        taux_apprentissage=args.learning_rate,
        max_exemples_train=args.max_train_examples,
        max_exemples_test=args.max_test_examples,
        graine=args.seed,
    )

    # Etape 5: lancer FedAvg, ecrire les XLSX puis afficher un resume court.
    round_rows, summary, _ = run_cnn_fedavg_experiment(config)
    print_summary(summary)

    # Etape 6: le plot est optionnel, car certains terminaux n'ont pas
    # d'affichage graphique.
    if args.plot:
        plot_results(round_rows, title=args.experiment)


if __name__ == "__main__":
    main()
