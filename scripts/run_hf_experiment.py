

import argparse

from common_cli import entier_optionnel, liste_virgules_validee, preparer_imports_projet


PROJECT_ROOT = preparer_imports_projet()
SCENARIOS_VALIDES = {"none", "float16", "topk_50", "topk_20", "topk_10"}
CONFIGS_MODELES = {
    "resnet18": "microsoft/resnet-18",
    "convnext_tiny": "facebook/convnext-tiny-224",
}


def parse_scenarios(value):
    return liste_virgules_validee(value, SCENARIOS_VALIDES, "scenario")


def print_config(args):
    """Affiche la configuration de l'experience.

    Entree:
        args: arguments lus depuis le terminal.

    Sortie:
        Aucune sortie Python. Le texte est affiche dans le terminal.
    """

    print("\n" + "=" * 80)
    print("FedCompress - Hugging Face experiment")
    print("=" * 80)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Model key           : {args.model}")
    print(f"Model id            : {CONFIGS_MODELES[args.model]}")
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
    print(f"Freeze backbone     : {args.freeze_backbone}")
    print(f"State scope         : {args.state_scope}")
    if args.freeze_backbone and args.state_scope == "trainable":
        print("Mode                : federated transfer learning")
        print("Communicated params : classification head only")
    elif args.freeze_backbone and args.state_scope == "all":
        print("Mode                : frozen backbone, full state communication")
    else:
        print("Mode                : full model training")
    print("=" * 80 + "\n")


def print_summary(lignes_resume):
    """Affiche un tableau final compact.

    Entree:
        lignes_resume: lignes resumees par scenario.

    Sortie:
        Aucune sortie Python.
    """

    if not lignes_resume:
        print("No summary generated")
        return

    print("\nFinal summary")
    print("-" * 80)
    header = f"{'scenario':<12} {'accuracy':>10} {'loss':>10} {'comm_mb':>10} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    for ligne in lignes_resume:
        print(
            f"{ligne['scenario']:<12} "
            f"{ligne['final_accuracy'] * 100:>9.2f}% "
            f"{ligne['final_loss']:>10.4f} "
            f"{ligne['communication_total_mb']:>10.3f} "
            f"{ligne['total_seconds']:>10.1f}"
        )
    print("-" * 80)


def build_parser():
    """Cree le parseur des arguments terminal.

    Entree:
        Aucune entree.

    Sortie:
        Parseur configure.
    """

    parser = argparse.ArgumentParser(
        description="Lance une experience FedCompress federee avec Hugging Face."
    )
    parser.add_argument(
        "--model",
        choices=sorted(CONFIGS_MODELES),
        required=True,
        help="Modele a lancer.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Nom de l'experience utilise pour les fichiers XLSX.",
    )
    parser.add_argument(
        "--partition",
        choices=["iid", "non_iid"],
        default="iid",
        help="Partition des donnees entre clients.",
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=["none"],
        help="Scenarios separes par des virgules : none,float16,topk_50,topk_20,topk_10.",
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-train-examples", type=entier_optionnel, default=1000)
    parser.add_argument("--max-test-examples", type=entier_optionnel, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--state-scope",
        choices=["trainable", "all"],
        default="trainable",
        help="Parametres communiques par FedAvg.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Entraine tout le modele au lieu de geler le backbone.",
    )
    parser.set_defaults(freeze_backbone=True)
    return parser


def main():
    """Point d'entree du script terminal.

    Entree:
        Arguments tapes dans le terminal.

    Sortie:
        Aucune sortie Python. Les resultats sont affiches et sauvegardes en XLSX.
    """

    # Etape 1: lire les arguments avant de charger PyTorch. Cela rend --help
    # rapide et evite de charger les gros modules si on ne lance pas le train.
    parser = build_parser()
    args = parser.parse_args()

    # Etape 2: afficher clairement ce qui va etre lance avant l'experience.
    print_config(args)

    # Etape 3: importer la fonction d'entrainement seulement maintenant.
    from src.fedcompress_hf import run_hf_fedavg_experiment

    # Etape 4: executer FedAvg et sauvegarder les XLSX dans results/<model>.
    _, lignes_resume = run_hf_fedavg_experiment(
        cle_modele=args.model,
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
        geler_backbone=args.freeze_backbone,
        portee_etat=args.state_scope,
    )

    # Etape 5: afficher un petit tableau final dans le terminal.
    print_summary(lignes_resume)


if __name__ == "__main__":
    main()
