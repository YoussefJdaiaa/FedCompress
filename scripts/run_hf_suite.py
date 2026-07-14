

import argparse

from common_cli import entier_optionnel, filtre_virgules_ou_aucun, preparer_imports_projet


PROJECT_ROOT = preparer_imports_projet()

CONFIGS_MODELES = {
    "resnet18": "microsoft/resnet-18",
    "convnext_tiny": "facebook/convnext-tiny-224",
}


EXPERIENCES_STANDARD = [
    {
        "name": "iid_baseline",
        "partition": "iid",
        "scenarios": ["none"],
        "rounds": 3,
    },
    {
        "name": "iid_float16",
        "partition": "iid",
        "scenarios": ["none", "float16"],
        "rounds": 3,
    },
    {
        "name": "iid_sparsification",
        "partition": "iid",
        "scenarios": ["none", "topk_50", "topk_20", "topk_10"],
        "rounds": 3,
    },
    {
        "name": "non_iid",
        "partition": "non_iid",
        "scenarios": ["none", "float16", "topk_20", "topk_10"],
        "rounds": 3,
    },
    {
        "name": "non_iid_20rounds",
        "partition": "non_iid",
        "scenarios": ["none", "float16", "topk_20", "topk_10"],
        "rounds": 20,
    },
]


def selected_models(value):
    """Transforme le choix de modele en liste de modeles.

    Entree:
        value: ``resnet18``, ``convnext_tiny`` ou ``both``.

    Sortie:
        Liste des cles de modeles.
    """

    if value == "both":
        return ["resnet18", "convnext_tiny"]
    return [value]


def selected_experiments(args):
    """Selectionne les experiences de la suite.

    Entree:
        args: arguments lus depuis le terminal.

    Sortie:
        Liste de dictionnaires d'experiences.
    """

    seulement = filtre_virgules_ou_aucun(args.only)
    experiences = []
    for experience in EXPERIENCES_STANDARD:
        if args.skip_long and experience["rounds"] > 3:
            continue
        if seulement and experience["name"] not in seulement:
            continue
        experiences.append(experience)
    return experiences


def print_plan(args, modeles, experiences):
    """Affiche le plan d'execution de la suite.

    Entrees:
        args: arguments lus depuis le terminal.
        modeles: cles des modeles selectionnes.
        experiences: experiences selectionnees.

    Sortie:
        Aucune sortie Python. Le plan est affiche dans le terminal.
    """

    print("\n" + "=" * 80)
    print("FedCompress - standard Hugging Face suite")
    print("=" * 80)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Models              : {', '.join(modeles)}")
    print(f"Experiments         : {', '.join(experience['name'] for experience in experiences)}")
    print(f"Clients             : {args.clients}")
    print(f"Local epochs        : {args.local_epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Max train examples  : {args.max_train_examples}")
    print(f"Max test examples   : {args.max_test_examples}")
    print(f"Seed                : {args.seed}")
    print(f"Freeze backbone     : {args.freeze_backbone}")
    print(f"State scope         : {args.state_scope}")
    print(f"Dry run             : {args.dry_run}")
    print("=" * 80)
    for modele in modeles:
        id_modele = CONFIGS_MODELES[modele]
        print(f"\nModel: {modele} ({id_modele})")
        for experience in experiences:
            scenarios = ",".join(experience["scenarios"])
            print(
                f"  - {experience['name']:<20} "
                f"partition={experience['partition']:<7} "
                f"rounds={experience['rounds']:<2} "
                f"scenarios={scenarios}"
            )
    print("=" * 80 + "\n")


def build_parser():
    """Cree le parseur des arguments terminal.

    Entree:
        Aucune entree.

    Sortie:
        Parseur configure.
    """

    parser = argparse.ArgumentParser(
        description="Lance la suite standard FedCompress avec Hugging Face."
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "convnext_tiny", "both"],
        required=True,
        help="Modele ou groupe de modeles a lancer.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Noms d'experiences a lancer, separes par des virgules.",
    )
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="Ignore les experiences longues de 20 rounds.",
    )
    parser.add_argument("--clients", type=int, default=5)
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
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Entraine tout le modele au lieu de geler le backbone.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan sans lancer les experiences.",
    )
    parser.set_defaults(freeze_backbone=True)
    return parser


def main():
    """Point d'entree du script terminal.

    Entree:
        Arguments tapes dans le terminal.

    Sortie:
        Aucune sortie Python. Chaque experience sauvegarde ses fichiers XLSX.
    """

    # Etape 1: lire les options sans importer PyTorch.
    parser = build_parser()
    args = parser.parse_args()

    # Etape 2: choisir les modeles et experiences a partir de la suite standard.
    modeles = selected_models(args.model)
    experiences = selected_experiments(args)

    if not experiences:
        raise SystemExit("No experiment selected")

    # Etape 3: afficher le plan avant de lancer des entrainements longs.
    print_plan(args, modeles, experiences)
    if args.dry_run:
        return

    # Etape 4: importer la fonction d'entrainement seulement si on lance vraiment.
    from src.fedcompress_hf import run_hf_fedavg_experiment

    # Etape 5: lancer les couples modele/experience un par un pour suivre la sortie.
    for modele in modeles:
        for experience in experiences:
            print("\n" + "#" * 80)
            print(f"Running model={modele} experiment={experience['name']}")
            print("#" * 80)
            run_hf_fedavg_experiment(
                cle_modele=modele,
                nom_experience=experience["name"],
                scenarios=experience["scenarios"],
                repartition=experience["partition"],
                nombre_clients=args.clients,
                nombre_rounds=experience["rounds"],
                epochs_locales=args.local_epochs,
                taille_batch=args.batch_size,
                taux_apprentissage=args.learning_rate,
                max_exemples_train=args.max_train_examples,
                max_exemples_test=args.max_test_examples,
                graine=args.seed,
                geler_backbone=args.freeze_backbone,
                portee_etat=args.state_scope,
            )


if __name__ == "__main__":
    main()
