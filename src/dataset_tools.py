"""Outils dataset pour FedCompress.

Il sert a verifier
Fashion-MNIST avant de lancer les simulations :

- chargement des donnees;
- comptage des classes;
- resume train/test;
- sauvegarde d'une grille d'exemples.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms


NOMS_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def find_project_root() -> Path:
    """Trouve la racine du projet.

    Entree:
        Aucune entree.

    Sortie:
        Chemin du dossier racine.
    """

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return current


def charger_fashion_mnist(
    dossier_donnees: Path,
) -> tuple[torchvision.datasets.FashionMNIST, torchvision.datasets.FashionMNIST]:
    """Charge Fashion-MNIST sans preprocessing specifique a un modele.

    Entree:
        dossier_donnees: dossier ou torchvision stocke les donnees.

    Sortie:
        Tuple `(donnees_train, donnees_test)`.
    """

    transformation = transforms.ToTensor()
    donnees_train = torchvision.datasets.FashionMNIST(
        root=str(dossier_donnees),
        train=True,
        download=True,
        transform=transformation,
    )
    donnees_test = torchvision.datasets.FashionMNIST(
        root=str(dossier_donnees),
        train=False,
        download=True,
        transform=transformation,
    )
    return donnees_train, donnees_test


def compter_labels(donnees: torchvision.datasets.FashionMNIST) -> dict[str, int]:
    """Compte le nombre d'exemples par classe.

    Entree:
        donnees: dataset Fashion-MNIST.

    Sortie:
        Dictionnaire `class_name -> count`.
    """

    compteurs = Counter(int(label) for label in donnees.targets)
    return {NOMS_CLASSES[label]: compteurs[label] for label in sorted(compteurs)}


def resumer_dataset(
    donnees_train: torchvision.datasets.FashionMNIST,
    donnees_test: torchvision.datasets.FashionMNIST,
) -> dict[str, object]:
    """Construit un resume simple du dataset.

    Entrees:
        donnees_train: split train.
        donnees_test: split test.

    Sortie:
        Dictionnaire avec tailles des splits et repartition des classes.
    """

    image, _ = donnees_train[0]
    return {
        "train_examples": len(donnees_train),
        "test_examples": len(donnees_test),
        "image_shape": tuple(image.shape),
        "classes": NOMS_CLASSES,
        "train_label_counts": compter_labels(donnees_train),
        "test_label_counts": compter_labels(donnees_test),
    }


def sauvegarder_grille_exemples(
    donnees: torchvision.datasets.FashionMNIST,
    chemin_sortie: Path,
    nombre_images: int = 10,
) -> Path:
    """Sauvegarde une grille d'images exemples.

    Entrees:
        donnees: dataset source.
        chemin_sortie: chemin PNG de sortie.
        nombre_images: nombre d'images a afficher.

    Sortie:
        Chemin du fichier PNG sauvegarde.
    """

    import matplotlib.pyplot as plt

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    colonnes = min(5, nombre_images)
    lignes = (nombre_images + colonnes - 1) // colonnes

    plt.figure(figsize=(2 * colonnes, 2 * lignes))
    for indice in range(nombre_images):
        image, label = donnees[indice]
        plt.subplot(lignes, colonnes, indice + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(NOMS_CLASSES[label])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=160)
    plt.close()
    return chemin_sortie


def afficher_resume_dataset(resume: dict[str, object]) -> None:
    """Affiche un resume lisible du dataset.

    Entree:
        resume: sortie de `resumer_dataset`.

    Sortie:
        Aucune sortie. Le resume est affiche dans le terminal.
    """

    print("Fashion-MNIST summary")
    print("-" * 80)
    print("Train examples:", resume["train_examples"])
    print("Test examples :", resume["test_examples"])
    print("Image shape   :", resume["image_shape"])
    print("\nTrain label counts:")
    for nom_classe, compteur in dict(resume["train_label_counts"]).items():
        print(f"  {nom_classe:<12} {compteur}")
    print("\nTest label counts:")
    for nom_classe, compteur in dict(resume["test_label_counts"]).items():
        print(f"  {nom_classe:<12} {compteur}")
