"""Benchmark de modeles Hugging Face pretrained sur Fashion-MNIST.



Le benchmark mesure :

- la performance de classification;
- le nombre de parametres;
- le temps d'entrainement;
- la taille estimee d'une mise a jour en float32, float16 et int8.
"""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.xlsx_tools import ecrire_dicts_xlsx


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

INFOS_MODELES = {
    "resnet18": {
        "name": "ResNet-18",
        "model_id": "microsoft/resnet-18",
        "type": "Classic CNN",
        "why": "Light Hugging Face baseline, known and fast.",
        "advantages": "Light, fast, low communication cost.",
        "limits": "Older architecture, sometimes weaker than modern models.",
    },
    "convnext_tiny": {
        "name": "ConvNeXT-Tiny",
        "model_id": "facebook/convnext-tiny-224",
        "type": "Modern CNN",
        "why": "Modern CNN to test a better performance/cost compromise.",
        "advantages": "Often stronger than classic CNNs.",
        "limits": "Heavier and slower than ResNet-18.",
    },
    "vit_base": {
        "name": "ViT-Base",
        "model_id": "google/vit-base-patch16-224-in21k",
        "type": "Vision Transformer",
        "why": "Different and heavier architecture for comparison.",
        "advantages": "Strong representation capacity at large scale.",
        "limits": "Heavy, slower, less convenient for a small local GPU.",
    },
}


@dataclass
class BenchmarkConfig:
    """Configuration du benchmark Hugging Face.

    Entrees:
        cles_modeles: cles des modeles dans `INFOS_MODELES`.
        taille_sous_ensemble_train: taille optionnelle du sous-ensemble train.
        taille_sous_ensemble_test: taille optionnelle du sous-ensemble test.
        taille_batch: taille des mini-batchs.
        nombre_epochs: nombre d'epochs de fine-tuning.
        taux_apprentissage: taux d'apprentissage AdamW.
        geler_backbone: si True, on entraine seulement la tete/classifier.
        graine: graine aleatoire.

    Sortie:
        Objet de configuration utilise par les fonctions de benchmark.
    """

    cles_modeles: list[str]
    taille_sous_ensemble_train: int | None = None
    taille_sous_ensemble_test: int | None = None
    taille_batch: int = 16
    nombre_epochs: int = 3
    taux_apprentissage: float = 3e-5
    geler_backbone: bool = True
    graine: int = 1


def find_project_root() -> Path:
    """Trouve la racine du projet.

    Entree:
        Aucune entree.

    Sortie:
        Chemin racine du projet.
    """

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return current


def reset_seed(graine: int) -> None:
    """Fixe les generateurs aleatoires.

    Entree:
        graine: graine aleatoire entiere.

    Sortie:
        Aucune sortie.
    """

    random.seed(graine)
    torch.manual_seed(graine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(graine)


def get_device() -> torch.device:
    """Choisit GPU si possible, sinon CPU.

    Entree:
        Aucune entree.

    Sortie:
        Device PyTorch.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recuperer_taille_image_processor(processor: object, taille_defaut: int = 224) -> int:
    """Lit la taille d'image attendue par un image processor Hugging Face.

    Entrees:
        processor: image processor Hugging Face.
        taille_defaut: taille par defaut si l'information manque.

    Sortie:
        Taille d'image entiere.
    """

    taille_crop = getattr(processor, "crop_size", None)
    if isinstance(taille_crop, dict):
        if "height" in taille_crop:
            return int(taille_crop["height"])
        if "width" in taille_crop:
            return int(taille_crop["width"])

    taille = getattr(processor, "size", taille_defaut)
    if isinstance(taille, dict):
        return int(taille.get("height") or taille.get("width") or taille.get("shortest_edge") or taille_defaut)
    if isinstance(taille, int):
        return taille
    return taille_defaut


def construire_transformation(processor: object) -> transforms.Compose:
    """Construit le preprocessing pour un modele pretrained.

    Entree:
        processor: image processor Hugging Face.

    Sortie:
        Transform torchvision compatible avec le modele.
    """

    taille_image = recuperer_taille_image_processor(processor)
    moyenne = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    ecart_type = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((taille_image, taille_image)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=moyenne, std=ecart_type),
        ]
    )


def selectionner_indices(taille_dataset: int, max_exemples: int | None, graine: int) -> list[int]:
    """Selectionne des indices aleatoires reproductibles.

    Entrees:
        taille_dataset: nombre total d'exemples.
        max_exemples: limite optionnelle.
        graine: graine aleatoire.

    Sortie:
        Liste d'indices selectionnes.
    """

    generateur = torch.Generator().manual_seed(graine)
    indices = torch.randperm(taille_dataset, generator=generateur).tolist()
    if max_exemples is not None:
        indices = indices[:max_exemples]
    return indices


def construire_dataloaders(
    dossier_donnees: Path,
    processor: object,
    indices_train: list[int],
    indices_test: list[int],
    taille_batch: int,
) -> tuple[DataLoader, DataLoader]:
    """Construit les DataLoaders train et test pour un modele.

    Entrees:
        dossier_donnees: dossier des donnees.
        processor: image processor Hugging Face.
        indices_train: indices du sous-ensemble train.
        indices_test: indices du sous-ensemble test.
        taille_batch: taille des batchs.

    Sortie:
        Tuple `(train_loader, test_loader)`.
    """

    transformation = construire_transformation(processor)
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
    chargeur_train = DataLoader(Subset(donnees_train, indices_train), batch_size=taille_batch, shuffle=True, num_workers=0)
    chargeur_test = DataLoader(Subset(donnees_test, indices_test), batch_size=taille_batch, shuffle=False, num_workers=0)
    return chargeur_train, chargeur_test


def geler_backbone_garder_classifieur(modele: torch.nn.Module) -> None:
    """Gele le backbone et garde la tete entrainable.

    Entree:
        modele: modele Hugging Face de classification d'images.

    Sortie:
        Aucune sortie. Les flags `requires_grad` sont modifies.

    Signification:
        Cette fonction transforme le benchmark en transfert learning. Le modele
        garde ses representations pretrained et adapte seulement sa sortie.
    """

    for parametre in modele.parameters():
        parametre.requires_grad = False

    mots_cles_tete = ["classifier", "head", "fc"]
    tete_trouvee = False
    for nom, parametre in modele.named_parameters():
        if any(mot_cle in nom.lower() for mot_cle in mots_cles_tete):
            parametre.requires_grad = True
            tete_trouvee = True

    if not tete_trouvee:
        parametres = list(modele.named_parameters())
        for _, parametre in parametres[-4:]:
            parametre.requires_grad = True


def compter_parametres(modele: torch.nn.Module) -> tuple[int, int]:
    """Compte les parametres totaux et entrainables.

    Entree:
        modele: modele PyTorch.

    Sortie:
        Tuple `(total_params, trainable_params)`.
    """

    total = sum(parametre.numel() for parametre in modele.parameters())
    entrainables = sum(parametre.numel() for parametre in modele.parameters() if parametre.requires_grad)
    return total, entrainables


def tailles_communication_mb(nombre_parametres: int) -> dict[str, float]:
    """Estime la taille d'une mise a jour dense.

    Entree:
        nombre_parametres: nombre de parametres communiques.

    Sortie:
        Dictionnaire avec les tailles float32, float16 et int8 en Mo.
    """

    return {
        "float32_mb": nombre_parametres * 4 / (1024**2),
        "float16_mb": nombre_parametres * 2 / (1024**2),
        "int8_mb": nombre_parametres / (1024**2),
    }


def entrainer_une_epoch(
    modele: torch.nn.Module,
    chargeur_train: DataLoader,
    optimiseur: torch.optim.Optimizer,
    materiel: torch.device,
) -> float:
    """Entraine un modele pendant une epoch.

    Entrees:
        modele: modele Hugging Face.
        chargeur_train: batchs d'entrainement.
        optimiseur: optimiseur.
        materiel: CPU ou GPU.

    Sortie:
        Loss moyenne d'entrainement.
    """

    modele.train()
    somme_loss = 0.0
    total_exemples = 0

    for images, labels in chargeur_train:
        images = images.to(materiel)
        labels = labels.to(materiel)

        optimiseur.zero_grad()
        sorties = modele(pixel_values=images, labels=labels)
        loss = sorties.loss
        loss.backward()
        optimiseur.step()

        taille_batch = labels.size(0)
        somme_loss += float(loss.item()) * taille_batch
        total_exemples += taille_batch

    return somme_loss / total_exemples


@torch.no_grad()
def evaluer(
    modele: torch.nn.Module,
    chargeur_test: DataLoader,
    materiel: torch.device,
) -> dict[str, float]:
    """Evalue un modele sur le test.

    Entrees:
        modele: modele a evaluer.
        chargeur_test: batchs de test.
        materiel: CPU ou GPU.

    Sortie:
        Dictionnaire contenant test loss et accuracy.
    """

    modele.eval()
    somme_loss = 0.0
    total_exemples = 0
    bonnes_predictions = 0

    for images, labels in chargeur_test:
        images = images.to(materiel)
        labels = labels.to(materiel)

        sorties = modele(pixel_values=images, labels=labels)
        loss = sorties.loss
        predictions = sorties.logits.argmax(dim=1)

        taille_batch = labels.size(0)
        somme_loss += float(loss.item()) * taille_batch
        total_exemples += taille_batch
        bonnes_predictions += int((predictions == labels).sum().item())

    return {
        "test_loss": somme_loss / total_exemples,
        "test_accuracy": bonnes_predictions / total_exemples,
    }


def lancer_benchmark_modele(
    cle_modele: str,
    config: BenchmarkConfig,
    indices_train: list[int],
    indices_test: list[int],
    dossier_donnees: Path,
    materiel: torch.device,
) -> dict[str, object]:
    """Lance le benchmark pour un modele pretrained.

    Entrees:
        cle_modele: cle dans `INFOS_MODELES`.
        config: configuration du benchmark.
        indices_train: indices train.
        indices_test: indices test.
        dossier_donnees: dossier des donnees.
        materiel: CPU ou GPU.

    Sortie:
        Dictionnaire de resultats pour ce modele.
    """

    infos_modele = INFOS_MODELES[cle_modele]
    id_modele = infos_modele["model_id"]
    label_vers_id = {nom: index for index, nom in enumerate(NOMS_CLASSES)}
    id_vers_label = {index: nom for index, nom in enumerate(NOMS_CLASSES)}

    print("\n" + "=" * 80)
    print("Model:", infos_modele["name"])
    print("Hugging Face ID:", id_modele)
    print("Why:", infos_modele["why"])

    processor = AutoImageProcessor.from_pretrained(id_modele)
    chargeur_train, chargeur_test = construire_dataloaders(
        dossier_donnees=dossier_donnees,
        processor=processor,
        indices_train=indices_train,
        indices_test=indices_test,
        taille_batch=config.taille_batch,
    )

    modele = AutoModelForImageClassification.from_pretrained(
        id_modele,
        num_labels=len(NOMS_CLASSES),
        id2label=id_vers_label,
        label2id=label_vers_id,
        ignore_mismatched_sizes=True,
    )

    if config.geler_backbone:
        geler_backbone_garder_classifieur(modele)

    modele = modele.to(materiel)
    total_parametres, parametres_entrainables = compter_parametres(modele)
    print("Total parameters:", total_parametres)
    print("Trainable parameters:", parametres_entrainables)

    optimiseur = torch.optim.AdamW(
        [parametre for parametre in modele.parameters() if parametre.requires_grad],
        lr=config.taux_apprentissage,
    )

    debut = time.perf_counter()
    losses_train = []
    for epoch in range(config.nombre_epochs):
        loss_train = entrainer_une_epoch(modele, chargeur_train, optimiseur, materiel)
        losses_train.append(loss_train)
        print(f"Epoch {epoch + 1}/{config.nombre_epochs} - train_loss={loss_train:.4f}")

    metriques = evaluer(modele, chargeur_test, materiel)
    duree = time.perf_counter() - debut
    tailles_communication = tailles_communication_mb(total_parametres)

    resultat = {
        "key": cle_modele,
        "name": infos_modele["name"],
        "model_id": id_modele,
        "type": infos_modele["type"],
        "why": infos_modele["why"],
        "advantages": infos_modele["advantages"],
        "limits": infos_modele["limits"],
        "num_epochs": config.nombre_epochs,
        "train_examples": len(chargeur_train.dataset),
        "test_examples": len(chargeur_test.dataset),
        "total_params": total_parametres,
        "trainable_params": parametres_entrainables,
        "last_train_loss": losses_train[-1],
        "test_loss": metriques["test_loss"],
        "test_accuracy": metriques["test_accuracy"],
        "elapsed_seconds": duree,
        **tailles_communication,
    }

    del modele
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return resultat


def normalize_higher_is_better(valeurs: list[float]) -> list[float]:
    """Normalise des valeurs quand une grande valeur est meilleure.

    Entree:
        valeurs: valeurs numeriques a comparer.

    Sortie:
        Scores compris entre 0 et 1.
    """

    valeur_min = min(valeurs)
    valeur_max = max(valeurs)
    if valeur_max == valeur_min:
        return [1.0 for _ in valeurs]
    return [(valeur - valeur_min) / (valeur_max - valeur_min) for valeur in valeurs]


def normalize_lower_is_better(valeurs: list[float]) -> list[float]:
    """Normalise des valeurs quand une petite valeur est meilleure.

    Entree:
        valeurs: valeurs numeriques a comparer.

    Sortie:
        Scores compris entre 0 et 1.
    """

    valeur_min = min(valeurs)
    valeur_max = max(valeurs)
    if valeur_max == valeur_min:
        return [1.0 for _ in valeurs]
    return [1 - ((valeur - valeur_min) / (valeur_max - valeur_min)) for valeur in valeurs]


def construire_classement(resultats: list[dict[str, object]]) -> list[dict[str, object]]:
    """Construit un classement accuracy/communication/temps.

    Entree:
        resultats: lignes de resultats du benchmark.

    Sortie:
        Lignes classees avec un score final.

    Signification:
        Le score final donne plus de poids a l'accuracy, puis au cout de
        communication, puis au temps. Il sert a choisir un modele pratique,
        pas seulement le modele le plus precis.
    """

    lignes_valides = [
        ligne
        for ligne in resultats
        if ligne.get("test_accuracy") is not None
        and ligne.get("float32_mb") is not None
        and ligne.get("elapsed_seconds") is not None
        and ligne.get("error") is None
    ]
    if not lignes_valides:
        return []

    scores_accuracy = normalize_higher_is_better([float(ligne["test_accuracy"]) for ligne in lignes_valides])
    scores_communication = normalize_lower_is_better([float(ligne["float32_mb"]) for ligne in lignes_valides])
    scores_temps = normalize_lower_is_better([float(ligne["elapsed_seconds"]) for ligne in lignes_valides])

    classement = []
    for ligne, score_accuracy, score_communication, score_temps in zip(
        lignes_valides,
        scores_accuracy,
        scores_communication,
        scores_temps,
    ):
        ligne_classee = dict(ligne)
        ligne_classee["accuracy_score"] = score_accuracy
        ligne_classee["communication_score"] = score_communication
        ligne_classee["speed_score"] = score_temps
        ligne_classee["final_score"] = 0.50 * score_accuracy + 0.30 * score_communication + 0.20 * score_temps
        classement.append(ligne_classee)

    classement.sort(key=lambda item: float(item["final_score"]), reverse=True)
    for rang, ligne in enumerate(classement, start=1):
        ligne["rank"] = rang
    return classement


def ecrire_xlsx_resultats(chemin: Path, lignes: list[dict[str, object]], nom_feuille: str) -> None:
    """Ecrit des resultats dans un fichier XLSX.

    Entrees:
        chemin: fichier de destination.
        lignes: lignes a ecrire.
        nom_feuille: nom de l'onglet Excel.

    Sortie:
        Aucune sortie Python. Le fichier est cree ou remplace.
    """

    noms_colonnes = []
    for ligne in lignes:
        for cle in ligne:
            if cle not in noms_colonnes:
                noms_colonnes.append(cle)

    ecrire_dicts_xlsx(chemin, lignes, noms_colonnes, nom_feuille)


def run_hf_benchmark(config: BenchmarkConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Lance le benchmark complet des modeles Hugging Face.

    Entree:
        config: configuration du benchmark.

    Sortie:
        Tuple ``(resultats, classement)`` :
        resultats bruts et classement final.
    """

    racine_projet = find_project_root()
    dossier_donnees = racine_projet / "data"
    dossier_resultats = racine_projet / "results"
    materiel = get_device()

    # On fixe la graine pour que le sous-echantillonnage et l'entrainement
    # soient reproductibles d'une execution a l'autre.
    reset_seed(config.graine)
    print("Device:", materiel)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    transformation_base = transforms.ToTensor()
    donnees_train_base = torchvision.datasets.FashionMNIST(
        root=str(dossier_donnees),
        train=True,
        download=True,
        transform=transformation_base,
    )
    donnees_test_base = torchvision.datasets.FashionMNIST(
        root=str(dossier_donnees),
        train=False,
        download=True,
        transform=transformation_base,
    )

    # Meme logique que dans les simulations federes : on peut limiter le nombre
    # d'images pour garder un benchmark rapide sur une machine personnelle.
    indices_train = selectionner_indices(len(donnees_train_base), config.taille_sous_ensemble_train, config.graine)
    indices_test = selectionner_indices(len(donnees_test_base), config.taille_sous_ensemble_test, config.graine)
    print("Train examples:", len(indices_train))
    print("Test examples:", len(indices_test))

    resultats = []
    for cle_modele in config.cles_modeles:
        try:
            # Chaque modele est teste separement pour liberer la memoire GPU
            # entre deux architectures.
            resultat = lancer_benchmark_modele(cle_modele, config, indices_train, indices_test, dossier_donnees, materiel)
            resultats.append(resultat)
        except Exception as exc:
            infos_modele = INFOS_MODELES[cle_modele]
            print(f"Error with {infos_modele['name']}: {type(exc).__name__}: {exc}")
            resultats.append(
                {
                    "key": cle_modele,
                    "name": infos_modele["name"],
                    "model_id": infos_modele["model_id"],
                    "type": infos_modele["type"],
                    "why": infos_modele["why"],
                    "advantages": infos_modele["advantages"],
                    "limits": infos_modele["limits"],
                    "error": str(exc),
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    classement = construire_classement(resultats)
    ecrire_xlsx_resultats(dossier_resultats / "hf_model_benchmark_fashion_mnist.xlsx", resultats, "Benchmark")
    if classement:
        ecrire_xlsx_resultats(dossier_resultats / "hf_model_benchmark_ranking.xlsx", classement, "Ranking")

    return resultats, classement
