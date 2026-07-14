"""Experiences FedAvg avec un petit CNN pour FedCompress.



Le fichier regroupe les etapes importantes :

- chargement de Fashion-MNIST;
- decoupage des donnees entre clients IID ou non-IID;
- entrainement local d'un modele par client;
- aggregation FedAvg sur le serveur;
- compression float16 ou sparsification top-k;
- sauvegarde des resultats par round et des resumes en XLSX.

La fonction principale est `run_cnn_fedavg_experiment`.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

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

SCENARIOS_VALIDES = {"none", "float16", "topk_50", "topk_20", "topk_10"}
OCTETS_VALEUR = 4
OCTETS_INDICE = 4


@dataclass
class CnnExperimentConfig:
    """Configuration d'une experience FedAvg avec le petit CNN.

    Entrees:
        nom_experience: nom utilise pour choisir les fichiers XLSX de sortie.
        scenarios: compressions a tester, par exemple ["none", "float16"].
        repartition: repartition des clients, `iid` ou `non_iid`.
        nombre_clients: nombre de clients federes.
        nombre_rounds: nombre de rounds FedAvg.
        epochs_locales: nombre d'epochs locales par client et par round.
        taille_batch: taille des mini-batchs.
        taux_apprentissage: taux d'apprentissage SGD.
        max_exemples_train: limite optionnelle du train, `None` signifie tout.
        max_exemples_test: limite optionnelle du test, `None` signifie tout.
        graine: graine aleatoire pour reproduire les tirages.

    Sortie:
        Objet de configuration transmis aux fonctions d'entrainement.
    """

    nom_experience: str
    scenarios: list[str]
    repartition: str = "iid"
    nombre_clients: int = 5
    nombre_rounds: int = 20
    epochs_locales: int = 1
    taille_batch: int = 64
    taux_apprentissage: float = 0.01
    max_exemples_train: int | None = None
    max_exemples_test: int | None = None
    graine: int = 1


@dataclass
class SavedPaths:
    """Chemins des fichiers produits par une experience.

    Entrees:
        rounds_xlsx: XLSX avec une ligne par scenario et par round.
        summary_xlsx: XLSX de resume, parfois absent pour garder la compatibilite
        avec les anciens notebooks.

    Sortie:
        Petit objet retourne par la fonction de sauvegarde.
    """

    rounds_xlsx: Path
    summary_xlsx: Path | None


class SmallCNN(nn.Module):
    """Petit reseau convolutionnel utilise comme baseline.

    Entree:
        Tenseur image de forme `[batch, 1, 28, 28]`.

    Sortie:
        Logits de classification de forme `[batch, 10]`.

    Role:
        Deux blocs convolutionnels extraient des caracteristiques simples. Deux
        couches fully connected classent ensuite l'image Fashion-MNIST.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Effectue un passage avant du modele.

        Entree:
            x: batch d'images normalisees en niveaux de gris.

        Sortie:
            Logits bruts avant softmax.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def find_project_root() -> Path:
    """Trouve la racine du projet.

    Entree:
        Aucune entree directe. La recherche commence depuis `Path.cwd()`.

    Sortie:
        Chemin du dossier racine du projet.

    Role:
        Remonte les dossiers jusqu'a trouver celui qui contient `README.md` et
        `src`.
    """

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return current


def reset_seed(seed: int) -> None:
    """Fixe les graines aleatoires.

    Entree:
        seed: graine entiere.

    Sortie:
        Aucune sortie. Les generateurs aleatoires sont modifies en place.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Choisit le materiel de calcul.

    Entree:
        Aucune entree.

    Sortie:
        `torch.device("cuda")` si un GPU existe, sinon `torch.device("cpu")`.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(materiel: torch.device | None = None) -> SmallCNN:
    """Cree un nouveau modele SmallCNN.

    Entree:
        materiel: device PyTorch optionnel. Si une valeur est donnee, le
        modele est deplace sur ce materiel.

    Sortie:
        Nouvelle instance ``SmallCNN`` avec des poids initialises au hasard.
    """

    modele = SmallCNN()
    if materiel is not None:
        modele = modele.to(materiel)
    return modele


def compter_parametres(modele: nn.Module) -> int:
    """Compte tous les parametres du modele.

    Entree:
        modele: modele PyTorch.

    Sortie:
        Nombre total de parametres scalaires.
    """

    return sum(parametre.numel() for parametre in modele.parameters())


def bytes_to_mb(nombre_octets: float) -> float:
    """Convertit des octets en Mo.

    Entree:
        nombre_octets: nombre d'octets.

    Sortie:
        Meme quantite en MiB.
    """

    return nombre_octets / (1024**2)


def build_transform() -> transforms.Compose:
    """Construit le preprocessing Fashion-MNIST.

    Entree:
        Aucune entree.

    Sortie:
        Transformation torchvision qui convertit les images en tenseurs et
        normalise les pixels autour de zero.
    """

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_fashion_mnist(
    dossier_donnees: Path,
) -> tuple[torchvision.datasets.FashionMNIST, torchvision.datasets.FashionMNIST]:
    """Charge les datasets train et test Fashion-MNIST.

    Entree:
        dossier_donnees: dossier local ou torchvision stocke les donnees.

    Sortie:
        Tuple `(donnees_train, donnees_test)`.
    """

    transformation = build_transform()
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


def select_indices(jeu_donnees: torch.utils.data.Dataset, max_exemples: int | None, graine: int) -> list[int]:
    """Selectionne un sous-ensemble aleatoire reproductible.

    Entrees:
        jeu_donnees: dataset dans lequel on tire les exemples.
        max_exemples: nombre maximal d'exemples, ou `None` pour tout prendre.
        graine: graine utilisee par `torch.randperm`.

    Sortie:
        Liste d'indices entiers selectionnes.

    Signification:
        Le tirage est suis une loi uniforme sans remise. Chaque exemple a la meme chance
        d'etre choisi, et la graine rend le choix reproductible.
    """

    generateur = torch.Generator().manual_seed(graine)
    indices = torch.randperm(len(jeu_donnees), generator=generateur).tolist()
    if max_exemples is not None:
        indices = indices[:max_exemples]
    return indices


def get_label(jeu_donnees: torchvision.datasets.FashionMNIST, indice: int) -> int:
    """Retourne le label entier d'un exemple Fashion-MNIST.

    Entrees:
        jeu_donnees: dataset Fashion-MNIST.
        indice: indice de l'exemple.

    Sortie:
        Label entier entre 0 et 9.
    """

    etiquette = jeu_donnees.targets[indice]
    if hasattr(etiquette, "item"):
        return int(etiquette.item())
    return int(etiquette)


def build_iid_client_indices(indices_train: list[int], nombre_clients: int) -> list[list[int]]:
    """Repartit les exemples entre clients en mode IID.

    Entrees:
        indices_train: indices train deja melanges.
        nombre_clients: nombre de clients.

    Sortie:
        Liste ou chaque element contient les indices d'un client.

    Role:
        Le notebook original utilisait `torch.chunk`. On garde la meme logique:
        les donnees sont melangees, puis coupees en morceaux.
    """

    indices_tensor = torch.tensor(indices_train, dtype=torch.long)
    return [bloc.tolist() for bloc in torch.chunk(indices_tensor, nombre_clients)]


def build_non_iid_client_indices(
    jeu_donnees: torchvision.datasets.FashionMNIST,
    indices_train: list[int],
    nombre_clients: int,
) -> list[list[int]]:
    """Attribue aux clients des groupes de labels disjoints.

    Entrees:
        jeu_donnees: dataset train Fashion-MNIST.
        indices_train: indices d'entrainement selectionnes.
        nombre_clients: nombre de clients. Avec 5 clients, chaque client recoit
        deux classes.

    Sortie:
        Liste des indices attribues a chaque client.

    Signification:
        Ce split rend l'apprentissage plus difficile, car les clients ne voient
        pas tous la meme distribution de classes.
    """

    if 10 % nombre_clients != 0:
        raise ValueError("For this non-IID split, num_clients must divide 10.")

    labels_par_client = 10 // nombre_clients
    classe_vers_client: dict[int, int] = {}
    for id_client in range(nombre_clients):
        premier_label = id_client * labels_par_client
        dernier_label = premier_label + labels_par_client
        for label in range(premier_label, dernier_label):
            classe_vers_client[label] = id_client

    clients = [[] for _ in range(nombre_clients)]
    for index in indices_train:
        label = get_label(jeu_donnees, index)
        clients[classe_vers_client[label]].append(index)
    return clients


def build_client_indices(
    jeu_donnees: torchvision.datasets.FashionMNIST,
    indices_train: list[int],
    nombre_clients: int,
    repartition: str,
) -> list[list[int]]:
    """Construit les listes d'indices clients IID ou non-IID.

    Entrees:
        jeu_donnees: dataset train Fashion-MNIST.
        indices_train: indices train selectionnes.
        nombre_clients: nombre de clients.
        repartition: `iid` ou `non_iid`.

    Sortie:
        Liste des indices par client.
    """

    if repartition == "iid":
        return build_iid_client_indices(indices_train, nombre_clients)
    if repartition == "non_iid":
        return build_non_iid_client_indices(jeu_donnees, indices_train, nombre_clients)
    raise ValueError(f"Unknown partition: {repartition}")


def build_loaders(
    donnees_train: torchvision.datasets.FashionMNIST,
    donnees_test: torchvision.datasets.FashionMNIST,
    config: CnnExperimentConfig,
) -> tuple[list[DataLoader], list[int], DataLoader, list[list[int]]]:
    """Cree les DataLoaders clients et le DataLoader de test global.

    Entrees:
        donnees_train: dataset train Fashion-MNIST.
        donnees_test: dataset test Fashion-MNIST.
        config: configuration de l'experience.

    Sortie:
        `(chargeurs_clients, tailles_clients, chargeur_test, indices_clients)`.
    """

    indices_train = select_indices(donnees_train, config.max_exemples_train, config.graine)
    indices_test = select_indices(donnees_test, config.max_exemples_test, config.graine)
    indices_clients = build_client_indices(
        jeu_donnees=donnees_train,
        indices_train=indices_train,
        nombre_clients=config.nombre_clients,
        repartition=config.repartition,
    )

    chargeurs_clients = []
    tailles_clients = []
    for indices_client in indices_clients:
        sous_ensemble = Subset(donnees_train, indices_client)
        chargeur = DataLoader(sous_ensemble, batch_size=config.taille_batch, shuffle=True, num_workers=0)
        chargeurs_clients.append(chargeur)
        tailles_clients.append(len(indices_client))

    sous_ensemble_test = Subset(donnees_test, indices_test)
    chargeur_test = DataLoader(sous_ensemble_test, batch_size=config.taille_batch, shuffle=False, num_workers=0)
    return chargeurs_clients, tailles_clients, chargeur_test, indices_clients


def client_label_summary(
    jeu_donnees: torchvision.datasets.FashionMNIST,
    indices_clients: list[list[int]],
) -> list[dict[str, int]]:
    """Compte les labels attribues a chaque client.

    Entrees:
        jeu_donnees: dataset train Fashion-MNIST.
        indices_clients: liste des indices de chaque client.

    Sortie:
        Liste de dictionnaires `class_name -> count`.
    """

    resumes = []
    for indices_client in indices_clients:
        labels = [get_label(jeu_donnees, index) for index in indices_client]
        compte_labels = Counter(labels)
        resumes.append({NOMS_CLASSES[label]: compte_labels[label] for label in sorted(compte_labels)})
    return resumes


def state_dict_to_cpu(etat_modele: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copie un etat de modele sur CPU.

    Entree:
        etat_modele: dictionnaire PyTorch des poids.

    Sortie:
        Nouveau dictionnaire CPU detache de l'autograd.
    """

    return {nom: tenseur.detach().cpu().clone() for nom, tenseur in etat_modele.items()}


def is_topk(scenario: str) -> bool:
    """Indique si un scenario correspond a top-k.

    Entree:
        scenario: nom du scenario de compression.

    Sortie:
        True pour les noms comme `topk_20`.
    """

    return scenario.startswith("topk_")


def scenario_to_ratio(scenario: str) -> float:
    """Convertit un nom top-k en ratio conserve.

    Entree:
        scenario: nom comme `topk_20`.

    Sortie:
        Ratio comme `0.20`.
    """

    if not is_topk(scenario):
        raise ValueError(f"Scenario is not top-k: {scenario}")
    return float(scenario.split("_")[1]) / 100.0


def validate_scenarios(scenarios: Iterable[str]) -> list[str]:
    """Valide les noms de scenarios.

    Entree:
        scenarios: noms des scenarios demandes.

    Sortie:
        Liste nettoyee des scenarios.
    """

    clean = [scenario.strip() for scenario in scenarios if scenario.strip()]
    if not clean:
        raise ValueError("At least one scenario is required.")
    inconnus = [scenario for scenario in clean if scenario not in SCENARIOS_VALIDES]
    if inconnus:
        raise ValueError(f"Unknown scenarios: {', '.join(inconnus)}")
    return clean


def compress_full_state(
    etat_modele: dict[str, torch.Tensor],
    scenario: str,
) -> dict[str, torch.Tensor]:
    """Compresse un etat dense du modele complet.

    Entrees:
        etat_modele: etat dense du modele.
        scenario: `none` ou `float16`. Top-k n'est pas applique ici, car top-k
        compresse les deltas et non les etats complets.

    Sortie:
        Etat transmis, stocke sur CPU.

    Signification:
        `float16` garde toutes les coordonnees mais diminue la precision
        numerique. C'est une compression peu destructive.
    """

    etat_compresse = {}
    for nom, tenseur in etat_modele.items():
        if scenario == "float16" and torch.is_floating_point(tenseur):
            etat_compresse[nom] = tenseur.cpu().half()
        else:
            etat_compresse[nom] = tenseur.cpu().clone()
    return etat_compresse


def decompress_full_state(etat_transmis: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reconvertit un etat transmis en float32 si necessaire.

    Entree:
        etat_transmis: etat recu apres compression.

    Sortie:
        Etat compatible avec le modele PyTorch.
    """

    etat_decompresse = {}
    for nom, tenseur in etat_transmis.items():
        if torch.is_floating_point(tenseur):
            etat_decompresse[nom] = tenseur.float()
        else:
            etat_decompresse[nom] = tenseur.clone()
    return etat_decompresse


def estimate_full_state_bytes(etat_modele: dict[str, torch.Tensor], scenario: str) -> int:
    """Estime le cout de communication d'un etat dense.

    Entrees:
        etat_modele: etat du modele.
        scenario: `none` ou `float16`.

    Sortie:
        Nombre estime d'octets transmis.
    """

    total_octets = 0
    for tenseur in etat_modele.values():
        if scenario == "float16" and torch.is_floating_point(tenseur):
            total_octets += tenseur.numel() * 2
        elif torch.is_floating_point(tenseur):
            total_octets += tenseur.numel() * 4
        else:
            total_octets += tenseur.numel() * tenseur.element_size()
    return total_octets


def compute_delta(
    etat_local: dict[str, torch.Tensor],
    etat_global: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Calcule le delta de mise a jour d'un client.

    Entrees:
        etat_local: etat apres entrainement local.
        etat_global: etat avant entrainement local.

    Sortie:
        Dictionnaire `etat_local - etat_global` pour les tenseurs flottants.
    """

    delta = {}
    for nom in etat_local:
        if torch.is_floating_point(etat_local[nom]):
            delta[nom] = etat_local[nom].float() - etat_global[nom].float()
    return delta


def sparsify_delta(
    delta_client: dict[str, torch.Tensor],
    ratio: float,
) -> tuple[dict[str, dict[str, torch.Tensor | tuple[int, ...]]], int]:
    """Garde seulement les plus grandes valeurs absolues d'un delta.

    Entrees:
        delta_client: delta dense du client.
        ratio: fraction de valeurs a conserver.

    Sortie:
        Paire `(mise_a_jour_sparse, total_octets)`.

    Signification:
        C'est le coeur de la compression top-k. On transmet les valeurs gardees
        et leurs indices pour reconstruire une mise a jour sparse.
    """

    mise_a_jour_sparse: dict[str, dict[str, torch.Tensor | tuple[int, ...]]] = {}
    total_octets = 0

    for nom, tenseur in delta_client.items():
        valeurs_aplaties = tenseur.flatten()
        nombre_valeurs = valeurs_aplaties.numel()
        nombre_garde = max(1, int(nombre_valeurs * ratio))
        nombre_garde = min(nombre_garde, nombre_valeurs)

        _, indices = torch.topk(valeurs_aplaties.abs(), nombre_garde, sorted=False)
        valeurs = valeurs_aplaties[indices]

        mise_a_jour_sparse[nom] = {
            "shape": tuple(tenseur.shape),
            "indices": indices.cpu(),
            "values": valeurs.cpu(),
        }
        total_octets += nombre_garde * (OCTETS_VALEUR + OCTETS_INDICE)

    return mise_a_jour_sparse, total_octets


def densify_sparse_delta(
    mise_a_jour_sparse: dict[str, dict[str, torch.Tensor | tuple[int, ...]]],
) -> dict[str, torch.Tensor]:
    """Reconstruit un delta dense depuis une charge sparse.

    Entree:
        mise_a_jour_sparse: charge produite par `sparsify_delta`.

    Sortie:
        Delta dense utilisable par le serveur.
    """

    delta_dense = {}
    for nom, charge in mise_a_jour_sparse.items():
        shape = charge["shape"]
        indices = charge["indices"]
        valeurs = charge["values"]
        tenseur_dense = torch.zeros(shape, dtype=torch.float32)
        tenseur_dense.view(-1)[indices] = valeurs.float()
        delta_dense[nom] = tenseur_dense
    return delta_dense


def aggregate_full_states(
    etats_clients: list[dict[str, torch.Tensor]],
    tailles_clients: list[int],
) -> dict[str, torch.Tensor]:
    """Moyenne les modeles clients avec les poids FedAvg.

    Entrees:
        etats_clients: un etat de modele par client.
        tailles_clients: nombre d'exemples de chaque client.

    Sortie:
        Nouvel etat global agrege.

    Signification:
        Chaque client est pondere par sa quantite de donnees locales.
    """

    total_exemples = sum(tailles_clients)
    nouvel_etat = {}

    for nom in etats_clients[0]:
        premiere_valeur = etats_clients[0][nom]
        if torch.is_floating_point(premiere_valeur):
            moyenne_ponderee = torch.zeros_like(premiere_valeur, dtype=torch.float32)
            for etat_client, taille_client in zip(etats_clients, tailles_clients):
                if taille_client == 0:
                    continue
                moyenne_ponderee += etat_client[nom].float() * (taille_client / total_exemples)
            nouvel_etat[nom] = moyenne_ponderee
        else:
            nouvel_etat[nom] = premiere_valeur.clone()
    return nouvel_etat


def aggregate_sparse_updates(
    etat_global: dict[str, torch.Tensor],
    charges_sparse: list[dict[str, dict[str, torch.Tensor | tuple[int, ...]]]],
    tailles_clients: list[int],
) -> dict[str, torch.Tensor]:
    """Agrege les deltas top-k et les ajoute au modele global.

    Entrees:
        etat_global: etat global avant entrainement local.
        charges_sparse: deltas sparse recus des clients.
        tailles_clients: nombre d'exemples de chaque client.

    Sortie:
        Etat global mis a jour.
    """

    total_exemples = sum(tailles_clients)
    deltas_denses = [densify_sparse_delta(charge) for charge in charges_sparse]
    nouvel_etat = {nom: valeur.clone() for nom, valeur in etat_global.items()}

    for nom in deltas_denses[0]:
        delta_agrege = torch.zeros_like(deltas_denses[0][nom], dtype=torch.float32)
        for delta_client, taille_client in zip(deltas_denses, tailles_clients):
            if taille_client == 0:
                continue
            delta_agrege += delta_client[nom].float() * (taille_client / total_exemples)
        nouvel_etat[nom] = etat_global[nom].float() + delta_agrege
    return nouvel_etat


@torch.no_grad()
def evaluate_model(
    modele: nn.Module,
    chargeur: DataLoader,
    materiel: torch.device,
) -> dict[str, float]:
    """Evalue un modele sur le jeu de test.

    Entrees:
        modele: modele a evaluer.
        chargeur: DataLoader du test.
        materiel: CPU ou GPU.

    Sortie:
        Dictionnaire avec `loss` et `accuracy`.

    Signification:
        Le test n'est pas utilise pour l'apprentissage. Il sert a mesurer la
        qualite du modele global apres aggregation.
    """

    modele.eval()
    fonction_perte = nn.CrossEntropyLoss()

    somme_loss = 0.0
    total_exemples = 0
    bonnes_predictions = 0

    for images, labels in chargeur:
        images = images.to(materiel)
        labels = labels.to(materiel)
        logits = modele(images)
        loss = fonction_perte(logits, labels)
        predictions = logits.argmax(dim=1)

        taille_batch = labels.size(0)
        somme_loss += float(loss.item()) * taille_batch
        total_exemples += taille_batch
        bonnes_predictions += int((predictions == labels).sum().item())

    return {
        "loss": somme_loss / total_exemples,
        "accuracy": bonnes_predictions / total_exemples,
    }


def train_local_model(
    etat_global_recu: dict[str, torch.Tensor],
    chargeur_train: DataLoader,
    config: CnnExperimentConfig,
    materiel: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Entraine le modele local d'un client.

    Entrees:
        etat_global_recu: etat global recu du serveur.
        chargeur_train: donnees locales du client.
        config: configuration de l'experience.
        materiel: CPU ou GPU.

    Sortie:
        Tuple `(etat_local, total_exemples, loss_moyenne)`.

    Signification:
        Le client apprend localement sans envoyer ses donnees brutes. Il renvoie
        seulement un etat ou une mise a jour du modele.
    """

    modele_local = create_model(materiel)
    modele_local.load_state_dict(etat_global_recu)
    modele_local.train()

    fonction_perte = nn.CrossEntropyLoss()
    optimiseur = torch.optim.SGD(modele_local.parameters(), lr=config.taux_apprentissage, momentum=0.9)

    somme_loss = 0.0
    total_exemples = 0

    for _ in range(config.epochs_locales):
        for images, labels in chargeur_train:
            images = images.to(materiel)
            labels = labels.to(materiel)

            optimiseur.zero_grad()
            logits = modele_local(images)
            loss = fonction_perte(logits, labels)
            loss.backward()
            optimiseur.step()

            taille_batch = labels.size(0)
            somme_loss += float(loss.item()) * taille_batch
            total_exemples += taille_batch

    etat_local = state_dict_to_cpu(modele_local.state_dict())
    loss_moyenne = somme_loss / total_exemples if total_exemples else 0.0
    return etat_local, total_exemples, loss_moyenne


def run_single_scenario(
    scenario: str,
    chargeurs_clients: list[DataLoader],
    chargeur_test: DataLoader,
    config: CnnExperimentConfig,
    materiel: torch.device,
) -> list[dict[str, float | int | str]]:
    """Lance FedAvg pour un scenario de compression.

    Entrees:
        scenario: compression testee.
        chargeurs_clients: un DataLoader train par client.
        chargeur_test: DataLoader de test global.
        config: configuration de l'experience.
        materiel: CPU ou GPU.

    Sortie:
        Liste des resultats par round.

    Signification:
        C'est la boucle federee principale : envoi du modele, entrainement
        local, compression, aggregation, evaluation.
    """

    reset_seed(config.graine)
    modele_global = create_model(materiel)
    resultats_rounds = []
    communication_cumulee_octets = 0

    print("\n" + "=" * 80)
    print("Scenario:", scenario)
    print("Parameters:", compter_parametres(modele_global))

    for numero_round in range(1, config.nombre_rounds + 1):
        debut_round = time.perf_counter()
        etat_global = state_dict_to_cpu(modele_global.state_dict())

        charges_clients = []
        tailles_clients = []
        losses_clients = []
        octets_downlink = 0
        octets_uplink = 0

        for chargeur_client in chargeurs_clients:
            # Serveur -> client: top-k ne compresse que la mise a jour du
            # client. Le modele envoye par le serveur reste dense ici.
            scenario_downlink = "none" if is_topk(scenario) else scenario
            etat_global_transmis = compress_full_state(etat_global, scenario_downlink)
            octets_downlink += estimate_full_state_bytes(etat_global, scenario_downlink)

            # Le client entraine localement sa copie du modele global.
            etat_global_recu = decompress_full_state(etat_global_transmis)
            etat_local, nombre_exemples, loss_locale = train_local_model(
                etat_global_recu=etat_global_recu,
                chargeur_train=chargeur_client,
                config=config,
                materiel=materiel,
            )

            if is_topk(scenario):
                # Client -> serveur: on transmet seulement les plus grandes
                # coordonnees du delta, avec leurs indices.
                ratio = scenario_to_ratio(scenario)
                delta_client = compute_delta(etat_local, etat_global)
                charge_client, octets_client_uplink = sparsify_delta(delta_client, ratio)
            else:
                # Cas dense: on transmet tout l'etat local, eventuellement en
                # float16.
                etat_local_transmis = compress_full_state(etat_local, scenario)
                charge_client = decompress_full_state(etat_local_transmis)
                octets_client_uplink = estimate_full_state_bytes(etat_local, scenario)

            charges_clients.append(charge_client)
            tailles_clients.append(nombre_exemples)
            losses_clients.append(loss_locale)
            octets_uplink += octets_client_uplink

        if is_topk(scenario):
            nouvel_etat_global = aggregate_sparse_updates(etat_global, charges_clients, tailles_clients)
        else:
            nouvel_etat_global = aggregate_full_states(charges_clients, tailles_clients)

        modele_global.load_state_dict(nouvel_etat_global)
        modele_global.to(materiel)

        metriques = evaluate_model(modele_global, chargeur_test, materiel)
        duree_round = time.perf_counter() - debut_round
        communication_round_octets = octets_downlink + octets_uplink
        communication_cumulee_octets += communication_round_octets

        row = {
            "scenario": scenario,
            "round": numero_round,
            "test_loss": metriques["loss"],
            "test_accuracy": metriques["accuracy"],
            "avg_client_loss": sum(losses_clients) / len(losses_clients),
            "round_seconds": duree_round,
            "downlink_round_mb": bytes_to_mb(octets_downlink),
            "uplink_round_mb": bytes_to_mb(octets_uplink),
            "communication_round_mb": bytes_to_mb(communication_round_octets),
            "communication_total_mb": bytes_to_mb(communication_cumulee_octets),
        }
        resultats_rounds.append(row)

        print(
            f"Round {numero_round:02d} | "
            f"acc={row['test_accuracy']:.4f} | "
            f"loss={row['test_loss']:.4f} | "
            f"time={row['round_seconds']:.1f}s | "
            f"comm_total={row['communication_total_mb']:.2f} MB"
        )

    return resultats_rounds


def summarize_results(
    resultats: list[dict[str, float | int | str]],
    scenarios: list[str],
) -> list[dict[str, float | str]]:
    """Construit une ligne de resume par scenario.

    Entrees:
        resultats: resultats par round.
        scenarios: ordre des scenarios a resumer.

    Sortie:
        Liste de dictionnaires de resume.
    """

    resume = []
    for scenario in scenarios:
        lignes_scenario = [ligne for ligne in resultats if ligne["scenario"] == scenario]
        if not lignes_scenario:
            continue
        derniere_ligne = lignes_scenario[-1]
        resume.append(
            {
                "scenario": scenario,
                "final_accuracy": derniere_ligne["test_accuracy"],
                "final_loss": derniere_ligne["test_loss"],
                "total_seconds": sum(float(ligne["round_seconds"]) for ligne in lignes_scenario),
                "communication_total_mb": derniere_ligne["communication_total_mb"],
                "communication_round_mb": derniere_ligne["communication_round_mb"],
                "uplink_round_mb": derniere_ligne["uplink_round_mb"],
                "downlink_round_mb": derniere_ligne["downlink_round_mb"],
            }
        )
    return resume


def output_paths(results_dir: Path, nom_experience: str) -> SavedPaths:
    """Associe un nom d'experience aux chemins XLSX attendus.

    Entrees:
        results_dir: dossier de sortie.
        nom_experience: nom standard ou personnalise de l'experience.

    Sortie:
        Objet `SavedPaths`.

    Signification:
        On garde les memes noms logiques que les notebooks pour ne pas casser le
        rapport et les scripts d'export.
    """

    mapping = {
        "iid_baseline": ("fedavg_simple_baseline.xlsx", None),
        "iid_float16": ("fedavg_float16_comparison_rounds.xlsx", "fedavg_float16_comparison_summary.xlsx"),
        "iid_sparsification": ("fedavg_sparsification_rounds.xlsx", "fedavg_sparsification_summary.xlsx"),
        "non_iid": ("fedavg_non_iid_rounds.xlsx", "fedavg_non_iid_summary.xlsx"),
        "non_iid_20rounds": ("fedavg_non_iid_20rounds_rounds.xlsx", "fedavg_non_iid_20rounds_summary.xlsx"),
    }
    if nom_experience in mapping:
        rounds_name, summary_name = mapping[nom_experience]
    else:
        safe_name = nom_experience.replace(" ", "_")
        rounds_name = f"fedavg_{safe_name}_rounds.xlsx"
        summary_name = f"fedavg_{safe_name}_summary.xlsx"

    return SavedPaths(
        rounds_xlsx=results_dir / rounds_name,
        summary_xlsx=results_dir / summary_name if summary_name else None,
    )


def ecrire_lignes_xlsx(path: Path, lignes: list[dict[str, object]], noms_colonnes: list[str], nom_feuille: str) -> None:
    """Ecrit des dictionnaires dans un fichier XLSX.

    Entrees:
        path: chemin du XLSX de sortie.
        lignes: lignes a ecrire.
        noms_colonnes: ordre des colonnes.
        nom_feuille: nom de l'onglet Excel.

    Sortie:
        Aucune sortie. Le fichier est ecrit sur disque.
    """

    ecrire_dicts_xlsx(path, lignes, noms_colonnes, nom_feuille)


def save_experiment_results(
    results_dir: Path,
    config: CnnExperimentConfig,
    resultats: list[dict[str, float | int | str]],
    resume: list[dict[str, float | str]],
) -> SavedPaths:
    """Sauvegarde les XLSX par round et les XLSX de resume.

    Entrees:
        results_dir: dossier de sortie.
        config: configuration de l'experience.
        resultats: lignes par round.
        resume: lignes de resume par scenario.

    Sortie:
        Chemins des fichiers ecrits.
    """

    paths = output_paths(results_dir, config.nom_experience)

    if config.nom_experience == "iid_baseline":
        baseline_rows = [
            {
                "round": row["round"],
                "test_loss": row["test_loss"],
                "test_accuracy": row["test_accuracy"],
                "avg_client_loss": row["avg_client_loss"],
                "round_seconds": row["round_seconds"],
                "communication_round_mb": row["communication_round_mb"],
                "communication_total_mb": row["communication_total_mb"],
            }
            for row in resultats
        ]
        ecrire_lignes_xlsx(
            paths.rounds_xlsx,
            baseline_rows,
            [
                "round",
                "test_loss",
                "test_accuracy",
                "avg_client_loss",
                "round_seconds",
                "communication_round_mb",
                "communication_total_mb",
            ],
            "Rounds",
        )
    else:
        ecrire_lignes_xlsx(
            paths.rounds_xlsx,
            resultats,
            [
                "scenario",
                "round",
                "test_loss",
                "test_accuracy",
                "avg_client_loss",
                "round_seconds",
                "downlink_round_mb",
                "uplink_round_mb",
                "communication_round_mb",
                "communication_total_mb",
            ],
            "Rounds",
        )

    if paths.summary_xlsx is not None:
        ecrire_lignes_xlsx(
            paths.summary_xlsx,
            resume,
            [
                "scenario",
                "final_accuracy",
                "final_loss",
                "total_seconds",
                "communication_total_mb",
                "communication_round_mb",
                "uplink_round_mb",
                "downlink_round_mb",
            ],
            "Summary",
        )

    return paths


def run_cnn_fedavg_experiment(
    config: CnnExperimentConfig,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]], SavedPaths]:
    """Lance une experience FedAvg complete avec le petit CNN.

    Entree:
        config: configuration de l'experience.

    Sortie:
        Tuple ``(round_rows, summary_rows, saved_paths)`` :
        resultats par round, resume final et chemins des fichiers XLSX.
    """

    config.scenarios = validate_scenarios(config.scenarios)
    racine_projet = find_project_root()
    dossier_donnees = racine_projet / "data"
    dossier_resultats = racine_projet / "results"
    materiel = get_device()

    reset_seed(config.graine)

    print("Experiment:", config.nom_experience)
    print("Partition:", config.repartition)
    print("Device:", materiel)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    donnees_train, donnees_test = load_fashion_mnist(dossier_donnees)
    chargeurs_clients, tailles_clients, chargeur_test, indices_clients = build_loaders(
        donnees_train=donnees_train,
        donnees_test=donnees_test,
        config=config,
    )

    print("Train examples:", sum(tailles_clients))
    print("Test examples:", len(chargeur_test.dataset))
    print("Client sizes:", tailles_clients)
    if config.repartition == "non_iid":
        for id_client, resume_labels in enumerate(client_label_summary(donnees_train, indices_clients)):
            print(f"Client {id_client} labels:", resume_labels)

    resultats: list[dict[str, float | int | str]] = []
    for scenario in config.scenarios:
        resultats_scenario = run_single_scenario(
            scenario=scenario,
            chargeurs_clients=chargeurs_clients,
            chargeur_test=chargeur_test,
            config=config,
            materiel=materiel,
        )
        resultats.extend(resultats_scenario)

    resume = summarize_results(resultats, config.scenarios)
    paths = save_experiment_results(dossier_resultats, config, resultats, resume)

    print("\nSaved:", paths.rounds_xlsx)
    if paths.summary_xlsx is not None:
        print("Saved:", paths.summary_xlsx)

    return resultats, resume, paths


def plot_results(resultats: list[dict[str, float | int | str]], title: str) -> None:
    """Trace les courbes d'accuracy, de loss et de communication.

    Entrees:
        resultats: lignes de resultats produites a chaque round.
        title: titre de la figure.

    Sortie:
        Aucune sortie Python. Une figure matplotlib est affichee.
    """

    import matplotlib.pyplot as plt

    if not resultats:
        print("No results to plot")
        return

    scenarios = []
    for row in resultats:
        if row["scenario"] not in scenarios:
            scenarios.append(row["scenario"])

    plt.figure(figsize=(14, 4))

    for scenario in scenarios:
        lignes_scenario = [row for row in resultats if row["scenario"] == scenario]
        rounds = [row["round"] for row in lignes_scenario]
        accuracies = [float(row["test_accuracy"]) * 100 for row in lignes_scenario]
        losses = [row["test_loss"] for row in lignes_scenario]
        communication = [row["communication_total_mb"] for row in lignes_scenario]

        plt.subplot(1, 3, 1)
        plt.plot(rounds, accuracies, marker="o", label=scenario)
        plt.title("Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(rounds, losses, marker="o", label=scenario)
        plt.title("Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(rounds, communication, marker="o", label=scenario)
        plt.title("Communication")
        plt.xlabel("Round")
        plt.ylabel("MB")
        plt.grid(True)

    for idx in range(1, 4):
        plt.subplot(1, 3, idx)
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
