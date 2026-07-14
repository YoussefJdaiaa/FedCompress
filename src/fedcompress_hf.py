"""Experiences FedAvg Hugging Face pour FedCompress.

Il contient toute la logique partagee :

- chargement de Fashion-MNIST;
- adaptation des images au format attendu par les modeles pretrained;
- decoupage des donnees entre clients IID ou non-IID;
- entrainement local cote client;
- aggregation FedAvg cote serveur;
- compression float16 ou top-k;
- sauvegarde des resultats en XLSX.

Les commentaires sont volontairement explicites pour qu'une autre personne
puisse reprendre le projet sans devoir relire les notebooks d'origine.
"""

import copy
import random
import time
from collections import Counter
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


CONFIGS_MODELES = {
    "resnet18": {
        "model_id": "microsoft/resnet-18",
        "display_name": "ResNet-18",
        "result_prefix": "hf_resnet18",
    },
    "convnext_tiny": {
        "model_id": "facebook/convnext-tiny-224",
        "display_name": "ConvNeXT-Tiny",
        "result_prefix": "hf_convnext_tiny",
    },
}


def find_project_root():
    """Trouve la racine du projet.

    Entree:
        Aucune entree directe. La recherche commence depuis le dossier courant.

    Sortie:
        Chemin du dossier racine du projet.

    Signification:
        Les scripts peuvent etre lances depuis plusieurs endroits. Cette
        fonction retrouve le dossier qui contient `README.md` et `src`.
    """

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return current


def reset_seed(graine):
    """Fixe les graines aleatoires.

    Entree:
        graine: valeur entiere utilisee pour rendre les tirages reproductibles.

    Sortie:
        Aucune sortie. Les generateurs aleatoires Python et PyTorch sont
        modifies en place.

    Signification:
        Avec la meme graine, on obtient les memes sous-ensembles de donnees et
        la meme initialisation experimentale.
    """

    random.seed(graine)
    torch.manual_seed(graine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(graine)


def get_device():
    """Choisit le materiel de calcul.

    Entree:
        Aucune entree.

    Sortie:
        `cuda` si un GPU est disponible, sinon `cpu`.

    Signification:
        Les experiences utilisent automatiquement la RTX si PyTorch detecte
        CUDA.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_size(processor):
    """Recupere la taille d'image attendue par le modele Hugging Face.

    Entree:
        processor: objet Hugging Face qui contient les parametres de
        preprocessing du modele.

    Sortie:
        Tuple `(height, width)`, par exemple `(224, 224)`.

    Signification:
        Fashion-MNIST est en 28x28, mais ResNet-18 et ConvNeXT attendent des
        images plus grandes. Cette fonction permet de redimensionner
        correctement les images.
    """

    taille = getattr(processor, "size", None)
    if isinstance(taille, dict):
        if "height" in taille and "width" in taille:
            return (taille["height"], taille["width"])
        if "shortest_edge" in taille:
            cote = taille["shortest_edge"]
            return (cote, cote)
    return (224, 224)


def build_transform(processor):
    """Construit le preprocessing des images.

    Entree:
        processor: image processor Hugging Face associe au modele.

    Sortie:
        Transformation torchvision appliquee a chaque image.

    Signification:
        Les images Fashion-MNIST sont en niveaux de gris. Les modeles
        pretrained attendent generalement trois canaux RGB et une normalisation
        ImageNet. Cette fonction fait cette adaptation.
    """

    taille_image = get_image_size(processor)
    moyenne = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    ecart_type = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(taille_image),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=moyenne, std=ecart_type),
        ]
    )


def load_datasets(dossier_donnees, id_modele):
    """Charge Fashion-MNIST avec le preprocessing du modele.

    Entrees:
        dossier_donnees: dossier local ou les donnees sont stockees.
        id_modele: identifiant Hugging Face du modele, par exemple
        `facebook/convnext-tiny-224`.

    Sortie:
        Tuple `(train_dataset, test_dataset)`.

    Signification:
        Le split train sert a entrainer les clients. Le split test reste separe
        et sert uniquement a evaluer le modele global.
    """

    processor = AutoImageProcessor.from_pretrained(id_modele)
    transformation = build_transform(processor)
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


def select_indices(jeu_donnees, max_exemples, graine):
    """Selectionne un sous-ensemble aleatoire reproductible.

    Entrees:
        jeu_donnees: dataset dans lequel on tire des indices.
        max_exemples: nombre maximal d'exemples a garder, ou `None` pour tout
        prendre.
        graine: graine aleatoire.

    Sortie:
        Liste d'indices selectionnes.

    Signification:
        `torch.randperm` produit une permutation uniforme sans remise. On prend
        ensuite les premiers indices. Chaque exemple a donc la meme probabilite
        d'etre choisi, et le tirage est reproductible avec la graine.
    """

    generateur = torch.Generator().manual_seed(graine)
    indices = torch.randperm(len(jeu_donnees), generator=generateur).tolist()
    if max_exemples is not None:
        indices = indices[:max_exemples]
    return indices


def build_iid_clients(indices_train, nombre_clients):
    """Repartit les exemples entre clients dans le cas IID.

    Entrees:
        indices_train: indices d'entrainement selectionnes.
        nombre_clients: nombre de clients federes.

    Sortie:
        Liste de listes d'indices, une liste par client.

    Signification:
        En IID, les clients recoivent un melange similaire des classes. Cela
        donne un cas plus facile pour FedAvg.
    """

    random.shuffle(indices_train)
    groupes_clients = []
    for id_client in range(nombre_clients):
        groupes_clients.append(indices_train[id_client::nombre_clients])
    return groupes_clients


def get_label(jeu_donnees, indice):
    """Recupere le label d'un exemple Fashion-MNIST.

    Entrees:
        jeu_donnees: dataset Fashion-MNIST.
        indice: indice de l'exemple.

    Sortie:
        Label entier entre 0 et 9.

    Signification:
        Cette fonction sert surtout a construire les partitions non-IID.
    """

    etiquette = jeu_donnees.targets[indice]
    if hasattr(etiquette, "item"):
        return int(etiquette.item())
    return int(etiquette)


def build_non_iid_clients(jeu_donnees, indices_train, nombre_clients):
    """Construit une repartition non-IID par groupes de labels.

    Entrees:
        jeu_donnees: dataset d'entrainement Fashion-MNIST.
        indices_train: indices d'entrainement selectionnes.
        nombre_clients: nombre de clients.

    Sortie:
        Liste des indices attribues a chaque client.

    Signification:
        Chaque client voit surtout deux classes. Ce cas est plus proche d'une
        situation federee difficile, car les clients n'ont pas la meme
        distribution locale.
    """

    labels_par_client = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
    }
    clients = [[] for _ in range(nombre_clients)]
    for indice in indices_train:
        label = get_label(jeu_donnees, indice)
        for id_client in range(nombre_clients):
            if label in labels_par_client[id_client % len(labels_par_client)]:
                clients[id_client].append(indice)
                break
    return clients


def build_loaders(donnees_train, donnees_test, indices_train, indices_test, nombre_clients, taille_batch, repartition):
    """Construit les DataLoaders des clients et du test global.

    Entrees:
        donnees_train: split d'entrainement.
        donnees_test: split de test.
        indices_train: indices d'entrainement selectionnes.
        indices_test: indices de test selectionnes.
        nombre_clients: nombre de clients.
        taille_batch: taille des batchs.
        repartition: `iid` ou `non_iid`.
    Sortie:
        Tuple `(chargeurs_clients, tailles_clients, chargeur_test, indices_clients)`.

    Signification:
        Chaque client a son DataLoader local. Le chargeur de test reste commun et
        sert seulement a evaluer le modele global apres chaque round.
    """

    if repartition == "iid":
        indices_clients = build_iid_clients(indices_train, nombre_clients)
    elif repartition == "non_iid":
        indices_clients = build_non_iid_clients(donnees_train, indices_train, nombre_clients)
    else:
        raise ValueError(f"Unknown partition: {repartition}")

    chargeurs_clients = []
    tailles_clients = []
    for indices_client in indices_clients:
        sous_ensemble = Subset(donnees_train, indices_client)
        chargeur = DataLoader(sous_ensemble, batch_size=taille_batch, shuffle=True)
        chargeurs_clients.append(chargeur)
        tailles_clients.append(len(indices_client))

    sous_ensemble_test = Subset(donnees_test, indices_test)
    chargeur_test = DataLoader(sous_ensemble_test, batch_size=taille_batch, shuffle=False)

    return chargeurs_clients, tailles_clients, chargeur_test, indices_clients


def create_model(id_modele, materiel, geler_backbone=True):
    """Cree un modele Hugging Face de classification d'images.

    Entrees:
        id_modele: identifiant Hugging Face du modele.
        materiel: CPU ou GPU.
        geler_backbone: si True, on gele le backbone et on entraine seulement
        la tete de classification.

    Sortie:
        Modele place sur le device demande.

    Signification:
        C'est ici que se fait le transfert learning federe. Le backbone garde
        ses poids pretrained, et seuls les parametres de type classifier/head/fc
        restent entrainables.
    """

    modele = AutoModelForImageClassification.from_pretrained(
        id_modele,
        num_labels=len(NOMS_CLASSES),
        ignore_mismatched_sizes=True,
    )

    if geler_backbone:
        for parametre in modele.parameters():
            parametre.requires_grad = False

        noms_entrainables = []
        for nom, parametre in modele.named_parameters():
            nom_minuscule = nom.lower()
            if any(cle in nom_minuscule for cle in ["classifier", "head", "fc"]):
                parametre.requires_grad = True
                noms_entrainables.append(nom)

        if not noms_entrainables:
            parametres = list(modele.named_parameters())
            for nom, parametre in parametres[-4:]:
                parametre.requires_grad = True
                noms_entrainables.append(nom)

    modele.to(materiel)
    return modele


def trainable_names(modele):
    """Liste les parametres entrainables.

    Entree:
        modele: modele PyTorch.

    Sortie:
        Liste des noms de parametres avec `requires_grad=True`.

    Signification:
        Quand le backbone est gele, cette liste correspond surtout a la tete de
        classification. C'est aussi ce qui est communique avec
        `portee_etat="trainable"`.
    """

    return [nom for nom, parametre in modele.named_parameters() if parametre.requires_grad]


def get_state(modele, portee):
    """Extrait l'etat du modele qui sera communique.

    Entrees:
        modele: modele PyTorch.
        portee: `all` pour tout communiquer, ou `trainable` pour communiquer
        seulement les parametres entrainables.

    Sortie:
        Dictionnaire de tenseurs sur CPU.

    Signification:
        Cette fonction definit ce qui circule entre serveur et clients. Dans
        nos simulations Hugging Face, on communique principalement la tete de
        classification.
    """

    etat = modele.state_dict()
    if portee == "all":
        return {cle: valeur.detach().cpu().clone() for cle, valeur in etat.items()}
    if portee != "trainable":
        raise ValueError(f"Unknown state scope: {portee}")

    noms = set(trainable_names(modele))
    return {cle: etat[cle].detach().cpu().clone() for cle in noms if cle in etat}


def load_partial_state(modele, etat_partiel):
    """Recharge un etat partiel dans un modele.

    Entrees:
        modele: modele a mettre a jour.
        etat_partiel: dictionnaire contenant une partie des poids.

    Sortie:
        Aucune sortie. Le modele est modifie en place.

    Signification:
        Utile quand on ne communique que les parametres entrainables. Le reste
        du modele, par exemple le backbone gele, reste inchange.
    """

    etat = modele.state_dict()
    for cle, valeur in etat_partiel.items():
        if cle in etat:
            etat[cle] = valeur.to(etat[cle].device).type_as(etat[cle])
    modele.load_state_dict(etat, strict=False)


def train_one_client(modele, chargeur, epochs_locales, taux_apprentissage, materiel):
    """Entraine localement le modele d'un client.

    Entrees:
        modele: copie locale du modele global.
        chargeur: donnees locales du client.
        epochs_locales: nombre d'epochs locales.
        taux_apprentissage: taux d'apprentissage SGD.
        materiel: CPU ou GPU.

    Sortie:
        Loss moyenne locale du client.

    Signification:
        Chaque client part du modele global, apprend sur ses donnees locales,
        puis renvoie une mise a jour au serveur.
    """

    parametres = [parametre for parametre in modele.parameters() if parametre.requires_grad]
    optimiseur = torch.optim.SGD(parametres, lr=taux_apprentissage, momentum=0.9)
    modele.train()
    somme_loss = 0.0
    total_exemples = 0

    for _ in range(epochs_locales):
        for images, labels in chargeur:
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

    if total_exemples == 0:
        return 0.0
    return somme_loss / total_exemples


@torch.no_grad()
def evaluate(modele, chargeur, materiel):
    """Evalue le modele global.

    Entrees:
        modele: modele a evaluer.
        chargeur: DataLoader du jeu de test.
        materiel: CPU ou GPU.

    Sortie:
        Tuple `(loss, accuracy)`.

    Signification:
        Le test n'est jamais utilise pour entrainer. Il sert seulement a suivre
        la performance du modele global apres chaque round.
    """

    modele.eval()
    somme_loss = 0.0
    bonnes_predictions = 0
    total = 0

    for images, labels in chargeur:
        images = images.to(materiel)
        labels = labels.to(materiel)
        sorties = modele(pixel_values=images, labels=labels)
        loss = sorties.loss
        logits = sorties.logits
        predictions = torch.argmax(logits, dim=1)

        taille_batch = labels.size(0)
        somme_loss += float(loss.item()) * taille_batch
        bonnes_predictions += int((predictions == labels).sum().item())
        total += taille_batch

    return somme_loss / total, bonnes_predictions / total


def aggregate_states(etats_clients, tailles_clients):
    """Agrege des modeles clients complets avec FedAvg.

    Entrees:
        etats_clients: un dictionnaire de poids par client.
        tailles_clients: nombre d'exemples par client.

    Sortie:
        Nouvel etat global agrege.

    Signification:
        Les clients avec plus de donnees pesent plus dans la moyenne. C'est la
        formule classique de FedAvg.
    """

    total_exemples = sum(tailles_clients)
    etat_agrege = {}
    for cle in etats_clients[0]:
        valeur_agregee = torch.zeros_like(etats_clients[0][cle], dtype=torch.float32)
        for etat_client, taille_client in zip(etats_clients, tailles_clients):
            valeur_agregee += etat_client[cle].float() * (taille_client / total_exemples)
        etat_agrege[cle] = valeur_agregee
    return etat_agrege


def diff_states(etat_local, etat_global):
    """Calcule la difference entre modele local et modele global.

    Entrees:
        etat_local: etat du client apres entrainement local.
        etat_global: etat du serveur avant entrainement local.

    Sortie:
        Dictionnaire de deltas `local - global`.

    Signification:
        Pour top-k, on compresse la mise a jour du client, pas forcement le
        modele complet.
    """

    return {cle: etat_local[cle].float() - etat_global[cle].float() for cle in etat_global}


def add_delta(etat, delta):
    """Ajoute un delta agrege au modele global.

    Entrees:
        etat: etat global actuel.
        delta: mise a jour agregee.

    Sortie:
        Nouvel etat global.
    """

    return {cle: etat[cle].float() + delta[cle].float() for cle in etat}


def topk_delta(delta, ratio):
    """Applique la sparsification top-k a une mise a jour.

    Entrees:
        delta: mise a jour dense du client.
        ratio: fraction de coordonnees a garder.

    Sortie:
        Tuple `(compressed_delta, masks)`.

    Signification:
        On garde les plus grandes valeurs en valeur absolue. Les petites
        coordonnees sont mises a zero, ce qui reduit l'information transmise.
    """

    delta_compresse = {}
    masques = {}
    for cle, tenseur in delta.items():
        valeurs_aplaties = tenseur.flatten()
        nombre_garde = max(1, int(valeurs_aplaties.numel() * ratio))
        if nombre_garde >= valeurs_aplaties.numel():
            delta_compresse[cle] = tenseur.cpu().clone()
            masques[cle] = torch.ones_like(tenseur, dtype=torch.bool)
            continue
        _, indices = torch.topk(valeurs_aplaties.abs(), nombre_garde, sorted=False)
        valeurs_sparse = torch.zeros_like(valeurs_aplaties)
        valeurs_sparse[indices] = valeurs_aplaties[indices]
        masque_aplati = torch.zeros_like(valeurs_aplaties, dtype=torch.bool)
        masque_aplati[indices] = True
        delta_compresse[cle] = valeurs_sparse.reshape_as(tenseur).cpu()
        masques[cle] = masque_aplati.reshape_as(tenseur).cpu()
    return delta_compresse, masques


def aggregate_deltas(deltas_clients, tailles_clients):
    """Agrege les deltas clients avec les poids FedAvg.

    Entrees:
        deltas_clients: un delta par client.
        tailles_clients: nombre d'exemples par client.

    Sortie:
        Delta global moyen.
    """

    total_exemples = sum(tailles_clients)
    delta_agrege = {}
    for cle in deltas_clients[0]:
        valeur_agregee = torch.zeros_like(deltas_clients[0][cle], dtype=torch.float32)
        for delta_client, taille_client in zip(deltas_clients, tailles_clients):
            valeur_agregee += delta_client[cle].float() * (taille_client / total_exemples)
        delta_agrege[cle] = valeur_agregee
    return delta_agrege


def is_topk(scenario):
    """Indique si un scenario correspond a top-k.

    Entree:
        scenario: nom du scenario, par exemple `topk_20`.

    Sortie:
        Booleen.
    """

    return scenario.startswith("topk_")


def scenario_ratio(scenario):
    """Convertit un nom top-k en ratio conserve.

    Entree:
        scenario: nom comme `topk_20`.

    Sortie:
        Ratio comme `0.20`.
    """

    return float(scenario.split("_")[1]) / 100.0


def state_numel(etat):
    """Compte le nombre de valeurs dans un etat de modele.

    Entree:
        etat: dictionnaire de tenseurs.

    Sortie:
        Nombre total de scalaires.
    """

    return sum(valeur.numel() for valeur in etat.values())


def estimate_full_state_bytes(etat, scenario):
    """Estime le cout de communication d'un etat dense.

    Entrees:
        etat: dictionnaire de poids.
        scenario: `none` ou `float16`.

    Sortie:
        Nombre d'octets transmis.

    Signification:
        En `none`, une valeur float coute 4 octets. En `float16`, elle coute
        2 octets.
    """

    octets_par_valeur = 2 if scenario == "float16" else 4
    return state_numel(etat) * octets_par_valeur


def estimate_topk_delta_bytes(delta, ratio):
    """Estime le cout d'une mise a jour sparse top-k.

    Entrees:
        delta: delta dense avant sparsification.
        ratio: fraction de valeurs conservees.

    Sortie:
        Nombre d'octets, valeurs plus indices.

    Signification:
        Pour top-k, il faut transmettre la valeur et son indice. C'est pour
        cela que topk_50 peut ne pas reduire la communication totale.
    """

    total_octets = 0
    for tenseur in delta.values():
        nombre_garde = max(1, int(tenseur.numel() * ratio))
        total_octets += nombre_garde * (4 + 4)
    return total_octets


def mb(nombre_octets):
    """Convertit des octets en Mo.

    Entree:
        nombre_octets: nombre d'octets.

    Sortie:
        Valeur en MiB.
    """

    return nombre_octets / (1024**2)


def client_label_summary(donnees_train, indices_clients):
    """Compte les labels presents sur chaque client.

    Entrees:
        donnees_train: dataset Fashion-MNIST.
        indices_clients: indices attribues aux clients.

    Sortie:
        Liste de dictionnaires `class_name -> count`.

    Signification:
        Utile pour verifier que le split non-IID est bien desequilibre.
    """

    resumes = []
    for indices_client in indices_clients:
        labels = [get_label(donnees_train, indice) for indice in indices_client]
        compteurs = Counter(NOMS_CLASSES[label] for label in labels)
        resumes.append(dict(compteurs))
    return resumes


def run_hf_fedavg_experiment(
    cle_modele,
    nom_experience,
    scenarios,
    repartition="iid",
    nombre_clients=5,
    nombre_rounds=3,
    epochs_locales=1,
    taille_batch=8,
    taux_apprentissage=0.01,
    max_exemples_train=1000,
    max_exemples_test=500,
    graine=1,
    geler_backbone=True,
    portee_etat="trainable",
):
    """Lance une experience FedAvg complete avec un modele Hugging Face.

    Entrees:
        cle_modele: `resnet18` ou `convnext_tiny`.
        nom_experience: nom utilise dans les fichiers XLSX de sortie.
        scenarios: compressions a tester.
        repartition: `iid` ou `non_iid`.
        nombre_clients: nombre de clients federes.
        nombre_rounds: nombre de rounds de communication.
        epochs_locales: epochs locales par client et par round.
        taille_batch: taille des mini-batchs.
        taux_apprentissage: taux d'apprentissage SGD.
        max_exemples_train: taille maximale du sous-ensemble train.
        max_exemples_test: taille maximale du sous-ensemble test.
        graine: graine aleatoire.
        geler_backbone: si True, le backbone pretrained est gele.
        portee_etat: `trainable` ou `all`, c'est-a-dire ce qui est communique.

    Sortie:
        Tuple `(lignes_rounds, lignes_resume)`. Les XLSX sont aussi sauvegardes
        dans `results/<cle_modele>`.

    Signification:
        C'est la fonction principale appelee par les scripts. Elle reproduit le
        schema d'apprentissage federe : serveur -> clients -> entrainement
        local -> compression -> aggregation -> evaluation.
    """

    racine_projet = find_project_root()
    dossier_donnees = racine_projet / "data"
    dossier_resultats = racine_projet / "results" / cle_modele
    dossier_resultats.mkdir(parents=True, exist_ok=True)

    config_modele = CONFIGS_MODELES[cle_modele]
    id_modele = config_modele["model_id"]
    nom_affichage = config_modele["display_name"]

    materiel = get_device()
    reset_seed(graine)

    print("Modele:", nom_affichage)
    print("Model id:", id_modele)
    print("Experience:", nom_experience)
    print("Repartition:", repartition)
    print("Materiel:", materiel)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    donnees_train, donnees_test = load_datasets(dossier_donnees, id_modele)
    # Tirage aleatoire uniforme sans remise. Le train et le test restent
    # separes, et la graine rend le choix reproductible.
    indices_train = select_indices(donnees_train, max_exemples_train, graine)
    indices_test = select_indices(donnees_test, max_exemples_test, graine + 1)

    chargeurs_clients, tailles_clients, chargeur_test, indices_clients = build_loaders(
        donnees_train=donnees_train,
        donnees_test=donnees_test,
        indices_train=indices_train,
        indices_test=indices_test,
        nombre_clients=nombre_clients,
        taille_batch=taille_batch,
        repartition=repartition,
    )

    print("Exemples train:", sum(tailles_clients))
    print("Exemples test:", len(indices_test))
    print("Tailles clients:", tailles_clients)
    if repartition == "non_iid":
        print("Resume des labels par client:")
        for id_client, resume_labels in enumerate(client_label_summary(donnees_train, indices_clients)):
            print(id_client, resume_labels)

    lignes_rounds = []

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print("Scenario:", scenario)
        reset_seed(graine)

        modele_global = create_model(id_modele, materiel, geler_backbone=geler_backbone)
        etat_global = get_state(modele_global, portee_etat)
        print("Parametres communiques:", state_numel(etat_global))

        communication_totale_octets = 0

        for numero_round in range(1, nombre_rounds + 1):
            debut_round = time.perf_counter()
            etats_clients = []
            deltas_clients = []
            losses_clients = []
            octets_uplink = 0
            octets_downlink = 0

            for chargeur_client in chargeurs_clients:
                # Le serveur envoie le modele global courant au client.
                # On copie le modele pour simuler un entrainement local isole.
                modele_local = copy.deepcopy(modele_global)

                # Cout serveur -> client. Pour top-k, le downlink reste dense:
                # seule la mise a jour du client est sparse dans nos tests.
                scenario_downlink = scenario if not is_topk(scenario) else "none"
                octets_downlink += estimate_full_state_bytes(etat_global, scenario_downlink)

                # Le client entraine localement le modele sur ses propres
                # donnees. Les donnees brutes ne quittent jamais le client.
                loss = train_one_client(
                    modele_local,
                    chargeur_client,
                    epochs_locales=epochs_locales,
                    taux_apprentissage=taux_apprentissage,
                    materiel=materiel,
                )
                losses_clients.append(loss)

                etat_local = get_state(modele_local, portee_etat)

                if is_topk(scenario):
                    # Compression top-k: on envoie seulement les plus grandes
                    # coordonnees du delta local, avec leurs indices.
                    ratio = scenario_ratio(scenario)
                    delta = diff_states(etat_local, etat_global)
                    delta_sparse, _ = topk_delta(delta, ratio)
                    deltas_clients.append(delta_sparse)
                    octets_uplink += estimate_topk_delta_bytes(delta, ratio)
                else:
                    # Compression dense: on envoie tout l'etat local, soit en
                    # float32, soit arrondi en float16.
                    if scenario == "float16":
                        etat_transmis = {
                            cle: valeur.half().float()
                            for cle, valeur in etat_local.items()
                        }
                    else:
                        etat_transmis = etat_local
                    etats_clients.append(etat_transmis)
                    octets_uplink += estimate_full_state_bytes(etat_local, scenario)

                del modele_local
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if is_topk(scenario):
                # Le serveur agrege les deltas sparse, puis les ajoute au
                # modele global.
                delta_agrege = aggregate_deltas(deltas_clients, tailles_clients)
                etat_global = add_delta(etat_global, delta_agrege)
            else:
                # Cas dense: FedAvg fait directement la moyenne ponderee des
                # modeles locaux recus.
                etat_global = aggregate_states(etats_clients, tailles_clients)

            load_partial_state(modele_global, etat_global)
            # Evaluation centrale sur le jeu de test separe.
            loss_test, accuracy_test = evaluate(modele_global, chargeur_test, materiel)

            communication_round_octets = octets_downlink + octets_uplink
            communication_totale_octets += communication_round_octets
            duree_round = time.perf_counter() - debut_round

            ligne = {
                "model_key": cle_modele,
                "model_id": id_modele,
                "experiment": nom_experience,
                "partition": repartition,
                "scenario": scenario,
                "round": numero_round,
                "test_loss": loss_test,
                "test_accuracy": accuracy_test,
                "avg_client_loss": sum(losses_clients) / len(losses_clients),
                "round_seconds": duree_round,
                "downlink_round_mb": mb(octets_downlink),
                "uplink_round_mb": mb(octets_uplink),
                "communication_round_mb": mb(communication_round_octets),
                "communication_total_mb": mb(communication_totale_octets),
                "train_examples": sum(tailles_clients),
                "test_examples": len(indices_test),
                "communicated_parameters": state_numel(etat_global),
            }
            lignes_rounds.append(ligne)

            print(
                f"Round {numero_round:02d} | acc={accuracy_test:.4f} | "
                f"loss={loss_test:.4f} | comm={mb(communication_totale_octets):.2f} MB | "
                f"time={duree_round:.1f}s"
            )

        del modele_global
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lignes_resume = []
    for scenario in scenarios:
        lignes_scenario = [ligne for ligne in lignes_rounds if ligne["scenario"] == scenario]
        derniere_ligne = lignes_scenario[-1]
        lignes_resume.append(
            {
                "model_key": cle_modele,
                "model_id": id_modele,
                "experiment": nom_experience,
                "partition": repartition,
                "scenario": scenario,
                "final_accuracy": derniere_ligne["test_accuracy"],
                "final_loss": derniere_ligne["test_loss"],
                "total_seconds": sum(ligne["round_seconds"] for ligne in lignes_scenario),
                "communication_total_mb": derniere_ligne["communication_total_mb"],
                "communication_round_mb": derniere_ligne["communication_round_mb"],
                "uplink_round_mb": derniere_ligne["uplink_round_mb"],
                "downlink_round_mb": derniere_ligne["downlink_round_mb"],
                "train_examples": derniere_ligne["train_examples"],
                "test_examples": derniere_ligne["test_examples"],
                "communicated_parameters": derniere_ligne["communicated_parameters"],
            }
        )

    prefixe = f"{config_modele['result_prefix']}_{nom_experience}"
    xlsx_rounds = dossier_resultats / f"{prefixe}_rounds.xlsx"
    xlsx_resume = dossier_resultats / f"{prefixe}_summary.xlsx"

    if lignes_rounds:
        ecrire_dicts_xlsx(xlsx_rounds, lignes_rounds, list(lignes_rounds[0].keys()), "Rounds")

    if lignes_resume:
        ecrire_dicts_xlsx(xlsx_resume, lignes_resume, list(lignes_resume[0].keys()), "Summary")

    print("\nSaved:", xlsx_rounds)
    print("Saved:", xlsx_resume)

    return lignes_rounds, lignes_resume


def plot_results(lignes_rounds, titre):
    """Trace les courbes d'accuracy, de loss et de communication.

    Entrees:
        lignes_rounds: lignes de resultats produites a chaque round.
        titre: titre de la figure.

    Sortie:
        Aucune sortie Python. Une figure matplotlib est affichee.
    """

    import matplotlib.pyplot as plt

    if not lignes_rounds:
        print("No results to plot")
        return

    scenarios = []
    for ligne in lignes_rounds:
        if ligne["scenario"] not in scenarios:
            scenarios.append(ligne["scenario"])

    plt.figure(figsize=(14, 4))

    for scenario in scenarios:
        lignes_scenario = [ligne for ligne in lignes_rounds if ligne["scenario"] == scenario]
        numeros_round = [ligne["round"] for ligne in lignes_scenario]
        accuracies = [ligne["test_accuracy"] * 100 for ligne in lignes_scenario]
        losses = [ligne["test_loss"] for ligne in lignes_scenario]
        communication = [ligne["communication_total_mb"] for ligne in lignes_scenario]

        plt.subplot(1, 3, 1)
        plt.plot(numeros_round, accuracies, marker="o", label=scenario)
        plt.title("Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(numeros_round, losses, marker="o", label=scenario)
        plt.title("Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(numeros_round, communication, marker="o", label=scenario)
        plt.title("Communication")
        plt.xlabel("Round")
        plt.ylabel("MB")
        plt.grid(True)

    for idx in range(1, 4):
        plt.subplot(1, 3, idx)
        plt.legend()

    plt.suptitle(titre)
    plt.tight_layout()
    plt.show()
