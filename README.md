# FedCompress

Projet de stage sur la compression des communications en apprentissage federe.

L'objectif est simple : entrainer un modele global avec FedAvg, mesurer le
volume de communication entre clients et serveur, puis verifier si des
compressions comme `float16` ou `top-k` reduisent ce volume sans trop degrader
l'accuracy.

## Idee du projet

Le projet compare trois familles d'experiences :

- un petit CNN entraine depuis zero sur Fashion-MNIST ;
- ResNet-18 pre-entraine depuis Hugging Face ;
- ConvNeXT-Tiny pre-entraine depuis Hugging Face.

Pour les modeles Hugging Face, le protocole par defaut est du transfer learning
federe :

- le backbone pre-entraine est gele ;
- seule la tete de classification est entrainee ;
- seule la partie entrainable est communiquee au serveur.

Ce choix rend les simulations possibles sur une machine locale avec peu de
VRAM.

## Structure

```text
.
|-- src/
|   |-- fedcompress_cnn.py      # FedAvg + compressions pour le petit CNN
|   |-- fedcompress_hf.py       # FedAvg + compressions pour ResNet/ConvNeXT
|   |-- hf_benchmark.py         # benchmark des modeles Hugging Face
|   |-- dataset_tools.py        # inspection de Fashion-MNIST
|
|-- scripts/
|   |-- common_cli.py           # helpers communs aux scripts
|   |-- inspect_dataset.py      # inspecte Fashion-MNIST
|   |-- benchmark_hf_models.py  # compare ResNet-18 et ConvNeXT-Tiny
|   |-- run_cnn_experiment.py   # lance une experience CNN
|   |-- run_cnn_suite.py        # lance la suite CNN standard
|   |-- run_hf_experiment.py    # lance une experience Hugging Face
|   |-- run_hf_suite.py         # lance la suite Hugging Face standard
|   |-- export_results_xlsx.py  # exporte les resultats en Excel
|
|-- data/                       # datasets locaux, non versionnes
|-- results/                    # XLSX generes, non versionnes
|-- figures/                    # figures generees
|-- reports/                    # rapport LaTeX
|-- README.md
```

La separation est volontaire :

- `src/` contient la logique scientifique reutilisable ;
- `scripts/` contient les points d'entree a lancer depuis le terminal.

## Workflow

Le workflow normal est le suivant :

```text
1. Inspecter le dataset
2. Benchmarker les modeles Hugging Face
3. Lancer les simulations avec le petit CNN
4. Lancer les simulations avec ResNet-18 et ConvNeXT-Tiny
5. Exporter les resultats en Excel
6. Analyser les resultats dans le rapport
```

## Installation

Le dossier `.venv/` n'est pas pousse sur Git. C'est volontaire : un
environnement virtuel contient des fichiers propres a la machine, a Windows ou
Linux, a la version de Python et parfois au GPU. Le partager directement dans
Git rendrait le projet lourd et fragile.

La bonne pratique est de versionner `requirements.txt`, puis de recreer le
`.venv` localement. Sur Windows, une personne qui vient de faire `git pull`
lance simplement :

```powershell
.\scripts\setup_env.ps1
```

Si PowerShell bloque le script a cause de la politique d'execution :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Ensuite elle utilise l'interpreteur du projet :

```powershell
.\.venv\Scripts\python.exe
```

Dans VS Code ou PyCharm, il suffit de selectionner cet interpreteur :

```text
.venv\Scripts\python.exe
```

Le fichier `requirements.txt` installe les bibliotheques principales du projet :
PyTorch, torchvision, transformers, matplotlib, pillow et numpy.

## Commandes principales

Toutes les commandes ci-dessous se lancent depuis la racine du projet.

### 1. Inspecter Fashion-MNIST

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py --save-grid
```

Cette commande affiche la repartition des classes et sauvegarde une grille
d'exemples dans `figures/`.

### 2. Benchmarker les modeles Hugging Face

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_hf_models.py --models resnet18,convnext_tiny
```

Pour un test rapide :

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_hf_models.py --models resnet18,convnext_tiny --train-subset-size 1000 --test-subset-size 500
```

### 3. Lancer la suite du petit CNN

Voir le plan sans entrainer :

```powershell
.\.venv\Scripts\python.exe scripts\run_cnn_suite.py --dry-run
```

Lancer toute la suite :

```powershell
.\.venv\Scripts\python.exe scripts\run_cnn_suite.py
```

Lancer une seule experience :

```powershell
.\.venv\Scripts\python.exe scripts\run_cnn_experiment.py --experiment iid_float16 --partition iid --scenarios none,float16 --rounds 20
```

### 4. Lancer la suite Hugging Face

Voir le plan sans entrainer :

```powershell
.\.venv\Scripts\python.exe scripts\run_hf_suite.py --model both --dry-run
```

Lancer les experiences standard sans le cas long de 20 rounds :

```powershell
.\.venv\Scripts\python.exe scripts\run_hf_suite.py --model both --skip-long
```

Lancer toute la suite :

```powershell
.\.venv\Scripts\python.exe scripts\run_hf_suite.py --model both
```

Lancer une seule experience :

```powershell
.\.venv\Scripts\python.exe scripts\run_hf_experiment.py --model convnext_tiny --experiment iid_float16 --partition iid --scenarios none,float16 --rounds 3
```

### 5. Exporter les resultats

```powershell
.\.venv\Scripts\python.exe scripts\export_results_xlsx.py
```

Le fichier genere est :

```text
results/fedcompress_results_summary.xlsx
```

## Parametres utiles

Parametres communs :

```text
--partition iid|non_iid
--scenarios none,float16,topk_50,topk_20,topk_10
--clients 5
--rounds 3
--local-epochs 1
--batch-size 8
--learning-rate 0.01
--seed 1
```

Parametres Hugging Face :

```text
--model resnet18|convnext_tiny|both
--max-train-examples 1000
--max-test-examples 500
--state-scope trainable|all
--no-freeze-backbone
```

Par defaut :

```text
freeze_backbone = True
state_scope = trainable
```

Donc le backbone est gele et seule la tete de classification est entrainee et
communiquee.

## Experiences standard

Les suites lancent ces experiences :

| Nom | Partition | Scenarios |
|---|---|---|
| `iid_baseline` | IID | `none` |
| `iid_float16` | IID | `none`, `float16` |
| `iid_sparsification` | IID | `none`, `topk_50`, `topk_20`, `topk_10` |
| `non_iid` | non-IID | `none`, `float16`, `topk_20`, `topk_10` |
| `non_iid_20rounds` | non-IID | `none`, `float16`, `topk_20`, `topk_10` |

## Resultats

Les sorties sont ecrites dans `results/` :

- XLSX par round ;
- XLSX de resume ;
- fichier Excel de synthese.

Le dossier `results/` n'est pas destine a etre pousse sur GitHub. Il peut etre
regenere avec les scripts.

## Notes importantes

- Fashion-MNIST contient 10 classes, 60 000 images train et 10 000 images test.
- ResNet-18 et ConvNeXT-Tiny sont pre-entraines sur de grands datasets.
- Le petit CNN n'est pas pre-entraine : il apprend depuis zero.
- Le cas non-IID est volontairement plus difficile, car les clients ne voient
  pas la meme distribution de classes.
- `float16` reduit la precision des poids.
- `top-k` transmet seulement les plus grandes composantes des mises a jour.

## Git

Avant de pousser :

```powershell
git status
```

Le code important est dans :

```text
src/
scripts/
reports/
README.md
```

Les dossiers `data/`, `results/`, `.venv/` et les caches Python ne doivent pas
etre versionnes.
