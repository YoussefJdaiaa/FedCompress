# FedCompress

Projet de stage sur la compression des communications en apprentissage federe.

Le but est d'entrainer un modele global avec FedAvg, puis de mesurer le cout de
communication entre clients et serveur. Les compressions testees sont
principalement `float16` et `top-k`.

## Contenu

```text
src/
  fedcompress_cnn.py      logique FedAvg pour le petit CNN
  fedcompress_hf.py       logique FedAvg pour ResNet-18 et ConvNeXT-Tiny
  hf_benchmark.py         comparaison rapide des modeles Hugging Face
  dataset_tools.py        inspection de Fashion-MNIST
  xlsx_tools.py           ecriture simple des resultats en Excel

scripts/
  setup_env.ps1           prepare le venv
  inspect_dataset.py      inspecte Fashion-MNIST
  run_cnn_experiment.py   lance une experience CNN
  run_cnn_suite.py        lance la suite CNN standard
  run_hf_experiment.py    lance une experience Hugging Face
  run_hf_suite.py         lance la suite Hugging Face standard
  benchmark_hf_models.py  compare ResNet-18 et ConvNeXT-Tiny
  export_results_xlsx.py  genere le classeur de synthese
```

Les dossiers `data/`, `results/`, `.venv/` et `figures/` sont generes
localement et ne sont pas versionnes.

## Installation

Depuis la racine du projet :

```powershell
.\scripts\setup_env.ps1
```

Si PowerShell bloque le script :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Interpreter Python a utiliser dans l'IDE :

```text
.venv\Scripts\python.exe
```

## Commandes utiles

Inspecter le dataset :

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py --save-grid
```

Comparer ResNet-18 et ConvNeXT-Tiny :

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_hf_models.py --models resnet18,convnext_tiny
```

Lancer la suite du petit CNN :

```powershell
.\.venv\Scripts\python.exe scripts\run_cnn_suite.py
```

Lancer une experience CNN :

```powershell
.\.venv\Scripts\python.exe scripts\run_cnn_experiment.py --experiment iid_float16 --partition iid --scenarios none,float16 --rounds 20
```

Lancer la suite Hugging Face :

```powershell
.\.venv\Scripts\python.exe scripts\run_hf_suite.py --model both
```

Lancer une experience Hugging Face :

```powershell
.\.venv\Scripts\python.exe scripts\run_hf_experiment.py --model convnext_tiny --experiment iid_float16 --partition iid --scenarios none,float16 --rounds 3
```

Exporter la synthese Excel :

```powershell
.\.venv\Scripts\python.exe scripts\export_results_xlsx.py
```

## Parametres principaux

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

Pour Hugging Face :

```text
--model resnet18|convnext_tiny|both
--max-train-examples 1000
--max-test-examples 500
--state-scope trainable|all
--no-freeze-backbone
```

Par defaut, le backbone Hugging Face est gele. Seule la tete de classification
est entrainee et communiquee.

## Sorties

Les resultats sont ecrits dans `results/` au format `.xlsx` :

- fichiers par round ;
- fichiers de resume ;
- `fedcompress_results_summary.xlsx` pour la synthese globale.

## Notes

- Fashion-MNIST contient 10 classes.
- Le petit CNN apprend depuis zero.
- ResNet-18 et ConvNeXT-Tiny sont pre-entraines.
- Le cas non-IID est plus difficile car les clients ne voient pas les memes
  classes.
