# Notes de stage

## Sujet

Mon stage porte sur l'apprentissage federe et la compression des communications.

L'idee principale est simple : dans l'apprentissage federe, plusieurs clients entrainent un modele sans envoyer leurs donnees brutes au serveur. Ils envoient seulement des mises a jour du modele.

Le probleme, c'est que ces echanges peuvent devenir lourds. Le but du stage est donc de voir si on peut compresser ces mises a jour pour envoyer moins de donnees, sans trop perdre en precision.

## Question principale

Comment la compression des communications influence-t-elle :

- la precision du modele ;
- la loss function ;
- le temps d'apprentissage ;
- le nombre de rounds ;
- le volume total de donnees envoyees ?

## etape 1 - Fondements de l'apprentissage federe

L'apprentissage federe est une methode d'apprentissage automatique ou les donnees restent chez les clients.

Contrairement a un apprentissage classique, on ne rassemble pas toutes les donnees dans un seul serveur. Chaque client garde ses propres donnees et entraine localement le modele.

Le serveur central coordonne l'entrainement. Il envoie le modele global aux clients, puis les clients entrainent ce modele sur leurs donnees locales. Ensuite, ils renvoient seulement les mises a jour du modele au serveur.

Le serveur agrege ces mises a jour pour construire un nouveau modele global. Cette operation est repetee plusieurs fois. Chaque repetition s'appelle un round de communication.

Le schema general est :

```text
serveur -> envoie le modele global aux clients
clients -> entrainent le modele sur leurs donnees locales
clients -> renvoient les mises a jour au serveur
serveur -> agrege les mises a jour
serveur -> obtient un nouveau modele global
```

L'algorithme de base le plus connu est FedAvg, pour Federated Averaging.

FedAvg consiste a faire une moyenne des modeles ou des mises a jour envoyees par les clients. Si certains clients ont plus de donnees que d'autres, leur mise a jour peut compter davantage dans la moyenne.

L'avantage principal de l'apprentissage federe est que les donnees brutes ne quittent pas les clients. Cela peut aider pour la confidentialite et pour les cas ou les donnees sont distribuees sur plusieurs appareils ou organisations.

Le probleme principal est le cout de communication. A chaque round, les clients et le serveur doivent echanger des informations sur le modele. Si le modele est gros, ou s'il y a beaucoup de clients et beaucoup de rounds, le volume de donnees echangees devient important.

C'est la que la compression intervient. L'idee est de reduire la taille des mises a jour envoyees par les clients, par exemple en utilisant moins de precision ou en envoyant seulement une partie des valeurs.

La question importante du stage est donc :

```text
Peut-on reduire le volume de communication sans trop degrader la qualite du modele ?
```

Ce compromis sera mesure avec l'accuracy, la loss, le temps d'entrainement, le nombre de rounds et le volume total de donnees echangees.

## etape 2 - Contraintes de communication et compression

Dans l'apprentissage federe, les clients et le serveur echangent regulierement des informations sur le modele.

A chaque round :

```text
serveur -> clients : modele global
clients -> serveur : mises a jour du modele
```

Le probleme est que ces messages peuvent etre lourds, surtout si le modele contient beaucoup de parametres, s'il y a beaucoup de clients ou si le nombre de rounds est eleve.

La taille d'un message depend principalement du nombre de parametres du modele et du format utilise pour les stocker.

Exemple :

```text
float32 = 4 octets par parametre
float16 = 2 octets par parametre
int8    = 1 octet par parametre
```

Donc, pour un modele avec 10 millions de parametres :

```text
float32 -> environ 40 MB
float16 -> environ 20 MB
int8    -> environ 10 MB
```

Les principales contraintes de communication sont :

- la taille du modele ;
- le nombre de clients ;
- le nombre de rounds ;
- la bande passante disponible ;
- la latence du reseau ;
- l'energie consommee par les clients ;
- les clients lents ou parfois deconnectes.

La compression des communications sert a reduire la taille des mises a jour envoyees entre les clients et le serveur.

Les techniques simples a etudier sont :

1. Reduction de precision : passer de `float32` a `float16`.
2. Quantification : representer les valeurs avec moins de bits, par exemple en `int8`.
3. Sparsification : envoyer seulement une partie des valeurs, par exemple les plus grandes mises a jour.

Comparaison rapide :

| Technique | Gain attendu | Risque |
|---|---|---|
| `float16` | volume divise par 2 | faible |
| `int8` | volume divise par 4 | moyen |
| sparsification | volume fortement reduit | moyen ou fort |

Pour le prototype, je vais probablement tester :

```text
1. FedAvg sans compression
2. FedAvg avec float16
3. FedAvg avec sparsification
4. FedAvg avec int8 si le temps le permet
```

L'objectif sera de mesurer si la reduction du volume de communication degrade beaucoup ou non l'accuracy, la loss et le temps de convergence.


## Methode probable

Je vais commencer par une simulation simple :

- dataset : MNIST ou Fashion-MNIST ;
- modele : petit reseau de neurones ou petit CNN ;
- algorithme : FedAvg ;
- plusieurs clients simules sur ma machine.

Ensuite je comparerai plusieurs cas :

1. sans compression ;
2. avec compression legere, par exemple `float16` ;
3. avec compression plus forte, par exemple quantification ou sparsification.

## Ce qu'il faudra mesurer

- Accuracy du modele global.
- Loss pendant l'apprentissage.
- Temps d'entrainement.
- Nombre de rounds necessaires.
- Quantite de donnees echangees.
- Effet de la compression sur la qualite du modele.

## Premiere chose a faire

Lire et comprendre FedAvg.

C'est la base du prototype. Une fois FedAvg compris, je pourrai coder une premiere version sans compression, puis ajouter les compressions ensuite.

## Questions a poser a l'encadrant

- Est-ce qu'il faut utiliser Flower ou une simulation PyTorch suffit ?
- Est-ce que MNIST/Fashion-MNIST convient ?
- Combien de techniques de compression faut-il comparer ?
- Le rapport final doit-il etre en francais ou en anglais ?
