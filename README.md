# Mini-projet Docker & Deep Learning — Prédiction Engie

**Par :** Badreddine Saadioui & Outzoula Abderrazzak  
**Cours :** Technologies IA — Conteneurisation et déploiement

On a repris le cas Engie (prédiction de la puissance des éoliennes) et on l’a mis dans Docker : un conteneur qui entraîne le modèle, un autre qui sert une petite API pour faire des prédictions. Tout est décrit dans les consignes du mini-projet (PDF à la racine).


## Ce que fait le projet

1. **Entraînement**  
   Un script lit les données dans `data/`, les prépare (nettoyage, split temporel, standardisation), cherche les meilleurs réglages du réseau avec Keras Tuner, entraîne le meilleur modèle et enregistre tout dans `model/` (le modèle, le scaler, et la liste des variables).

2. **API**  
   Une app Flask charge le modèle depuis `model/` et expose une page d’accueil, un endpoint de santé et un endpoint de prédiction. L'API accepte une liste de 79 nombres (les features dans le bon ordre) et renvoie une prédiction en kW.

En résumé : **données → entraînement (Docker) → modèle sauvegardé → API (Docker) qui utilise ce modèle.**


## Prérequis

- Docker et Docker Compose (version récente)
- **Données** : le dossier `data/` n'est pas dans le repo (fichiers trop volumineux pour GitHub). Les jeux de données `engie_X.csv` et `engie_Y.csv` du challenge Engie sont disponibles sur la plateforme **Edunao**. Les placer dans `data/` avant de lancer l'entraînement.


## Lancer le projet

À la racine du repo :

```bash
docker compose up --build
```

- Le conteneur **train** part en premier : il lit `data/`, écrit dans `model/`, puis s’arrête.
- Dès que l’entraînement est terminé, le conteneur **app** démarre et écoute sur le port 5000.

Ensuite :

- **Page d’accueil :** http://localhost:5000/
- **Vérifier que tout va bien :** http://localhost:5000/health
- **Prédiction :** envoyer un POST sur `/predict` avec un JSON du type `{"features": [v1, v2, ..., v79]}`.  
  Exemple en ligne de commande :

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}' \
  http://localhost:5000/predict
```

Il faut exactement 79 nombres (même ordre que les variables du modèle). L’ordre des colonnes est dans `model/feature_columns.json` après un entraînement.


## Comparaison local vs Docker

L’entraînement peut être lancé en local (notebook dans `notebooks/` ou script `train/train.py` avec `DATA_DIR` et `MODEL_DIR`) ou via Docker (`docker compose up --build`). Les métriques (MAE, RMSE, R²) et les temps d’exécution sont comparables entre les deux modes ; cette comparaison est détaillée dans le rapport.



## Structure du projet

```
.
├── data/                    # Fichiers Engie (engie_X.csv, engie_Y.csv)
├── train/                   # Script d’entraînement + Dockerfile
├── app/                     # API Flask + Dockerfile + template
├── model/                   # Modèle, scaler et liste de features (créés à l’entraînement)
├── notebooks/               # Notebook de référence (comparaison local)
├── docker-compose.yml       # Lance train puis app
├── .env                     # BATCH_SIZE, EPOCHS, MAX_TRIALS (ne pas pousser sur le repo)
├── README.md                # Ce fichier
└── Consignes_...pdf         # Consignes du mini-projet (à garder)
```
