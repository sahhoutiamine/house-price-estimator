# 🏠 House Price Estimator

Estimation automatique du prix de vente d'un logement à partir de ses
caractéristiques (surface, chambres, localisation, qualité, année de
construction, état, équipements...), à l'aide d'un modèle de régression
supervisée. Dataset : **House Prices - Advanced Regression Techniques**.

## Structure du projet

```
house-price-estimator/
├── data/
│   ├── raw/              # Données brutes (train.csv, test.csv) — non versionnées
│   └── processed/        # Données nettoyées / transformées
├── notebooks/            # Notebooks d'exploration, EDA, feature engineering, modélisation
├── src/                  # Code réutilisable (préprocessing, feature engineering, training)
├── models/                # Modèles entraînés sauvegardés (joblib/pickle)
├── reports/figures/       # Graphiques exportés
├── app/
│   └── streamlit_app.py  # Application Streamlit de prédiction
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Plan du projet (5 jours)

| Jour | Contenu |
|------|---------|
| 1 | Exploration, nettoyage, préparation des données |
| 2 | EDA + visualisations + Feature Engineering |
| 3 | Entraînement de 3 modèles + premières évaluations |
| 4 | Cross-validation + GridSearchCV + comparaison + analyse des erreurs |
| 5 | Streamlit + Docker + README + finalisation |

## Setup local (sans Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Placer `train.csv` (et `test.csv` si besoin) dans `data/raw/` avant de lancer
les notebooks.

## Docker

### Construire l'image

```bash
docker compose build
```

(équivalent brut : `docker build -t house-price-estimator:latest .`)

### Lancer le conteneur

```bash
docker compose up
```

(équivalent brut :
`docker run -p 8501:8501 -v $(pwd)/models:/app/models:ro house-price-estimator:latest`)

### Accéder à l'application

Ouvrir : http://localhost:8501

### Arrêter le conteneur

```bash
docker compose down
```

> ⚠️ Le modèle n'est **jamais réentraîné** dans le conteneur : il est
> entraîné une fois via les notebooks/scripts (Jour 1 à 4), sauvegardé dans
> `models/` avec `joblib`, puis simplement **chargé** par l'app Streamlit
> au démarrage.

## Données

Dataset : *House Prices - Advanced Regression Techniques*.
Variable cible : `SalePrice`.

Placer les fichiers `train.csv` (obligatoire) et `test.csv` (optionnel) dans
`data/raw/`.
