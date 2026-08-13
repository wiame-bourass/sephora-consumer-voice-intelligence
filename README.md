# Sephora Consumer Voice Intelligence

Projet NLP / GenAI autour des avis skincare Sephora.

L’objectif est de transformer des avis clients non structurés en informations exploitables : quels aspects d’un produit sont appréciés ou critiqués, avec quel sentiment, et sur quelle partie du texte repose la prédiction.

## Vue d’ensemble

```text
Reviews Sephora
→ audit et nettoyage
→ segmentation
→ taxonomie ABSA
→ benchmark de plusieurs LLM
→ annotation avec GPT-OSS 20B
→ 2 000 segments pseudo-annotés
→ agrégations business
→ DistilBERT V1 puis V2
→ évaluation
→ démo Streamlit
```

Le projet se concentre sur trois catégories skincare riches en avis : **Moisturizers, Treatments et Cleansers**.

## Dataset

Les données proviennent du dataset Sephora Products and Skincare Reviews.

Fichiers attendus dans `data/raw/` :

- `product_info.csv`
- `reviews_0-250.csv`
- `reviews_250-500.csv`
- `reviews_500-750.csv`
- `reviews_750-1250.csv`
- `reviews_1250-end.csv`

Les fichiers bruts ne sont pas versionnés dans le dépôt.

## Taxonomie ABSA

Le pipeline utilise une taxonomie fermée de 9 aspects :

- efficacy / results
- hydration / dryness
- texture / finish
- irritation / sensitivity
- acne / breakouts
- fragrance / smell
- application / absorption
- packaging
- price / value

Chaque prédiction associe un aspect à un sentiment (`positive`, `neutral`, `negative`).

Lors de l’annotation avec GPT-OSS 20B, une preuve textuelle exacte est également extraite du segment.

## Annotation avec GPT-OSS 20B

Plusieurs LLM ont été comparés sur un petit benchmark avant de retenir GPT-OSS 20B pour générer les pseudo-labels.

Le run principal du POC utilise :

- 2 000 segments ;
- batch size 4 ;
- retries et checkpoints ;
- validation du JSON et des preuves textuelles.

Ces pseudo-labels servent ensuite à entraîner un modèle DistilBERT plus léger.

## Modèle DistilBERT

Deux versions ont été comparées.

La V1 a montré une faiblesse importante sur la détection multi-label des aspects.

La V2 ajoute :
- une pondération des classes positives par aspect ;
- des seuils de décision ajustés sur validation ;
- une pondération des classes pour le sentiment.

Résultats V2 sur le split test du POC :

| Tâche | Métrique | Score |
|---|---:|---:|
| Aspect | F1 micro | 0.621 |
| Aspect | F1 macro | 0.643 |
| Sentiment | Accuracy | 0.814 |
| Sentiment | Macro F1 | 0.631 |

Ces scores mesurent l’accord du modèle DistilBERT avec les pseudo-labels générés par GPT-OSS 20B. Ils ne correspondent pas à une évaluation sur une vérité terrain annotée par des humains.

## Consumer insights

Les sorties ABSA sont agrégées par :

- aspect ;
- catégorie ;
- marque ;
- produit ;
- type de peau.

L’objectif est d’identifier les principaux motifs de satisfaction ou d’insatisfaction et de comparer les catégories skincare.

Les visualisations finales sont dans `reports/figures/final/`.

## Démo Streamlit

Une petite application permet de tester DistilBERT V2 sur un avis libre et de consulter quelques agrégations business.

Les poids DistilBERT ne sont pas versionnés dans le dépôt en raison de leur taille. Pour utiliser l'analyse d'un avis, exécuter le notebook `08_MODEL_V2_WEIGHTED_THRESHOLDS_COLAB.ipynb` afin de générer les artifacts nécessaires dans `artifacts/`.

Depuis la racine du projet :

```bash
streamlit run app/streamlit_app.py
```

## Structure

```text
sephora-consumer-voice-intelligence/
├── app/
├── config/
├── data/
├── notebooks/
├── prompts/
├── reports/
├── scripts/
├── src/
├── .env.example
├── requirements.txt
├── PROJECT_ARCHITECTURE.md
└── README.md
```

## Installation

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pour les appels LLM, copier `.env.example` vers `.env` puis renseigner l’endpoint, la clé API et le modèle.

## Limites

Le projet est un POC, pas un système de production.

Principales limites :
- 2 000 segments pseudo-annotés pour entraîner DistilBERT ;
- absence de jeu de vérité terrain humain indépendant ;
- sentiment neutre peu représenté ;
- erreurs encore visibles sur certains aspects proches et sur les phrases contrastives.

Une suite logique serait d’augmenter le volume et la diversité des pseudo-labels avant d’entraîner une nouvelle version du modèle.
