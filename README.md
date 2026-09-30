# 🎬 CinéTarget — Ciblage marketing par prédiction du genre de film

Application **Streamlit** qui prédit le genre d'un film ou d'une série (Action, Comédie, Drame, Horreur) à partir de ses caractéristiques IMDb, puis propose une **stratégie marketing adaptée** (canaux, timing, budget).

Projet réalisé dans le cadre du cours **Naive Bayes** : on compare un classifieur **Naive Bayes gaussien** à un **Random Forest** pour mesurer ce que l'hypothèse d'indépendance des variables coûte en précision.

## Fonctionnalités

L'application est organisée en 4 pages :

| Page | Contenu |
|---|---|
| **Vue d'ensemble marketing** | Problématique, KPIs du jeu de données, répartition des genres |
| **Données & Features** | Nettoyage, encodage et variables créées |
| **Comparaison des modèles** | Naive Bayes vs Random Forest : accuracy, matrices de confusion, rapport de classification |
| **Cibler un film** | Saisie des caractéristiques d'un film → genre prédit, probabilités et recommandations marketing |

## Données et modélisation

- **Source** : `imdb.csv`, 6 178 titres IMDb (note, votes, durée, certificat, type, niveaux de contenu : violence, nudité, langage, alcool, scènes effrayantes)
- **Cible** : genre principal parmi Horror, Action, Comedy, Drama
- **Rééquilibrage** : sous-échantillonnage à la taille de la classe minoritaire (868 titres par genre)
- **Feature engineering** : intensité de contenu, note × log(votes), film récent (≥ 2015), film long (≥ 120 min)
- **Split** : 80 % entraînement / 20 % test, stratifié

| Modèle | Accuracy (test) |
|---|---|
| Naive Bayes gaussien (MinMax scaling) | ~45 % |
| Random Forest (200 arbres) | ~70 % |

Le hasard donnerait 25 % sur 4 classes équilibrées. Naive Bayes est rapide et interprétable mais pâtit des corrélations entre variables (ex. violence et scènes effrayantes). Le Random Forest capte ces interactions.

## Lancer l'application

```bash
git clone https://github.com/clarachalayer/cinetarget-naive-bayes.git
cd cinetarget-naive-bayes
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · Streamlit · pandas · NumPy · scikit-learn · Plotly
