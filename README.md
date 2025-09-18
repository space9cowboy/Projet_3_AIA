
# 📊 Prédiction du Churn Client - Projet 3 AIA

> Projet de Machine Learning supervisé visant à prédire les clients susceptibles de résilier leur abonnement dans le secteur des télécommunications.

---

## 🚀 Objectifs du projet

- Comprendre les facteurs influençant le **churn** (désabonnement client)
- Construire un **modèle de classification performant**
- Comparer plusieurs algorithmes (Régression Logistique, Arbre de Décision, Random Forest)
- Optimiser les hyperparamètres avec **validation croisée**
- Visualiser et interpréter les résultats pour guider des décisions business

---

## 🗂️ Architecture du projet
PROJET_3_AIA/
│
├── data/
│ ├── raw/ # Données brutes téléchargées depuis Kaggle
│ └── processed/ # Données nettoyées et normalisées
│
├── notebooks/
│ ├── preprocessing.ipynb # Nettoyage et préparation des données
│ ├── EDA.ipynb # Analyse exploratoire (churn vs variables)
│ ├── modeling.ipynb # Modélisation, évaluation, visualisation
│
├── reports/ # (À compléter) Rapport final ou visuels
├── requirements.txt # Liste des bibliothèques Python
├── README.md # Présentation du projet
└── .gitignore

## 📦 Données utilisées

- **Dataset** : Telco Customer Churn
- **Source** : [Kaggle – Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Description** : Caractéristiques des clients, services souscrits, facturation, churn (Oui/Non)

---

## 🔧 Environnement & installation

```bash
# 1. Cloner le projet
git clone https://github.com/space9cowboy/Projet_3_AIA.git
cd PROJET_3_AIA

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt


📈 Modèles testés & performances
Modèle	Accuracy	Recall	F1-score	AUC
Régression Logistique	    0.80	0.72	0.70	0.84
Arbre de Décision	        0.78	0.69	0.68	0.82
Random Forest	            0.85	0.75	0.74	0.88
Random Forest optimisé	    0.87	0.78	0.76	0.90

📌 Modèle retenu : Random Forest optimisé via GridSearchCV
Bon compromis entre détection des churns et limitation des faux positifs.

## 📊 Variables les plus influentes

- `Contract_Two year`, `tenure`, `TechSupport`, `MonthlyCharges`
- ➤ Les clients sans engagement, récents, et sans support technique sont plus à risque de churner.

---

## 💡 Recommandations

- Cibler les **clients récents avec contrats mensuels**
- Proposer des **services de support technique / sécurité** pour fidéliser
- Utiliser ce modèle pour déclencher des **alertes churn** en production

## Résultats

- Le modèle Random Forest optimisé atteint 87% de précision et 78% de recall.
- Il permet de prédire les clients churn avec une performance robuste et interprétable.
- Ces résultats peuvent être utilisés pour déclencher des campagnes de fidélisation ciblées.

## Auteur

Nom : Rabearivony Tsanta Loïc
Formation : AIA01
Date : Septembre 2025

