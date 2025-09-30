# Prédiction du Churn Client - Projet 3 AIA

> Projet de Machine Learning supervisé visant à prédire les clients susceptibles de résilier leur abonnement dans le secteur des télécommunications.

---

## Objectifs du projet

- Comprendre les facteurs influençant le **churn** (désabonnement client)
- Construire un **modèle de classification performant**
- Comparer plusieurs algorithmes (Régression Logistique, Arbre de Décision, Random Forest)
- Optimiser les hyperparamètres avec **validation croisée (GridSearchCV)**
- Visualiser et interpréter les résultats pour guider des décisions business

---

## Architecture du projet

```bash
PROJET_3_AIA/
│
├── .venv/                  # environnement virtuel
├── data/                   # données (raw + processed)
├── notebooks/              # notebooks Jupyter (preprocessing, EDA, modeling)
├── reports/                # rapports et résultats
├── src/
│   └── dashboard/          # code pour le dashboard
│       └── streamlit_app.py
│
├── requirements.txt        # dépendances Python
├── README.md               # présentation du projet
└── .gitignore


---

## Données utilisées

- **Dataset** : Telco Customer Churn  
- **Source** : [Kaggle – Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Description** : Caractéristiques des clients, services souscrits, facturation et information sur le churn (Oui/Non).  

---

## Environnement & installation

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

# Modèles testé et performance
| Modèle                   | Accuracy | Recall   | F1-score | AUC      |
| ------------------------ | -------- | -------- | -------- | -------- |
| Régression Logistique    | 0.73     | **0.80** | 0.61     | **0.84** |
| Arbre de Décision        | 0.71     | 0.78     | 0.59     | 0.82     |
| Random Forest (Défaut)   | 0.79     | 0.51     | 0.56     | 0.82     |
| Random Forest (Optimisé) | **0.80** | 0.52     | 0.57     | 0.83     |

Régression Logistique
Car il offre le meilleur compromis pour identifier le maximum de clients churn (Recall élevé) et guider les actions de rétention.

Facteurs de risque :
Contract_Month-to-month, MonthlyCharges, InternetService_Fiber optic, PaymentMethod_Electronic check

Facteurs protecteurs :
tenure (ancienneté), Contract_One year, Contract_Two year, OnlineSecurity, TechSupport

---

## Déploiement du Dashboard

Une application interactive a été développée avec **Streamlit** pour visualiser les résultats du projet.

### Lancer le dashboard

```bash
# Depuis la racine du projet
cd src/dashboard
streamlit run streamlit_app.py
```
