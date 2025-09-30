import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Projet 3 AIA - Loïc",
    layout="wide"
)

# Titre et description
st.title("Projet 3 AIA - Tableau de Bord d'Analyse de Données - Churn Prediction ML")
st.markdown("---")

# Barre latérale
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choisissez une page :",
    ["Accueil", "Vue d'ensemble des données", "Analyse", "Résultats du modèle"]
)

if page == "Accueil":
    st.header("Bienvenue sur le Tableau de Bord du Projet")
    st.write("""
    Ce projet a été réalisé dans le cadre de la formation AIA01 et a pour objectif
    de prédire le **churn client** dans le secteur des télécommunications.
    """)

    st.subheader("Objectifs du projet")
    st.markdown("""
    - Comprendre les facteurs qui influencent le churn
    - Construire et comparer plusieurs modèles de classification (Régression Logistique, Arbre de Décision, Random Forest)
    - Optimiser les hyperparamètres avec **GridSearchCV**
    - Fournir des amélioration business pour guider une stratégie de rétention client
    """)

    st.subheader("Résumé du dataset")
    st.markdown("""
    - **Source** : [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    - **Nombre de clients** : ~7 000
    - **Variables** : caractéristiques clients, services souscrits, facturation, information de churn (Oui/Non)
    """)

    st.subheader("Resumé des étapes")
    st.markdown("""
    - **Nettoyage & préparation** des données (`preprocessing.ipynb`)
    - **Analyse exploratoire** des comportements des clients churners (`EDA.ipynb`)
    - **Modélisation** avec plusieurs algorithmes (Régression Logistique, Arbre de Décision, Random Forest) (`modeling.ipynb`)
    - **Optimisation** des performances via `GridSearchCV`
    - **Visualisation** des résultats avec Seaborn & Matplotlib => (integration `streamlit_app.py`)
    - **Comparaison** des modèles avec AUC, Recall, F1-score
    """)

    st.subheader("Fonctionnalités du dashboard")
    st.markdown("""
    - **Exploration et visualisation des données**
    - **Analyse statistique et corrélation**
    - **Comparaison des modèles de Machine Learning**
    - **Conclusion et recommandations business**
    """)

    # Vérifier si les fichiers de données existent
    data_path = Path("data/processed/prepared_dataset_normalized.csv")
    if data_path.exists():
        st.success(f"Fichier de données trouvé : {data_path}")
    else:
        st.warning(f"Fichier de données non trouvé : {data_path}")

elif page == "Vue d'ensemble des données":
    st.header("Vue d'ensemble des Données")

    # Essayer de charger les données traitées
    try:
        data_path = Path("data/processed/prepared_dataset_normalized.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nombre total de lignes", len(df))
            with col2:
                st.metric("Nombre total de colonnes", len(df.columns))
            with col3:
                st.metric("Utilisation mémoire",
                          f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            st.subheader("Échantillon du jeu de données")
            st.dataframe(df.head(), use_container_width=True)

            st.subheader("Informations sur les colonnes")
            col_info = pd.DataFrame({
                'Colonne': df.columns,
                'Type de données': df.dtypes.values,
                'Nombre de valeurs non nulles': df.count().values,
                'Nombre de valeurs nulles': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)

            # --- Pie chart du churn ---
            st.subheader("Répartition du churn (Oui/Non)")
            if "Churn" in df.columns:
                churn_counts = df["Churn"].value_counts()
                fig, ax = plt.subplots()
                ax.pie(
                    churn_counts,
                    labels=["Non churn", "Churn"],
                    autopct='%1.1f%%',
                    startangle=50,
                    colors=["#66c2a5", "#fc8d62"]
                )
                ax.axis("equal")
                st.pyplot(fig)
                st.markdown("""73.5 pourcent des clients sont restés (Churn = 0)
                                26.6 pourcent des clients ont churné (Churn = 1)=> soit 1 client sur 3,8
                            """)
            else:
                st.warning(
                    "La colonne 'Churn' n'a pas été trouvée dans le dataset.")

            st.subheader("Top 4 variables les plus corrélées au churn")
            correlations = df.corr()["Churn"].drop(
                "Churn").abs().sort_values(ascending=False).head(4)
            st.bar_chart(correlations)

        else:
            st.error(
                "Jeu de données non trouvé. Veuillez vous assurer que le fichier de données existe à l'emplacement correct.")

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")

elif page == "Analyse":
    st.header("Analyse des Données - Churn")

    try:
        data_path = Path("data/processed/prepared_dataset_normalized.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)

            # Sélectionner les colonnes pour l'analyse
            numeric_columns = df.select_dtypes(
                include=[np.number]).columns.tolist()

            if numeric_columns:
                selected_column = st.selectbox(
                    "Sélectionnez une colonne pour l'analyse :", numeric_columns)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"Distribution de {selected_column}")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    df[selected_column].hist(bins=30, ax=ax)
                    ax.set_xlabel(selected_column)
                    ax.set_ylabel("Fréquence")
                    st.pyplot(fig)

                with col2:
                    st.subheader("Statistiques descriptives")
                    st.write(df[selected_column].describe())

            else:
                st.warning("Aucune colonne numérique trouvée pour l'analyse.")

            # -------------------------------
            # Boxplot Tenure vs Churn
            # -------------------------------
            st.subheader("Ancienneté (Tenure) selon le churn")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="Churn", y="tenure", palette="Set2", ax=ax)
            st.pyplot(fig)
            st.markdown("""Ancienneté moyenne :
                            Churn
                            No    0.213019 > 0
                            Yes   -0.588451 < 0
                        """)
            st.markdown(
                "Les clients récents (tenure faible) churnent beaucoup plus que les anciens. Ils quittent l'entreprise tôt dans le cycle client.")

            # -------------------------------
            # Boxplot MonthlyCharges vs Churn
            # -------------------------------
            st.subheader("Frais mensuels (MonthlyCharges) selon le churn")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="Churn", y="MonthlyCharges",
                        palette="Set3", ax=ax)
            st.pyplot(fig)
            st.markdown("""MonthlyCharges moyen :
                            Churn
                            No   -0.116036 < 0
                            Yes    0.320542 > 0
                        """)
            st.markdown(
                "Les clients avec des frais mensuels élevés sont plus enclins à churner.")

            # -------------------------------
            # Contrat vs Churn
            # -------------------------------
            st.subheader("Type de contrat et churn")
            if "Contract_One year" in df.columns and "Contract_Two year" in df.columns:
                df_contract = df.copy()
                df_contract["Contract"] = "Month-to-month"
                df_contract.loc[df_contract["Contract_One year"]
                                == 1, "Contract"] = "One year"
                df_contract.loc[df_contract["Contract_Two year"]
                                == 1, "Contract"] = "Two year"

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=df_contract, x="Contract",
                            y="Churn", palette="Set1", ax=ax)
                st.pyplot(fig)
                st.markdown("""
                            Taux de churn selon Contract_One year :
                                Contract_One year
                                0    30.629496
                                1    11.277174
                            Taux de churn selon Contract_Two year :
                                Contract_Two year
                                0    34.056480
                                1     2.848665
                            
                            - Les clients en contrat 1 an → leur taux de churn chute à 11%.
                            - Les clients en contrat 2 ans → ils sont presque captifs, avec un churn ultra faible à 2,8%.
                            - Les clients en month-to-month sont ceux qui churnent le plus ≈ 40%,
                            """)
                st.markdown(
                    "Les contrats **longs (1–2 ans)** réduisent fortement le churn comparé aux contrats mensuels.")

            # -------------------------------
            # Payment Method vs Churn
            # -------------------------------
            st.subheader("Méthode de paiement et churn")
            payment_cols = [
                col for col in df.columns if col.startswith("PaymentMethod_")]
            if payment_cols:
                churn_by_payment = {}
                for col in payment_cols:
                    churn_by_payment[col] = df.groupby(
                        col)["Churn"].mean().get(1, 0) * 100
                churn_df = pd.DataFrame.from_dict(
                    churn_by_payment, orient="index", columns=["Taux de churn (%)"])

            st.bar_chart(churn_df)
            st.markdown("""
                        - Payment method electronic check (CHURN élevé) => 45,3% => cheque éléctronique => méthode a risque
                        - Payment method credit card (CHURN faible) => 15,3% => carte bancaire => paiement automoatisé = meilleure fidélité = moins de churn
                        - Payment method electronic check (CHURN moyen) => 19,2% => chéque postal => méthode intermediaire""")
            st.markdown(
                "Les clients payant par **Electronic check** sont les plus à risque de churn.")

            # -------------------------------
            # Online Security vs Churn
            # -------------------------------
            st.subheader("Sécurité en ligne et churn")
            col_online = "OnlineSecurity_Yes" if "OnlineSecurity_Yes" in df.columns else "OnlineSecurity"
            if col_online in df.columns:
                churn_pct = df.groupby(col_online)["Churn"].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                churn_pct.plot(kind="bar", color="skyblue", ax=ax)
                ax.set_title("Taux de churn selon OnlineSecurity")
                ax.set_xlabel(
                    "Souscription à Online Security (0 = Non, 1 = Oui)")
                ax.set_ylabel("Taux de churn (%)")
                st.pyplot(fig)
                st.markdown("""
                            - Les clients qui n’ont pas activé OnlineSecurity sont beaucoup plus susceptibles de churner.
                            - Les clients protégés par OnlineSecurity restent plus fidèles.""")
                st.markdown(
                    f"Churn avec OnlineSecurity=1 : **{round(churn_pct[1], 1)}%** vs sans OnlineSecurity=0 : **{round(churn_pct[0], 1)}%**.")

        else:
            st.error("Jeu de données non trouvé.")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")

elif page == "Résultats du modèle":
    st.header("Résultats des modèles de Machine Learning")

    st.subheader("Les Modèles")

    models_info = pd.DataFrame({
        "Modèle": ["Régression Logistique", "Arbre de Décision", "Random Forest"],
        "Principe": [
            "Modèle linéaire qui estime la probabilité de churn en fonction des variables.",
            "Applique des règles successives (si/alors) pour classer un client.",
            "Ensemble d’arbres de décision, combine leurs prédictions pour être plus robuste."
        ],
    })
    st.dataframe(models_info, use_container_width=True)

    st.subheader("Métriques de performance")

    metrics_info = pd.DataFrame({
        "Métrique": ["Accuracy", "Recall", "F1-score", "AUC"],
        "Définition": [
            "Proportion de prédictions correctes (globalement).",
            "Capacité à détecter les churners (clients partis).",
            "Moyenne harmonique entre précision et rappel (équilibre).",
            "Mesure globale de séparation entre churn et non-churn (0.5 = hasard, 1 = parfait)."
        ],
    })

    st.dataframe(metrics_info, use_container_width=True)

    # --- Tableau comparatif ---
    st.subheader("Comparaison des modèles")
    results = pd.DataFrame({
        "Modèle": [
            "Régression Logistique",
            "Random Forest (Optimisé)",
            "Arbre de Décision",
            "Random Forest (Défaut)"
        ],
        "Accuracy": [0.73, 0.80, 0.71, 0.79],
        "Recall": [0.80, 0.52, 0.78, 0.51],
        "F1-score": [0.61, 0.57, 0.59, 0.56],
        "AUC": [0.84, 0.83, 0.82, 0.82]
    })
    st.dataframe(results, use_container_width=True)

    # --- Barplot comparatif ---
    st.subheader("Comparaison visuelle des performances")
    results_melted = results.melt(
        id_vars="Modèle", var_name="Métrique", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_melted, x="Métrique", y="Score",
                hue="Modèle", ax=ax, palette="Set2")
    ax.legend(title="Modèle", fontsize=8, title_fontsize=8,
              bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)

    # --- Matrices de confusion ---
    st.subheader("Matrices de confusion")
    st.markdown("""  
    La matrice de confusion permet de comprendre *comment* un modèle se trompe :  
    - Trop de **FP** → trop d’alertes inutiles → risque d’agacer les clients.  
    - Trop de **FN** → beaucoup de churners ratés → perte de revenus.  

    on privilégie la réduction des **FN** (augmenter le Recall) car l’objectif est de détecter un maximum de churners.
    """)
    selected_model = st.selectbox(
        "Choisissez un modèle pour afficher sa matrice de confusion",
        ["Régression Logistique",
         "Random Forest (Optimisé)", "Arbre de Décision", "Random Forest (Défaut)"]
    )

    # Matrices de confusion enregistrées (tes résultats calculés avant)
    matrices = {
        "Régression Logistique": [[723, 310], [75, 299]],
        "Random Forest (Optimisé)": [[928, 105], [181, 193]],
        "Arbre de Décision": [[700, 333], [81, 293]],
        "Random Forest (Défaut)": [[923, 110], [184, 190]]
    }

    cm = np.array(matrices[selected_model])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
        "Non-Churn", "Churn"], yticklabels=["Non-Churn", "Churn"], ax=ax)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion - {selected_model}")
    st.pyplot(fig)

    # Matrice générique
    cm_example = np.array([[100, 20],
                           [30, 50]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_example,
        annot=[["TN", "FP"], ["FN", "TP"]],
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Prédit : Non Churn", "Prédit : Churn"],
        yticklabels=["Réel : Non Churn", "Réel : Churn"],
        ax=ax
    )
    st.pyplot(fig)

    confusion_info = pd.DataFrame({
        "Type": ["Vrais Négatifs (TN)", "Faux Positifs (FP)", "Faux Négatifs (FN)", "Vrais Positifs (TP)"],
        "Définition": [
            "Clients non churn correctement prédits comme non churn.",
            "Clients non churn prédits à tort comme churn (fausse alerte).",
            "Clients churn prédits à tort comme non churn (churn ratés).",
            "Clients churn correctement prédits comme churn."
        ],
    })

    st.dataframe(confusion_info, use_container_width=True)

    # --- Conclusion ---
    st.subheader("Conclusion et modèle retenu")
    st.markdown("""
        - **Régression Logistique** : meilleur Recall (0.80) et meilleur AUC (0.84) → excellent pour détecter les churners  
        - **Random Forest optimisé** : meilleure Accuracy (0.80), mais Recall limité (0.52) → plus précis mais moins sensible  
        - **Arbre de Décision** : rappel élevé (0.78) mais performance globale plus faible  
        - **Random Forest défaut** : similaire à la version optimisée mais un peu moins stable  

        **Modèle retenu : Régression Logistique**, car l’objectif principal est de **maximiser la détection des churners**.
        """)

# Pied de page
st.markdown("---")
st.markdown("*Tableau de bord créé avec Streamlit*")
