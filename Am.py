import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 🔹 Fonction pour charger les données
def load_data():
    file_path = r"C:\Users\AMADOUBA\Desktop\Master2 SID\Biostatistique\Projet_GitHub\Donnnées_Projet_M2SID2023_2024_préparées.xlsx"
    df = pd.read_excel(file_path, engine="openpyxl")
    return df

# 🔹 Fonction pour définir l'image d'arrière-plan
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    
    css_code = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)
    
# 🔹 Fonction pour afficher la courbe ROC
def plot_roc_curve(y_test, y_pred_proba, model_name):
    if y_pred_proba is not None and len(y_pred_proba) > 0:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        st.pyplot(plt)
        plt.close()
    else:
        st.write("La courbe ROC ne peut pas être affichée car les probabilités ne sont pas disponibles pour ce modèle.")
# 🔹 Fonction principale
def main():
    set_background("ML.jpg")  # Ajouter l'image en arrière-plan
    
    st.title("Application de Machine Learning pour une étude pronostique complète de la survenue de décès après le traitement")
    st.subheader("Auteur: Amadou BA et Mahmoud SIDIBE")

    # 🔸 Barre latérale pour la navigation
    st.sidebar.subheader("Accueil / BIENVENUE")
    menu = st.sidebar.radio("Choisissez une section", ["Données et Statistiques", "Modélisation", "Prédiction"])

    df = load_data()

    # 🔹 Section Données et Statistiques
    if menu == "Données et Statistiques":
        st.sidebar.subheader("Options d'affichage")
        
        show_data = st.sidebar.checkbox("Afficher les données")
        show_stats = st.sidebar.checkbox("Afficher les statistiques et les graphes de distribustiions des variables")

        if show_data:
            st.write("### Aperçu de notre base de données:")
            st.write(df.head())
            st.write("#### Liste des variables disponibles dans notre jeu de données:", df.columns.tolist())

        if show_stats:
            st.write("### Statistiques descriptives de l'âge:")
            st.write(df["AGE"].describe())

            # 🔹 Histogramme de l'âge
            st.write("### Distribution des âges")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df["AGE"], bins=20, kde=True, color="skyblue", edgecolor="black", ax=ax)
            st.pyplot(fig)

            # 🔹 Diagramme circulaire de l'évolution
            st.write("### Répartition des modalités de l'évolution")
            fig, ax = plt.subplots()
            df["Evolution"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=["lightcoral", "lightblue"], ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

            # 🔹 Autres variables binaires (ex: Hypertension, Diabète)
            binary_columns = ["Hypertension_Arterielle", "Diabete", "Cardiopathie","hémiplégie", "Paralysie_faciale", "Aphasie", "Hémiparésie", "Engagement_Cerebral", "Inondation_Ventriculaire", "Traitement_Trombolyse", "Traitement_Chirurgical"]
            for col in binary_columns:
                st.write(f"### Répartition de {col}")
                fig, ax = plt.subplots()
                df[col].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=["lightgreen", "orange"], ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

    # 🔹 Section Modélisation 
    elif menu == "Modélisation":
        st.sidebar.subheader("Choix de la modélisation")
        model_choice = st.sidebar.radio("Sélectionnez un type de modèle", ["Modèles Classiques (ML)", "Deep Learning"])
        
        X = df.drop(columns=["Evolution"])
        y = df["Evolution"]

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Modèles Classiques (ML)":
            st.write("### Implémentation des modèles classiques de Machine Learning")
            
            model_type = st.selectbox("Choisissez un modèle", ["Régression Logistique", "SVM", "Forêt Aléatoire"])
            if st.button("Entraîner le modèle"):
                if model_type == "Régression Logistique":
                    model = LogisticRegression()
                elif model_type == "SVM":
                    model = SVC(probability=True)
                else:
                    model = RandomForestClassifier()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Vérification de la disponibilité des probabilités
                y_pred_proba = None
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    st.write("### Courbe ROC")
                    plot_roc_curve(y_test, y_pred_proba, model_type)
                else:
                    y_pred_proba = None  # Initialisation pour éviter l'erreur
                    st.write("Le modèle sélectionné ne permet pas de calculer des probabilités.")

                st.write("### Évaluation du modèle")
                st.write("Précision:", accuracy_score(y_test, y_pred))
                st.write("Score de Précision:", precision_score(y_test, y_pred, average='weighted'))
                st.write("Score de Rappel:", recall_score(y_test, y_pred, average='weighted'))
                st.write("Cohen Kappa:", cohen_kappa_score(y_test, y_pred))
                st.write("Matrice de Confusion:")
                st.write(confusion_matrix(y_test, y_pred))
                

            
        elif model_choice == "Deep Learning":
            st.write("### Implémentation des modèles de Deep Learning")

            if st.button("Entraîner le modèle Deep Learning"):
                # Création d'un modèle simple pour la classification
                model = Sequential([
                    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # Entraînement du modèle
                history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

                # Évaluation
                loss, accuracy = model.evaluate(X_test, y_test)
                st.write("### Évaluation du Modèle")
                st.write("Loss:", loss)
                st.write("Accuracy:", accuracy)
                
                history = model.history.history  # Correct accès au dictionnaire

                plt.figure(figsize=(8, 6))
                plt.plot(history.get('accuracy', []), label='Train Accuracy')
                plt.plot(history.get('val_accuracy', []), label='Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.title('Performance du modèle Deep Learning')
                st.pyplot(plt)
                        

    # 🔹 Section Prédiction (En attente de l'implémentation)
    elif menu == "Prédiction":
        st.write("### Prédiction à implémenter demain : Interface pour la prédiction.")

if __name__ == '__main__':
    main()
