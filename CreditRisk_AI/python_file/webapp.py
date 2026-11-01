import streamlit as st
import pandas as pd
import joblib

# 1. Charger le modèle préalablement sauvegardé
@st.cache_resource
def load_model():
    return joblib.load(r'C:\Users\HP 850  G5\Desktop\RandomForest_Python\Credit_default_model.pkl')

model = load_model()

# 2. Titre et description de l'application
st.title("💳 Application de Scoring & Prédiction de Défaut de Crédit")
st.markdown("Cette application utilise un modèle de **Machine Learning (Random Forest)** pour estimer si un client présente un risque de défaut de paiement.")

st.sidebar.header("Paramètres du Client")

# 3. Champs de saisie interactifs basés sur tes colonnes exactes
age = st.sidebar.slider("Âge", 18, 70, 30)
revenu = st.sidebar.number_input("Revenu", min_value=1000, max_value=200000, value=25000, step=500)
garantie = st.sidebar.selectbox("Garantie", ["oui", "Non"])
anncien = st.sidebar.slider("Ancienneté", 0, 20, 2)
typecredit = st.sidebar.selectbox("Type de crédit", ["automobile", "education", "MEDICAL", "personnel"])
mtcredit = st.sidebar.number_input("Montant du crédit (Mtcredit)", min_value=100, max_value=100000, value=5000, step=100)
statut = st.sidebar.selectbox("Statut", ["celibataire", "marie"])
txdendettement = st.sidebar.number_input("Taux d'endettement (%)", min_value=0.0, max_value=100.0, value=15.0)
histcredit = st.sidebar.slider("Historique de crédit (histcredit)", 0, 5, 2)

# 4. Bouton de prédiction
if st.sidebar.button("Lancer l'évaluation du risque"):
    
    # Créer un DataFrame avec l'unique ligne de l'utilisateur
    input_data = pd.DataFrame({
        'Age': [age],
        'Revenu': [revenu],
        'Garantie': [garantie],
        'anncien': [anncien],
        'typecredit': [typecredit.lower().strip()],
        'Mtcredit': [mtcredit],
        'statut': [statut.lower().strip()],
        'txdendettement': [txdendettement],
        'histcredit': [histcredit]
    })
    
    try:
        # Appliquer le même One-Hot Encoding que pour l'entraînement
        input_encoded = pd.get_dummies(input_data, columns=['Garantie', 'typecredit', 'statut'], drop_first=True, dtype=int)
        
        # Aligner les colonnes exactement sur celles attendues par le modèle (met des 0 pour les colonnes absentes)
        input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Faire la prédiction
        prediction = model.predict(input_encoded)
        probabilite = model.predict_proba(input_encoded)
        
        st.subheader("Résultat de l'analyse :")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **Risque de défaut élevé !** (Probabilité : {probabilite[0][1]*100:.2f}%)")
            st.write("Ce profil présente des facteurs de risque importants.")
        else:
            st.success(f"✅ **Client sûr / Risque faible** (Probabilité de non-défaut : {probabilite[0][0]*100:.2f}%)")
            st.write("Ce profil ne présente pas de risque majeur.")
            
    except Exception as e:
        st.error(f"Erreur technique lors de la prédiction : {e}")