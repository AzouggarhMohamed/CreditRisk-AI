# 💳 Prédiction du Risque de Défaut de Crédit - Random Forest \& Streamlit

## 📌 Description

Ce projet a pour objectif de prédire le **risque de défaut de paiement** d'un client (`risdefaut`) à partir de ses caractéristiques socio-financières.

Il combine :

1. L'entraînement et l'optimisation d'un modèle de Machine Learning basé sur l'algorithme **Random Forest** (Validation croisée : `Accuracy = 0.70 ± 0.09`).
2. Le déploiement d'une **application web interactive** avec **Streamlit** pour permettre une évaluation en temps réel du risque client.

\---

## 📊 Données

Le jeu de données `Credir\_defaut.csv` contient **10 339 entrées** et 11 caractéristiques :

|Variable|Description|
|-|-|
|`Age`|Âge du client|
|`Revenu`|Revenu du client|
|`Garantie`|Présence d'une garantie (`oui` / `Non`)|
|`anncien`|Ancienneté professionnelle/bancaire (en années)|
|`typecredit`|Type de crédit (`automobile`, `education`, `MEDICAL`, `personnel`)|
|`Mtcredit`|Montant du crédit demandé|
|`statut`|Statut matrimonial (`celibataire` / `marie`)|
|`txdendettement`|Taux d'endettement du client (%)|
|`histcredit`|Historique de crédit (score numérique)|
|`risdefaut`|**Variable cible** : `1` = Risque de défaut, `0` = Client sûr|

\---

## 🛠️ Méthodologie

1. **Prétraitement \& Nettoyage des Données**

   * Nettoyage du taux d'endettement (suppression du symbole `%` et conversion numérique).
   * Traitement des données manquantes et alignement des chaînes de caractères (gestion de la casse et des espaces).
   * Encodage des variables catégorielles via *One-Hot Encoding* (`pd.get\_dummies`).
2. **Modélisation \& Validation**

   * Modèle : `RandomForestClassifier` (`n\_estimators=200`, `max\_depth=10`, `min\_samples\_split=5`, `random\_state=42`).
   * Évaluation par **10-Fold Cross-Validation** (Score moyen d'accuracy : **70%**).
   * Évaluation via Matrice de Confusion, Rapport de Classification (Précision, Rappel, F1-score) et Courbe ROC / AUC.
3. **Sauvegarde \& Déploiement**

   * Exportation du modèle entraîné au format `joblib` (`Credit\_default\_model.pkl`).
   * Développement de l'interface web interactive dans `app.py`.

\---

## 📁 Structure du Projet

```text
RandomForest\_Python/
├── data/
│   └── Credir\_defaut.csv            # Base de données
├── python\_file/
│   ├── webapp.py                       # Application Streamlit
│   ├── Foret\_aleatoire\_credit\_risque.py # Script d'entraînement
│   └── ex.py                        # Scripts de tests annexes
├── Credit\_default\_model.pkl          # Modèle entraîné (Joblib)
├── requirements.txt                  # Dépendances pour le déploiement Cloud
└── README.md                         # Documentation
```

\---

## ⚙️ Installation \& Exécution

### 1\. Installation des dépendances

```bash
pip install -r requirements.txt
```

*(Ou manuellement : `pip install streamlit pandas scikit-learn joblib matplotlib seaborn`)*

### 2\. Entraîner le modèle

Pour ré-entraîner le modèle et générer le fichier `.pkl` :

```bash
cd python\_file
python Foret\_aleatoire\_credit\_risque.py
```

### 3\. Lancer l'application web Streamlit

Pour exécuter l'application en local :

```bash
cd python\_file
python -m streamlit run app.py
```

L'application s'ouvrira automatiquement à l'adresse locale `http://localhost:8501`.

\---

## 🚀 Déploiement en Ligne (Streamlit Cloud)

L'application peut être hébergée sur **Streamlit Community Cloud** pour un partage public :

1. Connecter le dépôt GitHub à [share.streamlit.io](https://share.streamlit.io/).
2. Pointer vers le fichier principal `python\_file/app.py`.
3. Valider le déploiement pour obtenir un lien URL partageable.

