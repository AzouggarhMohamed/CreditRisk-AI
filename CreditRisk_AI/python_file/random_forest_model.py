import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r"C:\Users\HP 850  G5\Desktop\RandomForest_Python\data\Credir_defaut.csv", sep=";")
df = df.drop_duplicates()
df.columns = df.columns.str.strip()
df = df.drop(columns=['Unnamed: 10'],errors ='coerce')
df['txdendettement'] = df['txdendettement'].str.replace(',','.').str.replace('%','')
df['txdendettement'] = pd.to_numeric(df['txdendettement'],errors='coerce')
df = df.dropna(subset = ['risdefaut'])
new_df = pd.get_dummies(df,columns=['Garantie','typecredit','statut'], drop_first=True, dtype=int)

x = new_df.drop('risdefaut',axis = 1)
y = new_df['risdefaut']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_split=5,random_state=42)
rf_model.fit(x_train,y_train)
rf_pred = rf_model.predict(x_test)

joblib.dump(rf_model,'Credit_default_model.pkl')

#score = cross_val_score(rf_model, x, y, cv=10, scoring='accuracy')

#print(accuracy_score(y_test, rf_pred))
#print(classification_report(y_test, rf_pred))
#print('5 cross validation : ',score)
#print('moyenne de cross validation : ',score.mean(),'ecart-type : ',score.std())