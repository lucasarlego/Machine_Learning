from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import random

##-------------------------------------------------------------- Preparing Data --------------------------------------------------------------##
dataset = pd.read_csv('Week_3\exames.csv')

# Features serão os exames 
X = dataset.drop(columns=["id", "diagnostico", "exame_33"])
# Targets serão os diagnósticos
y = dataset.diagnostico

SEED = 123143
random.seed(SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Instância 
rfc = RandomForestClassifier(n_estimators=100)

# Treino
rfc.fit(X_train, y_train)
print(f"Training with {len(X_train)} elements and testing with {len(X_test)} elements!")

##-------------------------------------------------------------- Random Forest Classifier --------------------------------------------------------------##
# Random Forest Classifier Score
score_rfc = rfc.score(X_test, y_test) * 100
print(f"The Random Forest Classifier Score is {score_rfc:.2f}%")

# Random Forest Classifier Predict
predict_rfc = rfc.fit(X_train, y_train).predict(X_test)

# Random Forest Classifier Accuracy
accuracy_rfc = accuracy_score(y_test, predict_rfc) * 100
print(f"The Random Forest Classifier Accuracy is {accuracy_rfc:.2f}%")

# Random Forest Classifier Precision
precision_rfc = precision_score(y_test, predict_rfc, pos_label='M') * 100
print(f"The Random Forest Classifier Precision is {precision_rfc:.2f}%")

# Random Forest Classifier Recall
recall_rfc = recall_score(y_test, predict_rfc, pos_label='M') * 100
print(f"The Random Forest Classifier Recall is {recall_rfc:.2f}%\n") 

##-------------------------------------------------------------- Dummy Classifier --------------------------------------------------------------##
# Testing the Dummy Score
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_score = dummy.score(X_test, y_test) * 100
print(f"The Dummy Score is {dummy_score:.2f}%")

pattern = StandardScaler()
pattern.fit(y)
y2 = pattern.transform(y)
y2 = pd.DataFrame(data = y2, columns=y.keys())

dados_plot = pd.concat([y, X.iloc[:,0:10]], axis=1)
dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")


plt.figure(figsize=(10, 10))
sns.violinplot(x="exames", y="valores", hue="diagnostico", data= dados_plot, split=True)
plt.xticks(rotation=90)
plt.show()