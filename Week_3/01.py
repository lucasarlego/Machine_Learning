from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
accuracy = rfc.score(X_test, y_test) * 100
print(f"The Random Forest Classifier Score is {accuracy:.2f}%\n")


