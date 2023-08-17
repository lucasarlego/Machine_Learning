import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Tratando warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The default value of `dual` will change from `True` to `'auto'` in 1.5.")

# Leitura do arquivo
dados = pd.read_csv('.\Exercices/car-prices.csv')

#print(dados.head())

# Tratamento do campo 'sold'
SOLD = {
    'yes': 1,
    'no': 0
}

dados.sold = dados.sold.map(SOLD)

#print(dados.head())

# Tratando coluna de ano do modelo para idade modelo
this_year = datetime.today().year
dados['model_age'] = this_year - dados.model_year

# Tratando Milhas por Kms
KM = 1.60934
dados['km_per_year'] = (dados.mileage_per_year * KM)

# Dropando colunas não utilizadas
dados = dados.drop(columns= ["Unnamed: 0", "mileage_per_year", "model_year"], axis=1)

# Features (x)
x = dados [["price", 'model_age', "km_per_year"]]

# Targets (y)
y = dados["sold"]

# Testing LinearSVC()
SEED = 5
np.random.seed(SEED)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify= y)
print(f"Training with {len(train_x)} elements and testing with {len(test_x)} elements!")

# Model (Linear)
""" model = LinearSVC()
model.fit(train_x, train_y)

predict = model.predict(test_x)

accuracy = accuracy_score(test_y, predict) * 100
print(f"The Accuracy is {accuracy:.2f}%") """

# Dummy Stratified
""" dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(train_x, train_y)
predict = dummy_stratified.predict(test_x)

dummy_accuracy = accuracy_score(test_y, predict) * 100
print(f"The Dummy Accuracy is {dummy_accuracy:.2f}%") """

 # Dummy Most Frequent
"""dummy_mostfrequet = DummyClassifier(strategy="most_frequent")
dummy_mostfrequet.fit(train_x, train_y)
predict = dummy_mostfrequet.predict(test_x)

dummy_mf_accuracy = accuracy_score(test_y, predict) * 100
print(f"The Most Frequent Dummy Accuracy is {dummy_mf_accuracy:.2f}%") """

dummy_stratified = DummyClassifier()
dummy_stratified.fit(train_x, train_y)

dummy_accuracy = dummy_stratified.score(test_x, test_y) * 100
print(f"The Dummy Accuracy is {dummy_accuracy:.2f}% and this is my Baseline!")


raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, stratify = y)
print(f"Training with {len(raw_train_x)} elements and testing with {len(raw_test_x)} elements!")

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = SVC()
model.fit(train_x, train_y)
predict = model.predict(test_x)

accuracy = accuracy_score(test_y, predict)*100
print(f"The Accuracy is {accuracy:.2f}%")
