import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from IPython.display import display
import graphviz

# Tratando warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The default value of `dual` will change from `True` to `'auto'` in 1.5.")

# Leitura do arquivo
dados = pd.read_csv('.\Week_1/car-prices.csv')

#print(dados.head())

# Tratamento do campo 'sold'vou fazer
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

SEED = 5
np.random.seed(SEED)

# Training Raw
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, stratify = y)
print(f"Training with {len(raw_train_x)} elements and testing with {len(raw_test_x)} elements!")

model = DecisionTreeClassifier(max_depth=3)
model.fit(raw_train_x, train_y)
predict = model.predict(raw_test_x)

accuracy = accuracy_score(test_y, predict) * 100
print(f"The Accuracy is {accuracy:.2f}%\n")

path = "file.png"
features = x.columns
dot_data = export_graphviz(model, out_file=None, filled=True, rounded=True,
                           feature_names=features, class_names=['não', 'sim']) 
graphic = graphviz.Source(dot_data)

# graphic.view()

