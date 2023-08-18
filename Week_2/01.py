import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

dados = pd.read_csv('.\Week_2/Customer-Churn.csv')

# Quantidade de (linhas, colunas)
#print(dados.shape)

# Visualização do dataset
#print(dados.head())

SWAP = {
    'Sim': 1,
    'Nao': 0
}

# ETL

dados_mod = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(SWAP)
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis=1))
dataset = pd.concat ([dados_mod, dummie_dados], axis=1)
pd.set_option('display.max_columns', 39)

#print(dataset.head())

# Balanceamento dos dados
""" ax = sns.countplot(x='Churn', data=dataset)
plt.show() """

X = dataset.drop('Churn', axis=1)
y = dataset['Churn']

smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)

dataset = pd.concat([X, y], axis=1)

#print(dataset.head(2))
ax = sns.countplot(x='Churn', data=dataset)  
# plt.show()


Xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

norm = StandardScaler()

X_norm = norm.fit_transform(X)
print(X_norm[0])

Xmaria_norm = norm.transform(pd.DataFrame(Xmaria, columns = X.columns))
print(Xmaria_norm)

a = Xmaria_norm
b = X_norm[0]

dist_euclidean = np.sqrt(np.sum(np.square(a-b)))
print(dist_euclidean)

x_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.25, random_state=123)
print(f"Training with {len(x_train)} elements and testing with {len(X_test)} elements!")

# Instância do Model
knn = KNeighborsClassifier(metric='euclidean')

# Treinamento
knn.fit(x_train, y_train)
predict = knn.predict(X_test)

accuracy = accuracy_score(y_test, predict) * 100
print(f"The Accuracy is {accuracy:.2f}%\n")

def distance(dados_clientes, dados_maria, numero_clientes):
    distances = []

    for i in range(numero_clientes):
        dist1 = dados_maria - dados_clientes[i]
        soma_quadrado = np.sum(np.square(dist1))
        distances.append(np.sqrt(soma_quadrado))

    return distances

distance1 = distance(X_norm, Xmaria_norm, 10)
print(distance1)