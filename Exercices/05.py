import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore", category=FutureWarning, message="The default value of `dual` will change from `True` to `'auto'` in 1.5.")

dados = pd.read_csv('projects.csv')

swap = {
    1 : 0,
    0 : 1
}

dados['finished'] = dados['unfinished'].map(swap)
#print(dados.head())

#sns.scatterplot(x='expected_hours', y='price', hue='finished', data=dados)
#sns.relplot(x='expected_hours', y='price', hue= 'finished', col='finished', data=dados)
# plt.show()

x = dados [['expected_hours', 'price']]
y = dados['finished']

SEED = 5
np.random.seed(SEED)

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, stratify = y)
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = SVC()
model.fit(train_x, train_y)
predicts = model.predict(test_x)

accuracy = accuracy_score(test_y, predicts) * 100
print(f"A acurácia foi de {accuracy: .2f}")

data_x = test_x[:,0]
data_y = test_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data_x, data_y, c=test_y, s=1)
plt.show()