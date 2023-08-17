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

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify= y)
print(f"Treinaremos com {len(train_x)} elementos e testaremos com {len(test_x)} elementos")

# Model
model = SVC()
model.fit(train_x, train_y)
predicts = model.predict(test_x)

accuracy = accuracy_score(test_y, predicts) * 100
print(f"A Acurácia foi de{accuracy: .2f}%")



x_min = test_x.expected_hours.min()
x_max = test_x.expected_hours.max()

y_min = test_x.price.min()
y_max = test_x.price.max()
# print(x_min, x_max, y_min, y_max)

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)
xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(pontos)
Z = Z.reshape(xx.shape)

# decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(test_x.expected_hours, test_x.price, c=test_y, s=1)
plt.show()


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