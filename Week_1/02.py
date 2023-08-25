import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The default value of `dual` will change from `True` to `'auto'` in 1.5.")


dados = pd.read_csv('Week_1/tracking.csv')

print(dados.head())

x = dados[['home', 'how_it_works', 'contact']]
y = dados[['bought']]

#print(x.head())
#print(y.head())

#print(dados.shape)

SEED = 20

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify= y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(train_x), len(test_x)))
# Model
model = LinearSVC()
model.fit(train_x, train_y)
predicts = model.predict(test_x)

accuracy = accuracy_score(test_y, predicts) * 100
print(f"A Acurácia foi de{accuracy: .2f}%")





