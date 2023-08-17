# Frameworks
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
# Pelo longo?
# Perna curta?
# Late?

# Features
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

animal_misterioso = [1, 1, 1]

# 1 -> Porco 
# 0 -> Cachorro
train_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
train_y = [1, 1, 1, 0, 0, 0] # Labels

# Ignoring warning
warnings.filterwarnings("ignore", category=FutureWarning, message="The default value of `dual` will change from `True` to `'auto'` in 1.5.")


# model
model = LinearSVC()
model.fit(train_x, train_y)

print(model.predict([animal_misterioso]))



misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

test_x = [misterio1, misterio2, misterio3]
previsoes = model.predict(test_x)
print(previsoes)

test_y = [0, 1, 1]

print((previsoes == test_y).sum())

# Accuracy
taxa = accuracy_score(test_y, previsoes)
print(f"Taxa de acerto: {taxa*100: .2f}")
