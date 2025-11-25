import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

""" Wczytanie danych z pliku .txt i konwersja na csv'kę. """
df = pd.read_csv("fruit_dataset.csv")

""" X to cechy, Y to etykiety. Etykieta to nazwa owocu."""
X = df.drop("fruit_name", axis=1)
y_text = df["fruit_name"]

""" Zmiana nazw owoców na wartości liczbowe, na stringu metody rzucają wyjątki."""
y, labels = pd.factorize(y_text)

""" Zdefiniowanie kolumn na stringowe i numeryczne """
categorical = ["shape", "color", "taste"]
numeric = ["size", "weight", "avg_price"]

""" Preprocessing i transformacja """
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ]
)
X_processed = preprocess.fit_transform(X)


""" Podział danych na test i train w stosunku 80:20, żeby zapobiec uczeniu się pod klucz """
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

""" konfiguracja dla 3 przypadków o różnych C i gammie. Kernel pozostaje ten sam."""
konfiguracja = [
    ("C=0.1, gamma=0.1", SVC(kernel="rbf", C=0.1, gamma=0.1)),
    ("C=1, gamma=1",     SVC(kernel="rbf", C=1,   gamma=1)),
    ("C=100, gamma=10",  SVC(kernel="rbf", C=100, gamma=10)),
]


""" Trenuje model na wszystkich trzech zestawach danych i generuje raport metodą classification"""
for label, model in konfiguracja:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


""" Redukcja do dwóch wymiarów"""
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

pca1 = X_pca[:, 0]
pca2 = X_pca[:, 1]


""" Początek szkicowania wykresu, wskazanie wartości maksymalnych i minimalnych."""
margin = 1.0
x_min = pca1.min() - margin
x_max = pca1.max() + margin
y_min = pca2.min() - margin
y_max = pca2.max() + margin


""" Tworzenie siatki 200x200"""
xX, yY = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

""" Rysowanie i wywołanie graficznej wersji wykresów"""
plt.figure(figsize=(15, 5))

for i, (label, model) in enumerate(konfiguracja, start=1):

    model.fit(X_pca, y)

    Z_text = model.predict(np.c_[xX.ravel(), yY.ravel()])
    Z = labels.get_indexer(Z_text).reshape(xX.shape)

    plt.subplot(1, 3, i)
    plt.contourf(xX, yY, Z, alpha=0.3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors="black")
    plt.title(label)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

plt.tight_layout()
plt.show()
