import numpy as np              # NumPy – potrzebne do pracy na macierzach i tworzenia siatki punktów
import matplotlib.pyplot as plt # Matplotlib – rysowanie wykresów
from sklearn.svm import SVC     # SVC – model klasyfikatora SVM
from sklearn.decomposition import PCA  # PCA – redukcja wymiarów do 2D
import pandas as pd             # Pandas – wczytywanie i obsługa danych tabelarycznych


""" Wczytanie danych z pliku .txt i konwersja na csv'kę. """
df = pd.read_csv("seeds_dataset.txt", header=None, sep=r"\s+")

""" Te same kolumny co w drzewie decyzyjnym. """
df.columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient",
    "length_of_kernel_groove", "class"
]

""" X to cechy, Y to etykieta. """
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


""" PCA z dwóch komponentów i trening. """
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  


"""Oś X """
pca1 = X_pca[:, 0]
""" Oś Y """
pca2 = X_pca[:, 1]
""" Margines"""
margin = 1.0

""" Wartości krańcowe osi X i Y"""
x_min = pca1.min() - margin
x_max = pca1.max() + margin
y_min = pca2.min() - margin
y_max = pca2.max() + margin


""" Siatka punktów z NumPy'owej metody meshgrid"""
xX, yY = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

""" Konfiguracja dla 3 zestawów wartości C, gammy i Kernela """
konfiguracja = [
    ("C=0.1, gamma=0.1, Wybrany kernel: rbf", SVC(kernel="rbf", C=0.1, gamma=0.1)),
    ("C=1, gamma=1, Wybrany kernel: poly",     SVC(kernel="poly", C=1,   gamma=1)),
    ("C=100, gamma=10, Wybrany kernel: linear",  SVC(kernel="linear", C=100, gamma=10)),
]

""" Wykres średniej wielkości """
plt.figure(figsize=(15, 5))  


""" Dla każdego wykresu weź wartości i SVC i wykonaj na nich fit, predict, reshape i rysowanie wykresu"""
for i, (label, model) in enumerate(konfiguracja, start=1):


    """ Uczy model """
    model.fit(X_pca, y)

    """ Przewidywanie dla każdego punktu siatki """
    Z = model.predict(np.c_[xX.ravel(), yY.ravel()])

    """ Reshape """
    Z = Z.reshape(xX.shape)

    """ Wybieramy subplo t"""
    plt.subplot(1, 3, i)

    """ Rysujemy kontur, punkty na wykresie, i etykiety."""
    plt.contourf(xX, yY, Z, alpha=0.3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors="black")
    plt.title(label)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

""" Prezentacja """
plt.tight_layout()
plt.show()