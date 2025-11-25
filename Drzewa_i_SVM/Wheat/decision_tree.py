import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

""" Wczytaj zestaw seeds_dataset, pomiń nagłówki i określ separator jako dowolną ilość białych znaków."""
df = pd.read_csv("seeds_dataset.txt", header=None, sep=r"\s+")

"""Lista z nazwami kolumn dla cech i etykiet (etykiety to ostatni wpis 'class')"""
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_of_kernel",
    "width_of_kernel",
    "asymmetry_coefficient",
    "length_of_kernel_groove",
    "class"
]

"""Przypisanie kolumn do X i Y -- x to cechy, y to etykiety. Etykiety to ostatni element, stąd -1"""
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

""" Podział danych na test i train w stosunku 80:20, żeby zapobiec uczeniu się drzewa pod klucz."""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

""" Model drzewa po kryterium gini"""
model = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)

""" Trenowanie """
model.fit(X_train, y_train)

""" Przewidywanie wyników na danych testowych"""
y_prediction = model.predict(X_test)

""" Wyświetlanie wbudowanych metryk na danych testowych"""
print("Accuracy:", accuracy_score(y_test, y_prediction))
print("\nClassification report:")
print(classification_report(y_test, y_prediction))

""" Rysunek drzewa """
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=["Klasa1", "Klasa2", "Klasa3"],
    filled=True
)

""" Wyświetl drzewo"""
plt.show()