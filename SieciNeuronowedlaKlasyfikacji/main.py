import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import numpy as np

""" Wczytaj zestaw seeds_dataset, pomiń nagłówki i określ separator jako dowolną ilość białych znaków."""
wheat = pd.read_csv("seeds_dataset.txt", header=None, sep=r"\s+")

"""Lista z nazwami kolumn dla cech i etykiet (etykiety to ostatni wpis 'class')"""
wheat.columns = [
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
X = wheat.iloc[:, :-1]
y = pd.get_dummies(wheat.iloc[:, -1])

""" Podział danych na test i train w stosunku 80:20, żeby zapobiec uczeniu się drzewa pod klucz."""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
+
""" Duży model sieci neuronowej (ANN) – kilka warstw ukrytych. """
model_ANN = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(32, activation="relu", input_shape=(7,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

model_ANN.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

""" Mniejszy model sieci neuronowej – uproszczona architektura."""
model_small = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(32, activation="relu", input_shape=(7,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

model_small.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

""" Model drzewa po kryterium gini"""
model = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)

""" Trenowanie """
model.fit(X_train, y_train)
model_ANN.fit(X_train, y_train, epochs=15)
model_small.fit(X_train, y_train, epochs=15)

""" Przewidywanie wyników na danych testowych"""
prediction = model.predict(X_test)
predicted_class = prediction.argmax(axis=1)
true_class = np.argmax(y_test, axis=1)

prediction2 = model_ANN.predict(X_test)
predicted_class_ANN = prediction2.argmax(axis=1)

prediction3 = model_small.predict(X_test)
predicted_class3 = prediction3.argmax(axis=1)

""" Wyprint("Wynik celności", accuracy_score(true_class_ANN, predicted_class_ANN)), wyświetlanie wbudowanych metryk na danych testowych"""

print("Tablica Pomyłek drzewa:\n", confusion_matrix(true_class, predicted_class))
print("Wynik celności drzewa:", accuracy_score(true_class, predicted_class))

print("Wynik celności dużego modelu:", accuracy_score(true_class, predicted_class_ANN))
print("Tablica Pomyłek dużego modelu:\n", confusion_matrix(true_class, predicted_class_ANN))

print("Wynik celności małego modelu:", accuracy_score(true_class, predicted_class3))
print("Tablica Pomyłek małego modelu:\n", confusion_matrix(true_class, predicted_class3))


""" Normalizacja danych obrazowych oraz kodowanie one-hot etykiet. """
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
X_train, X_test = X_train/255.0, X_test/255.0

""" Prosta sieć CNN do klasyfikacji obrazów CIFAR-10. """
model_cifar = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_cifar.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

""" Tworzenie modelu """
model_cifar.fit(X_train, y_train, epochs=10, validation_split=0.2)

prediction = model_cifar.predict(X_test)
predicted_class = prediction.argmax(axis=1)
true_class = np.argmax(y_test, axis=1)

print("Wynik celności modelu dla CIFAR:", accuracy_score(true_class, predicted_class))


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

""" MLP do klasyfikacji obrazów Fashion MNIST. """
model_MNIST = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model_MNIST.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_MNIST.fit(X_train, y_train, epochs=10, validation_split=0.2)

prediction = model_MNIST.predict(X_test)
predicted_class = prediction.argmax(axis=1)
true_class = np.argmax(y_test, axis=1)

print("Wynik celności modelu dla Fashion MNIST:", accuracy_score(true_class, predicted_class))

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)


""" Ponowne trenowanie tego samego modelu MLP, tym razem na klasycznym zbiorze MNIST. """
model_MNIST.fit(X_train, y_train, epochs=10, validation_split=0.2)

prediction = model_MNIST.predict(X_test)
predicted_class = prediction.argmax(axis=1)
true_class = np.argmax(y_test, axis=1)

print("Wynik celności modelu dla MNIST:", accuracy_score(true_class, predicted_class))