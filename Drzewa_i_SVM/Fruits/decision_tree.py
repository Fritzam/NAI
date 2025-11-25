import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn import tree


# Dane dostępne: https://www.kaggle.com/datasets/pranavkapratwar/fruit-classification
""" Wczytanie csv'ki """
df = pd.read_csv("fruit_dataset.csv")

""" ostatnia kolumna to etykieta, wszystkie pozostałe to cechy"""
X = df.drop("fruit_name", axis=1)
y = df["fruit_name"]

""" Podział na cechy numeryczne i kategoryczne """
categorical = ["shape", "color", "taste"]
numerical = ["size", "weight", "avg_price"]

""" kodowanie cech kategorycznych """
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[categorical])

""" Łączenie zakodowanych cech wraz z numerycznymi"""
X_final = pd.concat(
    [
        pd.DataFrame(X_encoded, index=X.index),
        X[numerical]
    ],
    axis=1
)

""" Przerobienie kolumn na stringi, bez tego model nie działa. """
X_final.columns = X_final.columns.astype(str)

""" Podział danych na test i train w stosunku 80:20, żeby zapobiec uczeniu się drzewa pod klucz."""
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

""" Wybór modelu, w tym przypadku kryteria gini i głębokość 10 (5 miała skuteczność na poziomie 60%) """
model = DecisionTreeClassifier(max_depth=10, criterion="gini", random_state=42)

""" Trening """
model.fit(X_train, y_train)

""" Szacowanie """
y_pred = model.predict(X_test)

""" Gotowa klasyfikacja"""
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

""" Rysowanie drzewa """
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    filled=True,
    feature_names=list(X_final.columns),
    class_names=model.classes_
)
plt.show()