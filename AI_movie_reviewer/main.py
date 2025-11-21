import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

from movie import Movie
r"""
Autorami kodu są Michał Fritza(s29235) i Wiktor Świerczyński(s27293)
Instrukcja przygotowania:

Krok 1: Sklonować repozytorium przy pomocy komendy:
    git clone git@github.com:Fritzam/NAI.git

Krok 2: Przejść do katalogu z AI_movie_reviewer i wygenerować wirtualne środowisko:
    Komenda Linux: python3 -m venv .venv
    Komenda Windows: python -m venv venv

Krok 3: Aktywować środowisko wirtualne:
    Komenda Linux: source .venv/bin/activate
    Komenda windows:
        Jeśli CMD: venv\Scripts\activate
        Jeśli Powershell: venv\Scripts\Activate.ps1


Krok 4: Zainstalować dependencje wykorzystując plik requirements.txt
    Komendy dla obydwu systemów: pip install -r requirements.txt (wywołana z katalogu projektu)
    
Krok 5: Upewnić się,że dane.xlsx zostały pobrane 

"""

"""
Wczytujemy dane, tworzymy listę użytkowników, zbiór filmów z gatunkami oraz słownik słów
następnie przechodzimy przez excel usuwając puste komórki oraz uzupełniając użytkowników 
oraz obejrzane przez nich filmy
"""
def load_dataset(path):

    df = pd.read_excel(path, header=None)

    users = []
    all_movies = set()
    user_ratings = defaultdict(dict)

    for i in range(len(df)):
        row = df.iloc[i].dropna().tolist()
        user = row[0]
        users.append(user)

        entries = row[1:]

        for j in range(0, len(entries), 3):
            film = entries[j]
            genre = entries[j+1]
            rating = entries[j+2]

            all_movies.add((film, genre))
            user_ratings[user][(film, genre)] = rating

    return users, list(all_movies), user_ratings


'''
tworzymy macierz cech dla algorytmu K-Means
'''
def generate_feature_matrix(users, movies, user_ratings):
    matrix = []

    for user in users:
        row = []
        for film, genre in movies:
            row.append(user_ratings[user].get((film, genre), 0))
        matrix.append(row)

    return np.array(matrix)


'''
Tworzymy model K-Means grupujący użytkowników w grupach po 3 
'''
def train_kmeans(matrix, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(matrix)
    return kmeans, labels

'''
Następnie Znajujemy klaster do którego pasuje docelowy użytkownik
Tworzymy średnie oceny dla wszystkich filmów w danym klastrze dzięki czemu możemy 
przypasować czy użytkownikowi spodoba/nie spodoba siędany film
'''

def recommend_for_user(target_user, users, movies, labels, user_ratings, top_n=5):
    target_cluster = labels[users.index(target_user)]

    cluster_users = [u for u, lbl in zip(users, labels) if lbl == target_cluster]

    movie_scores = defaultdict(list)
    for user in cluster_users:
        for movie, score in user_ratings[user].items():
            movie_scores[movie].append(score)

    avg_scores = {m: np.mean(scores) for m, scores in movie_scores.items()}

    unseen = [m for m in movies if m not in user_ratings[target_user]]

    recommended = sorted(
        [(m, avg_scores.get(m, 0)) for m in unseen],
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    anti = sorted(
        [(m, avg_scores.get(m, 0)) for m in unseen],
        key=lambda x: x[1]
    )[:top_n]

    return recommended, anti


'''
Tutaj wykorzystujemy zewnętrzne api do uzyskania opisu filmu 
i tworzymy słownik z pelnymi informacjami tytuł gatunek ocena i opis
jak nie ma opisu to zwracamy "brak opisu"
'''
def add_movie_descriptions(movie_api, rec_list):
    detailed_list = []

    for (film, genre), score in rec_list:
        description = movie_api.get_summary(film)

        detailed_list.append({
            "title": film,
            "genre": genre,
            "score": score,
            "description": description
        })

    return detailed_list


'''
Main wykorzystujący poprzednie funkcje na koniec wypisuje 5 rekomendacji i 5 antyrekomendacji
'''
def main():
    users, movies, user_ratings = load_dataset("dane.xlsx")

    matrix = generate_feature_matrix(users, movies, user_ratings)

    print("Trenowanie modelu K-Means...")
    kmeans, labels = train_kmeans(matrix)

    target_user = input("Podaj imię i nazwisko użytkownika: ")

    if target_user not in users:
        print("Nie ma użytkownika")
        return

    recommended, anti = recommend_for_user(target_user, users, movies, labels, user_ratings)

    movie_api = Movie()

    recommended_full = add_movie_descriptions(movie_api, recommended)
    anti_full = add_movie_descriptions(movie_api, anti)

    print(f"REKOMENDACJE DLA: {target_user}")

    for r in recommended_full:
        print(f"\n {r['title']}  ({r['genre']})")
        print(f"Opis: {r['description']}")

    print(f"ANTYREKOMENDACJE DLA: {target_user}")

    for r in anti_full:
        print(f"\n {r['title']}  ({r['genre']})")
        print(f"Opis: {r['description']}")


if __name__ == "__main__":
    main()
