from dotenv import load_dotenv
import requests
import os


load_dotenv()


class Movie:
    """
    Klasa obsługująca pobieranie informacji o filmach z API OMDb.

    Atrybuty:
        api_key (str): Klucz API pobierany ze zmiennych środowiskowych.
    """

    def __init__(self):
        """
        Inicjalizuje instancję klasy Movie, wczytując klucz API z pliku .env.
        """
        self.api_key = os.getenv("API_KEY")

    def get_summary(self, movie_title: str):
        """
        Pobiera opis fabuły filmu z API OMDb.

        Parametry:
            movie_title (str): Tytuł filmu, dla którego ma zostać pobrany opis.

        Zwraca:
            str: Opis fabuły filmu lub komunikat o błędzie.
        """
        params = {
            "apikey": self.api_key,
            "t": movie_title
        }

        url = "http://www.omdbapi.com/"

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if data.get("Response") == "True":
                return data.get("Plot", "Brak opisu.")
            else:
                return f"Brak opisu (błąd API: {data.get('Error')})."

        except Exception as e:
            return f"Brak opisu (błąd: {str(e)})."