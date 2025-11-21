from dotenv import load_dotenv
import requests
import os

load_dotenv()

class Movie:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")

    def get_summary(self, movie_title: str):
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

