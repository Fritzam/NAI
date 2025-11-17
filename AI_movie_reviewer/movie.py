from dotenv import load_dotenv
import requests
import os

load_dotenv()

class Movie:
    api_key = api_key = os.getenv("API_KEY")


    def get_summary(self, movie_title: str):

        params = {
            "apikey": self.api_key,
            "t": movie_title
        }

        url = f"http://www.omdbapi.com/"

        response = requests.get(url, params=params)

        
        data = response.json()

        return data['Plot']

