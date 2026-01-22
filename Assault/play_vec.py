"""

Plik służy do uruchomienia gry z wykorzystaniem bota.

"""

from stable_baselines3 import DQN
from env_shooter_vec import ShooterEnvVec
import time

env = ShooterEnvVec(render_mode="human")

""" Wczytanie wytrenowanego modelu DQN """
model = DQN.load("shooter_dqn_vec")

""" Reset środowiska """
obs, info = env.reset()

while True:
    """
    Główna pętla symulacji.

    """

    """ Przewidyawanie ruchu agenta"""
    action, _ = model.predict(obs, deterministic=True)

    """ Wykonanie środowiska """
    obs, reward, terminated, truncated, info = env.step(int(action))

    """ Render """
    env.render()

    """ Małe opóźnienie dla oszczędzenia procesora """
    time.sleep(0.016)

    """ Jeśli epizod się zakończył – startujemy od nowa"""
    if terminated or truncated:
        obs, info = env.reset()