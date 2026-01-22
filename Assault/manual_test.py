#Test działania bez agenta - zbędny na potrzeby zadania.
from env_shooter_vec import ShooterEnvVec
import time

env = ShooterEnvVec(render_mode="human")
obs, _ = env.reset()

while True:
    action = env.action_space.sample()   # losowe akcje
    obs, reward, term, trunc, _ = env.step(action)
    env.render()
    time.sleep(0.02)