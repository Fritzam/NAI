# train_vec_dqn.py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from env_shooter_vec import ShooterEnvVec

def main():
    env = make_vec_env(lambda: ShooterEnvVec(render_mode=None), n_envs=8)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
    )

    model.learn(total_timesteps=2_000_000)
    model.save("shooter_dqn_vec")

if __name__ == "__main__":
    main()