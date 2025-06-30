import os
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from env.custom_environment import CollaborativePickUpEnv


# CONFIGURACIONES Y RECOMPENSAS
CONFIGS = ["config2", "config3"]
REWARD_MODES = ["coordination"]
ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

# Función para inicializar entorno con configuración y tipo de recompensa
def make_env(config_name, reward_mode):
    env = CollaborativePickUpEnv(config=config_name, reward_mode=reward_mode)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    return env

# Entrenamiento por combinación
def run_experiment(config, reward_mode, algo_name, total_timesteps=5000000):
    print(f"\n🔁 Config: {config} | Reward: {reward_mode} | Algo: {algo_name.upper()}")
    env = make_env(config, reward_mode)
    model_class = ALGORITHMS[algo_name]
    model = model_class("MlpPolicy", env, verbose=0, tensorboard_log=f"logs/{config}_{reward_mode}_{algo_name}")

    eval_callback = EvalCallback(env, best_model_save_path=f"models/{config}_{reward_mode}_{algo_name}",
                                 log_path=f"logs/{config}_{reward_mode}_{algo_name}", eval_freq=5000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"models/{config}_{reward_mode}_{algo_name}/final_model")
    print("✅ Entrenamiento completado")

# Lanza todos los experimentos
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    for config in CONFIGS:
        for reward in REWARD_MODES:
            for algo in ALGORITHMS:
                run_experiment(config, reward, algo, total_timesteps=100000)
