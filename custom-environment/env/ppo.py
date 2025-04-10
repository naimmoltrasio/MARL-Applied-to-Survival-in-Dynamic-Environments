from stable_baselines3 import PPO
from pettingzoo.utils import parallel_to_aec
import supersuit as ss
from custom_environment import CollaborativePickUpEnv  # importa tu clase

# Crear el entorno
env = CollaborativePickUpEnv()

# Supersuit wrappers
env_wrapped = ss.pettingzoo_env_to_vec_env_v1(env)
env_wrapped = ss.concat_vec_envs_v1(env_wrapped, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

# Entrenar con PPO
model = PPO("MlpPolicy", env_wrapped, verbose=1)
model.learn(total_timesteps=50000)

# Guardar modelo
model.save("ppo_collaborative_env")
