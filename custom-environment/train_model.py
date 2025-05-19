from stable_baselines3 import DQN, A2C, PPO
from env.custom_environment import CollaborativePickUpEnv
from tensorboard_callback import TensorboardCallback
import supersuit as ss

# Crear entorno
env = CollaborativePickUpEnv()
env = ss.pad_action_space_v0(env)
env = ss.flatten_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

# Crear modelo
dqn = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs")
ppo = PPO("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log="./tensorboard_logs")
a2c = A2C("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log="./tensorboard_logs")

"""
print("Modelo en:", ppo.device)
import torch
print("¿CUDA disponible?:", torch.cuda.is_available())
print("GPU utilizada:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU")
"""

# Entrenamiento
dqn.learn(total_timesteps=1000000, callback=TensorboardCallback())
ppo.learn(total_timesteps=1000000, callback=TensorboardCallback())
a2c.learn(total_timesteps=1000000, callback=TensorboardCallback())

# Guardar
dqn.save("models/dqn_collaborative_pickup")
ppo.save("models/ppo_collaborative_pickup")
a2c.save("models/a2c_collaborative_pickup")
