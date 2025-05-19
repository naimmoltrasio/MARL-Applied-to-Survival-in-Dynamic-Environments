import time
import pygame
import numpy as np
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from env.custom_environment import CollaborativePickUpEnv

# Inicializar entorno paralelo
env = CollaborativePickUpEnv(render_mode="human")

# Vectorizar para SB3
vec_env = pettingzoo_env_to_vec_env_v1(env)
vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1)

# Cargar modelo entrenado
model = PPO.load("ppo_collaborative_pickup")

print("🎯 Evaluando modelo...\n")

# Resetear entorno
obs, _ = vec_env.reset()
done = False
total_reward = 0
step = 0

# Bucle de evaluación
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vec_env.step(action)

    # Combinar señales de finalización
    done = np.any(terminated) or np.any(truncated)

    # Render
    vec_env.render()
    pygame.event.pump()
    time.sleep(0.25)

    # Log
    reward_value = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
    print(f"Paso {step + 1} | Acción: {action} | Recompensa: {reward_value:.2f}")
    total_reward += reward_value
    step += 1

print(f"\n✅ Episodio terminado en {step} pasos | Recompensa total: {total_reward:.2f}")
vec_env.close()
pygame.quit()




