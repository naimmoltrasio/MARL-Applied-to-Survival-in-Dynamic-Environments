from ppo import *

# Cargar modelo
model = PPO.load("ppo_collaborative_env")

# Resetear entorno
obs = env_wrapped.reset()

# Ejecutar un episodio
done = [False]
while not all(done):
    action, _states = model.predict(obs)
    obs, reward, done, info = env_wrapped.step(action)
    env.render()
