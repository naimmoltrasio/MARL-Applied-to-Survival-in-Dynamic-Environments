from env.custom_environment import *
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":

    env = CollaborativePickUpEnv()
    obs, infos = env.reset()

    for _ in range(100):
        actions = {
            "agent_1": env.action_space("agent_1").sample(),
            "agent_2": env.action_space("agent_2").sample(),
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

    pygame.quit()
    """
    TEST PARALLEL API
    env = CollaborativePickUpEnv()
    parallel_api_test(env, num_cycles=1_000_000)
    """

    """
    TEST RENDER CMD
    env = CollaborativePickUpEnv()
    observations, infos = env.reset()

    done = {"prisoner": False, "guard": False}

    for _ in range(1000000):  # ejecutar 50 pasos o hasta que termine
        env.render()

        # Generar acciones aleatorias válidas basadas en la máscara de acción
        actions = {}
        for agent in env.agents:
            mask = observations[agent]["action_mask"]
            valid_actions = [i for i, m in enumerate(mask) if m]
            actions[agent] = random.choice(valid_actions)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Imprimir recompensas y estado de finalización
        print(f"Recompensas: {rewards}")
        print(f"Terminaciones: {terminations}")
        print(f"Truncamientos: {truncations}\n")

        if all(terminations.values()) or all(truncations.values()):
            env.render()
            print("Fin del episodio.")
            break
    """



