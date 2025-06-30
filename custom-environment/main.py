from env.custom_environment import *

if __name__ == "__main__":

    env = CollaborativePickUpEnv()
    obs, infos = env.reset()

    for _ in range(1000):
        actions = {
            "agent_1": env.action_space("agent_1").sample(),
            "agent_2": env.action_space("agent_2").sample(),
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

    pygame.quit()




