import pygame
import time
from env.custom_environment import CollaborativePickUpEnv

key_to_action_agent1 = {
    pygame.K_a: 0,  # izquierda
    pygame.K_d: 1,  # derecha
    pygame.K_w: 2,  # arriba
    pygame.K_s: 3,  # abajo
    pygame.K_e: 4,  # recoger
}
key_to_action_agent2 = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_DOWN: 3,
    pygame.K_RSHIFT: 4,
}

def main():
    env = CollaborativePickUpEnv(render_mode="human", config="config3")
    obs, _ = env.reset()
    env.render()
    running = True
    action1 = 4  # default: no moverse
    action2 = 4

    print("\n🎮 CONTROL MANUAL INICIADO")
    print("Agente 1: WASD + E | Agente 2: Flechas + RSHIFT\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key in key_to_action_agent1:
                    action1 = key_to_action_agent1[event.key]
                else:
                    action1 = 4  # inacción

                if event.key in key_to_action_agent2:
                    action2 = key_to_action_agent2[event.key]
                else:
                    action2 = 4

                actions = {
                    "agent_1": action1,
                    "agent_2": action2
                }

                obs, rewards, terminations, truncations, infos = env.step(actions)
                env.render()
                pygame.event.pump()

                print(f"A1: {action1} | A2: {action2} | Rewards: {rewards}")
                print(f"Obs A1: {obs['agent_1']} | Obs A2: {obs['agent_2']}\n")

                # Reset acciones después del paso
                action1 = 4
                action2 = 4

                # Reiniciar episodio si termina
                if all(terminations.values()) or all(truncations.values()):
                    print("Episodio terminado\n")
                    obs, _ = env.reset()
                    env.render()
                    time.sleep(1)

    pygame.quit()
    print("Control manual finalizado")

if __name__ == "__main__":
    main()