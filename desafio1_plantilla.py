"""
Planilla para el desarollo del desafio 1:  Freno de emergencia.

"""

import argparse
import cv2
import numpy as np
import pyglet

from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv


class Desafio1:

    FILTRO_AMARILLO_LOWER = np.array([0, 0, 0])
    FILTRO_AMARILLO_UPPER = np.array([0, 0, 0])

    def __init__(self, map_name):
        self.env = DuckietownEnv(
            seed=1,
            map_name=map_name,
            draw_curve=False,
            draw_bbox=False,
            domain_rand=False,
            frame_skip=1,
            distortion=False,
        )

        # Esta variable nos ayudara a definir la acción que se ejecutará en el siguiente paso (loop)
        self.last_obs = self.env.reset()
        self.env.render()

        # Registrar el handler
        self.key_handler = key.KeyStateHandler()
        self.env.unwrapped.window.push_handlers(self.key_handler)


    def emergency_brake(self, obs: np.array) -> bool:
        """
        Método que implementa el freno de emergencia. Dada la observación del agente, se debe
        determinar si se activa el freno de emergencia o no.

        """        

        # Recomendación, dividir la función en 3 partes:
        # 1. Obtener la máscara de color amarillo
        # 2. Aplicar operaciones morfológicas para eliminar ruido
        # 3. Obtener bounding boxes y determinar si se activa el freno de emergencia

        # Utilice funciones auxiliares para organizar el código
        return False


    def update(self, dt):
        """
        Este método se encarga de ejecutar un loop de simulación en el ambiente Duckietown.
        En cada paso se ejecuta una acción que se define en este método y se obtienen las 
        observaciones del agente en el ambiente.

        Este método debe usar la última observación obtenida por el agente (la variable
        self.last_obs) y realizar una acción en base a esta. Luego debe guardar la observación
        actual para la siguiente iteración.

        """

        if self.last_obs is None:
            self.last_obs = self.env.reset()


        action = np.array([0.0, 0.0])
        
        # Tele - operación: Control manual del agente dado por las teclas
        if self.key_handler[key.UP]:
            action[0] += 0.44

        if self.key_handler[key.DOWN]:
            action[0] -= 0.44

        if self.key_handler[key.LEFT]:
            action[1] += 1

        if self.key_handler[key.RIGHT]:
            action[1] -= 1

        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if self.key_handler[key.LSHIFT]:
            action *= 1.5

        # Se ejecuta el freno de emergencia
        if self.emergency_brake(self.last_obs):
            pass

        # Aquí se obtienen las observaciones y se fija la acción
        # obs consiste en un imagen de 640 x 480 x 3
        self.last_obs, _, done, _ = self.env.step(action)
        self.env.render()

        if done:
            self.last_obs = self.env.reset()

    def run(self):
        """
        Arranca la simulación del ambiente Duckietown.
        """

        # Fijar la frecuencia de la simulación. Se ejecutara el método update() cada 1/fps segundos
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        pyglet.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-name",
        default="loop_obstacles",
        help=(
            "Nombre del mapa donde correr la simulación. El mapa debe "
            "estar en la carpeta de mapas (gym_duckietown/maps.)."
        ),
    )

    args = parser.parse_args()
    Desafio1(map_name=args.map_name).run()
