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
    FILTRO_AMARILLO_LOWER = np.array([16, 240, 0])
    FILTRO_AMARILLO_UPPER = np.array([119, 255, 255])

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

        # UTILES
        self.augmented_window = "augmented"
        self.init_window(self.augmented_window)

    def init_window(self, window, elem: np.array = None) -> None:
        """
        Método que crea la ventana para visualizar la imagen.

        """
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 320, 240)
        if elem is not None:
            cv2.imshow(window, elem)

    def use_window(self, window, elem: np.array) -> None:
        """
        Método que actualiza la ventana con la imagen.

        """
        cv2.imshow(window, elem)

    def get_duckie_mask(self, obs: np.array) -> np.array:
        """
        Método que obtiene la máscara de color amarillo en la imagen de observación del agente.

        """
        hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
        filter = cv2.inRange(hsv, self.FILTRO_AMARILLO_LOWER, self.FILTRO_AMARILLO_UPPER)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(filter, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        # bitwise = cv2.bitwise_and(obs, obs, mask=dilation)
        return dilation

    def emergency_brake(self, obs: np.array) -> bool:
        """
        Método que implementa el freno de emergencia. Dada la observación del agente, se debe
        determinar si se activa el freno de emergencia o no.

        """
        mask = self.get_duckie_mask(obs)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle = obs
        emergency_break = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if area > 100:
                rectangle = cv2.rectangle(obs, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    rectangle,
                    "Duckie",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (36, 255, 12),
                    1,
                )
                # según revisado en google, se puede calcular la distancia usando H*F/P
                # H es la altura del objeto en metros, F es la distancia focal de la cámara
                # P es la altura del objeto en pixeles (h)
                estimated_distance = 1.0 * 700 / h
                if estimated_distance >= 2.5 and estimated_distance <= 3.5:
                    cv2.putText(
                        rectangle,
                        "STOP!",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2,
                    )
                if estimated_distance < 2.5:
                    cv2.putText(
                        rectangle,
                        "EMERGENCY!",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                    )
                    alpha = 0.3
                    cv2.addWeighted(
                        cv2.rectangle(obs.copy(), (x, y), (x + w, y + h), (255, 0, 0), -1),
                        alpha,
                        obs,
                        1 - alpha,
                        0,
                        obs,
                    )
                    emergency_break = True

        if rectangle is not None:
            self.use_window(self.augmented_window, cv2.cvtColor(rectangle, cv2.COLOR_RGB2BGR))
        return emergency_break

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
            action[0] -= 0.44

        # Aquí se obtienen las observaciones y se fija la acción
        # obs consiste en un imagen de 640 x 480 x 3
        self.last_obs, _, done, _ = self.env.step(action)
        self.img = self.env.render()

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
        help=("Nombre del mapa donde correr la simulación. El mapa debe " "estar en la carpeta de mapas (gym_duckietown/maps.)."),
    )

    args = parser.parse_args()
    Desafio1(map_name=args.map_name).run()
