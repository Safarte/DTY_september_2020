from src.car import Car
from src.ui import Interface


class Environment(object):
    NUM_SENSORS = 5

    def __init__(self, circuit, render=False):
        self.circuit = circuit
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)

        # To render the environment
        self.render = render
        if render:
            self.ui = Interface(self.circuit, self.car)
            self.ui.show(block=False)

        # Build the possible actions of the environment
        self.actions = []
        for turn_step in range(-2, 3, 1):
            for speed_step in range(-1, 2, 1):
                self.actions.append((speed_step, turn_step))

        self.count = 0

    def reward(self) -> float:
        """Computes the reward at the present moment"""
        k_center = -10
        k_stop = -10
        k_speed = 5
        k_progress = 100

        distances = self.car.distances()
        center = sum([abs(distances[i] - distances[-i-1]) for i in range(self.NUM_SENSORS)]) / 2

        reward = k_center * center + k_stop * (1 if self.car.speed == 0 else 0) + k_speed * self.car.speed + k_progress * self.circuit.progression
        
        return reward

    def isEnd(self) -> bool:
        """Is the episode over ?"""

        # Should return true if we have reached the end of an episode, False
        # otherwise
        return not self.car.in_circuit()

    def reset(self):
        self.count = 0
        self.car.reset()
        self.circuit.reset()
        return self.current_state

    @property
    def current_state(self):
        result = self.car.distances()
        result.append(self.car.speed)
        return result

    def step(self, i: int, greedy):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        self.count += 1
        self.car.action(*self.actions[i])

        state = self.current_state
        isEnd = self.isEnd()
        reward = self.reward()

        if self.render:
            self.ui.update()

        return state, reward, isEnd

    def mayAddTitle(self, title):
        if self.render:
            self.ui.setTitle(title)
