import logging
import keras.models
import os
import random
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input


class DQLAgent(object):
    def __init__(
            self, state_size=-1, action_size=-1,
            max_steps=200, gamma=1.0, epsilon=1.0, learning_rate=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.memory = deque(maxlen=2000)
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = learning_rate  # learning_rate
        if self.state_size > 0 and self.action_size > 0:
            self.model = self.build_model()

        self.count = 0

    def build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = Sequential()

        model.add(Input((self.state_size,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def updateEpsilon(self):
        """This function change the value of self.epsilon to deal with the
        exploration-exploitation tradeoff as time goes"""

        self.epsilon = 0.1 + 0.9 * np.exp(-0.001*self.count)
        # It should change (or not) the value of self.epsilon
        # self.epsilon = 0.99

    def save(self, output: str):
        self.model.save(output)

    def load(self, filename):
        if os.path.isfile(filename):
            self.model = keras.models.load_model(filename)
            self.state_size = self.model.layers[0].input_shape[1]
            self.action_size = self.model.layers[-1].output.shape[1]
            return True
        else:
            logging.error('no such file {}'.format(filename))
            return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def expert(self, state):
        v = state[-1][5]
        dist = state[-1][2]
        gg, g, d, dd = state[-1][0], state[-1][1], state[-1][3], state[-1][4]
        a = 0.05
        d_arret = v*v/(2*a)
        acc_threshold = 0.7

        taux_d = (dist-d_arret)/(dist+d_arret)

        if taux_d <= 0 :
            accelerer = -1
        elif taux_d < acc_threshold :
            accelerer = 0
        else:
            accelerer = 1
        
        if (gg == 0 or g == 0 or d == 0 or dd == 0):
            return 7
        
        turn_threshold = 0.01
        turn_threshold2 = 0.001
        turn = 0
        t1 = (gg - dd) / (gg + dd)
        if t1 < -turn_threshold:
            turn += 2
        elif t1 > turn_threshold:
            turn -= 2
        
        t2 = (g - d) / (d + g)
        if t2 < -turn_threshold2:
            turn += 1
        elif t2 > turn_threshold2:
            turn -= 1
        
        if turn < 0:
            turn = max(-2, turn)
        else:
            turn = min(2, turn)

        # print((accelerer, turn))
        actions = [(-1, -2), (0, -2), (1, -2), (-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2)]
        return actions.index((accelerer, turn))

    def act(self, state, greedy=True):
        if np.random.rand() <= self.epsilon and not greedy:
            return self.expert(state)

        return np.argmax(self.model.predict(state))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.updateEpsilon()

    def setTitle(self, env, train, name, num_steps, returns):
        h = name
        if train:
            h = 'Iter {} ($\epsilon$={:.2f})'.format(self.count, self.epsilon)
        end = '\nreturn {:.2f}'.format(returns) if train else ''

        env.mayAddTitle('{}\nsteps: {} | {}{}'.format(
            h, num_steps, env.circuit.debug(), end))

    def run_once(self, env, train=True, greedy=False, name=''):
        self.count += 1
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        returns = 0
        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            action = self.act(state, greedy=greedy)
            next_state, reward, done = env.step(action, greedy)
            next_state = np.reshape(next_state, [1, self.state_size])

            if train:
                self.remember(state, action, reward, next_state, done)

            returns = returns * self.gamma + reward
            state = next_state
            if done:
                return returns, num_steps

            self.setTitle(env, train, name, num_steps, returns)

        return returns, num_steps

    def train(
            self, env, episodes, minibatch, output='weights.h5', render=False):
        for e in range(episodes):
            r, _ = self.run_once(env, train=True, greedy=False)
            print("episode: {}/{}, return: {}, e: {:.2}, laps: {:.2}".format(
                e, episodes, r, self.epsilon, env.circuit.laps + env.circuit.progression))

            if len(self.memory) > minibatch:
                self.replay(minibatch)
                self.save(output)

        # Finally runs a greedy one
        r, n = self.run_once(env, train=False, greedy=True)
        self.save(output)
        print("Greedy return: {} in {} steps".format(r, n))
