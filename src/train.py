from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
#from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import joblib

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def __init__(self, env = env, discount_factor=0.6, epsilon=0.15, horizon=1000):
        """
            The agent class that will be used to train the agent using Q-learning algorithm.
            Args:
                - env: gym environment
                - discount_factor: float, the discount factor for the Q-learning algorithm
                - epsilon: float, the epsilon value for the epsilon-greedy policy
                - episodes: int, the number of episodes to train the agent 
                - max_steps: int, the maximum number of steps in an episode
        """
        
        self.env = env
        self.action_space = env.action_space
        self.q_table = RandomForestRegressor(n_estimators=50, min_samples_split=2, max_features=8).fit(np.zeros((1,env.observation_space.shape[0]+1)),np.zeros(1))
        self.observation_space = env.observation_space
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.horizon = horizon

    def act(self, observation, use_random=False):

        if use_random or np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            Qsa = []
            for a in range(self.env.action_space.n):
                sa = np.append(observation,a).reshape(1,-1)
                Qsa.append(self.q_table.predict(sa))
            return np.argmax(Qsa)
                

    def save(self):
        """
        Saves the agent's current state to a file specified by the path.

        Args:
            - path (str): The file path where the agent's state should be saved.

        """
        joblib.dump(self.q_table, "./Q_function.joblib")


    def load(self):
        """
        Loads the agent's state from a file specified by the path (HARDCODED).
        """
 
        self.q_table = joblib.load('.\src\Q_function.joblib')
            

    def collect_samples(self, use_random=False):
        ''' 
        Collect samples from the environment using the current policy
        Args:
            - use_random: bool, whether to use random policy or the current policy
        Returns:
            - S: list, the list of states
            - A: list, the list of actions
            - R: list, the list of rewards
            - S2: list, the list of next states
            - D: list, the list of done flags
        '''
        S, A, R, S2, D = [], [], [], [], []
        for _ in range(self.horizon): #tqdm(range(self.horizon), desc="Collecting samples"):
            state, _ = self.env.reset()
            action = self.act(state, use_random)
            next_state, reward, done, trunc , _ = self.env.step(action)
            S.append(state)
            A.append(action)
            R.append(reward)
            S2.append(next_state)
            D.append(done)
            if done or trunc:
                state = self.env.reset()
            else:
                state = next_state
        S = np.array(S)
        A = np.array(A).reshape(-1, 1)
        R = np.array(R)
        S2 = np.array(S2)
        D = np.array(D)
            
        return S, A, R, S2, D
    
    def fqi(self, S, A, R, S2, D, nb_iter):
        ''' 
        Fitted Q-Iteration algorithm
        Args:
            - S: list, the list of states
            - A: list, the list of actions
            - R: list, the list of rewards
            - S2: list, the list of next states
            - D: list, the list of done flags
            - nb_iter: int, the number of iterations
        '''

        nb_samples = S.shape[0]
        Qfunctions = []
        SA = np.append(S,A,axis=1)
        for iter in range(nb_iter): #tqdm(range(nb_iter), desc="Fitted Q-Iteration"):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples, self.env.action_space.n))
                for a2 in range(self.env.action_space.n):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + self.discount_factor*(1-D)*max_Q2
            Q = RandomForestRegressor(n_estimators=50, min_samples_split=2, max_features=8)
            Q.fit(SA,value)
            Qfunctions.append(Q)
            self.q_table = Qfunctions[-1]

    def train(self):
        '''
        Train the agent using the Fitted Q-Iteration algorithm
        '''

        S, A, R, S2, D = self.collect_samples(use_random=True)
        self.fqi(S, A, R, S2, D, 100)
        for loop in range(10):
            print("loop",loop)
            S1, A1, R1, S21, D1 = self.collect_samples(use_random=False)
            S = np.append(S,S1,axis=0)
            A = np.append(A,A1,axis=0)
            R = np.append(R,R1)
            S2 = np.append(S2,S21,axis=0)
            D = np.append(D,D1)
            self.fqi(S, A, R, S2, D, 100)

        
