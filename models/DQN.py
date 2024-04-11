import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os


class DQN(nn.Module):
    class ExperienceReplay():

        # state is a numpy array of shape (memory_size, state_size)

        memory_size = 0
        sample_size = 0

        def __init__(self, memory_size, sample_size, state_size, action_size):
            self.memory_size = memory_size
            self.sample_size = sample_size
            shape_state = (memory_size,) + (state_size,)
            shape = (memory_size, 1)
            self.state = np.empty(shape_state)
            self.action = np.empty(shape)
            self.reward = np.empty(shape)
            self.next_state = np.empty(shape_state)
            self.done = np.empty(shape)
            self.action_mask = np.empty((memory_size, action_size))
            self.head = 0
            self.size = 0

        def add_step(self, state, action, reward, next_state, done, action_mask=None):
            self.state[self.head] = state
            self.action[self.head] = action
            self.reward[self.head] = reward
            self.next_state[self.head] = next_state
            self.done[self.head] = done
            self.head = (self.head + 1) % self.memory_size
            if action_mask is not None:
                self.action_mask[self.head] = action_mask
            if self.size < self.memory_size:
                self.size += 1

        def sample(self, batch_index):
            return self.state[batch_index], self.action[batch_index], self.reward[batch_index], self.next_state[
                batch_index], self.done[batch_index], self.action_mask[batch_index]

        def can_train(self):
            return self.size >= self.sample_size

    def __init__(self, memory_size, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, lr, state_size, action_size,
                 env_name, seed=0, agent_index=0
                 ):
        super(DQN, self).__init__()
        # create a folder for the tensorboard logs
        os.makedirs(env_name + "_DQN/" + str(agent_index), exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=env_name + "_DQN/" + str(agent_index))
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.experience_replay = self.ExperienceReplay(
            memory_size, batch_size, state_size, action_size)
        self.gamma = gamma
        self.seed = np.random.seed(seed)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.number_update = 0
        self.epsilon_update = 0
        self.Q_value_get = 0
        self.lr = lr
        self.action_size = action_size
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.policy_type = "DQN"

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.network(x)

    def act(self, state, mask=None, deterministic=False):
        with torch.no_grad():
            if deterministic:
                # Always choose the action with the highest Q-value
                state = torch.from_numpy(state).to(self.device).float()
                q_values = self.network(state)
                if mask is not None:
                    masked_q_values = q_values.clone()
                    # Set illegal moves to negative infinity
                    masked_q_values[mask == 0] = float('-inf')
                    return torch.argmax(masked_q_values).item()
                else:
                    return torch.argmax(q_values).item()
            else:
                if np.random.rand() < self.epsilon:
                    if mask is not None:
                        # Get indices of legal actions
                        legal_actions = np.where(mask == 1)[0]
                        if len(legal_actions) > 0:
                            # Convert mask to probabilities
                            legal_probs = np.ones(len(mask)) * mask
                            # Normalize probabilities
                            legal_probs /= np.sum(legal_probs)
                            return np.random.choice(self.action_size, p=legal_probs)
                        else:
                            # If all actions are illegal, choose randomly from all actions
                            return np.random.randint(len(mask))
                    else:
                        return torch.randint(0, self.action_size, (1,), dtype=torch.long)
                else:
                    state = torch.from_numpy(state).to(self.device).float()
                    q_values = self.network(state)
                    if mask is not None:
                        masked_q_values = q_values.clone()
                        # Set illegal moves to negative infinity
                        masked_q_values[mask == 0] = float('-inf')
                        return torch.argmax(masked_q_values).item()
                    else:
                        return torch.argmax(q_values).item()

    def train_agent(self):

        batch_index = np.random.choice(
            (self.experience_replay.size), self.batch_size)
        state, action, reward, next_state, done, mask = self.experience_replay.sample(
            batch_index)
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).long().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        mask = torch.from_numpy(mask).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        prediction = self.network(state).gather(1, action)
        with torch.no_grad():
            target = reward + self.gamma * \
                self.target_network(next_state).detach().max(1)[
                    0].unsqueeze(1) * (1 - done)
        loss = self.loss(prediction, target)
        self.writer.add_scalar("Q loss", loss, self.number_update)
        self.number_update += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(0.01)

    def soft_update(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def update_epsilon(self):
        self.writer.add_scalar("Epsilon", self.epsilon, self.epsilon_update)
        self.epsilon_update += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path, round_):
        os.makedirs(path, exist_ok=True)
        torch.save({"network": self.network.state_dict(), "target_network" : self.target_network.state_dict()}, path + f"/model{round_}.pth")

    def load_model(self, path='dqn.pth'):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.network.eval()
        self.target_network.eval()

    def get_action_distribution(self, state, mask):
        """
        Get the action distribution of the agent
        :param state: The state of the environment
        :return: The action distribution of the agent
        """
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device).float()
            q_values = self.network(state)
            masked_q_values = q_values.clone()
            masked_q_values[mask == 0] = float('-inf')
            return F.softmax(masked_q_values, dim=0).cpu().numpy()
