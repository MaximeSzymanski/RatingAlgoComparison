import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class PPO(nn.Module):
    class ExperienceReplay():
        def __init__(self, minibatch_size, buffer_size, state_size, num_workers=1, action_size=6, horizon=128):
            self.minibatch_size = minibatch_size
            self.buffer_size = buffer_size
            self.state_size = state_size
            self.action_size = action_size
            self.num_worker = num_workers
            self.horizon = horizon
            self.reset_buffer(horizon, state_size, action_size=action_size)

        def reset_buffer(self, horizon, state_size, action_size=6):
            transformed_buffer_size = (horizon,) + (self.num_worker,)
            buffer_state_size = transformed_buffer_size + state_size
            mask_size = (horizon,) + (self.num_worker,) + (action_size,)

            self.actions = np.empty(transformed_buffer_size, dtype=np.int32)
            self.actions_mask = np.empty(mask_size, dtype=np.int32)
            self.rewards = np.empty(transformed_buffer_size, dtype=np.float32)
            self.states = np.empty(buffer_state_size, dtype=np.float32)
            self.next_states = np.empty(buffer_state_size, dtype=np.float32)
            self.dones = np.empty(transformed_buffer_size, dtype=np.int32)
            self.old_log_probs = np.empty(transformed_buffer_size, dtype=np.float32)
            self.advantages = np.empty(transformed_buffer_size, dtype=np.float32)
            self.values = np.empty(transformed_buffer_size, dtype=np.float32)
            self.head = 0
            self.size = 0

        def add_step(self, state, action, reward, next_state, done, value, old_log_prob,action_mask=None):
            # assert the buffer is not full
            assert self.size < self.buffer_size, "Buffer is full"
            if action_mask is None:
                action_mask = np.ones_like(action)
            self.states[self.head] = state
            self.actions[self.head] = action
            value = np.squeeze(value)
            self.values[self.head] = value
            self.old_log_probs[self.head] = old_log_prob
            self.rewards[self.head] = reward
            self.next_states[self.head] = next_state
            self.dones[self.head] = done
            self.actions_mask[self.head] = action_mask
            self.head = (self.head + 1) % self.horizon
            self.size += 1
            # check if the buffer is full

        def get_minibatch(self):
            # assert the buffer is not empty
            assert self.size > self.minibatch_size, "Buffer is empty"
            # get random indices
            indices = np.random.randint(0, self.size, size=self.minibatch_size)
            # return the minibatch
            return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], \
                self.dones[indices], self.old_log_probs[indices], self.values[indices], self.actions_mask[indices]

        def flatten_buffer(self):
            # flatten the buffer
            self.states = self.states.reshape(-1, self.states.shape[-1])
            self.actions = self.actions.flatten()
            self.actions_mask = self.actions_mask.reshape(-1, self.actions_mask.shape[-1])
            self.rewards = self.rewards.flatten()
            self.next_states = self.next_states.reshape(-1,self.next_states.shape[-1])
            self.dones = self.dones.flatten()
            self.old_log_probs = self.old_log_probs.flatten()
            self.values = self.values.flatten()
            self.advantages = self.advantages.flatten()



        def clean_buffer(self):
            self.reset_buffer(self.horizon, self.state_size,self.action_size)

        def can_train(self):
            return self.size >= self.horizon
        def __len__(self):
            return self.size
    def __init__(self, state_size, action_size, num_steps, batch_size,env_name='connect_four_v3', num_workers=1):
        """
        This class implements the Proximal Policy Optimization algorithm
        :param state_size: The size of the state space
        :param action_size: The size of the action space
        :param num_workers: The number of workers
        :param num_steps: The number of steps
        :param batch_size: The batch size
        """
        super(PPO, self).__init__()


        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)

        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.number_epochs = 0
        self.device = torch.device("cuda:0")
        print(self.parameters())
        self.to(self.device)
        self.optimizer = Adam(self.parameters(), lr=0.0003)

        self.experience_replay = self.ExperienceReplay(minibatch_size=batch_size, buffer_size=num_steps, state_size=(state_size,), num_workers=num_workers, action_size=action_size, horizon=num_steps)

        self.writer = SummaryWriter(log_dir=env_name)
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ortogonal_initialization()

    def ortogonal_initialization(self):

        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0)


    def forward(self, x,action_mask=None):
        logits = self.actor(x)
        value = self.critic(x)
        if action_mask is not None:
            
            logits = logits * action_mask
        dist = Categorical(logits)

        return dist, value

    def get_action(self, obs, action_mask = None, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            action_mask = torch.from_numpy(action_mask).float().to(self.device)
            dist, value = self.forward(obs,action_mask)
            if deterministic:
                action = torch.argmax(dist.probs)

            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

    def decay_learning_rate(self, optimizer, decay_rate=0.99):
        print("Decaying learning rate")
        self.writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], self.number_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

    def save_model(self, path='ppo.pth'):
        torch.save(self.state_dict(), path)

    def load_model(self, path='ppo.pth'):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def compute_advantages(self, gamma=0.99, lamda=0.95):

        for worker in range(self.experience_replay.num_worker):
            values = self.experience_replay.values[:, worker]

            advantages = np.zeros(self.num_steps, dtype=np.float32)
            last_advantage = 0
            last_value = values[-1]
            for i in reversed(range(self.num_steps)):
                mask = 1 - self.experience_replay.dones[i, worker]
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                delta = self.experience_replay.rewards[i, worker] + gamma * last_value - values[i]
                last_advantage = delta + gamma * lamda * last_advantage
                advantages[i] = last_advantage
                last_value = values[i]
            
            
            self.experience_replay.advantages[:, worker] = advantages
        pass
        self.experience_replay.flatten_buffer()
        advantages = self.experience_replay.advantages
        return advantages



    def train_agent(self):

        advantages = self.compute_advantages(gamma=0.99, lamda=0.95)
        # convert the data to torch tensors
        states = torch.from_numpy( self.experience_replay.states).to( self.device)
        actions = torch.from_numpy( self.experience_replay.actions).to( self.device)
        old_log_probs = torch.from_numpy(self.experience_replay.old_log_probs).to(self.device).detach()

        advantages = torch.from_numpy(advantages).to( self.device)
        values = torch.from_numpy( self.experience_replay.values).to( self.device)
        actions_mask = torch.from_numpy( self.experience_replay.actions_mask).to( self.device)
        returns = advantages + values

        # split the data into batches
        numer_of_samples = self.num_steps *  self.experience_replay.num_worker

        number_mini_batch = numer_of_samples //  self.experience_replay.minibatch_size
        assert number_mini_batch > 0, "batch size is too small"
        assert numer_of_samples %  self.experience_replay.minibatch_size == 0, "batch size is not a multiple of the number of samples"

        indices = np.arange(numer_of_samples)
        np.random.shuffle(indices)
        for _ in range(4):
            for batch_index in range(number_mini_batch):
                start = batch_index *  self.experience_replay.minibatch_size
                end = (batch_index + 1) *  self.experience_replay.minibatch_size
                indice_batch = indices[start:end]
                advantages_batch = advantages[indice_batch]
                normalized_advantages = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                self.number_epochs += 1

                new_dist, new_values = self.forward(states[indice_batch],actions_mask[indice_batch])
                log_pi = new_dist.log_prob(actions[indice_batch])

                ratio = torch.exp(log_pi - old_log_probs[indice_batch].detach())
                surr1 = ratio * normalized_advantages
                surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * normalized_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values.squeeze(), returns[indice_batch])

                entropy_loss = new_dist.entropy().mean()
                self.writer.add_scalar('entropy', entropy_loss, self.number_epochs)
                self.writer.add_scalar('critic', critic_loss, self.number_epochs)
                self.writer.add_scalar('actor', actor_loss, self.number_epochs)

                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.experience_replay.clean_buffer()
            # agent.decay_learning_rate(optimizer)

        # create the dataset

