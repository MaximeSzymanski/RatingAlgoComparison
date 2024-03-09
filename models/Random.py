from torch.utils.tensorboard import SummaryWriter
import numpy as np
class Random():
    class ExperienceReplay():
        def can_train(self):
            return False

        def __init__(self, memory_size, sample_size, state_size, action_size):
            pass
    def __init__(self,action_size,seed=42):
        self.action_size = action_size
        self.seed = seed
        self.experience_replay = self.ExperienceReplay(0,0,0,0)
        self.writer = SummaryWriter(log_dir="Random")

    def act(self,state,mask,deterministic=False):
        random_action = np.random.choice(np.arange(self.action_size), p=mask/np.sum(mask))
        return random_action


    def can_train(self):
        return False





