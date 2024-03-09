from torch.utils.tensorboard import SummaryWriter
import numpy as np
class Deterministic():
    class ExperienceReplay():
        def can_train(self):
            return False

        def __init__(self, memory_size, sample_size, state_size, action_size):
            pass
    def __init__(self,action_size,action_index=1,seed=42):
        self.action_size = action_size
        self.action_index = action_index
        self.experience_replay = self.ExperienceReplay(0,0,0,0)
        self.writer = SummaryWriter(log_dir="Random")

    def act(self,state,mask,deterministic=False):
        # check if the action is valid with the mask
        if mask[self.action_index] == 0:
            return np.random.choice(np.arange(self.action_size), p=mask/np.sum(mask))
        else:
            return self.action_index



    def can_train(self):
        return False





