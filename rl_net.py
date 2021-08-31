import torch
import torch.nn as nn
from torch.utils.data import Dataset
from functools import reduce

# assert torch.cuda.is_available() # You need to request a GPU from Runtime > Change Runtime Type
class PolicyNetwork(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super(PolicyNetwork, self).__init__()

        # lay_1_out = 3000
        # lay_2_out = 2000
        lay_1_out = 574
        lay_2_out = 344
        lay_3_out = 206
        lay_4_out = 123
        lay_5_out = 74
        lay_6_out = 44
        lay_7_out = 26
        lay_8_out = 10
        # lay_11_out = 3
        # self.roadmap = roadmap

        self.policy_net = nn.Sequential(nn.Linear(state_dimension, lay_1_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_1_out,lay_2_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_2_out,lay_3_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_3_out,lay_4_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_4_out,lay_5_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_5_out,lay_6_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_6_out,lay_7_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_7_out,lay_8_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_8_out, action_dimension))
        #
        self.policy_softmax = nn.Softmax(dim=1)
    #
    def forward(self, x):
        scores = self.policy_net(x)
        return self.policy_softmax(scores)
    #
#
class ValueNetwork(nn.Module):
    def __init__(self, state_dimension):
        super(ValueNetwork, self).__init__()

        lay_1_out = 574
        lay_2_out = 344
        lay_3_out = 206
        lay_4_out = 123
        lay_5_out = 74
        lay_6_out = 44
        lay_7_out = 26
        lay_8_out = 10
        lay_9_out = 1
        # lay_10_out = 2
        # lay_11_out = 3

        self.value_net = nn.Sequential(nn.Linear(state_dimension, lay_1_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_1_out,lay_2_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_2_out,lay_3_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_3_out,lay_4_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_4_out,lay_5_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_5_out,lay_6_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_6_out,lay_7_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_7_out,lay_8_out),
                                        nn.ReLU(),
                                        nn.Linear(lay_8_out, 1))
        #
    #
    def forward(self,x):
        return self.value_net(x)
    #
#
class AdvantageDataset(Dataset):
    def __init__(self, experience):
        super(AdvantageDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)
    #
    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        #
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp[0], chosen_exp[4]
    #
    def __len__(self):
        return self._length
    #
#
class PolicyDataset(Dataset):
    def __init__(self, experience):
        super(PolicyDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)
    #
    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        #
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp
    #
    def __len__(self):
        return self._length
    #
#
def calculate_returns(trajectories, gamma):
    for ii, trajectory in enumerate(trajectories):
        current_reward = 0
        for jj in reversed(range(len(trajectory))):
            state, probs, action_index, reward = trajectory[jj]
            ret = reward + gamma*current_reward
            trajectories[ii][jj] = (state, probs, action_index, reward, ret)
            current_reward = ret
        #
    #
#
def calculate_advantages(trajectories, value_net):
    for ii, traj in enumerate(trajectories):
        for jj, exp in enumerate(traj):#experience
            advantage = exp[4] - value_net(torch.from_numpy(exp[0]).float().unsqueeze(0))[0,0].detach().double()
            # advantage = exp[4] - value_net(exp[0].detach().numpy().float().unsqueeze(0))[0,0]
            trajectories[ii][jj] = (exp[0], exp[1], exp[2], exp[3], exp[4], advantage)
        #
    #
#


# ============================================================================
# ============================================================================




#
