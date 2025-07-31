import numpy as np
import torch
import A2C_GPU as A

# a = np.array([1,2,3])
# print(a)
#
# c = torch.from_numpy(a)
# print(c)
#
# b = [1,2,3]
# print(b)
#
# d = torch.tensor(b)
# print(d)
# c = A.Config()
#
# a = A.ActorCritic(state_dim=20, action_dim=20, config=c)
# print(a.action_space)
# print(a.observation_space)
# print(a.P)
# print(a.P.copy())

a = torch.tensor([1,2,3])
b = a.T
c = a.reshape(3,1)
print(a)
print(b)
print(c)