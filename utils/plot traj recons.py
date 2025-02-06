import torch
from model import Traj
from hp import hyperparameters_real
import numpy as np
import os
import pickle as pk
import matplotlib.pyplot as plt

# loss_fn1 = torch.nn.L1Loss(reduction='none')
# loss_fn2 = torch.nn.L1Loss(reduction='sum')
# loss_fn3 = torch.nn.L1Loss(reduction='mean')

# a = torch.rand((2,4))
# b = torch.rand((2,1,4))

# mae = loss_fn1(a,b)
# print(loss_fn1(a,b))
# print(loss_fn2(a,b))
# print(loss_fn3(a,b))

# a = torch.tensor([1,2,3,4,5,100,200]).float()
# print(a.std(), a.mean())

Traj_weights = 'best_Traj.pth'
state_dict = torch.load(Traj_weights, map_location=torch.device('cpu'))
model = Traj(hyperparameters_real)
model.load_state_dict(state_dict, strict=False)

with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy_go', 'TA_25.pkl'), 'rb') as f:
    traj = torch.from_numpy(np.deg2rad(pk.load(f).astype('float32'))).unsqueeze(dim=0)

trajj = model(traj).reshape(1,300,-1)

_, axes = plt.subplots(5, 5, figsize=(10,10))
for axis, j in zip(axes.reshape(25), list(range(1,26))):
    axis.title.set_text(f'Joint: {j}')

for i in range(25):
    axes[int(i//5)][i%5].scatter(range(300), traj[0,:,i].detach().numpy(), c='black', s=2)
    axes[int(i//5)][i%5].scatter(range(300), trajj[0,:,i].detach().numpy(), c='r', s=2, alpha=0.1)
plt.tight_layout()
plt.show(block=True)