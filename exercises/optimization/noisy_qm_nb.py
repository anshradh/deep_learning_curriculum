# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import einops
from fancy_einsum import einsum
from tqdm import tqdm

from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
d = int(10**3)
H = torch.diag(1 / torch.arange(1, d + 1, device=device))
C = H.clone()
# %%
loss_fn = lambda theta: 0.5 * einsum("d1, d1 d2, d2 -> ", theta, H, theta)
# %%
test_theta = torch.randn(d, device=device)
loss_fn(test_theta)

# %%
gradient_query = lambda theta: einsum("d1 d2, d2 -> d1", H, theta) + torch.randn(d, device=device)
# %%
test_theta = torch.randn(d, device=device)
gradient_query(test_theta)
# %%
average_gradient_query = lambda theta, B: torch.stack([gradient_query(theta) for _ in range(B)]).mean(0)
# %%
test_theta = torch.randn(d, device=device)
average_gradient_query(test_theta, 10)
# %%
theta = torch.randn(d, device=device)
lr = 1e-2
beta = 0.9
m = torch.zeros(d, device=device)
batch_size = 32
for _ in tqdm(range(5000)):
    loss = loss_fn(theta)
    if _ % 100 == 0:
        print(loss.item())
    grad = average_gradient_query(theta, batch_size)
    m = beta * m + (1 - beta) * grad
    theta = theta - lr * m
# %%
lr = 1e-2
beta1 = 0.9
beta2 = 0.999
m = torch.zeros(d, device=device)
v = torch.zeros(d, device=device)
batch_size = 32
theta = torch.randn(d, device=device)
for _ in tqdm(range(5000)):
    loss = loss_fn(theta)
    if _ % 100 == 0:
        print(loss.item())
    grad = average_gradient_query(theta, batch_size)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_t = m / (1 - beta1**(1 + _))
    v_t = v / (1 - beta2**(1 + _))
    theta = theta - lr * m_t / (torch.sqrt(v_t) + 1e-8)
# %%
