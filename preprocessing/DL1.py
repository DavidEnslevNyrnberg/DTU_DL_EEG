import os, csv
os.chdir(os.getcwd())
# ~~~~ NUMPY exercises ~~~~
# EX.1
import numpy as np

# EX.2
print(np.__version__)
np.show_config()

# EX.3
np.zeros(10)

# EX.11
np.eye(3)

# EX.17 [Guess -> result]
0 * np.nan # nan -> nan
np.nan == np.nan # True -> False
np.inf > np.nan # False -> False
np.nan - np.nan # nan -> nan
np.nan in set([np.nan]) # nan -> True
0.3 == 3*0.1 # False -> False


## 1.4 Numpy
# set dimentions
print("\nNN USING NUMPY")
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

l_rate = 1e-6
for t in range(500):
    #forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    grad_y_pred = 2*(y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= l_rate * grad_w1
    w2 -= l_rate * grad_w2

import torch
print("\nNN USING TORCH")

dtype = torch.float
device = torch.device("cpu")

xNN = torch.randn(N, D_in, device=device, dtype=dtype)
yNN = torch.randn(N, D_out, device=device, dtype=dtype)

w1NN = torch.randn(D_in, H, device=device, dtype=dtype)
w2NN = torch.randn(H, D_out, device=device, dtype=dtype)

# help(xNN.mm)
for t in range(500):
    h = xNN.mm(w1NN)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2NN)

    loss = (y_pred - yNN).pow(2).sum().item()
    if t% 100 == 99:
        print(t, loss)

    grad_y_pred = 2 * (y_pred - yNN)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2NN.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = xNN.t().mm(grad_h)

    w1NN -= l_rate * grad_w1
    w2NN -= l_rate * grad_w2


# better optimizer
print("\nOPTIMIZE NN WITH AUTOGRAD")
w1NN2 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2NN2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

for t in range(500):
    y_pred = xNN.mm(w1NN2).clamp(min=0).mm(w2NN2)

    loss = (y_pred - yNN).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1NN2 -= l_rate * w1NN2.grad
        w2NN2 -= l_rate * w2NN2.grad

        w1NN2.grad.zero_()
        w2NN2.grad.zero_()

