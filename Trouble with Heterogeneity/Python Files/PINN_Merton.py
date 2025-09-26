# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:01:56 2025

@author: Patrick Brock

Implements a simple PINN for the Merton Model.
"""

#%% 
# ----------
# Libraries
# ----------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,           # Default font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # x and y labels font size
    'xtick.labelsize': 12,     # x-axis tick label font size
    'ytick.labelsize': 12,     # y-axis tick label font size
    'legend.fontsize': 12,     # Legend font size
})

from math import pi, cos

import pickle
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Set directory
directory = "C:/Users/pbrock/Downloads/"
os.chdir(directory)

#Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set this to true to re-train the Neural Network.
training = True

# -------------------------
# Switch for trial solution
# -------------------------
USE_TRIAL_SOLUTION = True   # Set to False for baseline

#%%
# -------------------------
# Model / economic params
# -------------------------
rho   = 0.05
gamma = 2.0
r     = 0.02
mu    = 0.06
sigma = 0.2
T     = 1.0

#%%  
# ---------------
# Hyperparameters
# ---------------

#-- Asset Grid
#Range
a_min, a_max = 0.01, 10.0
#Number of Gridpoints in Asset Grid
N_assets = 50

#-- Terminal Value Function
#Number of Gridpoints
N_terminal = N_assets

#-- Neural Network
#Neurons in hidden layers
hidden_dim = 64
#dropout probability (Set to zero as dropout makes the results worse)
dropout_p = 0.0
#learning rate
lr = 1e-4
#weight decay
weight_decay = 1e-6    # small L2 regularization
#epochs
epochs = 30_000
#Print interval of Epoch
print_every = 50
# Gradient clipping to avoid explosive updates
GRAD_CLIP_NORM = 5.0

#-- Numerical epsilons (stability)
EPS = 1e-8            # general small epsilon
MAX_PI = 5            # bound on pi via tanh scaling

#-- Loss weights
w_pde = 1.0
#Weight of terminal Value Function Loss
w_term = 5.0
#Weight of shape violations
w_grad = 5.0

#%%
# -------------------------
# Utility Function
# -------------------------
def u(c):
    """
    Torch
    """
    if gamma == 1.0:
        return torch.log(c + EPS)
    else: #No EPS
        return (c).pow(1.0 - gamma) / (1.0 - gamma)

def u_np(c):
    """
    Numpy
    """
    if gamma == 1.0:
        return np.log(c + EPS)
    else: #no EPS
        return (c)**(1.0 - gamma) / (1.0 - gamma)

#%%
# --------------------------------------------------------
# Neural Network (3 hidden layers + ResNet skip + dropout)
# --------------------------------------------------------
class ResNetPINN(nn.Module):
    def __init__(self, in_dim=2, hidden=32, dropout_p=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, a):
        x = torch.cat([t, a], dim=1)
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        h = self.act(self.fc2(h))
        h = self.dropout(h)
        skip = self.input_proj(x)
        h = h + skip
        h = self.act(self.fc3(h))
        h = self.dropout(h)
        out = self.head(h)
        if USE_TRIAL_SOLUTION:
            # NEW: Multiply NN output with shape restriction
            V = out * (1.0 + 1.0 / (torch.abs(a)))
            return V
        else:
            out

#%% 
# -------------------------
# Autograd derivatives
# -------------------------
def derivatives(model, t, a):
    V = model(t, a)  # (N,1)
    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True)[0]
    V_a = torch.autograd.grad(V, a, grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True)[0]
    V_aa = torch.autograd.grad(V_a, a, grad_outputs=torch.ones_like(V_a),
                               create_graph=True, retain_graph=True)[0]
    return V, V_t, V_a, V_aa

#%%
# -------------------------
# Data samplers
# -------------------------

def sampler_terminal(N):
    """
    Grid for the terminal Value Function
    """
    t = np.ones((N,1), dtype=np.float32) * T
    a = np.linspace(a_min, a_max, N, dtype=np.float32).reshape(-1,1)
    return t, a

def discretize_assets(amin, amax, n_a):
    """
    Exponential asset grid: clusters more points near amin.
    """
    # max value of transformed uniform grid
    ubar = np.log(1 + np.log(1 + amax - amin))

    # uniform grid in transformed space
    u_grid = np.linspace(0, ubar, n_a)

    # double-exponentiate transform
    a_grid = amin + np.exp(np.exp(u_grid) - 1) - 1
    return a_grid.astype(np.float32).reshape(-1, 1)

def sampler_collocation_exp_random_pairs(N, amin=a_min, amax=a_max, seed=46845):
    """
    Collocation sampler with optional seeding for reproducibility.
    - sample N random times t ~ U[0,T)
    - sample N asset values by drawing (with replacement) from exponential asset grid
    - returns N pairs (t, a)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)   # modern, safe RNG
    else:
        rng = np.random.default_rng()

    # random times
    t = rng.random((N, 1), dtype=np.float32) * T

    # exponential asset grid
    a_grid = discretize_assets(amin, amax, N_assets).reshape(-1)

    # pick N asset values uniformly from grid
    idx = rng.choice(len(a_grid), size=N, replace=True)
    a = a_grid[idx].reshape(-1, 1).astype(np.float32)

    return t, a

def sampler_terminal_exp(N_a, amin=a_min, amax=a_max):
    """
    Terminal sampler with mesh-grid:
    - time is fixed at T
    - assets: exponential grid of N_a points
    """
    # asset grid
    a_grid = discretize_assets(amin, amax, N_a)

    # repeat T for all asset points
    t = np.ones_like(a_grid, dtype=np.float32) * T

    return t, a_grid

#%% Training

# ---------
# Training
# ---------

#-- Initialisation
model = ResNetPINN(in_dim=2, hidden=hidden_dim, dropout_p=dropout_p).to(device)
#Load the same initial weights
model.load_state_dict(torch.load("PINN_Merton_InitialWeights.pth", map_location=device))


#-- Optimizer with Learning rate warm-up
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# Warmup + cosine annealing scheduler:
total_steps = epochs
warmup_steps = max(1, int(0.05 * total_steps))  # 5% warmup default

def lr_lambda(step):
    # linear warmup
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    # cosine decay after warmup
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + cos(pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#Loss History
loss_history = []
lr_history = []


#-- Terminal Value Function Gridpoints
#Get gridpoints
t_term_np, a_term_np = sampler_terminal_exp(N_terminal)
#Make into tensors
t_term = torch.tensor(t_term_np, dtype=torch.float32, device=device, requires_grad=True)
a_term = torch.tensor(a_term_np, dtype=torch.float32, device=device, requires_grad=True)
#Compute true terminal Value Function
V_term_true = torch.tensor(u_np(a_term_np), dtype=torch.float32, device=device).reshape(-1,1)

#-- Interior Gridpoints
#Random (t,a) pairs
t_col_np, a_col_np = sampler_collocation_exp_random_pairs(1_000)
#Make into tensors
t_col = torch.tensor(t_col_np, dtype=torch.float32, device=device, requires_grad=True)
a_col = torch.tensor(a_col_np, dtype=torch.float32, device=device, requires_grad=True)

if training:
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
    
        # compute derivatives
        V_col, V_t_col, V_a_col, V_aa_col = derivatives(model, t_col, a_col)
    
        # enforce positive V_a
        V_a_pos = torch.relu(V_a_col) + EPS
    
        # enforce negative V_aa
        V_aa_neg = -(torch.abs(V_aa_col) + EPS)
    
        # safe asset values
        a_safe = torch.clamp(a_col, min=a_min, max=a_max)
    
        # optimal controls
        c_star = V_a_pos.pow(-1.0 / gamma)
        pi_raw = - (mu - r) / (sigma**2 * a_safe) * (V_a_col / V_aa_neg)
        pi_star = MAX_PI * torch.tanh(pi_raw / MAX_PI)
    
        # PDE residual
        drift = r * a_col + pi_star * (mu - r) * a_col - c_star
        diffusion = 0.5 * (pi_star * sigma * a_col)**2
        rhs = u(c_star) + V_t_col + drift * V_a_col + diffusion * V_aa_col
        resid = rho * V_col - rhs
        loss_resid = torch.mean(resid**2)
    
        # terminal loss
        V_term_pred = model(t_term, a_term)
        loss_term = torch.mean((V_term_pred - V_term_true)**2)
    
        # penalty for negative V_a
        neg_grad_penalty = torch.mean(torch.relu(-V_a_col))
    
        # total loss
        loss = w_pde * loss_resid + w_term * loss_term + w_grad * neg_grad_penalty
    
        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
    
        #Save History
        loss_history.append(loss.item())
        lr_history.append(optimizer.param_groups[0]['lr'])
    
        #Print
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | LR {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Loss {loss.item():.3e} | "
                  f"PDE {loss_resid.item():.3e} | "
                  f"Terminal {loss_term.item():.3e} | "
                  f"Shape {neg_grad_penalty.item():.3e}")
    
    #Save model
    torch.save(model.state_dict(), "PINN_Merton_FinalWeights.pth")
    
    #Save Training History
    if USE_TRIAL_SOLUTION:
        with open("PINN_Merton_Training_TS_history.pkl", "wb") as f:
            pickle.dump((loss_history, lr_history), f)
    else:
        with open("PINN_Merton_Training_history.pkl", "wb") as f:
            pickle.dump((loss_history, lr_history), f)


#%% Load trained model
#Trained Model
# -------------------------
# Switch for trial solution
# -------------------------
if USE_TRIAL_SOLUTION:
    model.load_state_dict(torch.load("PINN_Merton_TS_FinalWeights.pth", map_location=device))
else:
    model.load_state_dict(torch.load("PINN_Merton_FinalWeights.pth", map_location=device))

#Load Training History
with open(directory + "PINN_Merton_Training_history.pkl", "rb") as f:
    loss_history, lr_history = pickle.load(f)
    
#%% Plots

# -------------------------
# Evaluation at t=0, 0.5T, T
# -------------------------
for t in [0.0,0.5*T,T]:
    model.eval()
    with torch.no_grad():
        #Value Function from PINN
        #Plot Gridpionts
        a_plot = np.linspace(a_min, a_max, 400, dtype=np.float32).reshape(-1,1)
        t0 = np.ones_like(a_plot, dtype=np.float32)*t
        #Tensors
        t0_t = torch.tensor(t0, dtype=torch.float32, device=device, requires_grad=True)
        a0_t = torch.tensor(a_plot, dtype=torch.float32, device=device, requires_grad=True)
        #Value Function
        V0_pred = model(t0_t, a0_t).cpu().numpy().reshape(-1)
    
        # analytic Value Function
        eta = (mu - r) / sigma
        r_tilde = (rho - (1-gamma)*r - 0.5 * (1-gamma)/gamma * eta**2) / gamma
        f_t = 1.0 / r_tilde * (1.0 - np.exp(-r_tilde*(T - t))) + np.exp(-r_tilde*(T - t))
        V_analytic = u_np(a_plot.reshape(-1)) * (f_t**gamma)
    
    # get PINN policies via autograd
    _, Vt_t, Va_t, Vaa_t = derivatives(model, t0_t, a0_t)
    with torch.no_grad():
        #Numpy
        Va_np = Va_t.cpu().numpy().reshape(-1) + EPS
        Vaa_np = Vaa_t.cpu().numpy().reshape(-1) - EPS
        #Compute Consumption
        c_pred = (Va_np)**(-1.0/gamma)  
        #Compute pi
        pi_raw_np = - (mu - r) / (sigma**2 * (a_plot.reshape(-1))) * (Va_np / Vaa_np)
        pi_pred_clipped = MAX_PI*np.tanh(pi_raw_np/MAX_PI)

    # plots
    plt.figure(figsize=(8,5))
    plt.plot(a_plot.reshape(-1), V0_pred, label="PINN", lw=2)
    plt.plot(a_plot.reshape(-1), V_analytic, '--', label='Analytical', lw=2)
    plt.xlabel(f"a (range: {min(a_plot)[0]:.2f} to {max(a_plot)[0]:.2f})"); plt.legend(); plt.title(f"Value function t={t}"); plt.grid(True)
    plt.savefig(f"Value function t={t}.pdf", 
           format='pdf', 
           dpi=300, 
           bbox_inches='tight',  # Removes extra white space around borders
           edgecolor='none',     # Removes border around the figure
           facecolor='white')    # Ensures white background
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(a_plot.reshape(-1)[5:], V0_pred[5:], label="PINN", lw=2)
    plt.plot(a_plot.reshape(-1)[5:], V_analytic[5:], '--', label='Analytical', lw=2)
    plt.xlabel(f"a (range: {a_plot[5][0]:.2f} to {max(a_plot)[0]:.2f})"); plt.legend(); plt.title(f"Value function t={t}"); plt.grid(True)
    plt.savefig(f"Value function t={t} Zoom.pdf", 
           format='pdf', 
           dpi=300, 
           bbox_inches='tight',  # Removes extra white space around borders
           edgecolor='none',     # Removes border around the figure
           facecolor='white')    # Ensures white background
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(a_plot.reshape(-1), c_pred, label="PINN", lw=2)
    plt.plot(a_plot.reshape(-1), (a_plot.reshape(-1)/f_t), '--', label='Analytic c', lw=2)
    plt.xlabel(f"a (range: {min(a_plot)[0]:.2f} to {max(a_plot)[0]:.2f})"); plt.legend(); plt.title(f"Consumption t={t}"); plt.grid(True)
    plt.savefig(f"Consumption t={t}.pdf", 
           format='pdf', 
           dpi=300, 
           bbox_inches='tight',  # Removes extra white space around borders
           edgecolor='none',     # Removes border around the figure
           facecolor='white')    # Ensures white background
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(a_plot.reshape(-1)[5:], c_pred[5:], label="PINN", lw=2)
    plt.plot(a_plot.reshape(-1)[5:], (a_plot.reshape(-1)/f_t)[5:], '--', label='Analytic c', lw=2)
    plt.xlabel(f"a (range: {a_plot[5][0]:.2f} to {max(a_plot)[0]:.2f})"); plt.legend(); plt.title(f"Consumption t={t}"); plt.grid(True)
    plt.savefig(f"Consumption t={t} Zoom.pdf", 
           format='pdf', 
           dpi=300, 
           bbox_inches='tight',  # Removes extra white space around borders
           edgecolor='none',     # Removes border around the figure
           facecolor='white')    # Ensures white background
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(a_plot.reshape(-1), pi_pred_clipped, label="PINN", lw=2)
    plt.plot(a_plot.reshape(-1), (1.0/gamma) * ((mu-r)/sigma) / sigma * np.ones_like(a_plot.reshape(-1)), '--', label='Analytic pi', lw=2)
    plt.xlabel(f"a (range: {min(a_plot)[0]:.2f} to {max(a_plot)[0]:.2f})"); plt.legend(); plt.title(f"Portfolio share t={t}"); plt.grid(True)
    plt.savefig(f"Portfolio share t={t}.pdf", 
           format='pdf', 
           dpi=300, 
           bbox_inches='tight',  # Removes extra white space around borders
           edgecolor='none',     # Removes border around the figure
           facecolor='white')    # Ensures white background
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(a_plot.reshape(-1)[5:], pi_pred_clipped[5:], label="PINN", lw=2)
    plt.plot(a_plot.reshape(-1)[5:], (1.0/gamma) * ((mu-r)/sigma) / sigma * np.ones_like(a_plot.reshape(-1))[5:], '--', label='Analytic pi', lw=2)
    plt.xlabel(f"a (range: {a_plot[5][0]:.2f} to {max(a_plot)[0]:.2f})"); plt.legend(); plt.title(f"Portfolio share t={t}"); plt.grid(True)
    plt.savefig(f"Portfolio share t={t} Zoom.pdf", 
           format='pdf', 
           dpi=300, 
           bbox_inches='tight',  # Removes extra white space around borders
           edgecolor='none',     # Removes border around the figure
           facecolor='white')    # Ensures white background
    plt.show()

plt.figure(figsize=(8,5))
plt.semilogy(loss_history)
plt.title('Training Loss'); plt.grid(True); plt.xlabel("Epoch")
plt.savefig("Training Loss.pdf", 
       format='pdf', 
       dpi=300, 
       bbox_inches='tight',  # Removes extra white space around borders
       edgecolor='none',     # Removes border around the figure
       facecolor='white')    # Ensures white background
plt.show()

plt.figure(figsize=(8,5))
plt.semilogy(lr_history)
plt.title('Learning Rate'); plt.grid(True); plt.xlabel("Epoch")
plt.savefig("Learning Rate.pdf", 
       format='pdf', 
       dpi=300, 
       bbox_inches='tight',  # Removes extra white space around borders
       edgecolor='none',     # Removes border around the figure
       facecolor='white')    # Ensures white background
plt.show()
