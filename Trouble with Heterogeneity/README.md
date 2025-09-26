# PINN for the Merton Model

This repository implements a **Physics-Informed Neural Network (PINN)** to solve the **Merton portfolio optimization problem**, an influential model in financial economics. The code is written in Python using PyTorch and compares PINN-based solutions with known analytical benchmarks.

---

## ✨ Overview

The Merton model describes optimal consumption and portfolio choice under uncertainty. Instead of solving its Hamilton–Jacobi–Bellman (HJB) PDE analytically, this project uses a **PINN** to approximate the value function and derive optimal controls:

- **Value Function**: Investor’s lifetime utility of consumption/wealth  
- **Consumption Policy**: Optimal rate of consumption over time  
- **Portfolio Share**: Optimal risky asset allocation  

The PINN incorporates **economic priors, boundary conditions, and PDE residuals** into the loss function, ensuring solutions respect both the **HJB equation** and **economic constraints**.

---

## 📂 Repository Structure

- `main.py` (this script): Defines model, training loop, and evaluation  
- `PINN_Merton_InitialWeights.pth`: Pre-trained initialization weights  
- `PINN_Merton_FinalWeights.pth`: Saved trained model weights  
- `.pkl` files: Training history (loss, learning rate)  

Outputs include PDF plots for value function, consumption, portfolio share, and training dynamics.

---

## ⚙️ Key Components

### 1. **Model Setup**
- Imports **PyTorch**, **NumPy**, and **Matplotlib**.  
- Enables GPU training if available.  
- Defines **economic parameters**:  
  ```python
  rho = 0.05     # Discount rate
  gamma = 2.0    # Risk aversion
  r = 0.02       # Risk-free rate
  mu = 0.06      # Expected return
  sigma = 0.2    # Volatility
  T = 1.0        # Time horizon
