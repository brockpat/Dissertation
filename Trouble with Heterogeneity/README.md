# Physics-Informed Neural Network for Merton's Portfolio Choice Problem

A PyTorch implementation of a Physics-Informed Neural Network (PINN) to solve Merton's continuous-time portfolio optimization problem. This repository demonstrates how neural networks can learn the value function and optimal policies for stochastic control problems by directly incorporating the Hamilton-Jacobi-Bellman (HJB) equation into the training process.

## 📋 Overview

Merton's portfolio problem is a classic problem in financial economics that involves optimal allocation between a risky asset and a risk-free asset over time. This project solves the problem using a PINN approach that:

- **Learns the value function** directly from the HJB equation
- **Enforces economic constraints** through penalty terms in the loss function
- **Provides optimal policies** for consumption and portfolio allocation
- **Compares PINN solutions** against known analytical solutions

## 🎯 Key Features

- **ResNet architecture** with skip connections for stable training
- **Trial solution formulation** for improved boundary behavior
- **Adaptive learning rate** with warm-up and cosine annealing
- **Gradient clipping** and L2 regularization for training stability
- **Automatic differentiation** for precise derivative calculations
- **Comprehensive visualization** of results and training progress

## 🏗️ Architecture

The neural network implements a 3-layer ResNet architecture:
- Input dimension: 2 (time `t` and wealth `a`)
- Hidden layers: 64 neurons each with Tanh activation
- Dropout for regularization (configurable)
- Skip connection from input to second hidden layer
- Optional trial solution modification for shape constraints

## 📊 Mathematical Formulation

### Merton Model Parameters
- `ρ`: Subjective discount rate
- `γ`: Risk aversion coefficient
- `r`: Risk-free rate
- `μ`: Risky asset return
- `σ`: Risky asset volatility
- `T`: Investment horizon

### HJB Equation
The value function V(t,a) satisfies:
ρV = max_{c,π} [u(c) + V_t + (r + π(μ-r))a V_a - c V_a + 1/2 π²σ²a² V_aa]
