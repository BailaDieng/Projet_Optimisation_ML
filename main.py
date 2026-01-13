# -*- coding: utf-8 -*-
"""
Mini-Projet d'Optimisation ML
Phases 1 à 3 : Gradient déterministe, SGD, RMSProp/Adam, ISTA/FISTA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time

# -----------------------------
# 1. Génération du jeu de données
# -----------------------------
n_samples = 500
n_features = 20
X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_informative=10, n_redundant=5, n_classes=2,
                           random_state=42)
y = 2*y - 1  # Convertir en {-1,1}
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Phase 1 : Gradient Déterministe
# -----------------------------
lambda_ridge = 0.1

def logistic_loss(w, X, y, lam):
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z))) + lam/2 * np.sum(w**2)

def logistic_grad(w, X, y, lam):
    z = y * (X @ w)
    return -(X.T @ (y / (1 + np.exp(z)))) / X.shape[0] + lam*w

# Descente de Gradient à pas fixe
def gradient_descent(X, y, lam, lr=0.1, n_iter=100):
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(n_iter):
        g = logistic_grad(w, X, y, lam)
        w = w - lr * g
        losses.append(logistic_loss(w, X, y, lam))
    return w, losses

w_gd, losses_gd = gradient_descent(X, y, lambda_ridge, lr=0.5, n_iter=100)

plt.figure()
plt.plot(losses_gd, label='Gradient Déterministe')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Convergence Gradient Déterministe")
plt.legend()
plt.show()

# Gradient Conjugué (Fletcher-Reeves simplifié)
def gradient_conjugate(X, y, lam, n_iter=100):
    d = X.shape[1]
    w = np.zeros(d)
    g = logistic_grad(w, X, y, lam)
    d_cg = -g
    losses = []
    for _ in range(n_iter):
        losses.append(logistic_loss(w, X, y, lam))
        alpha = (g @ g) / (d_cg @ (lam*d_cg + (X.T @ X @ d_cg)/X.shape[0]))
        w = w + alpha * d_cg
        g_new = logistic_grad(w, X, y, lam)
        beta = (g_new @ g_new) / (g @ g)
        d_cg = -g_new + beta * d_cg
        g = g_new
    return w, losses

w_cg, losses_cg = gradient_conjugate(X, y, lambda_ridge, n_iter=100)

plt.figure()
plt.plot(losses_gd, label='Gradient Déterministe')
plt.plot(losses_cg, label='Gradient Conjugué')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Comparaison Gradient Déterministe vs Gradient Conjugué")
plt.legend()
plt.show()

# -----------------------------
# Phase 2 : Passage à l'Échelle Stochastique
# -----------------------------

# SGD avec décroissance du pas
def sgd(X, y, lam, lr=0.1, n_iter=200, batch_size=20):
    w = np.zeros(X.shape[1])
    losses = []
    n_samples = X.shape[0]
    for k in range(n_iter):
        alpha_k = lr / (1 + 0.01*k)  # règle de décroissance du pas
        idx = np.random.choice(n_samples, batch_size, replace=False)
        X_batch, y_batch = X[idx], y[idx]
        grad = logistic_grad(w, X_batch, y_batch, lam)
        w = w - alpha_k * grad
        losses.append(logistic_loss(w, X, y, lam))
    return w, losses

w_sgd, losses_sgd = sgd(X, y, lambda_ridge)
plt.figure()
plt.plot(losses_sgd, label='SGD')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Convergence SGD")
plt.legend()
plt.show()

# RMSProp
def rmsprop(X, y, lam, lr=0.01, beta=0.9, eps=1e-8, n_iter=200, batch_size=20):
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)
    losses = []
    n_samples = X.shape[0]
    for k in range(n_iter):
        idx = np.random.choice(n_samples, batch_size, replace=False)
        X_batch, y_batch = X[idx], y[idx]
        grad = logistic_grad(w, X_batch, y_batch, lam)
        v = beta*v + (1-beta)*(grad**2)
        w = w - lr * grad / np.sqrt(v + eps)
        losses.append(logistic_loss(w, X, y, lam))
    return w, losses

w_rms, losses_rms = rmsprop(X, y, lambda_ridge)
plt.figure()
plt.plot(losses_rms, label='RMSProp')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Convergence RMSProp")
plt.legend()
plt.show()

# Adam
def adam(X, y, lam, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iter=200, batch_size=20):
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    losses = []
    n_samples = X.shape[0]
    for t in range(1, n_iter+1):
        idx = np.random.choice(n_samples, batch_size, replace=False)
        X_batch, y_batch = X[idx], y[idx]
        grad = logistic_grad(w, X_batch, y_batch, lam)
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*(grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(logistic_loss(w, X, y, lam))
    return w, losses

w_adam, losses_adam = adam(X, y, lambda_ridge)
plt.figure()
plt.plot(losses_sgd, label='SGD')
plt.plot(losses_rms, label='RMSProp')
plt.plot(losses_adam, label='Adam')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Comparaison Optimiseurs Modernes")
plt.legend()
plt.show()

# -----------------------------
# Phase 3 : Proximal L1 (ISTA / FISTA)
# -----------------------------
def soft_threshold(w, lam):
    return np.sign(w) * np.maximum(np.abs(w)-lam, 0.0)

def ista(X, y, lam, lr=0.1, n_iter=200):
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(n_iter):
        grad = logistic_grad(w, X, y, 0)
        w = soft_threshold(w - lr*grad, lr*lam)
        losses.append(logistic_loss(w, X, y, 0) + lam*np.sum(np.abs(w)))
    return w, losses

w_ista, losses_ista = ista(X, y, lam=0.1)
plt.figure()
plt.plot(losses_ista, label='ISTA')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Convergence ISTA")
plt.legend()
plt.show()

def fista(X, y, lam, lr=0.1, n_iter=200):
    w = np.zeros(X.shape[1])
    z = w.copy()
    t = 1
    losses = []
    for _ in range(n_iter):
        grad = logistic_grad(z, X, y, 0)  # <-- corrigé ici
        w_new = soft_threshold(z - lr*grad, lr*lam)
        t_new = (1 + np.sqrt(1+4*t**2))/2
        z = w_new + ((t-1)/t_new)*(w_new-w)
        t = t_new
        w = w_new
        losses.append(logistic_loss(w, X, y, 0) + lam*np.sum(np.abs(w)))
    return w, losses

w_fista, losses_fista = fista(X, y, lam=0.1)
plt.figure()
plt.plot(losses_fista, label='FISTA')
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.title("Convergence FISTA")
plt.legend()
plt.show()
