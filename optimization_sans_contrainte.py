"""
Module d'optimisation sans contrainte
Advanced Optimization without Constraints

Ce module implémente plusieurs algorithmes d'optimisation pour les problèmes sans contraintes:
- Gradient Descent (Descente de gradient)
- Newton's Method (Méthode de Newton)
- Conjugate Gradient (Gradient conjugué)
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
"""

import numpy as np
from typing import Callable, Tuple, Optional


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Optimisation par descente de gradient
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x0: Point initial
        learning_rate: Taux d'apprentissage
        max_iter: Nombre maximum d'itérations
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations effectuées
    """
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Vérifier la convergence
        if np.linalg.norm(grad) < tol:
            return x, f(x), i + 1
        
        # Mise à jour
        x = x - learning_rate * grad
    
    return x, f(x), max_iter


def newton_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Méthode de Newton pour l'optimisation
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        hess_f: Matrice hessienne de la fonction objectif
        x0: Point initial
        max_iter: Nombre maximum d'itérations
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations effectuées
    """
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Vérifier la convergence
        if np.linalg.norm(grad) < tol:
            return x, f(x), i + 1
        
        # Calculer la direction de Newton
        hess = hess_f(x)
        try:
            direction = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # Si la hessienne n'est pas inversible, utiliser le gradient
            direction = -grad
        
        # Mise à jour
        x = x + direction
    
    return x, f(x), max_iter


def conjugate_gradient(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Méthode du gradient conjugué (Fletcher-Reeves)
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x0: Point initial
        max_iter: Nombre maximum d'itérations
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations effectuées
    """
    x = x0.copy()
    grad = grad_f(x)
    direction = -grad
    
    for i in range(max_iter):
        # Vérifier la convergence
        if np.linalg.norm(grad) < tol:
            return x, f(x), i + 1
        
        # Recherche linéaire simple (Armijo)
        alpha = line_search_armijo(f, grad_f, x, direction)
        
        # Mise à jour
        x_new = x + alpha * direction
        grad_new = grad_f(x_new)
        
        # Calcul du coefficient beta (Fletcher-Reeves)
        beta = np.dot(grad_new, grad_new) / (np.dot(grad, grad) + 1e-10)
        
        # Nouvelle direction
        direction = -grad_new + beta * direction
        
        x = x_new
        grad = grad_new
    
    return x, f(x), max_iter


def bfgs(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Méthode BFGS (Broyden-Fletcher-Goldfarb-Shanno)
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x0: Point initial
        max_iter: Nombre maximum d'itérations
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations effectuées
    """
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Approximation initiale de l'inverse de la hessienne
    grad = grad_f(x)
    
    for i in range(max_iter):
        # Vérifier la convergence
        if np.linalg.norm(grad) < tol:
            return x, f(x), i + 1
        
        # Direction de recherche
        direction = -H @ grad
        
        # Recherche linéaire
        alpha = line_search_armijo(f, grad_f, x, direction)
        
        # Mise à jour
        s = alpha * direction
        x_new = x + s
        grad_new = grad_f(x_new)
        y = grad_new - grad
        
        # Mise à jour de H (formule BFGS)
        rho = 1.0 / (np.dot(y, s) + 1e-10)
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        x = x_new
        grad = grad_new
    
    return x, f(x), max_iter


def line_search_armijo(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    direction: np.ndarray,
    c: float = 1e-4,
    rho: float = 0.9,
    max_iter: int = 50
) -> float:
    """
    Recherche linéaire avec condition d'Armijo
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x: Point courant
        direction: Direction de recherche
        c: Paramètre de la condition d'Armijo
        rho: Facteur de réduction
        max_iter: Nombre maximum d'itérations
        
    Returns:
        alpha: Pas optimal
    """
    alpha = 1.0
    f_x = f(x)
    grad_x = grad_f(x)
    slope = np.dot(grad_x, direction)
    
    for _ in range(max_iter):
        if f(x + alpha * direction) <= f_x + c * alpha * slope:
            return alpha
        alpha *= rho
    
    return alpha


# Fonctions d'exemple pour tester
def rosenbrock(x: np.ndarray) -> float:
    """Fonction de Rosenbrock (minimum à (1, 1))"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Gradient de la fonction de Rosenbrock"""
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])


def hess_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Hessienne de la fonction de Rosenbrock"""
    return np.array([
        [2 - 400 * (x[1] - x[0]**2) + 800 * x[0]**2, -400 * x[0]],
        [-400 * x[0], 200]
    ])


if __name__ == "__main__":
    # Test avec la fonction de Rosenbrock
    x0 = np.array([0.0, 0.0])
    
    print("=== Optimisation sans contrainte ===\n")
    
    # Gradient Descent
    print("1. Descente de gradient:")
    x_opt, f_opt, n_iter = gradient_descent(rosenbrock, grad_rosenbrock, x0, learning_rate=0.001, max_iter=10000)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6e}")
    print(f"   Itérations: {n_iter}\n")
    
    # Newton's Method
    print("2. Méthode de Newton:")
    x_opt, f_opt, n_iter = newton_method(rosenbrock, grad_rosenbrock, hess_rosenbrock, x0, max_iter=100)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6e}")
    print(f"   Itérations: {n_iter}\n")
    
    # Conjugate Gradient
    print("3. Gradient conjugué:")
    x_opt, f_opt, n_iter = conjugate_gradient(rosenbrock, grad_rosenbrock, x0, max_iter=1000)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6e}")
    print(f"   Itérations: {n_iter}\n")
    
    # BFGS
    print("4. BFGS:")
    x_opt, f_opt, n_iter = bfgs(rosenbrock, grad_rosenbrock, x0, max_iter=1000)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6e}")
    print(f"   Itérations: {n_iter}\n")
