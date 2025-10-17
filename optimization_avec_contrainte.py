"""
Module d'optimisation avec contraintes
Advanced Optimization with Constraints

Ce module implémente plusieurs algorithmes d'optimisation pour les problèmes avec contraintes:
- Penalty Method (Méthode de pénalité)
- Augmented Lagrangian Method (Méthode du Lagrangien augmenté)
- Projected Gradient Descent (Descente de gradient projeté)
- Sequential Quadratic Programming (SQP simplifié)
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from optimization_sans_contrainte import bfgs, gradient_descent


class Constraint:
    """Classe représentant une contrainte"""
    def __init__(self, func: Callable[[np.ndarray], float], constraint_type: str = 'ineq'):
        """
        Args:
            func: Fonction de contrainte (doit être <= 0 pour 'ineq' ou = 0 pour 'eq')
            constraint_type: Type de contrainte ('ineq' pour inégalité, 'eq' pour égalité)
        """
        self.func = func
        self.type = constraint_type


def penalty_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    constraints: List[Constraint],
    mu_init: float = 1.0,
    mu_factor: float = 10.0,
    max_outer_iter: int = 20,
    max_inner_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Méthode de pénalité pour l'optimisation avec contraintes
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x0: Point initial
        constraints: Liste des contraintes
        mu_init: Paramètre de pénalité initial
        mu_factor: Facteur d'augmentation du paramètre de pénalité
        max_outer_iter: Nombre maximum d'itérations externes
        max_inner_iter: Nombre maximum d'itérations internes
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations externes effectuées
    """
    x = x0.copy()
    mu = mu_init
    
    for outer_iter in range(max_outer_iter):
        # Fonction de pénalité
        def penalized_f(x_val):
            penalty = 0.0
            for constraint in constraints:
                c_val = constraint.func(x_val)
                if constraint.type == 'eq':
                    penalty += c_val**2
                else:  # 'ineq'
                    penalty += max(0, c_val)**2
            return f(x_val) + mu * penalty
        
        # Gradient de la fonction pénalisée (approximation par différences finies)
        def grad_penalized_f(x_val):
            grad = grad_f(x_val)
            eps = 1e-8
            for i in range(len(x_val)):
                x_plus = x_val.copy()
                x_plus[i] += eps
                x_minus = x_val.copy()
                x_minus[i] -= eps
                
                penalty_grad = (penalized_f(x_plus) - penalized_f(x_minus)) / (2 * eps)
                
                # Ajouter uniquement la composante de pénalité
                f_grad = (f(x_plus) - f(x_minus)) / (2 * eps)
                grad[i] = f_grad + (penalty_grad - f_grad)
            
            return grad
        
        # Optimisation sans contrainte du problème pénalisé
        x, f_val, _ = bfgs(penalized_f, grad_penalized_f, x, max_iter=max_inner_iter, tol=tol)
        
        # Vérifier la satisfaction des contraintes
        max_violation = 0.0
        for constraint in constraints:
            c_val = constraint.func(x)
            if constraint.type == 'eq':
                max_violation = max(max_violation, abs(c_val))
            else:
                max_violation = max(max_violation, max(0, c_val))
        
        if max_violation < tol:
            return x, f(x), outer_iter + 1
        
        # Augmenter le paramètre de pénalité
        mu *= mu_factor
    
    return x, f(x), max_outer_iter


def augmented_lagrangian(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    constraints: List[Constraint],
    rho_init: float = 1.0,
    rho_factor: float = 2.0,
    max_outer_iter: int = 20,
    max_inner_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Méthode du Lagrangien augmenté
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x0: Point initial
        constraints: Liste des contraintes
        rho_init: Paramètre de pénalité initial
        rho_factor: Facteur d'augmentation du paramètre de pénalité
        max_outer_iter: Nombre maximum d'itérations externes
        max_inner_iter: Nombre maximum d'itérations internes
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations externes effectuées
    """
    x = x0.copy()
    rho = rho_init
    
    # Initialiser les multiplicateurs de Lagrange
    lambdas = [0.0 for _ in constraints]
    
    for outer_iter in range(max_outer_iter):
        # Fonction Lagrangien augmenté
        def augmented_lagrangian_f(x_val):
            result = f(x_val)
            for i, constraint in enumerate(constraints):
                c_val = constraint.func(x_val)
                if constraint.type == 'eq':
                    result += lambdas[i] * c_val + (rho / 2) * c_val**2
                else:  # 'ineq'
                    result += lambdas[i] * c_val + (rho / 2) * max(0, c_val)**2
            return result
        
        # Gradient du Lagrangien augmenté (approximation)
        def grad_augmented_lagrangian_f(x_val):
            eps = 1e-8
            grad = np.zeros_like(x_val)
            for i in range(len(x_val)):
                x_plus = x_val.copy()
                x_plus[i] += eps
                x_minus = x_val.copy()
                x_minus[i] -= eps
                grad[i] = (augmented_lagrangian_f(x_plus) - augmented_lagrangian_f(x_minus)) / (2 * eps)
            return grad
        
        # Optimisation sans contrainte
        x, f_val, _ = bfgs(augmented_lagrangian_f, grad_augmented_lagrangian_f, x, 
                           max_iter=max_inner_iter, tol=tol)
        
        # Mise à jour des multiplicateurs de Lagrange
        max_violation = 0.0
        for i, constraint in enumerate(constraints):
            c_val = constraint.func(x)
            if constraint.type == 'eq':
                lambdas[i] += rho * c_val
                max_violation = max(max_violation, abs(c_val))
            else:  # 'ineq'
                lambdas[i] = max(0, lambdas[i] + rho * c_val)
                max_violation = max(max_violation, max(0, c_val))
        
        if max_violation < tol:
            return x, f(x), outer_iter + 1
        
        # Augmenter le paramètre de pénalité
        rho *= rho_factor
    
    return x, f(x), max_outer_iter


def projected_gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float, int]:
    """
    Descente de gradient avec projection sur l'ensemble réalisable
    
    Args:
        f: Fonction objectif
        grad_f: Gradient de la fonction objectif
        x0: Point initial (doit être réalisable)
        project: Fonction de projection sur l'ensemble réalisable
        learning_rate: Taux d'apprentissage
        max_iter: Nombre maximum d'itérations
        tol: Tolérance pour la convergence
        
    Returns:
        x_opt: Point optimal
        f_opt: Valeur optimale
        n_iter: Nombre d'itérations effectuées
    """
    x = project(x0.copy())
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Vérifier la convergence
        if np.linalg.norm(grad) < tol:
            return x, f(x), i + 1
        
        # Mise à jour avec projection
        x_new = x - learning_rate * grad
        x = project(x_new)
    
    return x, f(x), max_iter


def project_box(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Projection sur une boîte [lower, upper]
    
    Args:
        x: Point à projeter
        bounds: Liste de tuples (lower, upper) pour chaque dimension
        
    Returns:
        Point projeté
    """
    x_proj = x.copy()
    for i, (lower, upper) in enumerate(bounds):
        x_proj[i] = np.clip(x_proj[i], lower, upper)
    return x_proj


# Fonctions d'exemple pour tester avec contraintes
def sphere(x: np.ndarray) -> float:
    """Fonction sphère"""
    return np.sum(x**2)


def grad_sphere(x: np.ndarray) -> np.ndarray:
    """Gradient de la fonction sphère"""
    return 2 * x


if __name__ == "__main__":
    print("=== Optimisation avec contraintes ===\n")
    
    # Exemple 1: Minimiser x^2 + y^2 avec contrainte x + y >= 1
    print("Exemple 1: min x^2 + y^2 sujet à x + y >= 1")
    print("(Solution analytique: x = y = 0.5, f = 0.5)\n")
    
    x0 = np.array([1.0, 1.0])
    
    # Contrainte: x + y - 1 >= 0 => -(x + y - 1) <= 0
    constraint1 = Constraint(lambda x: -(x[0] + x[1] - 1), 'ineq')
    
    # Méthode de pénalité
    print("1. Méthode de pénalité:")
    x_opt, f_opt, n_iter = penalty_method(sphere, grad_sphere, x0, [constraint1], 
                                          mu_init=1.0, max_outer_iter=20)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6f}")
    print(f"   Contrainte (x + y): {x_opt[0] + x_opt[1]:.6f}")
    print(f"   Itérations externes: {n_iter}\n")
    
    # Méthode du Lagrangien augmenté
    print("2. Méthode du Lagrangien augmenté:")
    x_opt, f_opt, n_iter = augmented_lagrangian(sphere, grad_sphere, x0, [constraint1],
                                                 rho_init=1.0, max_outer_iter=20)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6f}")
    print(f"   Contrainte (x + y): {x_opt[0] + x_opt[1]:.6f}")
    print(f"   Itérations externes: {n_iter}\n")
    
    # Exemple 2: Projection sur une boîte
    print("Exemple 2: min x^2 + y^2 sujet à 0 <= x, y <= 1")
    print("(Solution: x = y = 0, f = 0)\n")
    
    x0 = np.array([0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    def project_func(x):
        return project_box(x, bounds)
    
    print("3. Descente de gradient projeté:")
    x_opt, f_opt, n_iter = projected_gradient_descent(sphere, grad_sphere, x0, project_func,
                                                       learning_rate=0.1, max_iter=1000)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6e}")
    print(f"   Itérations: {n_iter}\n")
    
    # Exemple 3: Contrainte d'égalité
    print("Exemple 3: min (x-2)^2 + (y-2)^2 sujet à x + y = 2")
    print("(Solution analytique: x = y = 1, f = 2)\n")
    
    def f_shifted(x):
        return (x[0] - 2)**2 + (x[1] - 2)**2
    
    def grad_f_shifted(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 2)])
    
    constraint_eq = Constraint(lambda x: x[0] + x[1] - 2, 'eq')
    
    x0 = np.array([0.5, 0.5])
    
    print("4. Méthode de pénalité (contrainte d'égalité):")
    x_opt, f_opt, n_iter = penalty_method(f_shifted, grad_f_shifted, x0, [constraint_eq],
                                          mu_init=1.0, max_outer_iter=20)
    print(f"   Solution: x = {x_opt}")
    print(f"   Valeur optimale: f(x) = {f_opt:.6f}")
    print(f"   Contrainte (x + y): {x_opt[0] + x_opt[1]:.6f}")
    print(f"   Itérations externes: {n_iter}\n")
