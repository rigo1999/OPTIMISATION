# OPTIMISATION

Bibliothèque d'optimisation avancée en Python avec et sans contraintes.

## Description

Ce projet implémente plusieurs algorithmes d'optimisation avancés pour résoudre des problèmes d'optimisation mathématique avec et sans contraintes.

### Algorithmes d'optimisation sans contrainte

Le module `optimization_sans_contrainte.py` implémente les algorithmes suivants :

1. **Descente de gradient (Gradient Descent)** : Algorithme itératif de base utilisant le gradient de la fonction objectif
2. **Méthode de Newton** : Utilise la matrice hessienne pour une convergence quadratique
3. **Gradient conjugué (Conjugate Gradient)** : Méthode de Fletcher-Reeves pour l'optimisation
4. **BFGS** : Méthode quasi-Newton (Broyden-Fletcher-Goldfarb-Shanno) qui approxime la hessienne

### Algorithmes d'optimisation avec contraintes

Le module `optimization_avec_contrainte.py` implémente les algorithmes suivants :

1. **Méthode de pénalité (Penalty Method)** : Transforme un problème contraint en une séquence de problèmes sans contrainte
2. **Lagrangien augmenté (Augmented Lagrangian)** : Combine multiplicateurs de Lagrange et termes de pénalité
3. **Descente de gradient projeté (Projected Gradient Descent)** : Projection sur l'ensemble réalisable après chaque itération
4. **Projection sur boîte (Box Projection)** : Fonction utilitaire pour projeter sur des contraintes de bornes

## Installation

Aucune installation spéciale requise. Dépendances :
- Python 3.6+
- NumPy

```bash
pip install numpy
```

## Utilisation

### Exemple 1 : Optimisation sans contrainte

```python
import numpy as np
from optimization_sans_contrainte import bfgs, rosenbrock, grad_rosenbrock

# Point initial
x0 = np.array([0.0, 0.0])

# Optimisation de la fonction de Rosenbrock
x_opt, f_opt, n_iter = bfgs(rosenbrock, grad_rosenbrock, x0)

print(f"Solution optimale: {x_opt}")
print(f"Valeur optimale: {f_opt}")
print(f"Nombre d'itérations: {n_iter}")
```

### Exemple 2 : Optimisation avec contrainte d'inégalité

```python
import numpy as np
from optimization_avec_contrainte import penalty_method, Constraint, sphere, grad_sphere

# Minimiser x^2 + y^2 sujet à x + y >= 1
x0 = np.array([1.0, 1.0])

# Définir la contrainte: x + y >= 1 équivaut à -(x + y - 1) <= 0
constraint = Constraint(lambda x: -(x[0] + x[1] - 1), 'ineq')

# Optimisation
x_opt, f_opt, n_iter = penalty_method(sphere, grad_sphere, x0, [constraint])

print(f"Solution optimale: {x_opt}")
print(f"Valeur optimale: {f_opt}")
```

### Exemple 3 : Optimisation avec contrainte d'égalité

```python
import numpy as np
from optimization_avec_contrainte import penalty_method, Constraint

# Minimiser (x-2)^2 + (y-2)^2 sujet à x + y = 2
def f(x):
    return (x[0] - 2)**2 + (x[1] - 2)**2

def grad_f(x):
    return np.array([2*(x[0] - 2), 2*(x[1] - 2)])

x0 = np.array([0.5, 0.5])

# Contrainte d'égalité
constraint = Constraint(lambda x: x[0] + x[1] - 2, 'eq')

x_opt, f_opt, n_iter = penalty_method(f, grad_f, x0, [constraint])

print(f"Solution optimale: {x_opt}")
```

### Exemple 4 : Contraintes de bornes

```python
import numpy as np
from optimization_avec_contrainte import projected_gradient_descent, project_box

# Minimiser (x-2)^2 + (y-2)^2 sujet à 0 <= x, y <= 1
def f(x):
    return (x[0] - 2)**2 + (x[1] - 2)**2

def grad_f(x):
    return np.array([2*(x[0] - 2), 2*(x[1] - 2)])

x0 = np.array([0.5, 0.5])
bounds = [(0.0, 1.0), (0.0, 1.0)]

def project(x):
    return project_box(x, bounds)

x_opt, f_opt, n_iter = projected_gradient_descent(f, grad_f, x0, project, learning_rate=0.1)

print(f"Solution optimale: {x_opt}")
```

## Fichiers

- `optimization_sans_contrainte.py` : Algorithmes d'optimisation sans contrainte
- `optimization_avec_contrainte.py` : Algorithmes d'optimisation avec contraintes
- `examples.py` : Exemples d'utilisation complets
- `test_optimization.py` : Tests unitaires

## Exécuter les exemples

```bash
# Exemples d'optimisation sans contrainte
python optimization_sans_contrainte.py

# Exemples d'optimisation avec contraintes
python optimization_avec_contrainte.py

# Tous les exemples
python examples.py

# Tests
python test_optimization.py
```

## Tests

Les tests vérifient le bon fonctionnement des algorithmes :

```bash
python test_optimization.py
```

Tous les tests doivent passer avec succès.

## Algorithmes détaillés

### Optimisation sans contrainte

#### 1. Descente de gradient
- **Principe** : Mise à jour itérative dans la direction opposée au gradient
- **Formule** : x_{k+1} = x_k - α ∇f(x_k)
- **Avantages** : Simple, peu coûteux en mémoire
- **Inconvénients** : Convergence lente, sensible au taux d'apprentissage

#### 2. Méthode de Newton
- **Principe** : Utilise la courbure (hessienne) pour une meilleure direction
- **Formule** : x_{k+1} = x_k - H^{-1}(x_k) ∇f(x_k)
- **Avantages** : Convergence quadratique près de l'optimum
- **Inconvénients** : Coût de calcul de la hessienne, peut diverger

#### 3. Gradient conjugué
- **Principe** : Directions conjuguées pour éviter les zigzags
- **Avantages** : Efficace pour problèmes de grande dimension
- **Inconvénients** : Nécessite une recherche linéaire

#### 4. BFGS
- **Principe** : Approximation de la hessienne inverse mise à jour à chaque itération
- **Avantages** : Convergence superlinéaire, pas de calcul de hessienne
- **Inconvénients** : Mémoire O(n²)

### Optimisation avec contraintes

#### 1. Méthode de pénalité
- **Principe** : Ajoute un terme de pénalité pour les violations de contraintes
- **Formule** : min f(x) + μ Σ max(0, g_i(x))²
- **Avantages** : Simple à implémenter
- **Inconvénients** : Problèmes numériques pour μ grand

#### 2. Lagrangien augmenté
- **Principe** : Combine multiplicateurs de Lagrange et pénalité
- **Avantages** : Meilleure conditionnement numérique
- **Inconvénients** : Plus complexe

#### 3. Gradient projeté
- **Principe** : Projection sur l'ensemble réalisable après chaque pas
- **Avantages** : Garantit la faisabilité
- **Inconvénients** : Nécessite une projection efficace

## Fonctions de test

Le projet inclut plusieurs fonctions de test classiques :

1. **Fonction de Rosenbrock** : Fonction non convexe avec vallée étroite
   - f(x,y) = (1-x)² + 100(y-x²)²
   - Minimum global : (1, 1)

2. **Fonction sphère** : Fonction convexe simple
   - f(x) = Σ x_i²
   - Minimum global : origine

## Références

- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization. Springer.
- Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.

## Auteur

Projet OPTIMISATION - Algorithmes d'optimisation avancée

## Licence

Ce projet est fourni à des fins éducatives.