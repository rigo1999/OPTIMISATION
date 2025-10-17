"""
Tests pour les algorithmes d'optimisation
Tests for optimization algorithms
"""

import numpy as np
import sys
from optimization_sans_contrainte import (
    gradient_descent, newton_method, conjugate_gradient, bfgs,
    rosenbrock, grad_rosenbrock, hess_rosenbrock
)
from optimization_avec_contrainte import (
    penalty_method, augmented_lagrangian, projected_gradient_descent,
    Constraint, project_box, sphere, grad_sphere
)


def test_unconstrained_optimization():
    """Test des algorithmes d'optimisation sans contrainte"""
    print("Test 1: Optimisation sans contrainte")
    print("-" * 50)
    
    # Fonction quadratique simple: f(x) = x^2 + y^2
    # Minimum à (0, 0) avec f* = 0
    x0 = np.array([2.0, 2.0])
    
    # Test BFGS
    x_opt, f_opt, n_iter = bfgs(sphere, grad_sphere, x0, tol=1e-6)
    
    # Vérifier que la solution est proche de (0, 0)
    assert np.linalg.norm(x_opt) < 0.01, f"BFGS: Solution incorrecte: {x_opt}"
    assert abs(f_opt) < 0.01, f"BFGS: Valeur optimale incorrecte: {f_opt}"
    print(f"✓ BFGS: Solution = {x_opt}, f* = {f_opt:.6e}, iterations = {n_iter}")
    
    # Test Gradient Descent
    x_opt, f_opt, n_iter = gradient_descent(sphere, grad_sphere, x0, learning_rate=0.1, tol=1e-6)
    assert np.linalg.norm(x_opt) < 0.1, f"Gradient Descent: Solution incorrecte: {x_opt}"
    assert abs(f_opt) < 0.1, f"Gradient Descent: Valeur optimale incorrecte: {f_opt}"
    print(f"✓ Gradient Descent: Solution = {x_opt}, f* = {f_opt:.6e}, iterations = {n_iter}")
    
    # Test Conjugate Gradient
    x_opt, f_opt, n_iter = conjugate_gradient(sphere, grad_sphere, x0, tol=1e-6)
    assert np.linalg.norm(x_opt) < 0.1, f"Conjugate Gradient: Solution incorrecte: {x_opt}"
    assert abs(f_opt) < 0.1, f"Conjugate Gradient: Valeur optimale incorrecte: {f_opt}"
    print(f"✓ Conjugate Gradient: Solution = {x_opt}, f* = {f_opt:.6e}, iterations = {n_iter}")
    
    # Test Rosenbrock avec BFGS
    x0_ros = np.array([0.0, 0.0])
    x_opt, f_opt, n_iter = bfgs(rosenbrock, grad_rosenbrock, x0_ros, tol=1e-6)
    
    # Vérifier que la solution est proche de (1, 1)
    expected = np.array([1.0, 1.0])
    assert np.linalg.norm(x_opt - expected) < 0.1, f"BFGS Rosenbrock: Solution incorrecte: {x_opt}"
    print(f"✓ BFGS (Rosenbrock): Solution = {x_opt}, f* = {f_opt:.6e}, iterations = {n_iter}")
    
    print("✓ Tous les tests d'optimisation sans contrainte réussis!\n")


def test_constrained_optimization():
    """Test des algorithmes d'optimisation avec contrainte"""
    print("Test 2: Optimisation avec contraintes")
    print("-" * 50)
    
    # Test 1: min x^2 + y^2 sujet à x + y >= 1
    # Solution: x = y = 0.5, f* = 0.5
    x0 = np.array([1.0, 1.0])
    constraint = Constraint(lambda x: -(x[0] + x[1] - 1), 'ineq')
    
    # Test Penalty Method
    x_opt, f_opt, n_iter = penalty_method(
        sphere, grad_sphere, x0, [constraint], 
        mu_init=1.0, max_outer_iter=20, tol=1e-4
    )
    
    # Vérifier la solution
    assert abs(x_opt[0] + x_opt[1] - 1.0) < 0.1, f"Penalty: Contrainte non satisfaite: {x_opt[0] + x_opt[1]}"
    assert abs(f_opt - 0.5) < 0.2, f"Penalty: Valeur optimale incorrecte: {f_opt}"
    print(f"✓ Penalty Method: Solution = {x_opt}, f* = {f_opt:.6f}, iterations = {n_iter}")
    print(f"  Contrainte x+y = {x_opt[0] + x_opt[1]:.6f} (attendu: 1.0)")
    
    # Test Augmented Lagrangian
    x_opt, f_opt, n_iter = augmented_lagrangian(
        sphere, grad_sphere, x0, [constraint],
        rho_init=1.0, max_outer_iter=20, tol=1e-4
    )
    
    assert abs(x_opt[0] + x_opt[1] - 1.0) < 0.1, f"Augmented Lagrangian: Contrainte non satisfaite: {x_opt[0] + x_opt[1]}"
    assert abs(f_opt - 0.5) < 0.2, f"Augmented Lagrangian: Valeur optimale incorrecte: {f_opt}"
    print(f"✓ Augmented Lagrangian: Solution = {x_opt}, f* = {f_opt:.6f}, iterations = {n_iter}")
    print(f"  Contrainte x+y = {x_opt[0] + x_opt[1]:.6f} (attendu: 1.0)")
    
    # Test 2: Contraintes de bornes avec gradient projeté
    # min (x-2)^2 + (y-2)^2 sujet à 0 <= x, y <= 1
    # Solution: x = y = 1, f* = 2
    def f_shifted(x):
        return (x[0] - 2)**2 + (x[1] - 2)**2
    
    def grad_f_shifted(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 2)])
    
    x0 = np.array([0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    def project_func(x):
        return project_box(x, bounds)
    
    x_opt, f_opt, n_iter = projected_gradient_descent(
        f_shifted, grad_f_shifted, x0, project_func, 
        learning_rate=0.1, tol=1e-6
    )
    
    # Vérifier la solution
    expected = np.array([1.0, 1.0])
    assert np.linalg.norm(x_opt - expected) < 0.1, f"Projected Gradient: Solution incorrecte: {x_opt}"
    assert abs(f_opt - 2.0) < 0.2, f"Projected Gradient: Valeur optimale incorrecte: {f_opt}"
    print(f"✓ Projected Gradient: Solution = {x_opt}, f* = {f_opt:.6f}, iterations = {n_iter}")
    
    # Test 3: Contrainte d'égalité
    # min (x-2)^2 + (y-2)^2 sujet à x + y = 2
    # Solution: x = y = 1, f* = 2
    constraint_eq = Constraint(lambda x: x[0] + x[1] - 2, 'eq')
    x0 = np.array([0.5, 0.5])
    
    x_opt, f_opt, n_iter = penalty_method(
        f_shifted, grad_f_shifted, x0, [constraint_eq],
        mu_init=1.0, max_outer_iter=20, tol=1e-4
    )
    
    assert abs(x_opt[0] + x_opt[1] - 2.0) < 0.1, f"Penalty (eq): Contrainte non satisfaite: {x_opt[0] + x_opt[1]}"
    assert abs(f_opt - 2.0) < 0.3, f"Penalty (eq): Valeur optimale incorrecte: {f_opt}"
    print(f"✓ Penalty (égalité): Solution = {x_opt}, f* = {f_opt:.6f}, iterations = {n_iter}")
    print(f"  Contrainte x+y = {x_opt[0] + x_opt[1]:.6f} (attendu: 2.0)")
    
    print("✓ Tous les tests d'optimisation avec contrainte réussis!\n")


def test_projection():
    """Test de la fonction de projection"""
    print("Test 3: Fonction de projection")
    print("-" * 50)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Test 1: Point à l'intérieur
    x = np.array([0.5, 0.5])
    x_proj = project_box(x, bounds)
    assert np.allclose(x_proj, x), f"Projection incorrecte pour point intérieur: {x_proj}"
    print(f"✓ Projection point intérieur: {x} -> {x_proj}")
    
    # Test 2: Point à l'extérieur (au-dessus)
    x = np.array([1.5, 1.5])
    x_proj = project_box(x, bounds)
    expected = np.array([1.0, 1.0])
    assert np.allclose(x_proj, expected), f"Projection incorrecte pour point au-dessus: {x_proj}"
    print(f"✓ Projection point au-dessus: {x} -> {x_proj}")
    
    # Test 3: Point à l'extérieur (en-dessous)
    x = np.array([-0.5, -0.5])
    x_proj = project_box(x, bounds)
    expected = np.array([0.0, 0.0])
    assert np.allclose(x_proj, expected), f"Projection incorrecte pour point en-dessous: {x_proj}"
    print(f"✓ Projection point en-dessous: {x} -> {x_proj}")
    
    print("✓ Tous les tests de projection réussis!\n")


def run_all_tests():
    """Exécuter tous les tests"""
    print("=" * 60)
    print("TESTS DES ALGORITHMES D'OPTIMISATION")
    print("=" * 60)
    print()
    
    try:
        test_unconstrained_optimization()
        test_constrained_optimization()
        test_projection()
        
        print("=" * 60)
        print("✓ TOUS LES TESTS RÉUSSIS!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ ÉCHEC DU TEST: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
