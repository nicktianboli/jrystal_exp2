import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from lobpcg import lobpcg_matrix_free


def _assert_close(a, b, tol=1e-4):
  err = jnp.max(jnp.abs(a - b))
  assert err < tol, f"max abs diff {err} >= {tol}"


def _random_unitary(key, n):
  M = jax.random.normal(key, (n, n), dtype=jnp.complex64)
  Q, _ = jnp.linalg.qr(M)
  return Q


def _hermitian_with_spectrum(key, evals):
  n = evals.shape[0]
  Q = _random_unitary(key, n)
  D = jnp.diag(evals)
  return Q.conj().T @ D @ Q


def test_spd_largest_smallest():
  key = jax.random.PRNGKey(0)
  n = 24
  k = 3
  key, sub = jax.random.split(key)
  evals_true = jnp.linspace(-1.0, 3.0, n).astype(jnp.float32)
  A = _hermitian_with_spectrum(sub, evals_true)

  def matmul(X):
    assert X.ndim == 2
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[::-1][:k]
  evals_ref_smallest = evals_ref[:k]

  evals_l, _ = lobpcg_matrix_free(
    matmul=matmul, k=k, n=n, which="largest", maxit=80, tol=1e-6
  )
  evals_s, _ = lobpcg_matrix_free(
    matmul=matmul, k=k, n=n, which="smallest", maxit=80, tol=1e-6
  )
  _assert_close(jnp.sort(evals_l)[::-1], evals_ref_largest, tol=5e-4)
  _assert_close(jnp.sort(evals_s), evals_ref_smallest, tol=5e-4)


def test_hermitian_complex():
  key = jax.random.PRNGKey(1)
  n = 20
  k = 3
  key, sub = jax.random.split(key)
  evals_true = jnp.linspace(0.5, 2.5, n).astype(jnp.float32)
  A = _hermitian_with_spectrum(sub, evals_true)

  def matmul(X):
    assert X.ndim == 2
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[::-1][:k]

  evals_l, _ = lobpcg_matrix_free(
    matmul=matmul, k=k, n=n, which="largest", maxit=100, tol=1e-6
  )
  _assert_close(jnp.sort(evals_l)[::-1], evals_ref_largest, tol=1e-3)


def test_indefinite_smallest():
  key = jax.random.PRNGKey(2)
  n = 18
  k = 2
  key, sub = jax.random.split(key)
  evals_true = jnp.linspace(-2.0, 2.0, n).astype(jnp.float32)
  A = _hermitian_with_spectrum(sub, evals_true)

  def matmul(X):
    assert X.ndim == 2
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_smallest = evals_ref[:k]

  evals_s, _ = lobpcg_matrix_free(
    matmul=matmul, k=k, n=n, which="smallest", maxit=100, tol=1e-6
  )
  _assert_close(jnp.sort(evals_s), evals_ref_smallest, tol=1e-3)


def main():
  test_spd_largest_smallest()
  test_hermitian_complex()
  test_indefinite_smallest()
  print("All LOBPCG tests passed.")


if __name__ == "__main__":
  main()
