import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from davidson import davidson_topk_matrix_free


def _assert_close(a, b, tol=1e-4):
  err = jnp.max(jnp.abs(a - b))
  assert err < tol, f"max abs diff {err} >= {tol}"


def _random_unitary(key, n):
  key_r, key_i = jax.random.split(key)
  Mr = jax.random.normal(key_r, (n, n), dtype=jnp.float32)
  Mi = jax.random.normal(key_i, (n, n), dtype=jnp.float32)
  M = Mr + 1j * Mi
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
  print(evals_ref)
  evals_ref_largest = evals_ref[::-1][:k]
  evals_ref_smallest = evals_ref[:k]
  shift_l = evals_ref[0] - 1e-3
  shift_s = evals_ref[-1] + 1e-3

  evals_l, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    which="largest",
    maxit=120,
    tol=1e-6,
    shift=shift_l,
  )
  evals_s, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    which="smallest",
    maxit=120,
    tol=1e-6,
    shift=shift_s,
  )
  _assert_close(jnp.sort(evals_l)[::-1], evals_ref_largest, tol=8e-4)
  _assert_close(jnp.sort(evals_s), evals_ref_smallest, tol=8e-4)


def test_hermitian_complex():
  key = jax.random.PRNGKey(1)
  n = 20
  k = 3
  key, sub = jax.random.split(key)
  evals_true = jnp.linspace(0.5, 2.5, n).astype(jnp.float32)

  A = _hermitian_with_spectrum(sub, evals_true)

  def matmul(X):
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[::-1][:k]
  shift_l = evals_ref[0] - 1e-3

  evals_l, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    which="largest",
    maxit=160,
    tol=1e-6,
    shift=shift_l,
  )
  print(f"Eigenvals(Davidson): {jnp.sort(evals_l)[::-1]}")
  print(f"Eigenvals(Reference): {evals_ref_largest}")
  _assert_close(jnp.sort(evals_l)[::-1], evals_ref_largest, tol=1e-3)


def test_indefinite_smallest():
  key = jax.random.PRNGKey(2)
  n = 18
  k = 2
  key, sub = jax.random.split(key)
  evals_true = jnp.linspace(-2.0, 2.0, n).astype(jnp.float32)
  A = _hermitian_with_spectrum(sub, evals_true)

  def matmul(X):
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_smallest = evals_ref[:k]
  shift_s = evals_ref[-1] + 1e-3

  evals_s, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    which="smallest",
    maxit=160,
    tol=1e-6,
    shift=shift_s,
  )
  print(f"Eigenvals(Davidson): {jnp.sort(evals_s)}")
  print(f"Eigenvals(Reference): {evals_ref_smallest}")
  _assert_close(jnp.sort(evals_s), evals_ref_smallest, tol=1e-3)


def test_davidson_jit():
  key = jax.random.PRNGKey(3)
  n = 16
  k = 2
  key, sub = jax.random.split(key)
  evals_true = jnp.linspace(-1.0, 1.0, n).astype(jnp.float32)
  A = _hermitian_with_spectrum(sub, evals_true)

  def matmul(X):
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[::-1][:k]
  shift_l = evals_ref[0] - 1e-3

  key, sub = jax.random.split(key)
  v0r = jax.random.normal(sub, (n, k), dtype=jnp.float32)
  v0i = jax.random.normal(sub, (n, k), dtype=jnp.float32)
  v0 = v0r + 1j * v0i

  def solve(v0_local):
    return davidson_topk_matrix_free(
      matmul=matmul,
      k=k,
      v0=v0_local,
      which="largest",
      maxit=120,
      tol=1e-6,
      shift=shift_l,
    )

  evals_l, _ = jax.jit(solve)(v0)
  print(f"Eigenvals(Davidson): {jnp.sort(evals_l)[::-1]}")
  print(f"Eigenvals(Reference): {evals_ref_largest}")
  _assert_close(jnp.sort(evals_l)[::-1], evals_ref_largest, tol=1e-3)


def test_davidson_precond():
  n = 20
  k = 3
  evals_true = jnp.linspace(1.0, 2.0, n).astype(jnp.float32)
  A = jnp.diag(evals_true).astype(jnp.complex64)

  def matmul(X):
    return A @ X

  evals_ref = evals_true
  evals_ref_largest = evals_ref[::-1][:k]
  shift_l = evals_ref[0] - 1e-3

  # Diagonal preconditioner for the shifted operator (exact for diagonal A).
  precond = 1.0 / (evals_true - shift_l)

  evals_l, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    which="largest",
    maxit=160,
    tol=1e-6,
    shift=shift_l,
    preconditioner=precond,
  )
  evals_l = jnp.real(evals_l)
  assert jnp.all(jnp.isfinite(evals_l))
  assert evals_l.shape == (k,)
  assert jnp.all(evals_l <= evals_ref.max() + 1.0)
  assert jnp.all(evals_l >= evals_ref.min() - 1.0)


def main():
  test_spd_largest_smallest()
  test_hermitian_complex()
  test_indefinite_smallest()
  test_davidson_jit()
  test_davidson_precond()
  print("All Davidson tests passed.")


if __name__ == "__main__":
  main()
