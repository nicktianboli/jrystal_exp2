import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from batch_davidson import batch_davidson_matrix_free


def _assert_close(a, b, tol=1e-4):
  err = jnp.max(jnp.abs(a - b))
  assert err < tol, f"max abs diff {err} >= {tol}"


def _random_unitary_batch(key, b, n):
  key_r, key_i = jax.random.split(key)
  Mr = jax.random.normal(key_r, (b, n, n), dtype=jnp.float32)
  Mi = jax.random.normal(key_i, (b, n, n), dtype=jnp.float32)
  M = Mr + 1j * Mi
  return jax.vmap(lambda Mi: jnp.linalg.qr(Mi, mode="reduced")[0])(M)


def _hermitian_with_spectrum_batch(key, evals):
  b, n = evals.shape
  Q = _random_unitary_batch(key, b, n)
  return jnp.einsum("bij,bj,bkj->bik", Q, evals, jnp.conj(Q))


def test_batched_davidson_batched_matmul():
  key = jax.random.PRNGKey(0)
  b = 4
  n = 20
  k = 10
  base = jnp.linspace(-1.0, 2.0, n).astype(jnp.float32)
  offsets = jnp.linspace(-0.2, 0.3, b).astype(jnp.float32)[:, None]
  evals_true = base[None, :] + offsets
  key, sub = jax.random.split(key)
  A = _hermitian_with_spectrum_batch(sub, evals_true)

  def matmul(X):
    assert X.ndim == 3
    return jnp.einsum("bij,bjk->bik", A, X)

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[..., ::-1][..., :k]
  evals_ref_smallest = evals_ref[..., :k]
  shift_l = evals_ref[..., 0] - 1e-3
  shift_s = evals_ref[..., -1] + 1e-3

  evals_l, _ = batch_davidson_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    batch_size=b,
    which="largest",
    maxit=140,
    tol=1e-6,
    shift=shift_l,
  )
  evals_s, _ = batch_davidson_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    batch_size=b,
    which="smallest",
    maxit=140,
    tol=1e-6,
    shift=shift_s,
  )
  print(f"Eigenvals(Davidson): {jnp.sort(evals_l, axis=-1)[:, ::-1]}")
  print(f"Eigenvals(Reference): {evals_ref_largest}")
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=1e-3
  )
  print(f"Eigenvals(Davidson): {jnp.sort(evals_s, axis=-1)}")
  print(f"Eigenvals(Reference): {evals_ref_smallest}")
  _assert_close(jnp.sort(evals_s, axis=-1), evals_ref_smallest, tol=1e-3)


def test_batched_davidson_vmap_matmul():
  key = jax.random.PRNGKey(1)
  b = 3
  n = 16
  k = 2
  base = jnp.linspace(0.5, 2.5, n).astype(jnp.float32)
  offsets = jnp.linspace(0.0, 0.4, b).astype(jnp.float32)[:, None]
  evals_true = base[None, :] + offsets
  key, sub = jax.random.split(key)
  A = _hermitian_with_spectrum_batch(sub, evals_true)

  def matmul(X):
    return jax.vmap(lambda Ai, Xi: Ai @ Xi, in_axes=(0, 0))(A, X)

  key, sub = jax.random.split(key)
  v0r = jax.random.normal(sub, (b, n, k), dtype=jnp.float32)
  v0i = jax.random.normal(sub, (b, n, k), dtype=jnp.float32)
  v0 = v0r + 1j * v0i

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[..., ::-1][..., :k]
  shift_l = evals_ref[..., 0] - 1e-3

  evals_l, _ = batch_davidson_matrix_free(
    matmul=matmul,
    k=k,
    v0=v0,
    which="largest",
    maxit=140,
    tol=1e-6,
    shift=shift_l,
  )
  print(f"Eigenvals(Davidson): {jnp.sort(evals_l, axis=-1)[:, ::-1]}")
  print(f"Eigenvals(Reference): {evals_ref_largest}")
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=1e-3
  )


def test_batched_davidson_jit():
  key = jax.random.PRNGKey(2)
  b = 2
  n = 12
  k = 2
  base = jnp.linspace(-0.5, 1.5, n).astype(jnp.float32)
  offsets = jnp.linspace(-0.1, 0.2, b).astype(jnp.float32)[:, None]
  evals_true = base[None, :] + offsets
  key, sub = jax.random.split(key)
  A = _hermitian_with_spectrum_batch(sub, evals_true)

  def matmul(X):
    return jnp.einsum("bij,bjk->bik", A, X)

  key, sub = jax.random.split(key)
  v0r = jax.random.normal(sub, (b, n, k), dtype=jnp.float32)
  v0i = jax.random.normal(sub, (b, n, k), dtype=jnp.float32)
  v0 = v0r + 1j * v0i

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[..., ::-1][..., :k]
  shift_l = evals_ref[..., 0] - 1e-3

  def solve(v0_local):
    return batch_davidson_matrix_free(
      matmul=matmul,
      k=k,
      v0=v0_local,
      which="largest",
      maxit=120,
      tol=1e-6,
      shift=shift_l,
    )

  evals_l, _ = jax.jit(solve)(v0)
  print(f"Eigenvals(Davidson): {jnp.sort(evals_l, axis=-1)[:, ::-1]}")
  print(f"Eigenvals(Reference): {evals_ref_largest}")
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=1e-3
  )


def test_batched_davidson_scalar_shift():
  key = jax.random.PRNGKey(3)
  b = 2
  n = 14
  k = 10
  base = jnp.linspace(-1.0, 1.0, n).astype(jnp.float32)
  offsets = jnp.linspace(-0.2, 0.2, b).astype(jnp.float32)[:, None]
  evals_true = base[None, :] + offsets
  key, sub = jax.random.split(key)
  A = _hermitian_with_spectrum_batch(sub, evals_true)

  def matmul(X):
    return jnp.einsum("bij,bjk->bik", A, X)

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[..., ::-1][..., :k]
  shift_l = float(evals_ref[0, 0] - 1e-3)

  evals_l, _ = batch_davidson_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    batch_size=b,
    which="largest",
    maxit=140,
    tol=1e-6,
    shift=shift_l,
  )
  print(f"Eigenvals(Davidson): {jnp.sort(evals_l, axis=-1)[:, ::-1]}")
  print(f"Eigenvals(Reference): {evals_ref_largest}")
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=1e-3
  )


def test_batched_davidson_precond():
  b = 3
  n = 18
  k = 2
  base = jnp.linspace(1.0, 2.0, n).astype(jnp.float32)
  offsets = jnp.linspace(0.0, 0.2, b).astype(jnp.float32)[:, None]
  evals_true = base[None, :] + offsets
  A = jax.vmap(lambda d: jnp.diag(d).astype(jnp.complex64))(evals_true)

  def matmul(X):
    return jnp.einsum("bij,bjk->bik", A, X)

  evals_ref = evals_true
  evals_ref_largest = evals_ref[..., ::-1][..., :k]
  shift_l = evals_ref[..., 0] - 1e-3

  precond = 1.0 / (evals_true - shift_l[:, None])

  evals_l, _ = batch_davidson_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    batch_size=b,
    which="largest",
    maxit=160,
    tol=1e-6,
    shift=shift_l,
    preconditioner=precond,
  )
  evals_l = jnp.real(evals_l)
  assert jnp.all(jnp.isfinite(evals_l))
  assert evals_l.shape == (b, k)
  assert jnp.all(evals_l <= jnp.max(evals_ref) + 1.0)
  assert jnp.all(evals_l >= jnp.min(evals_ref) - 1.0)


def main():
  test_batched_davidson_batched_matmul()
  test_batched_davidson_vmap_matmul()
  test_batched_davidson_jit()
  test_batched_davidson_scalar_shift()
  test_batched_davidson_precond()
  print("All batch Davidson tests passed.")


if __name__ == "__main__":
  main()
