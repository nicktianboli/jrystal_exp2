import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from batch_lobpcg import batch_lobpcg_matrix_free


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


def test_batched_lobpcg_batched_matmul():
  key = jax.random.PRNGKey(0)
  b = 4
  n = 20
  k = 3
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

  evals_l, _ = batch_lobpcg_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    batch_size=b,
    which="largest",
    maxit=80,
    tol=1e-6,
  )
  evals_s, _ = batch_lobpcg_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    batch_size=b,
    which="smallest",
    maxit=80,
    tol=1e-6,
  )
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=5e-4
  )
  _assert_close(jnp.sort(evals_s, axis=-1), evals_ref_smallest, tol=5e-4)


def test_batched_lobpcg_vmap_matmul():
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

  evals_l, _ = batch_lobpcg_matrix_free(
    matmul=matmul,
    k=k,
    v0=v0,
    which="largest",
    maxit=80,
    tol=1e-6,
  )
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=5e-4
  )


def test_batched_lobpcg_jit():
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

  def solve(v0_local):
    return batch_lobpcg_matrix_free(
      matmul=matmul,
      k=k,
      v0=v0_local,
      which="largest",
      maxit=60,
      tol=1e-6,
    )

  evals_ref, _ = jnp.linalg.eigh(A)
  evals_ref_largest = evals_ref[..., ::-1][..., :k]

  evals_l, _ = jax.jit(solve)(v0)
  _assert_close(
    jnp.sort(evals_l, axis=-1)[:, ::-1], evals_ref_largest, tol=5e-4
  )


def main():
  test_batched_lobpcg_batched_matmul()
  test_batched_lobpcg_vmap_matmul()
  test_batched_lobpcg_jit()
  print("All batch LOBPCG tests passed.")


if __name__ == "__main__":
  main()
