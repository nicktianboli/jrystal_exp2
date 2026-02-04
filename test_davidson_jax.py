import jax
import jax.numpy as jnp

from davidson_jax import davidson_topk_matrix_free


def main():
  key = jax.random.PRNGKey(0)
  n = 64
  k = 10

  key, sub = jax.random.split(key)
  M = jax.random.normal(sub, (n, n), dtype=jnp.complex64)
  A = (M + M.T.conj()) / 2.0
  A = A + jnp.eye(n, dtype=A.dtype) / 10

  def matmul(X):
    return A @ X

  evals_ref, _ = jnp.linalg.eigh(A)
  min_eig = evals_ref[0]
  max_eig = evals_ref[-1]

  evals_dav_largest, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    tol=1e-8,
    maxit=2000,
    which="largest",
    shift=min_eig - 1e-6,
    seed=123,
  )

  evals_dav_smallest, _ = davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=n,
    tol=1e-8,
    maxit=2000,
    which="smallest",
    shift=max_eig + 1e-6,
    seed=123,
  )

  evals_ref_largest = evals_ref[::-1][:k]
  evals_ref_smallest = evals_ref[:k]

  print("Davidson top-k (largest):", evals_dav_largest)
  print("Reference top-k (largest):", evals_ref_largest)
  print(
    "Max abs diff:", jnp.max(jnp.abs(evals_dav_largest - evals_ref_largest))
  )

  print("Davidson top-k (smallest):", evals_dav_smallest)
  print("Reference top-k (smallest):", evals_ref_smallest)
  print(
    "Max abs diff:", jnp.max(jnp.abs(evals_dav_smallest - evals_ref_smallest))
  )


if __name__ == "__main__":
  main()
