import jax
import jax.numpy as jnp


def block_mgs(W, V=None, reorth=True):
  """Orthonormalize columns with QR."""
  if W.ndim == 1:
    W = W[:, None]
  if W.shape[1] == 0:
    return jnp.zeros_like(W)
  if V is not None and V.size:
    Vh = jnp.conj(V).T
    W = W - V @ (Vh @ W)
    if reorth:
      W = W - V @ (Vh @ W)
  Q, _ = jnp.linalg.qr(W, mode="reduced")
  return Q


def lobpcg_matrix_free(
  matmul,
  k,
  n=None,
  tol=1e-8,
  maxit=200,
  v0=None,
  preconditioner=None,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
):
  """LOBPCG for k extremal eigenpairs (matrix-free).

  Args:
    matmul: callable. matmul(X) with X shape (n, p) returns A@X.
    k: int. Number of eigenpairs.
    n: int. Problem size (required if v0 is None).
    tol: float. Convergence threshold on residual norms.
    maxit: int. Maximum iterations.
    v0: ndarray or None. Initial block, shape (n, k).
    preconditioner: callable, ndarray, or None. If callable, M^{-1}(X) with
      X shape (n, p). If ndarray, elementwise scaling; vector shape (n,) or
      (n, 1) is broadcast across columns.
    which: "largest" or "smallest".
    reorth: bool. Use reorthogonalization in MGS.
    seed: int. RNG seed if v0 is None.
    return_history: bool. If True, return residual history with length maxit.
  """
  if v0 is None and n is None:
    raise ValueError("Provide n when v0 is None.")
  if which not in ("largest", "smallest"):
    raise ValueError('which must be "largest" or "smallest"')

  def _matmul_cols(X):
    X2 = X if X.ndim == 2 else X[:, None]
    AX = matmul(X2)
    return AX if AX.ndim == 2 else AX[:, None]

  def _apply_precond(X):
    if preconditioner is None:
      return X
    X2 = X if X.ndim == 2 else X[:, None]
    if callable(preconditioner):
      Y = preconditioner(X2)
      return Y if Y.ndim == 2 else Y[:, None]
    P = jnp.asarray(preconditioner)
    if P.ndim == 1:
      P = P[:, None]
    return P * X2

  # Initialize X
  if v0 is not None:
    X = jnp.asarray(v0)
    if X.ndim == 1:
      X = X[:, None]
    if X.shape[1] > k:
      X = X[:, :k]
    n = X.shape[0]
  else:
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (n, k))

  X = block_mgs(X, V=None, reorth=reorth)
  AX0 = _matmul_cols(X)
  if AX0.dtype != X.dtype:
    X = X.astype(AX0.dtype)
  P = jnp.zeros_like(X)

  def _update(X, P):
    AX = _matmul_cols(X)
    w = jnp.sum(jnp.conj(X) * AX, axis=0)
    R = AX - X * w[None, :]
    Z = _apply_precond(R)
    res = jnp.linalg.norm(R, axis=0)

    B = jnp.concatenate([X, P, Z], axis=1)
    B = block_mgs(B, V=None, reorth=reorth)
    AB = _matmul_cols(B)
    T = jnp.conj(B).T @ AB
    evals, vecs = jnp.linalg.eigh(T)
    evals = jnp.real(evals)
    idx = jnp.argsort(evals)
    idx = idx[::-1] if which == "largest" else idx
    vecs = vecs[:, idx[:k]]
    X_new = B @ vecs
    X_new = block_mgs(X_new, V=None, reorth=reorth)

    # Update P as component of new direction orthogonal to X
    delta = X_new - X @ (jnp.conj(X).T @ X_new)
    P_new = block_mgs(delta, V=None, reorth=reorth)
    return X_new, P_new, res

  res0 = jnp.full((k,), jnp.inf, dtype=jnp.real(X).dtype)

  def _step(carry, _):
    X, P, res, converged = carry

    def _do_update(_):
      Xn, Pn, resn = _update(X, P)
      conv = jnp.max(resn) < tol
      return Xn, Pn, resn, conv

    def _skip(_):
      return X, P, res, converged

    Xn, Pn, resn, conv = jax.lax.cond(
      converged, _skip, _do_update, operand=None
    )
    return (Xn, Pn, resn, conv), resn

  def _body(i, carry):
    (Xn, Pn, resn, conv), _ = _step(carry, None)
    return (Xn, Pn, resn, conv)

  if return_history:
    (X, P, res, _), history = jax.lax.scan(
      _step, (X, P, res0, False), None, length=maxit
    )
  else:
    X, P, res, _ = jax.lax.fori_loop(0, maxit, _body, (X, P, res0, False))

  AX = _matmul_cols(X)
  evals = jnp.real(jnp.sum(jnp.conj(X) * AX, axis=0))
  order = jnp.argsort(evals)
  order = order[::-1] if which == "largest" else order
  evals = evals[order][:k]
  evecs = X[:, order][:, :k]
  if return_history:
    return evals, evecs, history
  return evals, evecs
