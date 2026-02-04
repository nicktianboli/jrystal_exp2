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


def _estimate_maxeig(matmul_fn, x0, iters=8):
  def power_step(i, x):
    y = matmul_fn(x)
    y = y / (jnp.linalg.norm(y) + 1e-12)
    return y

  x = jax.lax.fori_loop(0, iters, power_step, x0)
  y = matmul_fn(x)
  return jnp.real(jnp.sum(jnp.conj(x) * y))


def _subspace_topk_core(
  matmul,
  k,
  n,
  tol,
  maxit,
  v0,
  preconditioner,
  diagA,
  shift,
  which,
  reorth,
  seed,
  return_history,
):
  """Core fixed-subspace iteration on a single 2D problem."""
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

  # Determine n and initialize V
  if v0 is not None:
    V0 = jnp.asarray(v0)
    if V0.ndim == 1:
      V0 = V0[:, None]
    if V0.shape[1] > k:
      V0 = V0[:, :k]
    n = V0.shape[0]
  else:
    key = jax.random.PRNGKey(seed)
    V0 = jax.random.normal(key, (n, k))

  if shift is None:
    # Estimate a shift so the spectrum is sign-definite.
    x0 = _matmul_cols(V0[:, 0])[:, 0]
    x0 = x0 / (jnp.linalg.norm(x0) + 1e-12)
    if which == "largest":
      min_est = -_estimate_maxeig(lambda x: -_matmul_cols(x)[:, 0], x0)
      margin = 0.1 * (jnp.abs(min_est) + 1.0)
      shift = min_est - margin
    else:
      max_est = _estimate_maxeig(lambda x: _matmul_cols(x)[:, 0], x0)
      margin = 0.1 * (jnp.abs(max_est) + 1.0)
      shift = max_est + margin

  def matmul_eff(X):
    AX = _matmul_cols(X)
    if shift is None:
      return AX
    return AX - shift * X

  V0 = block_mgs(V0, V=None, reorth=reorth)
  AV0 = matmul_eff(V0)
  if AV0.dtype != V0.dtype:
    V0 = V0.astype(AV0.dtype)

  use_precond = preconditioner is not None

  def body(V, _):
    AV = matmul_eff(V)  # (n,k)
    T = jnp.conj(V).T @ AV  # (k,k) Hermitian
    w, Y = jnp.linalg.eigh(T)  # ascending
    idx = jnp.argsort(w)
    idx = idx[::-1] if which == "largest" else idx
    w = w[idx]
    Y = Y[:, idx]
    Yk = Y[:, :k]
    U = V @ Yk
    AU = AV @ Yk
    R = AU - U * w[:k][None, :]
    res = jnp.linalg.norm(R, axis=0)
    if use_precond:
      Z = _apply_precond(R)
      Vnext = block_mgs(AU + Z, V=None, reorth=reorth)
    else:
      Vnext = block_mgs(AU, V=None, reorth=reorth)
    return Vnext, res

  V_final, history = jax.lax.scan(body, V0, None, length=maxit)

  AV = matmul_eff(V_final)
  T = jnp.conj(V_final).T @ AV
  w, Y = jnp.linalg.eigh(T)
  idx = jnp.argsort(w)
  idx = idx[::-1] if which == "largest" else idx
  w = w[idx]
  Y = Y[:, idx]
  U = V_final @ Y[:, :k]
  U = block_mgs(U, V=None, reorth=reorth)
  AU = matmul_eff(U)
  evals = jnp.sum(jnp.conj(U) * AU, axis=0)
  if shift is not None:
    evals = evals + shift
  order = jnp.argsort(evals)
  order = order[::-1] if which == "largest" else order
  evals = evals[order][:k]
  U = U[:, order][:, :k]
  if return_history:
    return evals, U, history
  return evals, U


def davidson_topk_matrix_free(
  matmul,
  k,
  n=None,
  tol=1e-8,
  maxit=200,
  m_max=None,
  restart=True,
  v0=None,
  preconditioner=None,
  diagA=None,
  shift=None,
  delta=1e-12,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
):
  """Fixed-subspace iteration for top-k eigenpairs (matrix-free).

  Notes:
    m_max, restart, diagA, delta are accepted for API compatibility but unused.
    For batched problems, use batch_davidson_matrix_free in batch_davidson.py.
  """
  if v0 is None and n is None:
    raise ValueError("Provide n when v0 is None for matrix-free mode.")
  if which not in ("largest", "smallest"):
    raise ValueError('which must be "largest" or "smallest"')
  if v0 is not None and v0.ndim > 2:
    raise ValueError("Use batch_davidson_matrix_free for batched inputs.")

  core = jax.jit(
    _subspace_topk_core,
    static_argnames=(
      "k",
      "n",
      "maxit",
      "which",
      "reorth",
      "matmul",
      "return_history",
    ),
  )
  return core(
    matmul=matmul,
    k=k,
    n=n,
    tol=tol,
    maxit=maxit,
    v0=v0,
    preconditioner=preconditioner,
    diagA=diagA,
    shift=shift,
    which=which,
    reorth=reorth,
    seed=seed,
    return_history=return_history,
  )


def davidson_topk(
  A,
  k,
  tol=1e-8,
  maxit=200,
  m_max=None,
  restart=True,
  v0=None,
  preconditioner=None,
  diagA=None,
  shift=None,
  delta=1e-12,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
):
  """Top-k eigenpairs for a dense matrix or matrix-free callable."""
  if callable(A):
    return davidson_topk_matrix_free(
      matmul=A,
      k=k,
      n=None,
      tol=tol,
      maxit=maxit,
      m_max=m_max,
      restart=restart,
      v0=v0,
      preconditioner=preconditioner,
      diagA=diagA,
      shift=shift,
      delta=delta,
      which=which,
      reorth=reorth,
      seed=seed,
      return_history=return_history,
    )

  A_mat = jnp.asarray(A)

  def matmul(X):
    return A_mat @ X

  return davidson_topk_matrix_free(
    matmul=matmul,
    k=k,
    n=A_mat.shape[0],
    tol=tol,
    maxit=maxit,
    m_max=m_max,
    restart=restart,
    v0=v0,
    preconditioner=preconditioner,
    diagA=diagA,
    shift=shift,
    delta=delta,
    which=which,
    reorth=reorth,
    seed=seed,
    return_history=return_history,
  )


__all__ = ["davidson_topk", "davidson_topk_matrix_free"]
