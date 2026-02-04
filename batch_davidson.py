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


def block_mgs_batch(W, V=None, reorth=True):
  """Batched block MGS using vmap over the leading batch axes."""
  if W.ndim <= 2:
    return block_mgs(W, V=V, reorth=reorth)
  batch_shape = W.shape[:-2]
  W_flat = W.reshape((-1,) + W.shape[-2:])

  if V is None:
    Q_flat = jax.vmap(lambda Wi: block_mgs(Wi, V=None, reorth=reorth))(W_flat)
  else:
    V_arr = jnp.asarray(V)
    if V_arr.ndim <= 2:
      V_arr = jnp.broadcast_to(V_arr, batch_shape + V_arr.shape)
    V_flat = V_arr.reshape((-1,) + V_arr.shape[-2:])
    Q_flat = jax.vmap(
      lambda Wi, Vi: block_mgs(Wi, V=Vi, reorth=reorth)
    )(W_flat, V_flat)

  return Q_flat.reshape(batch_shape + Q_flat.shape[-2:])


def _estimate_maxeig_batched(matmul_fn, x0, iters=8):
  def power_step(i, x):
    y = matmul_fn(x)
    y = y / (jnp.linalg.norm(y, axis=-2, keepdims=True) + 1e-12)
    return y

  x = jax.lax.fori_loop(0, iters, power_step, x0)
  y = matmul_fn(x)
  return jnp.real(jnp.sum(jnp.conj(x) * y, axis=(-2, -1)))


def batch_davidson_matrix_free(
  matmul,
  k,
  n=None,
  tol=1e-8,
  maxit=200,
  v0=None,
  batch_size=None,
  preconditioner=None,
  shift=None,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
):
  """Fixed-subspace iteration for batched top-k eigenpairs (matrix-free).

  Args:
    matmul: callable. matmul(X) with X shape (b, n, p) returns A@X.
    k: int. Number of eigenpairs.
    n: int. Problem size (required if v0 is None).
    tol: float. Convergence threshold on residual norms (tracked only).
    maxit: int. Maximum iterations.
    v0: ndarray or None. Initial block, shape (b, n, k) or (n, k).
    batch_size: int or None. Required if v0 is None for batched mode.
    preconditioner: callable, ndarray, or None. If callable, M^{-1}(X) with
      X shape (..., n, p). If ndarray, elementwise scaling; vector shape (n,)
      or (..., n) is broadcast across columns.
    shift: float or ndarray or None. If None, estimated per batch.
      Scalar shifts are broadcast across the batch.
    which: "largest" or "smallest".
    reorth: bool. Use reorthogonalization in MGS.
    seed: int. RNG seed if v0 is None.
    return_history: bool. If True, return residual history with length maxit.
  """
  if v0 is None and n is None:
    raise ValueError("Provide n when v0 is None.")
  if v0 is None and batch_size is None:
    raise ValueError("Provide batch_size when v0 is None for batched mode.")
  if which not in ("largest", "smallest"):
    raise ValueError('which must be "largest" or "smallest"')

  def _matmul_cols(X):
    assert X.ndim == 3, "X must be (batch, n, p)"
    # X2 = X if X.ndim == 3 else X[..., None]
    AX = matmul(X)
    # return AX if AX.ndim == X.ndim else AX[..., None]
    return AX

  def _apply_precond(X):
    if preconditioner is None:
      return X
    X2 = X if X.ndim >= 2 else X[..., None]
    if callable(preconditioner):
      Y = preconditioner(X2)
      return Y if Y.ndim == X2.ndim else Y[..., None]
    P = jnp.asarray(preconditioner)
    if P.ndim == 1:
      P = P.reshape((1,) * (X2.ndim - 2) + (P.shape[0], 1))
    elif P.ndim == 2 and P.shape[-1] == 1 and P.shape[0] == X2.shape[-2]:
      P = P.reshape((1,) * (X2.ndim - 2) + P.shape)
    elif P.ndim == X2.ndim - 1:
      P = P[..., None]
    return P * X2

  # Initialize V
  if v0 is not None:
    V = jnp.asarray(v0)
    if V.ndim == 1:
      V = V[:, None]
    if V.ndim == 2:
      V = V[None, ...]
    if V.shape[-1] > k:
      V = V[..., :k]
    n = V.shape[-2]
    batch_size = V.shape[0]
  else:
    key = jax.random.PRNGKey(seed)
    V = jax.random.normal(key, (batch_size, n, k))

  V = block_mgs_batch(V, V=None, reorth=reorth)
  AV0 = _matmul_cols(V)
  if AV0.dtype != V.dtype:
    V = V.astype(AV0.dtype)

  if shift is None:
    x0 = _matmul_cols(V[..., :, 0:1])
    x0 = x0 / (jnp.linalg.norm(x0, axis=-2, keepdims=True) + 1e-12)
    if which == "largest":
      min_est = -_estimate_maxeig_batched(
        lambda x: -_matmul_cols(x)[..., 0:1], x0
      )
      margin = 0.1 * (jnp.abs(min_est) + 1.0)
      shift = min_est - margin
    else:
      max_est = _estimate_maxeig_batched(
        lambda x: _matmul_cols(x)[..., 0:1], x0
      )
      margin = 0.1 * (jnp.abs(max_est) + 1.0)
      shift = max_est + margin

  if shift is not None:
    shift = jnp.asarray(shift, dtype=jnp.real(V).dtype)
    if shift.ndim == 0:
      shift = jnp.full((batch_size,), shift, dtype=shift.dtype)
    else:
      shift = jnp.broadcast_to(shift, (batch_size,))

  def matmul_eff(X):
    AX = _matmul_cols(X)
    if shift is None:
      return AX
    return AX - shift[..., None, None] * X

  use_precond = preconditioner is not None

  def body(V, _):
    AV = matmul_eff(V)
    T = jnp.einsum("...ni,...nj->...ij", jnp.conj(V), AV)
    w, Y = jnp.linalg.eigh(T)
    idx = jnp.argsort(w, axis=-1)
    idx = idx[..., ::-1] if which == "largest" else idx
    w = jnp.take_along_axis(w, idx, axis=-1)
    idx_mat = jnp.broadcast_to(idx[..., None, :], Y.shape)
    Y = jnp.take_along_axis(Y, idx_mat, axis=-1)
    Yk = Y[..., :, :k]
    U = jnp.einsum("...ni,...ik->...nk", V, Yk)
    AU = jnp.einsum("...ni,...ik->...nk", AV, Yk)
    R = AU - U * w[..., :k][..., None, :]
    res = jnp.linalg.norm(R, axis=-2)
    if use_precond:
      Z = _apply_precond(R)
      Vnext = block_mgs_batch(AU + Z, V=None, reorth=reorth)
    else:
      Vnext = block_mgs_batch(AU, V=None, reorth=reorth)
    return Vnext, res

  V_final, history = jax.lax.scan(body, V, None, length=maxit)

  AV = matmul_eff(V_final)
  T = jnp.einsum("...ni,...nj->...ij", jnp.conj(V_final), AV)
  w, Y = jnp.linalg.eigh(T)
  idx = jnp.argsort(w, axis=-1)
  idx = idx[..., ::-1] if which == "largest" else idx
  w = jnp.take_along_axis(w, idx, axis=-1)
  idx_mat = jnp.broadcast_to(idx[..., None, :], Y.shape)
  Y = jnp.take_along_axis(Y, idx_mat, axis=-1)
  U = jnp.einsum("...ni,...ik->...nk", V_final, Y[..., :, :k])
  U = block_mgs_batch(U, V=None, reorth=reorth)
  AU = matmul_eff(U)
  evals = jnp.sum(jnp.conj(U) * AU, axis=-2)
  if shift is not None:
    evals = evals + shift[..., None]
  order = jnp.argsort(evals, axis=-1)
  order = order[..., ::-1] if which == "largest" else order
  evals = jnp.take_along_axis(evals, order, axis=-1)[..., :k]
  order = jnp.broadcast_to(order[..., None, :], U.shape[:-1] + (k,))
  evecs = jnp.take_along_axis(U, order, axis=-1)[..., :k]

  if return_history:
    hist = jnp.moveaxis(history, 0, -2)
    return evals, evecs, hist
  return evals, evecs


davidson_matrix_free_batched = batch_davidson_matrix_free


__all__ = ["batch_davidson_matrix_free", "davidson_matrix_free_batched"]
