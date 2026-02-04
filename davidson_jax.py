import jax
import jax.numpy as jnp


def block_mgs(W, V=None, reorth=True, eps=1e-12):
  """Orthonormalize columns using QR (JIT-friendly).

  Args:
    W: (n, p) ndarray. Vectors to orthonormalize (columns).
    V: (n, m) ndarray or None. Existing orthonormal basis to orthogonalize
      against first.
    reorth: bool. Whether to do a second pass against V.
    eps: float. Unused. Kept for API compatibility.

  Returns:
    Q: (n, q) ndarray. Orthonormal columns spanning (approximately) the range of
      W after removing components in span(V).
  """
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


def _subspace_topk_core(
  matmul,
  k,
  n,
  tol,
  maxit,
  v0,
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

  def _estimate_maxeig(matmul_fn, x0, iters=8):
    def power_step(i, x):
      y = matmul_fn(x)
      y = y / (jnp.linalg.norm(y) + 1e-12)
      return y

    x = jax.lax.fori_loop(0, iters, power_step, x0)
    return jnp.real(jnp.vdot(x, matmul_fn(x)))

  if shift is None:
    # Use power iteration to estimate a conservative shift so the spectrum
    # becomes sign-definite and the block power iteration is stable.
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

  def body(V, _):
    AV = matmul_eff(V)  # (n,k)
    T = jnp.conj(V).T @ AV  # (k,k) Hermitian
    w, Y = jnp.linalg.eigh(T)  # ascending
    idx = jnp.argsort(w)
    idx = idx[::-1] if which == "largest" else idx
    Y = Y[:, idx]
    Yk = Y[:, :k]
    U = V @ Yk
    AU = AV @ Yk
    w_sorted = w[idx]
    R = AU - U * w_sorted[:k][None, :]
    res = jnp.linalg.norm(R, axis=0)
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
  diagA=None,
  shift=None,
  delta=1e-12,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
):
  """Fixed-subspace iteration for top-k eigenpairs (matrix-free).

  Args:
    matmul: callable. Function matmul(X) returning A@X for X shape (n,) or
      (n, p). A should be real symmetric or complex Hermitian.
    k: int. Number of largest eigenpairs.
    n: int or None. Problem size. Required if v0 is None. If v0 has leading
      batch dimensions, n is inferred from v0.
    tol: float. Convergence threshold on residual norms.
    maxit: int. Maximum outer iterations.
    m_max: unused. Kept for API compatibility.
    restart: unused. Kept for API compatibility.
    v0: ndarray or None. Initial vectors, shape (n, k) or (..., n, k). If None,
      random normal (real).
    diagA: ndarray or None. If provided and shift is None, a shift is estimated
      from min(diagA) to bias toward largest-by-value.
    shift: float or None. If set, uses A - shift * I during iteration and
      shifts eigenvalues back at the end.
    delta: unused. Kept for API compatibility.
    which: "largest" or "smallest". Select eigenvalues by value order.
    reorth: bool. Use reorthogonalization in MGS (recommended).
    seed: int. RNG seed if v0 is None.
    return_history: bool. If True, return residual history.

  Returns:
    evals: (k,) ndarray or (..., k). Largest eigenvalues in descending order.
    evecs: (n, k) ndarray or (..., n, k). Corresponding eigenvectors
      (orthonormal columns).
    history: (iters, k) ndarray or (..., iters, k), optional. Residual norms per
      iteration.
  """
  if v0 is None and n is None:
    raise ValueError("Provide n when v0 is None for matrix-free mode.")
  if which not in ("largest", "smallest"):
    raise ValueError('which must be "largest" or "smallest"')

  if v0 is not None and v0.ndim > 2:
    batch_shape = v0.shape[:-2]
    flat_v0 = v0.reshape((-1,) + v0.shape[-2:])
    if diagA is not None and diagA.ndim > 1:
      flat_diag = diagA.reshape((-1,) + diagA.shape[-1:])
    else:
      flat_diag = [diagA] * flat_v0.shape[0]

    evals_list = []
    evecs_list = []
    hist_list = []
    for i in range(flat_v0.shape[0]):
      out = _subspace_topk_core(
        matmul=matmul,
        k=k,
        n=None,
        tol=tol,
        maxit=maxit,
        v0=flat_v0[i],
        diagA=flat_diag[i],
        shift=shift,
        which=which,
        reorth=reorth,
        seed=seed,
        return_history=return_history,
      )
      if return_history:
        evals_i, evecs_i, hist_i = out
        hist_list.append(hist_i)
      else:
        evals_i, evecs_i = out
      evals_list.append(evals_i)
      evecs_list.append(evecs_i)

    evals = jnp.stack(evals_list, axis=0).reshape(batch_shape + (k,))
    evecs = jnp.stack(evecs_list, axis=0).reshape(
      batch_shape + evecs_list[0].shape
    )
    if return_history:
      history = jnp.stack(hist_list, axis=0).reshape(
        batch_shape + hist_list[0].shape
      )
      return evals, evecs, history
    return evals, evecs

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
    diagA=diagA,
    shift=shift,
    which=which,
    reorth=reorth,
    seed=seed,
    return_history=return_history,
  )
