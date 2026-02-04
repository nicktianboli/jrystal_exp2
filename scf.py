import jrystal as jr
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from batch_lobpcg import batch_lobpcg_matrix_free as lobpcg
from batch_davidson import batch_davidson_matrix_free as davidson
# from lobpcg import lobpcg_matrix_free as lobpcg
from diis import diis_init, diis_update


import time
jax.config.update("jax_enable_x64", True)


E_CUT = 100
G_VEC_SIZE = [48, ] * 3

CONVERGENCE = 1e-6
NUM_BANDS = 12
PATH = "/home/aiops/litb/projects/jrystal_exp2/"

SCF_MAX_ITER = 200

K_MESH = [1, 1, 1]

crystal = jr.crystal.Crystal.create_from_file(f"{PATH}diamond.xyz")

mask = jr.grid.spherical_mask(crystal.cell_vectors, G_VEC_SIZE, E_CUT)
print(f"mask ratio: {jnp.mean(mask):.4f}. num of G vectors {jnp.sum(mask)}")
g_vec = jr.grid.g_vectors(crystal.cell_vectors, G_VEC_SIZE)
# kpts = jnp.zeros([1, 3])
kpts = jr.grid.k_vectors(crystal.cell_vectors, K_MESH)
occ = jr.occupation.uniform(1, crystal.num_electron, 0, NUM_BANDS)
key = jax.random.PRNGKey(123)
param_coeff = jr.pw.param_init(
  key, NUM_BANDS, num_kpts=kpts.shape[0], freq_mask=mask
)
param_coeff['w_re'].shape
coeff_init = param_coeff['w_re'] + 1.j * param_coeff['w_im']
coeff_init = jnp.linalg.qr(coeff_init)[0]  # [s k g band]



def expand(c):
  return jr.utils.expand_coefficient(c, mask)


def density(
  coeff_compact: Float[Array, "s k g band"],
  occ: Float[Array, "s b k"],
):
  coeff = expand(coeff_compact)
  return jr.pw.density_grid(coeff, crystal.vol, occ)


def kerker(g_vec, mask):
  eff_g = g_vec.at[mask].get()
  g2 = jnp.sum(eff_g**2, axis=-1)
  return g2 / (1 + g2)


precond = kerker(g_vec, mask)


def efun(coeff_compact, dens):
  coeff = expand(coeff_compact)
  etot = jr.hamiltonian.hamiltonian_matrix_trace(
    coeff, crystal.positions, crystal.charges, dens, g_vec, kpts, crystal.vol,
    xc="lda_x", kohn_sham=True,
  )
  return etot


def Hvp_fun(
  coeff_compact: Float[Array, "s b g band"],
  dens: Float[Array, "s x y z"],
) -> Float[Array, "s k g band"]:

  return jax.grad(efun)(coeff_compact.conj(), dens)/2.


def lobpcg_matmul(c: Float[Array, "batch g band"], dens):
  return Hvp_fun(jnp.expand_dims(c, axis=(0)), dens)


@jax.jit
def update(coeff_compact, effective_dens):
  s, k, g, b = coeff_compact.shape
  eigval, evec = lobpcg(
    lambda c: lobpcg_matmul(c, effective_dens).reshape(s*k, g, -1),
    k=b,
    v0=coeff_compact.reshape(s*k, g, b),
    which="smallest",
    preconditioner=precond,
    seed=10,
    maxit=20,
    tol=1e-8,
  )    # eigval: [s*k, b], evec: [s*k, g, b]

  coeff_compact = evec.reshape(s, k, g, b)
  eigval = eigval.reshape(s, k, b)

  return coeff_compact, eigval


def check_convergence(
  new_coeff, old_coeff, new_loss, old_loss
):
  coeff_diff = jnp.mean(jnp.abs(new_coeff - old_coeff))
  loss_diff = jnp.mean(jnp.abs(new_loss - old_loss))

  return coeff_diff, loss_diff


def mixing(new_coeff, old_coeff, mixing_factor=0.95):
  return new_coeff * mixing_factor + old_coeff * (1 - mixing_factor)


def scf(iter, convergence_tol):
  coeff_compact = coeff_init
  dens = density(coeff_compact, occ)
  etol = efun(coeff_compact, dens)

  time_jit = 0
  time_total = 0

  diis_state = diis_init(
    max_hist=10, density_shape=dens.shape, dtype=dens.dtype
  )

  for i in range(iter):
    print(f"SCF iteration {i+1} started.")
    if i == 0:
      start_time = time.time()
      coeff_compact_new, eigval_new = update(coeff_compact, dens)
      coeff_compact_new = coeff_compact_new.conj()
      end_time = time.time()
      time_jit += end_time - start_time
    else:
      start_time = time.time()
      coeff_compact_new, eigval_new = update(coeff_compact, dens)
      coeff_compact_new = coeff_compact_new.conj()
      end_time = time.time()
      time_total += end_time - start_time

    dens_new = density(coeff_compact_new, occ)
    etol_new = jnp.sum(eigval_new*occ)
    dc, de = check_convergence(
      dens_new, dens,
      etol_new, etol
    )

    if dc < 1e-3 and de < convergence_tol:
      print(
        f"SCF converged in {i+1} iterations. "
        f"change of density: {dc: .6e}, change of energy: {de: .6e}"
      )
      break

    # dens = mixing(dens_new, dens, 0.98**i)
    dens_error = dens_new - dens
    diis_state, dens = diis_update(diis_state, dens_new, dens_error, 1e-8)
    etol = etol_new
    coeff_compact = coeff_compact_new

    print(eigval_new)
    print(f"SCF iteration {i+1} completed. Sum of eigenvalues: {etol: .6e}")
    print(
      f"change of coeffcient (avg.): {dc: .6e}, change of energy: {de: .6e}"
    )

  ewald_grid = jr.grid.translation_vectors(crystal.cell_vectors, 1e4)
  ewald = jr.ewald.ewald_coulomb_repulsion(
    crystal.positions, crystal.charges, g_vec, crystal.vol, 1e-1, ewald_grid
  )

  coeff = expand(coeff_compact_new)
  # dens = density(coeff_compact_new, occ)

  # dens_rcpl = jr.pw.density_grid_reciprocal(coeff, crystal.vol, occ)
  dens_rcpl = jnp.fft.fftn(dens, axes=range(-3, 0))
  # print(jnp.sum(dens_rcpl) * crystal.vol / np.prod(g_vec.shape[:3]))
  # print(jnp.sum(dens) * crystal.vol / np.prod(g_vec.shape[:3]))

  # print("sum of eigenvalues:", jnp.sum(eigval_new))
  # print("efun(coeff_compact_new, dens):", efun(coeff_compact_new, dens))
  kin = jr.energy.kinetic(g_vec, kpts, coeff, occ)
  har = jr.energy.hartree(dens_rcpl, g_vec, crystal.vol)
  ext = jr.energy.external(
    dens_rcpl, crystal.positions, crystal.charges, g_vec, crystal.vol
  )
  lda_x = jr.energy.xc_energy(dens, g_vec, crystal.vol, "lda_x")
  total_energy = kin + har + ext + lda_x + ewald

  print(f"Total energy:   \t {total_energy: .6f} Eh")
  print(f"Ewald energy:   \t {ewald: .6f} Eh")
  print(f"Kinetic energy: \t {kin: .6f} Eh")
  print(f"Hartree energy: \t {har: .6f} Eh")
  print(f"External energy: \t {ext: .6f} Eh")
  print(f"XC energy:      \t {lda_x: .6f} Eh")
  print(f"Time taken for JIT: {time_jit: .6f} seconds")
  print(f"Time taken for total: {time_total: .6f} seconds")

  return coeff_compact, etol


if __name__ == "__main__":
  scf(SCF_MAX_ITER, CONVERGENCE)
