import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse


def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep, via matrix-vector multiplication
    Args:
        A: The 2d finite difference matrix, N^2 x N^2. 
        u: N x N grid state at timestep k.
        epsilon: stability constant.

    Returns:
        N x N Grid state at timestep k+1.
    """
    N = u.shape[0]
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u

def get_A(N):
    """
    Build a heat diffusion matrix A for the 2D heat equation.
    Args:
        N, int, number of grid points of one dimension
    Returns:
        A, matrix, heat diffusion matrix of size N^2 x N^2.
    """
    n = N * N
    diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0
    A = (np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) +
        np.diag(diagonals[3], N) + np.diag(diagonals[4], -N))
    
    return A

def get_sparse_A(N):
    """
    Build a sparse format heat diffusion matrix A_sp_matrix
    Args:
        N, int, number of grid points of one dimension
    Returns:
        A_sp_matrix, matrix in sparse format, heat diffusion matrix of size N^2 x N^2.
    """
    A = get_A(N)
    A_sp_matrix = sparse.BCOO.fromdense(A)
    
    return A_sp_matrix
    
def advance_time_numpy(u, epsilon):
    """
    Use vectorized array operations to advance the heat diffusion simulation speed
    Args:
        u: N x N grid state at timestep k.
        epsilon: stability constant.
    Returns:
        next: N x N Grid state at timestep k+1.
    """
    # Initialize a (N+2) x (N+2) array with 0
    array_0 = np.pad(u, pad_width = 1, mode='constant', constant_values=0)
    
    # Compute the next by roll along all directions
    next_time_step = array_0 + epsilon * (
        np.roll(array_0, 1, axis=0) + np.roll(array_0, -1, axis=0) +
        np.roll(array_0, 1, axis=1) + np.roll(array_0, -1, axis=1) - 4 * array_0)
        
    # Convert to N x N grid
    next = next_time_step[1:-1, 1:-1]
    return next

@jax.jit
def advance_time_jax(u, epsilon):
    """
    Advance the heat diffusion simulation speed based on NumPy operations
    Args:
        u: N x N grid state at timestep k.
        epsilon: stability constant.
    Returns:
        next: N x N Grid state at timestep k+1.
    """
    # Initialize a (N+2) x (N+2) array with 0
    array_0 = jnp.pad(u, pad_width = 1, mode='constant', constant_values=0)
    
    # Compute the next by roll along all directions
    next_time_step = array_0 + epsilon * (
        jnp.roll(array_0, 1, axis=0) + jnp.roll(array_0, -1, axis=0) +
        jnp.roll(array_0, 1, axis=1) + jnp.roll(array_0, -1, axis=1) - 4 * array_0)
        
    # Convert to N x N grid
    next = next_time_step[1:-1, 1:-1]
    return next
