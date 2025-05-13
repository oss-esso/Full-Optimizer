import numpy as np

def mean_field_aoa(h, J, p=1000, tau=0.5, tau_decay=0.99, beta_init=0.5, beta_final=20.0, flip_improve=False):
    """
    Mean-Field Approximate Optimization Algorithm for QUBO/Ising problems.

    Args:
        h: 1D array of local fields, shape (N,)
        J: 2D symmetric array of couplings, shape (N, N)
        p: number of Trotter steps
        tau: initial time step size (will decay if tau_decay < 1)
        tau_decay: decay rate for tau at each step (0.99 means tau decreases by 1% per step)
        beta_init: initial inverse temperature
        beta_final: final inverse temperature
        flip_improve: whether to perform two-spin flip improvement

    Returns:
        sigma: solution spin vector (+1/-1) length N
        cost: final cost value
    """
    N = len(h)
    # Ensure p is an integer
    p = int(p)
    
    # initialize spin vectors ni = (nx, ny, nz) = (1,0,0)
    n = np.zeros((N, 3))
    n[:, 0] = 1.0

    # Create a dynamic schedule for gamma_k, beta_k with temperature annealing
    ks = np.arange(1, p+1)
    
    # Calculate beta schedule (inverse temperature) from beta_init to beta_final
    beta_schedule = np.linspace(beta_init, beta_final, p)
    
    # Calculate adaptive gamma and beta values
    gamma = np.zeros(p)
    beta = np.zeros(p)
    
    # Initial values
    current_tau = tau
    for k in range(p):
        # Update tau with decay
        if k > 0:
            current_tau *= tau_decay
        
        # Calculate gamma and beta for this step
        gamma[k] = current_tau * (k + 1) / p
        
        # Use temperature schedule for beta calculation
        beta[k] = current_tau * beta_schedule[k] / beta_final

    # time evolution
    for k in range(p):
        # compute magnetizations m_i = h_i + sum_j J_ij n_z_j
        m = h + J.dot(n[:, 2])
        # apply V_P: rotation around z-axis by angle theta_i = 2*m_i*gamma_k
        theta = 2 * m * gamma[k]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # rotation in x-y plane
        x, y, z = n[:, 0], n[:, 1], n[:, 2]
        n[:, 0] = x * cos_t + y * sin_t
        n[:, 1] = -x * sin_t + y * cos_t
        # z unchanged

        # apply V_D: rotation around x-axis by angle phi_i = 2*Delta_i*beta_k, Delta_i=1
        phi = 2 * beta[k]
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        # rotation in y-z plane
        y, z = n[:, 1], n[:, 2]
        n[:, 1] = y * cos_p + z * sin_p
        n[:, 2] = -y * sin_p + z * cos_p
        # x unchanged

    # compute final bitstring
    sigma = np.sign(n[:, 2])
    sigma[sigma == 0] = 1
    cost = - (h + 0.5 * J.dot(sigma)).dot(sigma)

    if flip_improve:
        # two-spin flip improvement
        best_cost = cost
        best_sigma = sigma.copy()
        
        # First do single bit flips for fast improvement
        improved = True
        while improved:
            improved = False
            for i in range(N):
                s_flip = best_sigma.copy()
                s_flip[i] *= -1
                c = - (h + 0.5 * J.dot(s_flip)).dot(s_flip)
                if c < best_cost:
                    best_cost = c
                    best_sigma = s_flip.copy()
                    improved = True
                    
        # Then try two-spin flips for further improvement
        for i in range(N-1):
            for j in range(i+1, N):
                s_flip = best_sigma.copy()
                s_flip[i] *= -1
                s_flip[j] *= -1
                c = - (h + 0.5 * J.dot(s_flip)).dot(s_flip)
                if c < best_cost:
                    best_cost = c
                    best_sigma = s_flip.copy()
        
        sigma = best_sigma
        cost = best_cost

    return sigma.astype(int), cost


def qubo_to_ising(Q):
    """
    Convert QUBO matrix Q to Ising coefficients h, J (zero-field reduction).
    QUBO: minimize x^T Q x, x in {0,1}^N
    Equivalent Ising: sigma_i = 2*x_i - 1
    Returns h, J for H = -h^T sigma - sum_{i<j} J_ij sigma_i sigma_j
    """
    N = Q.shape[0]
    # Expand quadratic form
    # x = (sigma+1)/2
    # x^T Q x = 1/4 sigma^T Q sigma + 1/2 sum_i (Q.sum(axis=1)) sigma + const
    # So J_ij = -Q_ij/4, h_i = -sum_j Q_ij/2 - Q_ii/4
    Q = np.array(Q)
    h = -0.5 * Q.sum(axis=1)
    # adjust diagonal
    h -= 0.25 * np.diag(Q)
    J = -0.25 * (Q + Q.T)
    np.fill_diagonal(J, 0.0)
    return h, J


if __name__ == '__main__':
    # Example: random SK instance
    N = 50
    np.random.seed(0)
    J_mat = np.random.randn(N, N)
    J_mat = (J_mat + J_mat.T) / 2
    np.fill_diagonal(J_mat, 0)
    h_vec = np.zeros(N)
    
    # Test with default parameters
    sigma1, cost1 = mean_field_aoa(h_vec, J_mat, p=1000, tau=0.5, flip_improve=False)
    print("Standard solution cost:", cost1)
    
    # Test with enhanced parameters
    sigma2, cost2 = mean_field_aoa(h_vec, J_mat, p=2000, tau=0.99, tau_decay=0.995, 
                                 beta_init=0.1, beta_final=30.0, flip_improve=True)
    print("Enhanced solution cost:", cost2)
    print("Improvement: {:.2f}%".format(100 * (cost1 - cost2) / abs(cost1)))
