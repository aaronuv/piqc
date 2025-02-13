import attr
import jax.numpy as jnp

# this can be a container for the model, but it if has a method, then is a class
class Model:
    # H0, Hc or smth
    def propagate(self):
        ...

# then you call psi_T = nmr.propagate (psi0, ...)

# def u = qdc_update(u)

# ψ = ψ_update(ψ, dΛ)








@attr.s
class QDC:
    # attrs

    # methods

    def solve(self):
        ψ = self.solve_dynamics(u, ψ0,...) #TODO: define, shape=(n_traj, N)
        # also psi as output, can be a tensor of many legs, but I do not see the purpose
        # for the IS loop

        U = dt*jnp.sum(u**2)
        Su=0.5*R * U
        S_girsanov = R * jnp.einsum(f'{A}t, j{A}t -> j', u, dW) #TODO: define A
        F = jnp.einsum('jμ, μν, jν -> j', jnp.conj(ψ), Qm, ψ) #TODO: define Qm
        S = -0.5* F + Su + S_girsanov
    
        F_all[p] = jnp.mean(F)
        F_min_all[p] = jnp.min(F)

        print(f'F: {F_all[p]}, std: {jnp.sqrt(jnp.std(F))}')
    
        Smin = jnp.min(S)
        S = S - Smin
        w = jnp.exp(-S/λ) #TODO: define λ

        J[p] = Smin - λ*jnp.log(jnp.mean(w))
        w = w/jnp.sum(w)
        ess[p] = 1/jnp.sum(w**2)/n_traj

        # K equal time intervals
        # NT is divisible by K
        N2 = jnp.floor(NT/K)
        for k in range(1, K+1):
            iv = jnp.arange((k-1)*N2, k*N2)
            Δu = 1/dt/N2*jnp.tensordot(w, jnp.sum(dW[..., iv], axis=-1), axes=1)
            u[..., iv] = u[..., iv] + jnp.reshape(Δu, (-1, 1))
        
        u_all[p] = u
        win = min(j-jw, is_window)
        u_av = 1/win * jnp.sum(u_all[j-win+1:j,...], axis=1)