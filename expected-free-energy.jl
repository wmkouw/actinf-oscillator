using LinearAlgebra
using Optim
using GFX

import StatsFuns: @irrational
@irrational log2π  1.8378770664093454836 log(big(2.)*π)

include("util.jl")

function minEFE(current_state, 
                goal_state,
                model_params,
                transition_function; 
                num_iters=100, 
                plan_horizon=1)

    # Objective function
    G(π) = EFE(π, current_state, goal_state, model_params, transition_function, time_horizon=plan_horizon)

    # Minimize
    results = optimize(G, rand(plan_horizon,), Optim.Options(iterations=num_iters), LBFGS(); autodiff=:forward)

    return Optim.minimizer(results)
end

function EFE(policy, prior_state, goal_state, model_params, g; time_horizon=1, Δt=1.0)
    "Compute Expected Free Energy"

    # Unpack goal state
    μ_star, σ_star = goal_state

    # Unpack model parameters
    θ, η, ζ, ξ = model_params

    # Process noise
    Σ_z = inv(ζ) *[Δt^3/3   Δt^2/2;
                   Δt^2/2       Δt]

    # Utility matrices
    S = [1. Δt; 0. 1.]
    s = [0., Δt]

    # Start previous state var        
    μ_kmin = prior_state[1]
    Σ_kmin = prior_state[2]

    G = 0.
    for k in 1:time_horizon

        # State transition
        μ_k = S*μ_kmin + s*g(θ, μ_kmin) + s*η*policy[k]
        Σ_k = Σ_kmin + Σ_z

        # calculate the covariance matrix
        H = collect(s')
        Σ_11 = Σ_k
        Σ_21 = H*Σ_k
        Σ_12 = Σ_k*H'
        Σ_22 = H*Σ_k*H' + inv(mat(ξ))

        # Calculate conditional entropy
        Σ_cond = Σ_22 - Σ_21 * inv(Σ_11) * Σ_12
        ambiguity = 0.5(log2π + log(Σ_cond[1]) + 1)

        # Calculate marginal mean of observations
        y_hat = s'*μ_k

        # Risk as KL between marginal and goal prior
        risk = KLDivergence(μ_star, σ_star, y_hat, Σ_22[1])

        # Update loss.
        G += risk + ambiguity

        # Update previous state var
        μ_kmin = μ_k
        Σ_kmin = Σ_k

    end
    return G
end