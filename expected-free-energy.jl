using LinearAlgebra
using Optim
using GFX

import StatsFuns: log2π


function KLDivergence(μ0, σ0, μ1, σ1)
    "KL of 2 univariate Normals"
#     return log(σ_q / σ_p) + (σ_p + (μ_p - μ_q)^2)/ (2*σ_q) - 0.5
    return 0.5*( (σ0/σ1)^2 + (μ0 - μ1)^2/σ1^2 +1 + 2*log(σ1/σ0))
end

function minEFE(current_state, 
                goal_state,
                model_params;
                Δt=1.0,
                num_iters=100, 
                plan_horizon=2)

    # Objective function
    G(π) = EFE(π, current_state, goal_state, model_params, Δt=Δt, time_horizon=plan_horizon)

    # Minimize
    results = optimize(G, rand(plan_horizon,), Optim.Options(iterations=num_iters), LBFGS(); autodiff=:forward)

    return Optim.minimizer(results)
end

function EFE(action, state_k, goal_state, params; Δt = 1., time_horizon=1)
    "Expected Free Energy"
    
    # Unpack goal state
    μ_star = goal_state[1]
    Σ_star = goal_state[2]

    # Unpack model parameters
    θ, η, τ, ξ = params

    # Process noise
    Σ_z = inv(τ)*[Δt^3/3 Δt^2/2; Δt^2/2 Δt]

    # Utility matrices
    S = [1. Δt; 0. 1.]
    s = [0., Δt]
    
    # Model matrices
    A = S + s*θ'
    B = s*η
    C = [1., 0.]'

    # Unpack prior state     
    μ_tmin1 = mn(state_k)
    Σ_tmin1 = cv(state_k)
    
    cumEFE = 0
    for t in 1:time_horizon
        
        # State transition
        μ_t = A*μ_tmin1 + B*action[t]
        Σ_t = A*Σ_tmin1*A' + Σ_z

        # Predicted observation 
        y_t = C*μ_t

        # Block covariance matrix of joint
        Σ_11 = Σ_t
        Σ_21 = C*Σ_t
        Σ_12 = Σ_t*C'
        Σ_22 = C*Σ_t*C' .+ inv(ξ)

        # Calculate conditional entropy
        Σ_cond = Σ_22 - Σ_21*inv(Σ_11)*Σ_12
        ambiguity = 0.5(log2π + log(Σ_cond[1]) - 1)
        
        # Risk as KL between marginal and goal prior
        risk = 0.5*(log(det(Σ_star)/det(Σ_22)) + 1 + (y_t - μ_star)'*inv(Σ_star)*(y_t - μ_star) + tr(inv(Σ_star)*Σ_22))
        
        # Add to cumulative EFE
        cumEFE += risk + ambiguity
        
        # Update previous state
        μ_tmin1 = μ_t
        Σ_tmin1 = Σ_t
        
    end
    return cumEFE
end;