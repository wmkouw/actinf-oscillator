
using LinearAlgebra
using Optim
import ForneyLab: unsafeMean, unsafePrecision
# using Zygote: gradient, Params
# using Flux: update!, Descent, ADAM

using StatsFuns: @irrational
@irrational log2π  1.8378770664093454836 log(big(2.)*π)

function kl(μ_p, μ_q, σ_p, σ_q)
    "KL of 2 univariate Normals"
    log(σ_q / σ_p) + (σ_p + (μ_p - μ_q)^2)/ (2*σ_q) - 0.5
end

function umean(d::ProbabilityDistribution{Univariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafeMean(d)
end

function umean(d::ProbabilityDistribution{Multivariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafeMean(d)
end

function uprec(d::ProbabilityDistribution{Univariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafePrecision(d)
end

function uprec(d::ProbabilityDistribution{Multivariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafePrecision(d)
end

function minEFE(current_state, 
                goal_state,
                model_params,
                transition_function; 
                order=2, 
                opt=ADAM(0.01,(0.9, 0.8)), 
                num_iters=100, 
                plan_horizon=1)

    # Objective function
    G(π) = EFE(π, current_state, goal_state, model_params, transition_function, order=order, time_horizon=plan_horizon)

    # Minimize
    optimize(G, rand(plan_horizon,), Optim.Options(iterations=num_iters), LBFGS(); autodiff=:forward)

    return policy
end

function EFE(policy, current_state, goal_state, model_params, transition; order=2, time_horizon=1)
    "Compute Expected Free Energy"

    # Unpack goal state
    μ_star, σ_star = goal_state

    # Unpack model parameters
    θ, η, ζ, ξ = model_params

    # Process noise
    Σ_z = ζ *[1.  0.; 0.  0.]

    # Helper matrices
    S = [0. 0.; 1. 0.]
    s = [1., 0.]

    # Start previous state var        
    μ_kmin = current_state[1]
    Σ_kmin = current_state[2]

    loss = 0.
    for k in 1:time_horizon

        # State transition
        μ_k = S*μ_kmin + s*g(θ, μ_kmin) + s*η*policy[k]
        Σ = Σ_kmin + Σ_z

        # calculate the covariance matrix
        H = collect(s')
        Σ_11 = Σ
        Σ_21 = H*Σ
        Σ_12 = Σ*H'
        Σ_22 = H*Σ*H' + inv(mat(ξ))

        # Calculate conditional entropy
        Σ_cond = Σ_22 - Σ_21 * inv(Σ_11) * Σ_12
        ambiguity = 0.5(log2π + log(Σ_cond[1]) + 1)

        # Calculate marginal mean of observations
        y_hat = s'*μ_k

        # Risk as KL between marginal and goal prior
        risk = kl(μ_star, y_hat, σ_star, Σ_22[1])
        
        # Update loss.
        loss += risk + ambiguity

        # Update previous state var
        μ_kmin = μ_k
        Σ_kmin = Σ

    end
    return loss
end

# Gets the first element and sets everything else to 0
function uvector(dim, pos=1)
    u = zeros(dim)
    u[pos] = 1
    return dim == 1 ? u[pos] : u
end

# Creates the shift matrix with ones on the lower diagonal
function shift(dim)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end

function eye(dim)
    return Matrix{Float64}(I,dim,dim)
end

