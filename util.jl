
import LinearAlgebra: I
import ForneyLab: unsafeMean, unsafePrecision, unsafeCov, eye
import StatsFuns: @irrational
@irrational log2π  1.8378770664093454836 log(big(2.)*π)

function eye(dim)
    return Matrix{Float64}(I,dim,dim)
end

function KLDivergence(μ0, σ0, μ1, σ1)
    "KL of 2 univariate Normals"
#     return log(σ_q / σ_p) + (σ_p + (μ_p - μ_q)^2)/ (2*σ_q) - 0.5
    return 0.5*( (σ0/σ1)^2 + (μ1 - μ0)^2/σ1^2 -1 + 2*log(σ1/σ0))
end

function mn(d::ProbabilityDistribution{Univariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafeMean(d)
end

function mn(d::ProbabilityDistribution{Multivariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafeMean(d)
end

function pc(d::ProbabilityDistribution{Univariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafePrecision(d)
end

function pc(d::ProbabilityDistribution{Multivariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafePrecision(d)
end

function cv(d::ProbabilityDistribution{Univariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafeCov(d)
end

function cv(d::ProbabilityDistribution{Multivariate, T}) where T<:Union{Gaussian,Gamma}
    return ForneyLab.unsafeCov(d)
end

