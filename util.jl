
import LinearAlgebra: I
import ForneyLab: unsafeMean, unsafePrecision, unsafeCov, eye
import StatsFuns: log2π

function eye(dim)
    return Matrix{Float64}(I,dim,dim)
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

