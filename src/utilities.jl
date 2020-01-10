
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

Bits = UInt

bit2int(b::BitArray{1}) = parse(UInt,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0,C=0) = (1<<(d-1))|(1<<(2*o-1))|(c<0 ? 8 : (1<<(3*c-1)))|(1<<(5*C-1))

const vio = ('∞','∅')

signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-
-(x) = Base.:-(x)
-(x::Symbol) = :(-$x)
-(x::SArray) = Base.:-(x)
-(x::SArray{Tuple{M},T,1,M} where M) where T<:Any = broadcast(-,x)

for op ∈ (:conj,:inv,:sqrt,:abs,:exp,:expm1,:log,:log1p,:sin,:cos,:sinh,:cosh,:signbit)
    @eval @inline $op(z) = Base.$op(z)
end

for op ∈ (:/,:-,:^,:≈)
    @eval @inline $op(a,b) = Base.$op(a,b)
end

for (OP,op) ∈ ((:∏,:*),(:∑,:+))
    @eval begin
        @inline $OP(x...) = Base.$op(x...)
        @inline $OP(x::AbstractVector{T}) where T<:Any = $op(x...)
    end
end

const PROD,SUM,SUB = ∏,∑,-

@inline norm(z::Expr) = abs(z)
@inline norm(z::Symbol) = z
@inline norm(z::SArray{Tuple{M},Any,1,M} where M) = sqrt(SUM(z.^2...))

for T ∈ (Expr,Symbol)
    @eval begin
        ≈(a::$T,b::$T) = a == b
        ≈(a::$T,b) = false
        ≈(a,b::$T) = false
    end
end

# SubManifold

const cache_limit = 12

const lowerbits_cache = Vector{Vector{UInt}}[]
const lowerbits_extra = Dict{UInt,Dict{UInt,UInt}}[]
@pure lowerbits_calc(N,S,B,k=indices(S,N)) = bit2int(indexbits(N,findall(x->x∈k,indices(B,N))))
@pure function lowerbits(N,S,B)
    if N>cache_limit
        n = N-cache_limit
        for k ∈ length(lowerbits_extra)+1:n
            push!(lowerbits_extra,Dict{UInt,Dict{UInt,UInt}}())
        end
        @inbounds !haskey(lowerbits_extra[n],S) && push!(lowerbits_extra[n],S=>Dict{UInt,UInt}())
        @inbounds !haskey(lowerbits_extra[n][S],B) && push!(lowerbits_extra[n][S],B=>lowerbits_calc(N,S,B))
        @inbounds lowerbits_extra[n][S][B]
    else
        for k ∈ length(lowerbits_cache)+1:min(N,cache_limit)
            push!(lowerbits_cache,Vector{Int}[])
        end
        for s ∈ length(lowerbits_cache[N])+1:S
            k = indices(S,N)
            push!(lowerbits_cache[N],[lowerbits_calc(N,s,d,k) for d ∈ UInt(0):UInt(1)<<(N+1)-1])
        end
        @inbounds lowerbits_cache[N][S][B+1]
    end
end

const expandbits_cache = Dict{UInt,Dict{UInt,UInt}}[]
@pure expandbits_calc(N,S,B) = bit2int(indexbits(N,indices(S,N)[indices(B,N)]))
@pure function expandbits(N,S,B)
    for k ∈ length(expandbits_cache)+1:N
        push!(expandbits_cache,Dict{UInt,Dict{UInt,UInt}}())
    end
    @inbounds !haskey(expandbits_cache[N],S) && push!(expandbits_cache[N],S=>Dict{UInt,UInt}())
    @inbounds !haskey(expandbits_cache[N][S],B) && push!(expandbits_cache[N][S],B=>expandbits_calc(N,S,B))
    @inbounds expandbits_cache[N][S][B]
end

#=const expandbits_cache = Vector{Vector{UInt}}[]
const expandbits_extra = Dict{UInt,Dict{UInt,UInt}}[]
@pure expandbits_calc(N,S,B,k=indices(S,N)) = bit2int(indexbits(N,k[indices(B,N)]))
@pure function expandbits(N,S,B)
    if N>cache_limit
        n = N-cache_limit
        for k ∈ length(expandbits_extra)+1:n
            push!(expandbits_extra,Dict{UInt,Dict{UInt,UInt}}())
        end
        @inbounds !haskey(expandbits_extra[n],S) && push!(expandbits_extra[n],S=>Dict{UInt,UInt}())
        @inbounds !haskey(expandbits_extra[n][S],B) && push!(expandbits_extra[n][S],B=>expandbits_calc(N,S,B))
        @inbounds expandbits_extra[n][S][B]
    else
        for k ∈ length(expandbits_cache)+1:min(N,cache_limit)
            push!(expandbits_cache,Vector{Int}[])
        end
        for s ∈ length(expandbits_cache[N])+1:S
            k = indices(S,N)
            push!(expandbits_cache[N],[expandbits_calc(N,s,d,k) for d ∈ UInt(0):UInt(1)<<(N+1)-1])
        end
        @inbounds expandbits_cache[N][S][B+1]
    end
end=#


