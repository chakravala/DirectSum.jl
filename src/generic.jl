
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export bits, basis, grade, order, options, metric, polymode, dyadmode, diffmode, diffvars
export valuetype, value, hasinf, hasorigin, isorigin, norm, indices, tangent, isbasis, ≅

(M::Signature)(b::Int...) = SubManifold{M}(b)
(M::DiagonalForm)(b::Int...) = SubManifold{M}(b)
(M::SubManifold)(b::Int...) = SubManifold{supermanifold(M)}(b)

@pure Base.ndims(S::SubManifold{M,G}) where {G,M} = isbasis(S) ? ndims(M) : G
@pure grade(V::M) where M<:Manifold{N} where N = N-(isdyadic(V) ? 2 : 1)*diffvars(V)
@pure grade(m::T) where T<:Real = 0
@pure order(m) = 0
@pure order(V::M) where M<:Manifold = diffvars(V)
@pure order(m::SubManifold{V,G,B} where G) where {V,B} = count_ones(symmetricmask(V,B,B)[4])
@pure order(m::Simplex) = order(basis(m))+order(value(m))
@pure options(::T) where T<:TensorBundle{N,M} where N where M = M
@pure options_list(V::M) where M<:Manifold = hasinf(V),hasorigin(V),dyadmode(V),polymode(V)
@pure metric(::T) where T<:TensorBundle{N,M,S} where {N,M} where S = S
@pure metric(V::Signature,b::Bits) = isodd(count_ones(metric(V)&b)) ? -1 : 1
@pure metric(V::M,b::Bits) where M<:Manifold = PROD(V[indices(b)])
@pure polymode(M::Int) = iszero(M&16)
@pure polymode(::T) where T<:TensorBundle{N,M} where N where M = polymode(M)
@pure dyadmode(M::Int) = M%16 ∈ 8:11 ? -1 : Int(M%16 ∈ (4,5,6,7))
@pure dyadmode(::T) where T<:TensorBundle{N,M} where N where M = dyadmode(M)
@pure diffmode(::T) where T<:TensorBundle{N,M,S,F,D} where {N,M,S,F} where D = D
@pure diffvars(::T) where T<:TensorBundle{N,M,S,F} where {N,M,S} where F = F
@pure function diffvars(::SubManifold{M,N,S} where N) where {M,S}
    n,C = ndims(M),diffmode(M)
    sum(in.(1+n-(C<0 ? 2 : 1)*diffvars(M):n,Ref(indices(S))))
end
for mode ∈ (:options,:metric,:polymode,:dyadmode,:diffmode,:diffvars)
    mode≠:metric && @eval @pure $mode(t::T) where T<:TensorAlgebra = $mode(Manifold(t))
    mode≠:diffvars && @eval @pure $mode(::SubManifold{M}) where M = $mode(M)
end

@pure ≅(a,b) = grade(a) == grade(b) && order(a) == order(b) && diffmode(a) == diffmode(b)

export isdyadic, isdual, istangent
const mixedmode = dyadmode
@pure isdyadic(t::T) where T<:TensorAlgebra = dyadmode(Manifold(t))<0
@pure isdual(t::T) where T<:TensorAlgebra = dyadmode(Manifold(t))>0
@pure istangent(t::T) where T<:TensorAlgebra = diffvars(Manifold(t))≠0

@pure valuetype(::SubManifold) = Int
@pure valuetype(::Simplex{V,G,B,T} where {V,G,B}) where T = T
@inline value(x::M,T=Int) where M<:TensorBundle = T==Any ? 1 : one(T)
@inline value(::SubManifold,T=Int) = T==Any ? 1 : one(T)
@inline value(m::Simplex,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value_diff(m::T) where T<:TensorTerm = (v=value(m);istensor(v) ? v : m)
@pure isbasis(::SubManifold{V}) where V = typeof(V)<:SubManifold
@pure isbasis(::T) where T<: TensorBundle = false
@pure isbasis(::Simplex) = false
@pure basis(m::SubManifold) = isbasis(m) ? m : SubManifold(m)
@pure basis(m::Simplex{V,G,B} where {V,G}) where B = B
@pure UInt(m::T) where T<:TensorTerm = bits(basis(m))
@pure bits(m::T) where T<:TensorTerm = bits(basis(m))
@pure bits(b::SubManifold{V,G,B} where {V,G}) where B = B::UInt
@pure bits(::Type{SubManifold{V,G,B}} where {V,G}) where B = B
@pure det(s::Signature) = isodd(count_ones(metric(s))) ? -1 : 1
@pure det(s::DiagonalForm) = PROD(diagonalform(s))
@pure Base.abs(s::SubManifold) = isbasis(s) ? Base.sqrt(Base.abs2(s)) : sqrt(abs(det(s)))
@pure Base.abs(s::T) where T<:TensorBundle = sqrt(abs(det(s)))
@pure supermanifold(m::T) where T<:TensorBundle = m
@pure supermanifold(::SubManifold{M}) where M = M

@pure volume(t::SubManifold{V,G}) where {V,G} = G == ndims(V) ? t : zero(V)
@pure isvolume(t::SubManifold) = rank(t) == ndims(V)
for (part,G) ∈ ((:scalar,0),(:vector,1),(:bivector,2))
    ispart = Symbol(:is,part)
    @eval begin
        @pure $part(t::SubManifold{V,$G} where V) = t
        @pure $part(t::SubManifold{V}) where V = zero(V)
        @pure $ispart(t::SubManifold) = rank(t) == $G
    end
end
for T ∈ (Expr,Symbol)
    @eval @inline Base.iszero(t::Simplex{V,G,B,$T} where {V,G,B}) = false
end

@pure hasconformal(V) = hasinf(V) && hasorigin(V)
@pure hasinf(M::Int) = M%16 ∈ (1,3,5,7,9,11)
@pure hasinf(::T) where T<:TensorBundle{N,M} where N where M = hasinf(M)
@pure hasinf(::SubManifold{M,N,S} where N) where {M,S} = hasinf(M) && isodd(S)
@pure hasinf(t::Simplex) = hasinf(basis(t))
@pure hasinf(t::M) where M<:Manifold = hasinf(Manifold(t))
#@pure hasinf(m::T) where T<:TensorAlgebra = hasinf(Manifold(m))
@pure hasorigin(M::Int) = M%16 ∈ (2,3,6,7,10,11)
@pure hasorigin(::T) where T<:TensorBundle{N,M} where N where M = hasorigin(M)
@pure hasorigin(V::SubManifold{M,N,S} where N) where {M,S} = hasorigin(M) && (hasinf(M) ? (d=UInt(2);(d&S)==d) : isodd(S))
@pure hasorigin(t::Simplex) = hasorigin(basis(t))
@pure hasorigin(t::M) where M<:Manifold = hasorigin(Manifold(t))
#@pure hasorigin(m::T) where T<:TensorAlgebra = hasorigin(Manifold(m))
@pure Base.isinf(e::SubManifold{V}) where V = hasinf(e) && count_ones(bits(e)) == 1
@pure isorigin(e::SubManifold{V}) where V = hasorigin(V) && count_ones(bits(e))==1 && e[hasinf(V)+1]

@pure hasorigin(V::M, B::T) where {M<:Manifold,T<:Bits} = hasinf(V) ? (Bits(2)&B)==Bits(2) : isodd(B)

@pure function hasinf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && (isodd(A) || isodd(B))
end
@pure function hasorigin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && (hasorigin(V,A) || hasorigin(V,B))
end

@pure function hasinf2(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && isodd(A) && isodd(B)
end
@pure function hasorigin2(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && hasorigin(V,A) && hasorigin(V,B)
end

@pure function hasorigininf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && hasorigin(V,A) && isodd(B) && !hasorigin(V,B) && !isodd(A)
end
@pure function hasinforigin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && isodd(A) && hasorigin(V,B) && !isodd(B) && !hasorigin(V,A)
end

@pure function hasinf2origin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf2(V,A,B) && hasorigin(V,A,B)
end
@pure function hasorigin2inf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasorigin2(V,A,B) && hasinf(V,A,B)
end

@pure function diffmask(V::M) where M<:Manifold
    d = diffvars(V)
    if isdyadic(V)
        v = ((one(Bits)<<d)-1)<<(ndims(V)-2d)
        w = ((one(Bits)<<d)-1)<<(ndims(V)-d)
        return d<0 ? (typemax(Bits)-v,typemax(Bits)-w) : (v,w)
    end
    v = ((one(Bits)<<d)-1)<<(ndims(V)-d)
    d<0 ? typemax(Bits)-v : v
end

symmetricsplit(V::M,b::SubManifold) where M<:Manifold = symmetricsplit(V,bits(b))
@pure function symmetricsplit(V::M,a) where M<:Manifold
    sm,dm = symmetricmask(V,a),diffmask(V)
    isdyadic(V) ? (sm&dm[1],sm&dm[2]) : sm
end

@pure function symmetricmask(V::M,a) where M<:Manifold
    d = diffmask(V)
    a&(isdyadic(V) ? |(d...) : d)
end

@pure function symmetricmask(V::M,a,b) where M<:Manifold
    d = diffmask(V)
    D = isdyadic(V) ? |(d...) : d
    aD,bD = (a&D),(b&D)
    return a&~D, b&~D, aD|bD, aD&bD
end

@pure function diffcheck(V::M,A::Bits,B::Bits) where M<:Manifold
    d,db = diffvars(V),diffmask(V)
    v = isdyadic(V) ? db[1]|db[2] : db
    hi = hasinf2(V,A,B) && !hasorigin(V,A,B)
    ho = hasorigin2(V,A,B) && !hasinf(V,A,B)
    (hi || ho) || (d≠0 && count_ones(A&v)+count_ones(B&v)>diffmode(V))
end

## functors

@pure tangent(s::Signature{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = Signature{N+(isdyadic(s) ? 2f : f),M,S,f,D+d}()
@pure tangent(s::DiagonalForm{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = DiagonalForm{N+(isdyadic(s) ? 2f : f),M,S,f,D+d}()

@pure subtangent(V) = V(grade(V)+1:ndims(V)...)

for M ∈ (:Signature,:DiagonalForm)
    @eval @pure loworder(V::$M{N,M,S,D,O}) where {N,M,S,D,O} = O≠0 ? $M{N,M,S,D,O-1}() : V
end
@pure loworder(::SubManifold{M,N,S}) where {N,M,S} = SubManifold{loworder(M),N,S}()

# dual involution

@pure dual(V::T) where T<:Manifold = isdyadic(V) ? V : V'
@pure dual(V::T,B,M=Int(N/2)) where T<:Manifold{N} where N = ((B<<M)&((1<<N)-1))|(B>>M)

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::Signature{N,M,S,F,D}) where {N,M,S,F,D}
    C = dyadmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    Signature{N,doc2m(hasinf(V),hasorigin(V),Int(!Bool(C))),flip_sig(N,S),F,D}()
end

@pure function Base.adjoint(V::DiagonalForm{N,M,S,F,D}) where {N,M,S,F,D}
    C = dyadmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    DiagonalForm{N,doc2m(hasinf(V),hasorigin(V),Int(!Bool(C))),S,F,D}()
end

@pure function Base.adjoint(V::SubManifold{M,N,S}) where {N,M,S}
    C = dyadmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    SubManifold{M',N,S}()
end

## adjoint parities

@pure parityreverse(G) = isodd(Int((G-1)*G/2))
@pure parityinvolute(G) = isodd(G)
@pure parityclifford(G) = parityreverse(G)⊻parityinvolute(G)
const parityconj = parityreverse

## reverse

import Base: reverse, ~
export involute, clifford

@pure grade_basis(V,B) = B&(one(UInt)<<grade(V)-1)
@pure grade_basis(v,::SubManifold{V,G,B} where G) where {V,B} = grade_basis(V,B)
@pure grade(V,B) = count_ones(grade_basis(V,B))
@pure grade(v,::SubManifold{V,G,B} where G) where {V,B} = grade(V,B)

for r ∈ (:reverse,:involute,:(Base.conj),:clifford)
    p = Symbol(:parity,r==:(Base.conj) ? :conj : r)
    @eval begin
        @pure function $r(b::SubManifold{V,G,B}) where {V,G,B}
            $p(grade(V,B)) ? Simplex{V}(-value(b),b) : b
        end
        $r(b::Simplex) = value(b) ≠ 0 ? Simplex(value(b),$r(basis(b))) : g_zero(Manifold(b))
    end
end

@doc """
    ~(ω::TensorAlgebra)

Reverse of a `MultiVector` element: ~ω = (-1)^(grade(ω)*(grade(ω)-1)/2)*ω
""" Base.conj
#reverse(a::UniformScaling{Bool}) = UniformScaling(!a.λ)
#reverse(a::UniformScaling{T}) where T<:Field = UniformScaling(-a.λ)

"""
    reverse(ω::TensorAlgebra)

Reverse of a `MultiVector` element: ~ω = (-1)^(grade(ω)*(grade(ω)-1)/2)*ω
"""
@inline Base.:~(b::TensorAlgebra) = conj(b)
#@inline ~(b::UniformScaling) = reverse(b)

@doc """
    involute(ω::TensorAlgebra)

Involute of a `MultiVector` element: ~ω = (-1)^grade(ω)*ω
""" involute

@doc """
    clifford(ω::TensorAlgebra)

Clifford conjugate of a `MultiVector` element: clifford(ω) = involute(conj(ω))
""" clifford

odd(t::T) where T<:TensorGraded{V,G} where {V,G} = parityinvolute(G) ? t : zero(V)
even(t::T) where T<:TensorGraded{V,G} where {V,G} = parityinvolute(G) ? zero(V) : t

"""
    imag(ω::TensorAlgebra)

The `imag` part `(ω-(~ω))/2` is defined by `abs2(imag(ω)) == -(imag(ω)^2)`.
"""
Base.imag(t::T) where T<:TensorGraded{V,G} where {V,G} = parityreverse(G) ? t : zero(V)

"""
real(ω::TensorAlgebra)

The `real` part `(ω+(~ω))/2` is defined by `abs2(real(ω)) == real(ω)^2`.
"""
Base.real(t::T) where T<:TensorGraded{V,G} where {V,G} = parityreverse(G) ? zero(V) : t

for op ∈ (:div,:rem,:mod,:mod1,:fld,:fld1,:cld,:ldexp)
    @eval begin
        Base.$op(a::SubManifold{V,G},m) where {V,G} = SubManifold{V,G}($op(value(a),m))
        Base.$op(b::Simplex{V,G,B,T},m) where {V,G,B,T} = Simplex{V,G,B}($op(value(b),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad,:round)
    @eval begin
        Base.$op(a::SubManifold{V,G}) where {V,G} = SubManifold{V,G}($op(value(a)))
        Base.$op(b::Simplex{V,G,B,T}) where {V,G,B,T} = Simplex{V,G,B}($op(value(b)))
    end
end
Base.isfinite(b::T) where T<:TensorTerm = isfinite(value(b))
Base.rationalize(t::Type,a::SubManifold{V,G},tol::Real=eps(T)) where {V,G} = SubManifold{V,G}(rationalize(t,value(a),tol))
Base.rationalize(t::Type,b::Simplex{V,G,B,T};tol::Real=eps(T)) where {V,G,B,T} = Simplex{V,G,B}(rationalize(t,value(b),tol))

# comparison (special case for scalars)

Base.isless(a::T,b::S) where {T<:TensorTerm{V,0},S<:TensorTerm{W,0}} where {V,W} = isless(value(a),value(b))
Base.isless(a::T,b) where T<:TensorTerm{V,0} where V = isless(value(a),b)
Base.isless(a,b::T) where T<:TensorTerm{V,0} where V = isless(a,value(b))
Base.:<=(x::T,y::S) where {T<:TensorTerm{V,0},S<:TensorTerm{W,0}} where {V,W} = isless(x,y) | (x == y)
Base.:<=(x::T,y) where T<:TensorTerm{V,0} where V = isless(x,y) | (x == y)
Base.:<=(x,y::T) where T<:TensorTerm{V,0} where V = isless(x,y) | (x == y)

# random samplers

orand(T=Float64) = 2(rand(T).-0.5)
import Random: SamplerType, AbstractRNG
Base.rand(::AbstractRNG,::SamplerType{Manifold}) where V = SubManifold(Manifold(rand(1:5)))
Base.rand(::AbstractRNG,::SamplerType{SubManifold}) = rand(SubManifold{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{SubManifold{V}}) where V = SubManifold{V}(UInt(rand(0:1<<ndims(V)-1)))
Base.rand(::AbstractRNG,::SamplerType{SubManifold{V,G}}) where {V,G} = Λ(V).b[rand(binomsum(ndims(V),G)+1:binomsum(ndims(V),G+1))]
Base.rand(::AbstractRNG,::SamplerType{Simplex}) = rand(Simplex{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V}}) where V = orand()*rand(SubManifold{V})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G}}) where {V,G} = orand()*rand(SubManifold{V,G})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B}}) where {V,G,B} = orand()*B
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B,T}}) where {V,G,B,T} = rand(T)*B
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B,T} where B}) where {V,G,T} = rand(T)*rand(SubManifold{V,G})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B,T} where {G,B}}) where {V,T} = rand(T)*rand(SubManifold{V})
