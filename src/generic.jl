
#   This file is part of DirectSum.jl
#   It is licensed under the AGPL license
#   DirectSum Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export basis, grade, order, options, metric, polymode, dyadmode, diffmode, diffvars
export valuetype, value, hasinf, hasorigin, isorigin, norm, indices, tangent, isbasis, ≅
export antigrade, antireverse, antiinvolute, anticlifford

(M::Signature)(b::Int...) = Submanifold{M}(b)
(M::DiagonalForm)(b::Int...) = Submanifold{M}(b)
(M::Submanifold)(b::Int...) = Submanifold{supermanifold(M)}(b)
(M::Signature)(b::T) where T<:AbstractVector{Int} = Submanifold{M}(b)
(M::DiagonalForm)(b::T) where T<:AbstractVector{Int} = Submanifold{M}(b)
(M::Submanifold)(b::T) where T<:AbstractVector{Int} = Submanifold{supermanifold(M)}(b)
(M::Signature)(b::T) where T<:AbstractRange{Int} = Submanifold{M}(b)
(M::DiagonalForm)(b::T) where T<:AbstractRange{Int} = Submanifold{M}(b)
(M::Submanifold)(b::T) where T<:AbstractRange{Int} = Submanifold{supermanifold(M)}(b)

@pure _polymode(M::Int) = iszero(M&16)
@pure _dyadmode(M::Int) = M%16 ∈ 8:11 ? -1 : Int(M%16 ∈ (4,5,6,7))
@pure _hasinf(M::Int) = M%16 ∈ (1,3,5,7,9,11)
@pure _hasorigin(M::Int) = M%16 ∈ (2,3,6,7,10,11)

#@pure Base.ndims(S::Submanifold{M,G}) where {G,M} = isbasis(S) ? mdims(M) : G
@pure AbstractTensors.mdims(S::Submanifold{M,G}) where {G,M} = isbasis(S) ? mdims(M) : G
@pure order(m::Submanifold{V,G,B} where G) where {V,B} = order(V)>0 ? count_ones(symmetricmask(V,B,B)[4]) : 0
@pure order(m::Single) = order(basis(m))+order(value(m))
@pure options(::T) where T<:TensorBundle{N,M} where N where M = M
@pure options_list(V::M) where M<:Manifold = hasinf(V),hasorigin(V),dyadmode(V),polymode(V)
@pure metric(::T) where T<:TensorBundle{N,M,S} where {N,M} where S = S
@pure metric(V::Signature,b::UInt) = isodd(count_ones(metric(V)&b)) ? -1 : 1
@pure polymode(::T) where T<:TensorBundle{N,M} where N where M = _polymode(M)
@pure dyadmode(::T) where T<:TensorBundle{N,M} where N where M = _dyadmode(M)
@pure diffmode(::T) where T<:TensorBundle{N,M,S,F,D} where {N,M,S,F} where D = D
@pure diffvars(::T) where T<:TensorBundle{N,M,S,F} where {N,M,S} where F = F
@pure function diffvars(::Submanifold{M,N,S} where N) where {M,S}
    n,C = mdims(M),diffmode(M)
    sum(in.(1+n-(C<0 ? 2 : 1)*diffvars(M):n,Ref(indices(S))))
end
for mode ∈ (:options,:polymode,:dyadmode,:diffmode)
    @eval @pure $mode(::Submanifold{M}) where M = $mode(M)
end

export isdyadic, isdual, istangent

@inline value(x::M,T=Int) where M<:TensorBundle = T==Any ? 1 : one(T)
@inline value(::Submanifold,T=Int) = T==Any ? 1 : one(T)
@inline value(m::Single,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v

@pure basis(m::Zero{V}) where V = getbasis(V,UInt(0))
@pure basis(m::T) where T<:Submanifold = isbasis(m) ? m : Submanifold(m)
@pure basis(m::Type{T}) where T<:Submanifold = isbasis(m) ? m() : typeof(Submanifold(m()))
for T ∈ (:T,:(Type{T}))
    @eval begin
        @pure valuetype(::$T) where T<:Submanifold = Int
        @pure valuetype(::$T) where T<:Single{V,G,B,𝕂} where {V,G,B} where 𝕂 = 𝕂
        @pure isbasis(::$T) where T<:Submanifold{V} where V = issubmanifold(V)
        @pure isbasis(::$T) where T<:TensorBundle = false
        @pure isbasis(::$T) where T<:Single = false
        @pure basis(m::$T) where T<:Single{V,G,B} where {V,G} where B = B
        @pure UInt(b::$T) where T<:Submanifold{V,G,B} where {V,G} where B = B::UInt
        @pure UInt(b::$T) where T<:Single = UInt(basis(b))
    end
end
@pure det(s::Signature) = isodd(count_ones(metric(s))) ? -1 : 1
@pure det(s::DiagonalForm) = PROD(diagonalform(s))
@pure Base.abs(s::Submanifold) = isbasis(s) ? Base.sqrt(Base.abs2(s)) : sqrt(abs(det(s)))
@pure Base.abs(s::T) where T<:TensorBundle = sqrt(abs(det(s)))
@pure supermanifold(m::T) where T<:TensorBundle = m
@pure supermanifold(::Submanifold{M}) where M = M

@pure volume(t::Submanifold{V,G}) where {V,G} = G == mdims(V) ? t : Zero(V)
@pure isvolume(t::Submanifold) = rank(t) == mdims(V)
for (part,G) ∈ ((:scalar,0),(:vector,1),(:bivector,2),(:trivector,3))
    ispart = Symbol(:is,part)
    @eval begin
        @pure $part(t::Submanifold{V,$G} where V) = t
        @pure $part(t::Submanifold{V}) where V = Zero(V)
        @pure $ispart(t::Submanifold) = rank(t) == $G
    end
end
for T ∈ (Expr,Symbol)
    @eval @inline Base.iszero(t::Single{V,G,B,$T} where {V,G,B}) = false
end

@pure val(G::Int) = Val{G}()
grade(t,G::Int) = grade(t,val(G))
grade(t::TensorGraded{V,G},g::Val{G}) where {V,G} = t
grade(t::TensorGraded{V,L},g::Val{G}) where {V,G,L} = Zero(V)
antigrade(t,G::Int) = antigrade(t,val(G))
antigrade(t::TensorAlgebra{V},::Val{G}) where {V,G} = grade(t,val(grade(V)-G))

@pure hasinf(::T) where T<:TensorBundle{N,M} where N where M = _hasinf(M)
@pure hasinf(::Submanifold{M,N,S} where N) where {M,S} = hasinf(M) && isodd(S)
@pure hasinf(t::Single) = hasinf(basis(t))
@pure hasorigin(::T) where T<:TensorBundle{N,M} where N where M = _hasorigin(M)
@pure hasorigin(V::Submanifold{M,N,S} where N) where {M,S} = hasorigin(M) && (hasinf(M) ? (d=UInt(2);(d&S)==d) : isodd(S))
@pure hasorigin(t::Single) = hasorigin(basis(t))
@pure Base.isinf(e::Submanifold{V}) where V = hasinf(e) && count_ones(UInt(e)) == 1
@pure isorigin(e::Submanifold{V}) where V = hasorigin(V) && count_ones(UInt(e))==1 && e[hasinf(V)+1]

symmetricsplit(V,b::Submanifold) = symmetricsplit(V,UInt(b))

## functors

@pure tangent(s::Signature{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = Signature{N+(isdyadic(s) ? 2f : f),M,S,f,D+d}()
@pure tangent(s::DiagonalForm{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = DiagonalForm{N+(isdyadic(s) ? 2f : f),M,S,f,D+d}()

@pure subtangent(V) = V(grade(V)+1:mdims(V)...)

for M ∈ (:Signature,:DiagonalForm)
    @eval @pure loworder(V::$M{N,M,S,D,O}) where {N,M,S,D,O} = O≠0 ? $M{N,M,S,D,O-1}() : V
end
@pure loworder(::Submanifold{M,N,S}) where {N,M,S} = Submanifold{loworder(M),N,S}()
@pure loworder(::Type{T}) where T = loworder(T())

# dual involution

@pure flip_sig(N,S::UInt) = UInt(2^N-1) & (~S)

@pure dual(V::T) where T<:Manifold = isdyadic(V) ? V : V'
@pure dual(V::T,B,M=Int(rank(V)/2)) where T<:Manifold = ((B<<M)&((1<<rank(V))-1))|(B>>M)

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

@pure function Base.adjoint(V::Submanifold{M,N,S}) where {N,M,S}
    C = dyadmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    Submanifold{typeof(M)<:Int ? Signature(M)' : M',N,S}()
end

## reverse

import Base: reverse, ~
import AbstractTensors: involute, clifford
export involute, clifford

@pure grade_basis(v,::Submanifold{V,G,B} where G) where {V,B} = grade_basis(V,B)
@pure grade(v,::Submanifold{V,G,B} where G) where {V,B} = grade(V,B)
@pure antigrade(v,::Submanifold{V,G,B} where G) where {V,B} = antigrade(V,B)

@doc """
    ~(ω::TensorAlgebra)

Reverse of an element: ~ω = (-1)^(grade(ω)*(grade(ω)-1)/2)*ω
""" Base.conj
#reverse(a::UniformScaling{Bool}) = UniformScaling(!a.λ)
#reverse(a::UniformScaling{T}) where T<:Field = UniformScaling(-a.λ)

"""
    reverse(ω::TensorAlgebra)

Reverse of an element: ~ω = (-1)^(grade(ω)*(grade(ω)-1)/2)*ω
"""
@inline Base.:~(b::TensorAlgebra) = Base.conj(b)
#@inline ~(b::UniformScaling) = reverse(b)

@doc """
    involute(ω::TensorAlgebra)

Involute of an element: ~ω = (-1)^grade(ω)*ω
""" involute

@doc """
    clifford(ω::TensorAlgebra)

Clifford conjugate of an element: clifford(ω) = involute(reverse(ω))
""" clifford

"""
    antireverse(ω::TensorAlgebra)

Anti-reverse of an element: ~ω = (-1)^(antigrade(ω)*(antigrade(ω)-1)/2)*ω
""" antireverse

@doc """
    antiinvolute(ω::TensorAlgebra)

Anti-involute of an element: ~ω = (-1)^antigrade(ω)*ω
""" antiinvolute

@doc """
    anticlifford(ω::TensorAlgebra)

Anti-clifford conjugate of an element: anticlifford(ω) = antiinvolute(antireverse(ω))
""" anticlifford

for r ∈ (:reverse,:involute,:(Base.conj),:clifford)
    p = Symbol(:parity,r==:(Base.conj) ? :conj : r)
    ar = Symbol(:anti,r)
    @eval begin
        @pure function $r(b::Submanifold{V,G,B}) where {V,G,B}
            $p(grade(V,B)) ? Single{V}(-value(b),b) : b
        end
        $r(b::Single) = value(b) ≠ 0 ? Single(value(b),$r(basis(b))) : Zero(Manifold(b))
        @pure function $ar(b::Submanifold{V,G,B}) where {V,G,B}
            $p(antigrade(V,B)) ? Single{V}(-value(b),b) : b
        end
        $ar(b::Single) = value(b) ≠ 0 ? Single(value(b),$ar(basis(b))) : Zero(Manifold(b))
    end
end

for op ∈ (:div,:rem,:mod,:mod1,:fld,:fld1,:cld,:ldexp)
    @eval begin
        Base.$op(a::Submanifold{V,G},m) where {V,G} = Submanifold{V,G}($op(value(a),m))
        Base.$op(b::Single{V,G,B,T},m) where {V,G,B,T} = Single{V,G,B}($op(value(b),m))
    end
end
for op ∈ (:mod2pi,:rem2pi,:rad2deg,:deg2rad,:round)
    @eval begin
        Base.$op(a::Submanifold{V,G}) where {V,G} = Submanifold{V,G}($op(value(a)))
        Base.$op(b::Single{V,G,B,T}) where {V,G,B,T} = Single{V,G,B}($op(value(b)))
    end
end
Base.rationalize(t::Type,a::Submanifold{V,G},tol::Real=eps(T)) where {V,G} = Submanifold{V,G}(rationalize(t,value(a),tol))
Base.rationalize(t::Type,b::Single{V,G,B,T};tol::Real=eps(T)) where {V,G,B,T} = Single{V,G,B}(rationalize(t,value(b),tol))

# random samplers

orand(T=Float64) = 2(rand(T).-0.5)
import Random: SamplerType, AbstractRNG
Base.rand(::AbstractRNG,::SamplerType{Manifold}) = Submanifold(Manifold(rand(1:5)))
Base.rand(::AbstractRNG,::SamplerType{Submanifold}) = rand(Submanifold{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Submanifold{V}}) where V = Submanifold{V}(UInt(rand(0:1<<mdims(V)-1)))
Base.rand(::AbstractRNG,::SamplerType{Submanifold{V,G}}) where {V,G} = Λ(V).b[rand(binomsum(ndims(V),G)+1:binomsum(mdims(V),G+1))]
Base.rand(::AbstractRNG,::SamplerType{Single}) = rand(Single{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Single{V}}) where V = orand()*rand(Submanifold{V})
Base.rand(::AbstractRNG,::SamplerType{Single{V,G}}) where {V,G} = orand()*rand(Submanifold{V,G})
Base.rand(::AbstractRNG,::SamplerType{Single{V,G,B}}) where {V,G,B} = orand()*B
Base.rand(::AbstractRNG,::SamplerType{Single{V,G,B,T}}) where {V,G,B,T} = rand(T)*B
Base.rand(::AbstractRNG,::SamplerType{Single{V,G,B,T} where B}) where {V,G,T} = rand(T)*rand(Submanifold{V,G})
Base.rand(::AbstractRNG,::SamplerType{Single{V,G,B,T} where {G,B}}) where {V,T} = rand(T)*rand(Submanifold{V})
