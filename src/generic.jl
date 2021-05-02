
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export basis, grade, order, options, metric, polymode, dyadmode, diffmode, diffvars
export valuetype, value, hasinf, hasorigin, isorigin, norm, indices, tangent, isbasis, ≅

(M::Signature)(b::Int...) = SubManifold{M}(b)
(M::DiagonalForm)(b::Int...) = SubManifold{M}(b)
(M::SubManifold)(b::Int...) = SubManifold{supermanifold(M)}(b)
(M::Signature)(b::T) where T<:AbstractVector{Int} = SubManifold{M}(b)
(M::DiagonalForm)(b::T) where T<:AbstractVector{Int} = SubManifold{M}(b)
(M::SubManifold)(b::T) where T<:AbstractVector{Int} = SubManifold{supermanifold(M)}(b)
(M::Signature)(b::T) where T<:AbstractRange{Int} = SubManifold{M}(b)
(M::DiagonalForm)(b::T) where T<:AbstractRange{Int} = SubManifold{M}(b)
(M::SubManifold)(b::T) where T<:AbstractRange{Int} = SubManifold{supermanifold(M)}(b)

@pure _polymode(M::Int) = iszero(M&16)
@pure _dyadmode(M::Int) = M%16 ∈ 8:11 ? -1 : Int(M%16 ∈ (4,5,6,7))
@pure _hasinf(M::Int) = M%16 ∈ (1,3,5,7,9,11)
@pure _hasorigin(M::Int) = M%16 ∈ (2,3,6,7,10,11)

#@pure Base.ndims(S::SubManifold{M,G}) where {G,M} = isbasis(S) ? mdims(M) : G
@pure AbstractTensors.mdims(S::SubManifold{M,G}) where {G,M} = isbasis(S) ? mdims(M) : G
@pure order(m::SubManifold{V,G,B} where G) where {V,B} = order(V)>0 ? count_ones(symmetricmask(V,B,B)[4]) : 0
@pure order(m::Simplex) = order(basis(m))+order(value(m))
@pure options(::T) where T<:TensorBundle{N,M} where N where M = M
@pure options_list(V::M) where M<:Manifold = hasinf(V),hasorigin(V),dyadmode(V),polymode(V)
@pure metric(::T) where T<:TensorBundle{N,M,S} where {N,M} where S = S
@pure metric(V::Signature,b::UInt) = isodd(count_ones(metric(V)&b)) ? -1 : 1
@pure polymode(::T) where T<:TensorBundle{N,M} where N where M = _polymode(M)
@pure dyadmode(::T) where T<:TensorBundle{N,M} where N where M = _dyadmode(M)
@pure diffmode(::T) where T<:TensorBundle{N,M,S,F,D} where {N,M,S,F} where D = D
@pure diffvars(::T) where T<:TensorBundle{N,M,S,F} where {N,M,S} where F = F
@pure function diffvars(::SubManifold{M,N,S} where N) where {M,S}
    n,C = mdims(M),diffmode(M)
    sum(in.(1+n-(C<0 ? 2 : 1)*diffvars(M):n,Ref(indices(S))))
end
for mode ∈ (:options,:metric,:polymode,:dyadmode,:diffmode)
    @eval @pure $mode(::SubManifold{M}) where M = $mode(M)
end

export isdyadic, isdual, istangent

@inline value(x::M,T=Int) where M<:TensorBundle = T==Any ? 1 : one(T)
@inline value(::SubManifold,T=Int) = T==Any ? 1 : one(T)
@inline value(m::Simplex,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v

for T ∈ (:T,:(Type{T}))
    @eval begin
        @pure valuetype(::$T) where T<:SubManifold = Int
        @pure valuetype(::$T) where T<:Simplex{V,G,B,𝕂} where {V,G,B} where 𝕂 = 𝕂
        @pure isbasis(::$T) where T<:SubManifold{V} where V = typeof(V)<:SubManifold
        @pure isbasis(::$T) where T<:TensorBundle = false
        @pure isbasis(::$T) where T<:Simplex = false
        @pure basis(m::$T) where T<:SubManifold = isbasis(m) ? m : SubManifold(m)
        @pure basis(m::$T) where T<:Simplex{V,G,B} where {V,G} where B = B
        @pure UInt(b::$T) where T<:SubManifold{V,G,B} where {V,G} where B = B::UInt
        @pure UInt(b::$T) where T<:Simplex = UInt(basis(b))
    end
end
@pure det(s::Signature) = isodd(count_ones(metric(s))) ? -1 : 1
@pure det(s::DiagonalForm) = PROD(diagonalform(s))
@pure Base.abs(s::SubManifold) = isbasis(s) ? Base.sqrt(Base.abs2(s)) : sqrt(abs(det(s)))
@pure Base.abs(s::T) where T<:TensorBundle = sqrt(abs(det(s)))
@pure supermanifold(m::T) where T<:TensorBundle = m
@pure supermanifold(::SubManifold{M}) where M = M

@pure volume(t::SubManifold{V,G}) where {V,G} = G == mdims(V) ? t : zero(V)
@pure isvolume(t::SubManifold) = rank(t) == mdims(V)
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

@pure hasinf(::T) where T<:TensorBundle{N,M} where N where M = _hasinf(M)
@pure hasinf(::SubManifold{M,N,S} where N) where {M,S} = hasinf(M) && isodd(S)
@pure hasinf(t::Simplex) = hasinf(basis(t))
@pure hasorigin(::T) where T<:TensorBundle{N,M} where N where M = _hasorigin(M)
@pure hasorigin(V::SubManifold{M,N,S} where N) where {M,S} = hasorigin(M) && (hasinf(M) ? (d=UInt(2);(d&S)==d) : isodd(S))
@pure hasorigin(t::Simplex) = hasorigin(basis(t))
@pure Base.isinf(e::SubManifold{V}) where V = hasinf(e) && count_ones(UInt(e)) == 1
@pure isorigin(e::SubManifold{V}) where V = hasorigin(V) && count_ones(UInt(e))==1 && e[hasinf(V)+1]

symmetricsplit(V,b::SubManifold) = symmetricsplit(V,UInt(b))

## functors

@pure tangent(s::Signature{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = Signature{N+(isdyadic(s) ? 2f : f),M,S,f,D+d}()
@pure tangent(s::DiagonalForm{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = DiagonalForm{N+(isdyadic(s) ? 2f : f),M,S,f,D+d}()

@pure subtangent(V) = V(grade(V)+1:mdims(V)...)

for M ∈ (:Signature,:DiagonalForm)
    @eval @pure loworder(V::$M{N,M,S,D,O}) where {N,M,S,D,O} = O≠0 ? $M{N,M,S,D,O-1}() : V
end
@pure loworder(::SubManifold{M,N,S}) where {N,M,S} = SubManifold{loworder(M),N,S}()
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

@pure function Base.adjoint(V::SubManifold{M,N,S}) where {N,M,S}
    C = dyadmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    SubManifold{typeof(M)<:Int ? Signature(M)' : M',N,S}()
end

## reverse

import Base: reverse, ~
import AbstractTensors: involute, clifford
export involute, clifford

@pure grade_basis(v,::SubManifold{V,G,B} where G) where {V,B} = grade_basis(V,B)
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
Base.rationalize(t::Type,a::SubManifold{V,G},tol::Real=eps(T)) where {V,G} = SubManifold{V,G}(rationalize(t,value(a),tol))
Base.rationalize(t::Type,b::Simplex{V,G,B,T};tol::Real=eps(T)) where {V,G,B,T} = Simplex{V,G,B}(rationalize(t,value(b),tol))

# random samplers

orand(T=Float64) = 2(rand(T).-0.5)
import Random: SamplerType, AbstractRNG
Base.rand(::AbstractRNG,::SamplerType{Manifold}) where V = SubManifold(Manifold(rand(1:5)))
Base.rand(::AbstractRNG,::SamplerType{SubManifold}) = rand(SubManifold{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{SubManifold{V}}) where V = SubManifold{V}(UInt(rand(0:1<<mdims(V)-1)))
Base.rand(::AbstractRNG,::SamplerType{SubManifold{V,G}}) where {V,G} = Λ(V).b[rand(binomsum(ndims(V),G)+1:binomsum(mdims(V),G+1))]
Base.rand(::AbstractRNG,::SamplerType{Simplex}) = rand(Simplex{rand(Manifold)})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V}}) where V = orand()*rand(SubManifold{V})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G}}) where {V,G} = orand()*rand(SubManifold{V,G})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B}}) where {V,G,B} = orand()*B
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B,T}}) where {V,G,B,T} = rand(T)*B
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B,T} where B}) where {V,G,T} = rand(T)*rand(SubManifold{V,G})
Base.rand(::AbstractRNG,::SamplerType{Simplex{V,G,B,T} where {G,B}}) where {V,T} = rand(T)*rand(SubManifold{V})
