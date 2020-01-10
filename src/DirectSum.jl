module DirectSum

#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorBundle, Signature, DiagonalForm, ℝ, ⊕, value, tangent, Manifold, SubManifold
import Base: getindex, abs, @pure, +, *, ^, ∪, ∩, ⊆, ⊇, ==
import LinearAlgebra
import LinearAlgebra: det
using StaticArrays

## Manifold{N}

import AbstractTensors: Manifold, value, vectorspace, norm, interop, interform

## utilities

include("utilities.jl")

## VectorBundle{N}

"""
    VectorBundle{n,ℙ,g,ν,μ} <: Manifold{n}

Let `n` be the rank of a `Manifold{n}`.
The type `VectorBundle{n,ℙ,g,ν,μ}` uses *byte-encoded* data available at pre-compilation, where
`ℙ` specifies the basis for up and down projection,
`g` is a bilinear form that specifies the metric of the space,
and `μ` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of Leibniz-Taylor monomials).
Lastly, `ν` is the number of tangent variables.
"""
abstract type VectorBundle{Indices,Options,Metrics,Vars,Diff,Name} <: Manifold{Indices} end

const names_cache = NTuple{4,String}[]
function names_index(a::NTuple{4,String})
    if a ∈ names_cache
        findfirst(x->x==a,names_cache)
    else
        push!(names_cache,a)
        length(names_cache)
    end
end
@pure names_index(V::T) where T<:VectorBundle{N,M,S,F,D,Q} where {N,M,S,F,D} where Q = Q
@pure namelist(V) = names_cache[names_index(V)]

## Signature{N}

struct Signature{Indices,Options,Signatures,Vars,Diff,Name} <: VectorBundle{Indices,Options,Signatures,Vars,Diff,Name}
    @pure Signature{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure Signature{N,M,S,F,D}() where {N,M,S,F,D} = Signature{N,M,S,F,D,1}()
@pure Signature{N,M,S}() where {N,M,S} = Signature{N,M,S,0,0}()
@pure Signature{N,M}(b::BitArray{1},f=0,d=0) where {N,M} = Signature{N,M,bit2int(b[1:N]),f,d}()
@pure Signature{N,M}(b::Array{Bool,1},f=0,d=0) where {N,M} = Signature{N,M}(convert(BitArray{1},b),f,d)
@pure Signature{N,M}(s::String) where {N,M} = Signature{N,M}([k=='-' for k∈s])
@pure Signature(n::Int,d::Int=0,o::Int=0,s::Bits=zero(Bits)) = Signature{n,doc2m(d,o),s}()
@pure Signature(str::String) = Signature{length(str)}(str)

@pure function Signature{N}(s::String) where N
    ms = match(r"[0-9]+",s)
    if ms ≠ nothing && String(ms.match) == s
        length(s) < 4 && (s *= join(zeros(Int,5-length(s))))
        Signature(parse(Int,s[1]),parse(Int,s[2]),parse(Int,s[3]),UInt(parse(Int,s[4:end])))
    else
        Signature{N,doc2m(Int(vio[1]∈s),Int(vio[2]∈s))}(replace(replace(s,vio[1]=>'+'),vio[2]=>'-'))
    end
end

@inline function getindex(::Signature{N,M,S,F} where M,i::Int) where {N,S,F}
    d = one(Bits) << (i-1)
    return (d & S) == d
end

@inline getindex(vs::Signature,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Signature,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Signature{N,M,S,F} where S,i::Colon) where {N,M,F} = getindex(vs,1:N-(mixedmode(vs)<0 ? 2F : F))
Base.firstindex(m::VectorBundle) = 1
Base.lastindex(m::VectorBundle{N}) where N = N
Base.length(s::VectorBundle{N}) where N = N

@inline sig(s::Bool) = s ? '-' : '+'

function Base.show(io::IO,s::Signature)
    dm = diffmode(s)
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = mixedmode(s),diffvars(s)
    N = ndims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    print(io,sig.(s[hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):N])...)
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x] for x ∈ 1:abs(d)]...)
    d>0 && C<0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    names_index(s)>1 && print(io,subs[names_index(s)])
end

## DiagonalForm{N}

struct DiagonalForm{Indices,Options,Signatures,Vars,Diff,Name} <: VectorBundle{Indices,Options,Signatures,Vars,Diff,Name}
    @pure DiagonalForm{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure DiagonalForm{N,M,S,F,D}() where {N,M,S,F,D} = DiagonalForm{N,M,S,F,D,1}()
@pure DiagonalForm{N,M,S}() where {N,M,S} = DiagonalForm{N,M,S,0,0}()

@pure diagonalform(V::DiagonalForm{N,M,S} where N) where {M,S} = mixedmode(V)>0 ? SUB(diagonalform_cache[S]) : diagonalform_cache[S]

const diagonalform_cache = SVector[]
function DiagonalForm{N,M}(b::SVector{N}) where {N,M}
    a = mixedmode(M)>0 ? SUB(b) : b
    if a ∈ diagonalform_cache
        DiagonalForm{N,M,findfirst(x->x==a,diagonalform_cache)}()
    else
        push!(diagonalform_cache,a)
        DiagonalForm{N,M,length(diagonalform_cache)}()
    end
end

DiagonalForm{N,M}(b::Vector) where {N,M} = DiagonalForm{N,M}(SVector(b...))
DiagonalForm(b::SVector{N}) where N = DiagonalForm{N,0}(b)
DiagonalForm(b::Vector) = DiagonalForm{length(b),0}(b)
DiagonalForm(b::Tuple) = DiagonalForm{length(b),0}(SVector(b))
DiagonalForm(b...) = DiagonalForm(b)
DiagonalForm(s::String) = DiagonalForm(Meta.parse(s).args)

for t ∈ (Any,Integer)
    @eval @inline getindex(s::DiagonalForm{N,M,S} where {N,M},i::T) where {S,T<:$t} = diagonalform(s)[i]
end
@inline getindex(vs::DiagonalForm,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::DiagonalForm,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::DiagonalForm{N,M,S} where M,i::Colon) where {N,S} = diagonalform(vs)

function Base.show(io::IO,s::DiagonalForm)
    dm = diffmode(s)
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = mixedmode(s),diffvars(s)
    N = ndims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    for k ∈ hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):N
        print(io,s[k])
        k ≠ ndims(s) && print(io,',')
    end
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x] for x ∈ 1:abs(d)]...)
    d>0 && C<0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    names_index(s)>1 && print(io,subs[names_index(s)])
end

@pure Signature(V::DiagonalForm{N,M}) where {N,M} = Signature{N,M}(Vector(signbit.(V[:])))
@pure DiagonalForm(V::Signature{N,M}) where {N,M} = DiagonalForm{N,M}([t ? -1 : 1 for t∈V[:]])

## SubManifold{N}

"""
    SubManifold{G,V,B} <: Manifold{G}

Basis type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, bits `B::UInt64`.
"""
struct SubManifold{N,V,Indices} <: Manifold{N}
    @pure SubManifold{N,V,S}() where {N,V,S} = new{N,V,S}()
end

@pure SubManifold{N,M}(b::UInt) where {N,M} = SubManifold{N,M,b}()
SubManifold{N,M}(b::SVector{N}) where {N,M} = SubManifold{N,M}(bit2int(indexbits(ndims(M),b)))
SubManifold{M}(b::Vector) where M = SubManifold{length(b),M}(SVector(b...))
SubManifold{M}(b::Tuple) where M = SubManifold{length(b),M}(SVector(b...))
SubManifold{M}(b...) where M = SubManifold{M}(b)

for t ∈ (Any,Integer)
    @eval @inline function getindex(::SubManifold{N,M,S} where N,i::T) where {T<:$t,M,S}
        val = M[indices(S)[i]]
        typeof(M)<:Signature ? (val ? -1 : 1) : val
    end
end
@inline getindex(vs::SubManifold,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::SubManifold,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline function getindex(::SubManifold{N,M,S} where N,i::Colon) where {M,S}
    val = M[indices(S)]
    typeof(M)<:Signature ? [v ? -1 : 1 for v ∈ val] : val
end

function Base.iterate(r::SubManifold, i::Int=1)
    Base.@_inline_meta
    length(r) < i && return nothing
    Base.unsafe_getindex(r, i), i + 1
end

@pure bits(b::SubManifold{G,V,B} where {G,V}) where B = B
@inline indices(b::SubManifold{G,V} where G) where V = indices(bits(b),ndims(V))
@pure names_index(V::SubManifold{G,M} where G) where M = names_index(M)
@pure Manifold(V::SubManifold{G,M} where G) where M = typeof(M)<:SubManifold ? M : V
@inline interop(op::Function,a::A,b::B) where {A<:SubManifold{X,V} where X,B<:SubManifold{Y,V} where Y} where V = op(a,b)
@inline interform(a::A,b::B) where {A<:SubManifold{X,V} where X,B<:SubManifold{Y,V} where Y} where V = a(b)

function Base.show(io::IO,s::SubManifold{NN,M,S}) where {NN,M,S}
    typeof(M)<:SubManifold && (return printindices(io,M,bits(s)))
    dm = diffmode(s)
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = mixedmode(s),diffvars(s)
    N = NN-(d>0 ? (C<0 ? 2d : d) : 0)
    dM = diffvars(M)
    NM = ndims(M)-(dM>0 ? (C<0 ? 2dM : dM) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    ind = indices(S)
    toM = typeof(M)<:Signature
    for k ∈ hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):NM
        print(io,k ∈ ind ? (toM ? sig(M[k]) : M[k]) : '_')
        !toM && k ≠ NN && print(io,',')
    end
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x-NM] for x ∈ ind[N+1:N+abs(d)]]...)
    d>0 && C<0 && print(io,[sups[x-NM] for x ∈ ind[N+abs(d)+1:end]]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    names_index(s)>1 && print(io,subs[names_index(s)])
end

@pure Signature(V::SubManifold{N}) where N = Signature{N,options(V)}(Vector(signbit.(V[:])),diffvars(V),diffmode(V))
@pure SubManifold(V::M) where M<:Manifold{N} where N = SubManifold{N,V}()
@pure SubManifold{M}() where M<:Manifold{N} where N = SubManifold{N,V}()
@pure SubManifold{N,V}() where {N,V} = SubManifold{N,V}(UInt(1)<<N-1)

@pure (T::Signature{N,M,S,F,D})(::Signature{N,M,S,F,D}) where {N,M,S,F,D} = SubManifold(SubManifold(T))
@pure function (W::Signature)(::SubManifold{G,V,R}) where {G,V,R}
    V==W && (return SubManifold{G,SubManifold(W)}(R))
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    B = typeof(V)<:SubManifold ? expandbits(ndims(W),subvert(V),R) : R
    if WC<0 && VC≥0
        C = mixed(V,B)
        #getbasis(W,mixed(V,B))
        SubManifold{count_ones(C),SubManifold(W)}(C)
    elseif WC≥0 && VC≥0
        #getbasis(W,B)
        SubManifold{count_ones(B),SubManifold(W)}(B)
    else
        throw(error("arbitrary Manifold intersection not yet implemented."))
    end
end

#===(a::SubManifold{G,V},b::SubManifold{G,V}) where {G,V} = bits(a) == bits(b)
==(a::SubManifold{G,V} where V,b::SubManifold{L,W} where W) where {G,L} = false=#

for A ∈ (Signature,DiagonalForm,SubManifold)
    for B ∈ (Signature,DiagonalForm,SubManifold)
        @eval @pure ==(a::$A,b::$B) = (a⊆b) && (a⊇b)
    end
end

# conversions

@pure subvert(::SubManifold{M,V,S} where {M,V}) where S = S

@pure function mixed(V::M,ibk::UInt) where M<:Manifold
    N,D,VC = ndims(V),diffvars(V),mixedmode(V)
    return if D≠0
        A,B = ibk&(UInt(1)<<(N-D)-1),ibk&diffmask(V)
        VC>0 ? (A<<(N-D))|(B<<N) : A|(B<<(N-D))
    else
        VC>0 ? ibk<<N : ibk
    end
end

#=@pure function (W::SubManifold{M,V,S})(b::SubManifold{G,V,B}) where {M,V,S,G,B}
    count_ones(B&S)==G ? getbasis(W,lowerbits(ndims(V),S,B)) : g_zero(W)
end=#

# macros

Manifold(s::Number) = Signature(s)
function Manifold(s::String)
    try
        DiagonalForm(s)
    catch
        Signature(s)
    end
end

export vectorspace, @V_str, @S_str, @D_str

macro V_str(str)
    Manifold(str)
end

macro S_str(str)
    Signature(str)
end

macro D_str(str)
    DiagonalForm(str)
end

# generic

(M::Signature)(b::Int...) = SubManifold{M}(b)
(M::DiagonalForm)(b::Int...) = SubManifold{M}(b)
(M::SubManifold)(b::Int...) = SubManifold{M}(b)

@pure Base.ndims(S::SubManifold{G,M}) where {G,M} = typeof(M)<:SubManifold ? ndims(M) : G
@pure grade(V::M) where M<:Manifold{N} where N = N-(mixedmode(V)<0 ? 2 : 1)*diffvars(V)
@pure hasinf(M::Int) = M%16 ∈ (1,3,5,7,9,11)
@pure hasorigin(M::Int) = M%16 ∈ (2,3,6,7,10,11)
@pure mixedmode(M::Int) = M%16 ∈ 8:11 ? -1 : Int(M%16 ∈ (4,5,6,7))
@pure polymode(M::Int) = iszero(M&16)
@pure hasinf(::T) where T<:VectorBundle{N,M} where N where M = hasinf(M)
@pure hasorigin(::T) where T<:VectorBundle{N,M} where N where M = hasorigin(M)
@pure mixedmode(::T) where T<:VectorBundle{N,M} where N where M = mixedmode(M)
@pure polymode(::T) where T<:VectorBundle{N,M} where N where M = polymode(M)
@pure options(::T) where T<:VectorBundle{N,M} where N where M = M
@pure options_list(V::M) where M<:Manifold = hasinf(V),hasorigin(V),mixedmode(V),polymode(V)
@pure metric(::T) where T<:VectorBundle{N,M,S} where {N,M} where S = S
@pure value(::T) where T<:VectorBundle{N,M,S} where {N,M} where S = S
@pure diffvars(::T) where T<:VectorBundle{N,M,S,F} where {N,M,S} where F = F
@pure diffmode(::T) where T<:VectorBundle{N,M,S,F,D} where {N,M,S,F} where D = D
@pure order(V::M) where M<:Manifold = diffvars(V)

@pure hasinf(::SubManifold{N,M,S} where N) where {M,S} = hasinf(M) && 1∈indices(S)
@pure hasorigin(::SubManifold{N,M,S} where N) where {M,S} = hasorigin(M) && (hasinf(M) ? 2 : 1)∈indices(S)
@pure mixedmode(::SubManifold{N,M} where N) where M = mixedmode(M)
@pure options(::T) where T<:SubManifold{N,M} where N where M = options(M)
@pure metric(::T) where T<:SubManifold{N,M} where N where M = metric(M)
@pure value(::T) where T<:SubManifold{N,M} where N where M = metric(M)
@pure diffmode(::SubManifold{N,M} where N) where M = diffmode(M)
@pure function diffvars(::SubManifold{N,M,S}) where {N,M,S}
    n,C = ndims(M),diffmode(M)
    sum(in.(1+n-(C<0 ? 2 : 1)*diffvars(M):n,Ref(indices(S))))
end

@pure det(s::Signature) = isodd(count_ones(metric(s))) ? -1 : 1
@pure det(s::DiagonalForm) = PROD(diagonalform(s))

@pure abs(s::M) where M<:Manifold = sqrt(abs(det(s)))

@pure hasconformal(V) = hasinf(V) && hasorigin(V)

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
    if mixedmode(V)<0
        v = ((one(Bits)<<d)-1)<<(ndims(V)-2d)
        w = ((one(Bits)<<d)-1)<<(ndims(V)-d)
        return d<0 ? (typemax(Bits)-v,typemax(Bits)-w) : (v,w)
    end
    v = ((one(Bits)<<d)-1)<<(ndims(V)-d)
    d<0 ? typemax(Bits)-v : v
end

@pure function symmetricsplit(V::M,a) where M<:Manifold
    sm,dm = symmetricmask(V,a),diffmask(V)
    mixedmode(V)<0 ? (sm&dm[1],sm&dm[2]) : sm
end

@pure function symmetricmask(V::M,a) where M<:Manifold
    d = diffmask(V)
    a&(mixedmode(V)<0 ? |(d...) : d)
end

@pure function symmetricmask(V::M,a,b) where M<:Manifold
    d = diffmask(V)
    D = mixedmode(V)<0 ? |(d...) : d
    aD,bD = (a&D),(b&D)
    return a&~D, b&~D, aD|bD, aD&bD
end

@pure function diffcheck(V::M,A::Bits,B::Bits) where M<:Manifold
    d,db = diffvars(V),diffmask(V)
    v = mixedmode(V)<0 ? db[1]|db[2] : db
    hi = hasinf2(V,A,B) && !hasorigin(V,A,B)
    ho = hasorigin2(V,A,B) && !hasinf(V,A,B)
    (hi || ho) || (d≠0 && count_ones(A&v)+count_ones(B&v)>diffmode(V))
end

@pure tangent(s::Signature{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = Signature{N+(mixedmode(s)<0 ? 2f : f),M,S,f,D+d}()
@pure tangent(s::DiagonalForm{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = DiagonalForm{N+(mixedmode(s)<0 ? 2f : f),M,S,f,D+d}()

@pure subtangent(V) = V(grade(V)+1:ndims(V)...)

for M ∈ (:Signature,:DiagonalForm)
    @eval @pure loworder(V::$M{N,M,S,D,O}) where {N,M,S,D,O} = O≠0 ? $M{N,M,S,D,O-1}() : V
end
@pure loworder(::SubManifold{N,M,S}) where {N,M,S} = SubManifold{N,loworder(M),S}()

export metric

@pure metric(V::Signature,b::Bits) = isodd(count_ones(metric(V)&b)) ? -1 : 1
@pure metric(V::M,b::Bits) where M<:Manifold = PROD(V[indices(b)])

# dual involution

@pure dual(V::T) where T<:Manifold = mixedmode(V)<0 ? V : V'
@pure dual(V::T,B,M=Int(N/2)) where T<:Manifold{N} where N = ((B<<M)&((1<<N)-1))|(B>>M)

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::Signature{N,M,S,F,D}) where {N,M,S,F,D}
    C = mixedmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    Signature{N,doc2m(hasinf(V),hasorigin(V),Int(!Bool(C))),flip_sig(N,S),F,D}()
end

@pure function Base.adjoint(V::DiagonalForm{N,M,S,F,D}) where {N,M,S,F,D}
    C = mixedmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    DiagonalForm{N,doc2m(hasinf(V),hasorigin(V),Int(!Bool(C))),S,F,D}()
end

@pure function Base.adjoint(V::SubManifold{N,M,S}) where {N,M,S}
    C = mixedmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    SubManifold{N,M',S}()
end

## default definitions

const V0 = Signature(0)
const ℝ = Signature(1)

include("operations.jl")
include("indices.jl")

end # module
