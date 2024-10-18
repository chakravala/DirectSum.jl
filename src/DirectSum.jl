module DirectSum

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
#    ___  _                  _    ___
#   | . \<_> _ _  ___  ___ _| |_ / __> _ _ ._ _ _
#   | | || || '_>/ ._>/ | ' | |  \__ \| | || ' ' |
#   |___/|_||_|  \___.\_|_. |_|  <___/`___||_|_|_|

export TensorBundle, Signature, DiagonalForm, Manifold, Submanifold, ℝ, ⊕, mdims
import Base: getindex, convert, @pure, +, *, ∪, ∩, ⊆, ⊇, ==
import LinearAlgebra, AbstractTensors
import LinearAlgebra: det, rank, isdiag
using Leibniz, ComputedFieldTypes

## Manifold{N}

import AbstractTensors: TensorAlgebra, Manifold, TensorGraded, scalar, isscalar, involute
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume, equal, ⋆
import AbstractTensors: value, valuetype, interop, interform, even, odd, isnull, norm, SUM
import AbstractTensors: TupleVector, Values, Variables, FixedVector, basis, mdims, PROD
import AbstractTensors: Scalar, GradedVector, Bivector, Trivector, SUB

import Leibniz: Fields, pre, PRE, vsn, VTI, bit2int, combo, indexbits, indices
import Leibniz: printlabel, supermanifold, shift_indices, shift_indices!, printindices
import Leibniz: symmetricmask, parityleft, parityright, paritylefthodge, combine
import Leibniz: parityrighthodge, parityclifford, parityconj, parityreverse, parityinvolute
import Leibniz: parityrightnull, parityleftnull, parityrightnullpre, parityleftnullpre
import Leibniz: hasconformal, parval, TensorTerm, mixed, subs, sups, vio, gdims

import Leibniz: grade, order, options, metric, polymode, dyadmode, diffmode, diffvars
import Leibniz: pseudograde, hasinf, hasorigin, norm, indices, isbasis, ≅
import Leibniz: isdyadic, isdual, istangent, involute, basis, alphanumv, alphanumw

import Leibniz: algebra_limit, sparse_limit, cache_limit, fill_limit
import Leibniz: binomial, gdimsall, binomsum, binomcumsum, lowerbits, expandbits
import Leibniz: bladeindex, basisindex, indexbasis, indexbasis_set, loworder, intlog
import Leibniz: promote_type, mvec, svec, intlog, insert_expr, indexparity!

## TensorBundle{N}

"""
    TensorBundle{n,ℙ,g,ν,μ} <: Manifold{n}

A manifold representing the space of tensors with a given metric and basis for projection.

Let `n` be the rank of a `Manifold{n}`.
The type `TensorBundle{n,ℙ,g,ν,μ}` uses *byte-encoded* data available at pre-compilation, where
`ℙ` specifies the basis for up and down projection,
`g` is a bilinear form that specifies the metric of the space,
and `μ` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of Leibniz-Taylor monomials).
Lastly, `ν` is the number of tangent variables.
"""
abstract type TensorBundle{n,Options,Metrics,Vars,Diff,Name} <: Manifold{n,Int} end
rank(::TensorBundle{n}) where n = n
mdims(M::TensorBundle) = rank(M)
# Return the manifold dimension (i.e., rank) of a tensor bundle.

"""
    tensorhash(d, o; c=0, C=0) -> Int

Compute a unique integer representation for a set of options.

# Arguments:
- `d`: Number of tangent variables (corresponds to `ν`).
- `o`: Order of Leibniz-Taylor monomials multiplicity limit (corresponds to `μ`).
- `c`: Coefficient used to specify the metric (corresponds to `g`). Defaults to 0.
- `C`: Additional coefficient. Defaults to 0.

# Returns
A unique integer representation that can be used as a cache key.
"""
@pure function tensorhash(d,o,c=0,C=0)
    (1<<(d-1))|(1<<(2*o-1))|(c<0 ? 8 : (1<<(3*c-1)))|(1<<(5*C-1))
end

const namecache = NTuple{4,String}[]

"""
    nameindex(a::NTuple{4, String}) -> Int

Get a unique index for a set of name options, based on a cached result.
"""
function nameindex(a::NTuple{4,String})
    if a ∈ namecache # check if options are in cache
        findfirst(x->x==a,namecache)
    else # cache the options and add them to the list
        push!(namecache,a)
        length(namecache)
    end
end

# initialize vector and co-vector prefix
nameindex(pre)
nameindex(PRE)

"""
    nameindex(V::T) -> Int

Returns the name index for `V` when `V` is a `TensorBundle` or `Manifold`.
"""
@pure nameindex(V::T) where T<:TensorBundle{N,M,S,F,D,Q} where {N,M,S,F,D} where Q = Q
@pure nameindex(V::T) where T<:Manifold = nameindex(supermanifold(V))

"""
    nameindex(V::Int) -> Int

Returns a default name index for an integer input.
"""
@pure nameindex(V::Int) = 1

"""
    namelist(V) -> String

Get the cached name list for the options specified in `V`.
"""
@pure namelist(V) = namecache[nameindex(V)]

## Signature{N}

struct Signature{Indices,Options,Signatures,Vars,Diff,Name} <: TensorBundle{Indices,Options,Signatures,Vars,Diff,Name}
    @pure Signature{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure Signature{N,M,S,F,D}() where {N,M,S,F,D} = Signature{N,M,S,F,D,1}()
@pure Signature{N,M,S}() where {N,M,S} = Signature{N,M,S,0,0}()
@pure Signature{N,M}(b::BitArray{1},f=0,d=0) where {N,M} = Signature{N,M,bit2int(b[1:N]),f,d}()
@pure Signature{N,M}(b::Vector{Bool},f=0,d=0) where {N,M} = Signature{N,M}(convert(BitArray{1},b),f,d)
@pure Signature{N,M}(s::String) where {N,M} = Signature{N,M}([k=='-' for k∈s])
@pure Signature(str::String) = Signature{length(str)}(str)
@pure Signature(n::Int,d::Int=0,o::Int=0,s::UInt=zero(UInt)) = Signature{n,tensorhash(d,o),s}()
@pure function Signature{N}(s::String) where N
    ms = match(r"[0-9]+",s)
    if ms ≠ nothing && String(ms.match) == s
        length(s) < 4 && (s *= join(zeros(Int,5-length(s))))
        Signature(parse(Int,s[1]),parse(Int,s[2]),parse(Int,s[3]),UInt(parse(Int,s[4:end])))
    else
        Signature{N,tensorhash(Int(vio[1]∈s),Int(vio[2]∈s))}(replace(replace(s,vio[1]=>'+'),vio[2]=>'-'))
    end
end

@inline function getindex(::Signature{N,M,S,F} where M,i::Int) where {N,S,F}
    d = one(UInt) << (i-1)
    return (d & S) == d
end
@inline getindex(vs::Signature,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Signature,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Signature{N,M,S,F} where S,i::Colon) where {N,M,F} = getindex(vs,1:N-(isdyadic(vs) ? 2F : F))
Base.firstindex(m::TensorBundle) = 1
Base.lastindex(m::TensorBundle{N}) where N = N
Base.length(s::TensorBundle{N}) where N = N

# Type promotion to allow using `Int` with `Signature`
Base.promote_rule(::Type{Int}, ::Type{<:Signature}) = Signature
#Base.promote_rule(::Type{Int}, ::Type{<:TensorBundle}) = Signature

@inline sig(s::Bool) = s ? '-' : '+'
@inline sig(s::Int,k) = '1'
@inline sig(s,k) = s[k]
@inline sig(s::Signature,k) = sig(s[k])
@inline printsep(io,s::Signature,k,n) = nothing
@inline printsep(io,s::Int,k,n) = nothing
@inline printsep(io,s,k,n) = k≠n && print(io,',')

function Base.show(io::IO,s::Signature)
    dm = diffmode(s) # print lead symbols
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = dyadmode(s),diffvars(s)
    N = mdims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    print(io,sig.(s[hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):N])...)
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x] for x ∈ 1:abs(d)]...)
    d>0 && C<0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩') # print end symbols
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    nameindex(s)>1 && print(io,subs[nameindex(s)])
end

## DiagonalForm{N}

struct DiagonalForm{Indices,Options,Signatures,Vars,Diff,Name} <: TensorBundle{Indices,Options,Signatures,Vars,Diff,Name}
    @pure DiagonalForm{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure DiagonalForm{N,M,S,F,D}() where {N,M,S,F,D} = DiagonalForm{N,M,S,F,D,1}()
@pure DiagonalForm{N,M,S}() where {N,M,S} = DiagonalForm{N,M,S,0,0}()
@pure DiagonalForm{N,M}(b::Values{N}) where {N,M} = DiagonalForm{N,M,diagsig(M,b)}()
DiagonalForm{N,M}(b::Vector) where {N,M} = DiagonalForm{N,M}(Values(b...))
@pure DiagonalForm(b::Values{N}) where N = DiagonalForm{N,0}(b)
@pure DiagonalForm(b::Tuple) = DiagonalForm{length(b),0}(Values(b))
DiagonalForm(b::Vector) = DiagonalForm{length(b),0}(b)
DiagonalForm(b...) = DiagonalForm(b)
DiagonalForm(s::String) = DiagonalForm(Meta.parse(s).args)

@pure diagonalform(V::DiagonalForm{N,M,S} where N) where {M,S} = isdual(V) ? SUB(diagonalform_cache[S]) : diagonalform_cache[S]
const diagonalform_cache = Values[]
@pure function diagsig(M,b::Values)
    a = dyadmode(M)>0 ? SUB(b) : b
    if a ∈ diagonalform_cache
        findfirst(x->x==a,diagonalform_cache)
    else
        push!(diagonalform_cache,a)
        length(diagonalform_cache)
    end
end

for t ∈ (Any,Integer)
    @eval @inline getindex(s::DiagonalForm{N,M,S} where {N,M},i::T) where {S,T<:$t} = diagonalform(s)[i]
end
@inline getindex(vs::DiagonalForm,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::DiagonalForm,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::DiagonalForm{N,M,S} where M,i::Colon) where {N,S} = diagonalform(vs)

function Base.show(io::IO,s::DiagonalForm)
    dm = diffmode(s)
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = dyadmode(s),diffvars(s)
    N = mdims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    for k ∈ hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):N
        print(io,s[k])
        k ≠ mdims(s) && print(io,',')
    end
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x] for x ∈ 1:abs(d)]...)
    d>0 && C<0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    nameindex(s)>1 && print(io,subs[nameindex(s)])
end

## Submanifold{N}

"""
    Submanifold{V,G,B} <: TensorGraded{V,G} <: Manifold{G}

Basis type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, bits `B::UInt64`.
"""
struct Submanifold{V,n,Indices} <: TensorTerm{V,n,Int}
    @pure Submanifold{V,n,S}() where {V,n,S} = new{V,n,S}()
end

@pure Submanifold(V::Int) = Submanifold{V,V}()
@pure Submanifold(V::M) where M<:Manifold = Submanifold{V,rank(V)}()
#@pure Submanifold{M}() where M = Submanifold{M isa Int ? Submanifold(M) : M,rank(M)}()
@pure Submanifold{V,N}() where {V,N} = Submanifold{V,N}(UInt(1)<<N-1)
@pure Submanifold{M,N}(b::UInt) where {M,N} = Submanifold{M,N,b}()
@pure Submanifold{M,N}(b::Submanifold{M,N}) where {M,N} = b
Submanifold{M,N}(b::Values{N}) where {M,N} = Submanifold{M,N}(bit2int(indexbits(mdims(M),b)))
Submanifold{M}(b::UnitRange) where M = Submanifold{M,length(b)}(Values(b...))
Submanifold{M}(b::Vector) where M = Submanifold{M,length(b)}(Values(b...))
Submanifold{M}(b::Tuple) where M = Submanifold{M,length(b)}(Values(b...))
Submanifold{M}(b::Values) where M = Submanifold{M,length(b)}(b)
Submanifold{M}(b...) where M = Submanifold{M}(b)

@pure issubmanifold(V::Submanifold) = true
@pure issubmanifold(V) = false

for t ∈ ((:V,),(:V,:G))
    @eval begin
        function Submanifold{$(t...)}(b::VTI) where {$(t...)}
            Submanifold{V}(indexbits(mdims(V),b))
        end
        function Submanifold{$(t...)}(b::Int...) where {$(t...)}
            Submanifold{V}(indexbits(mdims(V),b))
        end
    end
end

for t ∈ (Any,Integer)
    @eval @inline function getindex(::Submanifold{M,N,S} where N,i::T) where {T<:$t,M,S}
        if typeof(M)<:Submanifold
            d = one(UInt) << (i-1)
            return (d & UInt(M)) == d
        elseif typeof(M)<:Int
            1
        else
            ind = indices(S)
            val = M[ind[i]]
            if typeof(M)<:Signature
                val ? -1 : 1
            else
                typeof(val)<:Values ? val[Values(ind...)] : val
            end
        end
    end
end
@inline getindex(vs::Submanifold,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Submanifold,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline function getindex(::Submanifold{M,N,S} where N,i::Colon) where {M,S}
    typeof(M)<:Int && (return ones(Int,M))
    ind = indices(S)
    val = M[ind]
    if typeof(M)<:Signature
        [v ? -1 : 1 for v ∈ val]
    else
        eltype(val) <: Values ? getindex.(val,Ref(Values(ind...))) : val
    end
end

function Base.iterate(r::Submanifold, i::Int=1)
    Base.@_inline_meta
    length(r) < i && return nothing
    Base.getindex(r, i), i + 1
end

#@inline interop(op::Function,a::A,b::B) where {A<:Submanifold{V},B<:Submanifold{V}} where V = op(a,b)
@inline interform(a::A,b::B) where {A<:Submanifold{V},B<:Submanifold{V}} where V = a(b)

@inline sig(s::Submanifold{V},k) where V = isdiag(V) ? sig(V,k) : s[k]

function Base.show(io::IO,s::Submanifold{V,NN,S}) where {V,NN,S}
    isbasis(s) && (return printindices(io,V,UInt(s)))
    P = typeof(V)<:Int ? V : parent(V)
    PnV = typeof(P) ≠ typeof(V)
    PnV && print(io,'Λ',sups[rank(V)])
    M = PnV ? supermanifold(P) : V
    dm = diffmode(s)
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = dyadmode(s),diffvars(s)
    N = NN-(d>0 ? (C<0 ? 2d : d) : 0)
    dM = diffvars(M)
    NM = mdims(M)-(dM>0 ? (C<0 ? 2dM : dM) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    ind = indices(S)
    for k ∈ hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):NM
        met = if k ∈ ind
            metr = sig(s,isdiag(V) ? k : findfirst(x->x==k,ind))
            typeof(metr)==Bool ? metr ? '-' : '+' : metr
        else
            '_'
        end
        print(io,met)
        printsep(io,M,k,NM)
    end
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x-NM] for x ∈ ind[N+1:N+abs(d)]]...)
    d>0 && C<0 && print(io,[sups[x-NM] for x ∈ ind[N+abs(d)+1:end]]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    nameindex(s)>1 && print(io,subs[nameindex(s)])
    PnV && print(io,'×',length(V))
end

# ==(a::Submanifold{V,G},b::Submanifold{V,G}) where {V,G} = UInt(a) == UInt(b)
# ==(a::Submanifold{V,G} where V,b::Submanifold{W,L} where W) where {G,L} = false
# ==(a::Submanifold{V,G},b::Submanifold{W,G}) where {V,W,G} = interop(==,a,b)

for A ∈ (Signature,DiagonalForm,Submanifold)
    @eval @pure Manifold(::Type{T}) where T<:$A = T()
    for B ∈ (Signature,DiagonalForm,Submanifold)
        @eval begin
            @pure equal(a::$A,b::$B) = (a⊆b) && (a⊇b)
            @pure ==(::Type{A},b::$B) where A<:$A = A() == b
            @pure ==(a::$A,::Type{B}) where B<:$B = a == B()
            @pure ==(::Type{A},::Type{B}) where {A<:$A,B<:$B} = A() == B()
        end
    end
end

# conversions

@pure Manifold(V::Submanifold{M}) where M = (t=typeof(M);t<:Submanifold||t<:Int ? M : V)
@pure Signature(V::Submanifold{M,N} where M) where N = Signature{N,options(V)}(Vector(signbit.(V[:])),diffvars(V),diffmode(V))
@pure Signature(V::DiagonalForm{N,M}) where {N,M} = Signature{N,M}(Vector(signbit.(V[:])))
@pure DiagonalForm(V::Signature{N,M}) where {N,M} = DiagonalForm{N,M}([t ? -1 : 1 for t∈V[:]])

@pure submanifold(V::Submanifold) = isbasis(V) ? V(Manifold(V)) : V
@pure submanifold(V::TensorBundle) = Submanifold(V)
@pure submanifold(V::Int) = Submanifold(V)

# indices

#@pure supblade(N,S,B) = bladeindex(N,expandbits(N,S,B))
#@pure supmulti(N,S,B) = basisindex(N,expandbits(N,S,B))

@inline indices(b::Submanifold{V}) where V = indices(UInt(b),mdims(V))

shift_indices(V::M,b::UInt) where M<:TensorBundle = shift_indices!(V,copy(indices(b,mdims(V))))
shift_indices(V::T,b::UInt) where T<:Submanifold{M,N,S} where {M,N,S} = shift_indices!(V,copy(indices(S,mdims(M))[indices(b,mdims(V))]))

printindices(io::IO,V::T,e::UInt,label::Bool=false) where T<:Manifold = printlabel(io,V,e,label,namelist(V)...)

# macros

TensorBundle(s::T) where T<:Number = Signature(s)
function TensorBundle(s::String)
    try
        parse(Int,s)
    catch
        try
            DiagonalForm(s)
        catch
            Signature(s)
        end
    end
end

Manifold(s::String) = TensorBundle(s)
Manifold(s::T) where T<:Number = TensorBundle(s)

export @V_str, @S_str, @D_str

macro V_str(str)
    TensorBundle(str)
end

macro S_str(str)
    Signature(str)
end

macro D_str(str)
    DiagonalForm(str)
end

## default definitions

const V0 = Signature(0)
const ℝ = Signature(1)
for n ∈ 0:9
    Rn = Symbol(:ℝ,n)
    @eval begin
        const $Rn = Submanifold($n)
        export $Rn
    end
end

"""
    Single{V,G,B,T} <: TensorTerm{V,G,T} <: TensorGraded{V,G,T}

Single type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, `B::Submanifold{V,G}`, field `T::Type`.
"""
struct Single{V,G,B,T} <: TensorTerm{V,G,T}
    v::T
    Single{A,B,C,D}(t) where {A,B,C,D} = new{submanifold(A),B,basis(C),D}(t)
    Single{A,B,C,D}(t::E) where E<:TensorAlgebra{A} where {A,B,C,D} = new{submanifold(A),B,basis(C),D}(t)
end

export Single
Single(v::Real) = Single{Submanifold(0)}(v)
Single(v::Complex) = Single{Submanifold(0)}(v)
@pure Single(b::Submanifold{V,G}) where {V,G} = Single{V}(b)
@pure Single{V}(b::Submanifold{V,G}) where {V,G} = Single{V,G,b,Int}(1)
Single{V}(v::T) where {V,T} = Single{V,0,Submanifold{V}(),T}(v)
Single{V}(v::S) where S<:TensorTerm where V = v
Single{V}(v::Tuple{UInt,T}) where {V,T} = @inbounds Single{V}(v[2],Submanifold{V}(v[1]))
Single{V,G,B}(v::T) where {V,G,B,T} = Single{V,G,B,T}(v)
Single(v,b::S) where S<:TensorTerm{V} where V = Single{V}(v,b)
Single{V}(v,b::S) where S<:TensorAlgebra where V = v*b
Single{V}(v,b::Submanifold{V,G}) where {V,G} = Single{V,G}(v,b)
Single{V}(v,b::Submanifold{W,G}) where {V,W,G} = Single{V,G}(v,b)
function Single{V,G}(v::T,b::Submanifold{V,G}) where {V,G,T}
    order(v)+order(b)>diffmode(V) ? Zero(V) : Single{V,G,b,T}(v)
end
function Single{V,G}(v::T,b::Submanifold{W,G}) where {V,W,G,T}
    order(v)+order(b)>diffmode(V) ? Zero(V) : Single{V,G,V(b),T}(v)
end
function Single{V,G}(v::T,b::Submanifold{V,G}) where T<:TensorTerm where {G,V}
    order(v)+order(b)>diffmode(V) ? Zero(V) : Single{V,G,b,Any}(v)
end
function Single{V,G,B}(b::T) where T<:TensorTerm{V} where {V,G,B}
    order(B)+order(b)>diffmode(V) ? Zero(V) : Single{V,G,B,Any}(b)
end
Base.show(io::IO,m::Single) = Leibniz.showvalue(io,Manifold(m),UInt(basis(m)),value(m))
for VG ∈ ((:V,),(:V,:G))
    @eval function Single{$(VG...)}(v,b::Single{V,G}) where {V,G}
        order(v)+order(b)>diffmode(V) ? Zero(V) : Single{V,G,basis(b)}(AbstractTensors.∏(v,b.v))
    end
end

#(::Type{T})(m::Single{V,G,B,<:Real} where {V,G,B}) where T<:Real = T(value(m))
Base.Real(m::Single{V,G,B,<:Real} where {V,G,B}) = value(m)
(::Type{T})(m::Single{V,G,B,<:Real} where {V,G,B}) where T<:AbstractFloat = T(value(m))
(::Type{Bool})(m::Single{V,G,B,<:Real} where {V,G,B}) = Bool(value(m))
(::Type{T})(m::Single{V,G,B,<:Real} where {V,G,B}) where T<:Signed = T(value(m))
(::Type{Rational})(m::Single{V,G,B,<:Real} where {V,G,B}) = Rational(value(m))
(::Type{Rational{T}})(m::Single{V,G,B,<:Real} where {V,G,B}) where T<:Integer = Rational{T}(value(m))
(::Type{T})(m::Single{V,G,B,<:Real} where {V,G,B}) where T<:AbstractIrrational = T(value(m))
(::Type{Complex})(m::Single{V,0,B,T} where {V,B}) where T<:Real = Complex(value(m),zero(T))
(::Type{Complex{T}})(m::Single{V,0,B,<:Real} where {V,B}) where T = Complex{T}(value(m),zero(T))
(::Type{Complex})(m::Single{V,G,B,T} where {V,G,B}) where T<:Real = Complex(zero(T),value(m))
(::Type{Complex{T}})(m::Single{V,G,B,<:Real} where {V,G,B}) where T = Complex{T}(zero(T),value(m))
(::Type{Complex})(m::Single{V,G,B,<:Complex} where {V,G,B}) = value(m)
(::Type{Complex{T}})(m::Single{V,G,B,<:Complex} where {V,G,B}) where T = Complex{T}(value(m))

equal(a::TensorTerm{V,G},b::TensorTerm{V,G}) where {V,G} = basis(a) == basis(b) ? value(a) == value(b) : 0 == value(a) == value(b)

for T ∈ (Fields...,Symbol,Expr)
    @eval begin
        Base.isapprox(a::S,b::T;atol::Real=0,rtol::Real=Base.rtoldefault(a,b,atol),nans::Bool=false,norm::Function=LinearAlgebra.norm) where {S<:TensorAlgebra,T<:$T} = Base.isapprox(a,Single{Manifold(a)}(b);atol=atol,rtol=rtol,nans=nans,norm=norm)
        Base.isapprox(a::S,b::T;atol::Real=0,rtol::Real=Base.rtoldefault(a,b,atol),nans::Bool=false,norm::Function=LinearAlgebra.norm) where {S<:$T,T<:TensorAlgebra} = Base.isapprox(b,a;atol=atol,rtol=rtol,nans=nans,norm=norm)
    end
end

for Field ∈ Fields
    TF = Field ∉ Fields ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    @eval begin
        Base.:*(a::F,b::Submanifold{V}) where {F<:$EF,V} = Single{V}(a,b)
        Base.:*(a::Submanifold{V},b::F) where {F<:$EF,V} = Single{V}(b,a)
        Base.:*(a::F,b::Single{V,G,B,T} where B) where {F<:$Field,V,G,T<:$Field} = Single{V,G}(Base.:*(a,b.v),basis(b))
        Base.:*(a::Single{V,G,B,T} where B,b::F) where {F<:$Field,V,G,T<:$Field} = Single{V,G}(Base.:*(a.v,b),basis(a))
        Base.adjoint(b::Single{V,G,B,T}) where {V,G,B,T<:$Field} = Single{dual(V),G,B',$TF}(Base.conj(value(b)))
    end
end

for M ∈ (:Signature,:DiagonalForm,:Submanifold)
    @eval begin
        @inline function (V::$M)(s::LinearAlgebra.UniformScaling{T}) where T
            b = getbasis(V,(one(UInt)<<(mdims(V)-diffvars(V)))-1)
            T<:Bool ? b : Single{V}(s.λ,b)
        end
        (W::$M)(b::Single) = Single{W}(value(b),W(basis(b)))
        ==(::Type{<:$M}, ::Type{Union{}}) = false
    end
end

# fixes an error in Julia, acceptable piracy
==(::Type{Union{}},::Type{Union{}}) = true

# One{V} <: TensorGraded{0,V}

"""
    One{V} <: TensorGraded{V,0} <: TensorAlgebra{V}

Unit quantity `One` of the `Grassmann` algebra over `V`.
"""
const One{V} = Submanifold{V,0,UInt(0)}

# Zero{V} <: TensorGraded{0,V}

export Zero, One

"""
    Zero{V} <: TensorGraded{V,0} <: TensorAlgebra{V}

Null quantity `Zero` of the `Grassmann` algebra over `V`.
"""
struct Zero{V} <: TensorTerm{V,0,Int}
    @pure Zero{V}() where V = new{submanifold(V)}()
end
@pure Zero(V::T) where T<:TensorBundle = Zero{V}()
@pure Zero(V::Type{<:TensorBundle}) = Zero(V())
@pure Zero(V::Int) = Zero(submanifold(V))
@pure Zero(V::Submanifold{M}) where M = Zero{isbasis(V) ? M : V}()
@pure One(V::T) where T<:Submanifold = Submanifold{Manifold(V)}()
@pure One(V::T) where T<:TensorBundle = Submanifold{V}()
@pure One(V::Int) = Submanifold{V,0}()
@pure One(b::Type{Submanifold{V}}) where V = Submanifold{V}()
@pure One(V::Type{<:TensorBundle}) = Submanifold{V()}()

(::Type{T})(m::Zero) where T<:Real = zero(T)
#(::Type{T})(m::Submanifold) where T<:Real = one(T) # causes ambiguity
Base.Real(m::Submanifold) = one(Real)
(::Type{T})(m::Submanifold) where T<:AbstractFloat = one(T)
(::Type{T})(m::Submanifold) where T<:Signed = one(T)
(::Type{Rational})(m::Submanifold) = one(Rational)
(::Type{Rational{T}})(m::Submanifold) where T = one(Rational{T})
(::Type{Complex})(m::Zero) = Complex(0)
(::Type{Complex{T}})(m::Zero) where T = Complex{T}(zero(T),zero(T))
(::Type{Complex})(m::One) = Complex(1)
(::Type{Complex{T}})(m::One) where T = Complex{T}(one(T),zero(T))
(::Type{Complex})(m::Submanifold) = Complex(0,1)
(::Type{Complex{T}})(m::Submanifold) where T = Complex{T}(zero(T),value(m))

for id ∈ (:Zero,:One)
    @eval begin
        @inline $id(t::T) where T<:TensorAlgebra = $id(Manifold(t))
        @inline $id(::Type{<:TensorAlgebra{V}}) where V = $id(V)
        @inline $id(::Type{<:TensorGraded{V}}) where V = $id(V)
    end
end

@pure Base.iszero(::Zero) = true
@pure Base.isone(::Zero) = false
@pure Base.isinf(::Zero) = false

@pure AbstractTensors.value(::Zero) = 0

Base.show(io::IO,::Zero{V}) where V = print(io,"𝟎")

==(::Zero,::Zero) = true
==(a::T,::Zero) where T<:TensorAlgebra = iszero(a)
==(::Zero,b::T) where T<:TensorAlgebra = iszero(b)
for T ∈ Fields
    @eval begin
        ==(a::T,::Zero) where T<:$T = iszero(a)
        ==(::Zero,b::T) where T<:$T = iszero(b)
    end
end

import Base: reverse, conj
import AbstractTensors: hodge, clifford, complementleft, complementlefthodge
for op ∈ (:hodge,:clifford,:complementleft,:complementlefthodge,:involute,:conj,:reverse)
    @eval $op(t::Zero) = t
end

@inline Base.abs2(t::Zero) = t

@pure One(::Type{T}) where T = one(T)
@pure Zero(::Type{T}) where T = zero(T)

# Infinity{V} <: TensorGraded{0,V}

export Infinity

"""
    Infinity{V} <: TensorGraded{V,0} <: TensorAlgebra{V}

Infinite quantity `Infinity` of the `Grassmann` algebra over `V`.
"""
struct Infinity{V} <: TensorTerm{V,0,Float64}
    @pure Infinity{V}() where V = new{submanifold(V)}()
end
@pure Infinity(V::T) where T<:TensorBundle = Infinity{V}()
@pure Infinity(V::Type{<:TensorBundle}) = Infinity(V())
@pure Infinity(V::Int) = Infinity(submanifold(V))
@pure Infinity(V::Submanifold{M}) where M = Infinity{isbasis(V) ? M : V}()

Real(::Infinity) = Inf
(::Type{T})(m::Infinity) where T<:AbstractFloat = T(Inf)
(::Type{Complex})(m::Infinity) = Complex(Inf)
(::Type{Complex{T}})(m::Infinity) where T = Complex{T}(T(Inf),T(Inf))

for id ∈ (:Infinity,)
    @eval begin
        @inline $id(t::T) where T<:TensorAlgebra = $id(Manifold(t))
        @inline $id(::Type{<:TensorAlgebra{V}}) where V = $id(V)
        @inline $id(::Type{<:TensorGraded{V}}) where V = $id(V)
    end
end

@pure Base.iszero(::Infinity) = false
@pure Base.isone(::Infinity) = false
@pure Base.isinf(::Infinity) = true

@pure AbstractTensors.value(::Infinity) = Inf

Base.show(io::IO,::Infinity{V}) where V = print(io,"∞")

==(::Infinity,::Infinity) = true
==(a::T,::Infinity) where T<:TensorAlgebra = isinf(norm(a))
==(::Infinity,b::T) where T<:TensorAlgebra = isinf(norm(b))
for T ∈ Fields
    @eval begin
        ==(a::T,::Infinity) where T<:$T = isinf(a)
        ==(::Infinity,b::T) where T<:$T = isinf(b)
    end
end

import Base: reverse, conj
import AbstractTensors: hodge, clifford, complementleft, complementlefthodge
for op ∈ (:hodge,:clifford,:complementleft,:complementlefthodge,:involute,:conj,:reverse)
    @eval $op(t::Infinity) = t
end

@inline Base.abs2(t::Infinity) = t

@pure Infinity(::Type{T}) where T<:AbstractFloat = T(Inf)

include("generic.jl")
include("operations.jl")
include("basis.jl")
include("grade.jl")

end # module
