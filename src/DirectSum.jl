module DirectSum

#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export TensorBundle, Signature, DiagonalForm, Manifold, SubManifold, ℝ, ⊕, mdims
import Base: getindex, convert, @pure, +, *, ∪, ∩, ⊆, ⊇, ==
import LinearAlgebra, AbstractTensors
import LinearAlgebra: det, rank
using Leibniz, ComputedFieldTypes

## Manifold{N}

import AbstractTensors: TensorAlgebra, Manifold, TensorGraded, scalar, isscalar, involute
import AbstractTensors: vector, isvector, bivector, isbivector, volume, isvolume, ⋆
import AbstractTensors: value, valuetype, interop, interform, even, odd, isnull, norm, SUM
import AbstractTensors: TupleVector, Values, Variables, FixedVector, basis, mdims, PROD

import Leibniz: Fields, pre, PRE, vsn, VTI, bit2int, combo, indexbits, indices
import Leibniz: printlabel, supermanifold, shift_indices, shift_indices!, printindices
import Leibniz: symmetricmask, parityleft, parityright, paritylefthodge, combine
import Leibniz: parityrighthodge, parityclifford, parityconj, parityreverse, parityinvolute
import Leibniz: parityrightnull, parityleftnull, parityrightnullpre, parityleftnullpre
import Leibniz: hasconformal, parval, TensorTerm, mixed, g_zero, g_one, subs, sups, vio

import Leibniz: grade, order, options, metric, polymode, dyadmode, diffmode, diffvars
import Leibniz: hasinf, hasorigin, norm, indices, isbasis, Bits, bits, ≅
import Leibniz: isdyadic, isdual, istangent, involute, basis, alphanumv, alphanumw

import Leibniz: algebra_limit, sparse_limit, cache_limit, fill_limit
import Leibniz: binomial, binomial_set, binomsum, binomsum_set, lowerbits, expandbits
import Leibniz: bladeindex, basisindex, indexbasis, indexbasis_set, loworder, intlog
import Leibniz: promote_type, mvec, svec, intlog, insert_expr, indexparity!

## TensorBundle{N}

"""
    TensorBundle{n,ℙ,g,ν,μ} <: Manifold{n}

Let `n` be the rank of a `Manifold{n}`.
The type `TensorBundle{n,ℙ,g,ν,μ}` uses *byte-encoded* data available at pre-compilation, where
`ℙ` specifies the basis for up and down projection,
`g` is a bilinear form that specifies the metric of the space,
and `μ` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of Leibniz-Taylor monomials).
Lastly, `ν` is the number of tangent variables.
"""
abstract type TensorBundle{n,Options,Metrics,Vars,Diff,Name} <: Manifold{n} end
rank(::TensorBundle{n}) where n = n
mdims(M::TensorBundle) = rank(M)

@pure doc2m(d,o,c=0,C=0) = (1<<(d-1))|(1<<(2*o-1))|(c<0 ? 8 : (1<<(3*c-1)))|(1<<(5*C-1))

const names_cache = NTuple{4,String}[]
function names_index(a::NTuple{4,String})
    if a ∈ names_cache
        findfirst(x->x==a,names_cache)
    else
        push!(names_cache,a)
        length(names_cache)
    end
end
@pure names_index(V::T) where T<:TensorBundle{N,M,S,F,D,Q} where {N,M,S,F,D} where Q = Q
@pure names_index(V::T) where T<:Manifold = names_index(supermanifold(V))
@pure names_index(V::Int) = 1
@pure namelist(V) = names_cache[names_index(V)]

# vector and co-vector prefix
names_index(pre)
names_index(PRE)

## Signature{N}

struct Signature{Indices,Options,Signatures,Vars,Diff,Name} <: TensorBundle{Indices,Options,Signatures,Vars,Diff,Name}
    @pure Signature{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure Signature{N,M,S,F,D}() where {N,M,S,F,D} = Signature{N,M,S,F,D,1}()
@pure Signature{N,M,S}() where {N,M,S} = Signature{N,M,S,0,0}()
@pure Signature{N,M}(b::BitArray{1},f=0,d=0) where {N,M} = Signature{N,M,bit2int(b[1:N]),f,d}()
@pure Signature{N,M}(b::Array{Bool,1},f=0,d=0) where {N,M} = Signature{N,M}(convert(BitArray{1},b),f,d)
@pure Signature{N,M}(s::String) where {N,M} = Signature{N,M}([k=='-' for k∈s])
@pure Signature(str::String) = Signature{length(str)}(str)
@pure Signature(n::Int,d::Int=0,o::Int=0,s::UInt=zero(UInt)) = Signature{n,doc2m(d,o),s}()

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
    d = one(UInt) << (i-1)
    return (d & S) == d
end
@inline getindex(vs::Signature,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Signature,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::Signature{N,M,S,F} where S,i::Colon) where {N,M,F} = getindex(vs,1:N-(isdyadic(vs) ? 2F : F))
Base.firstindex(m::TensorBundle) = 1
Base.lastindex(m::TensorBundle{N}) where N = N
Base.length(s::TensorBundle{N}) where N = N

Base.promote_rule(::Type{Int}, ::Type{<:Signature}) = Signature

@inline sig(s::Bool) = s ? '-' : '+'
@inline sig(s::Int,k) = '×'
@inline sig(s,k) = s[k]
@inline sig(s::Signature,k) = sig(s[k])
@inline printsep(io,s::Signature,k,n) = nothing
@inline printsep(io,s::Int,k,n) = nothing
@inline printsep(io,s,k,n) = k≠n && print(io,',')

function Base.show(io::IO,s::Signature)
    dm = diffmode(s)
    print(io,dm>0 ? "T$(sups[dm])⟨" : '⟨')
    C,d = dyadmode(s),diffvars(s)
    N = mdims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
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

struct DiagonalForm{Indices,Options,Signatures,Vars,Diff,Name} <: TensorBundle{Indices,Options,Signatures,Vars,Diff,Name}
    @pure DiagonalForm{N,M,S,F,D,L}() where {N,M,S,F,D,L} = new{N,M,S,F,D,L}()
end

@pure DiagonalForm{N,M,S,F,D}() where {N,M,S,F,D} = DiagonalForm{N,M,S,F,D,1}()
@pure DiagonalForm{N,M,S}() where {N,M,S} = DiagonalForm{N,M,S,0,0}()
DiagonalForm{N,M}(b::Vector) where {N,M} = DiagonalForm{N,M}(Values(b...))
DiagonalForm(b::Values{N}) where N = DiagonalForm{N,0}(b)
DiagonalForm(b::Vector) = DiagonalForm{length(b),0}(b)
DiagonalForm(b::Tuple) = DiagonalForm{length(b),0}(Values(b))
DiagonalForm(b...) = DiagonalForm(b)
DiagonalForm(s::String) = DiagonalForm(Meta.parse(s).args)

@pure diagonalform(V::DiagonalForm{N,M,S} where N) where {M,S} = isdual(V) ? SUB(diagonalform_cache[S]) : diagonalform_cache[S]
const diagonalform_cache = Values[]
function DiagonalForm{N,M}(b::Values{N}) where {N,M}
    a = dyadmode(M)>0 ? SUB(b) : b
    if a ∈ diagonalform_cache
        DiagonalForm{N,M,findfirst(x->x==a,diagonalform_cache)}()
    else
        push!(diagonalform_cache,a)
        DiagonalForm{N,M,length(diagonalform_cache)}()
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
    names_index(s)>1 && print(io,subs[names_index(s)])
end

## SubManifold{N}

"""
    SubManifold{V,G,B} <: TensorGraded{V,G} <: Manifold{G}

Basis type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, bits `B::UInt64`.
"""
struct SubManifold{V,n,Indices} <: TensorTerm{V,n}
    @pure SubManifold{V,n,S}() where {V,n,S} = new{V,n,S}()
end

@pure SubManifold(V::Int) where N = SubManifold{V,V}()
@pure SubManifold(V::M) where M<:Manifold = SubManifold{V,rank(V)}()
#@pure SubManifold{M}() where M = SubManifold{M isa Int ? SubManifold(M) : M,rank(M)}()
@pure SubManifold{V,N}() where {V,N} = SubManifold{V,N}(UInt(1)<<N-1)
@pure SubManifold{M,N}(b::UInt) where {M,N} = SubManifold{M,N,b}()
SubManifold{M,N}(b::Values{N}) where {M,N} = SubManifold{M,N}(bit2int(indexbits(mdims(M),b)))
SubManifold{M}(b::UnitRange) where M = SubManifold{M,length(b)}(Values(b...))
SubManifold{M}(b::Vector) where M = SubManifold{M,length(b)}(Values(b...))
SubManifold{M}(b::Tuple) where M = SubManifold{M,length(b)}(Values(b...))
SubManifold{M}(b::Values) where M = SubManifold{M,length(b)}(b)
SubManifold{M}(b...) where M = SubManifold{M}(b)

@pure issubmanifold(V::SubManifold) = true
@pure issubmanifold(V) = false

for t ∈ ((:V,),(:V,:G))
    @eval begin
        function SubManifold{$(t...)}(b::VTI) where {$(t...)}
            SubManifold{V}(indexbits(mdims(V),b))
        end
        function SubManifold{$(t...)}(b::Int...) where {$(t...)}
            SubManifold{V}(indexbits(mdims(V),b))
        end
    end
end

for t ∈ (Any,Integer)
    @eval @inline function getindex(::SubManifold{M,N,S} where N,i::T) where {T<:$t,M,S}
        if typeof(M)<:SubManifold
            d = one(UInt) << (i-1)
            return (d & UInt(M)) == d
        elseif typeof(M)<:Int
            1
        else
            val = M[indices(S)[i]]
            typeof(M)<:Signature ? (val ? -1 : 1) : val
        end
    end
end
@inline getindex(vs::SubManifold,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::SubManifold,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline function getindex(::SubManifold{M,N,S} where N,i::Colon) where {M,S}
    typeof(M)<:Int && (return ones(Int,M))
    val = M[indices(S)]
    typeof(M)<:Signature ? [v ? -1 : 1 for v ∈ val] : val
end

function Base.iterate(r::SubManifold, i::Int=1)
    Base.@_inline_meta
    length(r) < i && return nothing
    Base.getindex(r, i), i + 1
end

#@inline interop(op::Function,a::A,b::B) where {A<:SubManifold{V},B<:SubManifold{V}} where V = op(a,b)
@inline interform(a::A,b::B) where {A<:SubManifold{V},B<:SubManifold{V}} where V = a(b)

function Base.show(io::IO,s::SubManifold{V,NN,S}) where {V,NN,S}
    isbasis(s) && (return printindices(io,V,bits(s)))
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
        print(io,k ∈ ind ? sig(M,k) : '_')
        printsep(io,M,k,NN)
    end
    d>0 && print(io,[((C>0)⊻!polymode(s) ? sups : subs)[x-NM] for x ∈ ind[N+1:N+abs(d)]]...)
    d>0 && C<0 && print(io,[sups[x-NM] for x ∈ ind[N+abs(d)+1:end]]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
    names_index(s)>1 && print(io,subs[names_index(s)])
    PnV && print(io,'×',length(V))
end

# ==(a::SubManifold{V,G},b::SubManifold{V,G}) where {V,G} = UInt(a) == UInt(b)
# ==(a::SubManifold{V,G} where V,b::SubManifold{W,L} where W) where {G,L} = false
# ==(a::SubManifold{V,G},b::SubManifold{W,G}) where {V,W,G} = interop(==,a,b)

for A ∈ (Signature,DiagonalForm,SubManifold)
    @eval @pure Manifold(::Type{T}) where T<:$A = T()
    for B ∈ (Signature,DiagonalForm,SubManifold)
        @eval begin
            @pure ==(a::$A,b::$B) = (a⊆b) && (a⊇b)
            @pure ==(::Type{A},b::$B) where A<:$A = A() == b
            @pure ==(a::$A,::Type{B}) where B<:$B = a == B()
            @pure ==(::Type{A},::Type{B}) where {A<:$A,B<:$B} = A() == B()
        end
    end
end

@inline indices(b::SubManifold{V}) where V = indices(UInt(b),mdims(V))

shift_indices(V::M,b::UInt) where M<:TensorBundle = shift_indices!(V,copy(indices(b,mdims(V))))
shift_indices(V::T,b::UInt) where T<:SubManifold{M,N,S} where {M,N,S} = shift_indices!(V,copy(indices(S,mdims(M))[indices(b,mdims(V))]))

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
        const $Rn = SubManifold($n)
        export $Rn
    end
end

"""
    Simplex{V,G,B,𝕂} <: TensorTerm{V,G} <: TensorGraded{V,G}

Simplex type with pseudoscalar `V::Manifold`, grade/rank `G::Int`, `B::SubManifold{V,G}`, field `𝕂::Type`.
"""
struct Simplex{V,G,B,T} <: TensorTerm{V,G}
    v::T
    Simplex{A,B,C,D}(t::E) where E<:D where {A,B,C,D} = new{A,B,C,D}(t)
    Simplex{A,B,C,D}(t::E) where E<:TensorAlgebra{A} where {A,B,C,D} = new{A,B,C,D}(t)
end

export Simplex
@pure Simplex(b::SubManifold{V,G}) where {V,G} = Simplex{V}(b)
@pure Simplex{V}(b::SubManifold{V,G}) where {V,G} = Simplex{V,G,b,Int}(1)
Simplex{V}(v::T) where {V,T} = Simplex{V,0,SubManifold{V}(),T}(v)
Simplex{V}(v::S) where S<:TensorTerm where V = v
Simplex{V,G,B}(v::T) where {V,G,B,T} = Simplex{V,G,B,T}(v)
Simplex(v,b::S) where S<:TensorTerm{V} where V = Simplex{V}(v,b)
Simplex{V}(v,b::S) where S<:TensorAlgebra where V = v*b
Simplex{V}(v,b::SubManifold{V,G}) where {V,G} = Simplex{V,G}(v,b)
Simplex{V}(v,b::SubManifold{W,G}) where {V,W,G} = Simplex{V,G}(v,b)
function Simplex{V,G}(v::T,b::SubManifold{V,G}) where {V,G,T}
    order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,b,T}(v)
end
function Simplex{V,G}(v::T,b::SubManifold{W,G}) where {V,W,G,T}
    order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,V(b),T}(v)
end
function Simplex{V,G}(v::T,b::SubManifold{V,G}) where T<:TensorTerm where {G,V}
    order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,b,Any}(v)
end
function Simplex{V,G,B}(b::T) where T<:TensorTerm{V} where {V,G,B}
    order(B)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,B,Any}(b)
end
function Base.show(io::IO,m::Simplex)
    T = typeof(value(m))
    par = !(T <: TensorTerm) && |(broadcast(<:,T,parval)...)
    #val = T<:Float64 ? @fprintf() : m.v
    print(io,(par ? ['(',m.v,')'] : [m.v])...,basis(m))
end
for VG ∈ ((:V,),(:V,:G))
    @eval function Simplex{$(VG...)}(v,b::Simplex{V,G}) where {V,G}
        order(v)+order(b)>diffmode(V) ? zero(V) : Simplex{V,G,basis(b)}(AbstractTensors.∏(v,b.v))
    end
end

Base.:(==)(a::TensorTerm{V,G},b::TensorTerm{V,G}) where {V,G} = basis(a) == basis(b) ? value(a) == value(b) : 0 == value(a) == value(b)

for T ∈ (Fields...,Symbol,Expr)
    @eval begin
        Base.isapprox(a::S,b::T;atol::Real=0,rtol::Real=Base.rtoldefault(a,b,atol),nans::Bool=false,norm::Function=LinearAlgebra.norm) where {S<:TensorAlgebra,T<:$T} = Base.isapprox(a,Simplex{Manifold(a)}(b);atol=atol,rtol=rtol,nans=nans,norm=norm)
        Base.isapprox(a::S,b::T;atol::Real=0,rtol::Real=Base.rtoldefault(a,b,atol),nans::Bool=false,norm::Function=LinearAlgebra.norm) where {S<:$T,T<:TensorAlgebra} = Base.isapprox(b,a;atol=atol,rtol=rtol,nans=nans,norm=norm)
    end
end

for Field ∈ Fields
    TF = Field ∉ Fields ? :Any : :T
    EF = Field ≠ Any ? Field : ExprField
    @eval begin
        Base.:*(a::F,b::SubManifold{V}) where {F<:$EF,V} = Simplex{V}(a,b)
        Base.:*(a::SubManifold{V},b::F) where {F<:$EF,V} = Simplex{V}(b,a)
        Base.:*(a::F,b::Simplex{V,G,B,T} where B) where {F<:$Field,V,G,T<:$Field} = Simplex{V,G}(Base.:*(a,b.v),basis(b))
        Base.:*(a::Simplex{V,G,B,T} where B,b::F) where {F<:$Field,V,G,T<:$Field} = Simplex{V,G}(Base.:*(a.v,b),basis(a))
        Base.adjoint(b::Simplex{V,G,B,T}) where {V,G,B,T<:$Field} = Simplex{dual(V),G,B',$TF}(Base.conj(value(b)))
    end
end

include("generic.jl")
include("operations.jl")
include("basis.jl")
#include("e.jl")

end # module
