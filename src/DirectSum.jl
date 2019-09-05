module DirectSum

#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorBundle, Signature, DiagonalForm, ℝ, ⊕, value, tangent, Manifold, SubManifold
import Base: getindex, abs, @pure, +, *, ^, ∪, ∩, ⊆, ⊇
import LinearAlgebra: det
using StaticArrays

## utilities

Bits = UInt

bit2int(b::BitArray{1}) = parse(Bits,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))

const vio = ('∞','∅')

value(x::T) where T<:Number = x
signbit(x...) = Base.signbit(x...)
signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-
conj(z) = Base.conj(z)
inv(z) = Base.inv(z)
/(a,b) = Base.:/(a,b)
-(x) = Base.:-(x)
-(a,b) = Base.:-(a,b)
-(x::Symbol) = :(-$x)
-(x::SArray) = Base.:-(x)
-(x::SArray{Tuple{M},T,1,M} where M) where T<:Any = broadcast(-,x)

for (OP,op) ∈ ((:∏,:*),(:∑,:+))
    @eval begin
        $OP(x...) = Base.$op(x...)
        $OP(x::AbstractVector{T}) where T<:Any = $op(x...)
    end
end

const PROD,SUM,SUB = ∏,∑,-

## Manifold{N}

abstract type Manifold{Indices} end

## VectorBundle{N}

abstract type VectorBundle{Indices,Options,Metrics,Vars,Diff} <: Manifold{Indices} end

## Signature{N}

struct Signature{Indices,Options,Signatures,Vars,Diff} <: VectorBundle{Indices,Options,Signatures,Vars,Diff}
    @pure Signature{N,M,S,F,D}() where {N,M,S,F,D} = new{N,M,S,F,D}()
end

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
    print(io,'⟨')
    C,d = mixedmode(s),diffvars(s)
    N = ndims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    print(io,sig.(s[hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):N])...)
    d>0 && print(io,[(C>0 ? sups : subs)[x] for x ∈ 1:abs(d)]...)
    d>0 && C<0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

## DiagonalForm{N}

struct DiagonalForm{Indices,Options,Signatures,Vars,Diff} <: VectorBundle{Indices,Options,Signatures,Vars,Diff}
    @pure DiagonalForm{N,M,S,F,D}() where {N,M,S,F,D} = new{N,M,S,F,D}()
end

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

@inline getindex(vs::DiagonalForm,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::DiagonalForm,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline getindex(s::DiagonalForm{N,M,S} where {N,M},i) where S = diagonalform(s)[i]
@inline getindex(vs::DiagonalForm{N,M,S} where M,i::Colon) where {N,S} = diagonalform(vs)

function Base.show(io::IO,s::DiagonalForm)
    print(io,'⟨')
    C,d = mixedmode(s),diffvars(s)
    N = ndims(s)-(d>0 ? (C<0 ? 2d : d) : 0)
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    for k ∈ hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):N
        print(io,s[k])
        k ≠ ndims(s) && print(io,',')
    end
    d>0 && print(io,[(C>0 ? sups : subs)[x] for x ∈ 1:abs(d)]...)
    d>0 && C<0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

@pure Signature(V::DiagonalForm{N,M}) where {N,M} = Signature{N,M}(Vector(signbit.(V[:])))
@pure DiagonalForm(V::Signature{N,M}) where {N,M} = DiagonalForm{N,M}([t ? -1 : 1 for t∈V[:]])

## SubManifold{N}

struct SubManifold{N,V,Indices} <: Manifold{N}
    @pure SubManifold{N,V,S}() where {N,V,S} = new{N,V,S}()
end

@pure SubManifold{N,M}(b::UInt) where {N,M} = SubManifold{N,M,b}()
SubManifold{N,M}(b::SVector{N}) where {N,M} = SubManifold{N,M}(bit2int(indexbits(ndims(M),b)))
SubManifold{M}(b::Vector) where M = SubManifold{length(b),M}(SVector(b...))
SubManifold{M}(b::Tuple) where M = SubManifold{length(b),M}(SVector(b...))
SubManifold{M}(b...) where M = SubManifold{M}(b)

@inline function getindex(::SubManifold{N,M,S} where N,i) where {M,S}
    val = M[indices(S)[i]]
    typeof(M)<:Signature ? (val ? -1 : 1) : val
end
@inline getindex(vs::SubManifold,i::Vector) = [getindex(vs,j) for j ∈ i]
@inline getindex(vs::SubManifold,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
@inline function getindex(::SubManifold{N,M,S} where N,i::Colon) where {M,S}
    val = M[indices(S)]
    typeof(M)<:Signature ? [v ? -1 : 1 for v ∈ val] : val
end

function Base.show(io::IO,s::SubManifold{NN,M,S}) where {NN,M,S}
    print(io,'⟨')
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
    d>0 && print(io,[(C>0 ? sups : subs)[x-NM] for x ∈ ind[N+1:N+abs(d)]]...)
    d>0 && C<0 && print(io,[sups[x-NM] for x ∈ ind[N+abs(d)+1:end]]...)
    print(io,'⟩')
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

@pure Signature(V::SubManifold{N}) where N = Signature{N,options(V)}(Vector(signbit.(V[:])),diffvars(V),diffmode(V))
@pure SubManifold(V::M) where M<:Manifold{N} where N = SubManifold{N,V}()
@pure SubManifold{M}() where M<:Manifold{N} where N = SubManifold{N,V}()
@pure SubManifold{N,V}() where {N,V} = SubManifold{N,V}(indices(UInt(1)<<N-1))

# macros

vectorspace(s::Number) = Signature(s)
function vectorspace(s::String)
    try
        DiagonalForm(s)
    catch
        Signature(s)
    end
end

export vectorspace, @V_str, @S_str, @D_str

macro V_str(str)
    vectorspace(str)
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

@pure Base.ndims(::M) where M<:Manifold{N} where N = N
@pure odims(V::M) where M<:Manifold{N} where N = N-(mixedmode(V)<0 ? 2 : 1)*diffvars(V)
@pure hasinf(M::Int) = M ∈ (1,3,5,7,9,11)
@pure hasorigin(M::Int) = M ∈ (2,3,6,7,10,11)
@pure mixedmode(M::Int) = M ∈ 8:11 ? -1 : Int(M ∈ (4,5,6,7))
@pure hasinf(::T) where T<:VectorBundle{N,M} where N where M = hasinf(M)
@pure hasorigin(::T) where T<:VectorBundle{N,M} where N where M = hasorigin(M)
@pure mixedmode(::T) where T<:VectorBundle{N,M} where N where M = mixedmode(M)
@pure options(::T) where T<:VectorBundle{N,M} where N where M = M
@pure options_list(V::M) where M<:Manifold = hasinf(V),hasorigin(V),mixedmode(V)
@pure value(::T) where T<:VectorBundle{N,M,S} where {N,M} where S = S
@pure diffvars(::T) where T<:VectorBundle{N,M,S,F} where {N,M,S} where F = F
@pure diffmode(::T) where T<:VectorBundle{N,M,S,F,D} where {N,M,S,F} where D = D

@pure hasinf(::SubManifold{N,M,S} where N) where {M,S} = hasinf(M) && 1∈indices(S)
@pure hasorigin(::SubManifold{N,M,S} where N) where {M,S} = hasorigin(M) && (hasinf(M) ? 2 : 1)∈indices(S)
@pure mixedmode(::SubManifold{N,M} where N) where M = mixedmode(M)
@pure options(::T) where T<:SubManifold{N,M} where N where M = options(M)
@pure value(::T) where T<:SubManifold{N,M} where N where M = value(M)
@pure diffmode(::SubManifold{N,M} where N) where M = diffmode(M)
@pure function diffvars(::SubManifold{N,M,S}) where {N,M,S}
    n,C = ndims(M),diffmode(M)
    sum(in.(1+n-(C<0 ? 2 : 1)*diffvars(M):n,Ref(indices(S))))
end

@pure det(s::Signature) = isodd(count_ones(value(s))) ? -1 : 1
@pure det(s::DiagonalForm) = PROD(diagonalform(s))

@pure abs(s::M) where M<:Manifold = sqrt(abs(det(s)))

@pure hasorigin(V::M, B::T) where {M<:Manifold,T<:Bits} = hasinf(V) ? (Bits(2)&B)==Bits(2) : isodd(B)

@pure function hasinf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf(V) && (isodd(A) || isodd(B))
end
@pure function hasorigin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasorigin(V) && (hasorigin(V,A) || hasorigin(V,B))
end

@pure function hasinf2(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf(V) && isodd(A) && isodd(B)
end
@pure function hasorigin2(V::T,A::Bits,B::Bits) where T<:Manifold
    hasorigin(V) && hasorigin(V,A) && hasorigin(V,B)
end

@pure function hasorigininf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf(V) && hasorigin(V) && hasorigin(V,A) && isodd(B) && !hasorigin(V,B) && !isodd(A)
end
@pure function hasinforigin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf(V) && hasorigin(V) && isodd(A) && hasorigin(V,B) && !isodd(B) && !hasorigin(V,A)
end

@pure function hasinf2origin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf2(V,A,B) && hasorigin(V,A,B)
end
@pure function hasorigin2inf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasorigin2(V,A,B) && hasinf(V,A,B)
end

@pure function diffmask(V::T) where T<:Manifold
    d = diffvars(V)
    if mixedmode(V)<0
        v = ((one(Bits)<<d)-1)<<(ndims(V)-2d)
        w = ((one(Bits)<<d)-1)<<(ndims(V)-d)
        return d<0 ? (typemax(Bits)-v,typemax(Bits)-w) : (v,w)
    end
    v = ((one(Bits)<<d)-1)<<(ndims(V)-d)
    d<0 ? typemax(Bits)-v : v
end

@pure function diffcheck(V::T,A::Bits,B::Bits) where T<:Manifold
    d,db = diffvars(V),diffmask(V)
    v = mixedmode(V)<0 ? db[1]|db[2] : db
    hi = hasinf2(V,A,B) && !hasorigin(V,A,B)
    ho = hasorigin2(V,A,B) && !hasinf(V,A,B)
    (hi || ho) || (d≠0 && count_ones((A&v)&(B&v))≠0)
end

@pure tangent(s::Signature{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = Signature{N+f,M,S,f,D+d}()
@pure tangent(s::DiagonalForm{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = DiagonalForm{N+f,M,S,f,D+d}()

export metric

@pure metric(V::Signature,b::Bits) = isodd(count_ones(value(V)&b)) ? -1 : 1
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
