module DirectSum

#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorSpace, Signature, DiagonalForm, ℝ, ⊕, value, tangent
import Base: getindex, abs, @pure, +, *, ^, ∪, ∩, ⊆, ⊇
import LinearAlgebra: det
using StaticArrays

## utilities

Bits = UInt

bit2int(b::BitArray{1}) = parse(Bits,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))

const vio = ('∞','∅')

signbit(x...) = Base.signbit(x...)
signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-
PROD(x) = Base.prod(x)
SUB(x) = Base.:-(x)
SUB(x::Symbol) = :(-$x)
SUB(x::SArray) = Base.:-(x)
SUB(x::SArray{Tuple{M},T,1,M} where M) where T<:Any = broadcast(SUB,x)

## VectorSpace{N}

abstract type VectorSpace{Indices,Options,Metrics,Diff} end

## Signature{N}

struct Signature{Indices,Options,Signatures,Diff} <: VectorSpace{Indices,Options,Signatures,Diff}
    @pure Signature{N,M,S,D}() where {N,M,S,D} = new{N,M,S,D}()
end

@pure Signature{N,M,S}() where {N,M,S} = Signature{N,M,S,0}()
@pure Signature{N,M}(b::BitArray{1}) where {N,M} = Signature{N,M,bit2int(b[1:N])}()
@pure Signature{N,M}(b::Array{Bool,1}) where {N,M} = Signature{N,M}(convert(BitArray{1},b))
@pure Signature{N,M}(s::String) where {N,M} = Signature{N,M}([k=='-' for k∈s])
@pure Signature(n::Int,d::Int=0,o::Int=0,s::Bits=zero(Bits)) = Signature{n,doc2m(d,o),s}()
@pure Signature(str::String) = Signature{length(str)}(str)

@pure function Signature{N}(s::String) where N
    ms = match(r"[0-9]+",s)
    if ms ≠ nothing && String(ms.match) == s
        length(s) < 4 && (s *= join(zeros(Int,5-length(s))))
        Signature(parse(Int,s[1]),parse(Int,s[2]),parse(Int,s[3]),UInt(parse(Int,s[4:end])))
    else
        Signature{N,doc2m(Int(vio[1]∈s),Int(vio[2]∈s))}(replace(replace(s,vio[1]=>'+'),vio[2]=>'+'))
    end
end

function getindex(::Signature{N,M,S,D} where M,i::Int) where {N,S,D}
    d = one(Bits) << (i-1)
    return (d & S) == d
end

getindex(vs::Signature,i::Vector) = [getindex(vs,j) for j ∈ i]
getindex(vs::Signature,i::UnitRange{Int}) = [getindex(vs,j) for j ∈ i]
getindex(vs::Signature{N,M,S} where M,i::Colon) where {N,S} = getindex(vs,1:N)
Base.firstindex(m::VectorSpace) = 1
Base.lastindex(m::VectorSpace{N}) where N = N
Base.length(s::VectorSpace{N}) where N = N

@inline sig(s::Bool) = s ? '-' : '+'

@inline function Base.show(io::IO,s::Signature)
    print(io,'⟨')
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    d = diffmode(s)
    d<0 && print(io,[subs[x] for x ∈ abs(d):-1:1]...)
    print(io,sig.(s[hasinf(s)+hasorigin(s)+1+(d<0 ? abs(d) : 0):ndims(s)-(d>0 ? d : 0)])...)
    d>0 && print(io,[sups[x] for x ∈ 1:abs(d)]...)
    print(io,'⟩')
    C = dualtype(s)
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

## DiagonalForm{N}

struct DiagonalForm{Indices,Options,Signatures,Diff} <: VectorSpace{Indices,Options,Signatures,Diff}
    @pure DiagonalForm{N,M,S,D}() where {N,M,S,D} = new{N,M,S,D}()
end

@pure DiagonalForm{N,M,S}() where {N,M,S} = DiagonalForm{N,M,S,0}()

@pure diagonalform(V::DiagonalForm{N,M,S} where N) where {M,S} = dualtype(V)>0 ? SUB(diagonalform_cache[S]) : diagonalform_cache[S]

const diagonalform_cache = SVector[]
function DiagonalForm{N,M}(b::SVector{N}) where {N,M}
    a = dualtype(M)>0 ? SUB(b) : b
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

@inline getindex(s::DiagonalForm{N,M,S} where {N,M},i) where S = diagonalform(s)[i]
getindex(vs::DiagonalForm{N,M,S} where M,i::Colon) where {N,S} = diagonalform(vs)

@inline function Base.show(io::IO,s::DiagonalForm)
    print(io,'⟨')
    hasinf(s) && print(io,vio[1])
    hasorigin(s) && print(io,vio[2])
    for k ∈ hasinf(s)+hasorigin(s)+1:ndims(s)
        print(io,s[k])
        k ≠ ndims(s) && print(io,',')
    end
    print(io,'⟩')
    C = dualtype(s)
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

@pure Signature(V::DiagonalForm{N,M}) where {N,M} = Signature{N,M}(Vector(signbit.(V[:])))
@pure DiagonalForm(V::Signature{N,M}) where {N,M} = DiagonalForm{N,M}([t ? -1 : 1 for t∈V[:]])

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

@pure Base.ndims(::VectorSpace{N}) where N = N
@pure hasinf(M::Int) = M ∈ (1,3,5,7,9,11)
@pure hasorigin(M::Int) = M ∈ (2,3,6,7,10,11)
@pure dualtype(M::Int) = M ∈ 8:11 ? -1 : Int(M ∈ (4,5,6,7))
@pure hasinf(::VectorSpace{N,M} where N) where M = hasinf(M)
@pure hasorigin(::VectorSpace{N,M} where N) where M = hasorigin(M)
@pure dualtype(::VectorSpace{N,M} where N) where M = dualtype(M)
@pure options(::VectorSpace{N,M} where N) where M = M
@pure options_list(V::VectorSpace) = hasinf(V),hasorigin(V),dualtype(V)
@pure value(::VectorSpace{N,M,S} where {N,M}) where S = S
@pure diffmode(::VectorSpace{N,M,S,D} where {N,M,S}) where D = D

det(s::Signature) = isodd(count_ones(value(s))) ? -1 : 1
det(s::DiagonalForm) = PROD(diagonalform(s))

abs(s::VectorSpace) = sqrt(abs(det(s)))

tangent(s::Signature{N,M,S,D},d::Int=1) where {N,M,S,D} = Signature{N+abs(d),M,S,D+d}()
tangent(s::DiagonalForm{N,M,S,D},d::Int=1) where {N,M,S,D} = DiagonalForm{N+abs(d),M,S,D+d}()

export metric

@pure metric(V::Signature,b::Bits) = isodd(count_ones(value(V)&b)) ? -1 : 1
@pure metric(V::DiagonalForm,b::Bits) = PROD(V[indices(b)])

# dual involution

dual(V::VectorSpace) = dualtype(V)<0 ? V : V'
dual(V::VectorSpace{N},B,M=Int(N/2)) where N = ((B<<M)&((1<<N)-1))|(B>>M)

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::Signature{N,M,S}) where {N,M,S}
    C = dualtype(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    Signature{N,doc2m(hasinf(V),hasorigin(V),Int(!Bool(C))),flip_sig(N,S)}()
end

@pure function Base.adjoint(V::DiagonalForm{N,M,S}) where {N,M,S}
    C = dualtype(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    DiagonalForm{N,doc2m(hasinf(V),hasorigin(V),Int(!Bool(C))),S}()
end

## default definitions

const V0 = Signature(0)
const ℝ = Signature(1)

include("operations.jl")
include("indices.jl")

end # module
