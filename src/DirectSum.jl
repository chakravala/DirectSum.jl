module DirectSum

#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export VectorSpace, vectorspace, @V_str, ℝ, ⊕
import Base: getindex, @pure, +, *, ^, ∪, ∩, ⊆, ⊇

## utilities

Bits = UInt

bit2int(b::BitArray{1}) = parse(Bits,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))

## VectorSpace{N}

struct VectorSpace{Indices,Options,Signatures}
    @pure VectorSpace{N,M,S}() where {N,M,S} = new{N,M,S}()
end

@pure VectorSpace{N,M}(b::BitArray{1}) where {N,M} = VectorSpace{N,M,bit2int(b[1:N])}()
@pure VectorSpace{N,M}(b::Array{Bool,1}) where {N,M} = VectorSpace{N,M}(convert(BitArray{1},b))
@pure VectorSpace{N,M}(s::String) where {N,M} = VectorSpace{N,M}([k=='-' for k∈s])
@pure VectorSpace(n::Int,d::Int=0,o::Int=0,s::Bits=zero(Bits)) = VectorSpace{n,doc2m(d,o),s}()
@pure VectorSpace(str::String) = VectorSpace{length(str)}(str)

@pure function VectorSpace{N}(s::String) where N
    ms = match(r"[0-9]+",s)
    if ms ≠ nothing && String(ms.match) == s
        length(s) < 4 && (s *= join(zeros(Int,5-length(s))))
        VectorSpace(parse(Int,s[1]),parse(Int,s[2]),parse(Int,s[3]),UInt(parse(Int,s[4:end])))
    else
        VectorSpace{N,doc2m(Int('ϵ'∈s),Int('o'∈s))}(replace(replace(s,'ϵ'=>'+'),'o'=>'+'))
    end
end

@pure function getindex(::VectorSpace{N,M,S} where {N,M},i::Int) where S
    d = one(Bits) << (i-1)
    return (d & S) == d
end

@pure getindex(vs::VectorSpace{N,M,S} where {N,M},i::UnitRange{Int}) where S = [getindex(vs,j) for j ∈ i]
@pure getindex(vs::VectorSpace{N,M,S} where M,i::Colon) where {N,S} = [getindex(vs,j) for j ∈ 1:N]
Base.firstindex(m::VectorSpace) = 1
Base.lastindex(m::VectorSpace{N}) where N = N
Base.length(s::VectorSpace{N}) where N = N

@inline sig(s::Bool) = s ? '-' : '+'

@inline function Base.show(io::IO,s::VectorSpace)
    print(io,'⟨')
    hasdual(s) && print(io,'ϵ')
    hasorigin(s) && print(io,'o')
    print(io,sig.(s[hasdual(s)+hasorigin(s)+1:ndims(s)])...)
    print(io,'⟩')
    C = dualtype(s)
    C ≠ 0 ? print(io, C < 0 ? '*' : ''') : nothing
end

macro V_str(str)
    VectorSpace(str)
end

@pure Base.ndims(::VectorSpace{N}) where N = N
@pure hasdual(::VectorSpace{N,M} where N) where M = M ∈ (1,3,5,7,9,11)
@pure hasorigin(::VectorSpace{N,M} where N) where M = M ∈ (2,3,6,7,10,11)
@pure dualtype(::VectorSpace{N,M} where N) where M = M ∈ 8:11 ? -1 : Int(M ∈ (4,5,6,7))
@pure options(::VectorSpace{N,M} where N) where M = M
@pure options_list(V::VectorSpace) = hasdual(V),hasorigin(V),dualtype(V)
@inline value(::VectorSpace{N,M,S} where {N,M}) where S = S

# dual involution

dual(V::VectorSpace) = dualtype(V)<0 ? V : V'
dual(V::VectorSpace{N},B,M=Int(N/2)) where N = ((B<<M)&((1<<N)-1))|(B>>M)

@pure flip_sig(N,S::Bits) = Bits(2^N-1) & (~S)

@pure function Base.adjoint(V::VectorSpace{N,M,S}) where {N,M,S}
    C = dualtype(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    VectorSpace{N,doc2m(hasdual(V),hasorigin(V),Int(!Bool(C))),flip_sig(N,S)}()
end

## default definitions

function vectorspace end
const V0 = VectorSpace(0)
const ℝ = VectorSpace(1)

include("operations.jl")
include("indices.jl")

end # module
