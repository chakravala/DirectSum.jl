
#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

const pre = ("v","w")
const vsn = (:V,:VV,:W)
const digs = "1234567890"
const low_case,upp_case = "abcdefghijklmnopqrstuvwxyz","ABCDEFGHIJKLMNOPQRSTUVWXYZ"
const low_greek,upp_greek = "αβγδϵζηθικλμνξοπρστυφχψω","ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΡΣΤΥΦΨΩ"
const alphanumv = digs*low_case*upp_case #*low_greek*upp_greek
const alphanumw = digs*upp_case*low_case #*upp_greek*low_greek

const subs = Dict{Int,Char}(
   -1 => vio[1],
    0 => vio[2],
    1 => '₁',
    2 => '₂',
    3 => '₃',
    4 => '₄',
    5 => '₅',
    6 => '₆',
    7 => '₇',
    8 => '₈',
    9 => '₉',
    10 => '₀',
    [j=>alphanumv[j] for j ∈ 11:36]...
)

const sups = Dict{Int,Char}(
   -1 => vio[1],
    0 => vio[2],
    1 => '¹',
    2 => '²',
    3 => '³',
    4 => '⁴',
    5 => '⁵',
    6 => '⁶',
    7 => '⁷',
    8 => '⁸',
    9 => '⁹',
    10 => '⁰',
    [j=>alphanumw[j] for j ∈ 11:36]...
)

const VTI = Union{Vector{Int},Tuple,NTuple}

@inline function indexbits(d::Integer,b::VTI)
    out = falses(d)
    for k ∈ b
        out[k] = true
    end
    return out
end

@pure indices(b::Bits) = findall(digits(b,base=2).==1)
@pure shift_indices(V::VectorSpace,b::Bits) = shift_indices(V,indices(b))
function shift_indices(s::VectorSpace{N,M} where N,set::Vector{Int}) where M
    if !isempty(set)
        k = 1
        hasinf(s) && set[1] == 1 && (set[1] = -1; k += 1)
        shift = hasinf(s) + hasorigin(s)
        hasorigin(s) && length(set)>=k && set[k]==shift && (set[k]=0;k+=1)
        shift > 0 && (set[k:end] .-= shift)
    end
    return set
end

@inline printindex(i,e::String=pre[1],t=i>36) = (e≠pre[1])⊻t ? sups[t ? i-26 : i] : subs[t ? i-26 : i]
@inline printindices(io::IO,b::VTI,e::String=pre[1]) = print(io,e,[printindex(i,e) for i ∈ b]...)
@inline function printindices(io::IO,a::VTI,b::VTI,e::String=pre[1],f::String=pre[2])
    F = !isempty(b)
    !(F && isempty(a)) && printindices(io,a,e)
    F && printindices(io,b,f)
end
@inline function printindices(io::IO,V::VectorSpace,e::Bits)
    C = dualtype(V)
    if C < 0
        N = Int(ndims(V)/2)
        printindices(io,shift_indices(V,e & Bits(2^N-1)),shift_indices(V,e>>N))
    else
        printindices(io,shift_indices(V,e),C>0 ? pre[2] : pre[1])
    end
end
