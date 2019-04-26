
#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

# vector and co-vector prefix
const pre = ("v","w","ϵ","∂")

# vector space and dual-space symbols
const vsn = (:V,:VV,:W)

# alpha-numeric digits
const digs = "1234567890"
const low_case,upp_case = "abcdefghijklmnopqrstuvwxyz","ABCDEFGHIJKLMNOPQRSTUVWXYZ"
const low_greek,upp_greek = "αβγδϵζηθικλμνξοπρστυφχψω","ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΡΣΤΥΦΨΩ"
const alphanumv = digs*low_case*upp_case #*low_greek*upp_greek
const alphanumw = digs*upp_case*low_case #*upp_greek*low_greek

# subscript index
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

# superscript index
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

# converts indices into BitArray of length N
@inline function indexbits(N::Integer,indices::VTI)
    out = falses(N)
    for k ∈ indices
        out[k] = true
    end
    return out
end

# index sets
@pure indices(b::Bits) = findall(digits(b,base=2).==1)
@pure shift_indices(V::T,b::Bits) where T<:VectorSpace = shift_indices(V,indices(b))
function shift_indices(s::T,set::Vector{Int}) where T<:VectorSpace{N,M} where N where M
    if !isempty(set)
        k = 1
        hasinf(s) && set[1] == 1 && (set[1] = -1; k += 1)
        shift = hasinf(s) + hasorigin(s)
        hasorigin(s) && length(set)>=k && set[k]==shift && (set[k]=0;k+=1)
        shift > 0 && (set[k:end] .-= shift)
    end
    return set
end

# printing of indices
@inline function printindex(i,l::Bool=false,e::String=pre[1],t=i>36,j=t ? i-26 : i)
    (l&&(0≤j≤9)) ? j : ((e∉pre[[1,3]])⊻t ? sups[j] : subs[j])
 end
@inline printindices(io::IO,b::VTI,l::Bool=false,e::String=pre[1]) = print(io,e,[printindex(i,l,e) for i ∈ b]...)
@inline printindices(io::IO,a::VTI,b::VTI,l::Bool=false,e::String=pre[1],f::String=pre[2]) = printindices(io,a,b,Int[],Int[],l,e,f)
@inline function printindices(io::IO,a::VTI,b::VTI,c::VTI,d::VTI,l::Bool=false,e::String=pre[1],f::String=pre[2],g::String=pre[3],h::String=pre[4])
    A,B,C,D = isempty(a),!isempty(b),!isempty(c),!isempty(d)
    !((B || C || D) && A) && printindices(io,a,l,e)
    B && printindices(io,b,l,f)
    C && printindices(io,c,l,g)
    D && printindices(io,d,l,h)
end
@pure function printindices(io::IO,V::T,e::Bits,label::Bool=false) where T<:VectorSpace
    N,D,C,db = ndims(V),diffmode(V),dualtype(V),dualbits(V)
    if C < 0
        es = e & (~(db[1]|db[2]))
        n = Int((N-2D)/2)
        eps = shift_indices(V,e & db[1]).-(N-2D)
        par = shift_indices(V,e & db[2]).-(N-D)
        printindices(io,shift_indices(V,es & Bits(2^n-1)),shift_indices(V,es>>n),eps,par,label)
    else
        es = e & (~db)
        eps = shift_indices(V,e & db).-(N-D)
        if !isempty(eps)
            printindices(io,shift_indices(V,es),Int[],C>0 ? Int[] : eps,C>0 ? eps : Int[],label,C>0 ? pre[2] : pre[1])
        else
            printindices(io,shift_indices(V,es),label,C>0 ? pre[2] : pre[1])
        end
    end
end
