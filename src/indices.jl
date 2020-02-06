
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

# vector and co-vector prefix
const pre = ("v","w","∂","ϵ")
const PRE = ("X","x","Y","y")
names_index(pre)
names_index(PRE)

# vector space and dual-space symbols
const vsn = (:V,:VV,:W)
const VSN = (:Χ,:ΧΧ,:Υ) # \Chi,\Upsilon

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

# converts indices into BitArray of length N
@inline function indexbits(N::Integer,indices::SVTI)
    out = falses(N)
    for k ∈ indices
        out[k] = true
    end
    return out
end

# index sets
index_limit = 20
const digits_fast_cache = Vector{SVector}[]
const digits_fast_extra = Dict{UInt,SVector}[]
@pure digits_fast_calc(b,N) = SVector{N+1,Int}(digits(b,base=2,pad=N+1))
@pure function digits_fast(b,N)
    if N>index_limit
        n = N-index_limit
        for k ∈ length(digits_fast_extra)+1:n
            push!(digits_fast_extra,Dict{UInt,SVector{k+1,Int}}())
        end
        !haskey(digits_fast_extra[n],b) && push!(digits_fast_extra[n],b=>digits_fast_calc(b,N))
        @inbounds digits_fast_extra[n][b]
    else
        for k ∈ length(digits_fast_cache)+1:min(N,index_limit)
            push!(digits_fast_cache,[digits_fast_calc(d,k) for d ∈ 0:1<<(k+1)-1])
            GC.gc()
        end
        @inbounds digits_fast_cache[N][b+1]
    end
end

const indices_cache = Dict{UInt,Vector{Int}}()
@pure indices(b::Bits) = findall(digits(b,base=2).==1)
@pure function indices_calc(b::UInt,N::Int)
    d = digits_fast(b,N)
    l = length(d)
    a = Int[]
    for i ∈ 1:l
        d[i] == 1 && push!(a,i)
    end
    return a
end
@pure function indices(b::UInt,N::Int)
    !haskey(indices_cache,b) && push!(indices_cache,b=>indices_calc(b,N))
    return @inbounds indices_cache[b]
end
@inline indices(b::SubManifold{V}) where V = indices(bits(b),ndims(V))

@pure shift_indices(V::M,b::UInt) where M<:TensorBundle = shift_indices!(V,copy(indices(b,ndims(V))))
@pure shift_indices(V::T,b::UInt) where T<:SubManifold{M,N,S} where {M,N,S} = shift_indices!(V,copy(indices(S,ndims(M))[indices(b,ndims(V))]))
@pure shift_indices(V::T,b::UInt) where T<:Manifold = shift_indices(supermanifold(V),b)
function shift_indices!(s::T,set::Vector{Int}) where T<:Manifold
    M = supermanifold(s)
    if !isempty(set)
        k = 1
        hasinf(M) && set[1] == 1 && (set[1] = -1; k += 1)
        shift = hasinf(M) + hasorigin(M)
        hasorigin(M) && length(set)>=k && set[k]==shift && (set[k]=0;k+=1)
        shift > 0 && (set[k:end] .-= shift)
    end
    return set
end

# printing of indices
@inline function printindex(i,l::Bool=false,e::String=pre[1],pre=pre)
    t = i>36; j = t ? i-26 : i
    (l&&(0<j≤10)) ? j : ((e∉pre[[1,3]])⊻t ? sups[j] : subs[j])
 end
 @inline printindices(io::IO,b::UInt,l::Bool=false,e::String=pre[1],pre::NTuple{4,String}=pre) = printindices(io,indices(b),l,e,pre)
@inline printindices(io::IO,b::VTI,l::Bool=false,e::String=pre[1],pre::NTuple{4,String}=pre) = print(io,e,[printindex(i,l,e,pre) for i ∈ b]...)
@inline printindices(io::IO,a::VTI,b::VTI,l::Bool=false,e::String=pre[1],f::String=pre[2]) = printindices(io,a,b,Int[],Int[],l,e,f)
@inline function printindices(io::IO,a::VTI,b::VTI,c::VTI,d::VTI,l::Bool=false,e::String=pre[1],f::String=pre[2],g::String=pre[3],h::String=pre[4])
    A,B,C,D = isempty(a),!isempty(b),!isempty(c),!isempty(d)
    PRE = (e,f,g,h)
    C && printindices(io,c,l,g,PRE)
    D && printindices(io,d,l,h,PRE)
    !((B || C || D) && A) && printindices(io,a,l,e,PRE)
    B && printindices(io,b,l,f,PRE)
end
@pure printindices(io::IO,V::T,e::Bits,label::Bool=false) where T<:Manifold = printlabel(io,V,e,label,namelist(V)...)

@inline function printlabel(io::IO,V::T,e::Bits,label::Bool,vec,cov,duo,dif) where T<:Manifold
    M = supermanifold(V)
    N,D,C,db = ndims(M),diffvars(M),dyadmode(V),diffmask(V)
    if C < 0
        es = e & (~(db[1]|db[2]))
        n = Int((N-2D)/2)
        eps = shift_indices(V,e & db[1]).-(N-2D-hasinf(M)-hasorigin(M))
        par = shift_indices(V,e & db[2]).-(N-D-hasinf(M)-hasorigin(M))
        printindices(io,shift_indices(V,es & Bits(2^n-1)),shift_indices(V,es>>n),eps,par,label,vec,cov,duo,dif)
    else
        es = e & (~db)
        eps = shift_indices(V,e & db).-(N-D-hasinf(M)-hasorigin(M))
        if !isempty(eps)
            printindices(io,shift_indices(V,es),Int[],C>0 ? Int[] : eps,C>0 ? eps : Int[],label,C>0 ? cov : vec,cov,C>0 ? dif : duo,dif)
        else
            printindices(io,shift_indices(V,es),label,C>0 ? string(cov) : vec)
        end
    end
    return io
end

@inline printlabel(V::T,e::Bits,label::Bool,vec,cov,duo,dif) where T<:Manifold = printlabel(IOBuffer(),V,e,label,vec,cov,duo,dif) |> take! |> String

@pure function indexstring(V::M,D) where M<:Manifold
    io = IOBuffer()
    printlabel(io,V,D,true,PRE...)
    String(take!(io))
end

@pure indexsymbol(V::M,D) where M<:Manifold = Symbol(indexstring(V,D))

@pure indexsplit(B,N) = [UInt(1)<<(k-1) for k ∈ indices(B,N)]

function indexparity!(ind::Vector{Int},s::M) where M<:Manifold
    k = 1
    t = false
    while k < length(ind)
        if ind[k] == ind[k+1]
            ind[k] == 1 && hasinf(s) && (return t, ind, true)
            s[ind[k]] && (t = !t)
            deleteat!(ind,[k,k+1])
        elseif ind[k] > ind[k+1]
            ind[k:k+1] = ind[k+1:-1:k]
            t = !t
            k ≠ 1 && (k -= 1)
        else
            k += 1
        end
    end
    return t, ind, false
end

@noinline function indexparity(V::Manifold,v::Symbol)::Tuple{Bool,Vector,Manifold,Bool}
    vs = string(v)
    vt = vs[1:1]≠pre[1]
    Z=match(Regex("([$(pre[1])]([0-9a-vx-zA-VX-Z]+))?([$(pre[2])]([0-9a-zA-Z]+))?"),vs)
    ef = String[]
    for k ∈ (2,4)
        Z[k] ≠ nothing && push!(ef,Z[k])
    end
    length(ef) == 0 && (return false,Int[],V,true)
    let W = V,fs=false
        C = dyadmode(V)
        X = C≥0 && ndims(V)<4sizeof(Bits)+1
        X && (W = C>0 ? V'⊕V : V⊕V')
        V2 = (vt ⊻ (vt ? C≠0 : C>0)) ? V' : V
        L = length(ef) > 1
        M = X ? Int(ndims(W)/2) : ndims(W)
        m = ((!L) && vt && (C<0)) ? M : 0
        chars = (L || (Z[2] ≠ nothing)) ? alphanumv : alphanumw
        (es,e,et) = indexparity!([findfirst(isequal(ef[1][k]),chars) for k∈1:length(ef[1])].+m,C<0 ? V : V2)
        et && (return false,Int[],V,true)
        w,d = if L
            (fs,f,ft) = indexparity!([findfirst(isequal(ef[2][k]),alphanumw) for k∈1:length(ef[2])].+M,W)
            ft && (return false,Int[],V,true)
            W,[e;f]
        else
            V2,e
        end
        return es⊻fs, d, w, false
    end
end
