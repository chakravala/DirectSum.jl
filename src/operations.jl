
#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export ⊕

# direct sum ⨁

for op ∈ (:+,:⊕)
    @eval begin
        @pure function $op(a::VectorSpace{N,X,A},b::VectorSpace{M,Y,B}) where {N,X,A,M,Y,B}
            D1,O1,C1 = options_list(a)
            D2,O2,C2 = options_list(b)
            NM = N == M
            opt = if (D1,O1,C1,D2,O2,C2) == (0,0,0,0,0,0)
                doc2m(0,0,0)
            elseif (D1,O1,C1,D2,O2,C2) == (0,0,1,0,0,1)
                doc2m(0,0,1)
            elseif (D1,O1,C1,D2,O2,C2) == (0,0,0,0,0,1)
                doc2m(0,0,NM ? (B ≠ flip_sig(N,A) ? 0 : -1) : 0)
            elseif (D1,O1,C1,D2,O2,C2) == (0,0,1,0,0,0)
                doc2m(0,0,NM ? (A ≠ flip_sig(N,B) ? 0 : -1) : 0)
            else
                throw(error("arbitrary VectorSpace direct-sums not yet implemented"))
            end
            VectorSpace{N+M,opt,bit2int(BitArray([a[:]; b[:]]))}()
        end
    end
end
for M ∈ (0,4)
    @eval begin
        @pure function ^(v::VectorSpace{N,$M,S},i::I) where {N,S,I<:Integer}
            let V = v
                for k ∈ 2:i
                    V = V⊕v
                end
                return V
            end
        end
    end
end

## set theory ∪,∩,⊆,⊇

for op ∈ (:*,:∪)
    @eval begin
        @pure $op(a::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = a
        @pure function $op(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
            D1,O1,C1 = options_list(a)
            D2,O2,C2 = options_list(b)
            if ((C1≠C2)&&(C1≥0)&&(C2≥0)) && a==b'
                return C1>0 ? b⊕a : a⊕b
            elseif min(C1,C2)<0 && max(C1,C2)≥0
                Y = C1<0 ? b⊆a : a⊆b
                !Y && throw(error("VectorSpace union $(a)∪$(b) incompatible!"))
                return C1<0 ? a : b
            elseif ((N1,D1,O1)==(N2,D2,O2)) || (N1==N2)
                throw(error("VectorSpace intersection $(a)∩$(b) incompatible!"))
            else
                throw(error("arbitrary VectorSpace union not yet implemented."))
            end
        end
    end
end

∩(a::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = a
∩(a::VectorSpace{N},::VectorSpace{N}) where N = V0
@pure function ∩(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if ((C1≠C2)&&(C1≥0)&&(C2≥0))
        return V0
    elseif min(C1,C2)<0 && max(C1,C2)≥0
        Y = C1<0
        return (Y ? b⊕b' : a⊕a') == (Y ? a : b) ? Y ? b : a : V0
    else
        throw(error("arbitrary VectorSpace intersection not yet implemented."))
    end
end

@pure ⊇(a::VectorSpace,b::VectorSpace) = b ⊆ a
⊆(::VectorSpace{N,M,S},::VectorSpace{N,M,S}) where {N,M,S} = true
⊆(::VectorSpace{N},::VectorSpace{N}) where N = false
@pure function ⊆(a::VectorSpace{N1,M1,S1},b::VectorSpace{N2,M2,S2}) where {N1,M1,S1,N2,M2,S2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if ((C1≠C2)&&(C1≥0)&&(C2≥0)) || ((C1<0)&&(C2≥0))
        return false
    elseif C2<0 && C1≥0
        return (C1>0 ? a'⊕a : a⊕a') == b
    else
        throw(error("arbitrary VectorSpace subsets not yet implemented."))
    end
end
