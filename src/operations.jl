
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export ⊕

# direct sum ⨁

@pure function combine_options(a::T,b::S) where {T<:VectorBundle{N,X,A},S<:VectorBundle{M,Y,B}} where {N,X,A,M,Y,B}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    ds = (N == M) && (A == B)
    if (D1,O1,C1,D2,O2,C2) == (0,0,0,0,0,0)
        doc2m(0,0,0)
    elseif (D1,O1,C1,D2,O2,C2) == (0,0,1,0,0,1)
        doc2m(0,0,1)
    elseif (D1,O1,C1,D2,O2,C2) == (0,0,0,0,0,1)
        doc2m(0,0,ds ? -1 : 0)
    elseif (D1,O1,C1,D2,O2,C2) == (0,0,1,0,0,0)
        doc2m(0,0,ds ? -1 : 0)
    else
        throw(error("arbitrary VectorBundle direct-sums not yet implemented"))
    end
end

for op ∈ (:+,:⊕)
    @eval begin
        @pure function $op(a::T,b::S) where {T<:Signature{N,X,A,F,D},S<:Signature{M,Y,B,F,D}} where {N,X,A,M,Y,B,F,D}
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
                throw(error("arbitrary VectorBundle direct-sums not yet implemented"))
            end
            Signature{N+M,opt,bit2int(BitArray([a[:]; b[:]])),F,D}()
        end
        @pure function $op(a::DiagonalForm{N,X,A,F,D},b::DiagonalForm{M,Y,B,F,D}) where {N,X,A,M,Y,B,F,D}
            DiagonalForm{N+M,combine_options(a,b),F,D}([a[:];b[:]])
        end
        @pure function $op(a::DiagonalForm{N,X,A,F,D},b::Signature{M,Y,B,F,D}) where {N,X,A,M,Y,B,F,D}
            DiagonalForm{N+M,combine_options(a,b),F,D}([a[:];[t ? -1 : 1 for t ∈ b[:]]])
        end
        @pure function $op(a::Signature{N,X,A,F,D},b::DiagonalForm{M,Y,B,F,D}) where {N,X,A,M,Y,B,F,D}
            DiagonalForm{N+M,combine_options(a,b),F,D}([[t ? -1 : 1 for t ∈ a[:]];b[:]])
        end
    end
end
@pure function ⊕(a::SubManifold{N,V,X},b::SubManifold{M,W,Y}) where {N,V,X,M,W,Y}
    V ≠ W' && throw(error("$V ≠ $W'"))
    VW,Z = V⊕W,mixed(V,X)|mixed(W,Y)
    SubManifold{count_ones(Z),VW}(Z)
end
for M ∈ (0,4)
    @eval begin
        @pure function ^(v::T,i::I) where T<:VectorBundle{N,$M,S} where {N,S,I<:Integer}
            iszero(i) && (return V0)
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
        @pure $op(a::T,::Q) where {T<:VectorBundle{N,M,S},Q<:VectorBundle{N,M,S}} where {N,M,S} = a
        @pure $op(a::M,::SubManifold{Y,m,B}) where {Y,m,M<:VectorBundle,A,B} = a∪m
        @pure $op(::SubManifold{X,m,A},b::M) where {X,m,M<:VectorBundle,A,B} = m∪b
        @pure $op(::SubManifold{X,M,A} where X,::SubManifold{Y,M,B} where Y) where {M,A,B} = (C=A|B; SubManifold{count_ones(C),M}(C))
        @pure $op(a::SubManifold{X,N} where X,b::SubManifold{Y,M} where Y) where {N,M} = a⊆b ? b : (b⊆a ? a : a⊕b)
        @pure function $op(a::T,b::S) where {T<:VectorBundle{N1,M1,S1,V1,d1},S<:VectorBundle{N2,M2,S2,V2,d2}} where {N1,M1,S1,V1,d1,N2,M2,S2,V2,d2}
            D1,O1,C1 = options_list(a)
            D2,O2,C2 = options_list(b)
            if (M1,S1) == (M2,S2) && (V1,d1) ≠ (V2,d2)
                (T<:Signature ? Signature : DiagnoalManifold){max(N1,N2),M1,S1,max(V1,V2),max(d1,d2)}()
            elseif ((C1≠C2)&&(C1≥0)&&(C2≥0)) && a==b'
                return C1>0 ? b⊕a : a⊕b
            elseif min(C1,C2)<0 && max(C1,C2)≥0
                Y = C1<0 ? b⊆a : a⊆b
                !Y && throw(error("VectorBundle union $(a)∪$(b) incompatible!"))
                return C1<0 ? a : b
            elseif ((N1,D1,O1)==(N2,D2,O2)) || (N1==N2)
                throw(error("VectorBundle intersection $(a)∩$(b) incompatible!"))
            else
                throw(error("arbitrary VectorBundle union not yet implemented."))
            end
        end
    end
end

@pure ∪(x::T) where T<:Manifold = x
∪(a::A,b::B,c::C...) where {A<:Manifold,B<:Manifold,C<:Manifold} = ∪(a*b,c...)

@pure ∩(a::T,::Q) where {T<:VectorBundle{N,M,S},Q<:VectorBundle{N,M,S}} where {N,M,S} = a
@pure ∩(a::T,::S) where {T<:VectorBundle{N},S<:VectorBundle{N}} where N = V0
for Bundle ∈ (:Signature,:DiagonalForm)
    @eval begin
        @pure ∩(A::$Bundle,b::SubManifold) = b⊆A ? b : V0
        @pure ∩(a::SubManifold,B::$Bundle) = a⊆B ? a : V0
    end
end
@pure ∩(::SubManifold{X,M,A} where X,::SubManifold{Y,M,B} where Y) where {M,A,B} = (C=A&B; SubManifold{count_ones(C),M}(C))
@pure function ∩(a::T,b::S) where {T<:VectorBundle{N1,M1,S1,V1,d1},S<:VectorBundle{N2,M2,S2,V2,d2}} where {N1,M1,S1,V1,d1,N2,M2,S2,V2,d2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if (M1,S1) == (M2,S2) && (V1,d1) ≠ (V2,d2)
        (T<:Signature ? Signature : DiagnoalManifold){min(N1,N2),M1,S1,min(V1,V2),min(d1,d2)}()
    elseif ((C1≠C2)&&(C1≥0)&&(C2≥0))
        return V0
    elseif min(C1,C2)<0 && max(C1,C2)≥0
        Y = C1<0
        return (Y ? b⊕b' : a⊕a') == (Y ? a : b) ? Y ? b : a : V0
    else
        throw(error("arbitrary VectorBundle intersection not yet implemented."))
    end
end

∩(x::T) where T<:Manifold = x
∩(a::A,b::B,c::C...) where {A<:Manifold,B<:Manifold,C<:Manifold} = ∩(a∩b,c...)

@pure ⊇(a::T,b::S) where {T<:VectorBundle,S<:VectorBundle} = b ⊆ a
@pure ⊆(::T,::Q) where {T<:VectorBundle{N,M,S},Q<:VectorBundle{N,M,S}} where {N,M,S} = true
@pure ⊆(::T,::S) where {T<:VectorBundle{N},S<:VectorBundle{N}} where N = false

for Bundle ∈ (:Signature,:DiagonalForm)
    @eval begin
        @pure ⊆(A::$Bundle,b::SubManifold{Y,M,B}) where {B,M,Y} = M⊆A && ndims(A) == Y
        @pure ⊆(a::SubManifold{X,M,A} where X,B::$Bundle) where {A,M} = M⊆B
    end
end
@pure ⊆(::SubManifold{X,M,A},::SubManifold{Y,M,B} where Y) where {M,A,B,X} = count_ones(A&B) == X
@pure ⊆(a::SubManifold,b::SubManifold) = interop(⊆,a,b)
@pure function ⊆(a::T,b::S) where {T<:VectorBundle{N1,M1,S1,V1,d1},S<:VectorBundle{N2,M2,S2,V2,d2}} where {N1,M1,S1,V1,d1,N2,M2,S2,V2,d2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if (M1,S1) == (M2,S2) && (V1,d1) ≠ (V2,d2)
        V1 ≤ V2 && d1 ≤ d2
    elseif ((C1≠C2)&&(C1≥0)&&(C2≥0)) || ((C1<0)&&(C2≥0))
        return false
    elseif C2<0 && C1≥0
        return (C1>0 ? a'⊕a : a⊕a') == b
    else
        throw(error("arbitrary VectorBundle subsets not yet implemented."))
    end
end
