
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export ⊕, χ, gdims

# direct sum ⨁

@pure function combine_options(a::T,b::S) where {T<:TensorBundle{N,X,A},S<:TensorBundle{M,Y,B}} where {N,X,A,M,Y,B}
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
        throw(error("arbitrary TensorBundle direct-sums not yet implemented"))
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
                throw(error("arbitrary TensorBundle direct-sums not yet implemented"))
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
@pure function ⊕(a::SubManifold{V,N,X},b::SubManifold{W,M,Y}) where {N,V,X,M,W,Y}
    Z = (isdual(V)==isdual(W))||(V≠W') ? combine(V,W,X,Y) : (mixed(V,X)|mixed(W,Y))
    SubManifold{V⊕W,count_ones(Z)}(Z)
end
for M ∈ (0,4)
    @eval begin
        @pure function Base.:^(v::T,i::I) where T<:TensorBundle{N,$M,S} where {N,S,I<:Integer}
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

@pure ∪(a::T,::Q) where {T<:TensorBundle{N,M,S},Q<:TensorBundle{N,M,S}} where {N,M,S} = a
@pure ∪(a::M,::SubManifold{m,Y,B}) where {m,Y,M<:TensorBundle,A,B} = a∪m
@pure ∪(::SubManifold{m,X,A},b::M) where {m,X,M<:TensorBundle,A,B} = m∪b
@pure ∪(::SubManifold{M,X,A} where X,::SubManifold{M,Y,B} where Y) where {M,A,B} = (C=A|B; SubManifold{M,count_ones(C)}(C))
@pure function ∪(a::SubManifold{N},b::SubManifold{M}) where {N,M}
    ma,mb = dyadmode(a),dyadmode(b)
    mc = ma == mb
    (mc ? a⊆b : (mb<0 && b(a)⊆b)) ? b : ((mc ? b⊆a  : (ma<0 && a(b)⊆a)) ? a : ma>0 ? b⊕a : a⊕b)
end
@pure function ∪(a::T,b::S) where {T<:TensorBundle{N1,M1,S1,V1,d1},S<:TensorBundle{N2,M2,S2,V2,d2}} where {N1,M1,S1,V1,d1,N2,M2,S2,V2,d2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if (M1,S1) == (M2,S2) && (V1,d1) ≠ (V2,d2)
        (T<:Signature ? Signature : DiagnoalForm){max(N1,N2),M1,S1,max(V1,V2),max(d1,d2)}()
    elseif ((C1≠C2)&&(C1≥0)&&(C2≥0)) && a==b'
        return C1>0 ? b⊕a : a⊕b
    elseif min(C1,C2)<0 && max(C1,C2)≥0
        Y = C1<0 ? b⊆a : a⊆b
        !Y && throw(error("TensorBundle union $(a)∪$(b) incompatible!"))
        return C1<0 ? a : b
    elseif ((N1,D1,O1)==(N2,D2,O2)) || (N1==N2)
        throw(error("TensorBundle intersection $(a)∩$(b) incompatible!"))
    else
        throw(error("arbitrary TensorBundle union not yet implemented."))
    end
end

@pure ∩(a::T,::Q) where {T<:TensorBundle{N,M,S},Q<:TensorBundle{N,M,S}} where {N,M,S} = a
@pure ∩(a::T,::S) where {T<:TensorBundle{N},S<:TensorBundle{N}} where N = V0
for Bundle ∈ (:Signature,:DiagonalForm)
    @eval begin
        @pure ∩(A::$Bundle,b::SubManifold) = b⊆A ? b : V0
        @pure ∩(a::SubManifold,B::$Bundle) = a⊆B ? a : V0
    end
end
@pure ∩(::SubManifold{M,X,A} where X,::SubManifold{M,Y,B} where Y) where {M,A,B} = (C=A&B; SubManifold{M,count_ones(C)}(C))
@pure function ∩(a::T,b::S) where {T<:TensorBundle{N1,M1,S1,V1,d1},S<:TensorBundle{N2,M2,S2,V2,d2}} where {N1,M1,S1,V1,d1,N2,M2,S2,V2,d2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if (M1,S1) == (M2,S2) && (V1,d1) ≠ (V2,d2)
        (T<:Signature ? Signature : DiagnoalForm){min(N1,N2),M1,S1,min(V1,V2),min(d1,d2)}()
    elseif ((C1≠C2)&&(C1≥0)&&(C2≥0))
        return V0
    elseif min(C1,C2)<0 && max(C1,C2)≥0
        Y = C1<0
        return (Y ? b⊕b' : a⊕a') == (Y ? a : b) ? Y ? b : a : V0
    else
        throw(error("arbitrary TensorBundle intersection not yet implemented."))
    end
end

@pure ⊇(a::T,b::S) where {T<:TensorBundle,S<:TensorBundle} = b ⊆ a
@pure ⊆(::T,::Q) where {T<:TensorBundle{N,M,S},Q<:TensorBundle{N,M,S}} where {N,M,S} = true
@pure ⊆(::T,::S) where {T<:TensorBundle{N},S<:TensorBundle{N}} where N = false

for Bundle ∈ (:Signature,:DiagonalForm)
    @eval begin
        @pure ⊆(A::$Bundle,b::SubManifold{M,Y}) where {M,Y} = mdims(M) == Y ? A⊆M : throw(error("$A ⊆ $b not computable"))
        @pure ⊆(a::SubManifold{M,X,A} where X,B::$Bundle) where {A,M} = M⊆B
    end
end
@pure ⊆(::SubManifold{M,X,A},::SubManifold{M,Y,B} where Y) where {M,A,B,X} = count_ones(A&B) == X
@pure ⊆(a::SubManifold,b::SubManifold{M,Y}) where {M,Y} = mdims(M) == Y ? a⊆M : interop(⊆,a,b)
@pure function ⊆(a::T,b::S) where {T<:TensorBundle{N1,M1,S1,V1,d1},S<:TensorBundle{N2,M2,S2,V2,d2}} where {N1,M1,S1,V1,d1,N2,M2,S2,V2,d2}
    D1,O1,C1 = options_list(a)
    D2,O2,C2 = options_list(b)
    if (M1,S1) == (M2,S2) && (V1,d1) ≠ (V2,d2)
        V1 ≤ V2 && d1 ≤ d2
    elseif ((C1≠C2)&&(C1≥0)&&(C2≥0)) || ((C1<0)&&(C2≥0))
        return false
    elseif C2<0 && C1≥0
        return (C1>0 ? a'⊕a : a⊕a') == b
    else
        throw(error("arbitrary TensorBundle subsets not yet implemented."))
    end
end

# conversions

@pure Manifold(V::SubManifold{M}) where M = (t=typeof(M);t<:SubManifold||t<:Int ? M : V)
@pure Signature(V::SubManifold{M,N} where M) where N = Signature{N,options(V)}(Vector(signbit.(V[:])),diffvars(V),diffmode(V))
@pure Signature(V::DiagonalForm{N,M}) where {N,M} = Signature{N,M}(Vector(signbit.(V[:])))
@pure DiagonalForm(V::Signature{N,M}) where {N,M} = DiagonalForm{N,M}([t ? -1 : 1 for t∈V[:]])

for M ∈ (:Signature,:DiagonalForm,:SubManifold)
    @eval begin
        @inline (V::$M)(s::LinearAlgebra.UniformScaling{T}) where T = Simplex{V}(T<:Bool ? (s.λ ? 1 : -1) : s.λ,getbasis(V,(one(T)<<(mdims(V)-diffvars(V)))-1))
        (W::$M)(b::Simplex) = Simplex{W}(value(b),W(basis(b)))
    end
end

#@pure supblade(N,S,B) = bladeindex(N,expandbits(N,S,B))
#@pure supmulti(N,S,B) = basisindex(N,expandbits(N,S,B))

## Basis forms

@pure evaluate1(a::A,b::B) where {A<:TensorTerm{V,1},B<:TensorTerm{V,1}} where V = evaluate1(V,bits(a),bits(b))
@pure evaluate1(V::T,A,B) where T<:TensorBundle = evaluate(SubManifold(V),A,B)
@pure function evaluate1(V,A::UInt,B::UInt)
    X = isdyadic(V) ? A>>Int(mdims(V)/2) : A
    B∉(A,X) ? (true,false) : (false,V[intlog(B)+1])
end
@pure function evaluate2(a::A,b::B) where {A<:TensorTerm{V,1},B<:TensorTerm{V,1}} where V
    ib,(m1,m2) = indexbasis(N,1),eval_shift(a)
    @inbounds v = ib[m2]
    bits(b)≠v ? (true,false,UInt(0)) : (false,V[intlog(v)+1],ib[m1])
end
@pure eval_shift(t::T) where T<:TensorTerm = eval_shift(Manifold(t))
@pure function eval_shift(t::SubManifold)
    N = mdims(t)
    bi = indices(bits(t),N)
    M = Int(N/2)
    @inbounds (bi[1], bi[2]>M ? bi[2]-M : bi[2])
end

@pure function (W::SubManifold{Q,M})(b::SubManifold{V,G,R}) where {Q,M,V,G,R}
    if isbasis(W)
        if Q == V
            if G == M == 1
                y,v = evaluate1(W,b)
                y ? g_zero(V) : v*SubManifold{V}()
            elseif G == 1 && M == 2
                (!isdyadic(V)) && throw(error("wrong basis"))
                y,v,B = evaluate2(W,b)
                y ? g_zero(V) : v*getbasis(V,B)
            else
                throw(error("unsupported transformation"))
            end
        else
            return interform(W,b)
        end
    elseif V==W
        return SubManifold{SubManifold(W),G}(R)
    elseif W⊆V
        S = bits(W)
        count_ones(R&S)==G ? getbasis(W,lowerbits(mdims(V),S,R)) : g_zero(W)
    elseif V⊆W
        WC,VC = isdyadic(W),isdyadic(V)
        #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
        #    return V0
        B = isbasis(b) ? expandbits(mdims(W),bits(V),R) : R
        if WC && (!VC)
            getbasis(W,mixed(V,B))
        elseif (!WC) && (!VC)
            getbasis(W,B)
        else
            throw(error("arbitrary Manifold intersection not yet implemented."))
        end
    else
        throw(error("cannot convert from $(V) to $(W)"))
    end
end

#(a::SubManifold{V})(b::T) where {V,T<:TensorAlgebra} = interform(a,b)
@pure (T::Signature{N,M,S,F,D})(::Signature{N,M,S,F,D}) where {N,M,S,F,D} = SubManifold(SubManifold(T))
@pure (W::Signature)(b::SubManifold{V,G,R}) where {V,G,R} = SubManifold(W)(b)

# Simplex forms

(a::Simplex)(b::T) where {T<:TensorAlgebra} = interform(a,b)
function (a::SubManifold{V,1})(b::Simplex{V,1}) where V
    y,v = evaluate1(a,b)
    y ? g_zero(V) : (v*b.v)*SubManifold{V}()
end
function (a::Simplex{V,1})(b::SubManifold{V,1}) where V
    y,v = evaluate1(a,b)
    y ? g_zero(V) : (v*a.v)*SubManifold{V}()
end
@eval begin
    function (a::Simplex{V,1})(b::Simplex{V,1}) where V
        $(insert_expr((:t,))...)
        y,v = evaluate1(a,b)
        y && (return g_zero(V))
        y ? g_zero(V) : Simplex{V}((v*a.v*b.v)::t,SubManifold{V}())
    end
end
function (a::Simplex{V,2})(b::SubManifold{V,1}) where V
    (!isdyadic(V)) && throw(error("wrong basis"))
    y,v,B = evaluate2(a,b)
    @inbounds y ? g_zero(V) : (v*a.v)*getbasis(V,B)
end
function (a::SubManifold{V,2})(b::Simplex{V,1}) where V
    (!isdyadic(V)) && throw(error("wrong basis"))
    y,v,B = evaluate2(a,b)
    @inbounds y ? g_zero(V) : (v*b.v)*getbasis(V,B)
end
function (a::Simplex{V,2})(b::Simplex{V,1}) where V
    (!isdyadic(V)) && throw(error("wrong basis"))
    y,v,B = evaluate2(a,b)
    @inbounds y ? g_zero(V) : (v*a.v*b.v)*getbasis(V,B)
end

## complement parity

for side ∈ (:left,:right)
    p = Symbol(:parity,side)
    pg = Symbol(p,:hodge)
    pn = Symbol(p,:null)
    pnp = Symbol(pn,:pre)
    #=@eval begin
        @pure function $p(V::Signature,B,G=count_ones(B))
            b = B&(UInt(1)<<(mdims(V)-diffvars(V))-1)
            $p(0,sum(indices(b,mdims(V))),count_ones(b),mdims(V)-diffvars(V))
        end
        @pure function $pg(V::Signature,B,G=count_ones(B))
            o = hasorigin(V) && hasinf(V) && (iszero(B&UInt(1))&(!iszero(B&UInt(2))))
            b = B&(UInt(1)<<(mdims(V)-diffvars(V))-1)
            $pg(count_ones(metric(V)&b),sum(indices(b,mdims(V))),count_ones(b),mdims(V)-diffvars(V))⊻o
        end
    end=#
    for Q ∈ (:DiagonalForm,:SubManifold)
        @eval begin
            @pure function $p(V::$Q,B,G=count_ones(B))
                ind = indices(B&(UInt(1)<<(mdims(V)-diffvars(V))-1),mdims(V))
                $p(0,sum(ind),G,mdims(V)-diffvars(V)) ? -1 : 1
            end
            @pure function $pg(V::$Q,B,G=count_ones(B))
                ind = indices(B&(UInt(1)<<(mdims(V)-diffvars(V))-1),mdims(V))
                g,c = prod(V[ind]), hasconformal(V) && (B&UInt(3) == UInt(2))
                $p(0,sum(ind),G,mdims(V)-diffvars(V))⊻c ? -(g) : g
            end
        end
    end
    for p ∈ (p,pg)
        @eval @pure $p(::SubManifold{V,G,B}) where {V,G,B} = $p(V,B,G)
    end
end

## complement

import Leibniz: complementright, complementrighthodge, ⋆, complement
import AbstractTensors: complementleft, complementlefthodge
export complementleft, complementright, ⋆, complementlefthodge, complementrighthodge

for side ∈ (:left,:right)
    s,p = Symbol(:complement,side),Symbol(:parity,side)
    h,pg,pn = Symbol(s,:hodge),Symbol(p,:hodge),Symbol(p,:null)
    for (c,p) ∈ ((s,p),(h,pg))
        @eval begin
            @pure function $c(b::SubManifold{V,G,B}) where {V,G,B}
                d = getbasis(V,complement(mdims(V),B,diffvars(V),$(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))))
                isdyadic(V) && throw(error("Complement for mixed tensors is undefined"))
                v = $(c≠h ? :($pn(V,B,value(d))) : :(value(d)))
                typeof(V)<:Signature ? ($p(b) ? Simplex{V}(-v,d) : isone(v) ? d : Simplex{V}(v,d)) : Simplex{V}($p(b)*v,d)
            end
            $c(b::Simplex) = value(b)≠0 ? value(b)*$c(basis(b)) : g_zero(Manifold(b))
        end
    end
end

# QR compatibility

convert(::Type{Simplex{V,G,B,X}},t::SubManifold) where {V,G,B,X} = Simplex{V,G,B,X}(convert(X,value(t)))
convert(a::Type{Simplex{V,G,B,A}},b::Simplex{V,G,B,T}) where {V,G,A,B,T} = Simplex{V,G,B,A}(convert(A,value(b)))
convert(::Type{Simplex{V,G,B,X}},t::Y) where {V,G,B,X,Y} = Simplex{V,G,B,X}(convert(X,t))
Base.copysign(x::Simplex{V,G,B,T},y::Simplex{V,G,B,T}) where {V,G,B,T} = Simplex{V,G,B,T}(copysign(value(x),value(y)))
