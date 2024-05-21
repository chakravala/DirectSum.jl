
#   This file is part of DirectSum.jl
#   It is licensed under the AGPL license
#   DirectSum Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

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
            opt,sig = combine_options(a,b),[a[:];b[:]]
            DiagonalForm{N+M,opt,diagsig(opt,sig),F,D}()
        end
        @pure function $op(a::DiagonalForm{N,X,A,F,D},b::Signature{M,Y,B,F,D}) where {N,X,A,M,Y,B,F,D}
            opt,sig = combine_options(a,b),[a[:];[t ? -1 : 1 for t ∈ b[:]]]
            DiagonalForm{N+M,opt,diagsig(opt,sig),F,D}()
        end
        @pure function $op(a::Signature{N,X,A,F,D},b::DiagonalForm{M,Y,B,F,D}) where {N,X,A,M,Y,B,F,D}
            opt,sig = combine_options(a,b),[[t ? -1 : 1 for t ∈ a[:]];b[:]]
            DiagonalForm{N+M,opt,diagsig(opt,sig),F,D}()
        end
    end
end
@pure function ⊕(a::Submanifold{V,N,X},b::Submanifold{W,M,Y}) where {N,V,X,M,W,Y}
    Z = (isdual(V)==isdual(W))||(V≠W') ? combine(V,W,X,Y) : (mixed(V,X)|mixed(W,Y))
    A,B = typeof(V),typeof(W)
    VW = A<:Int ? B<:Int ? V+W : Signature(V)⊕W : B<:Int ? V⊕Signature(W) : V⊕W
    Submanifold{VW,count_ones(Z)}(Z)
end
for M ∈ (0,4)
    @eval begin
        @pure function Base.:^(v::T,i::I) where T<:TensorBundle{N,$M} where {N,I<:Integer}
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
@pure ∪(a::M,::Submanifold{m,Y,B}) where {m,Y,M<:TensorBundle,B} = a∪m
@pure ∪(::Submanifold{m,X,A},b::M) where {m,X,M<:TensorBundle,A} = m∪b
@pure ∪(::Submanifold{M,X,A} where X,::Submanifold{M,Y,B} where Y) where {M,A,B} = (C=A|B; Submanifold{M,count_ones(C)}(C))
@pure function ∪(a::Submanifold{N},b::Submanifold{M}) where {N,M}
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
        @pure ∩(A::$Bundle,b::Submanifold) = b⊆A ? b : V0
        @pure ∩(a::Submanifold,B::$Bundle) = a⊆B ? a : V0
    end
end
@pure ∩(::Submanifold{M,X,A} where X,::Submanifold{M,Y,B} where Y) where {M,A,B} = (C=A&B; Submanifold{M,count_ones(C)}(C))
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
        @pure ⊆(A::$Bundle,b::Submanifold{M,Y}) where {M,Y} = mdims(M) == Y ? A⊆M : throw(error("$A ⊆ $b not computable"))
        @pure ⊆(a::Submanifold{M,X,A} where X,B::$Bundle) where {A,M} = M⊆B
    end
end
@pure ⊆(::Submanifold{M,X,A},::Submanifold{M,Y,B} where Y) where {M,A,B,X} = count_ones(A&B) == X
@pure ⊆(a::Submanifold,b::Submanifold{M,Y}) where {M,Y} = mdims(M) == Y ? a⊆M : interop(⊆,a,b)
@pure ⊆(a::Submanifold{V},b::Int) where V = V ⊆ b
@pure ⊆(::Signature{A,M,S,D,O},::Signature{B,M,S,D,O}) where {A,B,M,S,D,O} = A≤B
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

## Basis forms

@pure evaluate1(a::A,b::B) where {A<:TensorTerm{V,1},B<:TensorTerm{V,1}} where V = evaluate1(V,UInt(a),UInt(b))
@pure evaluate1(V::T,A,B) where T<:TensorBundle = evaluate(Submanifold(V),A,B)
@pure function evaluate1(V,A::UInt,B::UInt)
    X = isdyadic(V) ? A>>Int(mdims(V)/2) : A
    B∉(A,X) ? (true,false) : (false,V[intlog(B)+1])
end
@pure function evaluate2(a::A,b::B) where {A<:TensorTerm{V,1},B<:TensorTerm{V,1}} where V
    ib,(m1,m2) = indexbasis(N,1),eval_shift(a)
    @inbounds v = ib[m2]
    UInt(b)≠v ? (true,false,UInt(0)) : (false,V[intlog(v)+1],ib[m1])
end
@pure eval_shift(t::T) where T<:TensorTerm = eval_shift(Manifold(t))
@pure function eval_shift(t::Submanifold)
    N = mdims(t)
    bi = indices(UInt(t),N)
    M = Int(N/2)
    @inbounds (bi[1], bi[2]>M ? bi[2]-M : bi[2])
end

@pure (W::Submanifold{Q,M})(b::Zero{V}) where {Q,M,V} = Zero(W)
@pure function (W::Submanifold{Q,M})(b::Submanifold{V,G,R}) where {Q,M,V,G,R}
    if isbasis(W) && !isbasis(b)
        RS = R&UInt(W)
        L = count_ones(RS)
        L == G ? b : Submanifold{V,L,RS}()
    elseif isbasis(W)
        if Q == V
            if G == M == 1
                y,v = evaluate1(W,b)
                y ? Zero(V) : v*Submanifold{V}()
            elseif G == 1 && M == 2
                (!isdyadic(V)) && throw(error("wrong basis"))
                y,v,B = evaluate2(W,b)
                y ? Zero(V) : v*getbasis(V,B)
            else
                throw(error("unsupported transformation"))
            end
        else
            return interform(W,b)
        end
    elseif V==W
        return Submanifold{Submanifold(W),G}(R)
    elseif W⊆V
        S = UInt(W)
        count_ones(R&S)==G ? getbasis(W,lowerbits(mdims(V),S,R)) : Zero(W)
    elseif V⊆W
        WC,VC = isdyadic(W),isdyadic(V)
        #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
        #    return V0
        B = isbasis(b) ? expandbits(mdims(W),UInt(V),R) : R
        if WC && (!VC)
            getbasis(W,mixed(V,B))
        elseif (!WC) && (!VC)
            getbasis(W,B)
        else
            throw(error("arbitrary Manifold intersection not yet implemented."))
        end
    elseif typeof(V)<:Int
        W(Submanifold{Signature(V),G,R}())
    else
        throw(error("cannot convert from $(V) to $(W)"))
    end
end

#(a::Submanifold{V})(b::T) where {V,T<:TensorAlgebra} = interform(a,b)
@pure (T::Signature{N,M,S,F,D})(::Signature{N,M,S,F,D}) where {N,M,S,F,D} = Submanifold(Submanifold(T))
@pure (W::Signature)(b::Submanifold{V,G,R}) where {V,G,R} = Submanifold(W)(b)

# Single forms

(a::Single)(b::T) where {T<:TensorAlgebra} = interform(a,b)
function (a::Submanifold{V,1})(b::Single{V,1}) where V
    y,v = evaluate1(a,b)
    y ? Zero(V) : (v*b.v)*Submanifold{V}()
end
function (a::Single{V,1})(b::Submanifold{V,1}) where V
    y,v = evaluate1(a,b)
    y ? Zero(V) : (v*a.v)*Submanifold{V}()
end
@eval begin
    function (a::Single{V,1})(b::Single{V,1}) where V
        $(insert_expr((:t,))...)
        y,v = evaluate1(a,b)
        y && (return Zero(V))
        y ? Zero(V) : Single{V}((v*a.v*b.v)::t,Submanifold{V}())
    end
end
function (a::Single{V,2})(b::Submanifold{V,1}) where V
    (!isdyadic(V)) && throw(error("wrong basis"))
    y,v,B = evaluate2(a,b)
    @inbounds y ? Zero(V) : (v*a.v)*getbasis(V,B)
end
function (a::Submanifold{V,2})(b::Single{V,1}) where V
    (!isdyadic(V)) && throw(error("wrong basis"))
    y,v,B = evaluate2(a,b)
    @inbounds y ? Zero(V) : (v*b.v)*getbasis(V,B)
end
function (a::Single{V,2})(b::Single{V,1}) where V
    (!isdyadic(V)) && throw(error("wrong basis"))
    y,v,B = evaluate2(a,b)
    @inbounds y ? Zero(V) : (v*a.v*b.v)*getbasis(V,B)
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
    for Q ∈ (:DiagonalForm,:Submanifold)
        @eval begin
            @pure function $p(V::$Q,B,G=count_ones(B))
                ind = indices(B&(UInt(1)<<(mdims(V)-diffvars(V))-1),mdims(V))
                $p(0,sum(ind),G,mdims(V)-diffvars(V)) ? -1 : 1
            end
            @pure function $pg(V::$Q,B,G=count_ones(B))
                ind = indices(B&(UInt(1)<<(mdims(V)-diffvars(V))-1),mdims(V))
                gg,c = V[ind], hasconformal(V) && (B&UInt(3) == UInt(2))
                g = isempty(gg) ? 1 : prod(signbool.(gg))
                $p(0,sum(ind),G,mdims(V)-diffvars(V))⊻c ? -(g) : g
            end
        end
    end
    for p ∈ (p,pg)
        @eval @pure $p(::Submanifold{V,G,B}) where {V,G,B} = $p(V,B,G)
    end
end

for Q ∈ (:DiagonalForm,:Submanifold)
    @eval begin
        @pure function paritymetric(V::$Q,B,G=count_ones(B))
            g = V[indices(B&(UInt(1)<<(mdims(V)-diffvars(V))-1),mdims(V))]
            isempty(g) ? 1 : prod(signbool.(g))
        end
        @pure function parityanti(V::$Q,B)
            paritymetric(V,complement(mdims(V),B,diffvars(V),hasinf(V)+hasorigin(V)))
        end
    end
end
@pure paritymetric(::Submanifold{V,G,B}) where {V,G,B} = paritymetric(V,B,G)
@pure parityanti(::Submanifold{V,G,B}) where {V,G,B} = parityanti(V,B)

## complement

import Leibniz: complementright, complementrighthodge, ⋆, complement
import AbstractTensors: complementleft, complementlefthodge, complementleftanti, complementrightanti, antimetric, pseudometric
export complementleft, complementright, ⋆, complementlefthodge, complementrighthodge
export complementleftanti, complementrightanti

@inline complementrightanti(t) = complementright(antimetric(t))
@inline complementleftanti(t) = complementleft(antimetric(t))

signbool(t::Bool) = t ? -1 : 1
signbool(t) = t

for side ∈ (:left,:right)
    s,p = Symbol(:complement,side),Symbol(:parity,side)
    h,pg,pn = Symbol(s,:hodge),Symbol(p,:hodge),Symbol(p,:null)
    for (c,p) ∈ ((s,p),(h,pg))
        @eval begin
            @pure function $c(b::Submanifold{V,G,B}) where {V,G,B}
                $(c≠h ? nothing : side≠:right ? :((!isdiag(V) && !hasconformal(V)) && (return $s(metric(b)))) : :((!isdiag(V) && !hasconformal(V)) && (return reverse(b)*V(LinearAlgebra.I))) )
                d = getbasis(V,complement(mdims(V),B,diffvars(V),$(c≠h ? 0 : :(hasinf(V)+hasorigin(V)))))
                isdyadic(V) && throw(error("Complement for mixed tensors is undefined"))
                v = $(c≠h ? :($pn(V,B,value(d))) : :(value(d)))
                typeof(V)<:Signature ? ($p(b) ? Single{V}(-v,d) : isone(v) ? d : Single{V}(v,d)) : Single{V}(signbool($p(b))*v,d)
            end
            $c(b::Single) = conj(value(b))*$c(basis(b))
        end
    end
end

@eval begin
    @pure function metric(b::Submanifold{V,G,B}) where {V,G,B}
        !isbasis(b) && (return metric(V))
        (!isdiag(V) || hasconformal(V)) && (return complementleft(complementrighthodge(b)))
        isdyadic(V) && throw(error("Complement for mixed tensors is undefined"))
        hasorigin(b) && !hasinf(b) && (return Zero(V))
        hasinf(b) && !hasorigin(b) && (return Zero(V))
        p = paritymetric(b)
        typeof(p)==Bool ? (p ? -b : b ) : Single{V}(p,b)
    end
    metric(b::Single) = value(b)*metric(basis(b))
    @pure function antimetric(b::Submanifold{V,G,B}) where {V,G,B}
        (!isdiag(V) || hasconformal(V)) && (return antimetric_term(b))
        isdyadic(V) && throw(error("Complement for mixed tensors is undefined"))
        hasorigin(b) && !hasinf(b) && (return Zero(V))
        hasinf(b) && !hasorigin(b) && (return Zero(V))
        p = parityanti(b)
        typeof(p)==Bool ? (p ? -b : b ) : Single{V}(p,b)
    end
    antimetric(b::Single) = value(b)*antimetric(basis(b))
end

# other

import Leibniz: parityinvolute, parityreverse

odd(t::T) where T<:TensorGraded{V,G} where {V,G} = parityinvolute(G) ? t : Zero(V)
even(t::T) where T<:TensorGraded{V,G} where {V,G} = parityinvolute(G) ? Zero(V) : t

"""
    imag(ω::TensorAlgebra)

The `imag` part `(ω-(~ω))/2` is defined by `abs2(imag(ω)) == -(imag(ω)^2)`.
"""
Base.imag(t::T) where T<:TensorGraded{V,G} where {V,G} = parityreverse(G) ? t : Zero(V)

"""
real(ω::TensorAlgebra)

The `real` part `(ω+(~ω))/2` is defined by `abs2(real(ω)) == real(ω)^2`.
"""
Base.real(t::T) where T<:TensorGraded{V,G} where {V,G} = parityreverse(G) ? Zero(V) : t

# QR compatibility

convert(::Type{Single{V,G,B,X}},t::Submanifold) where {V,G,B,X} = Single{V,G,B,X}(convert(X,value(t)))
convert(a::Type{Single{V,G,B,A}},b::Single{V,G,B,T}) where {V,G,A,B,T} = Single{V,G,B,A}(convert(A,value(b)))
convert(::Type{Single{V,G,B,X}},t::Y) where {V,G,B,X,Y} = Single{V,G,B,X}(convert(X,t))
Base.copysign(x::Single{V,G,B,T},y::Single{V,G,B,T}) where {V,G,B,T} = Single{V,G,B,T}(copysign(value(x),value(y)))
