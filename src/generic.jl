
#   This file is part of DirectSum.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

export basis, grade, hasinf, hasorigin, isorigin, scalar, norm, valuetype, indices, metric

(M::Signature)(b::Int...) = SubManifold{M}(b)
(M::DiagonalForm)(b::Int...) = SubManifold{M}(b)
(M::SubManifold)(b::Int...) = SubManifold{M}(b)

@pure Base.ndims(S::SubManifold{M,G}) where {G,M} = typeof(M)<:SubManifold ? ndims(M) : G
@pure grade(V::M) where M<:Manifold{N} where N = N-(mixedmode(V)<0 ? 2 : 1)*diffvars(V)
@pure grade(m::TensorGraded{V,G} where V) where G = G
@pure grade(m::Real) = 0
@pure order(m) = 0
@pure order(V::M) where M<:Manifold = diffvars(V)
@pure order(m::SubManifold{V,G,B} where G) where {V,B} = count_ones(symmetricmask(V,B,B)[4])
@pure order(m::Simplex) = order(basis(m))+order(value(m))
@pure options(::T) where T<:VectorBundle{N,M} where N where M = M
@pure options(::T) where T<:SubManifold{M} where M = options(M)
@pure options_list(V::M) where M<:Manifold = hasinf(V),hasorigin(V),mixedmode(V),polymode(V)
@pure metric(::T) where T<:VectorBundle{N,M,S} where {N,M} where S = S
@pure metric(::T) where T<:SubManifold{M} where M = metric(M)
@pure metric(V::Signature,b::Bits) = isodd(count_ones(metric(V)&b)) ? -1 : 1
@pure metric(V::M,b::Bits) where M<:Manifold = PROD(V[indices(b)])
@pure polymode(M::Int) = iszero(M&16)
@pure polymode(::T) where T<:VectorBundle{N,M} where N where M = polymode(M)
@pure mixedmode(M::Int) = M%16 ∈ 8:11 ? -1 : Int(M%16 ∈ (4,5,6,7))
@pure mixedmode(::T) where T<:VectorBundle{N,M} where N where M = mixedmode(M)
@pure mixedmode(::SubManifold{M}) where M = mixedmode(M)
@pure diffmode(::T) where T<:VectorBundle{N,M,S,F,D} where {N,M,S,F} where D = D
@pure diffmode(::SubManifold{M}) where M = diffmode(M)
@pure diffvars(::T) where T<:VectorBundle{N,M,S,F} where {N,M,S} where F = F
@pure function diffvars(::SubManifold{M,N,S}) where {M,N,S}
    n,C = ndims(M),diffmode(M)
    sum(in.(1+n-(C<0 ? 2 : 1)*diffvars(M):n,Ref(indices(S))))
end

@pure valuetype(::SubManifold) = Int
@pure valuetype(::Simplex{V,G,B,T} where {V,G,B}) where T = T
@inline value(::SubManifold,T=Int) = T==Any ? 1 : one(T)
@inline value(m::Simplex,T::DataType=valuetype(m)) = T∉(valuetype(m),Any) ? convert(T,m.v) : m.v
@inline value_diff(m::T) where T<:TensorTerm = (v=value(m);typeof(v)<:TensorAlgebra ? v : m)
@pure basis(m::SubManifold{V}) where V = typeof(V)<:SubManifold ? m : SubManifold(m)
@pure basis(m::Simplex{V,G,B}) where {V,G,B} = B
@pure UInt(m::T) where T<:TensorTerm = bits(basis(m))
@pure bits(m::T) where T<:TensorTerm = bits(basis(m))
@pure bits(b::SubManifold{V,G,B} where {V,G}) where B = B::UInt
@pure bits(::Type{SubManifold{V,G,B}}) where {V,G,B} = B
@pure det(s::Signature) = isodd(count_ones(metric(s))) ? -1 : 1
@pure det(s::DiagonalForm) = PROD(diagonalform(s))
@pure Base.abs(s::M) where M<:Manifold = sqrt(abs(det(s)))

@pure hasconformal(V) = hasinf(V) && hasorigin(V)
@pure hasinf(M::Int) = M%16 ∈ (1,3,5,7,9,11)
@pure hasinf(::T) where T<:VectorBundle{N,M} where N where M = hasinf(M)
@pure hasinf(::SubManifold{M,N,S} where N) where {M,S} = hasinf(M) && isodd(S)
@pure hasorigin(M::Int) = M%16 ∈ (2,3,6,7,10,11)
@pure hasorigin(::T) where T<:VectorBundle{N,M} where N where M = hasorigin(M)
@pure hasorigin(m::TensorAlgebra) = hasorigin(Manifold(m))
@pure hasorigin(V::SubManifold{M,N,S} where N) where {M,S} = hasorigin(M) && (hasinf(M) ? (d=UInt(2);(d&S)==d) : isodd(S))
@pure hasorigin(t::Simplex) = hasorigin(basis(t))
@pure Base.isinf(e::SubManifold{V}) where V = hasinf(e) && count_ones(bits(e)) == 1
@pure isorigin(e::SubManifold{V}) where V = hasorigin(V) && count_ones(bits(e))==1 && e[hasinf(V)+1]

@pure hasorigin(V::M, B::T) where {M<:Manifold,T<:Bits} = hasinf(V) ? (Bits(2)&B)==Bits(2) : isodd(B)

@pure function hasinf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && (isodd(A) || isodd(B))
end
@pure function hasorigin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && (hasorigin(V,A) || hasorigin(V,B))
end

@pure function hasinf2(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && isodd(A) && isodd(B)
end
@pure function hasorigin2(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && hasorigin(V,A) && hasorigin(V,B)
end

@pure function hasorigininf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && hasorigin(V,A) && isodd(B) && !hasorigin(V,B) && !isodd(A)
end
@pure function hasinforigin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasconformal(V) && isodd(A) && hasorigin(V,B) && !isodd(B) && !hasorigin(V,A)
end

@pure function hasinf2origin(V::T,A::Bits,B::Bits) where T<:Manifold
    hasinf2(V,A,B) && hasorigin(V,A,B)
end
@pure function hasorigin2inf(V::T,A::Bits,B::Bits) where T<:Manifold
    hasorigin2(V,A,B) && hasinf(V,A,B)
end

@pure function diffmask(V::M) where M<:Manifold
    d = diffvars(V)
    if mixedmode(V)<0
        v = ((one(Bits)<<d)-1)<<(ndims(V)-2d)
        w = ((one(Bits)<<d)-1)<<(ndims(V)-d)
        return d<0 ? (typemax(Bits)-v,typemax(Bits)-w) : (v,w)
    end
    v = ((one(Bits)<<d)-1)<<(ndims(V)-d)
    d<0 ? typemax(Bits)-v : v
end

@pure function symmetricsplit(V::M,a) where M<:Manifold
    sm,dm = symmetricmask(V,a),diffmask(V)
    mixedmode(V)<0 ? (sm&dm[1],sm&dm[2]) : sm
end

@pure function symmetricmask(V::M,a) where M<:Manifold
    d = diffmask(V)
    a&(mixedmode(V)<0 ? |(d...) : d)
end

@pure function symmetricmask(V::M,a,b) where M<:Manifold
    d = diffmask(V)
    D = mixedmode(V)<0 ? |(d...) : d
    aD,bD = (a&D),(b&D)
    return a&~D, b&~D, aD|bD, aD&bD
end

@pure function diffcheck(V::M,A::Bits,B::Bits) where M<:Manifold
    d,db = diffvars(V),diffmask(V)
    v = mixedmode(V)<0 ? db[1]|db[2] : db
    hi = hasinf2(V,A,B) && !hasorigin(V,A,B)
    ho = hasorigin2(V,A,B) && !hasinf(V,A,B)
    (hi || ho) || (d≠0 && count_ones(A&v)+count_ones(B&v)>diffmode(V))
end

## functors

@pure tangent(s::Signature{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = Signature{N+(mixedmode(s)<0 ? 2f : f),M,S,f,D+d}()
@pure tangent(s::DiagonalForm{N,M,S,F,D},d::Int=1,f::Int=F≠0 ? F : 1) where {N,M,S,F,D} = DiagonalForm{N+(mixedmode(s)<0 ? 2f : f),M,S,f,D+d}()

@pure subtangent(V) = V(grade(V)+1:ndims(V)...)

for M ∈ (:Signature,:DiagonalForm)
    @eval @pure loworder(V::$M{N,M,S,D,O}) where {N,M,S,D,O} = O≠0 ? $M{N,M,S,D,O-1}() : V
end
@pure loworder(::SubManifold{M,N,S}) where {N,M,S} = SubManifold{loworder(M),N,S}()

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

@pure function Base.adjoint(V::SubManifold{M,N,S}) where {N,M,S}
    C = mixedmode(V)
    C < 0 && throw(error("$V is the direct sum of a vector space and its dual space"))
    SubManifold{M',N,S}()
end

# conversions

@pure Manifold(V::SubManifold{M}) where M = typeof(M)<:SubManifold ? M : V
@pure Signature(V::SubManifold{M,N} where M) where N = Signature{N,options(V)}(Vector(signbit.(V[:])),diffvars(V),diffmode(V))
@pure Signature(V::DiagonalForm{N,M}) where {N,M} = Signature{N,M}(Vector(signbit.(V[:])))
@pure DiagonalForm(V::Signature{N,M}) where {N,M} = DiagonalForm{N,M}([t ? -1 : 1 for t∈V[:]])

@pure function mixed(V::M,ibk::UInt) where M<:Manifold
    N,D,VC = ndims(V),diffvars(V),mixedmode(V)
    return if D≠0
        A,B = ibk&(UInt(1)<<(N-D)-1),ibk&diffmask(V)
        VC>0 ? (A<<(N-D))|(B<<N) : A|(B<<(N-D))
    else
        VC>0 ? ibk<<N : ibk
    end
end

@pure function (W::SubManifold{V,M,S})(b::SubManifold{V,G,B}) where {M,V,S,G,B}
    count_ones(B&S)==G ? getbasis(W,lowerbits(ndims(V),S,B)) : g_zero(W)
end
@pure function (a::SubManifold{W,M,S})(b::SubManifold{V,G,R}) where {M,V,S,G,W,R}
    V==W && (return SubManifold{SubManifold(W),G}(R))
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    B = typeof(V)<:SubManifold ? expandbits(ndims(W),bits(V),R) : R
    if WC<0 && VC≥0
        getbasis(W,mixed(V,B))
    elseif WC≥0 && VC≥0
        getbasis(W,B)
    else
        throw(error("arbitrary Manifold intersection not yet implemented."))
    end
    #interform(a,b)
end

@pure (T::Signature{N,M,S,F,D})(::Signature{N,M,S,F,D}) where {N,M,S,F,D} = SubManifold(SubManifold(T))
@pure function (W::Signature)(::SubManifold{V,G,R}) where {V,G,R}
    V==W && (return SubManifold{SubManifold(W),G}(R))
    !(V⊆W) && throw(error("cannot convert from $(V) to $(W)"))
    WC,VC = mixedmode(W),mixedmode(V)
    #if ((C1≠C2)&&(C1≥0)&&(C2≥0))
    #    return V0
    B = typeof(V)<:SubManifold ? expandbits(ndims(W),subvert(V),R) : R
    if WC<0 && VC≥0
        C = mixed(V,B)
        #getbasis(W,mixed(V,B))
        SubManifold{SubManifold(W),coun_ones(C)}(C)
    elseif WC≥0 && VC≥0
        #getbasis(W,B)
        SubManifold{SubManifold(W),count_ones(B)}(B)
    else
        throw(error("arbitrary Manifold intersection not yet implemented."))
    end
end


