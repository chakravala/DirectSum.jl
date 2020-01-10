
#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

Bits = UInt

bit2int(b::BitArray{1}) = parse(UInt,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0,C=0) = (1<<(d-1))|(1<<(2*o-1))|(c<0 ? 8 : (1<<(3*c-1)))|(1<<(5*C-1))

const vio = ('∞','∅')

value(x::T) where T<:Number = x
signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-
-(x) = Base.:-(x)
-(x::Symbol) = :(-$x)
-(x::SArray) = Base.:-(x)
-(x::SArray{Tuple{M},T,1,M} where M) where T<:Any = broadcast(-,x)

for op ∈ (:conj,:inv,:sqrt,:abs,:exp,:expm1,:log,:log1p,:sin,:cos,:sinh,:cosh,:signbit)
    @eval @inline $op(z) = Base.$op(z)
end

for op ∈ (:/,:-,:^,:≈)
    @eval @inline $op(a,b) = Base.$op(a,b)
end

for (OP,op) ∈ ((:∏,:*),(:∑,:+))
    @eval begin
        @inline $OP(x...) = Base.$op(x...)
        @inline $OP(x::AbstractVector{T}) where T<:Any = $op(x...)
    end
end

const PROD,SUM,SUB = ∏,∑,-

@inline norm(z::Expr) = abs(z)
@inline norm(z::Symbol) = z
@inline norm(z) = LinearAlgebra.norm(z)
@inline norm(z::SArray{Tuple{M},Any,1,M} where M) = sqrt(SUM(z.^2...))

for T ∈ (Expr,Symbol)
    @eval begin
        ≈(a::$T,b::$T) = a == b
        ≈(a::$T,b) = false
        ≈(a,b::$T) = false
    end
end
