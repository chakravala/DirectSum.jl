
#   This file is part of DirectSum.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

Bits = UInt

bit2int(b::BitArray{1}) = parse(UInt,join(reverse([t ? '1' : '0' for t ∈ b])),base=2)

@pure doc2m(d,o,c=0) = (1<<(d-1))+(1<<(2*o-1))+(c<0 ? 8 : (1<<(3*c-1)))

const vio = ('∞','∅')

value(x::T) where T<:Number = x
signbit(x...) = Base.signbit(x...)
signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-
conj(z) = Base.conj(z)
inv(z) = Base.inv(z)
/(a,b) = Base.:/(a,b)
-(x) = Base.:-(x)
-(a,b) = Base.:-(a,b)
-(x::Symbol) = :(-$x)
-(x::SArray) = Base.:-(x)
-(x::SArray{Tuple{M},T,1,M} where M) where T<:Any = broadcast(-,x)

for (OP,op) ∈ ((:∏,:*),(:∑,:+))
    @eval begin
        $OP(x...) = Base.$op(x...)
        $OP(x::AbstractVector{T}) where T<:Any = $op(x...)
    end
end

const PROD,SUM,SUB = ∏,∑,-

