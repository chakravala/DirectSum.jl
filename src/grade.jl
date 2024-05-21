
export grade, gdims, Grade

struct Grade{N,G} <: Integer
    @pure Grade{N,G}() where {N,G} = new{N,G}()
    @pure Grade{N}(G::Int) where N = new{N,G}()
    @pure Grade(N::Int,G::Int) = new{N,G}()
end

Base.show(io::IO,::Grade{N,G}) where {N,G} = print(io,"Λ$G")

Base.convert(::Type{Int},::Grade{N,G}) where {N,G} = G
Base.convert(::Type{UInt},::Grade{N,G}) where {N,G} = UInt(G)

@pure mdims(::Grade{N}) where N = N
@pure tdims(::Grade{N}) where N = 1<<N
@pure grade(::Grade{N,G}) where {N,G} = G
@pure grades(t::TensorGraded) = Values{1}(Grade(mdims(t),grade(t)))

for fun ∈ (:gdims,:combo,:binomsum,:spinsum,:antisum,:indexbasis)
    @eval begin
        @pure $fun(::Grade{N,G}) where {N,G} = $fun(N,G)
        @pure $fun(::Grade{N,N},::Grade{N,G}) where {N,G} = $fun(N,G)
    end
end
for fun ∈ (:binomial_set,:binomsum_set,:spinsum_set,:antisum_set,:indexbasis_set,:indexeven_set,:indexodd_set,:indexbasis,:indexeven,:indexodd)
    @eval @pure $fun(::Grade{N,N}) where N = $fun(N)
end

Base.:+(::Grade{N,G},::Grade{N,F}) where {N,G,F} = Grade{N,G+F}()
Base.:-(::Grade{N,G},::Grade{N,F}) where {N,G,F} = Grade{N,G-F}()

Base.:+(::Grade{N,G},F::Int) where {N,G} = G+F
Base.:+(F::Int,::Grade{N,G}) where {N,G} = F+G
Base.:-(::Grade{N,G},F::Int) where {N,G} = G-F
Base.:-(F::Int,::Grade{N,G}) where {N,G} = F-G

Base.:(==)(::Grade,::Grade) = false
Base.:(==)(::Grade{N,G},::Grade{N,G}) where {N,G} = true
Base.:(==)(::Grade{N,G},F::Int) where {N,G} = G==F
Base.:(==)(F::Int,::Grade{N,G}) where {N,G} = F==G

