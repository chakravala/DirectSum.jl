using DirectSum
using Test

# write your own tests here
@test (ℝ'⊕ℝ^3) == V"-+++"
@test (ℝ⊕ℝ') ⊇ TensorBundle(1)
@test (print(devnull,ℝ) == nothing)
@test (DirectSum.dual(ℝ) == ℝ')
@test (ℝ∩(ℝ') == TensorBundle(0))
@test (ℝ∪(ℝ') == ℝ⊕ℝ')
@test indices(Λ(3).v12) == [1,2]
@test (@basis ℝ^3; v1 ⊆ v12 && v12 ⊆ V)
!Sys.iswindows() && @test Λ(62).v32a87Ng == -1Λ(62).v2378agN
@test Λ.V3 == Λ.C3'
@test Λ(14) ⊕ Λ(14)' == Λ(TensorBundle(14)⊕TensorBundle(14)')

