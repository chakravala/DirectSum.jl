using DirectSum
using Test

# write your own tests here
@test (ℝ'⊕ℝ^3) == V"-+++"
@test (ℝ+ℝ') ⊇ VectorBundle(1)
@test (print(devnull,ℝ) == nothing)
@test (DirectSum.dual(ℝ) == ℝ')
@test (ℝ∩(ℝ') == VectorBundle(0))
@test (ℝ∪(ℝ') == ℝ+ℝ')
