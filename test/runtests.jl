using DirectSum
using Test

# write your own tests here
@test (ℝ'⊕ℝ^3) == V"-+++"
@test (ℝ+ℝ') ⊇ ℝ
