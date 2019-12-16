# DirectSum.jl

*Abstract tangent bundle vector space type operations at compile-time*

[![DOI](https://zenodo.org/badge/169765288.svg)](https://zenodo.org/badge/latestdoi/169765288)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://grassmann.crucialflow.com/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://grassmann.crucialflow.com/dev)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build Status](https://travis-ci.org/chakravala/DirectSum.jl.svg?branch=master)](https://travis-ci.org/chakravala/DirectSum.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/ipaggdeq2f1509pl?svg=true)](https://ci.appveyor.com/project/chakravala/directsum-jl)
[![Coverage Status](https://coveralls.io/repos/chakravala/DirectSum.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chakravala/DirectSum.jl?branch=master)
[![codecov.io](https://codecov.io/github/chakravala/DirectSum.jl/coverage.svg?branch=master)](https://codecov.io/github/chakravala/DirectSum.jl?branch=master)

This package is a work in progress providing the necessary tools to work with arbitrary dual `Manifold` elements specified with an encoding having optional origin, point at infinity, and tangent bundle parameter. Due to the parametric type system for the generating `VectorBundle`, the Julia compiler can fully preallocate and often cache values efficiently ahead of run-time.

Although intended for use with the [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package, `DirectSum` can be used independently.

## Direct-sum yields `VectorBundle` parametric type polymorphism ⨁

Let `N` be the rank of a `Manifold{N}`.
The type `VectorBundle{N,P,g,ν,μ}` uses *byte-encoded* data available at pre-compilation, where
`P` specifies the basis for up and down projection,
`g` is a bilinear form that specifies the metric of the space,
and `μ` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of Leibniz-Taylor monomials). Lastly, `ν` is the number of tangent variables.

The metric signature of the basis elements of a vector space `V` can be specified with the `V"..."` constructor by using `+` and `-` to specify whether the basis element of the corresponding index squares to `+1` or `-1`.
For example, `S"+++"` constructs a positive definite 3-dimensional `VectorBundle`.
```julia
julia> ℝ^3 == V"+++" == vectorspace(3)
true
```
It is also possible to specify an arbitrary `DiagonalForm` having numerical values for the basis with degeneracy `D"1,1,1,0"`, although the `Signature` format has a more compact representation.
Further development will result in more metric types.

The direct sum operator `⊕` can be used to join spaces (alternatively `+`), and `'` is an involution which toggles a dual vector space with inverted signature.
```julia
julia> V = ℝ'⊕ℝ^3
⟨-+++⟩

julia> V'
⟨+---⟩'

julia> W = V⊕V'
⟨-++++---⟩*
```
The direct sum of a `VectorSpace` and its dual `V⊕V'` represents the full mother space `V*`.

### Compile-time type operations make code optimization easier

Additionally to the direct-sum operation, several others operations are supported, such as `∪, ∩, ⊆, ⊇` for set operations.
Due to the design of the `VectorBundle` dispatch, these operations enable code optimizations at compile-time provided by the bit parameters.
```Julia
julia> ℝ+ℝ' ⊇ vectorspace(1)
true

julia> ℝ ∩ ℝ' == vectorspace(0)
true

julia> ℝ ∪ ℝ' == ℝ+ℝ'
true
```
**Note**, although some of the operations sometimes result in the same value as shown in the above examples, the `∪` and `+` are entirely different operations in general.

Calling manifolds with sets of indices constructs the subspace representations.
Given `M(s::Int...)` one can encode `SubManifold{length(s),M,s}` with induced orthogonal space, such that computing unions of submanifolds is done by inspecting the parameter ``s``.
Operations on `Manifold` types is automatically handled at compile time.
```julia
julia> (ℝ^5)(3,5)
⟨__+_+⟩

julia> dump(ans)
SubManifold{2,⟨+++++⟩,0x0000000000000014} ⟨__+_+⟩
```
Here, calling a `Manifold` with a set of indices produces a `SubManifold` representation.

### Extended dual index printing with full alphanumeric characters #62'

To help provide a commonly shared and readable indexing to the user, some print methods are provided:
```julia
julia> DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1)),false,"v")
v₁₂₃₄₅₆₇₈₉₀abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

julia> DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1)),false,"w")
w¹²³⁴⁵⁶⁷⁸⁹⁰ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```
An application of this is in the `Grasmann` package, where dual indexing is used.

### Additional features for conformal projective geometry null-basis

Declaring an additional plane at infinity is done by specifying it in the string constructor with ``\infty`` at the first index (i.e. Riemann sphere `S"∞+++"`). The hyperbolic geometry can be declared by ``\emptyset`` subsequently (i.e. Minkowski spacetime `S"∅+++"`).
Additionally, the *null-basis* based on the projective split for confromal geometric algebra would be specified with `∞∅` initially (i.e. 5D CGA `S"∞∅+++"`). These two declared basis elements are interpreted in the type system.
```julia
julia> Signature("∞∅++")
⟨∞∅++⟩
```
The index number `N` of the `VectorBundle` corresponds to the total number of generator elements. However, even though `V"∞∅+++"` is of type `VectorBundle{5,3}` with `5` generator elements, it can be internally recognized in the direct sum algebra as being an embedding of a 3-index `VectorBundle{3,0}` with additional encoding of the null-basis (origin and point at infinity) in the parameter `M` of the `VectorBundle{N,M}` type.

### Tangent bundle

The `tangent` map takes `V` to its tangent space and can be applied repeatedly for higher orders, such that `tangent(V,μ,ν)` can be used to specify `μ` and `ν`.
```julia
julia> V = tangent(ℝ^3)
⟨+++₁⟩

julia> V'
⟨---¹⟩'

julia> V+V'
⟨+++---₁¹⟩*
```

### Future work

This package is still in its beginning stages and may have deprecating changes.
