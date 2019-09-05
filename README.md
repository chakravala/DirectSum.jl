# DirectSum.jl

*Abstract tangent bundle vector space type operations at compile-time*

[![Build Status](https://travis-ci.org/chakravala/DirectSum.jl.svg?branch=master)](https://travis-ci.org/chakravala/DirectSum.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/ipaggdeq2f1509pl?svg=true)](https://ci.appveyor.com/project/chakravala/directsum-jl)
[![Coverage Status](https://coveralls.io/repos/chakravala/DirectSum.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chakravala/DirectSum.jl?branch=master)
[![codecov.io](http://codecov.io/github/chakravala/DirectSum.jl/coverage.svg?branch=master)](http://codecov.io/github/chakravala/DirectSum.jl?branch=master)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Liberapay patrons](https://img.shields.io/liberapay/patrons/chakravala.svg)](https://liberapay.com/chakravala)

This package is a work in progress providing the necessary tools to work with arbitrary dual `VectorSpace` elements with optional origin, point at infinity, and tangent bundle parameter. Due to the parametric type system for the generating `VectorSpace`, the Julia compiler can fully preallocate and often cache values efficiently ahead of run-time.

Although intended for use with the [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package, `DirectSum` can be used independently.

## Direct-sum yields `VectorSpace` parametric type polymorphism ⨁

Let `N` be the dimension (number of indices) of a `VectorSpace{N}`.
The metric signature of the basis elements of a vector space `V` can be specified with the `V"..."` constructor by using `+` and `-` to specify whether the basis element of the corresponding index squares to `+1` or `-1`.
For example, `V"+++"` constructs a positive definite 3-dimensional `VectorSpace`.
```Julia
julia> ℝ^3 == V"+++" == vectorspace(3)
true
```
The direct sum operator `⊕` can be used to join spaces (alternatively `+`), and `'` is an involution which toggles a dual vector space with inverted signature.
```Julia
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

```Julia
julia> ℝ+ℝ' ⊇ vectorspace(1)
true

julia> ℝ ∩ ℝ' == vectorspace(0)
true

julia> ℝ ∪ ℝ' == ℝ+ℝ'
true
```
**Note**, although some of the operations sometimes result in the same value as shown in the above examples, the `∪` and `+` are entirely different operations in general.

### Extended dual index printing with full alphanumeric characters #62'

To help provide a commonly shared and readable indexing to the user, some print methods are provided:
```Julia
julia> DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1)),"v")
v₁₂₃₄₅₆₇₈₉₀abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

julia> DirectSum.printindices(stdout,DirectSum.indices(UInt(2^62-1)),"w")
w¹²³⁴⁵⁶⁷⁸⁹⁰ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```
An application of this is in the `Grasmann` package, where dual indexing is used.

### Additional features for conformal projective geometry null-basis

Declaring an additional *null-basis* is done by specifying it in the string constructor with `∞` at the first index (i.e. `V"∞+++"`).
Likewise, an optional *origin* can be declared by `∅` subsequently (i.e. `V"∅+++"` or `V"∞∅+++"`).
These two basis elements will be interpreted in the type system such that they propagate under transformations when combining a mixed index sets (provided the `Signature` is compatible).

```Julia
julia> Signature("∞∅++")
⟨∞∅++⟩
```

The index number `N` of the `VectorSpace` corresponds to the total number of generator elements. However, even though `V"∞∅+++"` is of type `VectorSpace{5,3}` with `5` generator elements, it can be internally recognized in the direct sum algebra as being an embedding of a 3-index `VectorSpace{3,0}` with additional encoding of the null-basis (origin and point at infinity) in the parameter `M` of the `VectorSpace{N,M}` type.

### Tangent bundle

The `tangent` map takes `V` to its tangent space.

```Julia
julia> V = tangent(ℝ^3)
⟨+++₁⟩

julia> V'
⟨---¹⟩'

julia> V+V'
⟨+++---₁¹⟩*
```

### Future work

This package is still in its beginning stages and may have deprecating changes.

One of the new features is the `SubManifold` implementation,
```Julia
julia> (ℝ^5)(3,5)
⟨__+_+⟩

julia> dump(ans)
SubManifold{2,⟨+++++⟩,0x0000000000000014} ⟨__+_+⟩
```
where calling a manifold with a set of indices produces a subspace representation.
