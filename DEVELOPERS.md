# Information for Developers

## Building Hasktorch Manually

* Install a compatible version of `ATen`. One such way is to use the submodule
  included in this repo.

  * Clone it into `vendor/aten` with:

    ```
    git submodule update --init --recursive
    ```

  * Build `ATen`. Refer to the full documentation at
    [https://github.com/hasktorch/ATen](https://github.com/hasktorch/ATen) for
    how to do so, but here is a quick example:

    ```
    cd vendor/aten
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=.
    make install
    ```

    (To build without CUDA support, use `-DNO_CUDA=true`)

    This will create `vendor/aten/build/lib` and `vendor/aten/build/include`
    directories containing the `libATen.so` shared object and header files,
    respectively.

* Create a `cabal.project.local` file with settings for your particular
  installation.

  This serves two primary purposes:

  * Locate `ATen` build artifacts, if they were installed to a non-standard
    system location, as in the example above, which installed into
    `vendor/aten/build`.

    ```
    package hasktorch-raw-th
      extra-lib-dirs: /path/to/vendor/aten/build/lib
      extra-include-dirs: /path/to/vendor/aten/build/include
    ```

  * Build without CUDA support, if desired.

    ```
    package hasktorch-core
      flags: -cuda
    ```

* Build `hasktorch-core`.

  ```
  cabal new-build hasktorch-core
  ```

* Test `hasktorch-core`.

  ```
  cabal new-test hasktorch-core
  ```

## References

### Torch Internals

- [Torch Internals Writeup by Adam Paszke](https://apaszke.github.io/torch-internals.html)
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).

###  Background on Dependent types in Haskell and NN Implementations

- [Practical dependent types in Haskell](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html)
- [Monday Morning Haskell: Deep Learning and Deep Types: Tensor Flow and Dependent Types](https://mmhaskell.com/blog/2017/9/11/deep-learning-and-deep-types-tensor-flow-and-dependent-types)
- [Basic Type Level Programming in Haskell](http://www.parsonsmatt.org/2017/04/26/basic_type_level_programming_in_haskell.html)
- [static dimension checking](http://dis.um.es/~alberto/hmatrix/static.html)
- [defunctionalization](https://typesandkinds.wordpress.com/2013/04/01/defunctionalization-for-the-win/)
- [An introduction to DataKinds and GHC.TypeLits](http://ponies.io/posts/2014-07-30-typelits.html)
- [Applying Type-Level and Generic Programming in Haskell](https://www.cs.ox.ac.uk/projects/utgp/school/andres.pdf)

### Automatic Differentiation

- [Automatic Propagation of Uncertainty with AD](https://blog.jle.im/entry/automatic-propagation-of-uncertainty-with-ad.html)
- [Automatic Differentiation is Trivial in Haskell](http://www.danielbrice.net/blog/2015-12-01/)
