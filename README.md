# hasktorch

[![Build Status](https://circleci.com/gh/austinvhuang/hasktorch/tree/master.svg?style=shield&circle-token=9455d7cc953a58204f4d8dd683e9fa03fd5b2744)](https://circleci.com/gh/austinvhuang/hasktorch/tree/master)

A Haskell library for tensors and neural networks. Hasktorch uses the low-level
[TorcH](https://github.com/pytorch/pytorch/tree/master/aten/src/TH) C
tensor/math library shared by [Torch](http://torch.ch/) and
[PyTorch](http://pytorch.org/). Hasktorch is an independent open source
community project.

**NOTE: hasktorch is in early development and should only be used by
contributing developers at the current time.**

## Project Organization

| Directory | Description |
| --------- | ----------- |
| [`aten/`][aten] | Modules to interact with ATen library & spec files (experimental).
| [`codegen/`][codegen] | Parsers that parse .h files in the C TH library and generate low-level raw Haskell bindings.
| [`core/`][core] | Memory-managed tensors and core data types that wrap raw C bindings to TH.
| [`nn/`][nn] | Neural network library (not much here atm)
| [`interface/`][interface] | Type interface used by raw bindings
| [`output/`][output] | Staging directory for `codegen/` output, contents should not be under source control.
| [`raw/`][raw] | Comprehensive raw bindings to several hundred C TorcH (TH) operations, including separate bindings for all tensor types.
| [`tests/`][tests] | Continuous integration tests
| [`tools/`][tools] | Misc tools
| [`vendor/`][vendor] | 3rd party dependencies as git submodules (links to TH C and other libraries)

## Build Instructions 

Currently building hasktorch is only supported on OSX and linux. Building
Hasktorch requires retrieving submodules which include TorcH library C
dependencies, followed by building with
[Stack](https://docs.haskellstack.org/en/stable/README/).

These steps can be done automatically using the [Makefile][makefile] or manually.

See the [developer guide][developers] for manual build instructions.

### Build Hasktorch with `make` (Recommended)

Run:

```
make init
```

This should retrieve submodules including TorcH library dependencies, build
them, build hasktorch itself, then run tests.

## Contributing

Contributions are welcome. For a list of things that need to get done, see:

https://github.com/austinvhuang/hasktorch/projects/1


Contact maintainers for access to the private hasktorch slack channel at:

https://hasktorch.slack.com 


<!-- project directory links -->
[developers]: ./DEVELOPERS.md
[makefile]: ./Makefile
[aten]: ./aten/
[codegen]: ./codegen/
[core]: ./core/
[examples]: ./examples/
[interface]: ./interface/
[nn]: ./nn/
[output]: ./output/
[raw]: ./raw/
[tests]: ./tests/
[tools]: ./tools/
[vendor]: ./vendor/
