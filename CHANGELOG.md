# Changelog
All notable changes to this project will be documented in this file.

## [0.3.0] - 2022-04-12

### Documentation

- Fix list of algorithms in README ([53f300d](../../commit/53f300d3dc50fa4f19a682fecf3b30899ba6e939))

### Features

- Implement Steffensen's method for 1D systems ([422eeae](../../commit/422eeae025bc86cab61deb086b0ffaae83c1f393))
- Update nalgebra to v0.30 ([e31c75a](../../commit/e31c75ac2734a0a81d439dc5544798b02f2c1123))
- Require contiguous storage for x values ([b63317d](../../commit/b63317d7b02f84c8b6a63eb80cd5145e67b07322))
- Rename Error to ProblemError ([0aa2397](../../commit/0aa2397fb78fbf0ac606d3f28caf96d3bca683b3))

### Miscellaneous Tasks

- Fix new clippy warnings ([c84446a](../../commit/c84446ad273c556600ccc27417c0f78694bcc1ca))

## [0.2.1] - 2022-02-07

### Documentation

- Remove asterisk typo ([2870944](../../commit/2870944578dc605c7e0443acc9da8ee8e8b0850f))

### Features

- Improve robustness of Nelder-Mead algorithm ([34a7fca](../../commit/34a7fca846e2b18e09a6b182a54a23aeefa087f1))
- Avoid unnecessary function evaluation in Nelder-Mead ([fb8e90e](../../commit/fb8e90ea85bc6a9fd758f2e8513c139f65d794e4))
- Add reset functions to all solvers ([b12eca5](../../commit/b12eca5ae452038f5efb377a7523f720da9fcf85))

## [0.2.0] - 2022-01-27

### Features

- Rename `System::apply_mut` to `System::apply` ([6b5ad6e](../../commit/6b5ad6ec5e7b1e94c3dd8f511df5bfea01db916f))
  - The function was superseded by `System::eval` in ([0e869f6](../../commit/0e869f656852369ed47f23aff76f04d56d62620d))
- Add Function and Optimizer traits for general optimization ([0e869f6](../../commit/0e869f656852369ed47f23aff76f04d56d62620d))
  - Huge **breaking change** because of restructuring the problem traits to `Problem`, `System` and `Function`

### Miscellaneous Tasks

- Fix commit links for changelog ([42db7f7](../../commit/42db7f794fa232c1df057da3844a01e357e05431))

## [0.1.1] - 2022-01-08

### Bug Fixes

- Ensure that computed steps are inside trust region ([5ed3226](../../commit/5ed32266efe2fcf2e0e3b58335e3d00fe80e3310))
- Allow to use custom population initializer in Cuckoo ([7f3193d](../../commit/7f3193d4c092e1f7c2864c6bc2fb3590b3ebce58))

### Documentation

- Remove note about log being an optional feature ([a4a3430](../../commit/a4a3430a5ee0d30fceba3b9bcce8ff2265e7e109))
- Add examples for Solver/System traits ([24028d7](../../commit/24028d7b276ead14af4742176ab60f75bdb216b6))
- Fix a link in the documentation ([d498da8](../../commit/d498da80003bfda43512777d3c3b83a07d175396))
- Use modules link in the documentation index page ([53320da](../../commit/53320da28857197cec5e9d5a3fdd94bd6f778753))

### Features

- Pass system reference to population initializers ([9f2cc5b](../../commit/9f2cc5bc3ab7e111b21c9013cf9bf2df2f74a70f))
- Add analysis for initial guesses ([f78130b](../../commit/f78130b751024d334a0c1b90600283d537f2bc9e))
- Add `allow_ascent` option to trust region algorithm ([d0f7421](../../commit/d0f74211df8c45967db05fc2c95c35071bbeaf5a))

## [0.1.0] - 2021-12-20

### Added

- Create core abstractions and types
- Implement trust region algorithm
- Implement cuckoo search algorithm
- Implement Nelder-Mead algorithm
- Add tests and benchmark
