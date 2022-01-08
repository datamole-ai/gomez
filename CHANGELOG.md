# Changelog
All notable changes to this project will be documented in this file.

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
