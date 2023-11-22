# Changelog
All notable changes to this project will be documented in this file.

## [0.5.0] - 2023-11-22

[2ee24f0](2ee24f06175bcf8796390c7d2a0f199cbd555424)...[2ee24f0](2ee24f06175bcf8796390c7d2a0f199cbd555424)

### Bug Fixes

- Use lowercase for commit messages because it's standard ([a352b8e](../../commit/a352b8e3204d9474fb6fb1ab29a9f9b71eecaef8))
- Fix gsl-sys ([fe86e32](../../commit/fe86e321286e5197a988cd38643f652e5f652940))
- Fix random sampling for unconstrained domains ([278fbe0](../../commit/278fbe0db900f2b51f7b2af4831257b6673b2c3b))
- Fix steffensen algorithm when given point is already solution ([42da442](../../commit/42da442259617b8ef703c34e6729670f02052efb))
- Fix step calculation in jacobian matrix approximation ([4161c56](../../commit/4161c563c88bcac36af7fe472d33e69e6fc9b697))

### Documentation

- Overhaul the documentation ([30d44a9](../../commit/30d44a9987fa43d9d1e847e561c8a1ab27ef4619))

### Features

- Remove cuckoo search algorithm ([29b22be](../../commit/29b22be73b685ebc1214524b8ca47ee509d80b26))
- Remove population module ([7b5a64b](../../commit/7b5a64b5cec39c9de36dd5c6ac7126b04cab8583))
- Hide testing module behind a feature flag ([e69ede6](../../commit/e69ede63c13cdb288a8f05aa8d70b57f963d871c))
- Introduce TestFunction trait in testing module ([9ea04f0](../../commit/9ea04f0a9b913ed0932fe4765f1e1bc0e00269c2))
- Change the interoperability between System and Function traits ([db104b5](../../commit/db104b5dced03f917b04df88746130638e6c7c5e))
- Make systems and functions infallible ([8b6c271](../../commit/8b6c271d52869427658dca4fc1dfadc2ca8f5034))
- Rename next methods on Solver and Optimizer so they do not clash ([d15c996](../../commit/d15c996786065c385146a1e80fa2472178fa114b))
- Rename scalar associated type to field ([5d8c8e1](../../commit/5d8c8e1453209a0bb65dcaaba31445c4568cf53e))
- Introduce own RealField trait ([abe0f65](../../commit/abe0f65939e7a2b5900cf8df8365f9e1ca30ead0))
- Remove prelude and move important types to lib root ([057d595](../../commit/057d5955da58c81a3856bb49543b25bb520aae12))
- Use fastrand crate for random number generation ([67a0d45](../../commit/67a0d454f9694d7b37ac1b83357013f2644db823))
- Make magnitude estimation function public ([69b037a](../../commit/69b037a810d670dbf01ec90efa7d77518a1bfc45))
- Remove RepulsiveSystem type ([bc35ea2](../../commit/bc35ea223b3b0cc892f1c10251c811d5833e20a1))
- Move non-linear vars detection and magnitude estimation ([639a8af](../../commit/639a8afc6ad9094ea28903cff82624829b74eb1d))
- Rename solver module to algo ([8c59165](../../commit/8c591659c59ca8ae017cbe49c5d8ddfaac20c10a))
- Use std vectors for in public API of the domain ([16de444](../../commit/16de4449fffb582898ffb7497d40a04799a599da))
- Implement high-level API for solving and optimization ([f9fba37](../../commit/f9fba37724bf327a7e5894bb6dd8a4f8e608d468))
- Update nalgebra ([d2c229b](../../commit/d2c229b1f5cfda5ca891b1c4ae11f71b9f320362))
- Return x as part of the result of next and find ([b392536](../../commit/b392536a8ac40ad5bffcb6f8749af87eb740ae5c))
- Implement different rules for step size in derivatives approx ([2ee24f0](../../commit/2ee24f06175bcf8796390c7d2a0f199cbd555424))

### Miscellaneous Tasks

- Fix rustdoc warning ([ca44cac](../../commit/ca44cac1c5bce5e2911bf76671aecdf047bbc30c))
- Use all numbers from semver version for dependencies ([24791d1](../../commit/24791d1116bd8fd276e70653b5cb7ea6ad6bf6d6))
- Sort dependencies alphabetically ([a9d1ad9](../../commit/a9d1ad926ba302acddcf27546ffbebaf6e30eb46))

### Refactor

- Move benchmark suite to a separate crate ([d7a92b4](../../commit/d7a92b456f68906c71cfc1533d7782627aa89d10))
- Rename SolveError to TestingError ([7e72ec7](../../commit/7e72ec7af8a2e25c9d0a4c866dd934db01c9c68b))
- Completely overhaul the domain type ([606da4f](../../commit/606da4f7d1117bffe2908076f01ef0bb424ebc89))
- Always use dynamic dimension ([4f8f2ce](../../commit/4f8f2ce941858024bd7acb5e2354bb4601179a4a))
- Do not use num_traits dependency ([b69bb8d](../../commit/b69bb8de60e794ca574fa3f4d24076f7f5dd58b9))

## [0.4.1] - 2023-05-15

[4028405](40284050177ba91fdea9683492cace42e13d8816)...[22445f5](22445f5b5db537edfa52b9839621a422fe32d78b)

### Bug Fixes

- Fix Nelder-Mead for univariate optimization ([22445f5](../../commit/22445f5b5db537edfa52b9839621a422fe32d78b))

## [0.4.0] - 2023-05-13

[3c523d3](3c523d36da132643bda75ec7ac663782a1721b86)...[ebc07ed](ebc07ed09b39c4a0103b01d5e74920632453c1c7)

### Features

- Remove Function and Optimizer from the prelude ([dd6524a](../../commit/dd6524a3e18f6a4a1539134931e7d1f845c16d5e))
- Adapt the trust region algorithm for function optimization ([96fe08f](../../commit/96fe08f5c85b8535d6514e2ab3095820c1ca4686))
- Implement LIPO solver ([ebc07ed](../../commit/ebc07ed09b39c4a0103b01d5e74920632453c1c7))

## [0.3.2] - 2023-04-28

[48bbf11](48bbf11afd3237d0984755c74968920ce52c0cd9)...[dfd5399](dfd5399806e919598c452bf233153bcaea983ac2)

### Bug Fixes

- Initialize residuals when initializing simplex in Nelder-Mead ([5b758a2](../../commit/5b758a2979eb8fd9e52a067b5d9ea92ce985b0bf))

### Features

- Introduce a new, hidden setting for TrustRegion ([dfd5399](../../commit/dfd5399806e919598c452bf233153bcaea983ac2))

### Miscellaneous Tasks

- Fix clippy warnings and other stylistic changes ([141305a](../../commit/141305a5de990f04c07a0f764e7b34a902b4d347))

### Testing

- Make tests to actually check if the successful output is a root ([33ae76a](../../commit/33ae76a7fb741a88f21e8b9c35d87283ef351a9b))
- Add a test for system that has infinite solutions ([266a1b1](../../commit/266a1b17894002f48958df19eef5c94633d02e2d))

## [0.3.1] - 2022-05-08

### Features

- Update nalgebra to v0.31 ([009ae58](../../commit/009ae580e811ab192b6987e85a8a0ff0acca1493))

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
