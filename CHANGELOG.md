# v1.0.7

- `sbibm` now depends on `torch>=1.8`
- Code formatting with current versions of black and isort (thanks to @psteinb,
  #22)
- Name attribute for Bernoulli GLM Raw and SLCP Distractors fixed (thanks to
  @atiyo, #17) for pointing this out
- Adopted PyTorch >= 1.8 usage of log abs det jacobian, got rid of helpers (#27)
- Adds additional metrics (#11)


# v1.0.6

- Additional fix for PyTorch >= 1.8 compatibility concerning (S)MC-ABC (thanks to @atiyo, #12)


# v1.0.5

- Fixed PyTorch >= 1.8 compatibility and added KDE warnings (#8, #10)
- Updated citation to AISTATS 2021


# v1.0.4

- Fixed imports of `kgof` (thanks to @bkmi, #1)


# v1.0.3

- Updating README following publication of preprint: https://arxiv.org/abs/2101.04653


# v1.0.2

- Fixed PyPI packaging issues


# v1.0.1

- Implemented `sbibm.get_results` for convenient fetching of benchmarking results


# v1.0.0

- Initial release
