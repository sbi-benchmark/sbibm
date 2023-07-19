# v1.1.0 

- Fixed Gaussian Mixture task for `simulation_batch_size` > 1 (thanks to @h3jia, #54, #63). Since experiments in the paper were run with a simulation batch size of 1000, this has an effect on the results. We will issue an update of the results.
- Additional changes for compatibility with `sbi` v0.21.0 (thanks to @bkmi, #60, @janfb #55, #57, #59)


# v1.0.8

- Compatibility with `sbi` v0.20.0 and v.0.21.0 (#50)
- Added tests for `get_labels_parameters` and `get_labels_data` (thanks to @psteinb, #39)
- Instructions for virtual environment (thanks to @psteinb, #47)
- Two Moons Likelihood for `log=False` fixed (thanks to @michaeldeistler, #41)
- Swap `axis` for `dim` in c2st function (thanks to @michaeldeistler, #43)


# v1.0.7

- `sbibm` now depends on `torch>=1.8`
- Tests for task's and (S)MC-ABC interfaces (thanks to @psteinb, #30, #29)
- Additional metrics (#11)
- Name attribute for Bernoulli GLM Raw and SLCP Distractors fixed (thanks to @atiyo for pointing this out, #17)
- Adopted `sbi` v0.17.2 (thanks to @DongxiaW for poiting out a compatability problem, #27, #31, #32, #39)
- Code formatting with current versions of black and isort (thanks to @psteinb, #22)


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
