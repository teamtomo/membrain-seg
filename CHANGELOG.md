# Changelog

All notable changes to this project will be documented in this file.


## [0.0.2] - 2024-06-26
### Fixed
- Set numpy dependency to be <2.0.0 as this was causing issues with MONAI.


## [0.0.2] - 2024-06-18
### Added
- On-the-fly-rescaling of patches on GPU during inference.
- Skeletonization functionality as postprocessing step after segmentation.
- Functionality to use Surface-Dice loss during training.

### Fixed
- Fixed a bug in the segmentation module causing memory leaks.

## [0.0.1] - 2024-01-17
### Added
- Initial version of MemBrain-Seg released to PyPI.
