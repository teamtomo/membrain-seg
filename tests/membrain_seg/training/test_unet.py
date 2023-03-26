import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_semantic_segmentation_unet():
    """Simple smoke test for instantiating SemanticSegmentationUnet."""
    from membrain_seg.training import SemanticSegmentationUnet

    _ = SemanticSegmentationUnet()
