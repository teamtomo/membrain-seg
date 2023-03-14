from membrain_seg.networks import SemanticSegmentationUnet


def test_semantic_segmentation_unet():
    """Simple smoke test for instantiating SemanticSegmentationUnet."""
    _ = SemanticSegmentationUnet()
