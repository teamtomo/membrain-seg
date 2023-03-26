import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_loss_fn_correctness():
    from membrain_seg.training.optim_utils import test_loss_fn

    test_loss_fn()
