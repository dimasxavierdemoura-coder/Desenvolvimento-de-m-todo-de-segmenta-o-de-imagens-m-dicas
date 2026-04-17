import numpy as np
import torch
import torch.nn as nn

from segmentation_pipeline import ensure_input_channels, predict_mask


class DummyModel(nn.Module):
    def forward(self, x):
        return x[:, :1]


def test_ensure_input_channels_repeats_single_channel():
    tensor = torch.ones((1, 8, 8), dtype=torch.float32)
    output = ensure_input_channels(tensor, 4)
    assert output.shape == (4, 8, 8)
    assert torch.all(output == 1.0)


def test_predict_mask_with_numpy_input():
    model = DummyModel()
    mask = predict_mask(np.ones((1, 8, 8), dtype=np.float32), model, torch.device("cpu"))
    assert mask.shape == (8, 8)
    assert mask.dtype == np.uint8
    assert np.all(mask == 1)
