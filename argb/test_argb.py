import torch
from argb import ArgbPixelEncoder

import torch.nn.functional as F


def test_argb_pixel_encoder():
    # Create a random input tensor with shape (batch_size, channels, height, width)
    batch_size, channels, height, width = 1, 3, 64, 64
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Initialize the model
    model = ArgbPixelEncoder()

    # Forward pass
    output_tensor, _ = model(input_tensor)

    # Check the output shape
    assert output_tensor.shape == (
        batch_size,
        channels,
        height,
        width,
    ), f"Expected output shape {(batch_size, channels, height, width)}, but got {output_tensor.shape}"

    # Check that the output is a tensor
    assert isinstance(
        output_tensor, torch.Tensor
    ), f"Expected output to be a tensor, but got {type(output_tensor)}"

    # Check that the output has the same spatial dimensions as the input
    assert output_tensor.size(2) == input_tensor.size(2) and output_tensor.size(
        3
    ) == input_tensor.size(
        3
    ), "Output spatial dimensions do not match input spatial dimensions"


if __name__ == "__main__":
    test_argb_pixel_encoder()
    print("All tests passed.")
