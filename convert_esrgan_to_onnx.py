from src.esrgan import ESRGAN
import torch

print("Converting")

model = ESRGAN("oldies.pth")
model.eval().cuda()
# https://github.com/onnx/onnx/issues/654
dynamic_axes = {
    "input": {0: "batch_size", 2: "width", 3: "height"},
    "output": {0: "batch_size", 2: "width", 3: "height"},
}
dummy_input = torch.rand(1, 3, 64, 64).cuda()

# fp32
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)
# fp16
torch.onnx.export(
    model.half(),
    dummy_input.half(),
    "model_fp16.onnx",
    opset_version=14,
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)
print("Finished")
