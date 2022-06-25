from src.SRVGGNetCompact_arch import SRVGGNetCompact
import torch

print("Converting")

model = SRVGGNetCompact(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"
)
model.load_state_dict(torch.load("model.pth"))
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
