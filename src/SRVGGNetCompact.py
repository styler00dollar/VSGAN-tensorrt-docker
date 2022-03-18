import functools
import numpy as np
import vapoursynth as vs
import os
import numpy as np
import torch
import sys
from contextlib import contextmanager

# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()
    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

# Code mainly from https://github.com/HolyWu/vs-realesrgan
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def SRVGGNetCompactRealESRGAN(clip: vs.VideoNode, scale: int = 2, fp16: bool = False, backend_inference: str = "cuda", tta_mode:bool = False, param_path:str = "eest.param", bin_path:str = "test.bin", gpuid:int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if scale not in [2, 4]:
        raise vs.Error('RealESRGAN: scale must be 2 or 4')
    
    with stdout_redirected(to=os.devnull):
        # load network
        if backend_inference != "ncnn":
            from .SRVGGNetCompact_arch import SRVGGNetCompact
            model_path = f'/workspace/RealESRGANv2-animevideo-xsx{scale}.pth'
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale, act_type='prelu')
            model.load_state_dict(torch.load(model_path, map_location="cpu")['params'])
            model.eval()
        
        if backend_inference == "tensorrt":
            import onnx as ox
            import onnx_tensorrt.backend as backend
            # export to onnx and load with tensorrt (you cant use https://github.com/NVIDIA/Torch-TensorRT because the scripting step will fail)
            torch.onnx.export(model, (torch.rand(1,3,clip.height,clip.width)), f"/workspace/test.onnx", verbose=False, opset_version=14)
            model = ox.load("/workspace/test.onnx")
            model = backend.prepare(model, device='CUDA:0', fp16_mode=fp16)
        elif backend_inference == "onnx":
            import onnx as ox
            import onnxruntime as ort
            torch.onnx.export(model, (torch.rand(1,3,clip.height,clip.width)), f"/workspace/test.onnx", verbose=False, opset_version=14, input_names=['input'], output_names=['output'])
            model = ox.load("/workspace/test.onnx")
            sess = ort.InferenceSession(f"/workspace/test.onnx", providers=["CUDAExecutionProvider"])
        elif backend_inference == "quantized_onnx":
            import onnxruntime as ort
            import onnx as ox
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_qat, QuantType, quantize, QuantizationMode
            torch.onnx.export(model.cuda(), (torch.rand(1,3,clip.height,clip.width)).cuda(), f"/workspace/test.onnx", verbose=False, opset_version=14, input_names=['input'], output_names=['output'])
            quantized_model = quantize_dynamic("/workspace/test.onnx", "/workspace/test_quant.onnx", weight_type=QuantType.QUInt8)
            model = ox.load("/workspace/test_quant.onnx")
            sess = ort.InferenceSession(f"/workspace/test.onnx", providers=["CUDAExecutionProvider"])
        elif backend_inference == "cuda":
            if fp16:
                model = model.half()
            model.cuda()
        elif backend_inference == "ncnn":
            from realsr_ncnn_vulkan_python import RealSR
            model = RealSR(gpuid=gpuid, tta_mode=tta_mode, scale=scale, param_path=param_path, bin_path=bin_path)

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        img = frame_to_tensor(clip.get_frame(n))
        if backend_inference != "ncnn":
            img = np.expand_dims(img, 0)

        if backend_inference == "tensorrt":
          output = model.run(img)[0]
        elif backend_inference == "onnx" or backend_inference == "quantized_onnx":
          output = sess.run(None, {'input': img})[0]
        elif backend_inference == "cuda":
          img = torch.Tensor(img).to("cuda", non_blocking=True)
          if fp16:
            img = img.half()
          output = model(img)
          output = output.detach().cpu().numpy()
        elif backend_inference == "ncnn":
            img = img * 255
            img = np.rollaxis(img.clip(0,255).astype(np.uint8), 0,3)
            output = model.process(img)
            output = output.swapaxes(0, 2).swapaxes(1, 2)/255

        if backend_inference != "ncnn":
            output = np.squeeze(output, 0)
        return tensor_to_clip(clip=clip, image=output)

    return core.std.FrameEval(
            core.std.BlankClip(
                clip=clip,
                width=clip.width * scale,
                height=clip.height * scale
            ),
            functools.partial(
                execute,
                clip=clip
            )
    )

def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    return np.stack([
        np.asarray(frame[plane])
        for plane in range(frame.format.num_planes)
    ])

def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f

def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(
        clip=clip,
        width=image.shape[-1],
        height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )
