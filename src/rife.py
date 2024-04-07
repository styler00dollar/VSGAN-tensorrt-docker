import torch
from .rife_arch import IFNet
from .download import check_and_download


# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class RIFE:
    def __init__(self, scale, fastmode, ensemble, model_version, fp16):
        self.scale = scale
        self.fastmode = fastmode
        self.ensemble = ensemble
        self.model_version = model_version
        self.fp16 = fp16
        self.cache = False
        self.amount_input_img = 2

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        model_info = {
            "rife40": ("/workspace/tensorrt/models/rife40.pth", "4.0"),
            "rife41": ("/workspace/tensorrt/models/rife41.pth", "4.0"),
            "rife42": ("/workspace/tensorrt/models/rife42.pth", "4.2"),
            "rife43": ("/workspace/tensorrt/models/rife43.pth", "4.3"),
            "rife44": ("/workspace/tensorrt/models/rife44.pth", "4.3"),
            "rife45": ("/workspace/tensorrt/models/rife45.pth", "4.5"),
            "rife46": ("/workspace/tensorrt/models/rife46.pth", "4.6"),
            "rife47": ("/workspace/tensorrt/models/rife47.pth", "4.7"),
            "rife48": ("/workspace/tensorrt/models/rife48.pth", "4.7"),
            "rife49": ("/workspace/tensorrt/models/rife49.pth", "4.7"),
            "rife410": ("/workspace/tensorrt/models/rife410.pth", "4.10"),
            "rife411": ("/workspace/tensorrt/models/rife411.pth", "4.10"),
            "rife412": ("/workspace/tensorrt/models/rife412.pth", "4.10"),
            "sudo_rife4": (
                "/workspace/tensorrt/models/sudo_rife4_269.662_testV1_scale1.pth",
                "4.0",
            ),
        }

        if model_version in model_info:
            model_path, arch_ver = model_info[model_version]

        check_and_download(model_path)
        self.model = IFNet(arch_ver=arch_ver)
        self.model.load_state_dict(torch.load(model_path), False)

        self.model.eval().cuda()

        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.model.half()

    def execute(self, I0, I1, timestep):
        scale_list = [8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]

        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.inference_mode():
            middle = self.model(
                I0,
                I1,
                scale_list=scale_list,
                fastmode=self.fastmode,
                ensemble=self.ensemble,
                timestep=timestep,
            )

        middle = middle.detach().squeeze(0).cpu().numpy()
        return middle
