# NOTE S TO SELF
# Each algorithm has a maximum of 5 keys: Selected, Available, Input, Code, and maybe Mods.
# Minimum of 4 keys: Selected, Available, Input Code
# Mods replace input arguments with different ones, they are just combined to maintain a constant visual style

Algorithms = {
    # sepconv; clip = sepconv_model(clip)
    "Sepconv": { Selected: False, Available: True,
        Code: "clip = sepconv_model(clip)"
    },
    # RIFE4; clip = RIFE(clip, multi = 2, scale = 1.0, fp16 = False, fastmode = False, ensemble = True, model_version = "sudo_rife4", psnr_dedup = False, psnr_value = 70, ssim_dedup = False, ms_ssim_dedup = False, ssim_value = 0.999, backend_inference = "cuda")
    "RIFE4": { Selected: False, Available: True,
        Input: {
            "multi": 2,
            "scale": 1.0,
            "fp16": False,
            "fastmode": False,
            "ensemble": True,
            "model_version": "sudo_rife4", # rife40 | rife41 | sudo_rife4
            "psnr_dedup": {
                Used: False,
                "psnr_value": 70,
            },
            "ssim_dedup": {
                Used: False,
                "ms_ssim_dedup": False,
                "ssim_value": 0.999
            },
            "backend_inference": "cuda" # rife4 can do cuda | ncnn, but only cuda is supported in docker
        }, Code: ["clip = RIFE(clip, ", ")"]
    },
    # VFI example for jit models; clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")
    "VFI": { Selected: False, Available: True,
        Input: {
            "fp16": False,
            "model_path": "/workspace/rvpV1_105661_G.pt"
        }, Code: ["clip = video_model(clip, ", ")"]
    },
    # SwinIR; clip = SwinIR(clip, task="lightweight_sr", scale=2)
    "SwinIR": { Selected: False, Available: True,
        Input: {
            "task": "lightweight_sr",
            "scale": 2
        }, Code: ["clip = SwinIR(clip, ", ")"]
    },
    # ESRGAN / RealESRGAN
    # clip = ESRGAN_inference(clip=clip, model_path="/workspace/4x_fatal_Anime_500000_G.pth", tile_x=400, tile_y=400, tile_pad=10, fp16=False, tta=False, tta_mode=1)
    # clip = ESRGAN_inference(clip=clip, model_path="/workspace/RealESRGAN_x4plus_anime_6B.pth", tile_x=480, tile_y=480, tile_pad=16, fp16=False, tta=False, tta_mode=1)
    "ESRGAN": { Selected: False, Available: True,
        Input: {
            "model_path": "/workspace/4x_fatal_Anime_500000_G.pth",
            "tilesize": [400, 400, 10],
            "fp16": False,
            "tta": False,
            "tta_mode": 1 # the number of times the image gets processed while being mirrored
        }, Code: ["clip = ESRGAN_inference(clip, ", ")"],
        Mods: {
            "tile_x": tilesize[0],
            "tile_y": tilesize[1],
            "tile_pad": tilesize[2]
        }
    },
    # RealESRGAN Anime Video example; clip = SRVGGNetCompactRealESRGAN(clip, scale=2, fp16=True, backend_inference = "tensorrt")
    "RealESRGAN": { Selected: False, Available: True,
        Input: {
            "scale": 2,
            "fp16": True,
            "backend_inference": "tensorrt" # tensorrt | cuda | onnx | quantized_onnx
        }, Code: ["clip = SRVGGNetCompactRealESRGAN(clip, ", ")"]
    },
    # EGVSR; clip = egvsr_model(clip, interval=15)
    "EGVSR": { Selected: False, Available: True,
        Input: {
            "interval": 15
        }, Code: ["clip = egvsr_model(clip, ", ")"]
    },
    # BasicVSR++; clip = BasicVSRPP(clip, model = 1, interval = 30, tile_x = 0, tile_y = 0, tile_pad = 16, device_type = 'cuda', device_index = 0, fp16 = False, cpu_cache = False)
    "BasicVSR": { Selected: False, Available: True,
        Input: {
            "model": 1,
            # 0 = REDS, 1 = Vimeo-90K (BI), 2 = Vimeo-90K (BD),
            # 3 = NTIRE 2021 - Track 1, 4 = Track 2, 5 = Track 3
            "interval": 30,
            "tilesize": [0, 0, 16],
            "device_type": "cuda",
            "device_index": 0,
            "fp16": False,
            "cpu_cache": False
        }, Code: ["code = BasicVSRPP(clip, ", ")"],
        Mods: {
            "tile_x": tilesize[0],
            "tile_y": tilesize[1],
            "tile_pad": tilesize[2]
        }
    },
    # RealBasicVSR; clip = realbasicvsr_model(clip, interval=15, fp16=True)
    "RealBasicVSR": { Selected: False, Available: True,
        Input: {
            "interval": 15, 
            "fp16": True
        }, Code: ["clip = realbasicvsr_model(clip, ", ")"]
    },
    # cugan; clip = cugan_inference(clip, fp16 = True, scale = 2, kind_model = "no_denoise", backend_inference = "cuda", tile_x=512, tile_y=512, tile_pad=10, pre_pad=0)
    "Cugan": { Selected: False, Available: True,
        Input: {
            "fp16": True,
            "scale": 2, # 2 | 3 | 4
            "kind_model": "no_denoise", # no_denoise | denoise3x | conservative
            "backend_inference": "cuda" # backend_inference: cuda | onnx, only cuda supports tiling
        }, Code: ["clip = cugan_inference(clip, ", ")"]
    },
    # FILM; clip = FILM_inference(clip, model_choise = "vgg")
    "FILM": {
        Selected: False, Available: True,
        Input: {
            "model_choise": "vgg" # l1 | vgg | style
        }, Code: ["clip = FILM_inference(clip, ", ")"]
    },
# vs-mlrt (you need to create the engine yourself); clip = core.trt.Model(clip, engine_path="/workspace/tensorrt/real2x.engine", tilesize=[854, 480], num_streams=6)
    "vs-mlrt": { Selected: False, Available: True,
        Input: {
            "engine_path": "/workspace/tensorrt/real2x.engine", # if you change this, you need to edit it xd
            "tilesize": [854, 480],
            "num_streams": 6
        }, Code: ["clip = core.trt.Model(clip, ", ")"]
    },
    # vs-mlrt (DPIR); DPIR does need an extra channel
    #sigma = 10.0
    #noise_level_map = core.std.BlankClip(clip, width=1280, height=720, format=vs.GRAYS)
    #clip = core.trt.Model([clip, core.std.BlankClip(noise_level_map, color=sigma/255.0)], engine_path="model.engine", tilesize=[1280, 720], num_streams=2)

    # !!!! this one doesnt work cause i cant think of how to input these into the existing system !!!!
    # / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /
    "vs-mlrt-DPIR": { Selected: False, Available: False,
        Input: {},
        Code: ["Print(\"vs-mlrt-DPIR does not work yet, sorry", "!\")"]
    },
    #/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /
    # PAN; clip = PAN_inference(clip, scale=2, fp16=True)
    "PAN": { Selected: False, Available: True,
        Input: {
            "scale": 2, # 2 | 3 | 4
            "fp16": True
        }, Code: ["clip = PAN_inference(clip, ", ")"]
    }
}