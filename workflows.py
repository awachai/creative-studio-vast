"""ComfyUI API-format workflows, exported from the official ComfyUI templates
(MiniMax H3 T2V / I2V / R2V and Z-Image-Turbo Int8) and adapted to the model
files installed on this machine (video VAE = fp16)."""
from __future__ import annotations

import copy
import math

ASPECTS = {
    "16:9": "16:9 (Widescreen)",
    "9:16": "9:16 (Portrait Widescreen)",
    "1:1": "1:1 (Square)",
    "4:3": "4:3 (Standard)",
    "3:4": "3:4 (Portrait Standard)",
    "3:2": "3:2 (Photo)",
    "2:3": "2:3 (Portrait Photo)",
    "21:9": "21:9 (Ultrawide)",
}
ASPECT_RATIOS = {k: (lambda a, b: a / b)(*map(int, k.split(":"))) for k in ASPECTS}

VIDEO_QUALITY_MP = {"draft": 0.4, "standard": 0.6, "high": 0.9}
IMAGE_QUALITY_MP = {"draft": 0.6, "standard": 1.0, "high": 2.0}

LENGTH_EXPR = "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL2V_TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
REF2V_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def closest_aspect(width: int, height: int) -> str:
    r = width / max(1, height)
    return min(ASPECT_RATIOS, key=lambda k: abs(math.log(ASPECT_RATIOS[k] / r)))


def _save_video(src: str) -> dict:
    return {"class_type": "SaveVideo", "inputs": {
        "filename_prefix": "video/MiniMax_H3", "format": "auto", "format.codec": "auto",
        "codec": "auto", "video": [src, 0]}}


def _h3_common(unet: str, lora: str, turbo_steps: int, sampler_cond_node: str) -> dict:
    """Loader / sampler / decode chain shared by the H3 workflows.
    `sampler_cond_node` is the H3 conditioning node (outputs: 0=conditioning, 1=latent)."""
    c = sampler_cond_node
    return {
        "res": {"class_type": "ResolutionSelector", "inputs": {
            "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.4, "multiple": 32}},
        "vae_v": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "vae_a": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": lora, "strength_model": 1, "model": ["unet", 0]}},
        "turbo": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "model_sw": {"class_type": "ComfySwitchNode", "inputs": {
            "switch": ["turbo", 0], "on_false": ["unet", 0], "on_true": ["lora", 0]}},
        "steps_full": {"class_type": "PrimitiveInt", "inputs": {"value": 20}},
        "steps_turbo": {"class_type": "PrimitiveInt", "inputs": {"value": turbo_steps}},
        "steps_sw": {"class_type": "ComfySwitchNode", "inputs": {
            "switch": ["turbo", 0], "on_false": ["steps_full", 0], "on_true": ["steps_turbo", 0]}},
        "duration": {"class_type": "PrimitiveFloat", "inputs": {"value": 5.0}},
        "length": {"class_type": "ComfyMathExpression", "inputs": {
            "expression": LENGTH_EXPR, "values.a": ["duration", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {
            "scheduler": "simple", "steps": ["steps_sw", 0], "denoise": 1, "model": ["model_sw", 0]}},
        "guider": {"class_type": "BasicGuider", "inputs": {
            "model": ["model_sw", 0], "conditioning": [c, 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0],
            "sigmas": ["sched", 0], "latent_image": [c, 1]}},
        "dec_v": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae_v", 0]}},
        "dec_a": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["vae_a", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {
            "fps": 24, "bit_depth": 8, "color_space": "sRGB", "codec": "none",
            "images": ["dec_v", 0], "audio": ["dec_a", 0]}},
        "save": _save_video("video"),
    }


def _apply_video_params(wf, duration, aspect, quality, seed, turbo):
    wf["duration"]["inputs"]["value"] = float(duration)
    wf["res"]["inputs"]["aspect_ratio"] = ASPECTS[aspect]
    wf["res"]["inputs"]["megapixels"] = VIDEO_QUALITY_MP[quality]
    wf["noise"]["inputs"]["noise_seed"] = int(seed)
    wf["turbo"]["inputs"]["value"] = bool(turbo)
    return wf


def text_to_video(prompt, duration, aspect, quality, seed, turbo) -> dict:
    wf = _h3_common(FL2VA, FL2V_TURBO, 8, "h3")
    wf["h3"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
        "prompt": prompt, "width": ["res", 0], "height": ["res", 1], "length": ["length", 1],
        "clip": ["clip", 0], "vae": ["vae_v", 0]}}
    return _apply_video_params(wf, duration, aspect, quality, seed, turbo)


def image_to_video(prompt, image_name, duration, aspect, quality, seed, turbo,
                   last_image_name=None) -> dict:
    wf = _h3_common(FL2VA, FL2V_TURBO, 8, "h3")
    wf["img_first"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["h3"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
        "prompt": prompt, "width": ["res", 0], "height": ["res", 1], "length": ["length", 1],
        "clip": ["clip", 0], "vae": ["vae_v", 0], "first_frame": ["img_first", 0]}}
    if last_image_name:
        wf["img_last"] = {"class_type": "LoadImage", "inputs": {"image": last_image_name}}
        wf["h3"]["inputs"]["last_frame"] = ["img_last", 0]
    return _apply_video_params(wf, duration, aspect, quality, seed, turbo)


def reference_to_video(prompt, image_names, duration, aspect, quality, seed, turbo) -> dict:
    wf = _h3_common(REF2VA, REF2V_TURBO, 4, "h3")
    inputs = {
        "prompt": prompt, "width": ["res", 0], "height": ["res", 1], "length": ["length", 1],
        "ref_image_size": "match", "clip": ["clip", 0], "vae": ["vae_v", 0], "audio_vae": ["vae_a", 0]}
    for i, name in enumerate(image_names):
        wf[f"ref_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs[f"ref_images.ref_image_{i}"] = [f"ref_{i}", 0]
    wf["h3"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
    return _apply_video_params(wf, duration, aspect, quality, seed, turbo)


def text_to_image(prompt, aspect, quality, seed) -> dict:
    mp = IMAGE_QUALITY_MP[quality] * 1_000_000
    r = ASPECT_RATIOS[aspect]
    w = int(round(math.sqrt(mp * r) / 64) * 64)
    h = int(round(math.sqrt(mp / r) / 64) * 64)
    return {
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "z_image_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "shift": 3, "sampling": "flow", "model": ["unet", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "ks": {"class_type": "KSampler", "inputs": {
            "seed": int(seed), "steps": 8, "cfg": 1, "sampler_name": "res_multistep",
            "scheduler": "simple", "denoise": 1, "model": ["shift", 0],
            "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": "image/Z_Image", "images": ["dec", 0]}},
    }
