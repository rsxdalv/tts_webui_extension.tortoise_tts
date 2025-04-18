presets = {
    "ultra_fast": {
        "num_autoregressive_samples": 16,
        "diffusion_iterations": 16,
        "cond_free": True,
    },
    "fast": {
        "num_autoregressive_samples": 96,
        "diffusion_iterations": 80,
        "cond_free": True,
    },
    "standard": {
        "num_autoregressive_samples": 256,
        "diffusion_iterations": 200,
        "cond_free": True,
    },
    "high_quality": {
        "num_autoregressive_samples": 256,
        "diffusion_iterations": 400,
        "cond_free": True,
    },
}
