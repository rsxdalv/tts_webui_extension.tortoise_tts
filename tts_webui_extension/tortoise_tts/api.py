import functools
import os

import gradio as gr
import numpy as np
from tts_webui.decorators import *
from tts_webui.decorators.decorator_interrupt import interruptible
from tts_webui.decorators.decorator_save_wav import (
    decorator_save_wav_generator_accumulated,
)
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner_generator,
    decorator_extension_outer_generator,
)
from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.split_text_functions import split_by_lines

SAMPLE_RATE = 24_000

TORTOISE_VOICE_DIR = get_path_from_root("voices", "tortoise")
TORTOISE_VOICE_DIR_ABS = TORTOISE_VOICE_DIR
TORTOISE_LOCAL_MODELS_DIR = get_path_from_root("data", "models", "tortoise")


def get_model_list():
    try:
        return ["Default"] + [
            x for x in os.listdir(TORTOISE_LOCAL_MODELS_DIR) if x != ".gitkeep"
        ]
    except FileNotFoundError as e:
        print(e)
        return ["Default"]


def get_full_model_dir(model_dir: str):
    return os.path.join(TORTOISE_LOCAL_MODELS_DIR, model_dir)


def get_voice_list():
    from tortoise.utils.audio import get_voices

    os.makedirs(TORTOISE_VOICE_DIR, exist_ok=True)

    return ["random"] + list(get_voices(extra_voice_dirs=[TORTOISE_VOICE_DIR]))


@manage_model_state("tortoise")
def get_tts(
    model_name,
    models_dir=None,
    kv_cache=False,
    use_deepspeed=False,
    half=False,
    device=None,
    tokenizer_path=None,
    tokenizer_basic=False,
    progress=gr.Progress(),
):
    progress(0.0, desc="Importing library...")
    from tortoise.api import MODELS_DIR, TextToSpeech

    progress(0.1, desc="Initializing TextToSpeech...")
    return TextToSpeech(
        models_dir=models_dir or MODELS_DIR,
        kv_cache=kv_cache,
        use_deepspeed=use_deepspeed,
        half=half,
        device=device,
        tokenizer_vocab_file=tokenizer_path,
        tokenizer_basic=tokenizer_basic,
    )


@functools.lru_cache(maxsize=1)
def _get_voice_latents(voice):
    from tortoise.utils.audio import load_voices

    voices = voice.split("&") if "&" in voice else [voice]

    voice_samples, conditioning_latents = load_voices(
        voices, extra_voice_dirs=[TORTOISE_VOICE_DIR]
    )
    return voice_samples, conditioning_latents


@interruptible("tortoise-tts")
def tts_stream(
    text: str,
    voice: str = "random",
    seed: int | None = None,  # for signature compatibility
    cvvp_amount: float = 0.0,
    num_autoregressive_samples: int = 16,
    diffusion_iterations: int = 16,
    temperature: float = 0.8,
    length_penalty: float = 1.0,
    repetition_penalty: float = 2.0,
    top_p: float = 0.8,
    max_mel_tokens: int = 500,
    cond_free: bool = True,
    cond_free_k: int = 2,
    diffusion_temperature: float = 1.0,
    # model params
    model: str = "Default",
    kv_cache: bool = False,
    use_deepspeed: bool = False,
    half: bool = False,
    tokenizer: str = None,
    use_basic_cleaners: bool = False,
    progress=gr.Progress(),
    **kwargs,
):
    from tortoise.api import MODELS_DIR
    from tortoise.utils.text import split_and_recombine_text

    models_dir = MODELS_DIR if model == "Default" else get_full_model_dir(model)

    progress(0.0, desc="Loading TTS model...")
    tts_model = get_tts(
        model_name=f"Tortoise '{model}' {'with' if kv_cache else 'without'} KV Cache {'with' if use_deepspeed else 'without'} Deepspeed {'half' if half else 'full'} {'basic' if use_basic_cleaners else 'advanced'} tokenizer",
        models_dir=models_dir,
        kv_cache=kv_cache,
        use_deepspeed=use_deepspeed,
        half=half,
        tokenizer_path=tokenizer,
        tokenizer_basic=use_basic_cleaners,
        progress=progress,
    )

    progress(0.2, desc="Loading voice latents...")
    voice_samples, conditioning_latents = _get_voice_latents(voice)

    progress(0.3, desc="Generating audio...")
    for prompt in split_and_recombine_text(text):
        progress(0.5, desc=f"Generating audio for chunk: {prompt}")
        result, _ = tts_model.tts_with_preset(
            prompt,
            return_deterministic_state=True,
            k=1,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            cvvp_amount=float(cvvp_amount),
            num_autoregressive_samples=num_autoregressive_samples,
            diffusion_iterations=diffusion_iterations,
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            repetition_penalty=float(repetition_penalty),
            top_p=float(top_p),
            max_mel_tokens=max_mel_tokens,
            cond_free=cond_free,
            cond_free_k=cond_free_k,
            diffusion_temperature=float(diffusion_temperature),
        )

        gen_list = result if isinstance(result, list) else [result]

        yield {"audio_out": (SAMPLE_RATE, gen_list[0].squeeze(0).cpu().t().numpy())}


@functools.wraps(tts_stream)
def tts(*args, **kwargs):
    try:
        wavs = list(tts_stream(*args, **kwargs))
        if not wavs:
            raise gr.Error("No audio generated")
        full_wav = np.concatenate([x["audio_out"][1] for x in wavs], axis=0)
        return {"audio_out": (wavs[0]["audio_out"][0], full_wav)}
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise gr.Error(f"Error: {e}")


@functools.wraps(tts_stream)
@decorator_extension_outer_generator
@decorator_apply_torch_seed_generator
@decorator_save_metadata_generator
@decorator_save_wav_generator_accumulated
@decorator_add_model_type_generator("tortoise")
@decorator_add_base_filename_generator_accumulated
@decorator_add_date_generator
@decorator_log_generation_generator
@decorator_extension_inner_generator
@log_generator_time
def tts_decorated_stream(*args, **kwargs):
    yield from tts_stream(*args, **kwargs)
