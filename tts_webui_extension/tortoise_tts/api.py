import functools
import os

import gradio as gr
from tts_webui.decorators import *
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
):
    from tortoise.api import MODELS_DIR, TextToSpeech

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


def tts(
    text: str,
    voice: str = "random",
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
    model: str = "Default",
    kv_cache: bool = False,
    use_deepspeed: bool = False,
    half: bool = False,
    tokenizer: str = None,
    use_basic_cleaners: bool = False,
    **kwargs,
):
    from tortoise.api import MODELS_DIR

    models_dir = MODELS_DIR if model == "Default" else get_full_model_dir(model)

    tts_model = get_tts(
        model_name=f"{models_dir}_{kv_cache}_{use_deepspeed}_{half}_{tokenizer}_{use_basic_cleaners}",
        models_dir=models_dir,
        kv_cache=kv_cache,
        use_deepspeed=use_deepspeed,
        half=half,
        tokenizer_path=tokenizer,
        tokenizer_basic=use_basic_cleaners,
    )

    voice_samples, conditioning_latents = _get_voice_latents(voice)

    for prompt in split_by_lines(text):
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


@functools.wraps(tts)
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
def tts_decorated(*args, **kwargs):
    return tts(*args, **kwargs)
