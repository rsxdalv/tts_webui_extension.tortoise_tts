import json
import os

import gradio as gr
import numpy as np
from scipy.io.wavfile import write as write_wav
from tts_webui.utils.create_base_filename import create_base_filename
from tts_webui.utils.date import get_date_string
from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.utils.save_waveform_plot import middleware_save_waveform_plot
from tts_webui.utils.split_text_functions import split_by_lines
from tts_webui.utils.torch_clear_memory import torch_clear_memory

SAMPLE_RATE = 24_000
OUTPUT_PATH = "outputs/"

MODEL = None
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


def switch_model(
    model_dir: str,
    kv_cache=False,
    use_deepspeed=False,
    half=False,
    tokenizer=None,
    use_basic_cleaners=False,
):
    from tortoise.api import MODELS_DIR

    get_tts(
        models_dir=(
            MODELS_DIR if model_dir == "Default" else get_full_model_dir(model_dir)
        ),
        force_reload=True,
        kv_cache=kv_cache,
        use_deepspeed=use_deepspeed,
        half=half,
        tokenizer_path=tokenizer.name if tokenizer else None,
        tokenizer_basic=use_basic_cleaners,
    )
    return gr.Dropdown()


def get_voice_list():
    from tortoise.utils.audio import get_voices

    # migration for legacy users
    if not os.path.exists(TORTOISE_VOICE_DIR):
        # mv from voices-tortoise to voices/tortoise
        old_dir = get_path_from_root("voices-tortoise")
        if os.path.exists(old_dir):
            os.makedirs(os.path.dirname(TORTOISE_VOICE_DIR), exist_ok=True)
            os.rename(old_dir, TORTOISE_VOICE_DIR)

    os.makedirs(TORTOISE_VOICE_DIR, exist_ok=True)

    return ["random"] + list(get_voices(extra_voice_dirs=[TORTOISE_VOICE_DIR]))


def save_wav_tortoise(audio_array, filename):
    write_wav(filename, SAMPLE_RATE, audio_array)


def unload_tortoise_model():
    global MODEL
    if MODEL is not None:
        del MODEL
        torch_clear_memory()
        MODEL = None


def get_tts(
    models_dir=None,
    force_reload=False,
    kv_cache=False,
    use_deepspeed=False,
    half=False,
    device=None,
    tokenizer_path=None,
    tokenizer_basic=False,
):
    from tortoise.api import MODELS_DIR, TextToSpeech

    if models_dir is None:
        models_dir = MODELS_DIR
    global MODEL
    if MODEL is None or force_reload:
        print("Loading tortoise model: ", models_dir)
        print("Clearing memory...")
        unload_tortoise_model()
        print("Memory cleared")
        print("Loading model...")
        MODEL = TextToSpeech(
            models_dir=models_dir,
            kv_cache=kv_cache,
            use_deepspeed=use_deepspeed,
            half=half,
            device=device,
            tokenizer_vocab_file=tokenizer_path,
            tokenizer_basic=tokenizer_basic,
        )
        print("Model loaded")
    return MODEL


last_voices = None
voice_samples = None
conditioning_latents = None


def get_voices_cached(voice):
    from tortoise.utils.audio import load_voices

    global last_voices, voice_samples, conditioning_latents

    if voice == last_voices:
        last_voices = voice
        return voice_samples, conditioning_latents

    voices = voice.split("&") if "&" in voice else [voice]

    voice_samples, conditioning_latents = load_voices(
        voices, extra_voice_dirs=[TORTOISE_VOICE_DIR]
    )
    last_voices = voices
    return voice_samples, conditioning_latents


def _generate_tortoise_chunk(
    text: str,
    candidates: int,
    **kwargs,
):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    voice = kwargs.get("voice", "random")
    voice_samples, conditioning_latents = get_voices_cached(voice)

    tts_model = get_tts()

    excluded_keys = ["text", "voice", "split_prompt", "seed", "model", "count"]
    tts_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}

    result, state = tts_model.tts_with_preset(
        text,
        return_deterministic_state=True,
        k=candidates,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        use_deterministic_seed=get_seed(kwargs.get("seed")),
        **tts_kwargs,
    )

    seed, _, _, _ = state
    kwargs["seed"] = seed

    gen_list = result if isinstance(result, list) else [result]
    audio_arrays = [tensor_to_audio_array(x) for x in gen_list]
    return [
        _process_gen(candidates, audio_array, id, kwargs)
        for id, audio_array in enumerate(audio_arrays)
    ]


def tts(
    text: str,
    count: int = 1,
    voice: str = "random",
    seed: int | None = None,
    cvvp_amount: float = 0.0,
    split_prompt: bool = False,
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
):
    print("Generating tortoise with params:")
    params = locals()
    print(params)

    prompt_raw = text

    prompts = split_by_lines(prompt_raw) if split_prompt else [prompt_raw]
    audio_pieces = [[] for _ in range(count)]

    for prompt in prompts:
        datas = _generate_tortoise_chunk(text=prompt, candidates=count, **params)
        for data in datas:
            yield data

        for i in range(count):
            audio_array = datas[i]["audio_out"][1]
            audio_pieces[i].append(audio_array)

    # if there is only one prompt, then we don't need to concatenate
    if len(prompts) == 1:
        return {}

    for i in range(count):
        res = _process_gen(
            count,
            np.concatenate(audio_pieces[i]),
            id=f"_long_{str(i)}",
            params_dict=params,
        )
        yield res

    return {}


def get_seed(seed):
    return seed if seed != -1 else None


def _process_gen(candidates, audio_array, id, params_dict):
    model = "tortoise"
    date = get_date_string()

    name = params_dict.get("voice", "random")
    filename, filename_png, filename_json = get_filenames(
        create_base_filename_tortoise(name, id, model, date)
    )
    save_wav_tortoise(audio_array, filename)
    middleware_save_waveform_plot(audio_array, filename_png)

    metadata = {
        "_version": "0.0.1",
        "_type": model,
        "date": date,
        "candidates": candidates,
        "index": id if isinstance(id, int) else 0,
        **params_dict,
        "seed": str(params_dict.get("seed")),
    }

    with open(filename_json, "w") as f:
        json.dump(metadata, f)

    folder_root = os.path.dirname(filename)

    return {
        "audio_out": (SAMPLE_RATE, audio_array),
        "folder_root": folder_root,
        "metadata": gr.JSON(value=metadata),
    }


def create_base_filename_tortoise(name, j, model, date):
    return f"{create_base_filename(f'{name}__n{j}', OUTPUT_PATH, model, date)}"


def tensor_to_audio_array(gen):
    return gen.squeeze(0).cpu().t().numpy()


def get_filenames(base_filename):
    filename = f"{base_filename}.wav"
    filename_png = f"{base_filename}.png"
    filename_json = f"{base_filename}.json"
    return filename, filename_png, filename_json
