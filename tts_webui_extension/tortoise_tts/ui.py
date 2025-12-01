import gradio as gr
from gradio_iconbutton import IconButton
from tts_webui.decorators.decorator_interrupt import InterruptButton
from tts_webui.decorators.gradio_dict_decorator import dictionarize_wraps
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.randomize_seed import randomize_seed_ui

from .api import (
    TORTOISE_LOCAL_MODELS_DIR,
    TORTOISE_VOICE_DIR_ABS,
    get_model_list,
    get_voice_list,
    tts_decorated_stream,
)
from .presets import presets


def tortoise_model_settings_ui():
    with gr.Column():
        gr.Markdown("Model")
        with gr.Row():
            model = gr.Dropdown(
                choices=get_model_list(),
                value="Default",
                show_label=False,
                container=False,
            )
            OpenFolderButton(TORTOISE_LOCAL_MODELS_DIR, api_name="tortoise_open_models")
            IconButton("refresh").click(
                fn=lambda: gr.Dropdown(choices=get_model_list()),
                outputs=[model],
                api_name="tortoise_refresh_models",
            )

        tokenizer = gr.File(label="Tokenizer", file_types=[".json"], type="filepath")

        gr.Markdown("Optimization Settings")
        with gr.Row():
            kv_cache = gr.Checkbox(label="KV Cache", value=True)
            use_deepspeed = gr.Checkbox(label="Use Deepspeed", value=False)
            half = gr.Checkbox(label="Half", value=False)
            use_basic_cleaners = gr.Checkbox(label="Use basic cleaners", value=False)

        unload_model_button("tortoise")

    return model, kv_cache, use_deepspeed, half, use_basic_cleaners, tokenizer


def autoregressive_params():
    return (
        gr.Slider(label="Samples", value=16, minimum=4, maximum=256, step=1),
        gr.Slider(label="Temperature", value=0.8, minimum=0.0, maximum=1.0, step=0.1),
        gr.Slider(
            label="Length Penalty", value=1.0, minimum=0.0, maximum=10.0, step=0.1
        ),
        gr.Slider(
            label="Repetition Penalty", value=2.0, minimum=0.0, maximum=10.0, step=0.1
        ),
        gr.Slider(label="Top P", value=0.8, minimum=0.0, maximum=1.0, step=0.1),
        gr.Slider(label="Max Mel Tokens", value=500, minimum=10, maximum=600, step=1),
    )


def diffusion_params():
    return (
        gr.Slider(
            label="Diffusion Iterations", value=30, minimum=4, maximum=400, step=1
        ),
        gr.Checkbox(label="Cond Free", value=False),
        gr.Slider(label="Cond Free K", value=2, minimum=0, maximum=10, step=1),
        gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=1.0, step=0.1),
    )


def ui():
    with gr.Row():
        tortoise_tts_ui()


def tortoise_tts_ui():
    with gr.Column():
        text = gr.Textbox(label="Prompt", lines=3, placeholder="Enter text here...")
        with gr.Row():
            InterruptButton("tortoise-tts", value="Interrupt")
            generate_button = gr.Button(value="Generate", variant="primary")

        gr.Markdown("Voice")
        with gr.Row():
            voice = gr.Dropdown(
                choices=["Press refresh to load the list"],
                value="Press refresh to load the list",
                show_label=False,
                container=False,
                allow_custom_value=True,
            )
            OpenFolderButton(TORTOISE_VOICE_DIR_ABS, api_name="tortoise_open_voices")
            IconButton("refresh").click(
                fn=lambda: gr.Dropdown(choices=get_voice_list()),
                outputs=[voice],
                api_name="tortoise_refresh_voices",
            )

        gr.Markdown("Select a preset to quickly adjust parameters:")
        preset = gr.Dropdown(
            show_label=False,
            choices=[
                "ultra_fast",
                "fast",
                "standard",
                "high_quality",
            ],
            value="ultra_fast",
            container=False,
        )

        with gr.Accordion(label="Autoregressive Parameters", open=False):
            (
                num_autoregressive_samples,
                temperature,
                length_penalty,
                repetition_penalty,
                top_p,
                max_mel_tokens,
            ) = autoregressive_params()

        with gr.Accordion(label="Diffusion Parameters", open=False):
            (
                diffusion_iterations,
                cond_free,
                cond_free_k,
                diffusion_temperature,
            ) = diffusion_params()

    with gr.Column():
        audio = gr.Audio(type="filepath", label="Generated audio")

        # gr.Button("Save to favorites").click(
        #     fn=save_to_favorites,
        #     inputs=[folder_root],
        # )

        seed, randomize_seed_callback = randomize_seed_ui()

        with gr.Accordion(label="Model Settings", open=False):
            (
                model,
                kv_cache,
                use_deepspeed,
                half,
                use_basic_cleaners,
                tokenizer,
            ) = tortoise_model_settings_ui()

    preset.change(
        fn=lambda x: [
            gr.Slider(value=presets[x]["num_autoregressive_samples"]),
            gr.Slider(value=presets[x]["diffusion_iterations"]),
            gr.Checkbox(
                value=presets[x]["cond_free"] if "cond_free" in presets[x] else True
            ),
        ],
        inputs=[preset],
        outputs=[num_autoregressive_samples, diffusion_iterations, cond_free],
    )

    folder_root = gr.Textbox(visible=False)
    metadata = gr.JSON(visible=False)
    generate_button.click(**randomize_seed_callback).then(
        **dictionarize_wraps(
            fn=tts_decorated_stream,
            inputs={
                text: "text",
                voice: "voice",
                seed: "seed",
                num_autoregressive_samples: "num_autoregressive_samples",
                diffusion_iterations: "diffusion_iterations",
                temperature: "temperature",
                length_penalty: "length_penalty",
                repetition_penalty: "repetition_penalty",
                top_p: "top_p",
                max_mel_tokens: "max_mel_tokens",
                cond_free: "cond_free",
                cond_free_k: "cond_free_k",
                diffusion_temperature: "diffusion_temperature",
                model: "model",
                kv_cache: "kv_cache",
                use_deepspeed: "use_deepspeed",
                half: "half",
                tokenizer: "tokenizer",
                use_basic_cleaners: "use_basic_cleaners",
            },
            outputs={
                "audio_out": audio,
                "folder_root": folder_root,
                "metadata": metadata,
            },
            api_name="generate_tortoise",
        )
    )
