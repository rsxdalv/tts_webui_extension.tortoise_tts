import gradio as gr
from gradio_iconbutton import IconButton
from tts_webui.decorators.gradio_dict_decorator import dictionarize_wraps
from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.utils.list_dir_models import unload_model_button

from .api import (
    TORTOISE_VOICE_DIR_ABS,
    TORTOISE_LOCAL_MODELS_DIR,
    get_voice_list,
    get_model_list,
    tts,
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

        with gr.Row():
            kv_cache = gr.Checkbox(label="KV Cache", value=False)
            use_deepspeed = gr.Checkbox(label="Use Deepspeed", value=False)
            half = gr.Checkbox(label="Half", value=False)
            use_basic_cleaners = gr.Checkbox(label="Use basic cleaners", value=False)
            tokenizer = gr.File(
                label="Tokenizer",
                file_types=[".json"],
                type="filepath",
            )

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
        with gr.Column():
            (
                model,
                kv_cache,
                use_deepspeed,
                half,
                use_basic_cleaners,
                tokenizer,
            ) = tortoise_model_settings_ui()
            with gr.Column():
                gr.Markdown("Voice")
                with gr.Row():
                    voice = gr.Dropdown(
                        choices=["Press refresh to load the list"],
                        value="Press refresh to load the list",
                        show_label=False,
                        container=False,
                        allow_custom_value=True,
                    )
                    OpenFolderButton(
                        TORTOISE_VOICE_DIR_ABS, api_name="tortoise_open_voices"
                    )
                    IconButton("refresh").click(
                        fn=lambda: gr.Dropdown(choices=get_voice_list()),
                        outputs=[voice],
                        api_name="tortoise_refresh_voices",
                    )
            with gr.Column():
                gr.Markdown("Apply Preset")
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

        with gr.Column():
            cvvp_amount = gr.Slider(
                label="CVVP Amount (Deprecated, always 0)",
                value=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                interactive=False,
            )
            seed, randomize_seed_callback = randomize_seed_ui()

            with gr.Accordion(label="Diffusion Parameters", open=False):
                (
                    diffusion_iterations,
                    cond_free,
                    cond_free_k,
                    diffusion_temperature,
                ) = diffusion_params()

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

    text = gr.Textbox(label="Prompt", lines=3, placeholder="Enter text here...")

    inputs_dict = {
        text: "text",
        voice: "voice",
        seed: "seed",
        cvvp_amount: "cvvp_amount",
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
    }

    with gr.Column():
        audio = gr.Audio(type="filepath", label="Generated audio")
        folder_root = gr.Textbox(visible=False)
        metadata = gr.JSON(visible=False)
        with gr.Row():
            from tts_webui.history_tab.save_to_favorites import save_to_favorites

            gr.Button("Save to favorites").click(
                fn=save_to_favorites,
                inputs=[folder_root],
            )

    gr.Button(
        value="Generate",
        variant="primary",
    ).click(**randomize_seed_callback).then(
        **dictionarize_wraps(
            fn=tts,
            inputs=inputs_dict,
            outputs={
                "audio_out": audio,
                "folder_root": folder_root,
                "metadata": metadata,
            },
            api_name="generate_tortoise",
        )
    )
