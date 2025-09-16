import gradio as gr

from .gen_tortoise import (
    generate_tortoise_long,
    get_voice_list,
    TORTOISE_VOICE_DIR_ABS,
)
from .TortoiseParameters import (
    TortoiseParameterComponents,
    TortoiseParameters,
)
from .autoregressive_params import autoregressive_params
from .diffusion_params import diffusion_params
from .presets import presets
from gradio_iconbutton import IconButton
from .tortoise_model_settings_ui import tortoise_model_settings_ui
from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.decorators.gradio_dict_decorator import dictionarize

MAX_OUTPUTS = 9


def ui():
    with gr.Row():
        with gr.Column():
            model = tortoise_model_settings_ui()
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
                    OpenFolderButton(TORTOISE_VOICE_DIR_ABS, api_name="tortoise_open_voices")
                    IconButton("refresh").click(
                        fn=lambda: gr.Dropdown(choices=get_voice_list()),
                        outputs=[voice],
                        api_name="tortoise_refresh_voices",
                    )
            with gr.Column():
                gr.Markdown("Preset")
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

            split_prompt = gr.Checkbox(label="Split prompt by lines", value=False)

            (
                diffusion_iterations,
                cond_free,
                cond_free_k,
                diffusion_temperature,
            ) = diffusion_params()

            name = gr.Textbox(label="Generation Name", placeholder="Enter name here...")

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

    inputs = list(
        TortoiseParameterComponents(
            text=text,
            voice=voice,
            preset=preset,
            seed=seed,
            cvvp_amount=cvvp_amount,
            split_prompt=split_prompt,
            num_autoregressive_samples=num_autoregressive_samples,
            diffusion_iterations=diffusion_iterations,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_mel_tokens=max_mel_tokens,
            cond_free=cond_free,
            cond_free_k=cond_free_k,
            diffusion_temperature=diffusion_temperature,
            model=model,
            name=name,
        )
    )

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

    def generate_button(count):
        def gen(*args):
            yield from generate_tortoise_long(
                count,
                TortoiseParameters.from_list(list(args)),
            )

        return (
            gr.Button(
                value=f"Generate {count if count > 1 else ''}",
                variant="primary" if count == 1 else "secondary",
            )
            .click(**randomize_seed_callback)
            .then(
                fn=gen,
                inputs=inputs,
                outputs=[audio, folder_root, metadata],
                api_name=f"generate_tortoise_{count}",
                # **dictionarize(
                #     fn=gen,
                #     inputs=inputs_dict,
                #     outputs={
                #         "audio_out": audio,
                #         "folder_root": folder_root,
                #         "metadata": metadata,
                #     },
                # ),
            )
        )

    with gr.Row():
        for i in range(MAX_OUTPUTS):
            generate_button(MAX_OUTPUTS - i)


def extension__tts_generation_webui():
    ui()

    return {
        "package_name": "extension_tortoise",
        "name": "Tortoise TTS",
        "requirements": "git+https://github.com/rsxdalv/extension_tortoise@main",
        "description": "Tortoise TTS is a high-quality text-to-speech model with voice cloning capabilities",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "neonbjb",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/neonbjb/tortoise-tts",
        "extension_website": "https://github.com/rsxdalv/extension_tortoise",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    from tts_webui.gradio.css import full_css

    with gr.Blocks(css=full_css) as demo:
        with gr.Tab("Tortoise TTS"):
            ui()

    demo.launch(
        server_port=7771,
    )
