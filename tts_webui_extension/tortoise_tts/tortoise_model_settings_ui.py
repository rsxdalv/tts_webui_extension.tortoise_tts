import gradio as gr
from .api import (
    TORTOISE_LOCAL_MODELS_DIR,
    get_model_list,
    switch_model,
    unload_tortoise_model,
)
from gradio_iconbutton import IconButton
from tts_webui.utils.OpenFolderButton import OpenFolderButton


def tortoise_model_settings_ui_inner():
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
        )

    unload_model = gr.Button(
        "Unload model",
        variant="secondary",
    )

    unload_model.click(
        fn=unload_tortoise_model,
        api_name="tortoise_unload_model",
    )

    apply_model_settings = gr.Button(
        "Apply model settings",
        variant="secondary",
    )

    apply_model_settings.click(
        fn=switch_model,
        inputs=[model, kv_cache, use_deepspeed, half, tokenizer, use_basic_cleaners],
        outputs=[model],
        api_name="tortoise_apply_model_settings",
    )

    model.select(
        fn=switch_model,
        inputs=[model, kv_cache, use_deepspeed, half, tokenizer, use_basic_cleaners],
        outputs=[model],
    )

    return model


def tortoise_model_settings_ui():
    with gr.Column():
        gr.Markdown("Model")
        with gr.Column():
            return tortoise_model_settings_ui_inner()
