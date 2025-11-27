import gradio as gr

from .ui import ui


def extension__tts_generation_webui():
    ui()

    return {
        "package_name": "tts_webui_extension.tortoise_tts",
        "name": "Tortoise TTS",
        "requirements": "git+https://github.com/rsxdalv/tts_webui_extension.tortoise_tts@main",
        "description": "Tortoise TTS is a high-quality text-to-speech model with voice cloning capabilities",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "neonbjb",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/neonbjb/tortoise-tts",
        "extension_website": "https://github.com/rsxdalv/tts_webui_extension.tortoise_tts",
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
