import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.tortoise_tts",
    packages=setuptools.find_namespace_packages(),
    version="0.0.5",
    author="rsxdalv",
    description="Tortoise TTS is a high-quality text-to-speech model with voice cloning capabilities",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts_webui_extension.tortoise_tts",
    project_urls={},
    scripts=[],
    install_requires=[
        "tts-webui.tortoise-tts"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

