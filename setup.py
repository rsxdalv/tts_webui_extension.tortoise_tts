import setuptools

setuptools.setup(
    name="tts_webui_extension.tortoise_tts",
    packages=setuptools.find_namespace_packages(),
    version="0.0.2",
    author="rsxdalv",
    description="Tortoise TTS is a high-quality text-to-speech model with voice cloning capabilities",
    url="https://github.com/rsxdalv/tts_webui_extension.tortoise",
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

