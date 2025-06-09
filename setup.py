import setuptools

setuptools.setup(
    name="extension_tortoise",
    packages=setuptools.find_namespace_packages(),
    version="0.0.2",
    author="rsxdalv",
    description="Tortoise TTS is a high-quality text-to-speech model with voice cloning capabilities",
    url="https://github.com/rsxdalv/extension_tortoise",
    project_urls={},
    scripts=[],
    install_requires=[
        "tortoise-tts @ https://github.com/rsxdalv/tortoise-tts/releases/download/v3.0.1/tortoise_tts-3.0.1-py3-none-any.whl",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
