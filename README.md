# Extension adapter for Tortoise TTS

License - the source code within this repository is licensed under the MIT license.

This extension provides a high-quality text-to-speech model with voice cloning capabilities.

## Features

- High-quality speech synthesis
- Voice cloning capabilities
- Multiple quality presets (ultra_fast, fast, standard, high_quality)
- Adjustable parameters for both autoregressive and diffusion models
- Support for custom models and tokenizers
- Split prompt functionality for long text

## Usage

1. Select a model (Default or custom)
2. Choose a voice from the dropdown or upload your own voice samples
3. Select a preset quality level
4. Adjust parameters as needed
5. Enter your text in the prompt field
6. Click "Generate" to create speech

## Advanced Options

### Model Settings
- KV Cache: Enable for faster inference at the cost of more VRAM
- DeepSpeed: Enable for optimized performance on supported hardware
- Half Precision: Enable for reduced memory usage
- Custom Tokenizer: Upload a custom tokenizer file for specialized use cases

### Autoregressive Parameters
- Num Autoregressive Samples: Higher values produce better quality but slower generation
- Temperature: Controls randomness in the autoregressive model
- Length Penalty: Penalizes longer sequences
- Repetition Penalty: Discourages repetitive outputs
- Top P: Controls diversity of outputs
- Max Mel Tokens: Maximum length of generated speech

### Diffusion Parameters
- Diffusion Iterations: Higher values produce better quality but slower generation
- Cond Free: Enable for better quality
- Cond Free K: Controls the strength of conditioning
- Temperature: Controls randomness in the diffusion model

This extension uses the [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) model.
