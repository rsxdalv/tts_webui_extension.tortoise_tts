import gradio as gr

from typing import TypedDict, Optional


class _TortoiseParametersTypedDict(TypedDict):
    text: str
    voice: str
    preset: str
    seed: Optional[int]
    cvvp_amount: float
    split_prompt: bool
    num_autoregressive_samples: int
    diffusion_iterations: int
    temperature: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    max_mel_tokens: int
    cond_free: bool
    cond_free_k: int
    diffusion_temperature: float
    model: str
    name: str


class TortoiseParameters:
    def __init__(
        self,
        text: str,
        voice: str = "random",
        preset: str = "ultra_fast",
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
        name: str = "",
    ):  # sourcery skip: remove-unnecessary-cast
        self.text = text
        self.voice = voice
        self.preset = preset
        self.seed = seed
        self.cvvp_amount = float(cvvp_amount)
        self.split_prompt = split_prompt
        self.num_autoregressive_samples = num_autoregressive_samples
        self.diffusion_iterations = diffusion_iterations
        self.temperature = float(temperature)
        self.length_penalty = float(length_penalty)
        self.repetition_penalty = float(repetition_penalty)
        self.top_p = float(top_p)
        self.max_mel_tokens = max_mel_tokens
        self.cond_free = cond_free
        self.cond_free_k = cond_free_k
        self.diffusion_temperature = float(diffusion_temperature)
        self.model = model
        self.name = name

    def __repr__(self):
        params = ",\n    ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"TortoiseParameters(\n    {params}\n)"

    def __iter__(self):
        return iter(TortoiseParameterZipper.to_list(self))

    def to_dict(self):
        return self.__dict__

    def to_metadata(self):
        return {
            **self.__dict__,
            "seed": str(self.seed),
        }

    @staticmethod
    def from_list(components: list):
        return TortoiseParameters(
            **TortoiseParameterZipper.from_list_to_dict(components)
        )


class TortoiseParameterComponents:
    def __init__(
        self,
        text: gr.Textbox,
        voice: gr.Dropdown,
        preset: gr.Dropdown,
        seed: gr.Textbox,
        cvvp_amount: gr.Slider,
        split_prompt: gr.Checkbox,
        num_autoregressive_samples: gr.Slider,
        diffusion_iterations: gr.Slider,
        temperature: gr.Slider,
        length_penalty: gr.Slider,
        repetition_penalty: gr.Slider,
        top_p: gr.Slider,
        max_mel_tokens: gr.Slider,
        cond_free: gr.Checkbox,
        cond_free_k: gr.Slider,
        diffusion_temperature: gr.Slider,
        model: gr.Dropdown,
        name: gr.Textbox,
    ):
        self.text = text
        self.voice = voice
        self.preset = preset
        self.seed = seed
        self.cvvp_amount = cvvp_amount
        self.split_prompt = split_prompt
        self.num_autoregressive_samples = num_autoregressive_samples
        self.diffusion_iterations = diffusion_iterations
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.max_mel_tokens = max_mel_tokens
        self.cond_free = cond_free
        self.cond_free_k = cond_free_k
        self.diffusion_temperature = diffusion_temperature
        self.model = model
        self.name = name

    def __repr__(self):
        params = ",\n    ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"TortoiseParameterComponents(\n    {params}\n)"

    def __iter__(self):
        return iter(TortoiseParameterZipper.to_list(self))


class TortoiseParameterZipper:
    @staticmethod
    def to_list(components: TortoiseParameterComponents | TortoiseParameters):
        return [
            components.text,
            components.voice,
            components.preset,
            components.seed,
            components.cvvp_amount,
            components.split_prompt,
            components.num_autoregressive_samples,
            components.diffusion_iterations,
            components.temperature,
            components.length_penalty,
            components.repetition_penalty,
            components.top_p,
            components.max_mel_tokens,
            components.cond_free,
            components.cond_free_k,
            components.diffusion_temperature,
            components.model,
            components.name,
        ]

    @staticmethod
    def from_list_to_dict(components: list):
        def next_idx():
            next_idx.idx += 1
            return next_idx.idx - 1

        next_idx.idx = 0
        return {
            "text": components[next_idx()],
            "voice": components[next_idx()],
            "preset": components[next_idx()],
            "seed": components[next_idx()],
            "cvvp_amount": components[next_idx()],
            "split_prompt": components[next_idx()],
            "num_autoregressive_samples": components[next_idx()],
            "diffusion_iterations": components[next_idx()],
            "temperature": components[next_idx()],
            "length_penalty": components[next_idx()],
            "repetition_penalty": components[next_idx()],
            "top_p": components[next_idx()],
            "max_mel_tokens": components[next_idx()],
            "cond_free": components[next_idx()],
            "cond_free_k": components[next_idx()],
            "diffusion_temperature": components[next_idx()],
            "model": components[next_idx()],
            "name": components[next_idx()],
        }
