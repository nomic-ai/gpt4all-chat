"""
Python only API for running all GPT4All models.
"""
import os

from . import pyllmodel


class GPT4All():

    def __init__(self, model_path: str, model_type: str = None):
        self.model = None

        # Model type provided for when model is custom
        if model_type:
            self.model = GPT4All.get_model_from_type(model_type)
        # Else get model from gpt4all model filenames
        else:
            model_filename = os.path.basename(model_path)
            self.model = GPT4All.get_model_from_filename(model_filename)

        self.model.load_model(model_path)

    def generate(self, prompt: str, verbose: bool = True):
        """
        High level method for running generate on gpt4all models.
        NOTE: Right now, this is essentially duplicate code but will be useful
        if underlying models' prompt or generate begin to differ.
        """
        return self.model.generate(prompt, verbose=verbose)

    @staticmethod
    def get_model_from_type(model_type: str) -> pyllmodel.LLModel:
        # This needs to be updated for each new model
        # TODO: Might be worth converting model_type to enum

        if model_type == "gptj":
            return pyllmodel.GPTJModel()
        elif model_type == "llama":
            return pyllmodel.LlamaModel()
        else:
            raise ValueError(f"No corresponding model for model_type: {model_type}")
        
    @staticmethod
    def get_model_from_filename(model_filename: str) -> pyllmodel.LLModel:
        # This needs to be updated for each new model
        GPTJ_MODELS = [
            "ggml-gpt4all-j-v1.3-groovy.bin",
            "ggml-gpt4all-l13b-snoozy.bin",
            "ggml-gpt4all-j-v1.2-jazzy.bin",
            "ggml-gpt4all-j-v1.1-breezy.bin",
            "ggml-gpt4all-j.bin"
        ]

        LLAMA_MODELS = [
            "ggml-vicuna-7b-1.1-q4_2.bin",
            "ggml-vicuna-13b-1.1-q4_2.bin",
            "ggml-wizardLM-7B.q4_2.bin",
            "ggml-stable-vicuna-13B.q4_2.bin"
        ]

        if model_filename in GPTJ_MODELS:
            return pyllmodel.GPTJModel()
        elif model_filename in LLAMA_MODELS:
            return pyllmodel.LlamaModel()
        else:
            err_msg = f"""No corresponding model for provided filename {model_filename}.
            If this is a custom model, make sure to specify a valid model_type.
            """
            raise ValueError(err_msg)
