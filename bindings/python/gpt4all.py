"""
Python only API for running all GPT4All models.
"""

import pyllmodel


class GPT4All():

    def __init__(self, model_path: str = None, model_type: str = None):
        pass

    @staticmethod
    def get_model_from_type(model_type: str = None) -> pyllmodel.LLModel:
        # This needs to be updated for each new model
        # TODO: Might be worth converting model_type to enum

        if model_type == None:
            return GPT4All.default_model_type()
        elif model_type == "gptj":
            return pyllmodel.GPTJModel
        elif model_type == "llama":
            return pyllmodel.LlamaModel
        else:
            raise ValueError(f"No corresponding model for model_type:   {model_type}")
        
    @staticmethod
    def get_model_from_filename(model_filename: str) -> pyllmodel.LLModel:
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
            return pyllmodel.GPTJModel
        elif model_filename in LLAMA_MODELS:
            return pyllmodel.LlamaModel
        else:
            err_msg = f"""No corresponding model for provided filename {model_filename}.
            If this is a custom model, make sure to specify a valid model_type.
            """
            raise ValueError(err_msg)

    @staticmethod
    def default_model_type():
        return pyllmodel.GPTJModel
