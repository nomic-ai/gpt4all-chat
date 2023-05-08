"""
Python only API for running all GPT4All models.
"""
import os
from typing import Dict, List

from . import pyllmodel


class GPT4All():

    def __init__(self, model_path: str = None, model_type: str = None):
        self.model = None

        # Model type provided for when model is custom
        if model_type:
            self.model = GPT4All.get_model_from_type(model_type)
        # Else get model from gpt4all model filenames
        else:
            model_filename = os.path.basename(model_path)
            self.model = GPT4All.get_model_from_filename(model_filename)

        self.model.load_model(model_path)

    def generate(self, prompt: str, **generate_kwargs):
        """
        Surfaced method of running generate without accessing model object.
        """
        return self.model.generate(prompt, **generate_kwargs)
    
    def chat_completion(self, 
                        messages: List[Dict], 
                        default_prompt_header=True, 
                        default_prompt_footer=True, 
                        verbose=True) -> str:
        """
        Format list of message dictionaries into a prompt and call model
        generate on prompt. Returns a response dictionary with metadata and
        generated content.

        Parameters
        ----------
        messages: List of Dict
            Each dictionary should have a "role" key
            with value of "system", "assistant", or "user" and a "content" key with a
            string value. Messages are organized such that "system" messages are at top of prompt,
            and "user" and "assistant" messages are displayed in order. Assistant messages get formatted as
            "Reponse: {content}". 

        default_prompt_header: bool
            If True (default), add default prompt header after any user specified system messages and
            before user/assistant messages.

        default_prompt_footer: bool
            If True (default), add default footer at end of prompt.

        verbose: bool
            If True (default), print full prompt and generated response.

        Returns
        -------
        Response dictionary with:
            "usage": a dictionary with number of full prompt tokens, number of 
                generated tokens in response, and total tokens.
            "choices": List of message dictionary where "content" is generated response and "role" is set
            as "assistant". Right now, only one choice is returned by model.
        
        """
       
        full_prompt = self._build_prompt(messages)

        if verbose:
            print(full_prompt)

        response = self.model.generate(full_prompt)

        if verbose:
            print(response)

        response_dict = {
            "usage": {"prompt_tokens": len(full_prompt), 
                      "completion_tokens": len(response), 
                      "total_tokens" : len(full_prompt) + len(response)},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response
                    }
                }
            ]
        }

        return response_dict
    
    @staticmethod
    def _build_prompt(messages: List[Dict], 
                      default_prompt_header=True, 
                      default_prompt_footer=False) -> str:
        full_prompt = ""

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"] + "\n"
                full_prompt += system_message

        if default_prompt_header:
            full_prompt += """### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation 
            to respond to; decide which and write an appropriate response.
            \n### Prompt: """

        for message in messages:
            if message["role"] == "user":
                user_message = "\n" + message["content"]
                full_prompt += user_message
            if message["role"] == "assistant":
                assistant_message = "\n### Response: " + message["content"]
                full_prompt += assistant_message

        if default_prompt_footer:
            full_prompt += "\n### Response:"

        return full_prompt

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
            "ggml-gpt4all-j-v1.2-jazzy.bin",
            "ggml-gpt4all-j-v1.1-breezy.bin",
            "ggml-gpt4all-j.bin"
        ]

        LLAMA_MODELS = [
            "ggml-gpt4all-l13b-snoozy.bin",
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
