import pytest

from gpt4all.gpt4all import GPT4All

def test_invalid_model_type():
    model_type = "bad_type"
    with pytest.raises(ValueError):
        GPT4All.get_model_from_type(model_type)

def test_valid_model_type():
    model_type = "gptj"
    assert GPT4All.get_model_from_type(model_type).model_type == model_type

def test_invalid_model_filename():
    model_filename = "bad_filename.bin"
    with pytest.raises(ValueError):
        GPT4All.get_model_from_filename(model_filename)

def test_valid_model_filename():
    model_filename = "ggml-gpt4all-l13b-snoozy.bin"
    model_type = "gptj"
    assert GPT4All.get_model_from_filename(model_filename).model_type == model_type
