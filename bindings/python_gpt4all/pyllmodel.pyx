from libcpp cimport bool

from io import StringIO
import re
import sys

# Import C++ int types from stdint
cdef extern from "<stdint.h>":
    ctypedef unsigned char uint8_t
    ctypedef signed char int8_t
    ctypedef unsigned short uint16_t
    ctypedef signed short int16_t
    ctypedef unsigned int uint32_t
    ctypedef signed int int32_t
    ctypedef unsigned long long uint64_t
    ctypedef signed long long int64_t

# Import C external functions from llmodel
cdef extern from "llmodel_c.h":
    ctypedef void *llmodel_model
    ctypedef bool (*llmodel_prompt_callback)(int32_t)
    ctypedef bool (*llmodel_response_callback)(int32_t, const char *)
    ctypedef bool (*llmodel_recalculate_callback)(bool)

    ctypedef struct llmodel_prompt_context:
        float *logits
        size_t logits_size
        int32_t *tokens
        size_t tokens_size
        int32_t n_past
        int32_t n_ctx
        int32_t n_predict
        int32_t top_k
        float top_p
        float temp
        int32_t n_batch
        float repeat_penalty
        int32_t repeat_last_n
        float context_erase

    llmodel_model llmodel_gptj_create()
    void llmodel_gptj_destroy(llmodel_model gptj)

    llmodel_model llmodel_llama_create()
    void llmodel_llama_destroy(llmodel_model llama)

    bool llmodel_loadModel(llmodel_model model, const char *model_path)
    bool llmodel_isModelLoaded(llmodel_model model)

    void llmodel_prompt(llmodel_model model, const char *prompt,
                    llmodel_response_callback prompt_callback,
                    llmodel_response_callback response_callback,
                    llmodel_recalculate_callback recalculate_callback,
                    llmodel_prompt_context *ctx)


cdef class LLModel:
    """
    Base class and universal wrapper for GPT4All language models
    built around llmodel C-API.

    Attributes
    ----------
    model: llmodel_model
        Ctype pointer to underlying model
    model_type : str
        Model architecture identifier
    """

    cdef llmodel_model model
    model_type: str = None

    def load_model(self, model_path: str) -> bool:
        """
        Load model from a file.

        Parameters
        ----------
        model_path : str
            Model filepath

        Returns
        -------
        True if model loaded successfully, False otherwise
        """
        llmodel_loadModel(self.model, model_path.encode('utf-8'))
    
        if llmodel_isModelLoaded(self.model):
            return True
        else:
            return False

    def generate(self, 
                 prompt: str,
                 add_default_header: bool = True, 
                 add_default_footer: bool = True,
                 logits_size: int = 0, 
                 tokens_size: int = 0, 
                 n_past: int = 0, 
                 n_ctx: int = 1024, 
                 n_predict: int = 128, 
                 top_k: int = 40, 
                 top_p: float = .9, 
                 temp: float = .1, 
                 n_batch: int = 8, 
                 repeat_penalty: float = 1.2, 
                 repeat_last_n: int = 10, 
                 context_erase: float = .5,
                 verbose: bool = True) -> str:
        """
        Generate response from model from a prompt.

        Parameters
        ----------
        prompt: str
            Question, task, or conversation for model to respond to
        add_default_header: bool, optional
            Whether to add a prompt header (default is True)
        add_default_footer: bool, optional
            Whether to add a prompt footer (default is True)
        # TODO: figure out what prompt context params are doing
        verbose: bool, optional
            Whether to print prompt and response

        Returns
        -------
        Model response str
        """

        full_prompt = ""
        if add_default_header:
            full_prompt += """### Instruction:\n 
            The prompt below is a question to answer, a task to complete, or a conversation 
            to respond to; decide which and write an appropriate response.
            \n### Prompt: """
        
        full_prompt += prompt

        if add_default_footer:
            full_prompt += "\n### Response:"

        if verbose:
            print(full_prompt)
        
        full_prompt = full_prompt.encode('utf-8')

        # Change stdout to StringIO so we can collect response
        old_stdout = sys.stdout 
        collect_response = StringIO()
        sys.stdout = collect_response
        
        self._generate_c(full_prompt,
                         logits_size, 
                         tokens_size, 
                         n_past, 
                         n_ctx, 
                         n_predict, 
                         top_k, 
                         top_p, 
                         temp, 
                         n_batch, 
                         repeat_penalty, 
                         repeat_last_n, 
                         context_erase)
        response = collect_response.getvalue()
        sys.stdout = old_stdout

        response = re.sub(r"\n(?!\n)", "", response)

        if verbose:
            print(response)
        
        # Remove the unnecessary new lines from response
        return response

    cdef void _generate_c(self, 
                          prompt, # utf-8 bytestring
                          size_t logits_size, 
                          size_t tokens_size, 
                          int32_t n_past, 
                          int32_t n_ctx, 
                          int32_t n_predict, 
                          int32_t top_k, 
                          float top_p, 
                          float temp, 
                          int32_t n_batch, 
                          float repeat_penalty, 
                          int32_t repeat_last_n, 
                          float context_erase):
        """
        C level defined function to wrap generate arguments
        into a prompt context and pass C callback functions.
        """
    
        cdef llmodel_prompt_context prompt_context
    
        prompt_context.logits_size = logits_size
        prompt_context.tokens_size = tokens_size
        prompt_context.n_past = n_past
        prompt_context.n_ctx = n_ctx
        prompt_context.n_predict = n_predict
        prompt_context.top_k = top_k
        prompt_context.top_p = top_p
        prompt_context.temp = temp
        prompt_context.n_batch = n_batch
        prompt_context.repeat_penalty = repeat_penalty
        prompt_context.repeat_last_n = repeat_last_n
        prompt_context.context_erase = context_erase

        llmodel_prompt(self.model, 
                       prompt, 
                       self._prompt_callback, 
                       self._response_callback, 
                       self._recalculate_callback, 
                       &prompt_context)

    # Empty prompt callback
    @staticmethod
    cdef bool _prompt_callback(int32_t token_id, const char* response):
        return True

    # Empty response callback method that just prints response to be collected
    @staticmethod
    cdef bool _response_callback(int32_t token_id, const char* response):
        print(response.decode('utf-8'))
        return True

    # Empty recalculate callback
    @staticmethod
    cdef bool _recalculate_callback(bool is_recalculating):
        return is_recalculating


cdef class GPTJModel(LLModel):

    model_type = "gptj"

    def __cinit__(self):
        self.model = llmodel_gptj_create()

    def __dealloc__(self):
        llmodel_gptj_destroy(self.model)


cdef class LlamaModel(LLModel):

    model_type = "llama"

    def __cinit__(self):
        self.model = llmodel_llama_create()

    def __dealloc__(self):
        llmodel_llama_destroy(self.model)


def test():
    gptj_model = GPTJModel()
    gptj_model.load_model("models_temp/ggml-gpt4all-j-v1.3-groovy.bin")  
    response = gptj_model.generate("Hello there")
    print(response)    
    response = gptj_model.generate("Can you list 5 different colors?")
    print(response)
    
    return