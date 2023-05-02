from libcpp cimport bool

from io import StringIO
import re
import sys

cdef extern from "<stdint.h>":
    ctypedef unsigned char uint8_t
    ctypedef signed char int8_t
    ctypedef unsigned short uint16_t
    ctypedef signed short int16_t
    ctypedef unsigned int uint32_t
    ctypedef signed int int32_t
    ctypedef unsigned long long uint64_t
    ctypedef signed long long int64_t

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

# Base model class
cdef class LLModel:
    cdef llmodel_model model
    model_type: str

    def load_model(self, model_path: str) -> bool:
        llmodel_loadModel(self.model, model_path.encode('utf-8'))

        if llmodel_isModelLoaded(self.model):
            return True
        else:
            return False

    cdef void _generate_c(self, 
                          prompt, # utf-8 bytestring
                          size_t logits_size = 0, 
                          size_t tokens_size = 0, 
                          int32_t n_past = 0, 
                          int32_t n_ctx = 1024, 
                          int32_t n_predict = 128, 
                          int32_t top_k = 40, 
                          float top_p = .9, 
                          float temp = .9, 
                          int32_t n_batch = 8, 
                          float repeat_penalty = 1.2, 
                          int32_t repeat_last_n = 10, 
                          float context_erase = .5):
    
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
                       empty_prompt_callback, 
                       empty_response_callback, 
                       empty_recalculate_callback, 
                       &prompt_context)

    def generate(self, 
                 prompt: str,
                 add_default_header: bool = True, 
                 add_default_footer: bool = True) -> str:

        full_prompt = ""
        if add_default_header:
            full_prompt += """### Instruction:\n 
            The prompt below is a question to answer, a task to complete, or a conversation 
            to respond to; decide which and write an appropriate response.
            \n### Prompt: """
        
        full_prompt += prompt

        if add_default_footer:
            full_prompt += "\n### Response:"
        
        full_prompt = full_prompt.encode('utf-8')

        old_stdout = sys.stdout 
        collect_response = StringIO()
        sys.stdout  = collect_response
        
        self._generate_c(full_prompt)
        response = collect_response.getvalue()
        sys.stdout = old_stdout
        return re.sub(r"\n(?!\n)", "", response)


cdef class GPTJModel(LLModel):

    def __cinit__(self):
        self.model = llmodel_gptj_create()

    def __dealloc__(self):
        llmodel_gptj_destroy(self.model)

    def __init__(self):
        self.model_type = "gptj"


cdef class LlamaModel(LLModel):

    def __cinit__(self):
        self.model = llmodel_llama_create()

    def __dealloc__(self):
        llmodel_llama_destroy(self.model)

    def __init__(self):
        self.model_type = "llama"


cdef bool empty_prompt_callback(int32_t token_id, const char* response):
    return True

cdef bool empty_response_callback(int32_t token_id, const char* response):
    print(response.decode('utf-8'))
    return True

cdef bool empty_recalculate_callback(bool is_recalculating):
    return is_recalculating

def test():
    gptj_model = GPTJModel()
    gptj_model.load_model("models_temp/ggml-gpt4all-j-v1.3-groovy.bin")  
    response = gptj_model.generate("Hello there")
    print(response)    
    response = gptj_model.generate("Can you list 5 colors?")
    print(response)
    
    return