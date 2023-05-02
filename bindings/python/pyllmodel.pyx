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

    def load_model(self, model_path: str):
        llmodel_loadModel(self.model, model_path.encode('utf-8'))

        if llmodel_isModelLoaded(self.model):
            return True
        else:
            return False

    cdef void _generate_c(self, str prompt):
        cdef llmodel_prompt_context prompt_context
    
        prompt_context.logits_size = 0
        prompt_context.tokens_size = 0
        prompt_context.n_past = 0
        prompt_context.n_ctx = 1024
        prompt_context.n_predict = 128
        prompt_context.top_k = 40
        prompt_context.top_p = 0.9
        prompt_context.temp = .9
        prompt_context.n_batch = 8
        prompt_context.repeat_penalty = 1.2
        prompt_context.repeat_last_n = 10
        prompt_context.context_erase = 0.5

        full_prompt = "### Instruction:\n The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response."
        full_prompt += f"\n### Prompt: {prompt}"
        full_prompt += "\n### Response:"
        full_prompt = full_prompt.encode('utf-8')
        llmodel_prompt(self.model, 
                       full_prompt, 
                       empty_prompt_callback, 
                       empty_response_callback, 
                       empty_recalculate_callback, 
                       &prompt_context)

    def generate(self, prompt: str):
        old_stdout = sys.stdout 
        collect_response = StringIO()
        sys.stdout  = collect_response
        
        self._generate_c(prompt)
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
    response = gptj_model.generate("hello there")    
    
    return response