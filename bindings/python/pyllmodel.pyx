from libcpp cimport bool

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
    llmodel_model llmodel_llama_destroy()
    bool llmodel_loadModel(llmodel_model model, const char *model_path)
    bool llmodel_isModelLoaded(llmodel_model model)
    void llmodel_prompt(llmodel_model model, const char *prompt,
                    llmodel_response_callback prompt_callback,
                    llmodel_response_callback response_callback,
                    llmodel_recalculate_callback recalculate_callback,
                    llmodel_prompt_context *ctx)
"""
cdef class GPTJModel:
    cdef llmodel_model model
    
    def __cinit__(self):
        self.model = llmodel_gptj_create()

    def __dealloc__(self):
        llmodel_gptj_destroy(self.model)

    cdef llmodel_model get_model(self):
        return self.model
"""

def create_and_destroy_gptj_model():
    gpjt_model = llmodel_gptj_create()
    llmodel_gptj_destroy(gpjt_model)

cdef llmodel_model load_gptj_model(model_path: str):
    model_path_bytestring = model_path.encode()
    gptj_model = llmodel_gptj_create()
    llmodel_loadModel(gptj_model, model_path_bytestring)
    return gptj_model

cdef bool empty_prompt_callback(int32_t token_id):
    return True

cdef bool empty_response_callback(int32_t token_id, const char* response):
    if response == b"###":
        return False
    return True

cdef bool empty_recalculate_callback(bool is_recalculating):
    return is_recalculating

cdef void load_and_prompt_gptj(model_path: str):
    gptj_model = load_gptj_model(model_path)

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

    prompt = "hello".encode('utf-8')
    print("TIME TO PROMPT")
    llmodel_prompt(gptj_model, prompt, empty_response_callback, empty_response_callback, empty_recalculate_callback, &prompt_context)

def python_load_and_prompt_gpt(model_path: str):
    load_and_prompt_gptj(model_path)
