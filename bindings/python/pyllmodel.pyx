from libcpp cimport bool

cdef extern from "llmodel_c.h":
    ctypedef void *llmodel_model
    ctypedef bool (*llmodel_prompt_callback)(int32_t)
    ctypedef bool (*llmodel_response_callback)(int32_t, const char *)

    llmodel_model llmodel_gptj_create()
    void llmodel_gptj_destroy(llmodel_model gptj)
    llmodel_model llmodel_llama_create()
    llmodel_model llmodel_llama_destroy()
    bool llmodel_loadModel(llmodel_model model, const char *model_path)
    bool llmodel_isModelLoaded(llmodel_model model)


def create_and_destroy_gptj_model():
    gpjt_model = llmodel_gptj_create()
    llmodel_gptj_destroy(gpjt_model)

def load_gptj_model(model_path: str):
    model_path_bytestring = model_path.encode()
    gptj_model = llmodel_gptj_create()
    llmodel_loadModel(gptj_model, model_path_bytestring)
    return gptj_model

def load_llama_model(model_path: str):
    model_path_bytestring = model_path.encode()
    llama_model = llmodel_gptj_create()
    llmodel_loadModel(llama_model, model_path_bytestring)
    return llama_model


