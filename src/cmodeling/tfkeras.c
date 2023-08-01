#include <tensorflow/c/c_api.h>
#include <stdio.h>


typedef struct TFKerasRegressor{
	TF_Graph *tfkgraph;
    TF_Session *tfksession;
    TF_SessionOptions *tfkoptions;
    TF_Status *tfkstatus; // status for both loading and predicting on session

} TFKerasRegressor ;


int LoadTFKSavedModelRegressor (TFKerasRegressor *new_regressor_data, char tfkeras_model_filepath[]) {
    TF_SessionOptions *session_options = TF_NewSessionOptions();
    TF_Status *status = TF_NewStatus();
    TF_Graph *graph = TF_NewGraph();
    const char* tags[] = { "serve" };
    TF_Session *session = TF_LoadSessionFromSavedModel(session_options, NULL, tfkeras_model_filepath, tags, 1, graph, NULL, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("Error loading SavedModel: %s\n", TF_Message(status));
        return 1;
    }
    
    new_regressor_data->tfkgraph = graph;
    new_regressor_data->tfksession = session;
    new_regressor_data->tfkstatus = status;
    new_regressor_data->tfkoptions = session_options;

    return 0;
}

void freeTensorBuffer(void* data, size_t len, void* arg) {
    free(data);
    printf("hahahahah");
}

float PredictTFKRegressor (TFKerasRegressor *complete_regressor_data, char **features_float, int float_feature_len, char **label, int label_len, char **features_str, int str_feature_len, float float_features_predict[float_feature_len], char **str_features_predict) {

    TF_Output input_nodes[float_feature_len + str_feature_len];
    TF_Tensor *input_tensors[float_feature_len + str_feature_len];
    TF_Graph *graph = complete_regressor_data->tfkgraph;
    TF_Session *session = complete_regressor_data->tfksession;
    TF_Status *status = complete_regressor_data->tfkstatus;
    TF_SessionOptions *session_opts = complete_regressor_data->tfkoptions;
    const int64_t input_dims[] = {1}; 
    const int ndims = 1;
    
    for (int i = 0; i < float_feature_len + str_feature_len; i++){
        if (i >= float_feature_len + str_feature_len - 1){
            input_nodes[i].oper = TF_GraphOperationByName(graph, features_str[i]);
            input_nodes[i].index = 0;
            const char* input_strings[1] = {str_features_predict[i]};
            input_tensors[i] = TF_NewTensor(TF_STRING, NULL, 0, input_strings, sizeof(char*), &freeTensorBuffer, NULL);
        } else {
            input_nodes[i].oper = TF_GraphOperationByName(graph, features_float[i]);
            input_nodes[i].index = 0;
            //input_tensors[i] = TF_NewTensor(TF_FLOAT, input_dims, 1, &float_features_predict[i], sizeof(float), my_deallocator, NULL);
            const size_t tensorSize = sizeof(float);
            void* tensorBuffer = malloc(tensorSize);
            memcpy(tensorBuffer, &float_features_predict[i], tensorSize);
            TF_Tensor* tensor = TF_NewTensor(TF_FLOAT, input_dims, ndims, tensorBuffer, tensorSize, &freeTensorBuffer, NULL);

            if (tensor == NULL) {
                printf("Failed to create the tensor.\n");
                return 1;
            }
            input_tensors[i] = tensor;
            float* tensorValue = (float*)TF_TensorData(tensor);
            printf("Float value: %f\n", *tensorValue);

        }
    }
    
    const int num_outputs = label_len;
    TF_Output output_nodes[num_outputs];
    output_nodes[0].oper = TF_GraphOperationByName(graph, label[0]);
    output_nodes[0].index = 0;
    
    
    TF_Tensor* output_tensor;
    TF_SessionRun(session, NULL, input_nodes, input_tensors, float_feature_len + str_feature_len,
                  output_nodes, &output_tensor, num_outputs,
                  NULL, 0, NULL, status);
    
    float final_prediction = -1.0;

    if (TF_GetCode(status) == TF_OK) {
        float* result_data = (float*)TF_TensorData(output_tensor);
        printf("Prediction result: %f\n", *result_data);
        final_prediction = *result_data;
        TF_DeleteTensor(output_tensor);
    } else {
        printf("Error running inference: %s\n", TF_Message(status));
    }
    for (int i = 0; i < float_feature_len + str_feature_len; ++i) {
        TF_DeleteTensor(input_tensors[i]);
    }

    return final_prediction;

}

int FreeTFKRegressor(TFKerasRegressor *complete_regressor_data){
    TF_DeleteGraph(complete_regressor_data->tfkgraph);
    TF_CloseSession(complete_regressor_data->tfksession, complete_regressor_data->tfkstatus);
    TF_DeleteSession(complete_regressor_data->tfksession, complete_regressor_data->tfkstatus);
    TF_DeleteStatus(complete_regressor_data->tfkstatus);

    TF_DeleteSessionOptions(complete_regressor_data->tfkoptions);

} 

float LoadExistingTFKerasModelandPredict(char tfkeras_model_filepath[], char **features_float, int float_feature_len, char **label, int label_len, char **features_str, int str_feature_len, float *float_features_predict, char **str_features_predict){
    struct TFKerasRegressor tfkRegressor;
    int ret = LoadTFKSavedModelRegressor(&tfkRegressor, tfkeras_model_filepath);
    if (ret == 1){
        return -1;
    }
    float result = PredictTFKRegressor(&tfkRegressor, features_float, float_feature_len, label, label_len, features_str, str_feature_len, float_features_predict, str_features_predict);
    if (result == -1){
        return -1;
    }
    FreeTFKRegressor(&tfkRegressor);
    //return result;
    return -1;
}


int main() {
    
    int features_float_len = 3;
    int str_features_len = 1;
    int label_len = 1;
    char *features_float[] = {
    "serving_default_core_cpu_cycles:0",
    "serving_default_core_cpu_instr:0",
    "serving_default_core_cpu_time:0",
    };
    char *features_str[] = {
        "serving_default_core_cpu_architecture:0",
    };
    char *label[] = {
        "StatefulPartitionedCall_2:0",
    };

    float float_features_predict[] = {
        18382,
        104,
        818383,
    };
    char *str_features_predict[] = {
        "Ivy Bridge"
    };
    char model_loc[ ] = "AbsComponentModelWeight/core";
    float result = LoadExistingTFKerasModelandPredict(model_loc, features_float, features_float_len, label, label_len, features_str, str_features_len,float_features_predict, str_features_predict);
    printf("Prediction: %f", result);
    return 0;

}
