#include <stdio.h>
#include <stdbool.h>
#include <xgboost/c_api.h>
#include <string.h>

const int num_of_features = 4;

typedef struct XGBoostRegressor{
	BoosterHandle XGBBooster;
} XGBoostRegressor ;

int LoadXGBoostRegressor (XGBoostRegressor *new_regressor_data, char xgb_model_filepath[]) {
	BoosterHandle XGBbooster;
    int ret = XGBoosterCreate(0, 0, &XGBbooster);
    if (ret != 0) {
        printf("Error: XGBoosterCreate failed with return value: %d\n", ret);
        return ret;
    }
    ret = XGBoosterLoadModel(XGBbooster, xgb_model_filepath);
    if (ret != 0) {
        printf("Error: XGBoosterLoadModel failed with return value: %d\n", ret);
        return ret;
    }

	new_regressor_data->XGBBooster = XGBbooster;
	return 0;

}

float PredictXGBoostRegressor (XGBoostRegressor complete_regressor_data, float features[1][num_of_features]) {

    DMatrixHandle dmatrix;
    int ret = XGDMatrixCreateFromMat((float *)features, 1, num_of_features, 0, &dmatrix);
    if (ret != 0) {
        printf("Error: XGDMatrixCreateFromMat failed with return value: %d\n", ret);
        return ret;
    }

    bst_ulong out_len;
    const float *out_result;
    ret = XGBoosterPredict(complete_regressor_data.XGBBooster, dmatrix, 0, 0, false, &out_len, &out_result);
    if (ret != 0) {
        printf("Error: XGBoosterPredict failed with return value: %d\n", ret);
        return ret;
    }
	float prediction = out_result[0];
    for (bst_ulong i = 0; i < out_len; ++i) {
        printf("Prediction for instance %lu: %f\n", i, out_result[i]);
    }
    XGDMatrixFree(dmatrix);
	return prediction;
}

int FreeXGBoostRegressor( XGBoostRegressor complete_regressor_data) {
	int ret = XGBoosterFree(complete_regressor_data.XGBBooster);
	if (ret != 0 ){
		printf("Error: XGBoosterFree failed with return value: %d\n", ret);
		return ret;
	}
	return 0;
}

int main(){
	struct XGBoostRegressor myRegressor;
	int ret = LoadXGBoostRegressor(&myRegressor, "XGBoost/XGBoostRegressionStandalonePipeline_Node_Level_KFoldCrossValidation_package/XGBoostRegressionStandalonePipeline_Node_Level_KFoldCrossValidation.model");
	float features_one[1][4] = {{276099, 138350277, 1520717, 4191}};
    float features_two[1][4] = {{28372, 138350277, 1538823, 5637}};
	float result_one = PredictXGBoostRegressor(myRegressor, features_one);
    float result_two = PredictXGBoostRegressor(myRegressor, features_two);
	printf("Prediction One: %f, Prediction Two: %f", result_one, result_two);
	FreeXGBoostRegressor(myRegressor);	
	return 0;
}


