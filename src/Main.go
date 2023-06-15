package main

import (
	"src/github.com/KaiyiLiu1234/src/models"
)

func main() {
	models.GenerateInitializeLRBPFIRQModel("models/refined_model_data/", "models/saved_models/LR/", true, false)
	models.GenerateInitializeLRBPFIRQModel("models/refined_model_data/", "models/saved_models/LR/", true, true)
	models.GenerateInitializeLRBPFIRQModel("models/refined_model_data/", "models/saved_models/LR/", false, true)
	models.GenerateInitializeLRBPFIRQModel("models/refined_model_data/", "models/saved_models/LR/", false, false)

	//retrievedRegressorTrueKFold := models.RetrieveIRQBPFXGBoostModel("models/saved_models/XGBoost/", true, models.KFoldCrossValidation)
	//retrievedRegressorTrueTrainTest := models.RetrieveIRQBPFXGBoostModel("models/saved_models/XGBoost/", true, models.TrainTestSplit)
	//retrievedRegressorFalseKFold := models.RetrieveIRQBPFXGBoostModel("models/saved_models/XGBoost/", false, models.KFoldCrossValidation)
	//retrievedRegressorFalseTrainTest := models.RetrieveIRQBPFXGBoostModel("models/saved_models/XGBoost/", false, models.TrainTestSplit)

	//fmt.Print(retrievedRegressorFalseKFold)
	//fmt.Print(retrievedRegressorFalseTrainTest)
	//fmt.Print(retrievedRegressorTrueKFold)
	//fmt.Print(retrievedRegressorTrueTrainTest)

}
