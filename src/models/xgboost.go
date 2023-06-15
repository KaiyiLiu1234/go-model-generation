package models

import (
	"fmt"
	"path/filepath"

	"github.com/dmitryikh/leaves"
)

// xgboost models are incapable of incremental training. It can only use pretrained xgboost models.

type XGBoostTrainingType int64

const (
	TrainTestSplit XGBoostTrainingType = iota
	KFoldCrossValidation
)

type XGBoostRegressor struct {
	LoadedXGBRegressor *leaves.Ensemble
	ModelFeatures      ModelFeatureType
	ModelLabel         ModelLabelType
	NodelLevel         bool
	TrainingType       XGBoostTrainingType
}

func (xgb *XGBoostRegressor) NewXGBoostRegressor(xgbModelFilepath string, ModelFeatures ModelFeatureType, ModelLabel ModelLabelType, nodeLevel bool, trainingType XGBoostTrainingType) error {
	nodeString := ""
	if nodeLevel {
		nodeString = "Node_Level"
	} else {
		nodeString = "Container_Level"
	}
	trainingString := ""
	if trainingType == TrainTestSplit {
		trainingString = "TrainTestSplitFit"
	} else {
		trainingString = "KFoldCrossValidation"
	}
	// Need to generalize XGBoostRegressor
	modelFolderName := fmt.Sprintf("XGBoostRegressionStandalonePipeline_%s_%s_package", nodeString, trainingString)
	modelName := fmt.Sprintf("XGBoostRegressionStandalonePipeline_%s_%s.model", nodeString, trainingString)
	model, err := leaves.XGBLinearFromFile(filepath.Join(xgbModelFilepath, modelFolderName+"/", modelName), false)
	if err != nil {
		fmt.Print(err)
		return err
	}
	xgb.LoadedXGBRegressor = model
	xgb.ModelFeatures = ModelFeatures
	xgb.ModelLabel = ModelLabel
	xgb.NodelLevel = nodeLevel
	xgb.TrainingType = trainingType
	return nil
}

func (xgb *XGBoostRegressor) Predict(prediction map[string]float64) float64 {
	featuresList := Model_Feature_Groups[xgb.ModelFeatures]
	values := make([]float64, len(featuresList))
	for index, feature := range featuresList {
		values[index] = prediction[feature]
	}
	predictedValue := xgb.LoadedXGBRegressor.PredictSingle(values, 0)

	return predictedValue
}
