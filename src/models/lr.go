package models

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	regression "github.com/sajari/regression"
)

type LinearRegressorData struct {
	LinearRegressor *regression.Regression
	ModelFeatures   ModelFeatureType
	ModelLabel      ModelLabelType
}

// Optional for Instantiating LR Model
func (lr *LinearRegressorData) NewLinearRegressorData(modelFeatures ModelFeatureType, modelLabel ModelLabelType) {
	lr.LinearRegressor = new(regression.Regression)
	lr.ModelFeatures = modelFeatures
	lr.ModelLabel = modelLabel
	lr.LinearRegressor.SetObserved(Model_Label_Groups[modelLabel][0])
	for index, feature := range Model_Feature_Groups[modelFeatures] {
		lr.LinearRegressor.SetVar(index, feature)
	}

}

// Preconditions:
// LinearRegressorData is properly instantiated
// features are presented in same order as the features added to Model according to ModelFeatureType (alphabetical ascending order)
// last column is the label data
// each slice is the same length
func (lr *LinearRegressorData) TrainModel(model_feature_and_label_data_in_order [][]float64, includeMultiplierFeatureCross bool) error {
	if len(model_feature_and_label_data_in_order) <= 2 {
		return errors.New("not enough data")
	}
	numOfFeaturesandLabels := len(model_feature_and_label_data_in_order[0])
	if numOfFeaturesandLabels < 2 {
		return errors.New("missing feature(s) data and/or label data")
	}
	// Check if number of columns is correct
	if numOfFeaturesandLabels-1 != len(Model_Feature_Groups[lr.ModelFeatures]) {
		return errors.New("number of features in data do not match with total number of features for model")
	}
	datapoints := regression.MakeDataPoints(model_feature_and_label_data_in_order, numOfFeaturesandLabels-1)
	lr.LinearRegressor.Train(datapoints...)

	// Retrieve variables
	if includeMultiplierFeatureCross {
		featuresToCross := make([]int, numOfFeaturesandLabels)
		for i := 0; i < numOfFeaturesandLabels; i++ {
			featuresToCross[i] = i
		}
		lr.LinearRegressor.AddCross(regression.MultiplierCross(featuresToCross...))
	}
	err := lr.LinearRegressor.Run()
	if err != nil {
		return err
	}
	return nil
}

func (lr *LinearRegressorData) RetrieveCoefficients() []float64 {
	return lr.LinearRegressor.GetCoeffs()
}

func (lr *LinearRegressorData) RetrieveR2() float64 {
	return lr.LinearRegressor.R2
}

func (lr *LinearRegressorData) RetrieveRegressionFormula() string {
	return lr.LinearRegressor.Formula
}

// first index is variance second index is predicted variance
func (lr *LinearRegressorData) RetrieveVariances() []float64 {
	return []float64{lr.LinearRegressor.Varianceobserved, lr.LinearRegressor.VariancePredicted}
}

func (lr *LinearRegressorData) Predict(prediction map[string]float64) (float64, error) {
	featuresList := Model_Feature_Groups[lr.ModelFeatures]
	values := make([]float64, len(featuresList))
	for index, feature := range featuresList {
		values[index] = prediction[feature]
	}
	predictedValue, err := lr.LinearRegressor.Predict(values)
	if err != nil {
		return -1, err
	}
	return predictedValue, nil
}

// Only Save Model when properly instantiated
func SaveModel(model_name string, lr LinearRegressorData) error {
	jsonFile, err := json.Marshal(lr)
	if err != nil {
		return err
	}
	err = os.WriteFile(fmt.Sprintf("saved_models/%s.json", model_name), jsonFile, 0644)
	if err != nil {
		return err
	}
	return nil
}

func LoadModel(model_name string) (LinearRegressorData, error) {
	jsonFile, err := os.ReadFile(model_name)
	if err != nil {
		return LinearRegressorData{}, err
	}

	lrModel := LinearRegressorData{}

	err = json.Unmarshal(jsonFile, &lrModel)
	if err != nil {
		return LinearRegressorData{}, err
	}
	return lrModel, nil
}

//Experimental Incremental Training Methods
/*
func (lr *LinearRegressorData) OnlineTrainModel(model_feature_data_in_order []float64, model_label_data_in_order float64) error {

}*/
