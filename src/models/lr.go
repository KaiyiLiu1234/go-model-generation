package models

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	regression "github.com/sajari/regression"
	"gonum.org/v1/gonum/stat"
)

type LinearRegressorData struct {
	LinearRegressor               *regression.Regression
	ModelFeatures                 ModelFeatureType
	ModelLabel                    ModelLabelType
	NodeLevel                     bool
	ModelName                     string
	IncludeMultiplierFeatureCross bool
}

// LR Model is generalized
// Optional for Instantiating LR Model
func (lr *LinearRegressorData) NewLinearRegressorData(modelFeatures ModelFeatureType, modelLabel ModelLabelType, nodeLevel bool, includeMultiplierFeatureCross bool) {
	lr.LinearRegressor = new(regression.Regression)
	lr.ModelFeatures = modelFeatures
	lr.ModelLabel = modelLabel
	lr.LinearRegressor.SetObserved(Model_Label_Groups[modelLabel][0])
	for index, feature := range Model_Feature_Groups[modelFeatures] {
		lr.LinearRegressor.SetVar(index, feature)
	}
	lr.NodeLevel = nodeLevel
	nodeString := ""
	if nodeLevel {
		nodeString = "Node_Level"
	} else {
		nodeString = "Container_Level"
	}
	IncludeMultiplier := ""
	if includeMultiplierFeatureCross {
		IncludeMultiplier = "IncludeMultiplier"
	} else {
		IncludeMultiplier = "NoMultiplier"
	}
	modelName := fmt.Sprintf("%v_%v_%s_%s.json", modelFeatures, modelLabel, nodeString, IncludeMultiplier)
	lr.ModelName = modelName
	lr.IncludeMultiplierFeatureCross = includeMultiplierFeatureCross

}

// assume that each nested array is a feature row in the data
// start index and end index are the features to normalize
// start index is inclusive and end index is exclusive
func normalizeFeatureData(data [][]float64, startIndex int, endIndex int) {
	rowNum := len(data)
	colNum := len(data[startIndex:endIndex])
	for col := 0; col < colNum; col++ {
		standardDeviation := stat.StdDev(data[col], nil)
		mean := stat.Mean(data[col], nil)
		// normalize data in place
		for row := 0; row < rowNum; row++ {
			data[row][col] = (data[row][col] - mean) / standardDeviation
		}
	}
}

// Preconditions:
// LinearRegressorData is properly instantiated
// features are presented in same order as the features added to Model according to ModelFeatureType (alphabetical ascending order)
// last column is the label data
// each slice is the same length
func (lr *LinearRegressorData) TrainModel(model_feature_and_label_data_in_order [][]float64) error {
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

	// Normalize feature data
	normalizeFeatureData(model_feature_and_label_data_in_order, 0, len(model_feature_and_label_data_in_order[0])-1)

	datapoints := regression.MakeDataPoints(model_feature_and_label_data_in_order, numOfFeaturesandLabels-1)
	lr.LinearRegressor.Train(datapoints...)

	// Retrieve variables
	if lr.IncludeMultiplierFeatureCross {
		featuresToCross := make([]int, numOfFeaturesandLabels-1)
		for i := 0; i < numOfFeaturesandLabels-1; i++ {
			featuresToCross[i] = i
		}
		//fmt.Print(featuresToCross)
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
func SaveModel(path string, lr LinearRegressorData) error {
	jsonFile, err := json.Marshal(lr)
	if err != nil {
		return err
	}
	err = os.WriteFile(filepath.Join(path, lr.ModelName), jsonFile, 0644)
	if err != nil {
		return err
	}
	return nil
}

func LoadModel(path string) (LinearRegressorData, error) {
	jsonFile, err := os.ReadFile(path)
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
