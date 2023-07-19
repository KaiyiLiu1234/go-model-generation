package gomodeling

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
)

// assume mxn array
func convertStringRowsToFloatColumns(data [][]string) ([][]float64, error) {
	if len(data) < 1 {
		return [][]float64{}, errors.New("not enough data")
	}

	convertedArray := make([][]float64, len(data[0]))

	for i := 0; i < len(data[0]); i++ {
		row := []float64{}
		for j := 0; j < len(data); j++ {
			result, err := strconv.ParseFloat(data[j][i], 32)
			if err != nil {
				return [][]float64{}, errors.New("data contains string that cannot be parsed to float")
			}
			row = append(row, result)
		}
		convertedArray[i] = row

	}

	return convertedArray, nil
}

func sumArraysAndConvertString(columns [][]float64) ([]string, error) {
	var summedArray []string
	if len(columns) < 2 {
		return summedArray, errors.New("not enough columns")
	}
	for i, _ := range columns[0] {
		summedVal := 0.0
		for j, _ := range columns {
			summedVal = summedVal + columns[j][i]
		}
		summedArray = append(summedArray, strconv.FormatFloat(summedVal, 'E', -1, 64))
	}
	return summedArray, nil
}

func GenerateInitializeLRBPFIRQModel(localRefinedDataFilepath string, modelFilepath string, nodeLevel bool, includeMultiplierFeatureCross bool) LinearRegressorData {
	// initialize model
	newLRModel := LinearRegressorData{}
	newLRModel.NewLinearRegressorData(irqFeatures, totalPackagePower, nodeLevel, includeMultiplierFeatureCross)
	// access local refined csv
	// purpose of refined csv is to just track the entire process
	nodeString := ""
	if nodeLevel {
		nodeString = "Node_Level"
	} else {
		nodeString = "Container_Level"
	}
	fileName := fmt.Sprintf("%v_%v_%s.csv", irqFeatures, totalPackagePower, nodeString)
	irqDataFile, err := os.Open(filepath.Join(localRefinedDataFilepath, fileName))
	if err != nil {
		fmt.Print(err)
		log.Fatal(err, "\nrefined data for desired model type does not exist")
	}
	defer irqDataFile.Close()

	irqDf := dataframe.ReadCSV(irqDataFile)
	// ensure features and labels are organized prior to transposing
	featuresDataInOrder := make([][]string, len(Model_Feature_Groups[irqFeatures])+1)
	for index, feature := range Model_Feature_Groups[irqFeatures] {
		featuresDataInOrder[index] = irqDf.Col(feature).Records()
	}
	// Retrieve all Package Power
	var packagePowers []string
	for _, name := range irqDf.Names() {
		if strings.HasSuffix(name, "_package_power") {
			packagePowers = append(packagePowers, name)

		}
	}
	packagePowerDataInOrder := make([][]float64, len(packagePowers))
	for index, name := range packagePowers {
		stringPowerData := irqDf.Col(name).Records()
		floatPowerData := make([]float64, len(stringPowerData))
		for index, value := range stringPowerData {
			returnedVal, err := strconv.ParseFloat(value, 64)
			if err != nil {
				fmt.Print(err)
				log.Fatal(err)
			}
			floatPowerData[index] = returnedVal
		}
		packagePowerDataInOrder[index] = floatPowerData
	}
	powerLabel, err := sumArraysAndConvertString(packagePowerDataInOrder)
	if err != nil {
		fmt.Print(err)
		log.Fatal(err)
	}
	featuresDataInOrder[len(Model_Feature_Groups[irqFeatures])] = powerLabel
	trainingData, err := convertStringRowsToFloatColumns(featuresDataInOrder)
	if err != nil {
		fmt.Print(err)
		log.Fatal(err)
	}
	// Train LR Model
	err = newLRModel.TrainModel(trainingData)
	if err != nil {
		fmt.Print(err)
		log.Fatal(err)
	}
	err = SaveModel(modelFilepath, newLRModel)
	if err != nil {
		fmt.Print(err)
		log.Fatal(err)
	}
	return newLRModel
}

// func RetrieveIRQBPFXGBoostModel(modelFilepath string, nodeLevel bool, trainingType XGBoostTrainingType) XGBoostRegressor {
// 	// Retrieve Model
// }

/*func PredictOnTFKeras() {
	test := TensorflowKerasLinearRegressor{}
	test.LoadTFKerasLinearRegressor("models/saved_models/LR/AbsComponentModelWeight/core", irqFeatures, totalPackagePower)
	predictions := []float64{1, 2, 3, 4}
	test.PredictTFKerasLinearRegressor(predictions)
}*/
