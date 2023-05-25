package models

import (
	"fmt"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
)

func GenerateInitializeIRQModel(model_filepath string) {
	// initialize model
	newLRModel := LinearRegressorData{}
	newLRModel.NewLinearRegressorData(irqFeatures, totalPackagePower)
	// access local csv

	irqDataFile, err := os.Open(model_filepath)
	if err != nil {
		log.Fatal(err)
	}
	defer irqDataFile.Close()
	irqDf := dataframe.ReadCSV(irqDataFile)

	featuresDf := irqDf.Select(Model_Feature_Groups[irqFeatures])
	fmt.Print(featuresDf)
	/*for _, v := range featuresDf.Maps() {
		inner := make([]float64, len(Model_Feature_Groups[irqFeatures]))
		for index, key := range Model_Feature_Groups[irqFeatures]{
			inner[index] = float64(v[key])
		}
	}*/
	// retrieve package power metrics
	//

}

func AccessXGBoostModel() {

}
