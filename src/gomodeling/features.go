package gomodeling

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Note the full kepler metric cannot be used here to maintain consistency (follow model server)
var irqRelatedMetrics = []string{"bpf_block_irq", "bpf_cpu_time_us", "bpf_net_rx_irq", "bpf_net_tx_irq"}
var irqPowerLabel = []string{"total_package_power"}
var coreMetrics = []string{"core_cpu_cycles", "core_cpu_instr", "core_cpu_time", "core_cpu_architecture"}

type DataLocation int64

const (
	Prometheus DataLocation = iota
	Local
)

type ModelFeatureType string

const (
	irqFeatures  ModelFeatureType = "irqFeatures"
	coreFeatures ModelFeatureType = "coreFeatures"
)

type ModelLabelType string

const (
	totalPackagePower         ModelLabelType = "totalPackagePower"
	coreRegressionLayerEnergy ModelLabelType = "coreRegressionLayerEnergy"
)

func sort_model_names(features []string) []string {
	copied_features := make([]string, len(features))
	copy(copied_features, features)
	sort.Strings(copied_features)
	return copied_features
}

var Model_Feature_Groups = map[ModelFeatureType][]string{
	irqFeatures: sort_model_names(irqRelatedMetrics),
}

var Model_Label_Groups = map[ModelLabelType][]string{
	totalPackagePower: sort_model_names(irqPowerLabel),
}

// assume the last column is the label
func ConvertCSVToLibsvm(csvFilepath string, libsvmLocationFilepath string) error {
	file, err := os.Open(csvFilepath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var features [][]float64
	var labels []float64

	for _, row := range rows {
		if len(row) < 2 {
			return errors.New("not enough columns for feature(s) and label")
		}
		// process last label as column
		label, err := strconv.ParseFloat(row[len(row)-1], 64)
		if err != nil {
			log.Fatal(err)
		}
		labels = append(labels, label)

		// process the features (the rest of the columns)
		var featureRow []float64
		for _, valStr := range row[:len(row)-2] {
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				log.Fatal(err)
			}
			featureRow = append(featureRow, val)
		}
		features = append(features, featureRow)
	}

	featureMatrix := mat.NewDense(len(features), len(features[0]), nil)
	for i, featureRow := range features {
		featureMatrix.SetRow(i, featureRow)
	}

	_, cols := featureMatrix.Dims()

	meanVals := make([]float64, cols)
	for j := 0; j < cols; j++ {
		column := mat.Col(nil, j, featureMatrix)
		meanVals[j] = stat.Mean(column, nil)
	}

	stdDevVals := make([]float64, cols)
	for j := 0; j < cols; j++ {
		column := mat.Col(nil, j, featureMatrix)
		stdDevVals[j] = stat.StdDev(column, nil)
	}

	for i, featureRow := range features {
		for j, val := range featureRow {
			features[i][j] = (val - meanVals[j]) / stdDevVals[j]
		}
	}

	outputFile, err := os.Create(libsvmLocationFilepath)
	if err != nil {
		log.Fatal(err)
	}
	defer outputFile.Close()

	writer := bufio.NewWriter(outputFile)
	for i, featureRow := range features {
		label := labels[i]
		labelStr := strconv.FormatFloat(label, 'f', -1, 64)

		featureStrs := []string{}
		for j, val := range featureRow {
			featureStr := strconv.Itoa(j+1) + ":" + strconv.FormatFloat(val, 'f', -1, 64)
			featureStrs = append(featureStrs, featureStr)
		}
		featureLine := strings.Join(featureStrs, " ")

		line := labelStr + " " + featureLine + "\n"
		fmt.Fprint(writer, line)
	}

	err = writer.Flush()
	if err != nil {
		return err
	}
	return nil
}
