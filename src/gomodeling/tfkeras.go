package gomodeling

/*
import (
	"fmt"

	tf "github.com/wamuir/graft/tensorflow"
)

type TensorflowKerasLinearRegressor struct {
	SavedModel *tf.SavedModel
}

func (tflr *TensorflowKerasLinearRegressor) LoadTFKerasLinearRegressor(tf_keras_filepath string, model_features ModelFeatureType, model_label ModelLabelType) error {
	tfmodel, err := tf.LoadSavedModel(tf_keras_filepath, []string{"serve"}, nil)
	if err != nil {
		fmt.Print(err)
		return err
	}
	tflr.SavedModel = tfmodel
	return nil
}

//Put into Low Level Package
func (tflr *TensorflowKerasLinearRegressor) PredictTFKerasLinearRegressor(prediction []float64) float64 {
	/*featuresList := Model_Feature_Groups[tflr.ModelFeatures]
	values := make([]float64, len(featuresList))
	for index, feature := range featuresList {
		values[index] = prediction[feature]
	}
	// prediction
	_, err := tf.NewTensor(prediction)
	if err != nil {
		fmt.Print(err)
		return -1
	}

	return 0

}
*/
