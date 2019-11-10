package mygoml

type SupervisedDataPoint interface {
	Features() []float64
	Target() []float64
}

type SupervisedDataSet interface {
	DataPoints() []SupervisedDataPoint
}

type SupervisedModel interface {
	Train(SupervisedDataSet) error
	Predict(features []float64) ([]float64, error)
}

func Accuracy(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		panic("[accuracy]: predictions set and targets set are not the same size")
	}

	correct := 0
	total := len(predictions)
	for i := range predictions {
		if Equal(predictions[i], targets[i]) {
			correct = correct + 1
		}
	}
	return float64(correct) / float64(total) * 100
}
