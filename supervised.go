package mygoml

type SupervisedDataPoint interface {
	Features() []float64
	Target() float64
}

type SupervisedDataSet interface {
	DataPoints() []SupervisedDataPoint
}

type SupervisedModel interface {
	Train(SupervisedDataSet) error
	Predict(features []float64) (float64, error)
}
