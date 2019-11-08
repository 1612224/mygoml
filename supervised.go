package mygoml

type Target interface {
	Value() float64
}

type SupervisedDataPoint interface {
	Features() []float64
	Target() Target
}

type SupervisedDataSet interface {
	DataPoints() []SupervisedDataPoint
}

type SupervisedModel interface {
	Train(SupervisedDataSet) error
	Predict(features []float64) (Target, error)
}
