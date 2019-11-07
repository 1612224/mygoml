package mygoml

type UnsupervisedDataPoint interface {
	Features() []float64
}

type UnsupervisedDataSet interface {
	DataPoints() []UnsupervisedDataPoint
}

type Cluster interface {
	Add(UnsupervisedDataPoint)
	Reset()
	Members() []UnsupervisedDataPoint
}

type UnsupervisedModel interface {
	Clustering(UnsupervisedDataSet) []Cluster
}
