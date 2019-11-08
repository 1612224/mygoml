package models

import (
	"fmt"
	"mygoml"
	"sort"

	"gonum.org/v1/gonum/floats"
)

type KNNWeigher func(knn *KNNModel, current, neighbor []float64) float64

var KNN_MajorVoting = KNNWeigher(func(knn *KNNModel, current, neighbor []float64) float64 {
	return 1
})

var KNN_DistanceWeight = KNNWeigher(func(knn *KNNModel, current, neighbor []float64) float64 {
	diff := make([]float64, len(current))
	floats.SubTo(diff, current, neighbor)
	return floats.Norm(diff, float64(knn.Norm))
})

type KNNModel struct {
	memory           []mygoml.SupervisedDataPoint
	K                int
	Norm             float64
	WeightCalculator KNNWeigher
}

func (knn *KNNModel) Train(dataset mygoml.SupervisedDataSet) error {
	dps := dataset.DataPoints()
	if len(dps) == 0 {
		return mygoml.ErrDatasetEmpty
	}
	for _, v := range dps {
		knn.memory = append(knn.memory, v)
	}

	return nil
}

func (knn *KNNModel) Predict(features []float64) (float64, error) {
	featuresCount := len(knn.memory[0].Features())
	if featuresCount != len(features) {
		msg := fmt.Sprintf("model expects %d features but got %d features", featuresCount, len(features))
		return 0, mygoml.ErrIncompatibleDataAndModel(msg)
	}

	distanceCalculator := func(a, b []float64) float64 {
		diff := make([]float64, len(a))
		floats.SubTo(diff, a, b)
		return floats.Norm(diff, float64(knn.Norm))
	}
	sort.Slice(knn.memory, func(i, j int) bool {
		di := distanceCalculator(features, knn.memory[i].Features())
		dj := distanceCalculator(features, knn.memory[j].Features())
		return di < dj
	})

	neighbors := knn.memory[:knn.K]
	chosen := neighbors[0].Target()
	labels := make(map[float64]float64)
	var max float64
	weigher := knn.WeightCalculator
	if weigher == nil {
		weigher = KNN_MajorVoting
	}
	for _, n := range neighbors {
		key := n.Target()
		labels[key] = labels[key] + weigher(knn, features, n.Features())
		if labels[key] > max {
			max = labels[key]
			chosen = key
		}
	}

	return chosen, nil
}
