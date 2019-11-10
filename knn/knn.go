package knn

import (
	"fmt"
	"mygoml"
	"sort"

	"gonum.org/v1/gonum/floats"
)

var MajorVoting = func(knn *Model, current, neighbor []float64) float64 {
	return 1
}

var DistanceWeight = func(knn *Model, current, neighbor []float64) float64 {
	diff := make([]float64, len(current))
	floats.SubTo(diff, current, neighbor)
	return floats.Norm(diff, float64(knn.Norm))
}

type Model struct {
	memory           []mygoml.SupervisedDataPoint
	K                int
	Norm             float64
	WeightCalculator func(knn *Model, current, neighbor []float64) float64
}

func (knn *Model) Train(dataset mygoml.SupervisedDataSet) error {
	dps := dataset.DataPoints()
	if len(dps) == 0 {
		return mygoml.ErrDatasetEmpty
	}
	for _, v := range dps {
		knn.memory = append(knn.memory, v)
	}

	return nil
}

func (knn *Model) Predict(features []float64) ([]float64, error) {
	featuresCount := len(knn.memory[0].Features())
	if featuresCount != len(features) {
		msg := fmt.Sprintf("model expects %d features but got %d features", featuresCount, len(features))
		return nil, mygoml.ErrIncompatibleDataAndModel(msg)
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
	labels := make([]map[float64]float64, len(chosen))
	for i := range labels {
		labels[i] = make(map[float64]float64)
	}
	max := make([]float64, len(chosen))
	weigher := knn.WeightCalculator
	if weigher == nil {
		weigher = MajorVoting
	}
	for _, n := range neighbors {
		neighborTarget := n.Target()
		weight := weigher(knn, features, n.Features())
		for i, m := range labels {
			key := neighborTarget[i]
			m[key] = m[key] + weight
			if m[key] > max[i] {
				max[i] = m[key]
				chosen[i] = key
			}
		}
	}

	return chosen, nil
}
