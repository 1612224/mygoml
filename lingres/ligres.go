package lingres

import (
	"fmt"
	"mygoml"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	weights []float64
}

func (m *Model) Train(s mygoml.SupervisedDataSet) error {
	dps := s.DataPoints()
	if len(dps) == 0 {
		return mygoml.ErrDatasetEmpty
	}
	featuresCount := len(dps[0].Features())
	targetCount := len(dps[0].Target())

	// build feature matrix (MxN) & target vector (Mx1 matrix)
	var featureMatrixData []float64
	var targetMatrixData []float64
	for _, dp := range dps {
		featureMatrixData = append(featureMatrixData, dp.Features()...)
		featureMatrixData = append(featureMatrixData, 1)
		targetMatrixData = append(targetMatrixData, dp.Target()...)
	}
	featureMatrix := mat.NewDense(len(dps), featuresCount+1, featureMatrixData)
	targetMatrix := mat.NewDense(len(dps), targetCount, targetMatrixData)
	var lhs mat.Dense
	var rhs mat.Dense
	lhs.Mul(featureMatrix.T(), featureMatrix)
	rhs.Mul(featureMatrix.T(), targetMatrix)
	var weightVector mat.Dense
	err := weightVector.Solve(&lhs, &rhs)
	if err != nil {
		switch err.(type) {
		case mat.Condition:
			return mygoml.ErrMaybeInaccurate
		default:
			return mygoml.ErrUnknown
		}
	}

	m.weights = nil
	for i := 0; i < featuresCount+1; i++ {
		m.weights = append(m.weights, weightVector.At(i, 0))
	}
	return nil
}

func (m Model) Predict(features []float64) ([]float64, error) {
	if len(features) != len(m.weights)-1 {
		msg := fmt.Sprintf("model expects %d features but got %d features", len(m.weights)-1, len(features))
		return nil, mygoml.ErrIncompatibleDataAndModel(msg)
	}

	featureVector := mat.NewVecDense(len(features)+1, append(features, 1)).T()
	weightVector := mat.NewVecDense(len(m.weights), m.weights)
	var result mat.Dense
	result.Mul(featureVector, weightVector)
	return result.RawRowView(0), nil
}
