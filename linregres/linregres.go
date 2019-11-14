package linregres

import (
	"fmt"
	"mygoml"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	weights mat.Dense
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
	err := m.weights.Solve(&lhs, &rhs)
	if err != nil {
		switch err.(type) {
		case mat.Condition:
			return mygoml.ErrMaybeInaccurate
		default:
			return mygoml.ErrUnknown
		}
	}
	return nil
}

func (m *Model) Predict(features []float64) ([]float64, error) {
	if r, _ := m.weights.Dims(); len(features) != r-1 {
		msg := fmt.Sprintf("model expects %d features but got %d features", r-1, len(features))
		return nil, mygoml.ErrIncompatibleDataAndModel(msg)
	}

	featureVector := mat.NewVecDense(len(features)+1, append(features, 1)).T()
	var result mat.Dense
	result.Mul(featureVector, &m.weights)
	return result.RawRowView(0), nil
}
