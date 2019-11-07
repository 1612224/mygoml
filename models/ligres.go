package models

import (
	"fmt"
	"mygoml"

	"gonum.org/v1/gonum/mat"
)

type LinearRegressionModel struct {
	weights []float64
}

func (m *LinearRegressionModel) Train(s mygoml.SupervisedDataSet) error {
	dps := s.DataPoints()
	if len(dps) == 0 {
		return mygoml.ErrDatasetEmpty
	}
	featuresCount := len(dps[0].Features())

	// build feature matrix (MxN) & target vector (Mx1 matrix)
	var featureMatrixData []float64
	var targetVectorData []float64
	for _, dp := range dps {
		featureMatrixData = append(featureMatrixData, dp.Features()...)
		featureMatrixData = append(featureMatrixData, 1)
		targetVectorData = append(targetVectorData, dp.Target())
	}
	featureMatrix := mat.NewDense(len(dps), featuresCount+1, featureMatrixData)
	targetVector := mat.NewVecDense(len(dps), targetVectorData)
	var lhs mat.Dense
	var rhs mat.VecDense
	lhs.Mul(featureMatrix.T(), featureMatrix)
	rhs.MulVec(featureMatrix.T(), targetVector)
	var weightVector mat.VecDense
	err := weightVector.SolveVec(&lhs, &rhs)
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

func (m LinearRegressionModel) Predict(features []float64) (float64, error) {
	if len(features) != len(m.weights)-1 {
		msg := fmt.Sprintf("model expects %d features but got %d features", len(m.weights)-1, len(features))
		return 0, mygoml.ErrIncompatibleDataAndModel(msg)
	}

	featureVector := mat.NewVecDense(len(features)+1, append(features, 1)).T()
	weightVector := mat.NewVecDense(len(m.weights), m.weights)
	var result mat.VecDense
	result.MulVec(featureVector, weightVector)
	return result.At(0, 0), nil
}
