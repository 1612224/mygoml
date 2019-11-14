package helpers

import (
	"mygoml"

	"gonum.org/v1/gonum/mat"
)

func ConvertSupervisedDataset(dataset mygoml.SupervisedDataSet, x0 bool) (mat.Matrix, mat.Matrix) {
	var xdata []float64
	var ydata []float64
	dps := dataset.DataPoints()
	featuresCount := len(dps[0].Features())
	targetCount := len(dps[0].Target())
	for _, dp := range dps {
		xdata = append(xdata, dp.Features()...)
		if x0 {
			xdata = append(xdata, 1.0)
		}
		ydata = append(ydata, dp.Target()...)
	}

	var xMatrix mat.Matrix
	if x0 {
		xMatrix = mat.NewDense(len(dps), featuresCount+1, xdata)
	} else {
		xMatrix = mat.NewDense(len(dps), featuresCount, xdata)
	}
	yMatrix := mat.NewDense(len(dps), targetCount, ydata)
	return xMatrix.T(), yMatrix.T()
}
