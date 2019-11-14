package logregres

import (
	"fmt"
	"math"
	"math/rand"
	"mygoml"
	"mygoml/graddesc"
	"mygoml/helpers"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	weights mat.Matrix
}

func (m *Model) Weights() mat.Matrix {
	return mat.DenseCopyOf(m.weights)
}

func copyFloats(x []float64) []float64 {
	n := make([]float64, len(x))
	copy(n, x)
	return n
}

func wtx(w mat.Matrix, xi []float64) []float64 {
	xivec := mat.NewVecDense(len(xi), copyFloats(xi))
	var yivec mat.VecDense
	yivec.MulVec(w.T(), xivec)
	yi := mat.Col(nil, 0, &yivec)
	return yi
}

func sigmod(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func toFloatSlice(m mat.Matrix) []float64 {
	var data []float64
	mt := m.T()
	r, c := mt.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data = append(data, mt.At(i, j))
		}
	}
	return data
}

func (m *Model) Train(dataset mygoml.SupervisedDataSet) error {
	X, Y := helpers.ConvertSupervisedDataset(dataset, true)
	wr, xcount := X.Dims()
	wc, _ := Y.Dims()

	var wData []float64
	for i := 0; i < wr*wc; i++ {
		wData = append(wData, rand.Float64()+0.01)
	}
	wMatrix := mat.NewDense(wr, wc, wData)
	// s = Wt * X
	// z = sigmod(s)

	gradient := func(xi, yi []float64, w mat.Matrix) []float64 {
		xivec := mat.NewVecDense(len(xi), copyFloats(xi))
		var zivec mat.VecDense
		zivec.MulVec(w.T(), xivec)
		zival := mat.Col(nil, 0, &zivec)
		wMatrix := mat.NewDense(wr, wc, nil)
		for i, zv := range zival {
			zi := 1 / (1 + math.Exp(-zv))
			zi = zi - yi[i]
			temp := make([]float64, len(xi))
			floats.ScaleTo(temp, zi, xi)
			wMatrix.SetCol(i, temp)
		}
		return toFloatSlice(wMatrix)
	}

	epochProvider := &graddesc.StochasticProvider{
		TotalSize: xcount,
		EpochGen: func(i int) graddesc.Function {
			xi := mat.Col(nil, i, X)
			yi := mat.Col(nil, i, Y)

			return graddesc.Function{
				InputSize: wr * wc,
				Gradient: func(w []float64) []float64 {
					return gradient(xi, yi, wMatrix)
				},
			}
		},
		AfterUpdateFunc: func(w []float64) {
			for r := 0; r < wr; r++ {
				for c := 0; c < wc; c++ {
					wMatrix.Set(r, c, w[r*wc+c])
				}
			}
		},
	}

	op := graddesc.Optimizer{
		EpochProvider: epochProvider,
		LearningRate:  0.01,
		MaxStep:       1000,
		Updater:       &graddesc.BaseUpdater{},
	}

	wData = op.Optimize(wData)
	m.weights = mat.NewDense(wr, wc, wData)

	return nil
}

func (m *Model) Predict(features []float64) ([]float64, error) {
	if r, _ := m.weights.Dims(); len(features) != r-1 {
		msg := fmt.Sprintf("model expects %d features but got %d features", r-1, len(features))
		return nil, mygoml.ErrIncompatibleDataAndModel(msg)
	}

	featureVector := mat.NewVecDense(len(features)+1, append(features, 1))
	var result mat.Dense
	result.Mul(m.weights.T(), featureVector)
	predicted := mat.Col(nil, 0, &result)
	return predicted, nil
}
