package softmax

import (
	"fmt"
	"math"
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

func softmax(z []float64) []float64 {
	max := floats.Max(z)
	temp := make([]float64, len(z))
	for i := range z {
		temp[i] = math.Exp(z[i] - max)
	}
	sum := floats.Sum(temp)
	out := make([]float64, len(z))
	for i := range z {
		out[i] = temp[i] / sum
	}
	return out
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
		wData = append(wData, float64(i+1))
	}
	wMatrix := mat.NewDense(wr, wc, wData)

	gradient := func(xi, yi []float64, w mat.Matrix) []float64 {
		xivec := mat.NewVecDense(len(xi), copyFloats(xi))

		// zi = W^t * xi
		var zivec mat.VecDense
		zivec.MulVec(w.T(), xivec)
		zival := mat.Col(nil, 0, &zivec)

		// ai = softmax(zi)
		ai := softmax(zival)

		// ei = ai - yi
		ei := make([]float64, len(ai))
		floats.SubTo(ei, ai, yi)
		eivec := mat.NewVecDense(len(ei), ei)

		// dL/dW = xi*ei^t
		dW := mat.NewDense(wr, wc, nil)
		dW.Mul(xivec, eivec.T())

		return toFloatSlice(dW.T())
	}

	epochProvider := &graddesc.StochasticProvider{
		TotalSize: xcount,
		EpochGen: func(i int) graddesc.Function {
			xi := mat.Col(nil, i, X)
			yi := mat.Col(nil, i, Y)

			return graddesc.Function{
				InputSize: wr * wc,
				Gradient: func(w []float64) []float64 {
					grad := gradient(xi, yi, wMatrix)
					return grad
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
		LearningRate:  0.05,
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
	z := mat.Col(nil, 0, &result)
	return softmax(z), nil
}
