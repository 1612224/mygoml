package graddesc

import "gonum.org/v1/gonum/floats"

type BaseUpdater struct{}

func (u *BaseUpdater) Update(x []float64, f Function, learningRate float64) {
	grad := f.Gradient(x)
	floats.Scale(learningRate, grad)
	floats.Sub(x, grad)
}

func (u *BaseUpdater) Reset() {}
