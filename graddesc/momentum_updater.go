package graddesc

import "gonum.org/v1/gonum/floats"

type MomentumUpdater struct {
	Gamma           float64
	StartVelocity   []float64
	currentVelocity []float64
}

func (u *MomentumUpdater) Update(x []float64, f Function, learningRate float64) {
	grad := f.Gradient(x)
	if u.currentVelocity == nil {
		u.currentVelocity = make([]float64, len(x))
	}
	floats.Scale(learningRate, grad)
	floats.Scale(u.Gamma, u.currentVelocity)
	floats.Add(u.currentVelocity, grad)
	floats.Sub(x, u.currentVelocity)
}

func (u *MomentumUpdater) Reset() {
	u.currentVelocity = make([]float64, len(u.StartVelocity))
	copy(u.currentVelocity, u.StartVelocity)
}
