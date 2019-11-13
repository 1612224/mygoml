package graddesc

import "gonum.org/v1/gonum/floats"

type NAGUpdater struct {
	Gamma           float64
	StartVelocity   []float64
	currentVelocity []float64
}

func (u *NAGUpdater) Update(x []float64, f Function, learningRate float64) {
	if u.currentVelocity == nil {
		u.currentVelocity = make([]float64, len(x))
	}
	floats.Scale(u.Gamma, u.currentVelocity)
	lookahead := make([]float64, len(x))
	floats.SubTo(lookahead, x, u.currentVelocity)
	grad := f.Gradient(lookahead)
	floats.AddScaled(u.currentVelocity, learningRate, grad)
	floats.Sub(x, u.currentVelocity)
}

func (u *NAGUpdater) Reset() {
	u.currentVelocity = make([]float64, len(u.StartVelocity))
	copy(u.currentVelocity, u.StartVelocity)
}
