package graddesc

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Function struct {
	InputSize int
	Mapper    func(x []float64) []float64
	Gradient  func(x []float64) []float64
}

func (f Function) Run(x []float64) []float64 {
	if len(x) != f.InputSize {
		panic("function input size mismatch")
	}
	out := f.Mapper(x)
	return out
}

func randomVector(size int) []float64 {
	var out []float64
	for i := 0; i < size; i++ {
		out = append(out, rand.Float64())
	}
	return out
}

type Optimizer struct {
	EpochProvider EpochProvider
	LearningRate  float64
	MaxStep       int
	CheckInterval int
	Updater       Updater
}

type Updater interface {
	Update(x []float64, f Function, learningRate float64)
	Reset()
}

func (o *Optimizer) Optimize(startPoint []float64) []float64 {
	// reset updater
	o.Updater.Reset()

	// add default values
	if o.CheckInterval <= 0 {
		o.CheckInterval = 1
	}

	// setup variables
	count := 0
	x := make([]float64, len(startPoint))
	copy(x, startPoint)
	output := o.EpochProvider.Funcs()[0]().Gradient(x)
	zeros := make([]float64, len(output))
	// do gradient descent loop
	for count < o.MaxStep {
		done := true
		for _, loss := range o.EpochProvider.Funcs() {
			grad := loss().Gradient(x)
			if count%o.CheckInterval == 0 && floats.EqualApprox(grad, zeros, 0.0000001) {
				continue
			}
			o.Updater.Update(x, loss(), o.LearningRate)
			o.EpochProvider.AfterUpdate(x)
			done = false
		}
		if done {
			break
		}
		o.EpochProvider.OnEpochEnd(x)
		count = count + 1
	}
	o.Updater.Reset()
	return x
}
