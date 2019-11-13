package main

import (
	"fmt"
	"math"
	"mygoml/graddesc"
)

func main() {
	myFunc := graddesc.Function{
		InputSize: 1,
		Mapper: func(x []float64) []float64 {
			a := x[0]
			return []float64{a*a + 10*math.Sin(a)}
		},
		Gradient: func(x []float64) []float64 {
			a := x[0]
			return []float64{2*a + 10*math.Cos(a)}
		},
	}

	out := make(chan string)
	// base
	batchProvider := graddesc.BatchProvider(myFunc)
	go func() {
		op := graddesc.Optimizer{
			LearningRate:  0.1,
			MaxStep:       100,
			EpochProvider: &batchProvider,
		}
		var optimizedX []float64

		op.Updater = &graddesc.BaseUpdater{}
		optimizedX = op.Optimize([]float64{5})
		out <- fmt.Sprintf("without momentum = %v\n", optimizedX)
	}()

	// momentum
	go func() {
		op := graddesc.Optimizer{
			LearningRate:  0.1,
			MaxStep:       100,
			EpochProvider: &batchProvider,
		}
		var optimizedX []float64

		op.Updater = &graddesc.MomentumUpdater{Gamma: 0.9, StartVelocity: make([]float64, 1)}
		optimizedX = op.Optimize([]float64{5})
		out <- fmt.Sprintf("with momentum = %v\n", optimizedX)
	}()

	// nag
	go func() {
		op := graddesc.Optimizer{
			LearningRate:  0.1,
			MaxStep:       100,
			EpochProvider: &batchProvider,
		}
		var optimizedX []float64

		op.Updater = &graddesc.NAGUpdater{Gamma: 0.9, StartVelocity: make([]float64, 1)}
		optimizedX = op.Optimize([]float64{5})
		out <- fmt.Sprintf("with nag = %v\n", optimizedX)
	}()

	for i := 0; i < 3; i++ {
		fmt.Println(<-out)
	}
}
