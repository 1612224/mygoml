package graddesc

import "math/rand"

type funcCreator = func() Function

type EpochProvider interface {
	Funcs() []funcCreator
	OnEpochEnd(x []float64)
	AfterUpdate(x []float64)
}

type BatchProvider Function

func (p *BatchProvider) Funcs() []funcCreator {
	return []funcCreator{func() Function { return Function(*p) }}
}

func (p *BatchProvider) OnEpochEnd([]float64)  {}
func (p *BatchProvider) AfterUpdate([]float64) {}

type MiniBatchProvider struct {
	BatchSize       int
	TotalSize       int
	EpochGen        func(indices []int) Function
	EpochEndFunc    func([]float64)
	AfterUpdateFunc func([]float64)
}

func (p *MiniBatchProvider) Funcs() []funcCreator {
	var fs []funcCreator
	var indices []int
	for i := 0; i < p.TotalSize; i++ {
		if (i > 0 && i%p.BatchSize == 0) || (i == p.TotalSize-1) {
			fs = append(fs, func() Function { return p.EpochGen(indices) })
			indices = nil
		}
		indices = append(indices, i)
	}
	return fs
}

func (p *MiniBatchProvider) OnEpochEnd(x []float64) {
	if p.EpochEndFunc != nil {
		p.EpochEndFunc(x)
	}
}

func (p *MiniBatchProvider) AfterUpdate(x []float64) {
	if p.AfterUpdateFunc != nil {
		p.AfterUpdateFunc(x)
	}
}

type StochasticProvider struct {
	TotalSize       int
	EpochGen        func(index int) Function
	EpochEndFunc    func([]float64)
	AfterUpdateFunc func([]float64)
}

func (p *StochasticProvider) Funcs() []funcCreator {
	shuffles := rand.Perm(p.TotalSize)
	var fs []funcCreator
	for i, _ := range shuffles {
		k := i
		fs = append(fs, func() Function { return p.EpochGen(k) })
	}
	return fs
}

func (p *StochasticProvider) OnEpochEnd(x []float64) {
	if p.EpochEndFunc != nil {
		p.EpochEndFunc(x)
	}
}

func (p *StochasticProvider) AfterUpdate(x []float64) {
	if p.AfterUpdateFunc != nil {
		p.AfterUpdateFunc(x)
	}
}
