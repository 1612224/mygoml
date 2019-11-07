package mygoml

import (
	"testing"
)

type FakeSupervisedDataPoint struct{}

func (f FakeSupervisedDataPoint) Features() []float64 {
	return []float64{1, 2, 3, 4, 5}
}

func (f FakeSupervisedDataPoint) Target() float64 {
	return 12
}

type FakeSupervisedDataSet []SupervisedDataPoint

func (f FakeSupervisedDataSet) DataPoints() []SupervisedDataPoint {
	return f
}

type FakeSupervisedModel struct {
	TrainCalled bool
}

func (f *FakeSupervisedModel) Train(SupervisedDataSet) {
	f.TrainCalled = true
}

func (f FakeSupervisedModel) Predict(features []float64) float64 {
	return 0
}

func TestSupervised(t *testing.T) {

	// testing data point
	t.Run("testing data point", func(t *testing.T) {
		var fdp FakeSupervisedDataPoint

		// test features
		expectedFeatures := []float64{1, 2, 3, 4, 5}
		gotFeatures := fdp.Features()
		DeepEqual(t, "features", expectedFeatures, gotFeatures)

		// test target
		expectedTarget := float64(12)
		gotTarget := fdp.Target()
		FloatEqual(t, "target", expectedTarget, gotTarget)
	})

	// testing model
	t.Run("testing model", func(t *testing.T) {
		fm := FakeSupervisedModel{TrainCalled: false}

		// test train
		fm.Train(FakeSupervisedDataSet{})
		if !fm.TrainCalled {
			t.Error("expect Train() method to be called, but it's not")
		}

		// test predict
		expectedTarget := float64(0)
		gotTarget := fm.Predict([]float64{1, 2, 3, 4, 5})
		FloatEqual(t, "predict", expectedTarget, gotTarget)
	})
}
