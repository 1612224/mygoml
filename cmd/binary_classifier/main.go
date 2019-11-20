package main

import (
	"fmt"
	"io"
	"mygoml"
	"mygoml/logregres"
	"mygoml/mnist"
	"os"
)

var Class_0 = 0
var Class_1 = 4

type MNISTImage mnist.DigitImage

func (m MNISTImage) Features() []float64 {
	var fs []float64
	for i := range m.Image {
		for j := range m.Image[i] {
			fs = append(fs, float64(m.Image[i][j]))
		}
	}
	return fs
}

func (m MNISTImage) Target() []float64 {
	if m.Digit != Class_1 {
		return []float64{0}
	}
	return []float64{1}
}

type MNISTDataset []mnist.DigitImage

func (ds MNISTDataset) DataPoints() []mygoml.SupervisedDataPoint {
	var out []mygoml.SupervisedDataPoint
	for _, v := range ds {
		out = append(out, MNISTImage(v))
	}
	return out
}

func printData(w io.Writer, data MNISTImage) {
	fmt.Fprintln(w, data.Digit)
	mnist.PrintImage(w, data.Image)
}

func main() {
	trainset, _ := mnist.ReadTrainSet("mnist")
	testset, _ := mnist.ReadTestSet("mnist")

	var ds MNISTDataset
	for _, v := range trainset.Data {
		if v.Digit == Class_0 || v.Digit == Class_1 {
			ds = append(ds, v)
		}
	}

	model := &logregres.Model{}
	model.Train(ds)

	file, _ := os.Create("cmd/binary_classifier/binary_classifier.txt")
	defer file.Close()

	for _, v := range testset.Data {
		if v.Digit != Class_0 && v.Digit != Class_1 {
			continue
		}

		dp := MNISTImage(v)
		predicted, _ := model.Predict(dp.Features())
		class := Class_1
		threshold := 0.5
		if predicted[0] < threshold {
			class = Class_0
		}
		if class != dp.Digit {
			fmt.Fprintln(file, "Predicted: ", class)
			printData(file, dp)
		}
	}
}
