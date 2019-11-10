package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"mygoml"
	"mygoml/knn"
	"os"
	"time"

	"gonum.org/v1/gonum/floats"
)

type Sepal struct {
	Length float64
	Width  float64
}

type Pedal struct {
	Length float64
	Width  float64
}

type IrisDataPoint struct {
	Sepal
	Pedal
	Type int
}

func (dp IrisDataPoint) Features() []float64 {
	return []float64{dp.Sepal.Length, dp.Sepal.Width, dp.Pedal.Length, dp.Pedal.Width}
}

func (dp IrisDataPoint) Target() []float64 {
	return []float64{float64(dp.Type)}
}

type IrisDataSet struct {
	TrainSet []IrisDataPoint
	TestSet  []IrisDataPoint
	Labels   []string
}

func (ds *IrisDataSet) ReadFromFile(filepath string) error {
	file, err := os.Open(filepath)
	if err != nil {
		// TODO: should return error
		// panic(err)
		return err
	}

	labelToType := make(map[string]int)
	var dataset []IrisDataPoint
	s := bufio.NewScanner(file)
	for s.Scan() {
		var data IrisDataPoint
		var label string
		_, err := fmt.Sscanf(s.Text(), "%f,%f,%f,%f,%s",
			&data.Sepal.Length, &data.Sepal.Width,
			&data.Pedal.Length, &data.Pedal.Width,
			&label)
		if err != nil {
			continue
		}
		if t, ok := labelToType[label]; ok {
			data.Type = t
		} else {
			current := len(labelToType)
			labelToType[label] = current
			ds.Labels = append(ds.Labels, label)
		}
		dataset = append(dataset, data)
	}

	if err := s.Err(); err != nil {
		return err
	}

	rand.Shuffle(len(dataset), func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })
	ds.TrainSet = dataset[:100]
	ds.TestSet = dataset[100:]
	return nil
}

func (ds IrisDataSet) DataPoints() []mygoml.SupervisedDataPoint {
	var out []mygoml.SupervisedDataPoint
	for _, v := range ds.TrainSet {
		out = append(out, v)
	}
	return out
}

func MyWeigher(knn *knn.Model, current, neighbor []float64) float64 {
	sigma := 0.5
	diff := make([]float64, len(current))
	floats.SubTo(diff, current, neighbor)
	return math.Exp(floats.Norm(diff, float64(knn.Norm)) / (sigma * sigma))
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// read data from file
	ds := IrisDataSet{}
	ds.ReadFromFile("datasets/iris.data")

	// define model
	model := &knn.Model{K: 10, Norm: 2}

	// train model
	model.Train(ds)

	// predict with major voting
	fmt.Println("######## Major Voting ############")
	var predictions []float64
	var targets []float64
	for _, d := range ds.TestSet {
		fs := d.Features()
		p, _ := model.Predict(fs)
		t := int(p[0])
		// fmt.Printf("Predicted: %s, Ground Truth: %s\n", ds.Labels[t], ds.Labels[d.Type])
		fmt.Printf("Predicted: %d, Ground Truth: %d\n", t, d.Type)
		predictions = append(predictions, p...)
		targets = append(targets, d.Target()...)
	}
	fmt.Printf("[Major Voting] Accuracy: %.2f%%\n", mygoml.Accuracy(predictions, targets))

	// predict with distance weight
	fmt.Println("\n######## Distance Weight ############")
	predictions = nil
	targets = nil
	model.WeightCalculator = knn.DistanceWeight
	for _, d := range ds.TestSet {
		fs := d.Features()
		p, _ := model.Predict(fs)
		t := int(p[0])
		// fmt.Printf("Predicted: %s, Ground Truth: %s\n", ds.Labels[t], ds.Labels[d.Type])
		fmt.Printf("Predicted: %d, Ground Truth: %d\n", t, d.Type)
		predictions = append(predictions, p...)
		targets = append(targets, d.Target()...)
	}
	fmt.Printf("[Distance Weight] Accuracy: %.2f%%\n", mygoml.Accuracy(predictions, targets))

	// predict with custom weight
	fmt.Println("\n######## Custom Weight ############")
	predictions = nil
	targets = nil
	model.WeightCalculator = MyWeigher
	for _, d := range ds.TestSet {
		fs := d.Features()
		p, _ := model.Predict(fs)
		t := int(p[0])
		// fmt.Printf("Predicted: %s, Ground Truth: %s\n", ds.Labels[t], ds.Labels[d.Type])
		fmt.Printf("Predicted: %d, Ground Truth: %d\n", t, d.Type)
		predictions = append(predictions, p...)
		targets = append(targets, d.Target()...)
	}
	fmt.Printf("[Custom Weight] Accuracy: %.2f%%\n", mygoml.Accuracy(predictions, targets))
}
