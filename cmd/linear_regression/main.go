package main

import (
	"bufio"
	"fmt"
	"image/color"
	"mygoml"
	"mygoml/models"
	"os"

	"gonum.org/v1/plot/vg/draw"

	"gonum.org/v1/plot/vg"

	"gonum.org/v1/plot/plotter"

	"gonum.org/v1/plot"
)

type HeightWeight struct {
	Height float64
	Weight float64
}

func (hw HeightWeight) Features() []float64 {
	return []float64{hw.Height}
}

func (hw HeightWeight) Target() float64 {
	return hw.Weight
}

type HeightWeightSet []HeightWeight

func (hws HeightWeightSet) Len() int {
	return len(hws)
}

func (hws HeightWeightSet) XY(i int) (x, y float64) {
	return hws[i].Height, hws[i].Weight
}

func (hws HeightWeightSet) DataPoints() []mygoml.SupervisedDataPoint {
	var dataPoints []mygoml.SupervisedDataPoint
	for _, hw := range hws {
		dataPoints = append(dataPoints, hw)
	}
	return dataPoints
}

func (hws *HeightWeightSet) ReadFromFile(filepath string) error {
	file, err := os.Open(filepath)
	if err != nil {
		// TODO: should return error
		// panic(err)
		return err
	}

	var dataset []HeightWeight
	s := bufio.NewScanner(file)
	for s.Scan() {
		var data HeightWeight
		_, err := fmt.Sscanf(s.Text(), "%f %f", &data.Height, &data.Weight)
		if err == nil {
			dataset = append(dataset, data)
		}
	}

	if err := s.Err(); err != nil {
		return err
	}
	*hws = dataset
	return nil
}

func main() {
	// define variables
	var hws HeightWeightSet
	var model models.LinearRegressionModel

	// read training data from file
	err := hws.ReadFromFile("datasets/height_weight.txt")
	if err != nil {
		panic(err)
	}

	// train model
	err = model.Train(hws)
	if err != nil {
		panic(err)
	}

	// predict
	predict_155, _ := model.Predict([]float64{155})
	predict_160, _ := model.Predict([]float64{160})
	fmt.Printf("Height: 155, Predicted Weight: %f\n", predict_155)
	fmt.Printf("Height: 160, Predicted Weight: %f\n", predict_160)

	// try plotting
	// create new plot (the setting, ie label...)
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Height & Weight Relationship"
	p.X.Label.Text = "Height"
	p.Y.Label.Text = "Weight"
	p.Add(plotter.NewGrid())

	// create line plotter (the brush)
	scatter, err := plotter.NewScatter(hws)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
	p.Add(scatter)
	// save to file
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "example_plot.png"); err != nil {
		panic(err)
	}
}
