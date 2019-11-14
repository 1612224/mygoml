package main

import (
	"image/color"
	"math"
	"mygoml"
	"mygoml/logregres"

	"gonum.org/v1/plot"

	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"gonum.org/v1/plot/vg/draw"
)

type Student struct {
	StudyTime float64
	Passed    int
}

func (s Student) Features() []float64 {
	return []float64{s.StudyTime}
}

func (s Student) Target() []float64 {
	return []float64{float64(s.Passed)}
}

type StudentSet []Student

func (ss StudentSet) DataPoints() []mygoml.SupervisedDataPoint {
	var out []mygoml.SupervisedDataPoint
	for _, v := range ss {
		out = append(out, v)
	}
	return out
}

func (ss StudentSet) Len() int {
	return len([]Student(ss))
}

func (ss StudentSet) XY(i int) (x, y float64) {
	return ss[i].StudyTime, float64(ss[i].Passed)
}

func (ss StudentSet) Plotter(shape draw.GlyphDrawer, color color.RGBA) *plotter.Scatter {
	scatter, err := plotter.NewScatter(ss)
	if err != nil {
		panic(err)
	}

	scatter.GlyphStyle.Color = color
	scatter.GlyphStyle.Shape = shape
	return scatter
}

func main() {
	// define variables
	var ss, failed, passed, testfailed, testpassed StudentSet

	// add data
	studyTimes := []float64{0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
		2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50}
	results := []int{0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1}
	for i := range results {
		var s Student
		s.StudyTime = studyTimes[i]
		s.Passed = results[i]
		if s.Passed == 0 {
			failed = append(failed, s)
		} else {
			passed = append(passed, s)
		}
		ss = append(ss, s)
	}

	// plot them out
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Study Time & Exam Result Relationship"
	p.X.Label.Text = "Study Time"
	p.Y.Label.Text = "Result (1 - Passed, 0 - Failed)"
	p.Add(plotter.NewGrid())
	p.Add(failed.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(passed.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/logistic_regression/logistic_regression_data.png"); err != nil {
		panic(err)
	}

	// define model
	model := &logregres.Model{}

	// train model
	model.Train(ss)

	// get weights matrix
	weights := model.Weights()
	w1, w0 := weights.At(0, 0), weights.At(1, 0)
	sigmoid := func(x float64) float64 {
		return 1 / (1 + math.Exp(-(w1*x + w0)))
	}

	// test model
	threshold := 0.3
	for i := 0; i < 5; i++ {
		var s Student
		s.StudyTime = float64(i + 1)
		p, _ := model.Predict(s.Features())
		if p[0] > threshold {
			s.Passed = 1
			testpassed = append(testpassed, s)
		} else {
			s.Passed = 0
			testfailed = append(testfailed, s)
		}
	}

	// plot test
	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Study Time & Exam Result Relationship"
	p.X.Label.Text = "Study Time"
	p.Y.Label.Text = "Result (1 - Passed, 0 - Failed)"
	p.Add(plotter.NewGrid())
	p.Add(plotter.NewFunction(sigmoid))
	p.Add(testfailed.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(testpassed.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/logistic_regression/logistic_regression_test.png"); err != nil {
		panic(err)
	}
}
