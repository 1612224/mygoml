package main

import (
	"fmt"
	"image/color"
	"mygoml"
	"mygoml/softmax"
	"time"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var MaxLabel = 3

type RandomPoint struct {
	X     float64
	Y     float64
	Label int
}

func NewRandomPoint(src []float64) RandomPoint {
	if len(src) != 2 {
		panic("source slice must have the length of 2")
	}
	var rp RandomPoint
	rp.X = src[0]
	rp.Y = src[1]
	return rp
}

func (rp RandomPoint) Features() []float64 {
	return []float64{rp.X, rp.Y}
}

func (rp RandomPoint) Target() []float64 {
	t := make([]float64, MaxLabel)
	t[rp.Label] = 1
	return t
}

type RandomSet []RandomPoint

func (rs RandomSet) DataPoints() []mygoml.SupervisedDataPoint {
	var out []mygoml.SupervisedDataPoint
	for _, v := range rs {
		out = append(out, v)
	}
	return out
}

func (rs RandomSet) Len() int {
	return len([]RandomPoint(rs))
}

func (rs RandomSet) XY(i int) (x, y float64) {
	return rs[i].X, rs[i].Y
}

func (rs RandomSet) Plotter(shape draw.GlyphDrawer, color color.RGBA) *plotter.Scatter {
	scatter, err := plotter.NewScatter(rs)
	if err != nil {
		panic(err)
	}

	scatter.GlyphStyle.Color = color
	scatter.GlyphStyle.Shape = shape
	return scatter
}

func main() {
	// seed random generator
	s := rand.NewSource(uint64(time.Now().Unix()))

	// generate train data & test data
	// generate points around centers
	cov := mat.NewSymDense(2, []float64{1, 0, 0, 1})
	N := 50
	ND := make([]*distmv.Normal, MaxLabel)
	ND[0], _ = distmv.NewNormal([]float64{2, 2}, cov, s)
	ND[1], _ = distmv.NewNormal([]float64{8, 3}, cov, s)
	ND[2], _ = distmv.NewNormal([]float64{3, 6}, cov, s)

	var rs RandomSet
	var test RandomSet
	labels := make([]RandomSet, MaxLabel)
	for i := 0; i < MaxLabel; i++ {
		for j := 0; j < N; j++ {
			rp := NewRandomPoint(ND[i].Rand(nil))
			rp.Label = i
			labels[rp.Label] = append(labels[rp.Label], rp)
			rs = append(rs, rp)
		}
	}

	// plot them out
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Random Points"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(labels[0].Plotter(draw.CrossGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(labels[1].Plotter(draw.CircleGlyph{}, color.RGBA{G: 255, A: 255}))
	p.Add(labels[2].Plotter(draw.PlusGlyph{}, color.RGBA{B: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/softmax/softmax_data.png"); err != nil {
		panic(err)
	}

	// define model & train
	model := &softmax.Model{}
	model.Train(rs)

	// predict
	predictions := make([]RandomSet, MaxLabel)
	for _, v := range test {
		p, _ := model.Predict(v.Features())
		fmt.Println(v.Label, p)
		label := floats.MaxIdx(p)
		predictions[label] = append(predictions[label], v)
	}

	// get weights matrix
	weights := model.Weights()
	w01, w02, w00 := weights.At(0, 0), weights.At(1, 0), weights.At(2, 0)
	w11, w12, w10 := weights.At(0, 1), weights.At(1, 1), weights.At(2, 1)
	w21, w22, w20 := weights.At(0, 2), weights.At(1, 2), weights.At(2, 2)
	fmt.Printf("%f + %fx + %fy = 0\n", (w00 - w10), (w01 - w11), (w02 - w12))
	fmt.Printf("%f + %fx + %fy = 0\n", (w10 - w20), (w11 - w21), (w12 - w22))
	fmt.Printf("%f + %fx + %fy = 0\n", (w20 - w00), (w21 - w01), (w22 - w02))
	d01 := func(x float64) float64 {
		return -((w00 - w10) + x*(w01-w11)) / (w02 - w12)
	}
	d12 := func(x float64) float64 {
		return -((w10 - w20) + x*(w11-w21)) / (w12 - w22)
	}
	d20 := func(x float64) float64 {
		return -((w20 - w00) + x*(w21-w01)) / (w22 - w02)
	}

	// plot predictions
	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Random Points"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(plotter.NewFunction(d01))
	p.Add(plotter.NewFunction(d12))
	p.Add(plotter.NewFunction(d20))
	p.Add(labels[0].Plotter(draw.CrossGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(labels[1].Plotter(draw.CircleGlyph{}, color.RGBA{G: 255, A: 255}))
	p.Add(labels[2].Plotter(draw.PlusGlyph{}, color.RGBA{B: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/softmax/softmax_test.png"); err != nil {
		panic(err)
	}
}
