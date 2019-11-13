package main

import (
	"image/color"
	"mygoml"
	"mygoml/perceptron"
	"time"

	"gonum.org/v1/plot"

	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"gonum.org/v1/plot/vg/draw"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/stat/distmv"

	"gonum.org/v1/gonum/mat"
)

type RandomPoint struct {
	X     float64
	Y     float64
	Class int
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
	return []float64{float64(rp.Class)}
}

type RandomPointSet []RandomPoint

func (rps RandomPointSet) DataPoints() []mygoml.SupervisedDataPoint {
	var out []mygoml.SupervisedDataPoint
	for _, v := range rps {
		out = append(out, v)
	}
	return out
}

func (rps RandomPointSet) Len() int {
	return len([]RandomPoint(rps))
}

func (rps RandomPointSet) XY(i int) (x, y float64) {
	return rps[i].X, rps[i].Y
}

func (rps RandomPointSet) Plotter(shape draw.GlyphDrawer, color color.RGBA) *plotter.Scatter {
	scatter, err := plotter.NewScatter(rps)
	if err != nil {
		panic(err)
	}

	scatter.GlyphStyle.Color = color
	scatter.GlyphStyle.Shape = shape
	return scatter
}

func main() {
	// define variables
	var rps, group1, group2, test, t1, t2 RandomPointSet

	// seed random generator
	s := rand.NewSource(uint64(time.Now().Unix()))

	// generate points around centers
	cov := mat.NewSymDense(2, []float64{0.3, 0.2, 0.2, 0.3})
	N := 10
	NTest := 5
	ND1, _ := distmv.NewNormal([]float64{2, 2}, cov, s)
	ND2, _ := distmv.NewNormal([]float64{4, 2}, cov, s)
	for i := 0; i < N; i++ {
		rp1 := NewRandomPoint(ND1.Rand(nil))
		rp1.Class = 1
		rp2 := NewRandomPoint(ND2.Rand(nil))
		rp2.Class = -1

		rps = append(rps, rp1, rp2)
		group1 = append(group1, rp1)
		group2 = append(group2, rp2)
	}
	for i := 0; i < NTest; i++ {
		rp1 := NewRandomPoint(ND1.Rand(nil))
		rp2 := NewRandomPoint(ND2.Rand(nil))

		test = append(test, rp1, rp2)
		t1 = append(t1, rp1)
		t2 = append(t2, rp2)
	}

	// plot them out
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Perception"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(group1.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(group2.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/perceptron/perceptron_data.png"); err != nil {
		panic(err)
	}

	// define model
	model := &perceptron.Perceptron{}

	// train model
	model.Train(rps)

	// test model
	var pd1, pd2 RandomPointSet
	for _, dp := range test.DataPoints() {
		prediction, _ := model.Predict(dp.Features())
		newrp := NewRandomPoint(dp.Features())
		newrp.Class = int(prediction[0])
		if newrp.Class == -1 {
			pd1 = append(pd1, newrp)
		} else if newrp.Class == 1 {
			pd2 = append(pd2, newrp)
		}
	}

	// plot test
	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Perceptron"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	p.Add(t1.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(t2.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/perceptron/perceptron_test.png"); err != nil {
		panic(err)
	}

	// plot predictions
	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Perceptron"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	p.Add(pd1.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(pd2.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/perceptron/perceptron_final.png"); err != nil {
		panic(err)
	}
}
