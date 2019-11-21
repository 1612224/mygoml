package main

import (
	"image/color"
	"math"
	"mygoml"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/plot/vg"

	"gonum.org/v1/plot"

	"golang.org/x/exp/rand"

	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

var LabelNum = 3

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
	t := make([]float64, LabelNum)
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
	// generate non-linearly seperable data
	rand.Seed(uint64(time.Now().UnixNano()))
	s := rand.NewSource(uint64(time.Now().Unix()))

	cov := mat.NewSymDense(1, []float64{1})
	ND, _ := distmv.NewNormal([]float64{0}, cov, s)

	var rs RandomSet
	gs := make([]RandomSet, LabelNum)
	N := 100

	for i := 0; i < LabelNum; i++ {
		for j := 0; j < N; j++ {
			r := float64(j) / float64(N)
			t := float64(i)*4 + 4*float64(j)/float64(N) + ND.Rand(nil)[0]*0.2
			rp := RandomPoint{X: r * math.Sin(t), Y: r * math.Cos(t), Label: i}
			rs = append(rs, rp)
			gs[i] = append(gs[i], rp)
		}
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Random Points"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(gs[0].Plotter(draw.CircleGlyph{}, mygoml.Red))
	p.Add(gs[1].Plotter(draw.CircleGlyph{}, mygoml.Green))
	p.Add(gs[2].Plotter(draw.CircleGlyph{}, mygoml.Blue))

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "cmd/mlp/mlp_data.png"); err != nil {
		panic(err)
	}
}
