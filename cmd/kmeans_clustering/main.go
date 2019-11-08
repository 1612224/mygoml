package main

import (
	"image/color"
	"mygoml"
	"mygoml/models"
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
	X float64
	Y float64
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

type RandomPointSet []RandomPoint

func (rps RandomPointSet) DataPoints() []mygoml.UnsupervisedDataPoint {
	var out []mygoml.UnsupervisedDataPoint
	for _, v := range rps {
		out = append(out, v)
	}
	return out
}

type PointCluster []RandomPoint

func ClusterToPointCluster(cluster mygoml.Cluster) PointCluster {
	var pc PointCluster
	members := cluster.Members()
	for _, m := range members {
		if v, ok := m.(RandomPoint); ok {
			pc = append(pc, v)
		} else {
			return make([]RandomPoint, 0)
		}
	}
	return pc
}

func (pc PointCluster) Len() int {
	return len([]RandomPoint(pc))
}

func (pc PointCluster) XY(i int) (x, y float64) {
	return pc[i].X, pc[i].Y
}

func (pc *PointCluster) Plotter(shape draw.GlyphDrawer, color color.RGBA) *plotter.Scatter {
	scatter, err := plotter.NewScatter(pc)
	if err != nil {
		panic(err)
	}

	scatter.GlyphStyle.Color = color
	scatter.GlyphStyle.Shape = shape
	return scatter
}

func main() {
	// define variables
	var rps RandomPointSet

	// seed random generator
	s := rand.NewSource(uint64(time.Now().Unix()))

	// create centers
	var C1, C2, C3 PointCluster

	// generate points around centers
	cov := mat.NewSymDense(2, []float64{1, 0, 0, 1})
	N := 500
	ND1, _ := distmv.NewNormal([]float64{2, 2}, cov, s)
	ND2, _ := distmv.NewNormal([]float64{8, 3}, cov, s)
	ND3, _ := distmv.NewNormal([]float64{3, 6}, cov, s)
	for i := 0; i < N; i++ {
		rp1 := NewRandomPoint(ND1.Rand(nil))
		rp2 := NewRandomPoint(ND2.Rand(nil))
		rp3 := NewRandomPoint(ND3.Rand(nil))

		C1 = append(C1, rp1)
		C2 = append(C2, rp2)
		C3 = append(C3, rp3)
		rps = append(rps, rp1, rp2, rp3)
	}

	// plot them out
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "K-Means Clustering"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(C1.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(C2.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	p.Add(C3.Plotter(draw.SquareGlyph{}, color.RGBA{B: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "misc/kmeans_clustering_data.png"); err != nil {
		panic(err)
	}

	// define model
	model := models.KMeansModel{ClusterCount: 3}

	// start clustering
	clusters := model.Clustering(rps)
	C1 = ClusterToPointCluster(clusters[0])
	C2 = ClusterToPointCluster(clusters[1])
	C3 = ClusterToPointCluster(clusters[2])

	// plot clusters
	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "K-Means Clustering"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(C1.Plotter(draw.TriangleGlyph{}, color.RGBA{R: 255, A: 255}))
	p.Add(C2.Plotter(draw.RingGlyph{}, color.RGBA{G: 255, A: 255}))
	p.Add(C3.Plotter(draw.SquareGlyph{}, color.RGBA{B: 255, A: 255}))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "misc/kmeans_clustering_final.png"); err != nil {
		panic(err)
	}
}
