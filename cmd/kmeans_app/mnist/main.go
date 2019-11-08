package main

import (
	"fmt"
	"io"
	"mygoml"
	"mygoml/mnist"
	"mygoml/models"
	"os"
)

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

type MNISTDataset []MNISTImage

func (ds MNISTDataset) DataPoints() []mygoml.UnsupervisedDataPoint {
	var out []mygoml.UnsupervisedDataPoint
	for _, v := range ds {
		out = append(out, v)
	}
	return out
}

func printData(w io.Writer, data MNISTImage) {
	fmt.Fprintln(w, data.Digit)
	mnist.PrintImage(w, data.Image)
}

func main() {
	dataset, err := mnist.ReadTestSet("mnist")
	if err != nil {
		panic(err)
	}

	var ds MNISTDataset
	for _, v := range dataset.Data {
		ds = append(ds, MNISTImage(v))
	}

	model := models.KMeansModel{ClusterCount: 10}
	clusters := model.Clustering(ds)

	file, _ := os.Create("cmd/kmeans_app/mnist/mnist_clustering.txt")
	defer file.Close()

	for _, c := range clusters {
		kmc, _ := c.(*models.KMeansCluster)
		center := kmc.Center()
		var centerImg MNISTImage
		centerImg.Digit = -1
		for i := 0; i < dataset.H; i++ {
			var row []uint8
			for j := 0; j < dataset.W; j++ {
				row = append(row, uint8(center[i*dataset.W+j]))
			}
			centerImg.Image = append(centerImg.Image, row)
		}

		fmt.Fprintln(file, "########### Cluster ############")
		printData(file, centerImg)
		for i, m := range c.Members() {
			if i >= 10 {
				break
			}
			if v, ok := m.(MNISTImage); ok {
				printData(file, v)
			}
		}
	}
}
