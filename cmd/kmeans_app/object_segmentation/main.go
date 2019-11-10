package main

import (
	"image"
	"image/color"
	"image/jpeg"
	_ "image/jpeg"
	"mygoml"
	"mygoml/kmeans"
	"os"
)

type Uint32Color struct {
	r, g, b uint32
}

func (u Uint32Color) RGBA() (r, g, b, a uint32) {
	return u.r, u.g, u.b, 4294967295
}

type ImagePoint struct {
	x, y  int
	color color.Color
}

func (ip ImagePoint) Features() []float64 {
	r, g, b, _ := ip.color.RGBA()
	return []float64{float64(r), float64(g), float64(b)}
}

type Image struct {
	img image.Image
}

func (im Image) DataPoints() []mygoml.UnsupervisedDataPoint {
	var out []mygoml.UnsupervisedDataPoint
	rect := im.img.Bounds()
	for x := rect.Min.X; x < rect.Max.X; x++ {
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			c := im.img.At(x, y)
			out = append(out, ImagePoint{x, y, c})
		}
	}
	return out
}

func main() {
	imageFile, _ := os.Open("cmd/kmeans_app/object_segmentation/girl3.jpg")
	defer imageFile.Close()

	img, _, _ := image.Decode(imageFile)
	ds := Image{img}

	model := kmeans.Model{ClusterCount: 3}
	clusters := model.Clustering(ds)

	rect := img.Bounds()
	newImg := image.NewRGBA(rect)
	for _, c := range clusters {
		rc, _ := c.(*kmeans.Cluster)
		center := rc.Center()
		centerColor := Uint32Color{uint32(center[0]), uint32(center[1]), uint32(center[2])}
		for _, m := range c.Members() {
			rm, _ := m.(ImagePoint)
			newImg.Set(rm.x, rm.y, centerColor)
		}
	}

	newImgFile, _ := os.Create("cmd/kmeans_app/object_segmentation/girl3_clustering_final.jpg")
	defer newImgFile.Close()
	jpeg.Encode(newImgFile, newImg, &jpeg.Options{Quality: 80})
}
