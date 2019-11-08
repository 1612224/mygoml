package models

import "math"

func distance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("[calculate distance between 2 points]: dimension mismatch")
	}
	var sum float64
	for i := range a {
		sum = sum + (a[i]-b[i])*(a[i]-b[i])
	}
	return math.Sqrt(sum)
}
