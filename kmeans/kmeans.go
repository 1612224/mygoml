package kmeans

import (
	"math"
	"mygoml"
	"time"

	"gonum.org/v1/gonum/floats"

	"golang.org/x/exp/rand"
)

type Cluster struct {
	members []mygoml.UnsupervisedDataPoint
	center  []float64
}

func (kc Cluster) Center() []float64 {
	return kc.center
}

func (kc *Cluster) Add(p mygoml.UnsupervisedDataPoint) {
	kc.members = append(kc.members, p)
}

func (kc *Cluster) Reset() {
	kc.members = nil
}

func (kc *Cluster) Members() []mygoml.UnsupervisedDataPoint {
	var out []mygoml.UnsupervisedDataPoint
	for _, v := range kc.members {
		out = append(out, v)
	}
	return out
}

type Model struct {
	ClusterCount int
}

func createRandomClusters(dps []mygoml.UnsupervisedDataPoint, clusterCount int) []*Cluster {
	s := rand.NewSource(uint64(time.Now().Unix()))
	gen := rand.New(s)
	var clusters []*Cluster
	chosen := gen.Perm(len(dps))[:clusterCount]
	for _, i := range chosen {
		var c Cluster
		c.center = dps[i].Features()
		clusters = append(clusters, &c)
	}
	return clusters
}

func findBestCluster(p mygoml.UnsupervisedDataPoint, clusters []*Cluster) *Cluster {
	pfs := p.Features()
	mini := 0
	mind := floats.Distance(pfs, clusters[0].center, 2)
	for i := 1; i < len(clusters); i++ {
		if d := floats.Distance(pfs, clusters[i].center, 2); d < mind {
			mini = i
			mind = d
		}
	}
	return clusters[mini]
}

func addMembersToClusters(dps []mygoml.UnsupervisedDataPoint, clusters []*Cluster) {
	for _, p := range dps {
		c := findBestCluster(p, clusters)
		c.Add(p)
	}
}

func calculateNewCenter(members []mygoml.UnsupervisedDataPoint) []float64 {
	if len(members) == 0 {
		return nil
	}
	sum := members[0].Features()
	clen := len(sum)
	mlen := len(members)
	for i := 1; i < mlen; i++ {
		mfs := members[i].Features()
		for j := 0; j < clen; j++ {
			sum[j] = sum[j] + mfs[j]
		}
	}
	for i := 0; i < clen; i++ {
		sum[i] = sum[i] / float64(mlen)
	}
	return sum
}

func updateClustersCenters(clusters []*Cluster) {
	for _, c := range clusters {
		newCenter := calculateNewCenter(c.Members())
		if newCenter != nil {
			c.center = newCenter
		}
	}
}

func getCenters(clusters []*Cluster) [][]float64 {
	var centers [][]float64
	for _, c := range clusters {
		centers = append(centers, c.center)
	}
	return centers
}

func sameCenter(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if math.Abs(a[i]-b[i]) > 0.0000001 {
			return false
		}
	}
	return true
}

func centerInsideSet(center []float64, set [][]float64) bool {
	for _, c := range set {
		if sameCenter(center, c) {
			return true
		}
	}
	return false
}

func sameCenterSet(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for _, va := range a {
		if !centerInsideSet(va, b) {
			return false
		}
	}
	for _, vb := range b {
		if !centerInsideSet(vb, a) {
			return false
		}
	}
	return true
}

func resetClusters(clusters []*Cluster) {
	for _, c := range clusters {
		c.Reset()
	}
}

func (km *Model) Clustering(ds mygoml.UnsupervisedDataSet) []mygoml.Cluster {
	dps := ds.DataPoints()
	// init centers & create clusters
	clusters := createRandomClusters(dps, km.ClusterCount)

	for {
		// get old centers
		oldCenters := getCenters(clusters)

		// add members to clusters
		resetClusters(clusters)
		addMembersToClusters(dps, clusters)

		// update clusters' centers
		updateClustersCenters(clusters)

		// check convergence
		newCenters := getCenters(clusters)
		if sameCenterSet(newCenters, oldCenters) {
			break
		}
	}

	var out []mygoml.Cluster
	for _, c := range clusters {
		out = append(out, c)
	}
	return out
}
