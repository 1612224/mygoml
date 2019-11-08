package mygoml

import (
	"math"
	"reflect"
	"testing"
)

var Epsilon = 0.00001

func Equal(a, b float64) bool {
	if math.Abs(a-b) < Epsilon {
		return true
	}
	return false
}

func DeepEqual(t *testing.T, name string, expected, got interface{}) {
	t.Helper()

	if ok := reflect.DeepEqual(expected, got); !ok {
		t.Errorf("[%s] expected: %v, got %v", name, expected, got)
	}
}

func FloatEqual(t *testing.T, name string, expected, got float64) {
	t.Helper()

	if !Equal(expected, got) {
		t.Errorf("[%s] expected: %f, got %f", name, expected, got)
	}
}
