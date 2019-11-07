package mygoml

import (
	"reflect"
	"testing"
)

func DeepEqual(t *testing.T, name string, expected, got interface{}) {
	t.Helper()

	if ok := reflect.DeepEqual(expected, got); !ok {
		t.Errorf("[%s] expected: %v, got %v", name, expected, got)
	}
}

func FloatEqual(t *testing.T, name string, expected, got float64) {
	t.Helper()

	if expected != got {
		t.Errorf("[%s] expected: %f, got %f", name, expected, got)
	}
}
