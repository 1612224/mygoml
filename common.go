package mygoml

import "errors"

var ErrDatasetEmpty = errors.New("[dataset empty]: there is no data inside dataset")
var ErrMaybeInaccurate = errors.New("[maybe inaccurate computation]: the computed solution maybe inaccurate")
var ErrUnknown = errors.New("[unknown]: unknown error")

type ErrIncompatibleDataAndModel string

func (e ErrIncompatibleDataAndModel) Error() string {
	return "[incompatible data and model]: model and data provided are not compatible - " + string(e)
}
