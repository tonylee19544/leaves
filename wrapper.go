package leaves

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
)

type ForestWrapper struct {
	Ensemble
	FeatureNames   []string
	TestPrediction float64
}

type WrapperConfig struct {
	FeatureNames   []string
	TestPrediction float64
}

func NewForestWrapper(modelPath, configPath string) (*ForestWrapper, error) {
	model, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		return nil, err
	}
	var config WrapperConfig
	// load config from file as json
	bytes, err := ioutil.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(bytes, &config); err != nil {
		return nil, err
	}
	// doing test prediction
	fvals := make([]float64, len(config.FeatureNames))
	for i := 0; i < len(config.FeatureNames); i += 1 {
		fvals[i] = float64(i)
	}
	pred := model.PredictSingle(fvals, 0)
	// compare with test prediction using epsilon
	if math.Abs(pred-config.TestPrediction) > 1e-4 {
		return nil, fmt.Errorf("test prediction failed: %f != %f", pred, config.TestPrediction)
	}
	return &ForestWrapper{
		Ensemble:       *model,
		FeatureNames:   config.FeatureNames,
		TestPrediction: config.TestPrediction,
	}, nil
}

func (fw *ForestWrapper) PredictSingle(features map[string]float64) (prediction float64, missingCount int) {
	fvals := make([]float64, len(fw.FeatureNames))
	for i, featureName := range fw.FeatureNames {
		fval, ok := features[featureName]
		if !ok {
			missingCount += 1
		}
		fvals[i] = fval
	}
	return fw.Ensemble.PredictSingle(fvals, 0), missingCount
}
