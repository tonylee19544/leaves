package leaves

import (
	"bufio"
	"fmt"
	"github.com/dmitryikh/leaves/internal/xgjson"
	"math"
	"os"

	"github.com/dmitryikh/leaves/internal/xgbin"
	"github.com/dmitryikh/leaves/transformation"
)

func xgSplitIndex(origNode *xgbin.Node) uint32 {
	return origNode.SIndex & ((1 << 31) - 1)
}

func xgDefaultLeft(origNode *xgbin.Node) bool {
	return (origNode.SIndex >> 31) != 0
}

func xgIsLeaf(origNode *xgbin.Node) bool {
	return origNode.CLeft == -1
}

func xgTreeFromTreeModel(origTree *xgbin.TreeModel, numFeatures uint32) (lgTree, error) {
	t := lgTree{}

	if origTree.Param.NumFeature > int32(numFeatures) {
		return t, fmt.Errorf(
			"tree number of features %d, but header number of features %d",
			origTree.Param.NumFeature,
			numFeatures,
		)
	}

	if origTree.Param.NumRoots != 1 {
		return t, fmt.Errorf("support only trees with 1 root (got %d)", origTree.Param.NumRoots)
	}

	if origTree.Param.NumNodes == 0 {
		return t, fmt.Errorf("tree with zero number of nodes")
	}
	numNodes := origTree.Param.NumNodes

	// XGBoost doesn't support categorical features
	t.nCategorical = 0

	if numNodes == 1 {
		// special case - constant value tree
		t.leafValues = append(t.leafValues, float64(origTree.Nodes[0].Info))
		return t, nil
	}

	createNode := func(origNode *xgbin.Node) (lgNode, error) {
		node := lgNode{}
		// count nan as missing value
		// NOTE: this differs with XGBosst realization: could be a problem
		missingType := uint8(missingNan)

		defaultType := uint8(0)
		if xgDefaultLeft(origNode) {
			defaultType = defaultLeft
		}
		node = numericalNode(xgSplitIndex(origNode), missingType, float64(origNode.Info), defaultType)

		if origNode.CLeft < 0 {
			return node, fmt.Errorf("logic error: got origNode.CLeft < 0")
		}
		if origNode.CRight < 0 {
			return node, fmt.Errorf("logic error: got origNode.CRight < 0")
		}
		if xgIsLeaf(&origTree.Nodes[origNode.CLeft]) {
			node.Flags |= leftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, float64(origTree.Nodes[origNode.CLeft].Info))
		}
		if xgIsLeaf(&origTree.Nodes[origNode.CRight]) {
			node.Flags |= rightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, float64(origTree.Nodes[origNode.CRight].Info))
		}
		return node, nil
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.nodes = make([]lgNode, 0, numNodes)
	node, err := createNode(&origTree.Nodes[0])
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&rightLeaf == 0 {
			origIdx := origTree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].CRight
			if !visited[origIdx] {
				node, err := createNode(&origTree.Nodes[origIdx])
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if t.nodes[convIdx].Flags&leftLeaf == 0 {
			origIdx := origTree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].CLeft
			if !visited[origIdx] {
				node, err := createNode(&origTree.Nodes[origIdx])
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	return t, nil
}

// XGEnsembleFromReader reads XGBoost model from `reader`. Works with 'gbtree' and 'dart' models
func XGEnsembleFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	e := &xgEnsemble{}

	// reading header info
	useLearnerParam := false

	if peek, err := reader.Peek(4); err == nil && string(peek) == "binf" {
		_, _ = reader.Read(make([]byte, 4))
		useLearnerParam = true
	}
	header, err := xgbin.ReadModelHeader(reader)
	if err != nil {
		return nil, err
	}
	if header.NameGbm == "gbtree" {
		e.name = "xgboost.gbtree"
	} else if header.NameGbm == "dart" {
		e.name = "xgboost.dart"
	} else {
		return nil, fmt.Errorf("only 'gbtree' or 'dart' is supported (got %s)", header.NameGbm)
	}

	if header.Param.NumFeatures == 0 {
		return nil, fmt.Errorf("zero number of features")
	}
	e.MaxFeatureIdx = int(header.Param.NumFeatures) - 1
	if useLearnerParam {
		e.BaseScore = math.Log(float64(header.Param.BaseScore)) - math.Log(float64(1-header.Param.BaseScore))
	} else {
		e.BaseScore = float64(header.Param.BaseScore)
	}
	// reading gbtree
	origModel, err := xgbin.ReadGBTreeModel(reader)
	if err != nil {
		return nil, err
	}
	if origModel.Param.DeprecatedNumFeature > int32(header.Param.NumFeatures) {
		return nil, fmt.Errorf(
			"gbtee number of features %d, but header number of features %d",
			origModel.Param.DeprecatedNumFeature,
			header.Param.NumFeatures,
		)
	}

	e.WeightDrop = make([]float64, origModel.Param.NumTrees)
	if header.NameGbm == "dart" {
		// read additional float32 slice of weighs of dropped trees. Only for 'dart' models
		weightDrop, err := xgbin.ReadFloat32Slice(reader)
		if err != nil {
			return nil, err
		}
		if len(weightDrop) != int(origModel.Param.NumTrees) {
			return nil, fmt.Errorf(
				"unexpected len(weightDrop) for 'dart' (got: %d, expected: %d)",
				len(weightDrop),
				origModel.Param.NumTrees,
			)
		}
		for i, v := range weightDrop {
			e.WeightDrop[i] = float64(v)
		}
	} else if header.NameGbm == "gbtree" {
		// use 1.0 as default. 1.0 scale will not break down anything
		for i := 0; i < int(origModel.Param.NumTrees); i++ {
			e.WeightDrop[i] = 1.0
		}
	} else {
		return nil, fmt.Errorf("unsupported model type (got: %s)", header.NameGbm)
	}
	// TODO: below is not true (see Agaricus test). Why?
	// if header.GbTreeModelParam.NumClass != origModel.GbTreeModelParam.DeprecatedNumOutputGroup {
	// 	return nil, fmt.Errorf("header number of class and model number of class should be the same (%d != %d)",
	// 		header.GbTreeModelParam.NumClass, origModel.GbTreeModelParam.DeprecatedNumOutputGroup)
	// }
	if useLearnerParam || origModel.Param.DeprecatedNumOutputGroup == 0 {
		if header.Param.NumClass == 0 {
			e.nRawOutputGroups = 1
		} else {
			e.nRawOutputGroups = int(header.Param.NumClass)
		}
	} else {
		e.nRawOutputGroups = int(origModel.Param.DeprecatedNumOutputGroup)
	}
	if origModel.Param.DeprecatedNumRoots != 1 {
		return nil, fmt.Errorf("support only trees with 1 root (got %d)", origModel.Param.DeprecatedNumRoots)
	}
	if len(origModel.TreeInfo) != int(origModel.Param.NumTrees) {
		return nil, fmt.Errorf("TreeInfo size should be %d (got %d)",
			int(origModel.Param.NumTrees),
			len(origModel.TreeInfo))
	}
	{
		// Check that TreeInfo has expected pattern (0 1 2 0 1 2...)
		curID := 0
		for i := 0; i < len(origModel.TreeInfo); i++ {
			if int(origModel.TreeInfo[i]) != curID {
				return nil, fmt.Errorf("TreeInfo expected to have pattern [0 1 2 0 1 2...] (got %v)", origModel.TreeInfo)
			}
			curID++
			if curID >= e.nRawOutputGroups {
				curID = 0
			}
		}
	}

	var transform transformation.Transform
	transform = &transformation.TransformRaw{e.nRawOutputGroups}
	if loadTransformation {
		if header.NameObj == "binary:logistic" {
			transform = &transformation.TransformLogistic{}
		} else {
			return nil, fmt.Errorf("unknown transformation function '%s'", header.NameObj)
		}
	}

	nTrees := origModel.Param.NumTrees
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in model")
	}

	// reading particular trees
	e.Trees = make([]lgTree, 0, nTrees)
	for i := int32(0); i < nTrees; i++ {
		tree, err := xgTreeFromTreeModel(origModel.Trees[i], header.Param.NumFeatures)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return &Ensemble{e, transform}, nil
}

// XGEnsembleFromFile reads XGBoost model from binary file. Works with 'gbtree' and 'dart' models
func XGEnsembleFromFile(filename string, loadTransformation bool) (*Ensemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return XGEnsembleFromReader(bufReader, loadTransformation)
}

func XGEnsembleFromJsonFile(filename string, loadTransformation bool) (*Ensemble, error) {
	gbTreeJson, err := xgjson.ReadGBTree(filename)
	if err != nil {
		return nil, err
	}
	e := &xgEnsemble{}
	if gbTreeJson.Learner.LearnerModelParam.NumClass == 0 {
		e.nRawOutputGroups = 1
	} else {
		e.nRawOutputGroups = int(gbTreeJson.Learner.LearnerModelParam.NumClass)
	}
	e.WeightDrop = gbTreeJson.Learner.GradientBooster.WeightDrop
	e.BaseScore = math.Log(float64(gbTreeJson.Learner.LearnerModelParam.BaseScore)) - math.Log(1-float64(gbTreeJson.Learner.LearnerModelParam.BaseScore))
	e.MaxFeatureIdx = int(gbTreeJson.Learner.LearnerModelParam.NumFeatures) - 1
	e.name = fmt.Sprintf("xgboost.%s", gbTreeJson.Learner.GradientBooster.Name)
	e.WeightDrop = make([]float64, gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.NumTrees)
	if gbTreeJson.Learner.GradientBooster.Name == "dart" {
		// read additional float32 slice of weighs of dropped trees. Only for 'dart' models
		e.WeightDrop = gbTreeJson.Learner.GradientBooster.WeightDrop
	} else if gbTreeJson.Learner.GradientBooster.Name == "gbtree" {
		for i := 0; i < int(gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.NumTrees); i++ {
			e.WeightDrop[i] = 1.0
		}
	} else {
		return nil, fmt.Errorf("unsupported model type (got: %s)", gbTreeJson.Learner.GradientBooster.Name)
	}
	var transform transformation.Transform
	transform = &transformation.TransformRaw{e.nRawOutputGroups}
	if loadTransformation {
		if gbTreeJson.Learner.Objective.Name == "binary:logistic" {
			transform = &transformation.TransformLogistic{}
		} else {
			return nil, fmt.Errorf("unknown transformation function '%s'", gbTreeJson.Learner.Objective.Name)
		}
	}
	nTrees := gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.NumTrees
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in model")
	}
	e.Trees = make([]lgTree, 0, nTrees)
	model := gbTreeJson.Learner.GradientBooster.Model.ToBinGBTreeModel()
	for i := int32(0); i < nTrees; i++ {
		tree, err := xgTreeFromTreeModel(model.Trees[i], gbTreeJson.Learner.LearnerModelParam.NumFeatures)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return &Ensemble{e, transform}, nil
}
