module Torch.Indef.Static.NN.Criterion where

import qualified Torch.Class.NN as Dynamic
import qualified Torch.Class.NN.Static.Criterion as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.NN ()
import Torch.Indef.Static.Tensor ()

instance Class.Criterion Tensor where
  _absCriterion_updateOutput i t o = Dynamic.absCriterion_updateOutput (asDynamic i) (asDynamic t) (asDynamic o)
  _absCriterion_updateGradInput i t go gi = Dynamic.absCriterion_updateGradInput (asDynamic i) (asDynamic t) (asDynamic go) (asDynamic gi)

  _bCECriterion_updateOutput i t o b w = Dynamic.bCECriterion_updateOutput (asDynamic i) (asDynamic t) (asDynamic o) b (asDynamic w)
  _bCECriterion_updateGradInput i t go gi b w = Dynamic.bCECriterion_updateGradInput (asDynamic i) (asDynamic t) (asDynamic go) (asDynamic gi) b (asDynamic w)


