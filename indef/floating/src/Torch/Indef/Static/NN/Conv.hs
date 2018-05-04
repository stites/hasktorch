module Torch.Indef.Static.NN.Conv where

import qualified Torch.Class.NN as Dynamic
import qualified Torch.Class.NN.Static.Conv as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.NN ()
import Torch.Indef.Static.Tensor ()

instance Class.TemporalConvolutions Tensor where
  _temporalConvolution_updateOutput i o w b = Dynamic.temporalConvolution_updateOutput (asDynamic i) (asDynamic o) (asDynamic w) (asDynamic b)



