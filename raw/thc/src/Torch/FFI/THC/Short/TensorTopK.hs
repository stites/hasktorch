{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorTopK
  ( c_topk
  , p_topk
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THShortTensor_topk"
  c_topk :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THShortTensor_topk"
  p_topk :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))