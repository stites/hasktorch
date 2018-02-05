{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensorRandom
  ( c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_geometric
  , c_bernoulli
  , c_bernoulli_FloatTensor
  , c_bernoulli_DoubleTensor
  , c_bernoulli_Tensor
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_geometric
  , p_bernoulli
  , p_bernoulli_FloatTensor
  , p_bernoulli_DoubleTensor
  , p_bernoulli_Tensor
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_random : self _generator -> void
foreign import ccall "THTensorRandom.h random"
  c_random :: Ptr CTHIntTensor -> Ptr CTHGenerator -> IO ()

-- | c_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h clampedRandom"
  c_clampedRandom :: Ptr CTHIntTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h cappedRandom"
  c_cappedRandom :: Ptr CTHIntTensor -> Ptr CTHGenerator -> CLLong -> IO ()

-- | c_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h geometric"
  c_geometric :: Ptr CTHIntTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h bernoulli"
  c_bernoulli :: Ptr CTHIntTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr CTHIntTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHIntTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- | c_bernoulli_Tensor : self _generator p -> void
foreign import ccall "THTensorRandom.h bernoulli_Tensor"
  c_bernoulli_Tensor :: Ptr CTHIntTensor -> Ptr CTHGenerator -> Ptr CTHIntTensor -> IO ()

-- |p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &random"
  p_random :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> IO ())

-- |p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- |p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &geometric"
  p_geometric :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- |p_bernoulli_Tensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &bernoulli_Tensor"
  p_bernoulli_Tensor :: FunPtr (Ptr CTHIntTensor -> Ptr CTHGenerator -> Ptr CTHIntTensor -> IO ())