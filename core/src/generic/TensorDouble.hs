{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TensorDouble (
  tdNew,
  tensorDoubleInit,

  -- TODO - use this convention for everything
  td_get,
  td_newWithTensor
  )
where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw
import TensorTypes
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import THDoubleLapack

w2cl = fromIntegral

td_get :: TensorDim Integer -> TensorDouble -> IO Double
td_get loc tensor =
  withForeignPtr
    (tdTensor tensor)
    (\t -> pure . realToFrac $ getter loc t)
  where
    getter D0 t = undefined
    getter (D1 d1) t = c_THDoubleTensor_get1d t $ w2cl d1
    getter (D2 d1 d2) t = c_THDoubleTensor_get2d t
                          (w2cl d1) (w2cl d2)
    getter (D3 d1 d2 d3) t = c_THDoubleTensor_get3d t
                             (w2cl d1) (w2cl d2) (w2cl d3)
    getter (D4 d1 d2 d3 d4) t = c_THDoubleTensor_get4d t
                                (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr -> c_THDoubleTensor_newWithTensor tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)


-- |Create a new (double) tensor of specified dimensions and fill it with 0
tdNew :: TensorDim Word -> TensorDouble
tdNew dims = unsafePerformIO $ do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims

tensorDoubleInit :: TensorDim Word -> Double -> TensorDouble
tensorDoubleInit dims value = unsafePerformIO $ do
  newPtr <- tensorRaw dims value
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr (fillRaw value)
  pure $ TensorDouble fPtr dims

test :: IO ()
test = do
  let foo = tdNew (D1 5)
  -- disp foo
  pure ()
