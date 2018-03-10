{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathMagma where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gesv :  state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_gesv"
  c_gesv :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_gels :  state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_gels"
  c_gels :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_syev :  state re_ rv_ a_ jobz uplo -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_syev"
  c_syev :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_geev :  state re_ rv_ a_ jobvr -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_geev"
  c_geev :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_gesvd :  state ru_ rs_ rv_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_gesvd"
  c_gesvd :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_gesvd2 :  state ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_gesvd2"
  c_gesvd2 :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_getri :  state ra_ a -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_getri"
  c_getri :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_potri :  state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_potri"
  c_potri :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_potrf :  state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_potrf"
  c_potrf :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_potrs :  state rb_ a b uplo -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_potrs"
  c_potrs :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_geqrf :  state ra_ rtau_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_geqrf"
  c_geqrf :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_qr :  state rq_ rr_ a -> void
foreign import ccall "THCTensorMathMagma.h THCFloatTensor_qr"
  c_qr :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | p_gesv : Pointer to function : state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_gesv"
  p_gesv :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_gels : Pointer to function : state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_gels"
  p_gels :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_syev : Pointer to function : state re_ rv_ a_ jobz uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_syev"
  p_syev :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- | p_geev : Pointer to function : state re_ rv_ a_ jobvr -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_geev"
  p_geev :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_gesvd : Pointer to function : state ru_ rs_ rv_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_gesvd"
  p_gesvd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_gesvd2 : Pointer to function : state ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_gesvd2"
  p_gesvd2 :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_getri : Pointer to function : state ra_ a -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_getri"
  p_getri :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_potri : Pointer to function : state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_potri"
  p_potri :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_potrf : Pointer to function : state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_potrf"
  p_potrf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_potrs : Pointer to function : state rb_ a b uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_potrs"
  p_potrs :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_geqrf : Pointer to function : state ra_ rtau_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_geqrf"
  p_geqrf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_qr : Pointer to function : state rq_ rr_ a -> void
foreign import ccall "THCTensorMathMagma.h &THCFloatTensor_qr"
  p_qr :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())