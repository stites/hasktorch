{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorMathCompare where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_ltValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_ltValue"
  c_ltValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_gtValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_gtValue"
  c_gtValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_leValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_leValue"
  c_leValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_geValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_geValue"
  c_geValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_eqValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_eqValue"
  c_eqValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_neValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_neValue"
  c_neValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_ltValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_ltValueT"
  c_ltValueT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_gtValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_gtValueT"
  c_gtValueT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_leValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_leValueT"
  c_leValueT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_geValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_geValueT"
  c_geValueT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_eqValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_eqValueT"
  c_eqValueT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | c_neValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCCharTensor_neValueT"
  c_neValueT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ()

-- | p_ltValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_ltValue"
  p_ltValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_gtValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_gtValue"
  p_gtValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_leValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_leValue"
  p_leValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_geValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_geValue"
  p_geValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_eqValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_eqValue"
  p_eqValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_neValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_neValue"
  p_neValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_ltValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_gtValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_leValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_leValueT"
  p_leValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_geValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_geValueT"
  p_geValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_eqValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())

-- | p_neValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCCharTensor_neValueT"
  p_neValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CChar -> IO ())