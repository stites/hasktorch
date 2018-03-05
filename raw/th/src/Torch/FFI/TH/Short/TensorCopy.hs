{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorCopy
  ( c_copy
  , c_copyByte
  , c_copyChar
  , c_copyShort
  , c_copyInt
  , c_copyLong
  , c_copyFloat
  , c_copyDouble
  , c_copyHalf
  , p_copy
  , p_copyByte
  , p_copyChar
  , p_copyShort
  , p_copyInt
  , p_copyLong
  , p_copyFloat
  , p_copyDouble
  , p_copyHalf
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copy"
  c_copy :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyByte"
  c_copyByte :: Ptr (CTHShortTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyChar"
  c_copyChar :: Ptr (CTHShortTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyShort"
  c_copyShort :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyInt"
  c_copyInt :: Ptr (CTHShortTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyLong"
  c_copyLong :: Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyFloat"
  c_copyFloat :: Ptr (CTHShortTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyDouble"
  c_copyDouble :: Ptr (CTHShortTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorShort_copyHalf"
  c_copyHalf :: Ptr (CTHShortTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copy"
  p_copy :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorShort_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHHalfTensor) -> IO (()))