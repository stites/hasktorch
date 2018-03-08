{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Storage
  ( c_data
  , c_size
  , c_set
  , c_get
  , c_new
  , c_newWithSize
  , c_newWithSize1
  , c_newWithSize2
  , c_newWithSize3
  , c_newWithSize4
  , c_newWithMapping
  , c_newWithData
  , c_setFlag
  , c_clearFlag
  , c_retain
  , c_free
  , c_resize
  , c_fill
  , c_getDevice
  , p_data
  , p_size
  , p_set
  , p_get
  , p_new
  , p_newWithSize
  , p_newWithSize1
  , p_newWithSize2
  , p_newWithSize3
  , p_newWithSize4
  , p_newWithMapping
  , p_newWithData
  , p_setFlag
  , p_clearFlag
  , p_retain
  , p_free
  , p_resize
  , p_fill
  , p_getDevice
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCharStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (Ptr (CChar))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCharStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCharStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> CPtrdiff -> CChar -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCharStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> CPtrdiff -> IO (CChar)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CChar -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CChar -> CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHCharStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THCharStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> IO (Ptr (CTHCharStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCharStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCharStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCharStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCharStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCharStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCharStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> CChar -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCharStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCharStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (Ptr (CChar)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCharStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCharStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> CPtrdiff -> CChar -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCharStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> CPtrdiff -> IO (CChar))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CChar -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CChar -> CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHCharStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THCharStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> IO (Ptr (CTHCharStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCharStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCharStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCharStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCharStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCharStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCharStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> CChar -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCharStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharStorage) -> IO (CInt))