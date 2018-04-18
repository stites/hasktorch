{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Tensor where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_storage :  self -> THStorage *
foreign import ccall "THTensor.h THShortTensor_storage"
  c_storage_ :: Ptr C'THShortTensor -> IO (Ptr C'THShortStorage)

-- | alias of c_storage_ with unused argument (for CTHState) to unify backpack signatures.
c_storage :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THShortStorage)
c_storage = const c_storage_

-- | c_storageOffset :  self -> ptrdiff_t
foreign import ccall "THTensor.h THShortTensor_storageOffset"
  c_storageOffset_ :: Ptr C'THShortTensor -> IO CPtrdiff

-- | alias of c_storageOffset_ with unused argument (for CTHState) to unify backpack signatures.
c_storageOffset :: Ptr C'THState -> Ptr C'THShortTensor -> IO CPtrdiff
c_storageOffset = const c_storageOffset_

-- | c_nDimension :  self -> int
foreign import ccall "THTensor.h THShortTensor_nDimension"
  c_nDimension_ :: Ptr C'THShortTensor -> IO CInt

-- | alias of c_nDimension_ with unused argument (for CTHState) to unify backpack signatures.
c_nDimension :: Ptr C'THState -> Ptr C'THShortTensor -> IO CInt
c_nDimension = const c_nDimension_

-- | c_size :  self dim -> int64_t
foreign import ccall "THTensor.h THShortTensor_size"
  c_size_ :: Ptr C'THShortTensor -> CInt -> IO CLLong

-- | alias of c_size_ with unused argument (for CTHState) to unify backpack signatures.
c_size :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> IO CLLong
c_size = const c_size_

-- | c_stride :  self dim -> int64_t
foreign import ccall "THTensor.h THShortTensor_stride"
  c_stride_ :: Ptr C'THShortTensor -> CInt -> IO CLLong

-- | alias of c_stride_ with unused argument (for CTHState) to unify backpack signatures.
c_stride :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> IO CLLong
c_stride = const c_stride_

-- | c_newSizeOf :  self -> THLongStorage *
foreign import ccall "THTensor.h THShortTensor_newSizeOf"
  c_newSizeOf_ :: Ptr C'THShortTensor -> IO (Ptr C'THLongStorage)

-- | alias of c_newSizeOf_ with unused argument (for CTHState) to unify backpack signatures.
c_newSizeOf :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THLongStorage)
c_newSizeOf = const c_newSizeOf_

-- | c_newStrideOf :  self -> THLongStorage *
foreign import ccall "THTensor.h THShortTensor_newStrideOf"
  c_newStrideOf_ :: Ptr C'THShortTensor -> IO (Ptr C'THLongStorage)

-- | alias of c_newStrideOf_ with unused argument (for CTHState) to unify backpack signatures.
c_newStrideOf :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THLongStorage)
c_newStrideOf = const c_newStrideOf_

-- | c_data :  self -> real *
foreign import ccall "THTensor.h THShortTensor_data"
  c_data_ :: Ptr C'THShortTensor -> IO (Ptr CShort)

-- | alias of c_data_ with unused argument (for CTHState) to unify backpack signatures.
c_data :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr CShort)
c_data = const c_data_

-- | c_setFlag :  self flag -> void
foreign import ccall "THTensor.h THShortTensor_setFlag"
  c_setFlag_ :: Ptr C'THShortTensor -> CChar -> IO ()

-- | alias of c_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_setFlag :: Ptr C'THState -> Ptr C'THShortTensor -> CChar -> IO ()
c_setFlag = const c_setFlag_

-- | c_clearFlag :  self flag -> void
foreign import ccall "THTensor.h THShortTensor_clearFlag"
  c_clearFlag_ :: Ptr C'THShortTensor -> CChar -> IO ()

-- | alias of c_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_clearFlag :: Ptr C'THState -> Ptr C'THShortTensor -> CChar -> IO ()
c_clearFlag = const c_clearFlag_

-- | c_new :   -> THTensor *
foreign import ccall "THTensor.h THShortTensor_new"
  c_new_ :: IO (Ptr C'THShortTensor)

-- | alias of c_new_ with unused argument (for CTHState) to unify backpack signatures.
c_new :: Ptr C'THState -> IO (Ptr C'THShortTensor)
c_new = const c_new_

-- | c_newWithTensor :  tensor -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithTensor"
  c_newWithTensor_ :: Ptr C'THShortTensor -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithTensor :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THShortTensor)
c_newWithTensor = const c_newWithTensor_

-- | c_newWithStorage :  storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage"
  c_newWithStorage_ :: Ptr C'THShortStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithStorage_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage :: Ptr C'THState -> Ptr C'THShortStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)
c_newWithStorage = const c_newWithStorage_

-- | c_newWithStorage1d :  storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage1d"
  c_newWithStorage1d_ :: Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithStorage1d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage1d :: Ptr C'THState -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithStorage1d = const c_newWithStorage1d_

-- | c_newWithStorage2d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage2d"
  c_newWithStorage2d_ :: Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithStorage2d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage2d :: Ptr C'THState -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithStorage2d = const c_newWithStorage2d_

-- | c_newWithStorage3d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage3d"
  c_newWithStorage3d_ :: Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithStorage3d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage3d :: Ptr C'THState -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithStorage3d = const c_newWithStorage3d_

-- | c_newWithStorage4d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage4d"
  c_newWithStorage4d_ :: Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithStorage4d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage4d :: Ptr C'THState -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithStorage4d = const c_newWithStorage4d_

-- | c_newWithSize :  size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize"
  c_newWithSize_ :: Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize :: Ptr C'THState -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)
c_newWithSize = const c_newWithSize_

-- | c_newWithSize1d :  size0_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize1d"
  c_newWithSize1d_ :: CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithSize1d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize1d :: Ptr C'THState -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithSize1d = const c_newWithSize1d_

-- | c_newWithSize2d :  size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize2d"
  c_newWithSize2d_ :: CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithSize2d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize2d :: Ptr C'THState -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithSize2d = const c_newWithSize2d_

-- | c_newWithSize3d :  size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize3d"
  c_newWithSize3d_ :: CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithSize3d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize3d :: Ptr C'THState -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithSize3d = const c_newWithSize3d_

-- | c_newWithSize4d :  size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize4d"
  c_newWithSize4d_ :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newWithSize4d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize4d :: Ptr C'THState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newWithSize4d = const c_newWithSize4d_

-- | c_newClone :  self -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newClone"
  c_newClone_ :: Ptr C'THShortTensor -> IO (Ptr C'THShortTensor)

-- | alias of c_newClone_ with unused argument (for CTHState) to unify backpack signatures.
c_newClone :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THShortTensor)
c_newClone = const c_newClone_

-- | c_newContiguous :  tensor -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newContiguous"
  c_newContiguous_ :: Ptr C'THShortTensor -> IO (Ptr C'THShortTensor)

-- | alias of c_newContiguous_ with unused argument (for CTHState) to unify backpack signatures.
c_newContiguous :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THShortTensor)
c_newContiguous = const c_newContiguous_

-- | c_newSelect :  tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newSelect"
  c_newSelect_ :: Ptr C'THShortTensor -> CInt -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_newSelect :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> CLLong -> IO (Ptr C'THShortTensor)
c_newSelect = const c_newSelect_

-- | c_newNarrow :  tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newNarrow"
  c_newNarrow_ :: Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newNarrow_ with unused argument (for CTHState) to unify backpack signatures.
c_newNarrow :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newNarrow = const c_newNarrow_

-- | c_newTranspose :  tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newTranspose"
  c_newTranspose_ :: Ptr C'THShortTensor -> CInt -> CInt -> IO (Ptr C'THShortTensor)

-- | alias of c_newTranspose_ with unused argument (for CTHState) to unify backpack signatures.
c_newTranspose :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> CInt -> IO (Ptr C'THShortTensor)
c_newTranspose = const c_newTranspose_

-- | c_newUnfold :  tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newUnfold"
  c_newUnfold_ :: Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)

-- | alias of c_newUnfold_ with unused argument (for CTHState) to unify backpack signatures.
c_newUnfold :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THShortTensor)
c_newUnfold = const c_newUnfold_

-- | c_newView :  tensor size -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newView"
  c_newView_ :: Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)

-- | alias of c_newView_ with unused argument (for CTHState) to unify backpack signatures.
c_newView :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)
c_newView = const c_newView_

-- | c_newExpand :  tensor size -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newExpand"
  c_newExpand_ :: Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)

-- | alias of c_newExpand_ with unused argument (for CTHState) to unify backpack signatures.
c_newExpand :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)
c_newExpand = const c_newExpand_

-- | c_expand :  r tensor size -> void
foreign import ccall "THTensor.h THShortTensor_expand"
  c_expand_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_expand_ with unused argument (for CTHState) to unify backpack signatures.
c_expand :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_

-- | c_expandNd :  rets ops count -> void
foreign import ccall "THTensor.h THShortTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THShortTensor) -> Ptr (Ptr C'THShortTensor) -> CInt -> IO ()

-- | alias of c_expandNd_ with unused argument (for CTHState) to unify backpack signatures.
c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THShortTensor) -> Ptr (Ptr C'THShortTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_

-- | c_resize :  tensor size stride -> void
foreign import ccall "THTensor.h THShortTensor_resize"
  c_resize_ :: Ptr C'THShortTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()

-- | alias of c_resize_ with unused argument (for CTHState) to unify backpack signatures.
c_resize :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()
c_resize = const c_resize_

-- | c_resizeAs :  tensor src -> void
foreign import ccall "THTensor.h THShortTensor_resizeAs"
  c_resizeAs_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | alias of c_resizeAs_ with unused argument (for CTHState) to unify backpack signatures.
c_resizeAs :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()
c_resizeAs = const c_resizeAs_

-- | c_resizeNd :  tensor nDimension size stride -> void
foreign import ccall "THTensor.h THShortTensor_resizeNd"
  c_resizeNd_ :: Ptr C'THShortTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | alias of c_resizeNd_ with unused argument (for CTHState) to unify backpack signatures.
c_resizeNd :: Ptr C'THState -> Ptr C'THShortTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
c_resizeNd = const c_resizeNd_

-- | c_resize1d :  tensor size0_ -> void
foreign import ccall "THTensor.h THShortTensor_resize1d"
  c_resize1d_ :: Ptr C'THShortTensor -> CLLong -> IO ()

-- | alias of c_resize1d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize1d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> IO ()
c_resize1d = const c_resize1d_

-- | c_resize2d :  tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THShortTensor_resize2d"
  c_resize2d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_resize2d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize2d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> IO ()
c_resize2d = const c_resize2d_

-- | c_resize3d :  tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THShortTensor_resize3d"
  c_resize3d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_resize3d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize3d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> IO ()
c_resize3d = const c_resize3d_

-- | c_resize4d :  tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THShortTensor_resize4d"
  c_resize4d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_resize4d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize4d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_resize4d = const c_resize4d_

-- | c_resize5d :  tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THShortTensor_resize5d"
  c_resize5d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_resize5d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize5d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_resize5d = const c_resize5d_

-- | c_set :  self src -> void
foreign import ccall "THTensor.h THShortTensor_set"
  c_set_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | alias of c_set_ with unused argument (for CTHState) to unify backpack signatures.
c_set :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()
c_set = const c_set_

-- | c_setStorage :  self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage"
  c_setStorage_ :: Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()

-- | alias of c_setStorage_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()
c_setStorage = const c_setStorage_

-- | c_setStorageNd :  self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THShortTensor_setStorageNd"
  c_setStorageNd_ :: Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | alias of c_setStorageNd_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorageNd :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
c_setStorageNd = const c_setStorageNd_

-- | c_setStorage1d :  self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage1d"
  c_setStorage1d_ :: Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage1d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage1d :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()
c_setStorage1d = const c_setStorage1d_

-- | c_setStorage2d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage2d"
  c_setStorage2d_ :: Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage2d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage2d :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_setStorage2d = const c_setStorage2d_

-- | c_setStorage3d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage3d"
  c_setStorage3d_ :: Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage3d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage3d :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_setStorage3d = const c_setStorage3d_

-- | c_setStorage4d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage4d"
  c_setStorage4d_ :: Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage4d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage4d :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
c_setStorage4d = const c_setStorage4d_

-- | c_narrow :  self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THShortTensor_narrow"
  c_narrow_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | alias of c_narrow_ with unused argument (for CTHState) to unify backpack signatures.
c_narrow :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO ()
c_narrow = const c_narrow_

-- | c_select :  self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THShortTensor_select"
  c_select_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> IO ()

-- | alias of c_select_ with unused argument (for CTHState) to unify backpack signatures.
c_select :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> IO ()
c_select = const c_select_

-- | c_transpose :  self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THShortTensor_transpose"
  c_transpose_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()

-- | alias of c_transpose_ with unused argument (for CTHState) to unify backpack signatures.
c_transpose :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ()
c_transpose = const c_transpose_

-- | c_unfold :  self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THShortTensor_unfold"
  c_unfold_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | alias of c_unfold_ with unused argument (for CTHState) to unify backpack signatures.
c_unfold :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO ()
c_unfold = const c_unfold_

-- | c_squeeze :  self src -> void
foreign import ccall "THTensor.h THShortTensor_squeeze"
  c_squeeze_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | alias of c_squeeze_ with unused argument (for CTHState) to unify backpack signatures.
c_squeeze :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()
c_squeeze = const c_squeeze_

-- | c_squeeze1d :  self src dimension_ -> void
foreign import ccall "THTensor.h THShortTensor_squeeze1d"
  c_squeeze1d_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | alias of c_squeeze1d_ with unused argument (for CTHState) to unify backpack signatures.
c_squeeze1d :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()
c_squeeze1d = const c_squeeze1d_

-- | c_unsqueeze1d :  self src dimension_ -> void
foreign import ccall "THTensor.h THShortTensor_unsqueeze1d"
  c_unsqueeze1d_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()

-- | alias of c_unsqueeze1d_ with unused argument (for CTHState) to unify backpack signatures.
c_unsqueeze1d :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ()
c_unsqueeze1d = const c_unsqueeze1d_

-- | c_isContiguous :  self -> int
foreign import ccall "THTensor.h THShortTensor_isContiguous"
  c_isContiguous_ :: Ptr C'THShortTensor -> IO CInt

-- | alias of c_isContiguous_ with unused argument (for CTHState) to unify backpack signatures.
c_isContiguous :: Ptr C'THState -> Ptr C'THShortTensor -> IO CInt
c_isContiguous = const c_isContiguous_

-- | c_isSameSizeAs :  self src -> int
foreign import ccall "THTensor.h THShortTensor_isSameSizeAs"
  c_isSameSizeAs_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO CInt

-- | alias of c_isSameSizeAs_ with unused argument (for CTHState) to unify backpack signatures.
c_isSameSizeAs :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO CInt
c_isSameSizeAs = const c_isSameSizeAs_

-- | c_isSetTo :  self src -> int
foreign import ccall "THTensor.h THShortTensor_isSetTo"
  c_isSetTo_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO CInt

-- | alias of c_isSetTo_ with unused argument (for CTHState) to unify backpack signatures.
c_isSetTo :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO CInt
c_isSetTo = const c_isSetTo_

-- | c_isSize :  self dims -> int
foreign import ccall "THTensor.h THShortTensor_isSize"
  c_isSize_ :: Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO CInt

-- | alias of c_isSize_ with unused argument (for CTHState) to unify backpack signatures.
c_isSize :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO CInt
c_isSize = const c_isSize_

-- | c_nElement :  self -> ptrdiff_t
foreign import ccall "THTensor.h THShortTensor_nElement"
  c_nElement_ :: Ptr C'THShortTensor -> IO CPtrdiff

-- | alias of c_nElement_ with unused argument (for CTHState) to unify backpack signatures.
c_nElement :: Ptr C'THState -> Ptr C'THShortTensor -> IO CPtrdiff
c_nElement = const c_nElement_

-- | c_retain :  self -> void
foreign import ccall "THTensor.h THShortTensor_retain"
  c_retain_ :: Ptr C'THShortTensor -> IO ()

-- | alias of c_retain_ with unused argument (for CTHState) to unify backpack signatures.
c_retain :: Ptr C'THState -> Ptr C'THShortTensor -> IO ()
c_retain = const c_retain_

-- | c_free :  self -> void
foreign import ccall "THTensor.h THShortTensor_free"
  c_free_ :: Ptr C'THShortTensor -> IO ()

-- | alias of c_free_ with unused argument (for CTHState) to unify backpack signatures.
c_free :: Ptr C'THState -> Ptr C'THShortTensor -> IO ()
c_free = const c_free_

-- | c_freeCopyTo :  self dst -> void
foreign import ccall "THTensor.h THShortTensor_freeCopyTo"
  c_freeCopyTo_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()

-- | alias of c_freeCopyTo_ with unused argument (for CTHState) to unify backpack signatures.
c_freeCopyTo :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ()
c_freeCopyTo = const c_freeCopyTo_

-- | c_set1d :  tensor x0 value -> void
foreign import ccall "THTensor.h THShortTensor_set1d"
  c_set1d_ :: Ptr C'THShortTensor -> CLLong -> CShort -> IO ()

-- | alias of c_set1d_ with unused argument (for CTHState) to unify backpack signatures.
c_set1d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CShort -> IO ()
c_set1d = const c_set1d_

-- | c_set2d :  tensor x0 x1 value -> void
foreign import ccall "THTensor.h THShortTensor_set2d"
  c_set2d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CShort -> IO ()

-- | alias of c_set2d_ with unused argument (for CTHState) to unify backpack signatures.
c_set2d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CShort -> IO ()
c_set2d = const c_set2d_

-- | c_set3d :  tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THShortTensor_set3d"
  c_set3d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CShort -> IO ()

-- | alias of c_set3d_ with unused argument (for CTHState) to unify backpack signatures.
c_set3d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CShort -> IO ()
c_set3d = const c_set3d_

-- | c_set4d :  tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THShortTensor_set4d"
  c_set4d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CShort -> IO ()

-- | alias of c_set4d_ with unused argument (for CTHState) to unify backpack signatures.
c_set4d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CShort -> IO ()
c_set4d = const c_set4d_

-- | c_get1d :  tensor x0 -> real
foreign import ccall "THTensor.h THShortTensor_get1d"
  c_get1d_ :: Ptr C'THShortTensor -> CLLong -> IO CShort

-- | alias of c_get1d_ with unused argument (for CTHState) to unify backpack signatures.
c_get1d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> IO CShort
c_get1d = const c_get1d_

-- | c_get2d :  tensor x0 x1 -> real
foreign import ccall "THTensor.h THShortTensor_get2d"
  c_get2d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> IO CShort

-- | alias of c_get2d_ with unused argument (for CTHState) to unify backpack signatures.
c_get2d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> IO CShort
c_get2d = const c_get2d_

-- | c_get3d :  tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THShortTensor_get3d"
  c_get3d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> IO CShort

-- | alias of c_get3d_ with unused argument (for CTHState) to unify backpack signatures.
c_get3d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> IO CShort
c_get3d = const c_get3d_

-- | c_get4d :  tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THShortTensor_get4d"
  c_get4d_ :: Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CShort

-- | alias of c_get4d_ with unused argument (for CTHState) to unify backpack signatures.
c_get4d :: Ptr C'THState -> Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CShort
c_get4d = const c_get4d_

-- | c_desc :  tensor -> THDescBuff
foreign import ccall "THTensor.h THShortTensor_desc"
  c_desc_ :: Ptr C'THShortTensor -> IO (Ptr C'THDescBuff)

-- | alias of c_desc_ with unused argument (for CTHState) to unify backpack signatures.
c_desc :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THDescBuff)
c_desc = const c_desc_

-- | c_sizeDesc :  tensor -> THDescBuff
foreign import ccall "THTensor.h THShortTensor_sizeDesc"
  c_sizeDesc_ :: Ptr C'THShortTensor -> IO (Ptr C'THDescBuff)

-- | alias of c_sizeDesc_ with unused argument (for CTHState) to unify backpack signatures.
c_sizeDesc :: Ptr C'THState -> Ptr C'THShortTensor -> IO (Ptr C'THDescBuff)
c_sizeDesc = const c_sizeDesc_

-- | p_storage : Pointer to function : self -> THStorage *
foreign import ccall "THTensor.h &THShortTensor_storage"
  p_storage :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THShortStorage))

-- | p_storageOffset : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THShortTensor_storageOffset"
  p_storageOffset :: FunPtr (Ptr C'THShortTensor -> IO CPtrdiff)

-- | p_nDimension : Pointer to function : self -> int
foreign import ccall "THTensor.h &THShortTensor_nDimension"
  p_nDimension :: FunPtr (Ptr C'THShortTensor -> IO CInt)

-- | p_size : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THShortTensor_size"
  p_size :: FunPtr (Ptr C'THShortTensor -> CInt -> IO CLLong)

-- | p_stride : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THShortTensor_stride"
  p_stride :: FunPtr (Ptr C'THShortTensor -> CInt -> IO CLLong)

-- | p_newSizeOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THShortTensor_newSizeOf"
  p_newSizeOf :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THLongStorage))

-- | p_newStrideOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THShortTensor_newStrideOf"
  p_newStrideOf :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THLongStorage))

-- | p_data : Pointer to function : self -> real *
foreign import ccall "THTensor.h &THShortTensor_data"
  p_data :: FunPtr (Ptr C'THShortTensor -> IO (Ptr CShort))

-- | p_setFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THShortTensor_setFlag"
  p_setFlag :: FunPtr (Ptr C'THShortTensor -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THShortTensor_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THShortTensor -> CChar -> IO ())

-- | p_new : Pointer to function :  -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_new"
  p_new :: FunPtr (IO (Ptr C'THShortTensor))

-- | p_newWithTensor : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithTensor"
  p_newWithTensor :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THShortTensor))

-- | p_newWithStorage : Pointer to function : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage"
  p_newWithStorage :: FunPtr (Ptr C'THShortStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor))

-- | p_newWithStorage1d : Pointer to function : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithStorage2d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithStorage3d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithStorage4d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithSize : Pointer to function : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor))

-- | p_newWithSize1d : Pointer to function : size0_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize1d"
  p_newWithSize1d :: FunPtr (CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithSize2d : Pointer to function : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize2d"
  p_newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithSize3d : Pointer to function : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize3d"
  p_newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newWithSize4d : Pointer to function : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize4d"
  p_newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newClone : Pointer to function : self -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newClone"
  p_newClone :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THShortTensor))

-- | p_newContiguous : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newContiguous"
  p_newContiguous :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THShortTensor))

-- | p_newSelect : Pointer to function : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newSelect"
  p_newSelect :: FunPtr (Ptr C'THShortTensor -> CInt -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newNarrow : Pointer to function : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newNarrow"
  p_newNarrow :: FunPtr (Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newTranspose : Pointer to function : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newTranspose"
  p_newTranspose :: FunPtr (Ptr C'THShortTensor -> CInt -> CInt -> IO (Ptr C'THShortTensor))

-- | p_newUnfold : Pointer to function : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newUnfold"
  p_newUnfold :: FunPtr (Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THShortTensor))

-- | p_newView : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newView"
  p_newView :: FunPtr (Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor))

-- | p_newExpand : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newExpand"
  p_newExpand :: FunPtr (Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor))

-- | p_expand : Pointer to function : r tensor size -> void
foreign import ccall "THTensor.h &THShortTensor_expand"
  p_expand :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO ())

-- | p_expandNd : Pointer to function : rets ops count -> void
foreign import ccall "THTensor.h &THShortTensor_expandNd"
  p_expandNd :: FunPtr (Ptr (Ptr C'THShortTensor) -> Ptr (Ptr C'THShortTensor) -> CInt -> IO ())

-- | p_resize : Pointer to function : tensor size stride -> void
foreign import ccall "THTensor.h &THShortTensor_resize"
  p_resize :: FunPtr (Ptr C'THShortTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ())

-- | p_resizeAs : Pointer to function : tensor src -> void
foreign import ccall "THTensor.h &THShortTensor_resizeAs"
  p_resizeAs :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_resizeNd : Pointer to function : tensor nDimension size stride -> void
foreign import ccall "THTensor.h &THShortTensor_resizeNd"
  p_resizeNd :: FunPtr (Ptr C'THShortTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_resize1d : Pointer to function : tensor size0_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize1d"
  p_resize1d :: FunPtr (Ptr C'THShortTensor -> CLLong -> IO ())

-- | p_resize2d : Pointer to function : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize2d"
  p_resize2d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> IO ())

-- | p_resize3d : Pointer to function : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize3d"
  p_resize3d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize4d : Pointer to function : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize4d"
  p_resize4d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize5d : Pointer to function : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize5d"
  p_resize5d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_set : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THShortTensor_set"
  p_set :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_setStorage : Pointer to function : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage"
  p_setStorage :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ())

-- | p_setStorageNd : Pointer to function : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h &THShortTensor_setStorageNd"
  p_setStorageNd :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_setStorage1d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage1d"
  p_setStorage1d :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- | p_setStorage2d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage2d"
  p_setStorage2d :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage3d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage3d"
  p_setStorage3d :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage4d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage4d"
  p_setStorage4d :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_narrow : Pointer to function : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h &THShortTensor_narrow"
  p_narrow :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_select : Pointer to function : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h &THShortTensor_select"
  p_select :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> IO ())

-- | p_transpose : Pointer to function : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h &THShortTensor_transpose"
  p_transpose :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CInt -> IO ())

-- | p_unfold : Pointer to function : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h &THShortTensor_unfold"
  p_unfold :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_squeeze : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THShortTensor_squeeze"
  p_squeeze :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_squeeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THShortTensor_squeeze1d"
  p_squeeze1d :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_unsqueeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THShortTensor_unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> CInt -> IO ())

-- | p_isContiguous : Pointer to function : self -> int
foreign import ccall "THTensor.h &THShortTensor_isContiguous"
  p_isContiguous :: FunPtr (Ptr C'THShortTensor -> IO CInt)

-- | p_isSameSizeAs : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THShortTensor_isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO CInt)

-- | p_isSetTo : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THShortTensor_isSetTo"
  p_isSetTo :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO CInt)

-- | p_isSize : Pointer to function : self dims -> int
foreign import ccall "THTensor.h &THShortTensor_isSize"
  p_isSize :: FunPtr (Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO CInt)

-- | p_nElement : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THShortTensor_nElement"
  p_nElement :: FunPtr (Ptr C'THShortTensor -> IO CPtrdiff)

-- | p_retain : Pointer to function : self -> void
foreign import ccall "THTensor.h &THShortTensor_retain"
  p_retain :: FunPtr (Ptr C'THShortTensor -> IO ())

-- | p_free : Pointer to function : self -> void
foreign import ccall "THTensor.h &THShortTensor_free"
  p_free :: FunPtr (Ptr C'THShortTensor -> IO ())

-- | p_freeCopyTo : Pointer to function : self dst -> void
foreign import ccall "THTensor.h &THShortTensor_freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr C'THShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_set1d : Pointer to function : tensor x0 value -> void
foreign import ccall "THTensor.h &THShortTensor_set1d"
  p_set1d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CShort -> IO ())

-- | p_set2d : Pointer to function : tensor x0 x1 value -> void
foreign import ccall "THTensor.h &THShortTensor_set2d"
  p_set2d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CShort -> IO ())

-- | p_set3d : Pointer to function : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h &THShortTensor_set3d"
  p_set3d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CShort -> IO ())

-- | p_set4d : Pointer to function : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h &THShortTensor_set4d"
  p_set4d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CShort -> IO ())

-- | p_get1d : Pointer to function : tensor x0 -> real
foreign import ccall "THTensor.h &THShortTensor_get1d"
  p_get1d :: FunPtr (Ptr C'THShortTensor -> CLLong -> IO CShort)

-- | p_get2d : Pointer to function : tensor x0 x1 -> real
foreign import ccall "THTensor.h &THShortTensor_get2d"
  p_get2d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> IO CShort)

-- | p_get3d : Pointer to function : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h &THShortTensor_get3d"
  p_get3d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> IO CShort)

-- | p_get4d : Pointer to function : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h &THShortTensor_get4d"
  p_get4d :: FunPtr (Ptr C'THShortTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CShort)

-- | p_desc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THShortTensor_desc"
  p_desc :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THDescBuff))

-- | p_sizeDesc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THShortTensor_sizeDesc"
  p_sizeDesc :: FunPtr (Ptr C'THShortTensor -> IO (Ptr C'THDescBuff))