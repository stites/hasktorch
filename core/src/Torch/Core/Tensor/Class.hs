module Torch.Core.Tensor.Class
  ( TensorClass(..)
  , Int8, Int32, Int64
  ) where


import Foreign.C.Types (CPtrdiff)
import GHC.Int (Int8, Int32, Int64)

import Torch.Raw.Internal (HaskReal, CTHDescBuff, Storage)
import Torch.Core.StorageTypes (StorageLong)

class TensorClass t where
  clearFlag :: t -> Int8 -> IO ()
  tensordata :: t -> IO (HaskReal t)
  desc :: t -> IO CTHDescBuff
  expand :: t -> t -> StorageLong -> IO ()
  expandNd :: t -> t -> Int32 -> IO ()
  get1d :: t -> Int64 -> IO (HaskReal t)
  get2d :: t -> Int64 -> Int64 -> IO (HaskReal t)
  get3d :: t -> Int64 -> Int64 -> Int64 -> IO (HaskReal t)
  get4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HaskReal t)
  isContiguous :: t -> IO Bool
  isSameSizeAs :: t -> t -> IO Bool
  isSetTo :: t -> t -> IO Bool
  isSize :: t -> StorageLong -> IO Bool
  nDimension :: t -> IO Int
  nElement :: t -> IO CPtrdiff
  narrow :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  new :: IO t
  newClone :: t -> IO t
  newContiguous :: t -> IO t
  newExpand :: t -> StorageLong -> IO t
  newNarrow :: t -> Int32 -> Int64 -> Int64 -> IO t
  newSelect :: t -> Int32 -> Int64 -> IO t
  newSizeOf :: t -> IO StorageLong
  newStrideOf :: t -> IO StorageLong
  newTranspose :: t -> Int32 -> Int32 -> IO t
  newUnfold :: t -> Int32 -> Int64 -> Int64 -> IO t
  newView :: t -> StorageLong -> IO t
  newWithSize :: StorageLong -> StorageLong -> IO t
  newWithSize1d :: Int64 -> IO t
  newWithSize2d :: Int64 -> Int64 -> IO t
  newWithSize3d :: Int64 -> Int64 -> Int64 -> IO t
  newWithSize4d :: Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage   :: Storage t -> CPtrdiff -> StorageLong -> StorageLong -> IO t
  newWithStorage1d :: Storage t -> CPtrdiff -> Int64 -> Int64 -> IO t
  newWithStorage2d :: Storage t -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage3d :: Storage t -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage4d :: Storage t -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithTensor :: t -> IO t
  resize :: t -> StorageLong -> StorageLong -> IO ()
  resize1d :: t -> Int64 -> IO ()
  resize2d :: t -> Int64 -> Int64 -> IO ()
  resize3d :: t -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resizeAs :: t -> t -> IO ()
  resizeNd :: t -> Int32 -> Int64 -> Int64 -> IO ()
  retain :: t -> IO ()
  select :: t -> t -> Int32 -> Int64 -> IO ()
  set :: t -> t -> IO ()
  set1d :: t -> Int64 -> HaskReal t -> IO ()
  set2d :: t -> Int64 -> Int64 -> HaskReal t -> IO ()
  set3d :: t -> Int64 -> Int64 -> Int64 -> HaskReal t -> IO ()
  set4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> HaskReal t -> IO ()
  setFlag :: t -> Int8 -> IO ()
  setStorage :: t -> Storage t -> CPtrdiff -> StorageLong -> StorageLong -> IO ()
  setStorage1d :: t -> Storage t -> CPtrdiff -> Int64 -> Int64 -> IO ()
  setStorage2d :: t -> Storage t -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage3d :: t -> Storage t -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage4d :: t -> Storage t -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorageNd :: t -> Storage t -> CPtrdiff -> Int32 -> Int64 -> Int64 -> IO ()
  size :: t -> Int32 -> IO Int64
  sizeDesc :: t -> IO CTHDescBuff
  squeeze :: t -> t -> IO ()
  squeeze1d :: t -> t -> Int32 -> IO ()
  storage :: t -> IO (Storage t)
  storageOffset :: t -> IO CPtrdiff
  stride :: t -> Int32 -> IO Int64
  transpose :: t -> t -> Int32 -> Int32 -> IO ()
  unfold :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  unsqueeze1d :: t -> t -> Int32 -> IO ()
