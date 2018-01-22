{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs, RankNTypes, PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeInType #-}
module Torch.Core.Tensor.Dynamic.Double
  ( TensorDouble(..)
  , DynamicTH(..)
  , shapeList
  , rank
  -- These don't need to be exported, but concrete types help with tests
  , td_fromListNd
  , td_fromList1d
  , td_resize
  , td_get
  , td_newWithTensor
  , td_new
  , td_new_
  , td_free_
  , td_init
  , td_transpose
  , td_trans
  , td_shape
  ) where

import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Class (TensorClass(..), Int8, Int32, Int64)
import Torch.Core.Tensor.Dim (Dim(..), SomeDims(..), someDimsM)
import Torch.Core.Tensor.Types (TensorDouble(..), THForeignRef(getForeign), THForeignType)
import Torch.Core.StorageTypes
import Torch.Raw.Internal (CTHDoubleTensor, CTHLongTensor)
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Raw.Tensor.Generic as Gen
import qualified Torch.Core.Tensor.Dim as Dim


instance IsList TensorDouble where
  type Item TensorDouble = Double
  fromList = td_fromList1d
  toList td = unsafePerformIO $ withForeignPtr (getForeign td) (pure . fmap realToFrac . Gen.flatten)
  {-# NOINLINE toList #-}


class IsList t => DynamicTH t where
  printTensor :: t -> IO ()
  fromListNd :: SomeDims -> [Item t] -> t
  fromList1d :: [Item t] -> t
  resize :: t -> SomeDims -> t
  get :: SomeDims -> t -> Item t
  newWithTensor :: t -> t
  new :: SomeDims -> t
  new_ :: SomeDims -> IO t
  free_ :: t -> IO ()
  init :: SomeDims -> Item t -> t
  transpose :: Word -> Word -> t -> t
  trans :: t -> t
  shape :: t -> SomeDims


instance DynamicTH TensorDouble where
  printTensor = td_p
  fromListNd (SomeDims d) = td_fromListNd d
  fromList1d = td_fromList1d
  resize t (SomeDims d) = td_resize t d
  get (SomeDims d) = td_get d
  newWithTensor = td_newWithTensor
  new (SomeDims d) = td_new d
  new_ (SomeDims d) = td_new_ d
  free_ = td_free_
  init (SomeDims d) = td_init d
  transpose = td_transpose
  trans = td_trans
  shape = td_shape

td_p :: TensorDouble -> IO ()
td_p t = withForeignPtr (getForeign t) Gen.dispRaw

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME(stites): This should go in MonadThrow
td_fromListNd :: k ~ Nat => Dim (d::[k]) -> [Double] -> TensorDouble
td_fromListNd d l =
  if fromIntegral (product (Dim.dimVals d)) == length l
  then td_resize (td_fromList1d l) d
  else error "Incorrect tensor dimensions specified."

-- |Initialize a 1D tensor from a list
td_fromList1d :: [Double] -> TensorDouble
td_fromList1d l = unsafePerformIO $ do
  sdims <- someDimsM [length l]
  let res = td_new' sdims
  mapM_ (mutTensor (getForeign res)) (zip [0..length l - 1] l)
  pure res
 where
  mutTensor :: ForeignPtr CTHDoubleTensor -> (Int, Double) -> IO ()
  mutTensor t (idx, value) = withForeignPtr t $ \tp ->
    Gen.c_set1d tp (fromIntegral idx) (realToFrac value)
{-# NOINLINE td_fromList1d #-}

-- |Copy contents of tensor into a new one of specified size
td_resize :: k ~ Nat => TensorDouble -> Dim (d::[k]) -> TensorDouble
td_resize t d = unsafePerformIO $ do
  let resDummy = td_new d
  newPtr <- withForeignPtr (getForeign t) Gen.c_newClone
  newFPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  withForeignPtr newFPtr (withForeignPtr (getForeign resDummy) . Gen.c_resizeAs)
  pure $ TensorDouble newFPtr
{-# NOINLINE td_resize #-}

td_get :: Dim (d::[k]) -> TensorDouble -> Double
td_get loc tensor = unsafePerformIO $ withForeignPtr
  (getForeign tensor)
  (\t -> pure . realToFrac $ t `Gen.genericGet` loc)
{-# NOINLINE td_get #-}

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) Gen.c_newWithTensor
  newFPtr <- newForeignPtr Gen.p_free newPtr
  -- ds <- someDimsM (Gen.dimList newPtr)
  pure $ TensorDouble newFPtr
{-# NOINLINE td_newWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new :: k ~ Nat => Dim (d::[k]) -> TensorDouble
td_new dims = unsafePerformIO $ do
  newPtr <- Gen.constant dims 0.0
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ TensorDouble fPtr -- (SomeDims dims)
{-# NOINLINE td_new #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new' :: SomeDims -> TensorDouble
td_new' sdims = unsafePerformIO $ do
  newPtr <- Gen.constant' sdims 0
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ TensorDouble fPtr -- sdims
{-# NOINLINE td_new' #-}


-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new_ :: k ~ Nat => Dim (d::[k]) -> IO TensorDouble
td_new_ ds = do
  newPtr <- Gen.constant ds 0.0
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ TensorDouble fPtr -- (SomeDims ds)

td_free_ :: TensorDouble -> IO ()
td_free_ t = finalizeForeignPtr $! getForeign t

td_init :: k ~ Nat => Dim (d::[k]) -> Double -> TensorDouble
td_init ds val = unsafePerformIO $ do
  newPtr <- Gen.constant ds (realToFrac val)
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  withForeignPtr fPtr (Gen.inplaceFill realToFrac val)
  pure $ TensorDouble fPtr -- (SomeDims ds)
{-# NOINLINE td_init #-}

td_transpose :: Word -> Word -> TensorDouble -> TensorDouble
td_transpose dim1 dim2 t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p dim1C dim2C)
  newFPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  -- ds <- someDimsM (Gen.dimList newPtr)
  pure $ TensorDouble newFPtr -- ds
 where
  dim1C, dim2C :: CInt
  dim1C = fromIntegral dim1
  dim2C = fromIntegral dim2
{-# NOINLINE td_transpose #-}

td_trans :: TensorDouble -> TensorDouble
td_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p 1 0)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ TensorDouble newFPtr
{-# NOINLINE td_trans #-}

td_shape :: TensorDouble -> SomeDims
td_shape t = unsafePerformIO $ withForeignPtr (getForeign t) (pure . Gen.getDynamicDim)
{-# NOINLINE td_shape #-}

shapeList :: DynamicTH t => t -> [Int]
shapeList = Dim.dimVals' . shape

rank :: DynamicTH t => t -> Int
rank = Dim.rank' . shape

type instance Gen.HaskReal TensorDouble = Double


withTHPtr :: THForeignRef t => t -> (Ptr (THForeignType t) -> b) -> IO b
withTHPtr t op = withForeignPtr (getForeign t) (pure . op)

withTHPtr2 :: (THForeignRef t0, THForeignRef t1) => t0 -> t1 -> (Ptr (THForeignType t0) -> Ptr (THForeignType t1) -> b) -> IO b
withTHPtr2 t0 t1 op = do
  withForeignPtr (getForeign t0) $ \p0 ->
    withForeignPtr (getForeign t1) $ \p1 ->
      pure (op p0 p1)

withTHPtr3 :: (THForeignRef t0, THForeignRef t1, THForeignRef t2) => t0 -> t1 -> t2 -> (Ptr (THForeignType t0) -> Ptr (THForeignType t1) -> Ptr (THForeignType t2) -> b) -> IO b
withTHPtr3 t0 t1 t2 op = do
  withForeignPtr (getForeign t0) $ \p0 ->
    withForeignPtr (getForeign t1) $ \p1 ->
      withForeignPtr (getForeign t2) $ \p2 ->
        pure (op p0 p1 p2)


withTHPtrM :: THForeignRef t => t -> (Ptr (THForeignType t) -> IO b) -> IO b
withTHPtrM t op = withForeignPtr (getForeign t) op

withTHPtr2M :: (THForeignRef t0, THForeignRef t1) => t0 -> t1 -> (Ptr (THForeignType t0) -> Ptr (THForeignType t1) -> IO b) -> IO b
withTHPtr2M t0 t1 op = do
  withForeignPtr (getForeign t0) $ \p0 ->
    withForeignPtr (getForeign t1) $ \p1 ->
      op p0 p1


withTHPtr3M :: (THForeignRef t0, THForeignRef t1, THForeignRef t2) => t0 -> t1 -> t2 -> (Ptr (THForeignType t0) -> Ptr (THForeignType t1) -> Ptr (THForeignType t2) -> IO b) -> IO b
withTHPtr3M t0 t1 t2 op = do
  withForeignPtr (getForeign t0) $ \p0 ->
    withForeignPtr (getForeign t1) $ \p1 ->
      withForeignPtr (getForeign t2) $ \p2 ->
        op p0 p1 p2


(.:) :: (b -> c) -> (a0 -> a1 -> b) -> a0 -> a1 -> c
(.:) = (.) . (.)

bigflip3 :: (a -> b0 -> b1 -> c) -> b0 -> b1 -> a -> c
bigflip3 fn b0 b1 a = fn a b0 b1

bigflip4 :: (a -> b0 -> b1 -> b2 -> c) -> b0 -> b1 -> b2 -> a -> c
bigflip4 fn b0 b1 b2 a = fn a b0 b1 b2

bigflip5 :: (a -> b0 -> b1 -> b2 -> b3 -> c) -> b0 -> b1 -> b2 -> b3 -> a -> c
bigflip5 fn b0 b1 b2 b3 a = fn a b0 b1 b2 b3

fi :: (Num b, Integral a) => a -> b
fi = fromIntegral

instance TensorClass TensorDouble where
  clearFlag :: TensorDouble -> Int8 -> IO ()
  clearFlag t i = withTHPtrM t (\p -> Gen.c_clearFlag p (fi i))

{-
  tensordata :: TensorDouble -> IO (HaskReal t)
  desc :: TensorDouble -> CTHDescBuff
  expand :: TensorDouble -> TensorDouble -> StorageLong -> IO ()
  expandNd :: TensorDouble -> TensorDouble -> Int32 -> IO ()
-}
  get1d :: TensorDouble -> Int64 -> IO Double
  get1d t x = withTHPtr t (realToFrac . (`Gen.c_get1d` fi x))

  get2d :: TensorDouble -> Int64 -> Int64 -> IO Double
  get2d t x y = withTHPtr t (realToFrac . bigflip3 Gen.c_get2d (fi x) (fi y))

  get3d :: TensorDouble -> Int64 -> Int64 -> Int64 -> IO Double
  get3d t x y z = withTHPtr t (realToFrac . bigflip4 Gen.c_get3d (fi x) (fi y) (fi z))

  get4d :: TensorDouble -> Int64 -> Int64 -> Int64 -> Int64 -> IO Double
  get4d t x y z q = withTHPtr t (realToFrac . bigflip5 Gen.c_get4d (fi x) (fi y) (fi z) (fi q))

  isContiguous :: TensorDouble -> IO Bool
  isContiguous t = withTHPtr t ((1 ==) . Gen.c_isContiguous)

  isSameSizeAs :: TensorDouble -> TensorDouble -> IO Bool
  isSameSizeAs t0 t1 = withTHPtr2 t0 t1 ((1 ==) .: Gen.c_isSameSizeAs)

  isSetTo :: TensorDouble -> TensorDouble -> IO Bool
  isSetTo t0 t1 = withTHPtr2 t0 t1 ((1 ==) .: Gen.c_isSetTo)

  isSize :: TensorDouble -> StorageLong -> IO Bool
  isSize t s = withTHPtr2 t s ((1 ==) .: Gen.c_isSize)

  nDimension :: TensorDouble -> IO Int
  nDimension t = withTHPtr t (fromIntegral . Gen.c_nDimension)

  nElement :: TensorDouble -> IO CPtrdiff
  nElement t = withTHPtr t Gen.c_nElement

  narrow :: TensorDouble -> TensorDouble -> Int32 -> Int64 -> Int64 -> IO ()
  narrow t0 t1 i0 i1 i2 = withTHPtr2M t0 t1 (\p0 p1 -> Gen.c_narrow p0 p1 (fi i0) (fi i1) (fi i2))

{-
  new :: IO TensorDouble
  new = TensorDouble <$> Gen.c_new

  newClone :: TensorDouble -> IO t
  newClone t = withTHPtrM t Gen.c_newClone

  newContiguous :: TensorDouble -> IO t
  newContiguous t = withTHPtrM t Gen.c_newContiguous
  newExpand :: TensorDouble -> StorageLong -> IO t
  newNarrow :: TensorDouble -> Int32 -> Int64 -> Int64 -> IO t
  newSelect :: TensorDouble -> Int32 -> Int64 -> IO t
  newSizeOf :: TensorDouble -> IO StorageLong
  newStrideOf :: TensorDouble -> IO StorageLong
  newTranspose :: TensorDouble -> Int32 -> Int32 -> IO t
  newUnfold :: TensorDouble -> Int32 -> Int64 -> Int64 -> IO t
  newView :: TensorDouble -> StorageLong -> IO t
  newWithSize :: StorageLong -> StorageLong -> IO t
  newWithSize1d :: Int64 -> IO t
  newWithSize2d :: Int64 -> Int64 -> IO t
  newWithSize3d :: Int64 -> Int64 -> Int64 -> IO t
  newWithSize4d :: Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage   :: Storage TensorDouble -> CPtrdiff -> StorageLong -> StorageLong -> IO t
  newWithStorage1d :: Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> IO t
  newWithStorage2d :: Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage3d :: Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage4d :: Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithTensor :: TensorDouble -> IO t
-}

  resize :: TensorDouble -> StorageLong -> StorageLong -> IO ()
  resize t0 s0 s1 = withTHPtr3M t0 s0 s1 Gen.c_resize

  resize1d :: TensorDouble -> Int64 -> IO ()
  resize1d t i0 = withTHPtrM t (\p -> Gen.c_resize1d p (fi i0))

  resize2d :: TensorDouble -> Int64 -> Int64 -> IO ()
  resize2d t i0 i1 = withTHPtrM t (\p -> Gen.c_resize2d p (fi i0) (fi i1))

  resize3d :: TensorDouble -> Int64 -> Int64 -> Int64 -> IO ()
  resize3d t i0 i1 i2 = withTHPtrM t (\p -> Gen.c_resize3d p (fi i0) (fi i1) (fi i2))

  resize4d :: TensorDouble -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d t i0 i1 i2 i3 = withTHPtrM t (\p -> Gen.c_resize4d p (fi i0) (fi i1) (fi i2) (fi i3))

  resize5d :: TensorDouble -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d t i0 i1 i2 i3 i4 = withTHPtrM t (\p -> Gen.c_resize5d p (fi i0) (fi i1) (fi i2) (fi i3) (fi i4))

  resizeAs :: TensorDouble -> TensorDouble -> IO ()
  resizeAs t0 t1 = withTHPtr2M t0 t1 Gen.c_resizeAs

  resizeNd :: TensorDouble -> Int32 -> Int64 -> Int64 -> IO ()
  resizeNd t i0 i1 i2 = withTHPtrM t (\p -> Gen.c_resize3d p (fi i0) (fi i1) (fi i2))

  retain :: TensorDouble -> IO ()
  retain t = withTHPtrM t Gen.c_retain

  select :: TensorDouble -> TensorDouble -> Int32 -> Int64 -> IO ()
  select t0 t1 i0 i1 = withTHPtr2M t0 t1 (\p0 p1 -> Gen.c_select p0 p1 (fi i0) (fi i1))

  set :: TensorDouble -> TensorDouble -> IO ()
  set t0 t1 = withTHPtr2M t0 t1 Gen.c_set

  set1d :: TensorDouble -> Int64 -> Double -> IO ()
  set1d t i0 v = withTHPtrM t (\p -> Gen.c_set1d p (fi i0) (realToFrac v))

  set2d :: TensorDouble -> Int64 -> Int64 -> Double -> IO ()
  set2d t i0 i1 v = withTHPtrM t (\p -> Gen.c_set2d p (fi i0) (fi i1) (realToFrac v))

  set3d :: TensorDouble -> Int64 -> Int64 -> Int64 -> Double -> IO ()
  set3d t i0 i1 i2 v = withTHPtrM t (\p -> Gen.c_set3d p (fi i0) (fi i1) (fi i2) (realToFrac v))

  set4d :: TensorDouble -> Int64 -> Int64 -> Int64 -> Int64 -> Double -> IO ()
  set4d t i0 i1 i2 i3 v = withTHPtrM t (\p -> Gen.c_set4d p (fi i0) (fi i1) (fi i2) (fi i3) (realToFrac v))

  setFlag :: TensorDouble -> Int8 -> IO ()
  setFlag t i = withTHPtrM t (`Gen.c_setFlag` fi i)

{-
  setStorage :: TensorDouble -> Storage TensorDouble -> CPtrdiff -> StorageLong -> StorageLong -> IO ()
  setStorage t s pd sl0 sl1 = withTHPtr3M
  setStorage1d_ :: TensorDouble -> Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> IO ()
  setStorage2d_ :: TensorDouble -> Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage3d_ :: TensorDouble -> Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage4d_ :: TensorDouble -> Storage TensorDouble -> CPtrdiff -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorageNd_ :: TensorDouble -> Storage TensorDouble -> CPtrdiff -> Int32 -> Int64 -> Int64 -> IO ()
-}

  size :: TensorDouble -> Int32 -> IO Int64
  size t i = withTHPtr t (\p -> fromIntegral $ Gen.c_size p (fi i))

  sizeDesc :: TensorDouble -> IO Gen.CTHDescBuff
  sizeDesc t = withTHPtr t Gen.c_sizeDesc

  squeeze :: TensorDouble -> TensorDouble -> IO ()
  squeeze t0 t1 = withTHPtr2M t0 t1 Gen.c_squeeze

  squeeze1d :: TensorDouble -> TensorDouble -> Int32 -> IO ()
  squeeze1d t0 t1 i = withTHPtr2M t0 t1 (\p0 p1 -> Gen.c_squeeze1d p0 p1 (fi i))

{-
  storage :: TensorDouble -> IO (Gen.Storage TensorDouble)
  storage t = withTHPtr t Gen.c_storage
  -}

  storageOffset :: TensorDouble -> IO CPtrdiff
  storageOffset t = withTHPtr t Gen.c_storageOffset

  stride :: TensorDouble -> Int32 -> IO Int64
  stride t a = withTHPtr t (\p -> fromIntegral $ Gen.c_stride p (fi a))

  transpose :: TensorDouble -> TensorDouble -> Int32 -> Int32 -> IO ()
  transpose t0 t1 i0 i1 = withTHPtr2M t0 t1 (\p0 p1 -> Gen.c_transpose p0 p1 (fi i0) (fi i1))

  unfold :: TensorDouble -> TensorDouble -> Int32 -> Int64 -> Int64 -> IO ()
  unfold t0 t1 i0 i1 i2 = withTHPtr2M t0 t1 (\p0 p1 -> Gen.c_unfold p0 p1 (fi i0) (fi i1) (fi i2))

  unsqueeze1d :: TensorDouble -> TensorDouble -> Int32 -> IO ()
  unsqueeze1d t0 t1 i = withTHPtr2M t0 t1 (\p0 p1 -> Gen.c_unsqueeze1d p0 p1 (fi i))
