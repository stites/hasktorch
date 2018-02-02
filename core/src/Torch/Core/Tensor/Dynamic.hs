{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Core.Tensor.Dynamic
  ( Tensor(..)
  , THType
  , THValue(..)
  , printTensor
  , new
  , new'
  , newWithTensor
  , fromList1d
  , fromListNd
  , fromListNd'
  , resize
  , resize'
  , get
  , get'
  , free
  , constant
  , constant'
  , _transpose
  , newTranspose
  , newTranspose'
  , _newTranspose
  ) where

import GHC.ForeignPtr (ForeignPtr)
import Foreign (withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits
import Data.Proxy (Proxy(..))
import Torch.Raw.Tensor.Generic hiding (constant, constant')
import Torch.Core.Tensor.Dim
import Control.Exception.Safe (throwString)
import qualified Torch.Core.Tensor.Dim as Dim

import qualified Torch.Raw.Tensor.Generic as GenRaw

newtype Tensor a = Tensor { tensor :: ForeignPtr (THType a) }
  deriving (Show, Eq)

type family THType a

type instance THType CChar   = CTHByteTensor
type instance THType CDouble = CTHDoubleTensor
type instance THType CFloat  = CTHFloatTensor
type instance THType CInt    = CTHIntTensor
type instance THType CLong   = CTHLongTensor
type instance THType CShort  = CTHShortTensor

-- type instance THType Double = CTHDoubleTensor
-- type instance THType Float  = CTHFloatTensor
-- type instance THType Int    = CTHIntTensor

class (Num (HaskReal (THType a)), Num a) => THValue a where
  ccast :: a -> HaskReal (THType a)
  hcast :: HaskReal (THType a) -> a

instance THValue CFloat where
  ccast = realToFrac
  hcast = realToFrac

instance THValue CDouble where
  ccast = realToFrac
  hcast = realToFrac

type TensorConstraints a =
  ( THValue a
  , THTensor (THType a)
  , THTensorMath (THType a)
  )

-- ========================================================================= --

printTensor
  :: (Show (HaskReal (THType a)), THTensor (THType a))
  => Tensor a
  -> IO ()
printTensor t = withForeignPtr (tensor t) GenRaw.dispRaw

-- | Create a new (double) tensor of specified dimensions and fill it with 0
new :: TensorConstraints a => Dim (d::[Nat]) -> IO (Tensor a)
new = (`constant` 0)

-- | Opaque version of 'new'
new' :: TensorConstraints a => SomeDims -> IO (Tensor a)
new' (SomeDims dims) = dims `constant` 0

newWithTensor :: TensorConstraints a => Tensor a -> IO (Tensor a)
newWithTensor t = do
  newPtr <- withForeignPtr (tensor t) GenRaw.c_newWithTensor
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ Tensor newFPtr

-- | Initialize a 1D tensor from a list
fromList1d
  :: forall a . TensorConstraints a
  => [a]
  -> IO (Tensor a)
fromList1d l = do
  sdims <- someDimsM [length l]
  res <- new' sdims
  mapM_ (mutTensor (tensor res)) (zip [0..length l - 1] l)
  pure res
 where
  mutTensor :: ForeignPtr (THType a) -> (Int, a) -> IO ()
  mutTensor t (idx, value) = withForeignPtr t $ \tp ->
    GenRaw.c_set1d tp (fromIntegral idx) (ccast value)

-- | Initialize a tensor of arbitrary dimension from a list
fromListNd :: TensorConstraints a => Dim (d::[Nat]) -> [a] -> IO (Tensor a)
fromListNd d l =
  if fromIntegral (product (Dim.dimVals d)) == length l
  then fromList1d l >>= (`resize` d)
  else throwString "Incorrect tensor dimensions specified."

fromListNd' :: TensorConstraints a => SomeDims -> [a] -> IO (Tensor a)
fromListNd' (SomeDims d) l = fromListNd d l

-- | Copy contents of tensor into a new one of specified size
resize
  :: forall a d . TensorConstraints a
  => Tensor a
  -> Dim (d::[Nat])
  -> IO (Tensor a)
resize t d = do
  res :: Tensor a <- new d
  p <- withForeignPtr (tensor t) GenRaw.c_newClone
  fp <- newForeignPtr GenRaw.p_free p
  withForeignPtr fp (withForeignPtr (tensor res) . GenRaw.c_resizeAs)
  pure (Tensor fp)

resize' :: TensorConstraints a => Tensor a -> SomeDims -> IO (Tensor a)
resize' t (SomeDims d) = resize t d

get :: TensorConstraints a => Dim (d::[Nat]) -> Tensor a -> IO a
get loc t = withForeignPtr (tensor t) (pure . hcast . (`GenRaw.genericGet` loc))

get' :: TensorConstraints a => SomeDims -> Tensor a -> IO a
get' (SomeDims d) t = get d t

free :: Tensor a -> IO ()
free t = finalizeForeignPtr $! tensor t

constant :: TensorConstraints a => Dim (d::[Nat]) -> a -> IO (Tensor a)
constant ds val = do
  newPtr <- GenRaw.constant ds (ccast val)
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  withForeignPtr fPtr (GenRaw.inplaceFill ccast val)
  pure $ Tensor fPtr

constant' :: TensorConstraints a => SomeDims -> a -> IO (Tensor a)
constant' (SomeDims ds) val = constant ds val

-- | an inplace transpose (FIXME: is @GenRaw.c_transpose p p d1C d2C@ correct?)
_transpose :: TensorConstraints a => Word -> Word -> Tensor a -> IO ()
_transpose d1 d2 t = withForeignPtr (tensor t) (\p -> GenRaw.c_transpose p p d1C d2C)
 where
  d1C, d2C :: CInt
  d1C = fromIntegral d1
  d2C = fromIntegral d2

newTranspose
  :: forall d0 d1 a . (TensorConstraints a, KnownNat d0, KnownNat d1)
  => Dim '[d0, d1] -> Tensor a -> IO (Tensor a)
newTranspose _ = _newTranspose
  (fromIntegral (natVal (Proxy :: Proxy d0)))
  (fromIntegral (natVal (Proxy :: Proxy d1)))

newTranspose' :: TensorConstraints a => SomeDims -> Tensor a -> IO (Tensor a)
newTranspose' s t =
  case dimVals' s of
    x:y:_ -> _newTranspose (fromIntegral x) (fromIntegral y) t
    _     -> throwString "transpose bindings only currently work for two dimensions"

_newTranspose :: TensorConstraints a => Word -> Word -> Tensor a -> IO (Tensor a)
_newTranspose d1 d2 t = do
  newPtr <- withForeignPtr (tensor t) (\p -> GenRaw.c_newTranspose p d1C d2C)
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ Tensor newFPtr
 where
  d1C, d2C :: CInt
  d1C = fromIntegral d1
  d2C = fromIntegral d2


