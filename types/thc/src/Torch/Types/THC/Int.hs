{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Int where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.THC

type CTensor = CIntTensor
type CStorage = CIntStorage

type CReal = CInt
type CAccReal = CLong
type HsReal = Int32
type HsAccReal = Int64

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

type Storage = IntStorage
storage = intCStorage
asStorage = IntStorage

type DynTensor = IntDynTensor
tensor = intCTensor
asDyn = IntDynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor


