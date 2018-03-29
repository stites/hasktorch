-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Core.Tensor.Static
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Tensors with dimensional phantom types.
--
-- Be aware of https://ghc.haskell.org/trac/ghc/wiki/Roles but since Dynamic
-- and static tensors are the same (minus the dimension operators in the
-- phantom type), I (@stites) don't think we need to be too concerned.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wno-orphans #-}
module Torch.Core.Tensor.Static
  ( ByteTensor
  , ShortTensor
  , IntTensor
  , LongTensor
  , FloatTensor
  , DoubleTensor
  -- helper constraints
  , StaticConstraint
  , StaticConstraint2
  , StaticConstraint3
  , NumReals

  -- experimental helper function (potentially delete)
  , withInplace

  -- generalized static functions
  , fromList
  , resizeAs
  , isSameSizeAs

  -- specialized static functions
  , Torch.Core.Tensor.Static.fromList1d
  , newTranspose2d
  , expand2d
  , getElem2d
  , setElem2d
  , new

  -- reexports
  , IsStatic(..)
  , module X
  ) where

import THTypes
import Foreign ()
import Torch.Class.C.Internal
import Torch.Core.Tensor.Dim
import Data.Proxy
import Data.List (genericLength)
import Control.Exception.Safe
import GHC.TypeLits
import GHC.Natural
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num

import qualified Torch.Core.Tensor.Dynamic as Dynamic

import qualified Torch.Core.Storage as Storage
import qualified Torch.Core.LongStorage as L

import Torch.Class.C.Tensor.Static (IsStatic(..))
import qualified Torch.Core.Tensor.Dynamic as Class (IsTensor)

-- import qualified Torch.Core.ByteTensor.Static as B
-- import qualified Torch.Core.ShortTensor.Static as S
-- import qualified Torch.Core.IntTensor.Static as I
-- import qualified Torch.Core.LongTensor.Static as L
-- import qualified Torch.Core.FloatTensor.Static as F
-- import qualified Torch.Core.DoubleTensor.Static as D
--
-- ========================================================================= --
-- re-export all SigTypes so that Aliases propogate
import qualified THByteTypes   as B
import qualified THShortTypes  as S
import qualified THIntTypes    as I
import qualified THLongTypes   as L
import qualified THFloatTypes  as F
import qualified THDoubleTypes as D


-- ========================================================================= --
-- re-export all IsTensor functions -- except for ones not specialized for static
import Torch.Class.C.IsTensor as X hiding (resizeAs, isSameSizeAs, new, fromList1d)
import qualified Torch.Class.C.IsTensor as IsTensor (fromList1d)
import Torch.Core.ByteTensor.Static.IsTensor ()
import Torch.Core.ShortTensor.Static.IsTensor ()
import Torch.Core.IntTensor.Static.IsTensor ()
import Torch.Core.LongTensor.Static.IsTensor ()
import Torch.Core.FloatTensor.Static.IsTensor ()
import Torch.Core.DoubleTensor.Static.IsTensor ()

-- ========================================================================= --
-- re-export all Random functions >> import cycle due to @multivariate_normal@
-- import Torch.Core.Tensor.Static.Random as X

-- ========================================================================= --
-- re-export all TensorCopy functions (for dynamic copies)
import Torch.Class.C.Tensor.Copy as X

import Torch.Core.ByteTensor.Static.Copy   ()
import Torch.Core.ShortTensor.Static.Copy  ()
import Torch.Core.IntTensor.Static.Copy    ()
import Torch.Core.LongTensor.Static.Copy   ()
import Torch.Core.FloatTensor.Static.Copy  ()
import Torch.Core.DoubleTensor.Static.Copy ()

-------------------------------------------------------------------------------

type ByteTensor   = B.Tensor
type ShortTensor  = S.Tensor
type IntTensor    = I.Tensor
type LongTensor   = L.Tensor
type FloatTensor  = F.Tensor
type DoubleTensor = D.Tensor
type LongStorage  = L.Storage


-- -- These might require changing
-- instance Dynamic.TensorConv (ByteTensor   (d::[Nat]))
-- instance Dynamic.TensorConv (ShortTensor  (d::[Nat]))
-- instance Dynamic.TensorConv (IntTensor    (d::[Nat]))
-- instance Dynamic.TensorConv (LongTensor   (d::[Nat]))
-- instance Dynamic.TensorConv (FloatTensor  (d::[Nat]))
-- instance Dynamic.TensorConv (DoubleTensor (d::[Nat]))

-- ========================================================================= --

-- Constraints that will be garunteed for every static tensor. Only 'Dynamic.IsTensor'
-- because we require downcasting for a lot of operations
type NumReals t d =
  ( HsReal (t d) ~ HsReal (AsDynamic (t d))
  , HsAccReal (t d) ~ HsAccReal (AsDynamic (t d))
  , Num (HsReal (t d))
  , Num (HsAccReal (t d))
  , Num (HsReal (AsDynamic (t d)))
  , Num (HsAccReal (AsDynamic (t d)))
  )

type StaticConstraint t d =
  ( IsStatic (t d)
  , HsReal (t d) ~ HsReal (AsDynamic (t d))
  , HsAccReal (t d) ~ HsAccReal (AsDynamic (t d))
  , Dynamic.IsTensor (AsDynamic (t d))
  , Num (HsReal (t d))
  -- , Num (HsAccReal (t d))
  -- , Num (HsReal (AsDynamic (t d)))
  -- , Num (HsAccReal (AsDynamic (t d)))
  , Dimensions d
  )

-- Constraints used on two static tensors. Essentially that both static tensors have
-- the same internal tensor representations.
type StaticConstraint2 t d d' =
  ( StaticConstraint t d
  , StaticConstraint t d'
  , AsDynamic (t d) ~ AsDynamic (t d')
  , HsReal (t d) ~ HsReal (t d')
  , HsAccReal (AsDynamic (t d)) ~ HsAccReal (AsDynamic (t d'))
  , HsReal (AsDynamic (t d)) ~ HsReal (AsDynamic (t d'))
  )
type StaticConstraint3 t d d' d'' =
  ( StaticConstraint2 t d  d'
  , StaticConstraint2 t d' d''
  , StaticConstraint2 t d  d''
  )


-------------------------------------------------------------------------------

withInplace :: forall t d . (StaticConstraint t d) => (AsDynamic (t d) -> IO ()) -> IO (t d)
withInplace op = do
  res <- Dynamic.new (dim :: Dim d)
  op res
  pure (asStatic res)

-------------------------------------------------------------------------------

new :: forall t d . (StaticConstraint t d) => IO (t d)
new = asStatic <$> Dynamic.new (dim :: Dim d)

-- | 'Dynamic.isSameSizeAs' without calling down through the FFI since we have
-- this information
isSameSizeAs
  :: forall t d d' . (IsStatic (t d), IsStatic (t d'), Dimensions d', Dimensions d)
  => t d -> t d' -> Bool
isSameSizeAs _ _ = dimVals (dim :: Dim d) == dimVals (dim :: Dim d')

-- | pure 'Dynamic.resizeAs'
resizeAs
  :: forall t d d' . StaticConstraint2 t d d'
  => (Dimensions d', Dimensions d)
  => t d -> IO (t d')
resizeAs src = do
  dummy :: AsDynamic (t d') <- Dynamic.new (dim :: Dim d')
  asStatic <$> Dynamic.resizeAs (asDynamic src) dummy


-- TODO: try to force strict evaluation to avoid potential FFI + IO + mutation bugs.
-- however `go` never executes with deepseq: else unsafePerformIO $ pure (deepseq go result)
fromList1d
  :: forall t n . (KnownNatDim n, StaticConstraint t '[n])
  => [HsReal (t '[n])] -> IO (t '[n])
fromList1d l
  | fromIntegral (natVal (Proxy :: Proxy n)) /= length l =
    throwString "List length does not match tensor dimensions"
  | otherwise = do
    asStatic <$> IsTensor.fromList1d l
    -- res :: t '[n] <- Dynamic.new (dim :: Dim '[n])
    -- mapM_  (upd res) (zip [0..length l - 1] l)
    -- pure res
  -- where
    -- upd :: t '[n] -> (Int, HsReal (t '[n])) -> IO ()
    -- upd t (idx, v) = someDimsM [idx] >>= \sd -> Dynamic.setDim'_ t sd v

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME: There might be a faster way to do this with newWithSize
fromList
  :: forall t d
   . (KnownNatDim (Product d), Dimensions d)
  => (StaticConstraint2 t d '[Product d])
  => [HsReal (t d)] -> IO (t d)
fromList l = do
  oneD :: t '[Product d] <- fromList1d l
  asStatic <$> resizeDim (asDynamic oneD) (dim :: Dim d)

newTranspose2d
  :: forall t r c . (KnownNat2 r c, StaticConstraint2 t '[r, c] '[c, r])
  => t '[r, c] -> IO (t '[c, r])
newTranspose2d t =
  asStatic <$> Dynamic.newTranspose (asDynamic t) 1 0

-- | Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
expand2d
  :: forall t d1 d2 . (KnownNatDim2 d1 d2)
  => StaticConstraint2 t '[d2, d1] '[d1]
  => Dynamic.TensorMath (AsDynamic (t '[d1])) -- for 'Dynamic.constant' which uses 'Torch.Class.C.Tensor.Math.fill'
  => t '[d1] -> IO (t '[d2, d1])
expand2d t = do
  res :: AsDynamic (t '[d2, d1]) <- Dynamic.constant (dim :: Dim '[d2, d1]) 0
  s :: LongStorage <- Storage.newWithSize2 s2 s1
  Dynamic.expand_ res (asDynamic t) s
  pure (asStatic res)
  where
    s1, s2 :: Integer
    s1 = natVal (Proxy :: Proxy d1)
    s2 = natVal (Proxy :: Proxy d2)

getElem2d
  :: forall t n m . (KnownNatDim2 n m)
  => StaticConstraint t '[n, m]
  => t '[n, m] -> Natural -> Natural -> IO (HsReal (t '[n, m]))
getElem2d t r c
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = someDimsM [r, c] >>= getDim' (asDynamic t)

setElem2d
  :: forall t n m . (KnownNatDim2 n m)
  => StaticConstraint t '[n, m]
  => t '[n, m] -> Natural -> Natural -> HsReal (t '[n, m]) -> IO ()
setElem2d t r c v
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = someDimsM [r, c] >>= \d -> setDim'_ (asDynamic t) d v

