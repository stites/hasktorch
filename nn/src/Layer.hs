{-# LANGUAGE DataKinds, GADTs, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module Layer where

import StaticTensorDouble
import StaticTensorDoubleMath
import StaticTensorDoubleRandom
import TensorDouble
import Random
import TensorTypes
import TensorUtils

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits

-- datakinds lifts constructors to type level
data Shape = L1 Nat | L2 Nat Nat | L3 Nat Nat Nat | L4 Nat Nat Nat Nat

-- Define shape types
data S (shapeType :: Shape) where
  S1L :: (KnownNat o)                         => TDS '[o]          -> S ('L1 o)
  S2L :: (KnownNat r, KnownNat c)             => TDS '[r, c]       -> S ('L2 r c)
  S3L :: (KnownNat x, KnownNat y, KnownNat z) => TDS '[x, y, z]    -> S ('L3 x y z)
  S4L :: (KnownNat a, KnownNat b,
          KnownNat c, KnownNat d)             => TDS '[a, b, c, d] -> S ('L4 a b c d)

-- singleton machinery to apply a tensor transformation function to a layer

-- l1apply :: (KnownNat o) => (TDS '[o] -> TDS '[o]) -> S ('L1 o) -> S ('L1 o)
l1apply :: (KnownNat o) => (TDS '[o] -> TDS '[o]) -> S ('L1 o) -> S ('L1 o)
l1apply f (S1L x) = S1L (f x)

l2apply :: (KnownNat r, KnownNat c) =>
  (TDS '[r, c] -> TDS '[r, c]) -> S ('L2 r c) -> S ('L2 r c)
l2apply f (S2L x) = S2L (f x)

l3apply :: (KnownNat x, KnownNat y, KnownNat z) =>
  (TDS '[x, y, z] -> TDS '[x, y, z]) -> S ('L3 x y z) -> S ('L3 x y z)
l3apply f (S3L x) = S3L (f x)

l4apply :: (KnownNat a, KnownNat b, KnownNat c, KnownNat d) =>
  (TDS '[a, b, c, d] -> TDS '[a, b, c, d]) -> S ('L4 a b c d) -> S ('L4 a b c d)
l4apply f (S4L x) = S4L (f x)

test :: forall l2 ltype . Sing l2 -> S ltype -> S ltype
test = undefined

lapply :: forall d l . (SingI l) => (TDS d -> TDS d) -> S l -> S l
lapply f layer = go (sing :: Sing l) layer
  where
    go :: forall l . SingI l => (Sing l) -> S l -> S l
    -- TODO need proofs to specialize dimensions used by f
    go L1Sing (S1L x) = l1apply undefined (S1L x)
    go L2Sing (S2L x) = l2apply undefined (S2L x)
    go L3Sing (S3L x) = l3apply undefined (S3L x)
    go L4Sing (S4L x) = l4apply undefined (S4L x)

-- define singleton constructors: take type inputs, return value of type Sing [Shape]
data instance Sing (shapeType :: Shape) where
  L1Sing :: KnownNat a                                       => Sing ('L1 a)
  L2Sing :: (KnownNat a, KnownNat b)                         => Sing ('L2 a b)
  L3Sing :: (KnownNat a, KnownNat b, KnownNat c)             => Sing ('L3 a b c)
  L4Sing :: (KnownNat a, KnownNat b, KnownNat c, KnownNat d) => Sing ('L4 a b c d)
  
-- derive value-level singleton from layer shape types (L1...L4) via sing
instance KnownNat a => SingI ('L1 a) where
  sing = L1Sing
instance (KnownNat a, KnownNat b) => SingI ('L2 a b) where
  sing = L2Sing
instance (KnownNat a, KnownNat b, KnownNat c) => SingI ('L3 a b c) where
  sing = L3Sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat d) => SingI ('L4 a b c d) where
  sing = L4Sing

-- typeclass 1: layers can be updated and randomly instantiated
class UpdateLayer layer where
  type Gradient layer :: *
  updateLayer :: LearningParameters -> layer -> Gradient layer -> layer
  createRandom :: IO layer

-- typeclass 2: layers can be run forward and backward propagated
class UpdateLayer x => Layer x (i :: Shape) (o :: Shape) where
  type Tape x i o :: *
  runForwards :: x -> S i -> (Tape x i o, S o)
  runBackwards :: x -> Tape x i o -> S o -> (Gradient x, S i)

-- learning parameter values
data LearningParameters = LearningParameters {
  lp_rate :: Double
  , lp_momentum :: Double
  , lp_regularizer :: Double
  } deriving (Eq, Show)

-- a network is a list of layers
data Network :: [*] -> [Shape] -> * where
  NNil :: SingI i => Network '[] '[i]
  NCons :: (SingI i, SingI h, Layer x i h) =>
    x -> (Network xs (h : hs)) -> Network (x : xs) (i : h : hs)

--
-- Layer definiions
--

{- Logit -}

data Logit = Logit
  deriving Show

instance UpdateLayer Logit where
  type Gradient Logit = ()
  updateLayer _ _ _ = Logit
  createRandom = return Logit

instance (a ~ b, SingI a) => Layer Logit a b where
  type Tape Logit a b = S a
  runForwards _ a = (a, undefined)
  runBackwards _ a g = ((), undefined)
  -- runForwards _ a = (a, logistic a)
  -- runBackwards _ a g = ((), logistic' a * g)

-- logistic :: Floating a => a -> a
-- logistic x = 1 / (1 + exp (-x))
-- logistic' x = (logistic x) * (1 - (logistic x))

{- Tanh -}

data Tanh = Tanh
  deriving Show

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  updateLayer _ _ _ = Tanh
  createRandom = return Tanh

instance (a ~ b, SingI a) => Layer Tanh a b where
  type Tape Tanh a b = S a
  runForwards _ a = (a, undefined)
  runBackwards _ a g = ((), undefined)
  -- runForwards _ a = (a, tanh a)
  -- runBackwards _ a g = ((), tanh' a * g)
