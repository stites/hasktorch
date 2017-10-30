{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}

module Main where

-- experimental AD implementation

import Control.Exception.Base (assert)
import Data.Monoid ((<>))
import Data.Maybe (fromJust)
import Foreign.C.Types
import Foreign.Ptr

-- import GHC.TypeLits (Nat, KnownNat, natVal)

import StaticTensorDouble
import TensorDouble
--import TensorDoubleMath (sigmoid, (!*), addmv)
-- import TensorDoubleRandom
import StaticTensorDoubleRandom
import Random
import TensorTypes
import TensorUtils

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits
import GHC.TypeLits.Witnesses

{- Statically Typed Implementation -}

type SW = StaticWeights
type SN = StaticNetwork

data StaticWeights (i :: Nat) (o :: Nat) = SW {
  biases :: TDS '[o],
  nodes :: TDS '[i, o]
  } deriving (Show)

mkW :: (KnownNat i, KnownNat o) => SW i o
mkW = SW biases nodes
  where (biases, nodes) = (tds_new, tds_new)

data StaticNetwork :: Nat -> [Nat] -> Nat -> * where
  O :: SW i o -> SN i '[] o
  (:~) :: (KnownNat h) => SW i h -> SN h hs o -> SN i (h ': hs) o

infixr 5 :~

dispW :: (KnownNat o, KnownNat i) => StaticWeights i o -> IO ()
dispW w = do
  putStrLn "Biases:"
  dispS (biases w)
  putStrLn "Weights:"
  dispS (nodes w)

-- dispN (O w) = dispW w
-- dispN (w :~ n') = putStrLn "Current Layer ::::\n" >> dispW w >> dispN n'

-- dispN :: forall i hs o. (KnownNat i, SingI hs, KnownNat o) => IO (SN i hs o)
-- dispN = go sing
--   where go :: forall h hs'. KnownNat h => Sing hs' -> IO (SN h hs' o)
--         go = \case
--           SNil            ->     O <$> randomWeights
--           SNat `SCons` ss -> (:~) <$> randomWeights <*> go ss


randomWeights :: (KnownNat i, KnownNat o) => IO (SW i o)
randomWeights = do
  gen <- newRNG
  b <- tds_uniform (biases storeResult) gen (-1.0) (1.0)
  w <- tds_uniform (nodes storeResult) gen (-1.0) (1.0)
  pure SW { biases = b, nodes = w }
  where
    storeResult = mkW

randomNet :: forall i hs o. (KnownNat i, SingI hs, KnownNat o) => IO (SN i hs o)
randomNet = go sing
  where go :: forall h hs'. KnownNat h => Sing hs' -> IO (SN h hs' o)
        go = \case
          SNil            ->     O <$> randomWeights
          SNat `SCons` ss -> (:~) <$> randomWeights <*> go ss

-- runLayer :: SW i o -> (TDS d) -> TensorDouble
-- runLayer sw v = addmv 1.0 wB 1.0 wN v

-- runNet :: SN i h o -> TensorDouble -> TensorDouble
-- runNet (O w) v = sigmoid (runLayer w v)
-- runNet (w :~ n') v = let v' = sigmoid (runLayer w v) in runNet n' v'

-- train :: Double
--       -> TensorDouble
--       -> TensorDouble
--       -> SN i h o
--       -> SN i h o
-- train rate x0 target = fst . go x0
--   where go x (O w@(SW wB wN)) = undefined

ih = mkW :: StaticWeights 10 7
hh = mkW :: StaticWeights  7 4
ho = mkW :: StaticWeights  4 2

net1 = O ho :: SN 4 '[] 2
net2 = hh :~ O ho :: SN 7 '[4] 2
net3 = ih :~ hh :~ O ho :: SN 10 '[7,4] 2

main = do
  (foo  :: SN 4 '[] 2) <- randomNet
  (bar :: SN 4 '[3, 2] 2) <- randomNet
  putStrLn "Done"
