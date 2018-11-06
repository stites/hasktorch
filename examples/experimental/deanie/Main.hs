-------------------------------------------------------------------------------
-- |
-- Small prototype of deanie from Jared Tobin's post.
-- https://jtobin.io/simple-probabilistic-programming
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE LambdaCase #-}
module Main where

import Control.Monad
import Control.Monad.Free
import qualified System.Random.MWC.Probability as MWC

main = putStrLn "hello"


data ModelF r =
    BernoulliF Double (Bool -> r)
  | BetaF Double Double (Double -> r)
  deriving (Functor)

type Model = Free ModelF

bernoulli :: Double -> Model Bool
bernoulli p = liftF $ BernoulliF p id

beta :: Double -> Double -> Model Double
beta a b = liftF (BetaF a b id)

uniform :: Model Double
uniform = beta 1 1

binomial :: Int -> Double -> Model Int
binomial n p = count <$> coins
  where
    count :: [Bool] -> Int
    count = length . filter id

    coins :: Model [Bool]
    coins = replicateM n (bernoulli p)

betaBinomial :: Int -> Double -> Double -> Model Int
betaBinomial n a b = beta a b >>= binomial n

toSampler :: Model a -> MWC.Prob IO a
toSampler = iterM $ \case
  BernoulliF p f -> MWC.bernoulli p >>= f
  BetaF a b f -> MWC.beta a b >>= f

simulate :: forall a . Model a -> IO a
simulate model = MWC.withSystemRandom $ \g ->
  MWC.asGenIO sampling g
  where
    sampling :: MWC.GenIO -> IO a
    sampling g = (MWC.sample ((toSampler model) :: MWC.Prob IO a ) :: MWC.GenIO -> IO a) g

rejectionInverter
  :: forall m a b . (Monad m, Eq b)
  => m a              -- ^ model which represents the proposal of hyperparameters
  -> (a -> m b)       -- ^ model which takes in the proposed hyperparams and outputs inference
  -> [b]              -- ^ data to fit
  -> m a              -- ^ an inverted model
rejectionInverter = rejectionInverterBy id

rejectionInverterBy
  :: forall m a b c . (Monad m, Eq c)
  => ([b] -> c)       -- ^ assisting function called on sample to preprocess rejection
  -> m a              -- ^ model which represents the proposal of hyperparameters
  -> (a -> m b)       -- ^ model which takes in the proposed hyperparams and outputs inference
  -> [b]              -- ^ data to fit
  -> m a              -- ^ an inverted model
rejectionInverterBy assister proposal model observed = go
  where
    observed' :: c
    observed' = assister observed

    go :: m a
    go = do
      params <- proposal
      generated <- replicateM (length observed) (model params)
      if assister generated == observed'
      then pure params
      else go



