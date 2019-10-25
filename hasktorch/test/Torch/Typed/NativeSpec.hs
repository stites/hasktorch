{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NativeSpec
  ( Torch.Typed.NativeSpec.spec
  )
where

import           Prelude                 hiding ( all
                                                , any
                                                , sin
                                                , sinh
                                                , cos
                                                , cosh
                                                , tan
                                                , tanh
                                                , asin
                                                , asinh
                                                , acos
                                                , acosh
                                                , atan
                                                , atanh
                                                , abs
                                                , max
                                                , min
                                                , exp
                                                , log
                                                , round
                                                , floor
                                                , sqrt
                                                )
import           Control.Exception.Safe
import           Foreign.Storable
import           Data.HList
import           Data.Proxy
import           Data.Reflection
import           GHC.TypeLits

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data UnaryAllDTypesSpec =
    SignSpec
  | OnesLikeSpec
  | ZerosLikeSpec

data UnaryStandardDTypesSpec =
    AbsSpec

data UnaryStandardFloatingPointDTypesSpec =
    FracSpec
  | CeilSpec
  | FloorSpec
  | RoundSpec
  | TruncSpec
  | ErfSpec
  | ErfcSpec
  | ErfinvSpec
  | ExpSpec
  | Expm1Spec
  | LogSpec
  | Log1pSpec
  | Log2Spec
  | Log10Spec
  | LgammaSpec
  | DigammaSpec
  | ReluSpec
  | SeluSpec
  | GeluSpec
  | SigmoidSpec
  | LogSigmoidSpec
  | SinSpec
  | SinhSpec
  | AsinSpec
  | CosSpec
  | CoshSpec
  | AcosSpec
  | TanSpec
  | TanhSpec
  | AtanSpec
  | SqrtSpec
  | RsqrtSpec
  | RandLikeSpec
  | RandnLikeSpec

-- data UnaryAllFloatingPointDTypesSpec =

instance ( TensorOptions shape dtype device
         , DTypeIsNotHalf dtype
         , DTypeIsNotBool dtype
         )
  => Apply
       UnaryStandardDTypesSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply AbsSpec _ _ = do
    let t = abs ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t

instance (TensorOptions shape dtype device)
  => Apply
       UnaryAllDTypesSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply SignSpec _ _ = do
    let t = sign ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply OnesLikeSpec _ _ = do
    let t = onesLike ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ZerosLikeSpec _ _ = do
    let t = zerosLike ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t

instance ( TensorOptions shape dtype device
         , IsFloatingPoint dtype
         , DTypeIsNotHalf dtype
         )
  => Apply
       UnaryStandardFloatingPointDTypesSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply FracSpec _ _ = do
    let t = frac ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply CeilSpec _ _ = do
    let t = ceil ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply FloorSpec _ _ = do
    let t = floor ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply TruncSpec _ _ = do
    let t = trunc ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ErfSpec _ _ = do
    let t = erf ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ErfcSpec _ _ = do
    let t = erfc ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ErfinvSpec _ _ = do
    let t = erfinv ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ExpSpec _ _ = do
    let t = exp ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Expm1Spec _ _ = do
    let t = expm1 ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply LogSpec _ _ = do
    let t = log ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Log1pSpec _ _ = do
    let t = log1p ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Log2Spec _ _ = do
    let t = log2 ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply Log10Spec _ _ = do
    let t = log10 ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply LgammaSpec _ _ = do
    let t = lgamma ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply DigammaSpec _ _ = do
    let t = digamma ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply ReluSpec _ _ = do
    let t = relu ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply SeluSpec _ _ = do
    let t = selu ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply GeluSpec _ _ = do
    let t = gelu ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply SigmoidSpec _ _ = do
    let t = sigmoid ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply LogSigmoidSpec _ _ = do
    let t = logSigmoid ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply SinSpec _ _ = do
    let t = sin ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply SinhSpec _ _ = do
    let t = sinh ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply AsinSpec _ _ = do
    let t = asin ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply CosSpec _ _ = do
    let t = cos ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply CoshSpec _ _ = do
    let t = cosh ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply AcosSpec _ _ = do
    let t = acos ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply TanSpec _ _ = do
    let t = tan ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply TanhSpec _ _ = do
    let t = tanh ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply AtanSpec _ _ = do
    let t = atan ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply SqrtSpec _ _ = do
    let t = sqrt ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply RsqrtSpec _ _ = do
    let t = rsqrt ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply RandLikeSpec _ _ = do
    t <- randLike ones :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t
  apply RandnLikeSpec _ _ = do
    t <- randnLike ones :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t

-- instance ( TensorOptions shape dtype device
--          , IsFloatingPoint dtype
--          )
--   => Apply
--        UnaryAllFloatingPointDTypesSpec
--        (Proxy '(device, dtype, shape))
--        (() -> IO ())
--  where

data ToDTypeSpec = ToDTypeSpec

instance ( TensorOptions shape  dtype  device
         , TensorOptions shape' dtype' device'
         , shape' ~ shape
         , device' ~ device
         , KnownDType dtype'
         )
  => Apply
       ToDTypeSpec
       ((Proxy device, (Proxy dtype, Proxy shape)), (Proxy device', (Proxy dtype', Proxy shape')))
       (() -> IO ())
 where
  apply ToDTypeSpec _ _ = do
    let t = ones @shape @dtype @device
        t' = toDType @dtype' t
    checkDynamicTensorAttributes t'

data SumAllSpec = SumAllSpec

instance ( TensorOptions shape dtype device
         , DTypeIsNotHalf dtype
         , KnownDType (SumDType dtype)
         , KnownDevice device
         )
  => Apply
       SumAllSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply SumAllSpec _ _ = do
    let t = ones @shape @dtype @device
        t' = sumAll t
    checkDynamicTensorAttributes t'

data SumDimSpec = SumDimSpec

instance ( TensorOptions shape  dtype  device
         , TensorOptions shape' dtype' device
         , KnownNat d
         , shape' ~ DropValue shape d
         , dtype' ~ SumDType dtype
         , DTypeIsNotHalf dtype
         )
  => Apply
       SumDimSpec
       (Proxy d, (Proxy device, (Proxy dtype, Proxy shape)))
       (() -> IO ())
 where
  apply SumDimSpec _ _ = do
    let t = ones @shape @dtype @device
        t' = sumDim @d t
    checkDynamicTensorAttributes t'

spec :: Spec
spec = do
  let standardShapes               = Proxy @'[2, 3] :. HNil
      standardDTypes'              = hCartesianProduct3 justCPU standardDTypes              standardShapes
      almostAllDTypes'             = hCartesianProduct3 justCPU almostAllDTypes             standardShapes
      allDTypes'                   = hCartesianProduct3 justCPU allDTypes                   standardShapes
      standardFloatingPointDTypes' = hCartesianProduct3 justCPU standardFloatingPointDTypes standardShapes
  describe "unary native ops" $ do
    it "abs"   (hfoldrM @IO AbsSpec   () standardDTypes')
    it "sign"  (hfoldrM @IO SignSpec  () allDTypes')
    it "frac"  (hfoldrM @IO FracSpec  () standardFloatingPointDTypes')
    it "ceil"  (hfoldrM @IO CeilSpec  () standardFloatingPointDTypes')
    it "floor" (hfoldrM @IO FloorSpec () standardFloatingPointDTypes')
    it "trunc" (hfoldrM @IO TruncSpec () standardFloatingPointDTypes')

    it "erf"     (hfoldrM @IO ErfSpec     () standardFloatingPointDTypes')
    it "erfc"    (hfoldrM @IO ErfcSpec    () standardFloatingPointDTypes')
    it "erfinv"  (hfoldrM @IO ErfinvSpec  () standardFloatingPointDTypes')
    it "exp"     (hfoldrM @IO ExpSpec     () standardFloatingPointDTypes')
    it "expm1"   (hfoldrM @IO Expm1Spec   () standardFloatingPointDTypes')
    it "log"     (hfoldrM @IO LogSpec     () standardFloatingPointDTypes')
    it "log1p"   (hfoldrM @IO Log1pSpec   () standardFloatingPointDTypes')
    it "log2"    (hfoldrM @IO Log2Spec    () standardFloatingPointDTypes')
    it "log10"   (hfoldrM @IO Log10Spec   () standardFloatingPointDTypes')
    it "lgamma"  (hfoldrM @IO LgammaSpec  () standardFloatingPointDTypes')
    it "digamma" (hfoldrM @IO DigammaSpec () standardFloatingPointDTypes')

    it "relu"       (hfoldrM @IO ReluSpec       () standardFloatingPointDTypes')
    it "selu"       (hfoldrM @IO SeluSpec       () standardFloatingPointDTypes')
    it "gelu"       (hfoldrM @IO GeluSpec       () standardFloatingPointDTypes')
    it "sigmoid"    (hfoldrM @IO SigmoidSpec    () standardFloatingPointDTypes')
    it "logSigmoid" (hfoldrM @IO LogSigmoidSpec () standardFloatingPointDTypes')

    it "sin"   (hfoldrM @IO SinSpec  () standardFloatingPointDTypes')
    it "sinh"  (hfoldrM @IO SinhSpec () standardFloatingPointDTypes')
    it "asin"  (hfoldrM @IO AsinSpec () standardFloatingPointDTypes')
    it "cos"   (hfoldrM @IO CosSpec  () standardFloatingPointDTypes')
    it "cosh"  (hfoldrM @IO CoshSpec () standardFloatingPointDTypes')
    it "acos"  (hfoldrM @IO AcosSpec () standardFloatingPointDTypes')
    it "tan"   (hfoldrM @IO TanSpec  () standardFloatingPointDTypes')
    it "tanh"  (hfoldrM @IO TanhSpec () standardFloatingPointDTypes')
    it "atan"  (hfoldrM @IO AtanSpec () standardFloatingPointDTypes')
    it "sqrt"  (hfoldrM @IO SqrtSpec () standardFloatingPointDTypes')
    it "rsqrt" (hfoldrM @IO SinhSpec () standardFloatingPointDTypes')

    it "onesLike"  (hfoldrM @IO OnesLikeSpec  () allDTypes')
    it "zerosLike" (hfoldrM @IO ZerosLikeSpec () allDTypes')
    it "randLike"  (hfoldrM @IO RandLikeSpec  () standardFloatingPointDTypes')
    it "randnLike" (hfoldrM @IO RandnLikeSpec () standardFloatingPointDTypes')

    it "toDType" (hfoldrM @IO ToDTypeSpec () (hCartesianProduct allDTypes' allDTypes'))

    it "sumAll" (hfoldrM @IO SumAllSpec () almostAllDTypes')
    it "sumDim" (hfoldrM @IO SumDimSpec () (hCartesianProduct (Proxy @1 :. HNil) almostAllDTypes'))

    it "maxPool2d" $ do
      let c = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
      checkDynamicTensorAttributes c
  describe "binary native ops" $ return ()
