module Torch.Class.Tensor.Math where

import THTypes
import Foreign
import Foreign.C.Types
import Torch.Class.Internal
import GHC.Int

class TensorMath t where
  fill         :: t -> HsReal t -> IO ()
  zero         :: t -> IO ()
  maskedFill   :: t -> Ptr CTHByteTensor -> HsReal t -> IO ()
  maskedCopy   :: t -> Ptr CTHByteTensor -> t -> IO ()
  maskedSelect :: t -> t -> Ptr CTHByteTensor -> IO ()
  nonzero      :: Ptr CTHLongTensor -> t -> IO ()
  indexSelect  :: t -> t -> Int32 -> Ptr CTHLongTensor -> IO ()
  indexCopy    :: t -> Int32 -> Ptr CTHLongTensor -> t -> IO ()
  indexAdd     :: t -> Int32 -> Ptr CTHLongTensor -> t -> IO ()
  indexFill    :: t -> Int32 -> Ptr CTHLongTensor -> HsReal t -> IO ()
  take         :: t -> t -> Ptr CTHLongTensor -> IO ()
  put          :: t -> Ptr CTHLongTensor -> t -> Int32 -> IO ()
  gather       :: t -> t -> Int32 -> Ptr CTHLongTensor -> IO ()
  scatter      :: t -> Int32 -> Ptr CTHLongTensor -> t -> IO ()
  scatterAdd   :: t -> Int32 -> Ptr CTHLongTensor -> t -> IO ()
  scatterFill  :: t -> Int32 -> Ptr CTHLongTensor -> HsReal t -> IO ()
  dot          :: t -> t -> IO (HsAccReal t)
  minall       :: t -> IO (HsReal t)
  maxall       :: t -> IO (HsReal t)
  medianall    :: t -> IO (HsReal t)
  sumall       :: t -> IO (HsAccReal t)
  prodall      :: t -> IO (HsAccReal t)
  add          :: t -> t -> HsReal t -> IO ()
  sub          :: t -> t -> HsReal t -> IO ()
  add_scaled   :: t -> t -> HsReal t -> HsReal t -> IO ()
  sub_scaled   :: t -> t -> HsReal t -> HsReal t -> IO ()
  mul          :: t -> t -> HsReal t -> IO ()
  div          :: t -> t -> HsReal t -> IO ()
  lshift       :: t -> t -> HsReal t -> IO ()
  rshift       :: t -> t -> HsReal t -> IO ()
  fmod         :: t -> t -> HsReal t -> IO ()
  remainder    :: t -> t -> HsReal t -> IO ()
  clamp        :: t -> t -> HsReal t -> HsReal t -> IO ()
  bitand       :: t -> t -> HsReal t -> IO ()
  bitor        :: t -> t -> HsReal t -> IO ()
  bitxor       :: t -> t -> HsReal t -> IO ()
  cadd         :: t -> t -> HsReal t -> t -> IO ()
  csub         :: t -> t -> HsReal t -> t -> IO ()
  cmul         :: t -> t -> t -> IO ()
  cpow         :: t -> t -> t -> IO ()
  cdiv         :: t -> t -> t -> IO ()
  clshift      :: t -> t -> t -> IO ()
  crshift      :: t -> t -> t -> IO ()
  cfmod        :: t -> t -> t -> IO ()
  cremainder   :: t -> t -> t -> IO ()
  cbitand      :: t -> t -> t -> IO ()
  cbitor       :: t -> t -> t -> IO ()
  cbitxor      :: t -> t -> t -> IO ()
  addcmul      :: t -> t -> HsReal t -> t -> t -> IO ()
  addcdiv      :: t -> t -> HsReal t -> t -> t -> IO ()
  addmv        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addmm        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addr         :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addbmm       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  baddbmm      :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  match        :: t -> t -> t -> HsReal t -> IO ()
  numel        :: t -> IO Int64
  max          :: t -> Ptr CTHLongTensor -> t -> Int32 -> Int32 -> IO ()
  min          :: t -> Ptr CTHLongTensor -> t -> Int32 -> Int32 -> IO ()
  kthvalue     :: t -> Ptr CTHLongTensor -> t -> Int64 -> Int32 -> Int32 -> IO ()
  mode         :: t -> Ptr CTHLongTensor -> t -> Int32 -> Int32 -> IO ()
  median       :: t -> Ptr CTHLongTensor -> t -> Int32 -> Int32 -> IO ()
  sum          :: t -> t -> Int32 -> Int32 -> IO ()
  prod         :: t -> t -> Int32 -> Int32 -> IO ()
  cumsum       :: t -> t -> Int32 -> IO ()
  cumprod      :: t -> t -> Int32 -> IO ()
  sign         :: t -> t -> IO ()
  trace        :: t -> IO (HsAccReal t)
  cross        :: t -> t -> t -> Int32 -> IO ()
  cmax         :: t -> t -> t -> IO ()
  cmin         :: t -> t -> t -> IO ()
  cmaxValue    :: t -> t -> HsReal t -> IO ()
  cminValue    :: t -> t -> HsReal t -> IO ()
  zeros        :: t -> Ptr CTHLongStorage -> IO ()
  zerosLike    :: t -> t -> IO ()
  ones         :: t -> Ptr CTHLongStorage -> IO ()
  onesLike     :: t -> t -> IO ()
  diag         :: t -> t -> Int32 -> IO ()
  eye          :: t -> Int64 -> Int64 -> IO ()
  arange       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  range        :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  randperm     :: t -> Ptr CTHGenerator -> Int64 -> IO ()
  reshape      :: t -> t -> Ptr CTHLongStorage -> IO ()
  sort         :: t -> Ptr CTHLongTensor -> t -> Int32 -> Int32 -> IO ()
  topk         :: t -> Ptr CTHLongTensor -> t -> Int64 -> Int32 -> Int32 -> Int32 -> IO ()
  tril         :: t -> t -> Int64 -> IO ()
  triu         :: t -> t -> Int64 -> IO ()
  cat          :: t -> t -> t -> Int32 -> IO ()
  catArray     :: t -> [t] -> Int32 -> Int32 -> IO ()
  equal        :: t -> t -> IO Int32
  ltValue      :: Ptr CTHByteTensor -> t -> HsReal t -> IO ()
  leValue      :: Ptr CTHByteTensor -> t -> HsReal t -> IO ()
  gtValue      :: Ptr CTHByteTensor -> t -> HsReal t -> IO ()
  geValue      :: Ptr CTHByteTensor -> t -> HsReal t -> IO ()
  neValue      :: Ptr CTHByteTensor -> t -> HsReal t -> IO ()
  eqValue      :: Ptr CTHByteTensor -> t -> HsReal t -> IO ()
  ltValueT     :: t -> t -> HsReal t -> IO ()
  leValueT     :: t -> t -> HsReal t -> IO ()
  gtValueT     :: t -> t -> HsReal t -> IO ()
  geValueT     :: t -> t -> HsReal t -> IO ()
  neValueT     :: t -> t -> HsReal t -> IO ()
  eqValueT     :: t -> t -> HsReal t -> IO ()
  ltTensor     :: Ptr CTHByteTensor -> t -> t -> IO ()
  leTensor     :: Ptr CTHByteTensor -> t -> t -> IO ()
  gtTensor     :: Ptr CTHByteTensor -> t -> t -> IO ()
  geTensor     :: Ptr CTHByteTensor -> t -> t -> IO ()
  neTensor     :: Ptr CTHByteTensor -> t -> t -> IO ()
  eqTensor     :: Ptr CTHByteTensor -> t -> t -> IO ()
  ltTensorT    :: t -> t -> t -> IO ()
  leTensorT    :: t -> t -> t -> IO ()
  gtTensorT    :: t -> t -> t -> IO ()
  geTensorT    :: t -> t -> t -> IO ()
  neTensorT    :: t -> t -> t -> IO ()
  eqTensorT    :: t -> t -> t -> IO ()

class TensorMath t => TensorMathNegative t where
  neg          :: t -> t -> IO ()
  abs          :: t -> t -> IO ()

class TensorMath t => TensorMathFloating t where
  cinv         :: t -> t -> IO ()
  sigmoid      :: t -> t -> IO ()
  log          :: t -> t -> IO ()
  lgamma       :: t -> t -> IO ()
  log1p        :: t -> t -> IO ()
  exp          :: t -> t -> IO ()
  cos          :: t -> t -> IO ()
  acos         :: t -> t -> IO ()
  cosh         :: t -> t -> IO ()
  sin          :: t -> t -> IO ()
  asin         :: t -> t -> IO ()
  sinh         :: t -> t -> IO ()
  tan          :: t -> t -> IO ()
  atan         :: t -> t -> IO ()
  atan2        :: t -> t -> t -> IO ()
  tanh         :: t -> t -> IO ()
  erf          :: t -> t -> IO ()
  erfinv       :: t -> t -> IO ()
  pow          :: t -> t -> HsReal t -> IO ()
  tpow         :: t -> HsReal t -> t -> IO ()
  sqrt         :: t -> t -> IO ()
  rsqrt        :: t -> t -> IO ()
  ceil         :: t -> t -> IO ()
  floor        :: t -> t -> IO ()
  round        :: t -> t -> IO ()
  trunc        :: t -> t -> IO ()
  frac         :: t -> t -> IO ()
  lerp         :: t -> t -> t -> HsReal t -> IO ()
  mean         :: t -> t -> Int32 -> Int32 -> IO ()
  std          :: t -> t -> Int32 -> Int32 -> Int32 -> IO ()
  var          :: t -> t -> Int32 -> Int32 -> Int32 -> IO ()
  norm         :: t -> t -> HsReal t -> Int32 -> Int32 -> IO ()
  renorm       :: t -> t -> HsReal t -> Int32 -> HsReal t -> IO ()
  dist         :: t -> t -> HsReal t -> IO (HsAccReal t)
  histc        :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  bhistc       :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  meanall      :: t -> IO (HsAccReal t)
  varall       :: t -> Int32 -> IO (HsAccReal t)
  stdall       :: t -> Int32 -> IO (HsAccReal t)
  normall      :: t -> HsReal t -> IO (HsAccReal t)
  linspace     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  logspace     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  rand         :: t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()
  randn        :: t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

