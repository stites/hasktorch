{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorMath
  ( c_fill
  , c_zero
  , c_maskedFill
  , c_maskedCopy
  , c_maskedSelect
  , c_nonzero
  , c_indexSelect
  , c_indexCopy
  , c_indexAdd
  , c_indexFill
  , c_take
  , c_put
  , c_gather
  , c_scatter
  , c_scatterAdd
  , c_scatterFill
  , c_dot
  , c_minall
  , c_maxall
  , c_medianall
  , c_sumall
  , c_prodall
  , c_add
  , c_sub
  , c_add_scaled
  , c_sub_scaled
  , c_mul
  , c_div
  , c_lshift
  , c_rshift
  , c_fmod
  , c_remainder
  , c_clamp
  , c_bitand
  , c_bitor
  , c_bitxor
  , c_cadd
  , c_csub
  , c_cmul
  , c_cpow
  , c_cdiv
  , c_clshift
  , c_crshift
  , c_cfmod
  , c_cremainder
  , c_cbitand
  , c_cbitor
  , c_cbitxor
  , c_addcmul
  , c_addcdiv
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , c_match
  , c_numel
  , c_max
  , c_min
  , c_kthvalue
  , c_mode
  , c_median
  , c_sum
  , c_prod
  , c_cumsum
  , c_cumprod
  , c_sign
  , c_trace
  , c_cross
  , c_cmax
  , c_cmin
  , c_cmaxValue
  , c_cminValue
  , c_zeros
  , c_zerosLike
  , c_ones
  , c_onesLike
  , c_diag
  , c_eye
  , c_arange
  , c_range
  , c_randperm
  , c_reshape
  , c_sort
  , c_topk
  , c_tril
  , c_triu
  , c_cat
  , c_catArray
  , c_equal
  , c_ltValue
  , c_leValue
  , c_gtValue
  , c_geValue
  , c_neValue
  , c_eqValue
  , c_ltValueT
  , c_leValueT
  , c_gtValueT
  , c_geValueT
  , c_neValueT
  , c_eqValueT
  , c_ltTensor
  , c_leTensor
  , c_gtTensor
  , c_geTensor
  , c_neTensor
  , c_eqTensor
  , c_ltTensorT
  , c_leTensorT
  , c_gtTensorT
  , c_geTensorT
  , c_neTensorT
  , c_eqTensorT
  , c_logicalall
  , c_logicalany
  , p_fill
  , p_zero
  , p_maskedFill
  , p_maskedCopy
  , p_maskedSelect
  , p_nonzero
  , p_indexSelect
  , p_indexCopy
  , p_indexAdd
  , p_indexFill
  , p_take
  , p_put
  , p_gather
  , p_scatter
  , p_scatterAdd
  , p_scatterFill
  , p_dot
  , p_minall
  , p_maxall
  , p_medianall
  , p_sumall
  , p_prodall
  , p_add
  , p_sub
  , p_add_scaled
  , p_sub_scaled
  , p_mul
  , p_div
  , p_lshift
  , p_rshift
  , p_fmod
  , p_remainder
  , p_clamp
  , p_bitand
  , p_bitor
  , p_bitxor
  , p_cadd
  , p_csub
  , p_cmul
  , p_cpow
  , p_cdiv
  , p_clshift
  , p_crshift
  , p_cfmod
  , p_cremainder
  , p_cbitand
  , p_cbitor
  , p_cbitxor
  , p_addcmul
  , p_addcdiv
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  , p_match
  , p_numel
  , p_max
  , p_min
  , p_kthvalue
  , p_mode
  , p_median
  , p_sum
  , p_prod
  , p_cumsum
  , p_cumprod
  , p_sign
  , p_trace
  , p_cross
  , p_cmax
  , p_cmin
  , p_cmaxValue
  , p_cminValue
  , p_zeros
  , p_zerosLike
  , p_ones
  , p_onesLike
  , p_diag
  , p_eye
  , p_arange
  , p_range
  , p_randperm
  , p_reshape
  , p_sort
  , p_topk
  , p_tril
  , p_triu
  , p_cat
  , p_catArray
  , p_equal
  , p_ltValue
  , p_leValue
  , p_gtValue
  , p_geValue
  , p_neValue
  , p_eqValue
  , p_ltValueT
  , p_leValueT
  , p_gtValueT
  , p_geValueT
  , p_neValueT
  , p_eqValueT
  , p_ltTensor
  , p_leTensor
  , p_gtTensor
  , p_geTensor
  , p_neTensor
  , p_eqTensor
  , p_ltTensorT
  , p_leTensorT
  , p_gtTensorT
  , p_geTensorT
  , p_neTensorT
  , p_eqTensorT
  , p_logicalall
  , p_logicalany
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_fill"
  c_fill :: Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h c_THTensorByte_zero"
  c_zero :: Ptr (CTHByteTensor) -> IO (())

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_maskedFill"
  c_maskedFill :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_maskedCopy"
  c_maskedCopy :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h c_THTensorByte_maskedSelect"
  c_maskedSelect :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h c_THTensorByte_nonzero"
  c_nonzero :: Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h c_THTensorByte_indexSelect"
  c_indexSelect :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_indexCopy"
  c_indexCopy :: Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_indexAdd"
  c_indexAdd :: Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h c_THTensorByte_indexFill"
  c_indexFill :: Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (())

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h c_THTensorByte_take"
  c_take :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h c_THTensorByte_put"
  c_put :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h c_THTensorByte_gather"
  c_gather :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_scatter"
  c_scatter :: Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_scatterAdd"
  c_scatterAdd :: Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h c_THTensorByte_scatterFill"
  c_scatterFill :: Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (())

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h c_THTensorByte_dot"
  c_dot :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CLong)

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorByte_minall"
  c_minall :: Ptr (CTHByteTensor) -> IO (CUChar)

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorByte_maxall"
  c_maxall :: Ptr (CTHByteTensor) -> IO (CUChar)

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorByte_medianall"
  c_medianall :: Ptr (CTHByteTensor) -> IO (CUChar)

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorByte_sumall"
  c_sumall :: Ptr (CTHByteTensor) -> IO (CLong)

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorByte_prodall"
  c_prodall :: Ptr (CTHByteTensor) -> IO (CLong)

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_add"
  c_add :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_sub"
  c_sub :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h c_THTensorByte_add_scaled"
  c_add_scaled :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (())

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h c_THTensorByte_sub_scaled"
  c_sub_scaled :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (())

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_mul"
  c_mul :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_div"
  c_div :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_lshift"
  c_lshift :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_rshift"
  c_rshift :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_fmod"
  c_fmod :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_remainder"
  c_remainder :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_clamp"
  c_clamp :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (())

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_bitand"
  c_bitand :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_bitor"
  c_bitor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_bitxor"
  c_bitxor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cadd"
  c_cadd :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> IO (())

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_csub"
  c_csub :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> IO (())

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cmul"
  c_cmul :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cpow"
  c_cpow :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cdiv"
  c_cdiv :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_clshift"
  c_clshift :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_crshift"
  c_crshift :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cfmod"
  c_cfmod :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cremainder"
  c_cremainder :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cbitand"
  c_cbitand :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cbitor"
  c_cbitor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cbitxor"
  c_cbitxor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_addcmul"
  c_addcmul :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_addcdiv"
  c_addcdiv :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h c_THTensorByte_addmv"
  c_addmv :: Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_addmm"
  c_addmm :: Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_addr"
  c_addr :: Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_addbmm"
  c_addbmm :: Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h c_THTensorByte_baddbmm"
  c_baddbmm :: Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h c_THTensorByte_match"
  c_match :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h c_THTensorByte_numel"
  c_numel :: Ptr (CTHByteTensor) -> IO (CPtrdiff)

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_max"
  c_max :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_min"
  c_min :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_kthvalue"
  c_kthvalue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CLLong -> CInt -> CInt -> IO (())

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_mode"
  c_mode :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_median"
  c_median :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_sum"
  c_sum :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorByte_prod"
  c_prod :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cumsum"
  c_cumsum :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cumprod"
  c_cumprod :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h c_THTensorByte_sign"
  c_sign :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorByte_trace"
  c_trace :: Ptr (CTHByteTensor) -> IO (CLong)

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cross"
  c_cross :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cmax"
  c_cmax :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cmin"
  c_cmin :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cmaxValue"
  c_cmaxValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cminValue"
  c_cminValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h c_THTensorByte_zeros"
  c_zeros :: Ptr (CTHByteTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h c_THTensorByte_zerosLike"
  c_zerosLike :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h c_THTensorByte_ones"
  c_ones :: Ptr (CTHByteTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h c_THTensorByte_onesLike"
  c_onesLike :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorByte_diag"
  c_diag :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h c_THTensorByte_eye"
  c_eye :: Ptr (CTHByteTensor) -> CLLong -> CLLong -> IO (())

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h c_THTensorByte_arange"
  c_arange :: Ptr (CTHByteTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h c_THTensorByte_range"
  c_range :: Ptr (CTHByteTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h c_THTensorByte_randperm"
  c_randperm :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h c_THTensorByte_reshape"
  c_reshape :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h c_THTensorByte_sort"
  c_sort :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (())

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h c_THTensorByte_topk"
  c_topk :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorByte_tril"
  c_tril :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CLLong -> IO (())

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorByte_triu"
  c_triu :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CLLong -> IO (())

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h c_THTensorByte_cat"
  c_cat :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h c_THTensorByte_catArray"
  c_catArray :: Ptr (CTHByteTensor) -> Ptr (Ptr (CTHByteTensor)) -> CInt -> CInt -> IO (())

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h c_THTensorByte_equal"
  c_equal :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CInt)

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_ltValue"
  c_ltValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_leValue"
  c_leValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_gtValue"
  c_gtValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_geValue"
  c_geValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_neValue"
  c_neValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_eqValue"
  c_eqValue :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_ltValueT"
  c_ltValueT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_leValueT"
  c_leValueT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_gtValueT"
  c_gtValueT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_geValueT"
  c_geValueT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_neValueT"
  c_neValueT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorByte_eqValueT"
  c_eqValueT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_ltTensor"
  c_ltTensor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_leTensor"
  c_leTensor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_gtTensor"
  c_gtTensor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_geTensor"
  c_geTensor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_neTensor"
  c_neTensor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_eqTensor"
  c_eqTensor :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_ltTensorT"
  c_ltTensorT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_leTensorT"
  c_leTensorT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_gtTensorT"
  c_gtTensorT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_geTensorT"
  c_geTensorT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_neTensorT"
  c_neTensorT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorByte_eqTensorT"
  c_eqTensorT :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_logicalall :  self -> int
foreign import ccall "THTensorMath.h c_THTensorByte_logicalall"
  c_logicalall :: Ptr (CTHByteTensor) -> IO (CInt)

-- | c_logicalany :  self -> int
foreign import ccall "THTensorMath.h c_THTensorByte_logicalany"
  c_logicalany :: Ptr (CTHByteTensor) -> IO (CInt)

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_fill"
  p_fill :: FunPtr (Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_zero"
  p_zero :: FunPtr (Ptr (CTHByteTensor) -> IO (()))

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (()))

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_take"
  p_take :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_put"
  p_put :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_gather"
  p_gather :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_scatter"
  p_scatter :: FunPtr (Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_scatterFill"
  p_scatterFill :: FunPtr (Ptr (CTHByteTensor) -> CInt -> Ptr (CTHLongTensor) -> CUChar -> IO (()))

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &p_THTensorByte_dot"
  p_dot :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CLong))

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorByte_minall"
  p_minall :: FunPtr (Ptr (CTHByteTensor) -> IO (CUChar))

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorByte_maxall"
  p_maxall :: FunPtr (Ptr (CTHByteTensor) -> IO (CUChar))

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorByte_medianall"
  p_medianall :: FunPtr (Ptr (CTHByteTensor) -> IO (CUChar))

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorByte_sumall"
  p_sumall :: FunPtr (Ptr (CTHByteTensor) -> IO (CLong))

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorByte_prodall"
  p_prodall :: FunPtr (Ptr (CTHByteTensor) -> IO (CLong))

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_add"
  p_add :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_sub"
  p_sub :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (()))

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (()))

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_mul"
  p_mul :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_div"
  p_div :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_lshift"
  p_lshift :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_rshift"
  p_rshift :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_fmod"
  p_fmod :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_remainder"
  p_remainder :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_clamp"
  p_clamp :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (()))

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_bitand"
  p_bitand :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_bitor"
  p_bitor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cadd"
  p_cadd :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> IO (()))

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_csub"
  p_csub :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cmul"
  p_cmul :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cpow"
  p_cpow :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cdiv"
  p_cdiv :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_clshift"
  p_clshift :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_crshift"
  p_crshift :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cfmod"
  p_cfmod :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cremainder"
  p_cremainder :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cbitand"
  p_cbitand :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cbitor"
  p_cbitor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cbitxor"
  p_cbitxor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_addcmul"
  p_addcmul :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_addcdiv"
  p_addcdiv :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_addmv"
  p_addmv :: FunPtr (Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_addmm"
  p_addmm :: FunPtr (Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_addr"
  p_addr :: FunPtr (Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_match"
  p_match :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &p_THTensorByte_numel"
  p_numel :: FunPtr (Ptr (CTHByteTensor) -> IO (CPtrdiff))

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_max"
  p_max :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_min"
  p_min :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_kthvalue"
  p_kthvalue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CLLong -> CInt -> CInt -> IO (()))

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_mode"
  p_mode :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_median"
  p_median :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_sum"
  p_sum :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_prod"
  p_prod :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_sign"
  p_sign :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorByte_trace"
  p_trace :: FunPtr (Ptr (CTHByteTensor) -> IO (CLong))

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cross"
  p_cross :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cmax"
  p_cmax :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cmin"
  p_cmin :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cminValue"
  p_cminValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_zeros"
  p_zeros :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_ones"
  p_ones :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_diag"
  p_diag :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_eye"
  p_eye :: FunPtr (Ptr (CTHByteTensor) -> CLLong -> CLLong -> IO (()))

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_arange"
  p_arange :: FunPtr (Ptr (CTHByteTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_range"
  p_range :: FunPtr (Ptr (CTHByteTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_randperm"
  p_randperm :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_reshape"
  p_reshape :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_sort"
  p_sort :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CInt -> CInt -> IO (()))

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_topk"
  p_topk :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_tril"
  p_tril :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_triu"
  p_triu :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CLLong -> IO (()))

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_cat"
  p_cat :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_catArray"
  p_catArray :: FunPtr (Ptr (CTHByteTensor) -> Ptr (Ptr (CTHByteTensor)) -> CInt -> CInt -> IO (()))

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &p_THTensorByte_equal"
  p_equal :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CInt))

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_ltValue"
  p_ltValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_leValue"
  p_leValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_gtValue"
  p_gtValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_geValue"
  p_geValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_neValue"
  p_neValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_eqValue"
  p_eqValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_ltValueT"
  p_ltValueT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_leValueT"
  p_leValueT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_gtValueT"
  p_gtValueT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_geValueT"
  p_geValueT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_neValueT"
  p_neValueT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_eqValueT"
  p_eqValueT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorByte_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_logicalall : Pointer to function : self -> int
foreign import ccall "THTensorMath.h &p_THTensorByte_logicalall"
  p_logicalall :: FunPtr (Ptr (CTHByteTensor) -> IO (CInt))

-- | p_logicalany : Pointer to function : self -> int
foreign import ccall "THTensorMath.h &p_THTensorByte_logicalany"
  p_logicalany :: FunPtr (Ptr (CTHByteTensor) -> IO (CInt))