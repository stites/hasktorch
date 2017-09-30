{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensorLapack (
    c_THByteTensor_gesv,
    c_THByteTensor_trtrs,
    c_THByteTensor_gels,
    c_THByteTensor_syev,
    c_THByteTensor_geev,
    c_THByteTensor_gesvd,
    c_THByteTensor_gesvd2,
    c_THByteTensor_getri,
    c_THByteTensor_potrf,
    c_THByteTensor_potrs,
    c_THByteTensor_potri,
    c_THByteTensor_qr,
    c_THByteTensor_geqrf,
    c_THByteTensor_orgqr,
    c_THByteTensor_ormqr,
    c_THByteTensor_pstrf,
    c_THByteTensor_btrifact,
    c_THByteTensor_btrisolve,
    p_THByteTensor_gesv,
    p_THByteTensor_trtrs,
    p_THByteTensor_gels,
    p_THByteTensor_syev,
    p_THByteTensor_geev,
    p_THByteTensor_gesvd,
    p_THByteTensor_gesvd2,
    p_THByteTensor_getri,
    p_THByteTensor_potrf,
    p_THByteTensor_potrs,
    p_THByteTensor_potri,
    p_THByteTensor_qr,
    p_THByteTensor_geqrf,
    p_THByteTensor_orgqr,
    p_THByteTensor_ormqr,
    p_THByteTensor_pstrf,
    p_THByteTensor_btrifact,
    p_THByteTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_gesv"
  c_THByteTensor_gesv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_trtrs"
  c_THByteTensor_trtrs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_gels"
  c_THByteTensor_gels :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_syev"
  c_THByteTensor_syev :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_geev"
  c_THByteTensor_geev :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ()

-- |c_THByteTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_gesvd"
  c_THByteTensor_gesvd :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ()

-- |c_THByteTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_gesvd2"
  c_THByteTensor_gesvd2 :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ()

-- |c_THByteTensor_getri : ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_getri"
  c_THByteTensor_getri :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_potrf : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_potrf"
  c_THByteTensor_potrf :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ()

-- |c_THByteTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_potrs"
  c_THByteTensor_potrs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ()

-- |c_THByteTensor_potri : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_potri"
  c_THByteTensor_potri :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ()

-- |c_THByteTensor_qr : rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_qr"
  c_THByteTensor_qr :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_geqrf"
  c_THByteTensor_geqrf :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_orgqr : ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_orgqr"
  c_THByteTensor_orgqr :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_ormqr"
  c_THByteTensor_ormqr :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_pstrf"
  c_THByteTensor_pstrf :: (Ptr CTHByteTensor) -> Ptr CTHIntTensor -> (Ptr CTHByteTensor) -> Ptr CChar -> CChar -> IO ()

-- |c_THByteTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_btrifact"
  c_THByteTensor_btrifact :: (Ptr CTHByteTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h THByteTensor_btrisolve"
  c_THByteTensor_btrisolve :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHIntTensor -> IO ()

-- |p_THByteTensor_gesv : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_gesv"
  p_THByteTensor_gesv :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_trtrs : Pointer to rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_trtrs"
  p_THByteTensor_trtrs :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THByteTensor_gels : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_gels"
  p_THByteTensor_gels :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_syev : Pointer to re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_syev"
  p_THByteTensor_syev :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THByteTensor_geev : Pointer to re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_geev"
  p_THByteTensor_geev :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ())

-- |p_THByteTensor_gesvd : Pointer to ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_gesvd"
  p_THByteTensor_gesvd :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ())

-- |p_THByteTensor_gesvd2 : Pointer to ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_gesvd2"
  p_THByteTensor_gesvd2 :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ())

-- |p_THByteTensor_getri : Pointer to ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_getri"
  p_THByteTensor_getri :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_potrf : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_potrf"
  p_THByteTensor_potrf :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ())

-- |p_THByteTensor_potrs : Pointer to rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_potrs"
  p_THByteTensor_potrs :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ())

-- |p_THByteTensor_potri : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_potri"
  p_THByteTensor_potri :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> IO ())

-- |p_THByteTensor_qr : Pointer to rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_qr"
  p_THByteTensor_qr :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_geqrf : Pointer to ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_geqrf"
  p_THByteTensor_geqrf :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_orgqr : Pointer to ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_orgqr"
  p_THByteTensor_orgqr :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_ormqr : Pointer to ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_ormqr"
  p_THByteTensor_ormqr :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THByteTensor_pstrf : Pointer to ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_pstrf"
  p_THByteTensor_pstrf :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHIntTensor -> (Ptr CTHByteTensor) -> Ptr CChar -> CChar -> IO ())

-- |p_THByteTensor_btrifact : Pointer to ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_btrifact"
  p_THByteTensor_btrifact :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_btrisolve : Pointer to rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h &THByteTensor_btrisolve"
  p_THByteTensor_btrisolve :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHIntTensor -> IO ())