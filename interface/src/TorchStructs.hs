{-# LINE 1 "TorchStructs.hsc" #-}
{-# OPTIONS_GHC -fno-warn-unused-imports #-}
{-# LINE 2 "TorchStructs.hsc" #-}

{-# LINE 3 "TorchStructs.hsc" #-}

{-# LINE 4 "TorchStructs.hsc" #-}
module TorchStructs where
import Foreign.Ptr
import Foreign.Ptr (Ptr,FunPtr,plusPtr)
import Foreign.Ptr (wordPtrToPtr,castPtrToFunPtr)
import Foreign.Storable
import Foreign.C.Types
import Foreign.C.String (CString,CStringLen,CWString,CWStringLen)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray,pokeArray)
import Data.Int
import Data.Word

{-# LINE 7 "TorchStructs.hsc" #-}

{- typedef struct THAllocator {
            void * (* malloc)(void *, ptrdiff_t);
            void * (* realloc)(void *, void *, ptrdiff_t);
            void (* free)(void *, void *);
        } THAllocator; -}

{-# LINE 14 "TorchStructs.hsc" #-}

{-# LINE 15 "TorchStructs.hsc" #-}

{-# LINE 16 "TorchStructs.hsc" #-}

{-# LINE 17 "TorchStructs.hsc" #-}
data C'THAllocator = C'THAllocator{
  c'THAllocator'malloc :: FunPtr (Ptr () -> CLong -> Ptr ()),
  c'THAllocator'realloc :: FunPtr (Ptr () -> Ptr () -> CLong -> Ptr ()),
  c'THAllocator'free :: FunPtr (Ptr () -> Ptr () -> IO ())
} deriving (Eq,Show)
p'THAllocator'malloc p = plusPtr p 0
p'THAllocator'malloc :: Ptr (C'THAllocator) -> Ptr (FunPtr (Ptr () -> CLong -> Ptr ()))
p'THAllocator'realloc p = plusPtr p 8
p'THAllocator'realloc :: Ptr (C'THAllocator) -> Ptr (FunPtr (Ptr () -> Ptr () -> CLong -> Ptr ()))
p'THAllocator'free p = plusPtr p 16
p'THAllocator'free :: Ptr (C'THAllocator) -> Ptr (FunPtr (Ptr () -> Ptr () -> IO ()))
instance Storable C'THAllocator where
  sizeOf _ = 24
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    return $ C'THAllocator v0 v1 v2
  poke _p (C'THAllocator v0 v1 v2) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    return ()

{-# LINE 18 "TorchStructs.hsc" #-}

{- typedef struct THFloatStorage {
            float * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THFloatStorage * view;
        } THFloatStorage; -}

{-# LINE 29 "TorchStructs.hsc" #-}

{-# LINE 30 "TorchStructs.hsc" #-}

{-# LINE 31 "TorchStructs.hsc" #-}

{-# LINE 32 "TorchStructs.hsc" #-}

{-# LINE 33 "TorchStructs.hsc" #-}

{-# LINE 34 "TorchStructs.hsc" #-}

{-# LINE 35 "TorchStructs.hsc" #-}

{-# LINE 36 "TorchStructs.hsc" #-}
data C'THFloatStorage = C'THFloatStorage{
  c'THFloatStorage'data :: Ptr CFloat,
  c'THFloatStorage'size :: CLong,
  c'THFloatStorage'refcount :: CInt,
  c'THFloatStorage'flag :: CChar,
  c'THFloatStorage'allocator :: Ptr C'THAllocator,
  c'THFloatStorage'allocatorContext :: Ptr (),
  c'THFloatStorage'view :: Ptr C'THFloatStorage
} deriving (Eq,Show)
p'THFloatStorage'data p = plusPtr p 0
p'THFloatStorage'data :: Ptr (C'THFloatStorage) -> Ptr (Ptr CFloat)
p'THFloatStorage'size p = plusPtr p 8
p'THFloatStorage'size :: Ptr (C'THFloatStorage) -> Ptr (CLong)
p'THFloatStorage'refcount p = plusPtr p 16
p'THFloatStorage'refcount :: Ptr (C'THFloatStorage) -> Ptr (CInt)
p'THFloatStorage'flag p = plusPtr p 20
p'THFloatStorage'flag :: Ptr (C'THFloatStorage) -> Ptr (CChar)
p'THFloatStorage'allocator p = plusPtr p 24
p'THFloatStorage'allocator :: Ptr (C'THFloatStorage) -> Ptr (Ptr C'THAllocator)
p'THFloatStorage'allocatorContext p = plusPtr p 32
p'THFloatStorage'allocatorContext :: Ptr (C'THFloatStorage) -> Ptr (Ptr ())
p'THFloatStorage'view p = plusPtr p 40
p'THFloatStorage'view :: Ptr (C'THFloatStorage) -> Ptr (Ptr C'THFloatStorage)
instance Storable C'THFloatStorage where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'THFloatStorage v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THFloatStorage v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()

{-# LINE 37 "TorchStructs.hsc" #-}

{- typedef struct THFloatTensor {
            long * size;
            long * stride;
            int nDimension;
            THFloatStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THFloatTensor; -}

{-# LINE 48 "TorchStructs.hsc" #-}

{-# LINE 49 "TorchStructs.hsc" #-}

{-# LINE 50 "TorchStructs.hsc" #-}

{-# LINE 51 "TorchStructs.hsc" #-}

{-# LINE 52 "TorchStructs.hsc" #-}

{-# LINE 53 "TorchStructs.hsc" #-}

{-# LINE 54 "TorchStructs.hsc" #-}

{-# LINE 55 "TorchStructs.hsc" #-}
data C'THFloatTensor = C'THFloatTensor{
  c'THFloatTensor'size :: Ptr CLong,
  c'THFloatTensor'stride :: Ptr CLong,
  c'THFloatTensor'nDimension :: CInt,
  c'THFloatTensor'storage :: Ptr C'THFloatStorage,
  c'THFloatTensor'storageOffset :: CLong,
  c'THFloatTensor'refcount :: CInt,
  c'THFloatTensor'flag :: CChar
} deriving (Eq,Show)
p'THFloatTensor'size p = plusPtr p 0
p'THFloatTensor'size :: Ptr (C'THFloatTensor) -> Ptr (Ptr CLong)
p'THFloatTensor'stride p = plusPtr p 8
p'THFloatTensor'stride :: Ptr (C'THFloatTensor) -> Ptr (Ptr CLong)
p'THFloatTensor'nDimension p = plusPtr p 16
p'THFloatTensor'nDimension :: Ptr (C'THFloatTensor) -> Ptr (CInt)
p'THFloatTensor'storage p = plusPtr p 24
p'THFloatTensor'storage :: Ptr (C'THFloatTensor) -> Ptr (Ptr C'THFloatStorage)
p'THFloatTensor'storageOffset p = plusPtr p 32
p'THFloatTensor'storageOffset :: Ptr (C'THFloatTensor) -> Ptr (CLong)
p'THFloatTensor'refcount p = plusPtr p 40
p'THFloatTensor'refcount :: Ptr (C'THFloatTensor) -> Ptr (CInt)
p'THFloatTensor'flag p = plusPtr p 44
p'THFloatTensor'flag :: Ptr (C'THFloatTensor) -> Ptr (CChar)
instance Storable C'THFloatTensor where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    v6 <- peekByteOff _p 44
    return $ C'THFloatTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THFloatTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()

{-# LINE 56 "TorchStructs.hsc" #-}

{- typedef struct THDoubleStorage {
            double * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THDoubleStorage * view;
        } THDoubleStorage; -}

{-# LINE 67 "TorchStructs.hsc" #-}

{-# LINE 68 "TorchStructs.hsc" #-}

{-# LINE 69 "TorchStructs.hsc" #-}

{-# LINE 70 "TorchStructs.hsc" #-}

{-# LINE 71 "TorchStructs.hsc" #-}

{-# LINE 72 "TorchStructs.hsc" #-}

{-# LINE 73 "TorchStructs.hsc" #-}

{-# LINE 74 "TorchStructs.hsc" #-}
data C'THDoubleStorage = C'THDoubleStorage{
  c'THDoubleStorage'data :: Ptr CDouble,
  c'THDoubleStorage'size :: CLong,
  c'THDoubleStorage'refcount :: CInt,
  c'THDoubleStorage'flag :: CChar,
  c'THDoubleStorage'allocator :: Ptr C'THAllocator,
  c'THDoubleStorage'allocatorContext :: Ptr (),
  c'THDoubleStorage'view :: Ptr C'THDoubleStorage
} deriving (Eq,Show)
p'THDoubleStorage'data p = plusPtr p 0
p'THDoubleStorage'data :: Ptr (C'THDoubleStorage) -> Ptr (Ptr CDouble)
p'THDoubleStorage'size p = plusPtr p 8
p'THDoubleStorage'size :: Ptr (C'THDoubleStorage) -> Ptr (CLong)
p'THDoubleStorage'refcount p = plusPtr p 16
p'THDoubleStorage'refcount :: Ptr (C'THDoubleStorage) -> Ptr (CInt)
p'THDoubleStorage'flag p = plusPtr p 20
p'THDoubleStorage'flag :: Ptr (C'THDoubleStorage) -> Ptr (CChar)
p'THDoubleStorage'allocator p = plusPtr p 24
p'THDoubleStorage'allocator :: Ptr (C'THDoubleStorage) -> Ptr (Ptr C'THAllocator)
p'THDoubleStorage'allocatorContext p = plusPtr p 32
p'THDoubleStorage'allocatorContext :: Ptr (C'THDoubleStorage) -> Ptr (Ptr ())
p'THDoubleStorage'view p = plusPtr p 40
p'THDoubleStorage'view :: Ptr (C'THDoubleStorage) -> Ptr (Ptr C'THDoubleStorage)
instance Storable C'THDoubleStorage where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'THDoubleStorage v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THDoubleStorage v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()

{-# LINE 75 "TorchStructs.hsc" #-}

{- typedef struct THDoubleTensor {
            long * size;
            long * stride;
            int nDimension;
            THDoubleStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THDoubleTensor; -}

{-# LINE 86 "TorchStructs.hsc" #-}

{-# LINE 87 "TorchStructs.hsc" #-}

{-# LINE 88 "TorchStructs.hsc" #-}

{-# LINE 89 "TorchStructs.hsc" #-}

{-# LINE 90 "TorchStructs.hsc" #-}

{-# LINE 91 "TorchStructs.hsc" #-}

{-# LINE 92 "TorchStructs.hsc" #-}

{-# LINE 93 "TorchStructs.hsc" #-}
data C'THDoubleTensor = C'THDoubleTensor{
  c'THDoubleTensor'size :: Ptr CLong,
  c'THDoubleTensor'stride :: Ptr CLong,
  c'THDoubleTensor'nDimension :: CInt,
  c'THDoubleTensor'storage :: Ptr C'THDoubleStorage,
  c'THDoubleTensor'storageOffset :: CLong,
  c'THDoubleTensor'refcount :: CInt,
  c'THDoubleTensor'flag :: CChar
} deriving (Eq,Show)
p'THDoubleTensor'size p = plusPtr p 0
p'THDoubleTensor'size :: Ptr (C'THDoubleTensor) -> Ptr (Ptr CLong)
p'THDoubleTensor'stride p = plusPtr p 8
p'THDoubleTensor'stride :: Ptr (C'THDoubleTensor) -> Ptr (Ptr CLong)
p'THDoubleTensor'nDimension p = plusPtr p 16
p'THDoubleTensor'nDimension :: Ptr (C'THDoubleTensor) -> Ptr (CInt)
p'THDoubleTensor'storage p = plusPtr p 24
p'THDoubleTensor'storage :: Ptr (C'THDoubleTensor) -> Ptr (Ptr C'THDoubleStorage)
p'THDoubleTensor'storageOffset p = plusPtr p 32
p'THDoubleTensor'storageOffset :: Ptr (C'THDoubleTensor) -> Ptr (CLong)
p'THDoubleTensor'refcount p = plusPtr p 40
p'THDoubleTensor'refcount :: Ptr (C'THDoubleTensor) -> Ptr (CInt)
p'THDoubleTensor'flag p = plusPtr p 44
p'THDoubleTensor'flag :: Ptr (C'THDoubleTensor) -> Ptr (CChar)
instance Storable C'THDoubleTensor where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    v6 <- peekByteOff _p 44
    return $ C'THDoubleTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THDoubleTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()

{-# LINE 94 "TorchStructs.hsc" #-}

{- typedef struct THIntStorage {
            int * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THIntStorage * view;
        } THIntStorage; -}

{-# LINE 105 "TorchStructs.hsc" #-}

{-# LINE 106 "TorchStructs.hsc" #-}

{-# LINE 107 "TorchStructs.hsc" #-}

{-# LINE 108 "TorchStructs.hsc" #-}

{-# LINE 109 "TorchStructs.hsc" #-}

{-# LINE 110 "TorchStructs.hsc" #-}

{-# LINE 111 "TorchStructs.hsc" #-}

{-# LINE 112 "TorchStructs.hsc" #-}
data C'THIntStorage = C'THIntStorage{
  c'THIntStorage'data :: Ptr CInt,
  c'THIntStorage'size :: CLong,
  c'THIntStorage'refcount :: CInt,
  c'THIntStorage'flag :: CChar,
  c'THIntStorage'allocator :: Ptr C'THAllocator,
  c'THIntStorage'allocatorContext :: Ptr (),
  c'THIntStorage'view :: Ptr C'THIntStorage
} deriving (Eq,Show)
p'THIntStorage'data p = plusPtr p 0
p'THIntStorage'data :: Ptr (C'THIntStorage) -> Ptr (Ptr CInt)
p'THIntStorage'size p = plusPtr p 8
p'THIntStorage'size :: Ptr (C'THIntStorage) -> Ptr (CLong)
p'THIntStorage'refcount p = plusPtr p 16
p'THIntStorage'refcount :: Ptr (C'THIntStorage) -> Ptr (CInt)
p'THIntStorage'flag p = plusPtr p 20
p'THIntStorage'flag :: Ptr (C'THIntStorage) -> Ptr (CChar)
p'THIntStorage'allocator p = plusPtr p 24
p'THIntStorage'allocator :: Ptr (C'THIntStorage) -> Ptr (Ptr C'THAllocator)
p'THIntStorage'allocatorContext p = plusPtr p 32
p'THIntStorage'allocatorContext :: Ptr (C'THIntStorage) -> Ptr (Ptr ())
p'THIntStorage'view p = plusPtr p 40
p'THIntStorage'view :: Ptr (C'THIntStorage) -> Ptr (Ptr C'THIntStorage)
instance Storable C'THIntStorage where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'THIntStorage v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THIntStorage v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()

{-# LINE 113 "TorchStructs.hsc" #-}

{- typedef struct THIntTensor {
            long * size;
            long * stride;
            int nDimension;
            THIntStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THIntTensor; -}

{-# LINE 124 "TorchStructs.hsc" #-}

{-# LINE 125 "TorchStructs.hsc" #-}

{-# LINE 126 "TorchStructs.hsc" #-}

{-# LINE 127 "TorchStructs.hsc" #-}

{-# LINE 128 "TorchStructs.hsc" #-}

{-# LINE 129 "TorchStructs.hsc" #-}

{-# LINE 130 "TorchStructs.hsc" #-}

{-# LINE 131 "TorchStructs.hsc" #-}
data C'THIntTensor = C'THIntTensor{
  c'THIntTensor'size :: Ptr CLong,
  c'THIntTensor'stride :: Ptr CLong,
  c'THIntTensor'nDimension :: CInt,
  c'THIntTensor'storage :: Ptr C'THIntStorage,
  c'THIntTensor'storageOffset :: CLong,
  c'THIntTensor'refcount :: CInt,
  c'THIntTensor'flag :: CChar
} deriving (Eq,Show)
p'THIntTensor'size p = plusPtr p 0
p'THIntTensor'size :: Ptr (C'THIntTensor) -> Ptr (Ptr CLong)
p'THIntTensor'stride p = plusPtr p 8
p'THIntTensor'stride :: Ptr (C'THIntTensor) -> Ptr (Ptr CLong)
p'THIntTensor'nDimension p = plusPtr p 16
p'THIntTensor'nDimension :: Ptr (C'THIntTensor) -> Ptr (CInt)
p'THIntTensor'storage p = plusPtr p 24
p'THIntTensor'storage :: Ptr (C'THIntTensor) -> Ptr (Ptr C'THIntStorage)
p'THIntTensor'storageOffset p = plusPtr p 32
p'THIntTensor'storageOffset :: Ptr (C'THIntTensor) -> Ptr (CLong)
p'THIntTensor'refcount p = plusPtr p 40
p'THIntTensor'refcount :: Ptr (C'THIntTensor) -> Ptr (CInt)
p'THIntTensor'flag p = plusPtr p 44
p'THIntTensor'flag :: Ptr (C'THIntTensor) -> Ptr (CChar)
instance Storable C'THIntTensor where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    v6 <- peekByteOff _p 44
    return $ C'THIntTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THIntTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()

{-# LINE 132 "TorchStructs.hsc" #-}

{- typedef struct THCharStorage {
            char * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THCharStorage * view;
        } THCharStorage; -}

{-# LINE 143 "TorchStructs.hsc" #-}

{-# LINE 144 "TorchStructs.hsc" #-}

{-# LINE 145 "TorchStructs.hsc" #-}

{-# LINE 146 "TorchStructs.hsc" #-}

{-# LINE 147 "TorchStructs.hsc" #-}

{-# LINE 148 "TorchStructs.hsc" #-}

{-# LINE 149 "TorchStructs.hsc" #-}

{-# LINE 150 "TorchStructs.hsc" #-}
data C'THCharStorage = C'THCharStorage{
  c'THCharStorage'data :: CString,
  c'THCharStorage'size :: CLong,
  c'THCharStorage'refcount :: CInt,
  c'THCharStorage'flag :: CChar,
  c'THCharStorage'allocator :: Ptr C'THAllocator,
  c'THCharStorage'allocatorContext :: Ptr (),
  c'THCharStorage'view :: Ptr C'THCharStorage
} deriving (Eq,Show)
p'THCharStorage'data p = plusPtr p 0
p'THCharStorage'data :: Ptr (C'THCharStorage) -> Ptr (CString)
p'THCharStorage'size p = plusPtr p 8
p'THCharStorage'size :: Ptr (C'THCharStorage) -> Ptr (CLong)
p'THCharStorage'refcount p = plusPtr p 16
p'THCharStorage'refcount :: Ptr (C'THCharStorage) -> Ptr (CInt)
p'THCharStorage'flag p = plusPtr p 20
p'THCharStorage'flag :: Ptr (C'THCharStorage) -> Ptr (CChar)
p'THCharStorage'allocator p = plusPtr p 24
p'THCharStorage'allocator :: Ptr (C'THCharStorage) -> Ptr (Ptr C'THAllocator)
p'THCharStorage'allocatorContext p = plusPtr p 32
p'THCharStorage'allocatorContext :: Ptr (C'THCharStorage) -> Ptr (Ptr ())
p'THCharStorage'view p = plusPtr p 40
p'THCharStorage'view :: Ptr (C'THCharStorage) -> Ptr (Ptr C'THCharStorage)
instance Storable C'THCharStorage where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'THCharStorage v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCharStorage v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()

{-# LINE 151 "TorchStructs.hsc" #-}

{- typedef struct THCharTensor {
            long * size;
            long * stride;
            int nDimension;
            THCharStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THCharTensor; -}

{-# LINE 162 "TorchStructs.hsc" #-}

{-# LINE 163 "TorchStructs.hsc" #-}

{-# LINE 164 "TorchStructs.hsc" #-}

{-# LINE 165 "TorchStructs.hsc" #-}

{-# LINE 166 "TorchStructs.hsc" #-}

{-# LINE 167 "TorchStructs.hsc" #-}

{-# LINE 168 "TorchStructs.hsc" #-}

{-# LINE 169 "TorchStructs.hsc" #-}
data C'THCharTensor = C'THCharTensor{
  c'THCharTensor'size :: Ptr CLong,
  c'THCharTensor'stride :: Ptr CLong,
  c'THCharTensor'nDimension :: CInt,
  c'THCharTensor'storage :: Ptr C'THCharStorage,
  c'THCharTensor'storageOffset :: CLong,
  c'THCharTensor'refcount :: CInt,
  c'THCharTensor'flag :: CChar
} deriving (Eq,Show)
p'THCharTensor'size p = plusPtr p 0
p'THCharTensor'size :: Ptr (C'THCharTensor) -> Ptr (Ptr CLong)
p'THCharTensor'stride p = plusPtr p 8
p'THCharTensor'stride :: Ptr (C'THCharTensor) -> Ptr (Ptr CLong)
p'THCharTensor'nDimension p = plusPtr p 16
p'THCharTensor'nDimension :: Ptr (C'THCharTensor) -> Ptr (CInt)
p'THCharTensor'storage p = plusPtr p 24
p'THCharTensor'storage :: Ptr (C'THCharTensor) -> Ptr (Ptr C'THCharStorage)
p'THCharTensor'storageOffset p = plusPtr p 32
p'THCharTensor'storageOffset :: Ptr (C'THCharTensor) -> Ptr (CLong)
p'THCharTensor'refcount p = plusPtr p 40
p'THCharTensor'refcount :: Ptr (C'THCharTensor) -> Ptr (CInt)
p'THCharTensor'flag p = plusPtr p 44
p'THCharTensor'flag :: Ptr (C'THCharTensor) -> Ptr (CChar)
instance Storable C'THCharTensor where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    v6 <- peekByteOff _p 44
    return $ C'THCharTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THCharTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()

{-# LINE 170 "TorchStructs.hsc" #-}

{- typedef struct THByteStorage {
            unsigned char * data;
            ptrdiff_t size;
            int refcount;
            char flag;
            THAllocator * allocator;
            void * allocatorContext;
            struct THByteStorage * view;
        } THByteStorage; -}

{-# LINE 181 "TorchStructs.hsc" #-}

{-# LINE 182 "TorchStructs.hsc" #-}

{-# LINE 183 "TorchStructs.hsc" #-}

{-# LINE 184 "TorchStructs.hsc" #-}

{-# LINE 185 "TorchStructs.hsc" #-}

{-# LINE 186 "TorchStructs.hsc" #-}

{-# LINE 187 "TorchStructs.hsc" #-}

{-# LINE 188 "TorchStructs.hsc" #-}
data C'THByteStorage = C'THByteStorage{
  c'THByteStorage'data :: Ptr CUChar,
  c'THByteStorage'size :: CLong,
  c'THByteStorage'refcount :: CInt,
  c'THByteStorage'flag :: CChar,
  c'THByteStorage'allocator :: Ptr C'THAllocator,
  c'THByteStorage'allocatorContext :: Ptr (),
  c'THByteStorage'view :: Ptr C'THByteStorage
} deriving (Eq,Show)
p'THByteStorage'data p = plusPtr p 0
p'THByteStorage'data :: Ptr (C'THByteStorage) -> Ptr (Ptr CUChar)
p'THByteStorage'size p = plusPtr p 8
p'THByteStorage'size :: Ptr (C'THByteStorage) -> Ptr (CLong)
p'THByteStorage'refcount p = plusPtr p 16
p'THByteStorage'refcount :: Ptr (C'THByteStorage) -> Ptr (CInt)
p'THByteStorage'flag p = plusPtr p 20
p'THByteStorage'flag :: Ptr (C'THByteStorage) -> Ptr (CChar)
p'THByteStorage'allocator p = plusPtr p 24
p'THByteStorage'allocator :: Ptr (C'THByteStorage) -> Ptr (Ptr C'THAllocator)
p'THByteStorage'allocatorContext p = plusPtr p 32
p'THByteStorage'allocatorContext :: Ptr (C'THByteStorage) -> Ptr (Ptr ())
p'THByteStorage'view p = plusPtr p 40
p'THByteStorage'view :: Ptr (C'THByteStorage) -> Ptr (Ptr C'THByteStorage)
instance Storable C'THByteStorage where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 20
    v4 <- peekByteOff _p 24
    v5 <- peekByteOff _p 32
    v6 <- peekByteOff _p 40
    return $ C'THByteStorage v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THByteStorage v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 20 v3
    pokeByteOff _p 24 v4
    pokeByteOff _p 32 v5
    pokeByteOff _p 40 v6
    return ()

{-# LINE 189 "TorchStructs.hsc" #-}

{- typedef struct THByteTensor {
            long * size;
            long * stride;
            int nDimension;
            THByteStorage * storage;
            ptrdiff_t storageOffset;
            int refcount;
            char flag;
        } THByteTensor; -}

{-# LINE 200 "TorchStructs.hsc" #-}

{-# LINE 201 "TorchStructs.hsc" #-}

{-# LINE 202 "TorchStructs.hsc" #-}

{-# LINE 203 "TorchStructs.hsc" #-}

{-# LINE 204 "TorchStructs.hsc" #-}

{-# LINE 205 "TorchStructs.hsc" #-}

{-# LINE 206 "TorchStructs.hsc" #-}

{-# LINE 207 "TorchStructs.hsc" #-}
data C'THByteTensor = C'THByteTensor{
  c'THByteTensor'size :: Ptr CLong,
  c'THByteTensor'stride :: Ptr CLong,
  c'THByteTensor'nDimension :: CInt,
  c'THByteTensor'storage :: Ptr C'THByteStorage,
  c'THByteTensor'storageOffset :: CLong,
  c'THByteTensor'refcount :: CInt,
  c'THByteTensor'flag :: CChar
} deriving (Eq,Show)
p'THByteTensor'size p = plusPtr p 0
p'THByteTensor'size :: Ptr (C'THByteTensor) -> Ptr (Ptr CLong)
p'THByteTensor'stride p = plusPtr p 8
p'THByteTensor'stride :: Ptr (C'THByteTensor) -> Ptr (Ptr CLong)
p'THByteTensor'nDimension p = plusPtr p 16
p'THByteTensor'nDimension :: Ptr (C'THByteTensor) -> Ptr (CInt)
p'THByteTensor'storage p = plusPtr p 24
p'THByteTensor'storage :: Ptr (C'THByteTensor) -> Ptr (Ptr C'THByteStorage)
p'THByteTensor'storageOffset p = plusPtr p 32
p'THByteTensor'storageOffset :: Ptr (C'THByteTensor) -> Ptr (CLong)
p'THByteTensor'refcount p = plusPtr p 40
p'THByteTensor'refcount :: Ptr (C'THByteTensor) -> Ptr (CInt)
p'THByteTensor'flag p = plusPtr p 44
p'THByteTensor'flag :: Ptr (C'THByteTensor) -> Ptr (CChar)
instance Storable C'THByteTensor where
  sizeOf _ = 48
  alignment _ = 8
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 8
    v2 <- peekByteOff _p 16
    v3 <- peekByteOff _p 24
    v4 <- peekByteOff _p 32
    v5 <- peekByteOff _p 40
    v6 <- peekByteOff _p 44
    return $ C'THByteTensor v0 v1 v2 v3 v4 v5 v6
  poke _p (C'THByteTensor v0 v1 v2 v3 v4 v5 v6) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 8 v1
    pokeByteOff _p 16 v2
    pokeByteOff _p 24 v3
    pokeByteOff _p 32 v4
    pokeByteOff _p 40 v5
    pokeByteOff _p 44 v6
    return ()

{-# LINE 208 "TorchStructs.hsc" #-}

