#include <stddef.h>

typedef struct THAllocator {
  void* (*malloc)(void*, ptrdiff_t);
  void* (*realloc)(void*, void*, ptrdiff_t);
  void (*free)(void*, void*);
} THAllocator;

typedef struct THFloatStorage
{
  double *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THFloatStorage *view;
} THFloatStorage;

typedef struct THFloatTensor
{
  long *size;
  long *stride;
  int nDimension;
  THFloatStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THFloatTensor;


typedef struct THDoubleStorage
{
  double *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THDoubleStorage *view;
} THDoubleStorage;

typedef struct THDoubleTensor
{
  long *size;
  long *stride;
  int nDimension;
  THDoubleStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THDoubleTensor;

typedef struct THIntStorage
{
  double *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THIntStorage *view;
} THIntStorage;

typedef struct THIntTensor
{
  long *size;
  long *stride;
  int nDimension;
  THIntStorage *storage;
  ptrdiff_t storageOffset;
  int refcount;
  char flag;
} THIntTensor;
