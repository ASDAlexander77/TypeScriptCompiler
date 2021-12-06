module @"c:\\temp\\1.ts"  {
  llvm.func @GC_init()
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    llvm.call @GC_init() : () -> ()
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %9 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.getelementptr %9[%10, %10] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %13 = llvm.insertvalue %11, %12[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %14 = llvm.insertvalue %8, %13[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %14, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %15 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %16 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %17 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %20 = llvm.ptrtoint %19 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %21 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %25 = llvm.icmp "ult" %20, %24 : i64
    %26 = llvm.select %25, %20, %24 : i1, i64
    %27 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%15, %16, %26, %27) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %28 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %28, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %29 = llvm.load %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %29, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.return
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @GC_malloc(i64) -> !llvm.ptr<i8>
  llvm.func @GC_free(!llvm.ptr<i8>)
}