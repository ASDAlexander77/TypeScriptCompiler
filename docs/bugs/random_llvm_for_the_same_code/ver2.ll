module @"c:\\temp\\1.ts"  {
  llvm.func @GC_init()
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    llvm.call @GC_init() : () -> ()
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %13 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.getelementptr %13[%14, %14] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %17 = llvm.insertvalue %15, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %18 = llvm.insertvalue %12, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %18, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %19 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %20 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %21 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %25 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %29 = llvm.icmp "ult" %24, %28 : i64
    %30 = llvm.select %29, %24, %28 : i1, i64
    %31 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%19, %20, %30, %31) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %32 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %32, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %33, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %34 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %35 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %36 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.getelementptr %36[%37] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %39 = llvm.ptrtoint %38 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %40 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %41 = llvm.mlir.constant(1 : i64) : i64
    %42 = llvm.getelementptr %40[%41] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %43 = llvm.ptrtoint %42 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %44 = llvm.icmp "ult" %39, %43 : i64
    %45 = llvm.select %44, %39, %43 : i1, i64
    %46 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%34, %35, %45, %46) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %47 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    llvm.store %47, %9 : !llvm.ptr<struct<(ptr<i8>, f64)>>
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