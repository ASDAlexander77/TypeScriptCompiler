; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, double }
%N = type { ptr, ptr }

$V..size = comdat any

$N..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638230081509142225 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFN\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 16, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V..rtti, ptr @V..size }
@N..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638230081509142225, i64 8)
@N..size = global i64 16, comdat
@N..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @N..instanceOf, ptr @N..new, ptr @N..rtti, ptr @N..size }
@IV.12252034448159921135..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12252034448159921135_object, i64 40), ptr @.instanceOf.none, ptr null }
@td_12252034448159921135_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrel_12252034448159921135, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

define internal void @__tslang_inc_ref(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %13

3:                                                ; preds = %1
  %4 = getelementptr i8, ptr %0, i64 -8
  %5 = load i64, ptr %4, align 8
  %6 = icmp ne i64 %5, -1
  %7 = icmp sge i64 %5, 0
  br i1 %7, label %11, label %8

8:                                                ; preds = %3
  br i1 %6, label %9, label %13

9:                                                ; preds = %8
  %10 = atomicrmw add ptr %4, i64 1 monotonic, align 8
  br label %13

11:                                               ; preds = %3
  %12 = add i64 %5, 1
  store i64 %12, ptr %4, align 8
  br label %13

13:                                               ; preds = %11, %9, %8, %1
  ret void
}

define internal void @__tslang_retain_block_slot(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal i1 @__tslang_dec_ref(ptr %0) {
  %2 = getelementptr i8, ptr %0, i64 -8
  %3 = load i64, ptr %2, align 8
  %4 = icmp ne i64 %3, -1
  %5 = icmp sge i64 %3, 0
  br i1 %5, label %11, label %6

6:                                                ; preds = %1
  br i1 %4, label %7, label %15

7:                                                ; preds = %6
  %8 = atomicrmw sub ptr %2, i64 1 release, align 8
  %9 = and i64 %8, 9223372036854775807
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %14, label %15

11:                                               ; preds = %1
  %12 = sub i64 %3, 1
  store i64 %12, ptr %2, align 8
  %13 = icmp eq i64 %12, 0
  ret i1 %13

14:                                               ; preds = %7
  fence acquire
  ret i1 true

15:                                               ; preds = %7, %6
  ret i1 false
}

declare void @GC_free(ptr)

define internal void @__tslang_free_block(ptr %0) {
  %2 = icmp eq ptr %0, null
  %3 = getelementptr i8, ptr %0, i64 -8
  %4 = select i1 %2, ptr null, ptr %3
  call void @GC_free(ptr %4)
  ret void
}

define internal void @tsrel_1375346448667278364(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  call void @__tslang_free_block(ptr %2)
  br label %7

7:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrel_12252034448159921135(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_1375346448667278364(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

; Function Attrs: mustprogress
define i1 @V..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @V..rtti, align 8
  %7 = ptrtoint ptr %5 to i64
  %8 = ptrtoint ptr %6 to i64
  %9 = icmp ne i64 %7, 0
  %10 = icmp ne i64 %8, 0
  %11 = zext i1 %9 to i32
  %12 = zext i1 %10 to i32
  %13 = and i32 %11, %12
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %18

15:                                               ; preds = %2
  %16 = call i32 @strcmp(ptr %5, ptr %6)
  %17 = icmp eq i32 %16, 0
  br label %20

18:                                               ; preds = %2
  %19 = icmp eq i64 %7, %8
  br label %20

20:                                               ; preds = %15, %18
  %21 = phi i1 [ %19, %18 ], [ %17, %15 ]
  br label %22

22:                                               ; preds = %20
  store i1 %21, ptr %4, align 1
  %23 = load i1, ptr %4, align 1
  ret i1 %23
}

; Function Attrs: mustprogress
define ptr @V..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %V, ptr %3, i32 0, i32 0
  store ptr @V..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @V.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %V, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @N..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @N..rtti, align 8
  %7 = ptrtoint ptr %5 to i64
  %8 = ptrtoint ptr %6 to i64
  %9 = icmp ne i64 %7, 0
  %10 = icmp ne i64 %8, 0
  %11 = zext i1 %9 to i32
  %12 = zext i1 %10 to i32
  %13 = and i32 %11, %12
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %18

15:                                               ; preds = %2
  %16 = call i32 @strcmp(ptr %5, ptr %6)
  %17 = icmp eq i32 %16, 0
  br label %20

18:                                               ; preds = %2
  %19 = icmp eq i64 %7, %8
  br label %20

20:                                               ; preds = %15, %18
  %21 = phi i1 [ %19, %18 ], [ %17, %15 ]
  br label %22

22:                                               ; preds = %20
  store i1 %21, ptr %4, align 1
  %23 = load i1, ptr %4, align 1
  ret i1 %23
}

; Function Attrs: mustprogress
define ptr @N..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %N, ptr %3, i32 0, i32 0
  store ptr @N..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @N.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %V, ptr %8, i32 0, i32 0
  store ptr @V..vtbl, ptr %9, align 8
  %10 = load double, ptr %3, align 8
  call void @V.constructor(ptr %8, double %10)
  %11 = call ptr @GC_malloc(i64 16)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr ptr, ptr %12, i64 0
  store ptr %8, ptr %13, align 8
  %14 = call ptr @GC_malloc(i64 32)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { ptr, i64, i64 }, ptr %15, i32 0, i32 0
  store ptr %12, ptr %16, align 8
  %17 = getelementptr { ptr, i64, i64 }, ptr %15, i32 0, i32 1
  store i64 1, ptr %17, align 8
  %18 = getelementptr { ptr, i64, i64 }, ptr %15, i32 0, i32 2
  store i64 1, ptr %18, align 8
  store ptr %15, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @mk(double %0) #1 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 0
  store ptr @N..vtbl, ptr %6, align 8
  %7 = load double, ptr %2, align 8
  call void @N.constructor(ptr %5, double %7)
  store ptr %5, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define double @read({ ptr, ptr } %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca { ptr, ptr }, align 8
  %5 = alloca double, align 8
  store { ptr, ptr } %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load { ptr, ptr }, ptr %4, align 8
  %7 = extractvalue { ptr, ptr } %6, 0
  %8 = extractvalue { ptr, ptr } %6, 1
  %9 = getelementptr ptr, ptr %7, i32 2
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %8 to i64
  %12 = ptrtoint ptr %10 to i64
  %13 = add i64 %11, %12
  %14 = inttoptr i64 %13 to ptr
  %15 = extractvalue { ptr, ptr } %6, 0
  %16 = extractvalue { ptr, ptr } %6, 1
  %17 = getelementptr ptr, ptr %15, i32 2
  %18 = load ptr, ptr %17, align 8
  %19 = ptrtoint ptr %18 to i64
  %20 = and i64 %19, -9223372036854775808
  %21 = icmp ne i64 %20, 0
  br i1 %21, label %22, label %28

22:                                               ; preds = %2
  %23 = sub i64 %19, -9223372036854775808
  %24 = inttoptr i64 %23 to ptr
  %25 = getelementptr { ptr, ptr }, ptr %24, i32 0, i32 0
  %26 = load ptr, ptr %25, align 8
  %27 = call ptr %26(ptr %16)
  br label %30

28:                                               ; preds = %2
  %29 = load ptr, ptr %14, align 8
  br label %30

30:                                               ; preds = %22, %28
  %31 = phi ptr [ %29, %28 ], [ %27, %22 ]
  br label %32

32:                                               ; preds = %30
  %33 = getelementptr %V, ptr %31, i32 0, i32 1
  %34 = load double, ptr %33, align 8
  %35 = load ptr, ptr %3, align 8
  %36 = getelementptr %N, ptr %35, i32 0, i32 1
  %37 = load ptr, ptr %36, align 8
  %38 = icmp eq ptr %37, null
  %39 = select i1 %38, ptr @__ts_empty_array, ptr %37
  %40 = getelementptr { ptr, i64, i64 }, ptr %39, i32 0, i32 0
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr ptr, ptr %41, i32 0
  %43 = load ptr, ptr %42, align 8
  %44 = getelementptr %V, ptr %43, i32 0, i32 1
  %45 = load double, ptr %44, align 8
  %46 = fadd double %34, %45
  store double %46, ptr %5, align 8
  %47 = load double, ptr %5, align 8
  ret double %47
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca { ptr }, align 8
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = call ptr @mk(double 7.000000e+00)
  store ptr %4, ptr %3, align 8
  store i32 0, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = load ptr, ptr %3, align 8
  %7 = getelementptr %N, ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %7, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr @__ts_empty_array, ptr %8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr ptr, ptr %12, i32 0
  %14 = load ptr, ptr %13, align 8
  store { ptr } zeroinitializer, ptr %1, align 8
  %15 = getelementptr { ptr }, ptr %1, i32 0, i32 0
  store ptr %14, ptr %15, align 8
  %16 = load { ptr }, ptr %1, align 8
  %17 = invoke ptr @mk(double 1.000000e+00)
          to label %18 unwind label %35

18:                                               ; preds = %0
  store ptr %17, ptr %3, align 8
  %19 = load ptr, ptr %3, align 8
  %20 = call ptr @GC_malloc(i64 16)
  %21 = getelementptr i8, ptr %20, i64 8
  store { ptr } %16, ptr %21, align 8
  %22 = insertvalue { ptr, ptr } { ptr @IV.12252034448159921135..vtbl, ptr undef }, ptr %21, 1
  %23 = invoke double @read({ ptr, ptr } %22, ptr %19)
          to label %24 unwind label %35

24:                                               ; preds = %18
  %25 = sitofp i32 %5 to double
  %26 = fadd double %25, %23
  %27 = fptosi double %26 to i64
  %28 = trunc i64 %27 to i32
  store i32 %28, ptr %2, align 4
  %29 = load i32, ptr %2, align 4
  %30 = call ptr @GC_malloc_atomic(i64 58)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %31, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %29)
  %33 = icmp eq ptr %31, null
  %34 = select i1 %33, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %31
  call void @puts(ptr %34)
  ret i32 0

35:                                               ; preds = %18, %0
  %36 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %36
}

declare void @mlirAsyncRuntimeAddRef(ptr, i64)

declare void @mlirAsyncRuntimeDropRef(ptr, i64)

declare ptr @mlirAsyncRuntimeCreateToken()

declare ptr @mlirAsyncRuntimeCreateValue(i64)

declare ptr @mlirAsyncRuntimeCreateGroup(i64)

declare void @mlirAsyncRuntimeEmplaceToken(ptr)

declare void @mlirAsyncRuntimeEmplaceValue(ptr)

declare void @mlirAsyncRuntimeSetTokenError(ptr)

declare void @mlirAsyncRuntimeSetValueError(ptr)

declare i1 @mlirAsyncRuntimeIsTokenError(ptr)

declare i1 @mlirAsyncRuntimeIsValueError(ptr)

declare i1 @mlirAsyncRuntimeIsGroupError(ptr)

declare void @mlirAsyncRuntimeAwaitToken(ptr)

declare void @mlirAsyncRuntimeAwaitValue(ptr)

declare void @mlirAsyncRuntimeAwaitAllInGroup(ptr)

declare void @mlirAsyncRuntimeExecute(ptr, ptr)

declare ptr @mlirAsyncRuntimeGetValueStorage(ptr)

declare i64 @mlirAsyncRuntimeAddTokenToGroup(ptr, ptr)

declare void @mlirAsyncRuntimeAwaitTokenAndExecute(ptr, ptr, ptr)

declare void @mlirAsyncRuntimeAwaitValueAndExecute(ptr, ptr, ptr)

declare void @mlirAsyncRuntimeAwaitAllInGroupAndExecute(ptr, ptr, ptr)

declare i64 @mlirAsyncRuntimGetNumWorkerThreads()

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

