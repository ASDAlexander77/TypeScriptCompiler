; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%N = type { ptr, ptr }
%V = type { ptr, double }

$V..size = comdat any

$N..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638230081509142225 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFN\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_15562253555735830060 = internal constant [116 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00catch_finally_locals_own.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@m_13957468038051399532 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 16, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V..rtti, ptr @V..size }
@N..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638230081509142225, i64 8)
@N..size = global i64 16, comdat
@N..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @N..instanceOf, ptr @N..new, ptr @N..rtti, ptr @N..size }
@G = private global ptr undef
@N.IY..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_8699907504592634356_class, i64 40), ptr @N.IY..vtbl..instanceOf.dispatch, ptr inttoptr (i64 8 to ptr) }
@td_8699907504592634356_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_8699907504592634356, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@IY.null..vtbl = private global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr null }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_14389619779565911031 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @G__cctor, ptr null }]

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__cxa_end_catch()

define internal void @tsrel_1752011451674293319(ptr %0) {
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

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

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

define internal void @tsrel_8804135856418482672(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %18

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %18

6:                                                ; preds = %4
  %7 = getelementptr { ptr, i64, i64 }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %2, i32 0, i32 1
  %10 = load i64, ptr %9, align 8
  br label %11

11:                                               ; preds = %14, %6
  %12 = phi i64 [ %16, %14 ], [ 0, %6 ]
  %13 = icmp slt i64 %12, %10
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  %15 = getelementptr ptr, ptr %8, i64 %12
  call void @tsrel_1375346448667278364(ptr %15)
  %16 = add i64 %12, 1
  br label %11

17:                                               ; preds = %11
  call void @__tslang_free_block(ptr %8)
  call void @__tslang_free_block(ptr %2)
  br label %18

18:                                               ; preds = %17, %4, %1
  ret void
}

define internal void @tsrel_8699907504592634356(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %N, ptr %2, i32 0, i32 1
  call void @tsrel_8804135856418482672(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @G__cctor() {
  %1 = call ptr @mk(double 1.000000e+00)
  store ptr %1, ptr @G, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @V..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @V..new() #2 {
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
define void @V.constructor(ptr %0, double %1) #2 {
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
define i1 @N..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @N..new() #2 {
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
define void @N.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %V, ptr %8, i32 0, i32 0
  store ptr @V..vtbl, ptr %9, align 8
  %10 = load ptr, ptr %3, align 8
  %11 = getelementptr %V, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  call void @V.constructor(ptr %8, double %12)
  %13 = call ptr @GC_malloc(i64 16)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr ptr, ptr %14, i64 0
  store ptr %8, ptr %15, align 8
  %16 = call ptr @GC_malloc(i64 32)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 0
  store ptr %14, ptr %18, align 8
  %19 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 1
  store i64 1, ptr %19, align 8
  %20 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 2
  store i64 1, ptr %20, align 8
  store ptr %17, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress noinline
define ptr @mk(double %0) #3 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  %4 = alloca <{ i64, %V }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %V zeroinitializer, ptr %5, align 8
  store double %0, ptr %2, align 8
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %N, ptr %7, i32 0, i32 0
  store ptr @N..vtbl, ptr %8, align 8
  store %V zeroinitializer, ptr %5, align 8
  %9 = getelementptr %V, ptr %5, i32 0, i32 0
  store ptr @V..vtbl, ptr %9, align 8
  %10 = load double, ptr %2, align 8
  call void @V.constructor(ptr %5, double %10)
  call void @N.constructor(ptr %7, ptr %5)
  store ptr %7, ptr %3, align 8
  %11 = load ptr, ptr %3, align 8
  ret ptr %11
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, %V }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %V zeroinitializer, ptr %4, align 8
  %5 = call ptr @GC_malloc(i64 8)
  %6 = getelementptr i8, ptr %5, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @a_0_14389619779565911031, i64 0, i1 false)
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  store ptr %6, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 1
  store i64 0, ptr %10, align 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 2
  store i64 0, ptr %11, align 8
  store ptr %8, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %12

12:                                               ; preds = %64, %0
  %13 = load i32, ptr %1, align 4
  %14 = icmp slt i32 %13, 64
  br i1 %14, label %15, label %68

15:                                               ; preds = %12
  %16 = call ptr @GC_malloc(i64 24)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr %N, ptr %17, i32 0, i32 0
  store ptr @N..vtbl, ptr %18, align 8
  store %V zeroinitializer, ptr %4, align 8
  %19 = getelementptr %V, ptr %4, i32 0, i32 0
  store ptr @V..vtbl, ptr %19, align 8
  call void @V.constructor(ptr %4, double 7.770000e+02)
  call void @N.constructor(ptr %17, ptr %4)
  %20 = load ptr, ptr %2, align 8
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %25

22:                                               ; preds = %15
  %23 = call ptr @GC_malloc(i64 32)
  %24 = getelementptr i8, ptr %23, i64 8
  store ptr %24, ptr %2, align 8
  br label %26

25:                                               ; preds = %15
  br label %26

26:                                               ; preds = %22, %25
  %27 = phi ptr [ %20, %25 ], [ %24, %22 ]
  br label %28

28:                                               ; preds = %26
  %29 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 1
  %30 = load i64, ptr %29, align 8
  %31 = add i64 %30, 1
  %32 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 0
  %33 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 2
  %34 = load ptr, ptr %32, align 8
  %35 = load i64, ptr %33, align 8
  %36 = icmp ugt i64 %31, %35
  br i1 %36, label %37, label %61

37:                                               ; preds = %28
  %38 = mul i64 %35, 2
  %39 = icmp uge i64 %38, %31
  %40 = select i1 %39, i64 %38, i64 %31
  %41 = icmp ult i64 %40, 4
  %42 = select i1 %41, i64 4, i64 %40
  %43 = icmp ule i64 %31, 4294967295
  br i1 %43, label %44, label %70

44:                                               ; preds = %37
  %45 = icmp ule i64 %42, 2305843009213693951
  br i1 %45, label %46, label %72

46:                                               ; preds = %44
  %47 = mul i64 8, %42
  %48 = add i64 %47, 8
  %49 = icmp eq ptr %34, null
  %50 = getelementptr i8, ptr %34, i64 -8
  %51 = select i1 %49, ptr null, ptr %50
  %52 = call ptr @GC_realloc(ptr %51, i64 %48)
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = icmp ne ptr %53, null
  %55 = icmp eq i64 %47, 0
  %56 = or i1 %54, %55
  br i1 %56, label %57, label %74

57:                                               ; preds = %46
  %58 = mul i64 8, %35
  %59 = getelementptr i8, ptr %53, i64 %58
  %60 = sub i64 %47, %58
  call void @llvm.memset.p0.i64(ptr %59, i8 0, i64 %60, i1 false)
  store ptr %53, ptr %32, align 8
  store i64 %42, ptr %33, align 8
  br label %62

61:                                               ; preds = %28
  br label %62

62:                                               ; preds = %57, %61
  %63 = phi ptr [ %34, %61 ], [ %53, %57 ]
  br label %64

64:                                               ; preds = %62
  %65 = getelementptr ptr, ptr %63, i64 %30
  store ptr %17, ptr %65, align 8
  store i64 %31, ptr %29, align 8
  %66 = load i32, ptr %1, align 4
  %67 = add i32 %66, 1
  store i32 %67, ptr %1, align 4
  br label %12

68:                                               ; preds = %12
  ret void

69:                                               ; preds = %74, %72, %70
  unreachable

70:                                               ; preds = %37
  %71 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15562253555735830060, i64 8), i32 27, ptr null)
  br label %69

72:                                               ; preds = %44
  %73 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15562253555735830060, i64 8), i32 27, ptr null)
  br label %69

74:                                               ; preds = %46
  %75 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15562253555735830060, i64 8), i32 27, ptr null)
  br label %69
}

; Function Attrs: mustprogress
define internal i1 @N.IY..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  %4 = getelementptr %N, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = call i1 %7(ptr %0, ptr %1)
  store i1 %8, ptr %3, align 1
  %9 = load i1, ptr %3, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define double @inFinally() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = load ptr, ptr @G, align 8
  %3 = ptrtoint ptr %2 to i64
  %4 = icmp ne i64 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %0
  br label %7

6:                                                ; preds = %0
  br label %7

7:                                                ; preds = %5, %6
  %8 = phi ptr [ @IY.null..vtbl, %6 ], [ @N.IY..vtbl, %5 ]
  %9 = insertvalue { ptr, ptr } undef, ptr %8, 0
  %10 = insertvalue { ptr, ptr } %9, ptr %2, 1
  %11 = call ptr @mk(double 5.000000e+00)
  store ptr %11, ptr @G, align 8
  call void @churn()
  %12 = extractvalue { ptr, ptr } %10, 0
  %13 = extractvalue { ptr, ptr } %10, 1
  %14 = getelementptr ptr, ptr %12, i32 2
  %15 = load ptr, ptr %14, align 8
  %16 = ptrtoint ptr %13 to i64
  %17 = ptrtoint ptr %15 to i64
  %18 = add i64 %16, %17
  %19 = inttoptr i64 %18 to ptr
  %20 = extractvalue { ptr, ptr } %10, 0
  %21 = extractvalue { ptr, ptr } %10, 1
  %22 = getelementptr ptr, ptr %20, i32 2
  %23 = load ptr, ptr %22, align 8
  %24 = ptrtoint ptr %23 to i64
  %25 = and i64 %24, -9223372036854775808
  %26 = icmp ne i64 %25, 0
  br i1 %26, label %27, label %33

27:                                               ; preds = %7
  %28 = sub i64 %24, -9223372036854775808
  %29 = inttoptr i64 %28 to ptr
  %30 = getelementptr { ptr, ptr }, ptr %29, i32 0, i32 0
  %31 = load ptr, ptr %30, align 8
  %32 = call ptr %31(ptr %21)
  br label %35

33:                                               ; preds = %7
  %34 = load ptr, ptr %19, align 8
  br label %35

35:                                               ; preds = %27, %33
  %36 = phi ptr [ %34, %33 ], [ %32, %27 ]
  br label %37

37:                                               ; preds = %35
  %38 = icmp eq ptr %36, null
  %39 = select i1 %38, ptr @__ts_empty_array, ptr %36
  %40 = getelementptr { ptr, i64, i64 }, ptr %39, i32 0, i32 0
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr ptr, ptr %41, i32 0
  %43 = load ptr, ptr %42, align 8
  %44 = getelementptr %V, ptr %43, i32 0, i32 1
  %45 = load double, ptr %44, align 8
  store double %45, ptr %1, align 8
  %46 = load double, ptr %1, align 8
  ret double %46
}

; Function Attrs: mustprogress
define double @inCatch() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %157 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr null
  %7 = extractvalue { ptr, i32 } %6, 0
  %8 = call ptr @__cxa_begin_catch(ptr %7)
  %9 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %5
  %10 = icmp eq ptr %9, @_ZTIi
  br i1 %10, label %11, label %18

11:                                               ; preds = %.noexc
  %12 = load i32, ptr %8, align 4
  %13 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %11
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr { i64, ptr, i32 }, ptr %14, i32 0, i32 0
  store i64 4, ptr %15, align 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %14, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %14, i32 0, i32 2
  store i32 %12, ptr %17, align 4
  br label %141

18:                                               ; preds = %.noexc
  %19 = icmp eq ptr %9, @_ZTIj
  br i1 %19, label %20, label %27

20:                                               ; preds = %18
  %21 = load i32, ptr %8, align 4
  %22 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %20
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 0
  store i64 4, ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 2
  store i32 %21, ptr %26, align 4
  br label %138

27:                                               ; preds = %18
  %28 = icmp eq ptr %9, @_ZTId
  br i1 %28, label %29, label %36

29:                                               ; preds = %27
  %30 = load double, ptr %8, align 8
  %31 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %29
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr { i64, ptr, double }, ptr %32, i32 0, i32 0
  store i64 8, ptr %33, align 8
  %34 = getelementptr { i64, ptr, double }, ptr %32, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %32, i32 0, i32 2
  store double %30, ptr %35, align 8
  br label %135

36:                                               ; preds = %27
  %37 = icmp eq ptr %9, @_ZTIf
  br i1 %37, label %38, label %45

38:                                               ; preds = %36
  %39 = load float, ptr %8, align 4
  %40 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %38
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr { i64, ptr, float }, ptr %41, i32 0, i32 0
  store i64 4, ptr %42, align 8
  %43 = getelementptr { i64, ptr, float }, ptr %41, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %41, i32 0, i32 2
  store float %39, ptr %44, align 4
  br label %132

45:                                               ; preds = %36
  %46 = icmp eq ptr %9, @_ZTIb
  br i1 %46, label %47, label %54

47:                                               ; preds = %45
  %48 = load i1, ptr %8, align 1
  %49 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %47
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr { i64, ptr, i1 }, ptr %50, i32 0, i32 0
  store i64 1, ptr %51, align 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %50, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %50, i32 0, i32 2
  store i1 %48, ptr %53, align 1
  br label %129

54:                                               ; preds = %45
  %55 = icmp eq ptr %9, @_ZTIx
  br i1 %55, label %56, label %63

56:                                               ; preds = %54
  %57 = load i64, ptr %8, align 8
  %58 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %56
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr { i64, ptr, i64 }, ptr %59, i32 0, i32 0
  store i64 8, ptr %60, align 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %59, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %59, i32 0, i32 2
  store i64 %57, ptr %62, align 8
  br label %126

63:                                               ; preds = %54
  %64 = icmp eq ptr %9, @_ZTIPc
  br i1 %64, label %65, label %72

65:                                               ; preds = %63
  %66 = load ptr, ptr %8, align 8
  %67 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %65
  %68 = getelementptr i8, ptr %67, i64 8
  %69 = getelementptr { i64, ptr, ptr }, ptr %68, i32 0, i32 0
  store i64 8, ptr %69, align 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %68, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %68, i32 0, i32 2
  store ptr %66, ptr %71, align 8
  br label %123

72:                                               ; preds = %63
  %73 = icmp eq ptr %9, @_ZTIPv
  br i1 %73, label %74, label %76

74:                                               ; preds = %72
  %75 = load ptr, ptr %8, align 8
  br label %120

76:                                               ; preds = %72
  %77 = load ptr, ptr %9, align 8
  %78 = icmp eq ptr %77, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %78, label %79, label %110

79:                                               ; preds = %76
  %80 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %9, i32 0, i32 2
  %81 = load i32, ptr %80, align 4
  %82 = and i32 %81, 1073741824
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %100

84:                                               ; preds = %79
  %85 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %9, i32 0, i32 4
  %86 = load ptr, ptr %85, align 8
  %87 = icmp ne ptr %86, null
  br i1 %87, label %88, label %90

88:                                               ; preds = %84
  invoke void %86(ptr %1, ptr %8)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %88
  %89 = load ptr, ptr %1, align 8
  br label %97

90:                                               ; preds = %84
  %91 = load ptr, ptr %8, align 8
  %92 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %90
  %93 = getelementptr i8, ptr %92, i64 8
  %94 = getelementptr { i64, ptr, ptr }, ptr %93, i32 0, i32 0
  store i64 8, ptr %94, align 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %93, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %93, i32 0, i32 2
  store ptr %91, ptr %96, align 8
  br label %97

97:                                               ; preds = %.noexc8, %.noexc9
  %98 = phi ptr [ %93, %.noexc9 ], [ %89, %.noexc8 ]
  br label %99

99:                                               ; preds = %97
  br label %107

100:                                              ; preds = %79
  %101 = load ptr, ptr %8, align 8
  %102 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %100
  %103 = getelementptr i8, ptr %102, i64 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 0
  store i64 8, ptr %104, align 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 2
  store ptr %101, ptr %106, align 8
  br label %107

107:                                              ; preds = %99, %.noexc13
  %108 = phi ptr [ %103, %.noexc13 ], [ %98, %99 ]
  br label %109

109:                                              ; preds = %107
  br label %117

110:                                              ; preds = %76
  %111 = load ptr, ptr %8, align 8
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 0
  store i64 8, ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 2
  store ptr %111, ptr %116, align 8
  br label %117

117:                                              ; preds = %109, %.noexc1
  %118 = phi ptr [ %113, %.noexc1 ], [ %108, %109 ]
  br label %119

119:                                              ; preds = %117
  br label %120

120:                                              ; preds = %74, %119
  %121 = phi ptr [ %118, %119 ], [ %75, %74 ]
  br label %122

122:                                              ; preds = %120
  br label %123

123:                                              ; preds = %.noexc7, %122
  %124 = phi ptr [ %121, %122 ], [ %68, %.noexc7 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %.noexc12, %125
  %127 = phi ptr [ %124, %125 ], [ %59, %.noexc12 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc2, %128
  %130 = phi ptr [ %127, %128 ], [ %50, %.noexc2 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc4, %131
  %133 = phi ptr [ %130, %131 ], [ %41, %.noexc4 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc6, %134
  %136 = phi ptr [ %133, %134 ], [ %32, %.noexc6 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc5, %137
  %139 = phi ptr [ %136, %137 ], [ %23, %.noexc5 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc3, %140
  %142 = phi ptr [ %139, %140 ], [ %14, %.noexc3 ]
  br label %143

143:                                              ; preds = %141
  store ptr %142, ptr %2, align 8
  %144 = load ptr, ptr @G, align 8
  %145 = invoke ptr @mk(double 6.000000e+00)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %143
  store ptr %145, ptr @G, align 8
  invoke void @churn()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  %146 = getelementptr %N, ptr %144, i32 0, i32 1
  %147 = load ptr, ptr %146, align 8
  %148 = icmp eq ptr %147, null
  %149 = select i1 %148, ptr @__ts_empty_array, ptr %147
  %150 = getelementptr { ptr, i64, i64 }, ptr %149, i32 0, i32 0
  %151 = load ptr, ptr %150, align 8
  %152 = getelementptr ptr, ptr %151, i32 0
  %153 = load ptr, ptr %152, align 8
  %154 = getelementptr %V, ptr %153, i32 0, i32 1
  %155 = load double, ptr %154, align 8
  store double %155, ptr %3, align 8
  call void @__cxa_end_catch()
  %156 = load double, ptr %3, align 8
  ret double %156

157:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %100, %56, %.noexc10, %143, %90, %88, %65, %29, %20, %38, %11, %47, %110, %5
  %158 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %158
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  br label %2

2:                                                ; preds = %13, %0
  %3 = load i32, ptr %1, align 4
  %4 = icmp slt i32 %3, 100
  br i1 %4, label %5, label %16

5:                                                ; preds = %2
  %6 = call ptr @mk(double 1.000000e+00)
  store ptr %6, ptr @G, align 8
  %7 = call double @inFinally()
  %8 = fcmp oeq double %7, 1.000000e+00
  br i1 %8, label %9, label %18

9:                                                ; preds = %5
  %10 = call ptr @mk(double 2.000000e+00)
  store ptr %10, ptr @G, align 8
  %11 = call double @inCatch()
  %12 = fcmp oeq double %11, 2.000000e+00
  br i1 %12, label %13, label %20

13:                                               ; preds = %9
  %14 = load i32, ptr %1, align 4
  %15 = add i32 %14, 1
  store i32 %15, ptr %1, align 4
  br label %2

16:                                               ; preds = %2
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

17:                                               ; preds = %20, %18
  unreachable

18:                                               ; preds = %5
  %19 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18175057562723240456, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15562253555735830060, i64 8), i32 57, ptr null)
  br label %17

20:                                               ; preds = %9
  %21 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13957468038051399532, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15562253555735830060, i64 8), i32 59, ptr null)
  br label %17
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress noinline }
attributes #4 = { mustprogress "personality" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

