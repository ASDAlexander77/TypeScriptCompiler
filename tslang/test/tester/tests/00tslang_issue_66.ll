; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Box = type { ptr, i32 }
%Cam = type { ptr, ptr, ptr }

$Box..size = comdat any

$Cam..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_1609559789082440018 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBox\00", align 8
@s_859077132320343308 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCam\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_378607937501964222 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00tslang_issue_66.ts\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Box..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1609559789082440018, i64 8)
@Box..size = global i64 16, comdat
@Box..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Box..instanceOf, ptr @Box..new, ptr @Box..rtti, ptr @Box..size }
@Cam..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_859077132320343308, i64 8)
@Cam..size = global i64 24, comdat
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@Cam..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Cam..instanceOf, ptr @Cam..new, ptr @Cam.loop, ptr @Cam.inner, ptr @Cam.boxed, ptr @Cam.thrown, ptr @Cam..rtti, ptr @Cam..size }
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

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

declare i32 @strcmp(ptr, ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

; Function Attrs: mustprogress
define ptr @rot(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr null, ptr %5, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr @__ts_empty_array, ptr %6
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr double, ptr %10, i32 0
  %12 = load double, ptr %11, align 8
  %13 = load double, ptr %3, align 8
  %14 = fmul double %12, %13
  %15 = load ptr, ptr %4, align 8
  %16 = icmp eq ptr %15, null
  %17 = select i1 %16, ptr @__ts_empty_array, ptr %15
  %18 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 0
  %19 = load ptr, ptr %18, align 8
  %20 = getelementptr double, ptr %19, i32 1
  %21 = load double, ptr %20, align 8
  %22 = load ptr, ptr %4, align 8
  %23 = icmp eq ptr %22, null
  %24 = select i1 %23, ptr @__ts_empty_array, ptr %22
  %25 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 0
  %26 = load ptr, ptr %25, align 8
  %27 = getelementptr double, ptr %26, i32 2
  %28 = load double, ptr %27, align 8
  %29 = call ptr @GC_malloc(i64 32)
  %30 = getelementptr i8, ptr %29, i64 8
  %31 = getelementptr double, ptr %30, i64 0
  store double %14, ptr %31, align 8
  %32 = getelementptr double, ptr %30, i64 1
  store double %21, ptr %32, align 8
  %33 = getelementptr double, ptr %30, i64 2
  store double %28, ptr %33, align 8
  %34 = call ptr @GC_malloc(i64 32)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr { ptr, i64, i64 }, ptr %35, i32 0, i32 0
  store ptr %30, ptr %36, align 8
  %37 = getelementptr { ptr, i64, i64 }, ptr %35, i32 0, i32 1
  store i64 3, ptr %37, align 8
  %38 = getelementptr { ptr, i64, i64 }, ptr %35, i32 0, i32 2
  store i64 3, ptr %38, align 8
  store ptr %35, ptr %5, align 8
  %39 = load ptr, ptr %5, align 8
  ret ptr %39
}

; Function Attrs: mustprogress
define i1 @Box..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Box..rtti, align 8
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
define ptr @Box..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Box, ptr %3, i32 0, i32 0
  store ptr @Box..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Box.constructor(ptr %0, i32 %1) #1 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %3, align 4
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %3, align 4
  store i32 %7, ptr %6, align 4
  ret void
}

; Function Attrs: mustprogress
define ptr @grow(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 0
  store ptr @Box..vtbl, ptr %6, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = add i32 %9, 1
  call void @Box.constructor(ptr %5, i32 %10)
  store ptr %5, ptr %3, align 8
  %11 = load ptr, ptr %3, align 8
  ret ptr %11
}

; Function Attrs: mustprogress
define void @Cam.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Cam, ptr %3, i32 0, i32 1
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr double, ptr %6, i64 0
  store double 1.000000e+00, ptr %7, align 8
  %8 = getelementptr double, ptr %6, i64 1
  store double 0.000000e+00, ptr %8, align 8
  %9 = getelementptr double, ptr %6, i64 2
  store double 1.000000e+00, ptr %9, align 8
  %10 = call ptr @GC_malloc(i64 32)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %11, i32 0, i32 0
  store ptr %6, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %11, i32 0, i32 1
  store i64 3, ptr %13, align 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %11, i32 0, i32 2
  store i64 3, ptr %14, align 8
  store ptr %11, ptr %4, align 8
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr %Cam, ptr %15, i32 0, i32 2
  %17 = call ptr @GC_malloc(i64 24)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr %Box, ptr %18, i32 0, i32 0
  store ptr @Box..vtbl, ptr %19, align 8
  call void @Box.constructor(ptr %18, i32 1)
  store ptr %18, ptr %16, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Cam..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Cam..rtti, align 8
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
define ptr @Cam..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Cam, ptr %3, i32 0, i32 0
  store ptr @Cam..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Cam.loop(ptr %0, i32 %1) #1 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store i32 %1, ptr %5, align 4
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %Cam, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  store ptr %9, ptr %4, align 8
  store i32 0, ptr %3, align 4
  br label %10

10:                                               ; preds = %14, %2
  %11 = load i32, ptr %3, align 4
  %12 = load i32, ptr %5, align 4
  %13 = icmp slt i32 %11, %12
  br i1 %13, label %14, label %19

14:                                               ; preds = %10
  %15 = load ptr, ptr %4, align 8
  %16 = call ptr @rot(ptr %15, double 2.000000e+00)
  store ptr %16, ptr %4, align 8
  %17 = load i32, ptr %3, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %3, align 4
  br label %10

19:                                               ; preds = %10
  %20 = load ptr, ptr %6, align 8
  %21 = getelementptr %Cam, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %4, align 8
  %23 = call ptr @rot(ptr %22, double 1.000000e+00)
  store ptr %23, ptr %21, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Cam.inner(ptr %0, i32 %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca double, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  %8 = alloca double, align 8
  store ptr %0, ptr %7, align 8
  store i32 %1, ptr %6, align 4
  store double 0.000000e+00, ptr %5, align 8
  store i32 0, ptr %4, align 4
  br label %9

9:                                                ; preds = %23, %2
  %10 = load i32, ptr %4, align 4
  %11 = load i32, ptr %6, align 4
  %12 = icmp slt i32 %10, %11
  br i1 %12, label %13, label %35

13:                                               ; preds = %9
  %14 = load ptr, ptr %7, align 8
  %15 = getelementptr %Cam, ptr %14, i32 0, i32 1
  %16 = load ptr, ptr %15, align 8
  store ptr %16, ptr %3, align 8
  %17 = load i32, ptr %4, align 4
  %18 = srem i32 %17, 2
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %23

20:                                               ; preds = %13
  %21 = load ptr, ptr %3, align 8
  %22 = call ptr @rot(ptr %21, double 3.000000e+00)
  store ptr %22, ptr %3, align 8
  br label %23

23:                                               ; preds = %20, %13
  %24 = load double, ptr %5, align 8
  %25 = load ptr, ptr %3, align 8
  %26 = icmp eq ptr %25, null
  %27 = select i1 %26, ptr @__ts_empty_array, ptr %25
  %28 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 0
  %29 = load ptr, ptr %28, align 8
  %30 = getelementptr double, ptr %29, i32 0
  %31 = load double, ptr %30, align 8
  %32 = fadd double %24, %31
  store double %32, ptr %5, align 8
  %33 = load i32, ptr %4, align 4
  %34 = add i32 %33, 1
  store i32 %34, ptr %4, align 4
  br label %9

35:                                               ; preds = %9
  %36 = load double, ptr %5, align 8
  store double %36, ptr %8, align 8
  %37 = load double, ptr %8, align 8
  ret double %37
}

; Function Attrs: mustprogress
define i32 @Cam.boxed(ptr %0, i1 %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store i1 %1, ptr %4, align 1
  %7 = load ptr, ptr %5, align 8
  %8 = getelementptr %Cam, ptr %7, i32 0, i32 2
  %9 = load ptr, ptr %8, align 8
  store ptr %9, ptr %3, align 8
  %10 = load i1, ptr %4, align 1
  br i1 %10, label %11, label %14

11:                                               ; preds = %2
  %12 = load ptr, ptr %3, align 8
  %13 = call ptr @grow(ptr %12)
  store ptr %13, ptr %3, align 8
  br label %14

14:                                               ; preds = %11, %2
  %15 = load ptr, ptr %3, align 8
  %16 = getelementptr %Box, ptr %15, i32 0, i32 1
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %6, align 4
  %18 = load i32, ptr %6, align 4
  ret i32 %18
}

; Function Attrs: mustprogress
define i32 @Cam.thrown(ptr %0, i1 %1) #2 personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i1, align 1
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  store ptr %0, ptr %7, align 8
  store i1 %1, ptr %6, align 1
  %9 = load ptr, ptr %7, align 8
  %10 = getelementptr %Cam, ptr %9, i32 0, i32 2
  %11 = load ptr, ptr %10, align 8
  store ptr %11, ptr %5, align 8
  %12 = load i1, ptr %6, align 1
  br i1 %12, label %13, label %161

13:                                               ; preds = %2
  %14 = load ptr, ptr %5, align 8
  %15 = invoke ptr @grow(ptr %14)
          to label %16 unwind label %18

16:                                               ; preds = %13
  store ptr %15, ptr %5, align 8
  %17 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %17, align 8
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIPc, ptr null)
          to label %167 unwind label %18

18:                                               ; preds = %16, %13
  %19 = landingpad { ptr, i32 }
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  %22 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %18
  %23 = icmp eq ptr %22, @_ZTIi
  br i1 %23, label %24, label %31

24:                                               ; preds = %.noexc5
  %25 = load i32, ptr %21, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %154

31:                                               ; preds = %.noexc5
  %32 = icmp eq ptr %22, @_ZTIj
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load i32, ptr %21, align 4
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 0
  store i64 4, ptr %37, align 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 2
  store i32 %34, ptr %39, align 4
  br label %151

40:                                               ; preds = %31
  %41 = icmp eq ptr %22, @_ZTId
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load double, ptr %21, align 8
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 0
  store i64 8, ptr %46, align 8
  %47 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 2
  store double %43, ptr %48, align 8
  br label %148

49:                                               ; preds = %40
  %50 = icmp eq ptr %22, @_ZTIf
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load float, ptr %21, align 4
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 0
  store i64 4, ptr %55, align 8
  %56 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 2
  store float %52, ptr %57, align 4
  br label %145

58:                                               ; preds = %49
  %59 = icmp eq ptr %22, @_ZTIb
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i1, ptr %21, align 1
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 0
  store i64 1, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 2
  store i1 %61, ptr %66, align 1
  br label %142

67:                                               ; preds = %58
  %68 = icmp eq ptr %22, @_ZTIx
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load i64, ptr %21, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 2
  store i64 %70, ptr %75, align 8
  br label %139

76:                                               ; preds = %67
  %77 = icmp eq ptr %22, @_ZTIPc
  br i1 %77, label %78, label %85

78:                                               ; preds = %76
  %79 = load ptr, ptr %21, align 8
  %80 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %78
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 0
  store i64 8, ptr %82, align 8
  %83 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %83, align 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 2
  store ptr %79, ptr %84, align 8
  br label %136

85:                                               ; preds = %76
  %86 = icmp eq ptr %22, @_ZTIPv
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = load ptr, ptr %21, align 8
  br label %133

89:                                               ; preds = %85
  %90 = load ptr, ptr %22, align 8
  %91 = icmp eq ptr %90, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %91, label %92, label %123

92:                                               ; preds = %89
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 2
  %94 = load i32, ptr %93, align 4
  %95 = and i32 %94, 1073741824
  %96 = icmp ne i32 %95, 0
  br i1 %96, label %97, label %113

97:                                               ; preds = %92
  %98 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 4
  %99 = load ptr, ptr %98, align 8
  %100 = icmp ne ptr %99, null
  br i1 %100, label %101, label %103

101:                                              ; preds = %97
  invoke void %99(ptr %3, ptr %21)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %101
  %102 = load ptr, ptr %3, align 8
  br label %110

103:                                              ; preds = %97
  %104 = load ptr, ptr %21, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %.noexc7, %.noexc
  %111 = phi ptr [ %106, %.noexc ], [ %102, %.noexc7 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %92
  %114 = load ptr, ptr %21, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc1
  %121 = phi ptr [ %116, %.noexc1 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %89
  %124 = load ptr, ptr %21, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc6
  %131 = phi ptr [ %126, %.noexc6 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %87, %132
  %134 = phi ptr [ %131, %132 ], [ %88, %87 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc4, %135
  %137 = phi ptr [ %134, %135 ], [ %81, %.noexc4 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc11, %138
  %140 = phi ptr [ %137, %138 ], [ %72, %.noexc11 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc3, %141
  %143 = phi ptr [ %140, %141 ], [ %63, %.noexc3 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc2, %144
  %146 = phi ptr [ %143, %144 ], [ %54, %.noexc2 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc9, %147
  %149 = phi ptr [ %146, %147 ], [ %45, %.noexc9 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc10, %150
  %152 = phi ptr [ %149, %150 ], [ %36, %.noexc10 ]
  br label %153

153:                                              ; preds = %151
  br label %154

154:                                              ; preds = %.noexc8, %153
  %155 = phi ptr [ %152, %153 ], [ %27, %.noexc8 ]
  br label %156

156:                                              ; preds = %154
  store ptr %155, ptr %4, align 8
  %157 = load ptr, ptr %5, align 8
  %158 = getelementptr %Box, ptr %157, i32 0, i32 1
  %159 = load i32, ptr %158, align 4
  %160 = mul i32 %159, 10
  store i32 %160, ptr %8, align 4
  call void @__cxa_end_catch()
  br label %165

161:                                              ; preds = %2
  %162 = load ptr, ptr %5, align 8
  %163 = getelementptr %Box, ptr %162, i32 0, i32 1
  %164 = load i32, ptr %163, align 4
  store i32 %164, ptr %8, align 4
  br label %165

165:                                              ; preds = %156, %161
  %166 = load i32, ptr %8, align 4
  ret i32 %166

167:                                              ; preds = %16
  unreachable

endcatch.resume:                                  ; preds = %69, %33, %42, %24, %101, %123, %18, %78, %60, %51, %113, %103
  %168 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %168
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca <{ i64, %Cam }>, align 16
  store i64 -1, ptr %1, align 8
  %2 = getelementptr i8, ptr %1, i64 8
  store %Cam zeroinitializer, ptr %2, align 8
  store %Cam zeroinitializer, ptr %2, align 8
  %3 = getelementptr %Cam, ptr %2, i32 0, i32 0
  store ptr @Cam..vtbl, ptr %3, align 8
  call void @Cam.constructor(ptr %2)
  %4 = getelementptr %Cam, ptr %2, i32 0, i32 0
  call void @Cam.loop(ptr %2, i32 3)
  %5 = getelementptr %Cam, ptr %2, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr @__ts_empty_array, ptr %6
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr double, ptr %10, i32 0
  %12 = load double, ptr %11, align 8
  %13 = fcmp oeq double %12, 8.000000e+00
  br i1 %13, label %14, label %40

14:                                               ; preds = %0
  %15 = getelementptr %Cam, ptr %2, i32 0, i32 0
  %16 = call double @Cam.inner(ptr %2, i32 3)
  %17 = fcmp oeq double %16, 5.600000e+01
  br i1 %17, label %18, label %42

18:                                               ; preds = %14
  %19 = getelementptr %Cam, ptr %2, i32 0, i32 0
  %20 = call i32 @Cam.boxed(ptr %2, i1 true)
  %21 = icmp eq i32 %20, 2
  br i1 %21, label %22, label %26

22:                                               ; preds = %18
  %23 = getelementptr %Cam, ptr %2, i32 0, i32 0
  %24 = call i32 @Cam.boxed(ptr %2, i1 false)
  %25 = icmp eq i32 %24, 1
  br label %26

26:                                               ; preds = %22, %18
  %27 = phi i1 [ %25, %22 ], [ %21, %18 ]
  br i1 %27, label %28, label %44

28:                                               ; preds = %26
  %29 = getelementptr %Cam, ptr %2, i32 0, i32 0
  %30 = call i32 @Cam.thrown(ptr %2, i1 true)
  %31 = icmp eq i32 %30, 20
  br i1 %31, label %32, label %36

32:                                               ; preds = %28
  %33 = getelementptr %Cam, ptr %2, i32 0, i32 0
  %34 = call i32 @Cam.thrown(ptr %2, i1 false)
  %35 = icmp eq i32 %34, 1
  br label %36

36:                                               ; preds = %32, %28
  %37 = phi i1 [ %35, %32 ], [ %31, %28 ]
  br i1 %37, label %38, label %46

38:                                               ; preds = %36
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

39:                                               ; preds = %46, %44, %42, %40
  unreachable

40:                                               ; preds = %0
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_378607937501964222, i64 8), i32 63, ptr null)
  br label %39

42:                                               ; preds = %14
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_378607937501964222, i64 8), i32 64, ptr null)
  br label %39

44:                                               ; preds = %26
  %45 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_378607937501964222, i64 8), i32 65, ptr null)
  br label %39

46:                                               ; preds = %36
  %47 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_378607937501964222, i64 8), i32 66, ptr null)
  br label %39
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

