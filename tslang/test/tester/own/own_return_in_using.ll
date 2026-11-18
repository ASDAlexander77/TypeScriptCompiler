; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr, ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_5010342068184733007 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFctor\00", align 8
@s_16077166718034104 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbad\00", align 8
@s_17112020470516500550 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispose\00", align 8
@s_14785268767687447306 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboom\00", align 8
@s_12638134423997487868 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF1\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_12638137722532372501 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF2\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_635592520754189898 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFq2\00", align 8
@s_12638204792741693372 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFq\00", align 8
@s_636580981707762362 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp1\00", align 8
@s_12638205892253321583 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp\00", align 8
@s_12638202593718436950 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFw\00", align 8
@s_12638216887369603693 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFz\00", align 8
@s_12638213588834719060 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFy\00", align 8
@s_626102635892993007 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFok\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_8211217425311437757 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_return_in_using.ts\00", align 8
@m_13574495241570801552 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFlengths and catches\00", align 8
@m_497100208963912460 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery dispose ran\00", align 8
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 16, comdat
@_ZTIPc = external constant ptr
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@disposed = private global i32 0
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

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

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Res..rtti, align 8
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
define ptr @Res..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.constructor(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  %8 = load ptr, ptr %3, align 8
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  %11 = zext i1 %10 to i32
  %12 = and i32 %11, 1
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %17

14:                                               ; preds = %2
  %15 = call i32 @strcmp(ptr %8, ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8))
  %16 = icmp eq i32 %15, 0
  br label %19

17:                                               ; preds = %2
  %18 = icmp eq i64 %9, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8) to i64)
  br label %19

19:                                               ; preds = %14, %17
  %20 = phi i1 [ %18, %17 ], [ %16, %14 ]
  br label %21

21:                                               ; preds = %19
  br i1 %20, label %22, label %24

22:                                               ; preds = %21
  %23 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_5010342068184733007, i64 8), ptr %23, align 8
  call void @__cxa_throw(ptr %23, ptr @_ZTIPc, ptr null)
  br label %25

24:                                               ; preds = %21
  ret void

25:                                               ; preds = %22
  unreachable
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %4, align 8
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  %8 = zext i1 %7 to i32
  %9 = and i32 %8, 1
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %1
  %12 = call i32 @strcmp(ptr %5, ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8))
  %13 = icmp eq i32 %12, 0
  br label %16

14:                                               ; preds = %1
  %15 = icmp eq i64 %6, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8) to i64)
  br label %16

16:                                               ; preds = %11, %14
  %17 = phi i1 [ %15, %14 ], [ %13, %11 ]
  br label %18

18:                                               ; preds = %16
  br i1 %17, label %19, label %21

19:                                               ; preds = %18
  %20 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_17112020470516500550, i64 8), ptr %20, align 8
  call void @__cxa_throw(ptr %20, ptr @_ZTIPc, ptr null)
  br label %24

21:                                               ; preds = %18
  %22 = load i32, ptr @disposed, align 4
  %23 = add i32 %22, 1
  store i32 %23, ptr @disposed, align 4
  ret void

24:                                               ; preds = %19
  unreachable
}

; Function Attrs: mustprogress
define ptr @returnFromBlock(ptr %0, ptr %1) #2 personality ptr @__gxx_personality_v0 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %6, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = icmp eq ptr %9, null
  %11 = select i1 %10, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %9
  %12 = call i64 @strlen(ptr %11)
  %13 = add i64 1, %12
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638134423997487868, i64 8))
  %15 = add i64 %13, %14
  %16 = add i64 %15, 8
  %17 = call ptr @GC_malloc(i64 %16)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr i8, ptr %18, i64 0
  %20 = sub i64 %15, 0
  %21 = call ptr @__memcpy_chk(ptr %19, ptr %11, i64 %12, i64 %20)
  %22 = add i64 0, %12
  %23 = getelementptr i8, ptr %18, i64 %22
  %24 = sub i64 %15, %22
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12638134423997487868, i64 8), i64 %14, i64 %24)
  %26 = add i64 %22, %14
  %27 = getelementptr i8, ptr %18, i64 %26
  store i8 0, ptr %27, align 1
  store ptr %18, ptr %5, align 8
  store i1 false, ptr %3, align 1
  %28 = call ptr @GC_malloc(i64 24)
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr %Res, ptr %29, i32 0, i32 0
  store ptr @Res..vtbl, ptr %30, align 8
  %31 = load ptr, ptr %6, align 8
  invoke void @Res.constructor(ptr %29, ptr %31)
          to label %32 unwind label %41

32:                                               ; preds = %2
  store ptr %29, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %33 = load ptr, ptr %5, align 8
  %34 = load i1, ptr %3, align 1
  br i1 %34, label %35, label %39

35:                                               ; preds = %32
  store i1 false, ptr %3, align 1
  %36 = load ptr, ptr %4, align 8
  %37 = getelementptr %Res, ptr %36, i32 0, i32 0
  invoke void @Res.dispose(ptr %36)
          to label %38 unwind label %41

38:                                               ; preds = %35
  br label %39

39:                                               ; preds = %38, %32
  store ptr %33, ptr %8, align 8
  %40 = load ptr, ptr %8, align 8
  ret ptr %40

41:                                               ; preds = %35, %2
  %42 = landingpad { ptr, i32 }
          cleanup
  %43 = load i1, ptr %3, align 1
  br i1 %43, label %44, label %47

44:                                               ; preds = %41
  store i1 false, ptr %3, align 1
  %45 = load ptr, ptr %4, align 8
  %46 = getelementptr %Res, ptr %45, i32 0, i32 0
  call void @Res.dispose(ptr %45)
  br label %47

47:                                               ; preds = %44, %41
  resume { ptr, i32 } %42
}

; Function Attrs: mustprogress
define ptr @nested(ptr %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = icmp eq ptr %9, null
  %11 = select i1 %10, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %9
  %12 = call i64 @strlen(ptr %11)
  %13 = add i64 1, %12
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638137722532372501, i64 8))
  %15 = add i64 %13, %14
  %16 = add i64 %15, 8
  %17 = call ptr @GC_malloc(i64 %16)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr i8, ptr %18, i64 0
  %20 = sub i64 %15, 0
  %21 = call ptr @__memcpy_chk(ptr %19, ptr %11, i64 %12, i64 %20)
  %22 = add i64 0, %12
  %23 = getelementptr i8, ptr %18, i64 %22
  %24 = sub i64 %15, %22
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12638137722532372501, i64 8), i64 %14, i64 %24)
  %26 = add i64 %22, %14
  %27 = getelementptr i8, ptr %18, i64 %26
  store i8 0, ptr %27, align 1
  store ptr %18, ptr %6, align 8
  store i1 false, ptr %4, align 1
  %28 = call ptr @GC_malloc(i64 24)
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr %Res, ptr %29, i32 0, i32 0
  store ptr @Res..vtbl, ptr %30, align 8
  invoke void @Res.constructor(ptr %29, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %31 unwind label %72

31:                                               ; preds = %1
  store ptr %29, ptr %5, align 8
  store i1 true, ptr %4, align 1
  store i1 false, ptr %2, align 1
  %32 = call ptr @GC_malloc(i64 24)
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr %Res, ptr %33, i32 0, i32 0
  store ptr @Res..vtbl, ptr %34, align 8
  invoke void @Res.constructor(ptr %33, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %35 unwind label %50

35:                                               ; preds = %31
  store ptr %33, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %36 = load ptr, ptr %6, align 8
  %37 = load i1, ptr %2, align 1
  br i1 %37, label %38, label %42

38:                                               ; preds = %35
  store i1 false, ptr %2, align 1
  %39 = load ptr, ptr %3, align 8
  %40 = getelementptr %Res, ptr %39, i32 0, i32 0
  invoke void @Res.dispose(ptr %39)
          to label %41 unwind label %50

41:                                               ; preds = %38
  br label %42

42:                                               ; preds = %41, %35
  %43 = load i1, ptr %4, align 1
  br i1 %43, label %44, label %48

44:                                               ; preds = %42
  store i1 false, ptr %4, align 1
  %45 = load ptr, ptr %5, align 8
  %46 = getelementptr %Res, ptr %45, i32 0, i32 0
  invoke void @Res.dispose(ptr %45)
          to label %47 unwind label %50

47:                                               ; preds = %44
  br label %48

48:                                               ; preds = %47, %42
  store ptr %36, ptr %8, align 8
  %49 = load ptr, ptr %8, align 8
  ret ptr %49

50:                                               ; preds = %44, %38, %31
  %51 = landingpad { ptr, i32 }
          catch ptr null
  %52 = extractvalue { ptr, i32 } %51, 0
  %53 = call ptr @__cxa_begin_catch(ptr %52)
  %54 = load i1, ptr %2, align 1
  br i1 %54, label %55, label %59

55:                                               ; preds = %50
  store i1 false, ptr %2, align 1
  %56 = load ptr, ptr %3, align 8
  %57 = getelementptr %Res, ptr %56, i32 0, i32 0
  invoke void @Res.dispose(ptr %56)
          to label %58 unwind label %endcatch.pad2

58:                                               ; preds = %55
  br label %59

59:                                               ; preds = %58, %50
  invoke void @__cxa_rethrow()
          to label %80 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %55
  %60 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

61:                                               ; No predecessors!
  %62 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %61
  %63 = phi { ptr, i32 } [ %62, %61 ], [ %60, %endcatch.pad2 ]
  %64 = extractvalue { ptr, i32 } %63, 0
  %65 = call ptr @__cxa_begin_catch(ptr %64)
  %66 = load i1, ptr %2, align 1
  br i1 %66, label %67, label %70

67:                                               ; preds = %.caught1
  store i1 false, ptr %2, align 1
  %68 = load ptr, ptr %3, align 8
  %69 = getelementptr %Res, ptr %68, i32 0, i32 0
  invoke void @Res.dispose(ptr %68)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %67
  br label %70

70:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %80 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %59, %70
  %71 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

72:                                               ; preds = %1
  %73 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %72
  %74 = phi { ptr, i32 } [ %73, %72 ], [ %71, %endcatch.pad ]
  %75 = load i1, ptr %4, align 1
  br i1 %75, label %76, label %79

76:                                               ; preds = %.caught
  store i1 false, ptr %4, align 1
  %77 = load ptr, ptr %5, align 8
  %78 = getelementptr %Res, ptr %77, i32 0, i32 0
  call void @Res.dispose(ptr %77)
  br label %79

79:                                               ; preds = %76, %.caught
  resume { ptr, i32 } %74

80:                                               ; preds = %59, %70
  unreachable

endcatch.resume:                                  ; preds = %67
  %81 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %81
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, ptr %6, align 4
  store i32 0, ptr %5, align 4
  br label %7

7:                                                ; preds = %402, %0
  %8 = load i32, ptr %5, align 4
  %9 = icmp slt i32 %8, 1000
  br i1 %9, label %10, label %405

10:                                               ; preds = %7
  %11 = load i32, ptr %6, align 4
  %12 = load i32, ptr %5, align 4
  %13 = call ptr @GC_malloc_atomic(i64 58)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %14, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %12)
  %16 = icmp eq ptr %14, null
  %17 = select i1 %16, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %18 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %19 = add i64 1, %18
  %20 = call i64 @strlen(ptr %17)
  %21 = add i64 %19, %20
  %22 = add i64 %21, 8
  %23 = call ptr @GC_malloc(i64 %22)
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr i8, ptr %24, i64 0
  %26 = sub i64 %21, 0
  %27 = call ptr @__memcpy_chk(ptr %25, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), i64 %18, i64 %26)
  %28 = add i64 0, %18
  %29 = getelementptr i8, ptr %24, i64 %28
  %30 = sub i64 %21, %28
  %31 = call ptr @__memcpy_chk(ptr %29, ptr %17, i64 %20, i64 %30)
  %32 = add i64 %28, %20
  %33 = getelementptr i8, ptr %24, i64 %32
  store i8 0, ptr %33, align 1
  %34 = call ptr @returnFromBlock(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_626102635892993007, i64 8))
  %35 = call i64 @strlen(ptr %34)
  %36 = sext i32 %11 to i64
  %37 = add i64 %36, %35
  %38 = trunc i64 %37 to i32
  store i32 %38, ptr %6, align 4
  %39 = load i32, ptr %6, align 4
  %40 = load i32, ptr %5, align 4
  %41 = call ptr @GC_malloc_atomic(i64 58)
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %42, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %40)
  %44 = icmp eq ptr %42, null
  %45 = select i1 %44, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %42
  %46 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638213588834719060, i64 8))
  %47 = add i64 1, %46
  %48 = call i64 @strlen(ptr %45)
  %49 = add i64 %47, %48
  %50 = add i64 %49, 8
  %51 = call ptr @GC_malloc(i64 %50)
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr i8, ptr %52, i64 0
  %54 = sub i64 %49, 0
  %55 = call ptr @__memcpy_chk(ptr %53, ptr getelementptr inbounds nuw (i8, ptr @s_12638213588834719060, i64 8), i64 %46, i64 %54)
  %56 = add i64 0, %46
  %57 = getelementptr i8, ptr %52, i64 %56
  %58 = sub i64 %49, %56
  %59 = call ptr @__memcpy_chk(ptr %57, ptr %45, i64 %48, i64 %58)
  %60 = add i64 %56, %48
  %61 = getelementptr i8, ptr %52, i64 %60
  store i8 0, ptr %61, align 1
  %62 = call ptr @nested(ptr %52)
  %63 = call i64 @strlen(ptr %62)
  %64 = sext i32 %39 to i64
  %65 = add i64 %64, %63
  %66 = trunc i64 %65 to i32
  store i32 %66, ptr %6, align 4
  %67 = load i32, ptr %5, align 4
  %68 = srem i32 %67, 10
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %402

70:                                               ; preds = %10
  %71 = load i32, ptr %5, align 4
  %72 = call ptr @GC_malloc_atomic(i64 58)
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %73, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %71)
  %75 = icmp eq ptr %73, null
  %76 = select i1 %75, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %73
  %77 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638216887369603693, i64 8))
  %78 = add i64 1, %77
  %79 = call i64 @strlen(ptr %76)
  %80 = add i64 %78, %79
  %81 = add i64 %80, 8
  %82 = call ptr @GC_malloc(i64 %81)
  %83 = getelementptr i8, ptr %82, i64 8
  %84 = getelementptr i8, ptr %83, i64 0
  %85 = sub i64 %80, 0
  %86 = call ptr @__memcpy_chk(ptr %84, ptr getelementptr inbounds nuw (i8, ptr @s_12638216887369603693, i64 8), i64 %77, i64 %85)
  %87 = add i64 0, %77
  %88 = getelementptr i8, ptr %83, i64 %87
  %89 = sub i64 %80, %87
  %90 = call ptr @__memcpy_chk(ptr %88, ptr %76, i64 %79, i64 %89)
  %91 = add i64 %87, %79
  %92 = getelementptr i8, ptr %83, i64 %91
  store i8 0, ptr %92, align 1
  %93 = invoke ptr @returnFromBlock(ptr %83, ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8))
          to label %94 unwind label %95

94:                                               ; preds = %70
  br label %236

95:                                               ; preds = %70
  %96 = landingpad { ptr, i32 }
          catch ptr null
  %97 = extractvalue { ptr, i32 } %96, 0
  %98 = call ptr @__cxa_begin_catch(ptr %97)
  %99 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %95
  %100 = icmp eq ptr %99, @_ZTIi
  br i1 %100, label %101, label %108

101:                                              ; preds = %.noexc11
  %102 = load i32, ptr %98, align 4
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, i32 }, ptr %104, i32 0, i32 0
  store i64 4, ptr %105, align 8
  %106 = getelementptr { i64, ptr, i32 }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, i32 }, ptr %104, i32 0, i32 2
  store i32 %102, ptr %107, align 4
  br label %231

108:                                              ; preds = %.noexc11
  %109 = icmp eq ptr %99, @_ZTIj
  br i1 %109, label %110, label %117

110:                                              ; preds = %108
  %111 = load i32, ptr %98, align 4
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, i32 }, ptr %113, i32 0, i32 0
  store i64 4, ptr %114, align 8
  %115 = getelementptr { i64, ptr, i32 }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, i32 }, ptr %113, i32 0, i32 2
  store i32 %111, ptr %116, align 4
  br label %228

117:                                              ; preds = %108
  %118 = icmp eq ptr %99, @_ZTId
  br i1 %118, label %119, label %126

119:                                              ; preds = %117
  %120 = load double, ptr %98, align 8
  %121 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %119
  %122 = getelementptr i8, ptr %121, i64 8
  %123 = getelementptr { i64, ptr, double }, ptr %122, i32 0, i32 0
  store i64 8, ptr %123, align 8
  %124 = getelementptr { i64, ptr, double }, ptr %122, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %124, align 8
  %125 = getelementptr { i64, ptr, double }, ptr %122, i32 0, i32 2
  store double %120, ptr %125, align 8
  br label %225

126:                                              ; preds = %117
  %127 = icmp eq ptr %99, @_ZTIf
  br i1 %127, label %128, label %135

128:                                              ; preds = %126
  %129 = load float, ptr %98, align 4
  %130 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %128
  %131 = getelementptr i8, ptr %130, i64 8
  %132 = getelementptr { i64, ptr, float }, ptr %131, i32 0, i32 0
  store i64 4, ptr %132, align 8
  %133 = getelementptr { i64, ptr, float }, ptr %131, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %133, align 8
  %134 = getelementptr { i64, ptr, float }, ptr %131, i32 0, i32 2
  store float %129, ptr %134, align 4
  br label %222

135:                                              ; preds = %126
  %136 = icmp eq ptr %99, @_ZTIb
  br i1 %136, label %137, label %144

137:                                              ; preds = %135
  %138 = load i1, ptr %98, align 1
  %139 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %137
  %140 = getelementptr i8, ptr %139, i64 8
  %141 = getelementptr { i64, ptr, i1 }, ptr %140, i32 0, i32 0
  store i64 1, ptr %141, align 8
  %142 = getelementptr { i64, ptr, i1 }, ptr %140, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %142, align 8
  %143 = getelementptr { i64, ptr, i1 }, ptr %140, i32 0, i32 2
  store i1 %138, ptr %143, align 1
  br label %219

144:                                              ; preds = %135
  %145 = icmp eq ptr %99, @_ZTIx
  br i1 %145, label %146, label %153

146:                                              ; preds = %144
  %147 = load i64, ptr %98, align 8
  %148 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %146
  %149 = getelementptr i8, ptr %148, i64 8
  %150 = getelementptr { i64, ptr, i64 }, ptr %149, i32 0, i32 0
  store i64 8, ptr %150, align 8
  %151 = getelementptr { i64, ptr, i64 }, ptr %149, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %151, align 8
  %152 = getelementptr { i64, ptr, i64 }, ptr %149, i32 0, i32 2
  store i64 %147, ptr %152, align 8
  br label %216

153:                                              ; preds = %144
  %154 = icmp eq ptr %99, @_ZTIPc
  br i1 %154, label %155, label %162

155:                                              ; preds = %153
  %156 = load ptr, ptr %98, align 8
  %157 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %155
  %158 = getelementptr i8, ptr %157, i64 8
  %159 = getelementptr { i64, ptr, ptr }, ptr %158, i32 0, i32 0
  store i64 8, ptr %159, align 8
  %160 = getelementptr { i64, ptr, ptr }, ptr %158, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %160, align 8
  %161 = getelementptr { i64, ptr, ptr }, ptr %158, i32 0, i32 2
  store ptr %156, ptr %161, align 8
  br label %213

162:                                              ; preds = %153
  %163 = icmp eq ptr %99, @_ZTIPv
  br i1 %163, label %164, label %166

164:                                              ; preds = %162
  %165 = load ptr, ptr %98, align 8
  br label %210

166:                                              ; preds = %162
  %167 = load ptr, ptr %99, align 8
  %168 = icmp eq ptr %167, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %168, label %169, label %200

169:                                              ; preds = %166
  %170 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %99, i32 0, i32 2
  %171 = load i32, ptr %170, align 4
  %172 = and i32 %171, 1073741824
  %173 = icmp ne i32 %172, 0
  br i1 %173, label %174, label %190

174:                                              ; preds = %169
  %175 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %99, i32 0, i32 4
  %176 = load ptr, ptr %175, align 8
  %177 = icmp ne ptr %176, null
  br i1 %177, label %178, label %180

178:                                              ; preds = %174
  invoke void %176(ptr %1, ptr %98)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %178
  %179 = load ptr, ptr %1, align 8
  br label %187

180:                                              ; preds = %174
  %181 = load ptr, ptr %98, align 8
  %182 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %180
  %183 = getelementptr i8, ptr %182, i64 8
  %184 = getelementptr { i64, ptr, ptr }, ptr %183, i32 0, i32 0
  store i64 8, ptr %184, align 8
  %185 = getelementptr { i64, ptr, ptr }, ptr %183, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %185, align 8
  %186 = getelementptr { i64, ptr, ptr }, ptr %183, i32 0, i32 2
  store ptr %181, ptr %186, align 8
  br label %187

187:                                              ; preds = %.noexc6, %.noexc2
  %188 = phi ptr [ %183, %.noexc2 ], [ %179, %.noexc6 ]
  br label %189

189:                                              ; preds = %187
  br label %197

190:                                              ; preds = %169
  %191 = load ptr, ptr %98, align 8
  %192 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %190
  %193 = getelementptr i8, ptr %192, i64 8
  %194 = getelementptr { i64, ptr, ptr }, ptr %193, i32 0, i32 0
  store i64 8, ptr %194, align 8
  %195 = getelementptr { i64, ptr, ptr }, ptr %193, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %195, align 8
  %196 = getelementptr { i64, ptr, ptr }, ptr %193, i32 0, i32 2
  store ptr %191, ptr %196, align 8
  br label %197

197:                                              ; preds = %189, %.noexc12
  %198 = phi ptr [ %193, %.noexc12 ], [ %188, %189 ]
  br label %199

199:                                              ; preds = %197
  br label %207

200:                                              ; preds = %166
  %201 = load ptr, ptr %98, align 8
  %202 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %200
  %203 = getelementptr i8, ptr %202, i64 8
  %204 = getelementptr { i64, ptr, ptr }, ptr %203, i32 0, i32 0
  store i64 8, ptr %204, align 8
  %205 = getelementptr { i64, ptr, ptr }, ptr %203, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %205, align 8
  %206 = getelementptr { i64, ptr, ptr }, ptr %203, i32 0, i32 2
  store ptr %201, ptr %206, align 8
  br label %207

207:                                              ; preds = %199, %.noexc15
  %208 = phi ptr [ %203, %.noexc15 ], [ %198, %199 ]
  br label %209

209:                                              ; preds = %207
  br label %210

210:                                              ; preds = %164, %209
  %211 = phi ptr [ %208, %209 ], [ %165, %164 ]
  br label %212

212:                                              ; preds = %210
  br label %213

213:                                              ; preds = %.noexc18, %212
  %214 = phi ptr [ %211, %212 ], [ %158, %.noexc18 ]
  br label %215

215:                                              ; preds = %213
  br label %216

216:                                              ; preds = %.noexc16, %215
  %217 = phi ptr [ %214, %215 ], [ %149, %.noexc16 ]
  br label %218

218:                                              ; preds = %216
  br label %219

219:                                              ; preds = %.noexc1, %218
  %220 = phi ptr [ %217, %218 ], [ %140, %.noexc1 ]
  br label %221

221:                                              ; preds = %219
  br label %222

222:                                              ; preds = %.noexc3, %221
  %223 = phi ptr [ %220, %221 ], [ %131, %.noexc3 ]
  br label %224

224:                                              ; preds = %222
  br label %225

225:                                              ; preds = %.noexc7, %224
  %226 = phi ptr [ %223, %224 ], [ %122, %.noexc7 ]
  br label %227

227:                                              ; preds = %225
  br label %228

228:                                              ; preds = %.noexc8, %227
  %229 = phi ptr [ %226, %227 ], [ %113, %.noexc8 ]
  br label %230

230:                                              ; preds = %228
  br label %231

231:                                              ; preds = %.noexc9, %230
  %232 = phi ptr [ %229, %230 ], [ %104, %.noexc9 ]
  br label %233

233:                                              ; preds = %231
  store ptr %232, ptr %4, align 8
  %234 = load i32, ptr %6, align 4
  %235 = add i32 %234, 1
  store i32 %235, ptr %6, align 4
  call void @__cxa_end_catch()
  br label %236

236:                                              ; preds = %94, %233
  %237 = load i32, ptr %5, align 4
  %238 = call ptr @GC_malloc_atomic(i64 58)
  %239 = getelementptr i8, ptr %238, i64 8
  %240 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %239, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %237)
  %241 = icmp eq ptr %239, null
  %242 = select i1 %241, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %239
  %243 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638202593718436950, i64 8))
  %244 = add i64 1, %243
  %245 = call i64 @strlen(ptr %242)
  %246 = add i64 %244, %245
  %247 = add i64 %246, 8
  %248 = call ptr @GC_malloc(i64 %247)
  %249 = getelementptr i8, ptr %248, i64 8
  %250 = getelementptr i8, ptr %249, i64 0
  %251 = sub i64 %246, 0
  %252 = call ptr @__memcpy_chk(ptr %250, ptr getelementptr inbounds nuw (i8, ptr @s_12638202593718436950, i64 8), i64 %243, i64 %251)
  %253 = add i64 0, %243
  %254 = getelementptr i8, ptr %249, i64 %253
  %255 = sub i64 %246, %253
  %256 = call ptr @__memcpy_chk(ptr %254, ptr %242, i64 %245, i64 %255)
  %257 = add i64 %253, %245
  %258 = getelementptr i8, ptr %249, i64 %257
  store i8 0, ptr %258, align 1
  %259 = invoke ptr @returnFromBlock(ptr %249, ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8))
          to label %260 unwind label %261

260:                                              ; preds = %236
  br label %402

261:                                              ; preds = %236
  %262 = landingpad { ptr, i32 }
          catch ptr null
  %263 = extractvalue { ptr, i32 } %262, 0
  %264 = call ptr @__cxa_begin_catch(ptr %263)
  %265 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %261
  %266 = icmp eq ptr %265, @_ZTIi
  br i1 %266, label %267, label %274

267:                                              ; preds = %.noexc4
  %268 = load i32, ptr %264, align 4
  %269 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %267
  %270 = getelementptr i8, ptr %269, i64 8
  %271 = getelementptr { i64, ptr, i32 }, ptr %270, i32 0, i32 0
  store i64 4, ptr %271, align 8
  %272 = getelementptr { i64, ptr, i32 }, ptr %270, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %272, align 8
  %273 = getelementptr { i64, ptr, i32 }, ptr %270, i32 0, i32 2
  store i32 %268, ptr %273, align 4
  br label %397

274:                                              ; preds = %.noexc4
  %275 = icmp eq ptr %265, @_ZTIj
  br i1 %275, label %276, label %283

276:                                              ; preds = %274
  %277 = load i32, ptr %264, align 4
  %278 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %276
  %279 = getelementptr i8, ptr %278, i64 8
  %280 = getelementptr { i64, ptr, i32 }, ptr %279, i32 0, i32 0
  store i64 4, ptr %280, align 8
  %281 = getelementptr { i64, ptr, i32 }, ptr %279, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %281, align 8
  %282 = getelementptr { i64, ptr, i32 }, ptr %279, i32 0, i32 2
  store i32 %277, ptr %282, align 4
  br label %394

283:                                              ; preds = %274
  %284 = icmp eq ptr %265, @_ZTId
  br i1 %284, label %285, label %292

285:                                              ; preds = %283
  %286 = load double, ptr %264, align 8
  %287 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %285
  %288 = getelementptr i8, ptr %287, i64 8
  %289 = getelementptr { i64, ptr, double }, ptr %288, i32 0, i32 0
  store i64 8, ptr %289, align 8
  %290 = getelementptr { i64, ptr, double }, ptr %288, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %290, align 8
  %291 = getelementptr { i64, ptr, double }, ptr %288, i32 0, i32 2
  store double %286, ptr %291, align 8
  br label %391

292:                                              ; preds = %283
  %293 = icmp eq ptr %265, @_ZTIf
  br i1 %293, label %294, label %301

294:                                              ; preds = %292
  %295 = load float, ptr %264, align 4
  %296 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %294
  %297 = getelementptr i8, ptr %296, i64 8
  %298 = getelementptr { i64, ptr, float }, ptr %297, i32 0, i32 0
  store i64 4, ptr %298, align 8
  %299 = getelementptr { i64, ptr, float }, ptr %297, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %299, align 8
  %300 = getelementptr { i64, ptr, float }, ptr %297, i32 0, i32 2
  store float %295, ptr %300, align 4
  br label %388

301:                                              ; preds = %292
  %302 = icmp eq ptr %265, @_ZTIb
  br i1 %302, label %303, label %310

303:                                              ; preds = %301
  %304 = load i1, ptr %264, align 1
  %305 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %303
  %306 = getelementptr i8, ptr %305, i64 8
  %307 = getelementptr { i64, ptr, i1 }, ptr %306, i32 0, i32 0
  store i64 1, ptr %307, align 8
  %308 = getelementptr { i64, ptr, i1 }, ptr %306, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %308, align 8
  %309 = getelementptr { i64, ptr, i1 }, ptr %306, i32 0, i32 2
  store i1 %304, ptr %309, align 1
  br label %385

310:                                              ; preds = %301
  %311 = icmp eq ptr %265, @_ZTIx
  br i1 %311, label %312, label %319

312:                                              ; preds = %310
  %313 = load i64, ptr %264, align 8
  %314 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %312
  %315 = getelementptr i8, ptr %314, i64 8
  %316 = getelementptr { i64, ptr, i64 }, ptr %315, i32 0, i32 0
  store i64 8, ptr %316, align 8
  %317 = getelementptr { i64, ptr, i64 }, ptr %315, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %317, align 8
  %318 = getelementptr { i64, ptr, i64 }, ptr %315, i32 0, i32 2
  store i64 %313, ptr %318, align 8
  br label %382

319:                                              ; preds = %310
  %320 = icmp eq ptr %265, @_ZTIPc
  br i1 %320, label %321, label %328

321:                                              ; preds = %319
  %322 = load ptr, ptr %264, align 8
  %323 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %321
  %324 = getelementptr i8, ptr %323, i64 8
  %325 = getelementptr { i64, ptr, ptr }, ptr %324, i32 0, i32 0
  store i64 8, ptr %325, align 8
  %326 = getelementptr { i64, ptr, ptr }, ptr %324, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %326, align 8
  %327 = getelementptr { i64, ptr, ptr }, ptr %324, i32 0, i32 2
  store ptr %322, ptr %327, align 8
  br label %379

328:                                              ; preds = %319
  %329 = icmp eq ptr %265, @_ZTIPv
  br i1 %329, label %330, label %332

330:                                              ; preds = %328
  %331 = load ptr, ptr %264, align 8
  br label %376

332:                                              ; preds = %328
  %333 = load ptr, ptr %265, align 8
  %334 = icmp eq ptr %333, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %334, label %335, label %366

335:                                              ; preds = %332
  %336 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %265, i32 0, i32 2
  %337 = load i32, ptr %336, align 4
  %338 = and i32 %337, 1073741824
  %339 = icmp ne i32 %338, 0
  br i1 %339, label %340, label %356

340:                                              ; preds = %335
  %341 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %265, i32 0, i32 4
  %342 = load ptr, ptr %341, align 8
  %343 = icmp ne ptr %342, null
  br i1 %343, label %344, label %346

344:                                              ; preds = %340
  invoke void %342(ptr %2, ptr %264)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %344
  %345 = load ptr, ptr %2, align 8
  br label %353

346:                                              ; preds = %340
  %347 = load ptr, ptr %264, align 8
  %348 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %346
  %349 = getelementptr i8, ptr %348, i64 8
  %350 = getelementptr { i64, ptr, ptr }, ptr %349, i32 0, i32 0
  store i64 8, ptr %350, align 8
  %351 = getelementptr { i64, ptr, ptr }, ptr %349, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %351, align 8
  %352 = getelementptr { i64, ptr, ptr }, ptr %349, i32 0, i32 2
  store ptr %347, ptr %352, align 8
  br label %353

353:                                              ; preds = %.noexc14, %.noexc13
  %354 = phi ptr [ %349, %.noexc13 ], [ %345, %.noexc14 ]
  br label %355

355:                                              ; preds = %353
  br label %363

356:                                              ; preds = %335
  %357 = load ptr, ptr %264, align 8
  %358 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %356
  %359 = getelementptr i8, ptr %358, i64 8
  %360 = getelementptr { i64, ptr, ptr }, ptr %359, i32 0, i32 0
  store i64 8, ptr %360, align 8
  %361 = getelementptr { i64, ptr, ptr }, ptr %359, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %361, align 8
  %362 = getelementptr { i64, ptr, ptr }, ptr %359, i32 0, i32 2
  store ptr %357, ptr %362, align 8
  br label %363

363:                                              ; preds = %355, %.noexc23
  %364 = phi ptr [ %359, %.noexc23 ], [ %354, %355 ]
  br label %365

365:                                              ; preds = %363
  br label %373

366:                                              ; preds = %332
  %367 = load ptr, ptr %264, align 8
  %368 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %366
  %369 = getelementptr i8, ptr %368, i64 8
  %370 = getelementptr { i64, ptr, ptr }, ptr %369, i32 0, i32 0
  store i64 8, ptr %370, align 8
  %371 = getelementptr { i64, ptr, ptr }, ptr %369, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %371, align 8
  %372 = getelementptr { i64, ptr, ptr }, ptr %369, i32 0, i32 2
  store ptr %367, ptr %372, align 8
  br label %373

373:                                              ; preds = %365, %.noexc
  %374 = phi ptr [ %369, %.noexc ], [ %364, %365 ]
  br label %375

375:                                              ; preds = %373
  br label %376

376:                                              ; preds = %330, %375
  %377 = phi ptr [ %374, %375 ], [ %331, %330 ]
  br label %378

378:                                              ; preds = %376
  br label %379

379:                                              ; preds = %.noexc17, %378
  %380 = phi ptr [ %377, %378 ], [ %324, %.noexc17 ]
  br label %381

381:                                              ; preds = %379
  br label %382

382:                                              ; preds = %.noexc5, %381
  %383 = phi ptr [ %380, %381 ], [ %315, %.noexc5 ]
  br label %384

384:                                              ; preds = %382
  br label %385

385:                                              ; preds = %.noexc22, %384
  %386 = phi ptr [ %383, %384 ], [ %306, %.noexc22 ]
  br label %387

387:                                              ; preds = %385
  br label %388

388:                                              ; preds = %.noexc21, %387
  %389 = phi ptr [ %386, %387 ], [ %297, %.noexc21 ]
  br label %390

390:                                              ; preds = %388
  br label %391

391:                                              ; preds = %.noexc20, %390
  %392 = phi ptr [ %389, %390 ], [ %288, %.noexc20 ]
  br label %393

393:                                              ; preds = %391
  br label %394

394:                                              ; preds = %.noexc19, %393
  %395 = phi ptr [ %392, %393 ], [ %279, %.noexc19 ]
  br label %396

396:                                              ; preds = %394
  br label %397

397:                                              ; preds = %.noexc10, %396
  %398 = phi ptr [ %395, %396 ], [ %270, %.noexc10 ]
  br label %399

399:                                              ; preds = %397
  store ptr %398, ptr %3, align 8
  %400 = load i32, ptr %6, align 4
  %401 = add i32 %400, 1
  store i32 %401, ptr %6, align 4
  call void @__cxa_end_catch()
  br label %402

402:                                              ; preds = %260, %399, %10
  %403 = load i32, ptr %5, align 4
  %404 = add i32 %403, 1
  store i32 %404, ptr %5, align 4
  br label %7

405:                                              ; preds = %7
  %406 = call ptr @returnFromBlock(ptr getelementptr inbounds nuw (i8, ptr @s_12638205892253321583, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_626102635892993007, i64 8))
  %407 = ptrtoint ptr %406 to i64
  %408 = icmp ne i64 %407, 0
  %409 = zext i1 %408 to i32
  %410 = and i32 %409, 1
  %411 = icmp ne i32 %410, 0
  br i1 %411, label %412, label %415

412:                                              ; preds = %405
  %413 = call i32 @strcmp(ptr %406, ptr getelementptr inbounds nuw (i8, ptr @s_636580981707762362, i64 8))
  %414 = icmp eq i32 %413, 0
  br label %417

415:                                              ; preds = %405
  %416 = icmp eq i64 %407, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_636580981707762362, i64 8) to i64)
  br label %417

417:                                              ; preds = %412, %415
  %418 = phi i1 [ %416, %415 ], [ %414, %412 ]
  br label %419

419:                                              ; preds = %417
  br i1 %418, label %420, label %443

420:                                              ; preds = %419
  %421 = call ptr @nested(ptr getelementptr inbounds nuw (i8, ptr @s_12638204792741693372, i64 8))
  %422 = ptrtoint ptr %421 to i64
  %423 = icmp ne i64 %422, 0
  %424 = zext i1 %423 to i32
  %425 = and i32 %424, 1
  %426 = icmp ne i32 %425, 0
  br i1 %426, label %427, label %430

427:                                              ; preds = %420
  %428 = call i32 @strcmp(ptr %421, ptr getelementptr inbounds nuw (i8, ptr @s_635592520754189898, i64 8))
  %429 = icmp eq i32 %428, 0
  br label %432

430:                                              ; preds = %420
  %431 = icmp eq i64 %422, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_635592520754189898, i64 8) to i64)
  br label %432

432:                                              ; preds = %427, %430
  %433 = phi i1 [ %431, %430 ], [ %429, %427 ]
  br label %434

434:                                              ; preds = %432
  br i1 %433, label %435, label %445

435:                                              ; preds = %434
  %436 = load i32, ptr %6, align 4
  %437 = icmp eq i32 %436, 9980
  br i1 %437, label %438, label %447

438:                                              ; preds = %435
  %439 = load i32, ptr @disposed, align 4
  %440 = icmp eq i32 %439, 3003
  br i1 %440, label %441, label %449

441:                                              ; preds = %438
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

442:                                              ; preds = %449, %447, %445, %443
  unreachable

443:                                              ; preds = %419
  %444 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8211217425311437757, i64 8), i32 57, ptr null)
  br label %442

445:                                              ; preds = %434
  %446 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8211217425311437757, i64 8), i32 58, ptr null)
  br label %442

447:                                              ; preds = %435
  %448 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13574495241570801552, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8211217425311437757, i64 8), i32 59, ptr null)
  br label %442

449:                                              ; preds = %438
  %450 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_497100208963912460, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8211217425311437757, i64 8), i32 60, ptr null)
  br label %442

endcatch.resume:                                  ; preds = %356, %303, %294, %285, %276, %155, %321, %146, %200, %344, %346, %190, %95, %267, %101, %110, %119, %178, %312, %261, %128, %180, %137, %366
  %451 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %451
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

