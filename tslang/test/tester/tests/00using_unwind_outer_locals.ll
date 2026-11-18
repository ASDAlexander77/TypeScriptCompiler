; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr, ptr }

$Res..size = comdat any

@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_12638145419113769978 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF;\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_12638119030834692914 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF#\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_12638134423997487868 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF1\00", align 8
@s_12638137722532372501 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF2\00", align 8
@s_12638201494206808739 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFt\00", align 8
@s_12638136623020744290 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF3\00", align 8
@s_12638131125462603235 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF4\00", align 8
@s_12638130025950975024 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF5\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_8008412449099960154 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught;\00", align 8
@s_12638133324485859657 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF6\00", align 8
@s_12638132224974231446 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF7\00", align 8
@s_584909432750204567 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF: \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_12376869648902559890 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00using_unwind_outer_locals.ts\00", align 8
@m_2432362340776716864 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmust throw\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_636574384637993096 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp7\00", align 8
@s_7824925360632822997 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught further out, switch: \00", align 8
@s_3019407056386408938 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa;caught;\00", align 8
@s_636575484149621307 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp6\00", align 8
@s_2773407230609657009 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught further out: \00", align 8
@s_12917172910269586205 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrow out of a loop body\00", align 8
@s_15882658668006414508 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa0;a1;\00", align 8
@s_2521489369436186603 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrow out of a switch case\00", align 8
@s_9925662178766230706 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp4;a;\00", align 8
@s_14630895708312341009 = internal constant [36 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrow through a plain scope\00", align 8
@s_11046246514244802049 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp3t;a;\00", align 8
@s_1254165271164641062 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrow out of a block\00", align 8
@s_14968365627901597948 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp2;a;\00", align 8
@s_620385175427431157 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa;\00", align 8
@s_636580981707762362 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp1\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@s_11862283273679587948 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreturn out of a block: \00", align 8
@s_12638205892253321583 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFp\00", align 8
@m_8074741577383239958 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreturn out of a block\00", align 8
@m_10267497238324938131 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught further out\00", align 8
@m_16285134945708457239 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught further out, switch\00", align 8
@trail = private global ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 16, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIi = external constant ptr
@td_13071058665056268293_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_13071058665056268293, ptr @tsretcb_13071058665056268293, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

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

define internal void @tsretcb_13071058665056268293(ptr %0) {
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

define internal void @tsrelcb_13071058665056268293(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %14

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %14

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %13

10:                                               ; preds = %6
  %11 = call i1 @__tslang_dec_ref(ptr %8)
  br i1 %11, label %12, label %13

12:                                               ; preds = %10
  call void @tsrel_1752011451674293319(ptr %8)
  call void @__tslang_free_block(ptr %8)
  br label %13

13:                                               ; preds = %12, %10, %6
  call void @__tslang_free_block(ptr %2)
  br label %14

14:                                               ; preds = %13, %4, %1
  ret void
}

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare void @__cxa_end_catch()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @Res..new() #2 {
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
define void @Res.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @trail, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %9 = call i64 @strlen(ptr %8)
  %10 = add i64 1, %9
  %11 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %12 = add i64 %10, %11
  %13 = add i64 %12, 8
  %14 = call ptr @GC_malloc(i64 %13)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i8, ptr %15, i64 0
  %17 = sub i64 %12, 0
  %18 = call ptr @__memcpy_chk(ptr %16, ptr %8, i64 %9, i64 %17)
  %19 = add i64 0, %9
  %20 = getelementptr i8, ptr %15, i64 %19
  %21 = sub i64 %12, %19
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %11, i64 %21)
  %23 = add i64 %19, %11
  %24 = getelementptr i8, ptr %15, i64 %23
  store i8 0, ptr %24, align 1
  %25 = icmp eq ptr %3, null
  %26 = select i1 %25, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  %27 = icmp eq ptr %15, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %29 = call i64 @strlen(ptr %26)
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr %28)
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %26, i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %28, i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  store ptr %35, ptr @trail, align 8
  ret void
}

; Function Attrs: mustprogress
define i64 @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %7, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 64
  br i1 %6, label %7, label %29

7:                                                ; preds = %4
  %8 = load ptr, ptr %2, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %8
  %11 = call i64 @strlen(ptr %10)
  %12 = add i64 1, %11
  %13 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638119030834692914, i64 8))
  %14 = add i64 %12, %13
  %15 = add i64 %14, 8
  %16 = call ptr @GC_malloc(i64 %15)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr i8, ptr %17, i64 0
  %19 = sub i64 %14, 0
  %20 = call ptr @__memcpy_chk(ptr %18, ptr %10, i64 %11, i64 %19)
  %21 = add i64 0, %11
  %22 = getelementptr i8, ptr %17, i64 %21
  %23 = sub i64 %14, %21
  %24 = call ptr @__memcpy_chk(ptr %22, ptr getelementptr inbounds nuw (i8, ptr @s_12638119030834692914, i64 8), i64 %13, i64 %23)
  %25 = add i64 %21, %13
  %26 = getelementptr i8, ptr %17, i64 %25
  store i8 0, ptr %26, align 1
  store ptr %17, ptr %2, align 8
  %27 = load i32, ptr %1, align 4
  %28 = add i32 %27, 1
  store i32 %28, ptr %1, align 4
  br label %4

29:                                               ; preds = %4
  %30 = load ptr, ptr %2, align 8
  %31 = call i64 @strlen(ptr %30)
  store i64 %31, ptr %3, align 8
  %32 = load i64, ptr %3, align 8
  ret i64 %32
}

; Function Attrs: mustprogress
define ptr @returnFromBlock(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = icmp eq ptr %7, null
  %9 = select i1 %8, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %7
  %10 = call i64 @strlen(ptr %9)
  %11 = add i64 1, %10
  %12 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638134423997487868, i64 8))
  %13 = add i64 %11, %12
  %14 = add i64 %13, 8
  %15 = call ptr @GC_malloc(i64 %14)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr i8, ptr %16, i64 0
  %18 = sub i64 %13, 0
  %19 = call ptr @__memcpy_chk(ptr %17, ptr %9, i64 %10, i64 %18)
  %20 = add i64 0, %10
  %21 = getelementptr i8, ptr %16, i64 %20
  %22 = sub i64 %13, %20
  %23 = call ptr @__memcpy_chk(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_12638134423997487868, i64 8), i64 %12, i64 %22)
  %24 = add i64 %20, %12
  %25 = getelementptr i8, ptr %16, i64 %24
  store i8 0, ptr %25, align 1
  store ptr %16, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %26 = call ptr @GC_malloc(i64 24)
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr %Res, ptr %27, i32 0, i32 0
  store ptr @Res..vtbl, ptr %28, align 8
  invoke void @Res.constructor(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %29 unwind label %38

29:                                               ; preds = %1
  store ptr %27, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %30 = load ptr, ptr %4, align 8
  %31 = load i1, ptr %2, align 1
  br i1 %31, label %32, label %36

32:                                               ; preds = %29
  store i1 false, ptr %2, align 1
  %33 = load ptr, ptr %3, align 8
  %34 = getelementptr %Res, ptr %33, i32 0, i32 0
  invoke void @Res.dispose(ptr %33)
          to label %35 unwind label %38

35:                                               ; preds = %32
  br label %36

36:                                               ; preds = %35, %29
  store ptr %30, ptr %6, align 8
  %37 = load ptr, ptr %6, align 8
  ret ptr %37

38:                                               ; preds = %32, %1
  %39 = landingpad { ptr, i32 }
          cleanup
  %40 = load i1, ptr %2, align 1
  br i1 %40, label %41, label %44

41:                                               ; preds = %38
  store i1 false, ptr %2, align 1
  %42 = load ptr, ptr %3, align 8
  %43 = getelementptr %Res, ptr %42, i32 0, i32 0
  call void @Res.dispose(ptr %42)
  br label %44

44:                                               ; preds = %41, %38
  resume { ptr, i32 } %39
}

; Function Attrs: mustprogress
define void @throwFromBlock(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %9 = call i64 @strlen(ptr %8)
  %10 = add i64 1, %9
  %11 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638137722532372501, i64 8))
  %12 = add i64 %10, %11
  %13 = add i64 %12, 8
  %14 = call ptr @GC_malloc(i64 %13)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i8, ptr %15, i64 0
  %17 = sub i64 %12, 0
  %18 = call ptr @__memcpy_chk(ptr %16, ptr %8, i64 %9, i64 %17)
  %19 = add i64 0, %9
  %20 = getelementptr i8, ptr %15, i64 %19
  %21 = sub i64 %12, %19
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638137722532372501, i64 8), i64 %11, i64 %21)
  %23 = add i64 %19, %11
  %24 = getelementptr i8, ptr %15, i64 %23
  store i8 0, ptr %24, align 1
  store ptr %15, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %25 = call ptr @GC_malloc(i64 24)
  %26 = getelementptr i8, ptr %25, i64 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  store ptr @Res..vtbl, ptr %27, align 8
  invoke void @Res.constructor(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %28 unwind label %70

28:                                               ; preds = %1
  store ptr %26, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %29 = load ptr, ptr @trail, align 8
  %30 = load ptr, ptr %4, align 8
  %31 = icmp eq ptr %30, null
  %32 = select i1 %31, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %30
  %33 = call i64 @strlen(ptr %32)
  %34 = add i64 1, %33
  %35 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %36 = add i64 %34, %35
  %37 = add i64 %36, 8
  %38 = call ptr @GC_malloc(i64 %37)
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr i8, ptr %39, i64 0
  %41 = sub i64 %36, 0
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %32, i64 %33, i64 %41)
  %43 = add i64 0, %33
  %44 = getelementptr i8, ptr %39, i64 %43
  %45 = sub i64 %36, %43
  %46 = call ptr @__memcpy_chk(ptr %44, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %35, i64 %45)
  %47 = add i64 %43, %35
  %48 = getelementptr i8, ptr %39, i64 %47
  store i8 0, ptr %48, align 1
  %49 = icmp eq ptr %29, null
  %50 = select i1 %49, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %29
  %51 = icmp eq ptr %39, null
  %52 = select i1 %51, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %39
  %53 = call i64 @strlen(ptr %50)
  %54 = add i64 1, %53
  %55 = call i64 @strlen(ptr %52)
  %56 = add i64 %54, %55
  %57 = add i64 %56, 8
  %58 = call ptr @GC_malloc(i64 %57)
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr i8, ptr %59, i64 0
  %61 = sub i64 %56, 0
  %62 = call ptr @__memcpy_chk(ptr %60, ptr %50, i64 %53, i64 %61)
  %63 = add i64 0, %53
  %64 = getelementptr i8, ptr %59, i64 %63
  %65 = sub i64 %56, %63
  %66 = call ptr @__memcpy_chk(ptr %64, ptr %52, i64 %55, i64 %65)
  %67 = add i64 %63, %55
  %68 = getelementptr i8, ptr %59, i64 %67
  store i8 0, ptr %68, align 1
  store ptr %59, ptr @trail, align 8
  %69 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %69, align 4
  invoke void @__cxa_throw(ptr %69, ptr @_ZTIi, ptr null)
          to label %77 unwind label %70

70:                                               ; preds = %28, %1
  %71 = landingpad { ptr, i32 }
          cleanup
  %72 = load i1, ptr %2, align 1
  br i1 %72, label %73, label %76

73:                                               ; preds = %70
  store i1 false, ptr %2, align 1
  %74 = load ptr, ptr %3, align 8
  %75 = getelementptr %Res, ptr %74, i32 0, i32 0
  call void @Res.dispose(ptr %74)
  br label %76

76:                                               ; preds = %73, %70
  resume { ptr, i32 } %71

77:                                               ; preds = %28
  unreachable
}

; Function Attrs: mustprogress
define void @throwThroughPlainScope(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = icmp eq ptr %7, null
  %9 = select i1 %8, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %7
  %10 = call i64 @strlen(ptr %9)
  %11 = add i64 1, %10
  %12 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638136623020744290, i64 8))
  %13 = add i64 %11, %12
  %14 = add i64 %13, 8
  %15 = call ptr @GC_malloc(i64 %14)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr i8, ptr %16, i64 0
  %18 = sub i64 %13, 0
  %19 = call ptr @__memcpy_chk(ptr %17, ptr %9, i64 %10, i64 %18)
  %20 = add i64 0, %10
  %21 = getelementptr i8, ptr %16, i64 %20
  %22 = sub i64 %13, %20
  %23 = call ptr @__memcpy_chk(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_12638136623020744290, i64 8), i64 %12, i64 %22)
  %24 = add i64 %20, %12
  %25 = getelementptr i8, ptr %16, i64 %24
  store i8 0, ptr %25, align 1
  store ptr %16, ptr %5, align 8
  %26 = load ptr, ptr %5, align 8
  %27 = icmp eq ptr %26, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %26
  %29 = call i64 @strlen(ptr %28)
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638201494206808739, i64 8))
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %28, i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr getelementptr inbounds nuw (i8, ptr @s_12638201494206808739, i64 8), i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  store ptr %35, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %45 = call ptr @GC_malloc(i64 24)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr %Res, ptr %46, i32 0, i32 0
  store ptr @Res..vtbl, ptr %47, align 8
  invoke void @Res.constructor(ptr %46, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %48 unwind label %90

48:                                               ; preds = %1
  store ptr %46, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %49 = load ptr, ptr @trail, align 8
  %50 = load ptr, ptr %4, align 8
  %51 = icmp eq ptr %50, null
  %52 = select i1 %51, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %50
  %53 = call i64 @strlen(ptr %52)
  %54 = add i64 1, %53
  %55 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %56 = add i64 %54, %55
  %57 = add i64 %56, 8
  %58 = call ptr @GC_malloc(i64 %57)
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr i8, ptr %59, i64 0
  %61 = sub i64 %56, 0
  %62 = call ptr @__memcpy_chk(ptr %60, ptr %52, i64 %53, i64 %61)
  %63 = add i64 0, %53
  %64 = getelementptr i8, ptr %59, i64 %63
  %65 = sub i64 %56, %63
  %66 = call ptr @__memcpy_chk(ptr %64, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %55, i64 %65)
  %67 = add i64 %63, %55
  %68 = getelementptr i8, ptr %59, i64 %67
  store i8 0, ptr %68, align 1
  %69 = icmp eq ptr %49, null
  %70 = select i1 %69, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %49
  %71 = icmp eq ptr %59, null
  %72 = select i1 %71, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %59
  %73 = call i64 @strlen(ptr %70)
  %74 = add i64 1, %73
  %75 = call i64 @strlen(ptr %72)
  %76 = add i64 %74, %75
  %77 = add i64 %76, 8
  %78 = call ptr @GC_malloc(i64 %77)
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr i8, ptr %79, i64 0
  %81 = sub i64 %76, 0
  %82 = call ptr @__memcpy_chk(ptr %80, ptr %70, i64 %73, i64 %81)
  %83 = add i64 0, %73
  %84 = getelementptr i8, ptr %79, i64 %83
  %85 = sub i64 %76, %83
  %86 = call ptr @__memcpy_chk(ptr %84, ptr %72, i64 %75, i64 %85)
  %87 = add i64 %83, %75
  %88 = getelementptr i8, ptr %79, i64 %87
  store i8 0, ptr %88, align 1
  store ptr %79, ptr @trail, align 8
  %89 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %89, align 4
  invoke void @__cxa_throw(ptr %89, ptr @_ZTIi, ptr null)
          to label %97 unwind label %90

90:                                               ; preds = %48, %1
  %91 = landingpad { ptr, i32 }
          cleanup
  %92 = load i1, ptr %2, align 1
  br i1 %92, label %93, label %96

93:                                               ; preds = %90
  store i1 false, ptr %2, align 1
  %94 = load ptr, ptr %3, align 8
  %95 = getelementptr %Res, ptr %94, i32 0, i32 0
  call void @Res.dispose(ptr %94)
  br label %96

96:                                               ; preds = %93, %90
  resume { ptr, i32 } %91

97:                                               ; preds = %48
  unreachable
}

; Function Attrs: mustprogress
define ptr @throwFromSwitchCase(ptr %0, double %1) #3 personality ptr @__gxx_personality_v0 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  store double %1, ptr %6, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = icmp eq ptr %9, null
  %11 = select i1 %10, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %9
  %12 = call i64 @strlen(ptr %11)
  %13 = add i64 1, %12
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638131125462603235, i64 8))
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
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12638131125462603235, i64 8), i64 %14, i64 %24)
  %26 = add i64 %22, %14
  %27 = getelementptr i8, ptr %18, i64 %26
  store i8 0, ptr %27, align 1
  store ptr %18, ptr %5, align 8
  %28 = load double, ptr %6, align 8
  store i1 false, ptr %3, align 1
  %29 = fptosi double %28 to i64
  %30 = trunc i64 %29 to i32
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %78

32:                                               ; preds = %2
  %33 = call ptr @GC_malloc(i64 24)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr %Res, ptr %34, i32 0, i32 0
  store ptr @Res..vtbl, ptr %35, align 8
  invoke void @Res.constructor(ptr %34, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %36 unwind label %87

36:                                               ; preds = %32
  store ptr %34, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %37 = load ptr, ptr @trail, align 8
  %38 = load ptr, ptr %5, align 8
  %39 = icmp eq ptr %38, null
  %40 = select i1 %39, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %38
  %41 = call i64 @strlen(ptr %40)
  %42 = add i64 1, %41
  %43 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %44 = add i64 %42, %43
  %45 = add i64 %44, 8
  %46 = call ptr @GC_malloc(i64 %45)
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr i8, ptr %47, i64 0
  %49 = sub i64 %44, 0
  %50 = call ptr @__memcpy_chk(ptr %48, ptr %40, i64 %41, i64 %49)
  %51 = add i64 0, %41
  %52 = getelementptr i8, ptr %47, i64 %51
  %53 = sub i64 %44, %51
  %54 = call ptr @__memcpy_chk(ptr %52, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %43, i64 %53)
  %55 = add i64 %51, %43
  %56 = getelementptr i8, ptr %47, i64 %55
  store i8 0, ptr %56, align 1
  %57 = icmp eq ptr %37, null
  %58 = select i1 %57, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %37
  %59 = icmp eq ptr %47, null
  %60 = select i1 %59, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %47
  %61 = call i64 @strlen(ptr %58)
  %62 = add i64 1, %61
  %63 = call i64 @strlen(ptr %60)
  %64 = add i64 %62, %63
  %65 = add i64 %64, 8
  %66 = call ptr @GC_malloc(i64 %65)
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr i8, ptr %67, i64 0
  %69 = sub i64 %64, 0
  %70 = call ptr @__memcpy_chk(ptr %68, ptr %58, i64 %61, i64 %69)
  %71 = add i64 0, %61
  %72 = getelementptr i8, ptr %67, i64 %71
  %73 = sub i64 %64, %71
  %74 = call ptr @__memcpy_chk(ptr %72, ptr %60, i64 %63, i64 %73)
  %75 = add i64 %71, %63
  %76 = getelementptr i8, ptr %67, i64 %75
  store i8 0, ptr %76, align 1
  store ptr %67, ptr @trail, align 8
  %77 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %77, align 4
  invoke void @__cxa_throw(ptr %77, ptr @_ZTIi, ptr null)
          to label %94 unwind label %87

78:                                               ; preds = %2
  %79 = load i1, ptr %3, align 1
  br i1 %79, label %80, label %84

80:                                               ; preds = %78
  store i1 false, ptr %3, align 1
  %81 = load ptr, ptr %4, align 8
  %82 = getelementptr %Res, ptr %81, i32 0, i32 0
  invoke void @Res.dispose(ptr %81)
          to label %83 unwind label %87

83:                                               ; preds = %80
  br label %84

84:                                               ; preds = %83, %78
  %85 = load ptr, ptr %5, align 8
  store ptr %85, ptr %8, align 8
  %86 = load ptr, ptr %8, align 8
  ret ptr %86

87:                                               ; preds = %36, %32, %80
  %88 = landingpad { ptr, i32 }
          cleanup
  %89 = load i1, ptr %3, align 1
  br i1 %89, label %90, label %93

90:                                               ; preds = %87
  store i1 false, ptr %3, align 1
  %91 = load ptr, ptr %4, align 8
  %92 = getelementptr %Res, ptr %91, i32 0, i32 0
  call void @Res.dispose(ptr %91)
  br label %93

93:                                               ; preds = %90, %87
  resume { ptr, i32 } %88

94:                                               ; preds = %36
  unreachable
}

; Function Attrs: mustprogress
define void @throwFromLoopBody(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  %8 = load ptr, ptr %7, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %8
  %11 = call i64 @strlen(ptr %10)
  %12 = add i64 1, %11
  %13 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638130025950975024, i64 8))
  %14 = add i64 %12, %13
  %15 = add i64 %14, 8
  %16 = call ptr @GC_malloc(i64 %15)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr i8, ptr %17, i64 0
  %19 = sub i64 %14, 0
  %20 = call ptr @__memcpy_chk(ptr %18, ptr %10, i64 %11, i64 %19)
  %21 = add i64 0, %11
  %22 = getelementptr i8, ptr %17, i64 %21
  %23 = sub i64 %14, %21
  %24 = call ptr @__memcpy_chk(ptr %22, ptr getelementptr inbounds nuw (i8, ptr @s_12638130025950975024, i64 8), i64 %13, i64 %23)
  %25 = add i64 %21, %13
  %26 = getelementptr i8, ptr %17, i64 %25
  store i8 0, ptr %26, align 1
  store ptr %17, ptr %6, align 8
  store i32 0, ptr %5, align 4
  br label %27

27:                                               ; preds = %92, %1
  %28 = load i32, ptr %5, align 4
  %29 = icmp slt i32 %28, 3
  br i1 %29, label %30, label %102

30:                                               ; preds = %27
  %31 = load ptr, ptr %6, align 8
  %32 = load i32, ptr %5, align 4
  %33 = call ptr @GC_malloc_atomic(i64 58)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %34, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %32)
  %36 = icmp eq ptr %31, null
  %37 = select i1 %36, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %31
  %38 = icmp eq ptr %34, null
  %39 = select i1 %38, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %34
  %40 = call i64 @strlen(ptr %37)
  %41 = add i64 1, %40
  %42 = call i64 @strlen(ptr %39)
  %43 = add i64 %41, %42
  %44 = add i64 %43, 8
  %45 = call ptr @GC_malloc(i64 %44)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr i8, ptr %46, i64 0
  %48 = sub i64 %43, 0
  %49 = call ptr @__memcpy_chk(ptr %47, ptr %37, i64 %40, i64 %48)
  %50 = add i64 0, %40
  %51 = getelementptr i8, ptr %46, i64 %50
  %52 = sub i64 %43, %50
  %53 = call ptr @__memcpy_chk(ptr %51, ptr %39, i64 %42, i64 %52)
  %54 = add i64 %50, %42
  %55 = getelementptr i8, ptr %46, i64 %54
  store i8 0, ptr %55, align 1
  store ptr %46, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %56 = call ptr @GC_malloc(i64 24)
  %57 = getelementptr i8, ptr %56, i64 8
  %58 = getelementptr %Res, ptr %57, i32 0, i32 0
  store ptr @Res..vtbl, ptr %58, align 8
  %59 = load i32, ptr %5, align 4
  %60 = call ptr @GC_malloc_atomic(i64 58)
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %61, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %59)
  %63 = icmp eq ptr %61, null
  %64 = select i1 %63, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %61
  %65 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %66 = add i64 1, %65
  %67 = call i64 @strlen(ptr %64)
  %68 = add i64 %66, %67
  %69 = add i64 %68, 8
  %70 = call ptr @GC_malloc(i64 %69)
  %71 = getelementptr i8, ptr %70, i64 8
  %72 = getelementptr i8, ptr %71, i64 0
  %73 = sub i64 %68, 0
  %74 = call ptr @__memcpy_chk(ptr %72, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i64 %65, i64 %73)
  %75 = add i64 0, %65
  %76 = getelementptr i8, ptr %71, i64 %75
  %77 = sub i64 %68, %75
  %78 = call ptr @__memcpy_chk(ptr %76, ptr %64, i64 %67, i64 %77)
  %79 = add i64 %75, %67
  %80 = getelementptr i8, ptr %71, i64 %79
  store i8 0, ptr %80, align 1
  invoke void @Res.constructor(ptr %57, ptr %71)
          to label %81 unwind label %95

81:                                               ; preds = %30
  store ptr %57, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %82 = load i32, ptr %5, align 4
  %83 = icmp eq i32 %82, 1
  br i1 %83, label %84, label %86

84:                                               ; preds = %81
  %85 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %85, align 4
  invoke void @__cxa_throw(ptr %85, ptr @_ZTIi, ptr null)
          to label %103 unwind label %95

86:                                               ; preds = %81
  %87 = load i1, ptr %2, align 1
  br i1 %87, label %88, label %92

88:                                               ; preds = %86
  store i1 false, ptr %2, align 1
  %89 = load ptr, ptr %3, align 8
  %90 = getelementptr %Res, ptr %89, i32 0, i32 0
  invoke void @Res.dispose(ptr %89)
          to label %91 unwind label %95

91:                                               ; preds = %88
  br label %92

92:                                               ; preds = %91, %86
  %93 = load i32, ptr %5, align 4
  %94 = add i32 %93, 1
  store i32 %94, ptr %5, align 4
  br label %27

95:                                               ; preds = %84, %88, %30
  %96 = landingpad { ptr, i32 }
          cleanup
  %97 = load i1, ptr %2, align 1
  br i1 %97, label %98, label %101

98:                                               ; preds = %95
  store i1 false, ptr %2, align 1
  %99 = load ptr, ptr %3, align 8
  %100 = getelementptr %Res, ptr %99, i32 0, i32 0
  call void @Res.dispose(ptr %99)
  br label %101

101:                                              ; preds = %98, %95
  resume { ptr, i32 } %96

102:                                              ; preds = %27
  ret void

103:                                              ; preds = %84
  unreachable
}

; Function Attrs: mustprogress
define ptr @caughtFurtherOut(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %7, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = icmp eq ptr %9, null
  %11 = select i1 %10, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %9
  %12 = call i64 @strlen(ptr %11)
  %13 = add i64 1, %12
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638133324485859657, i64 8))
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
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12638133324485859657, i64 8), i64 %14, i64 %24)
  %26 = add i64 %22, %14
  %27 = getelementptr i8, ptr %18, i64 %26
  store i8 0, ptr %27, align 1
  store ptr %18, ptr %6, align 8
  %28 = load ptr, ptr %6, align 8
  %29 = icmp eq ptr %28, null
  %30 = select i1 %29, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %28
  %31 = call i64 @strlen(ptr %30)
  %32 = add i64 1, %31
  %33 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638201494206808739, i64 8))
  %34 = add i64 %32, %33
  %35 = add i64 %34, 8
  %36 = call ptr @GC_malloc(i64 %35)
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr i8, ptr %37, i64 0
  %39 = sub i64 %34, 0
  %40 = call ptr @__memcpy_chk(ptr %38, ptr %30, i64 %31, i64 %39)
  %41 = add i64 0, %31
  %42 = getelementptr i8, ptr %37, i64 %41
  %43 = sub i64 %34, %41
  %44 = call ptr @__memcpy_chk(ptr %42, ptr getelementptr inbounds nuw (i8, ptr @s_12638201494206808739, i64 8), i64 %33, i64 %43)
  %45 = add i64 %41, %33
  %46 = getelementptr i8, ptr %37, i64 %45
  store i8 0, ptr %46, align 1
  store ptr %37, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %47 = call ptr @GC_malloc(i64 24)
  %48 = getelementptr i8, ptr %47, i64 8
  %49 = getelementptr %Res, ptr %48, i32 0, i32 0
  store ptr @Res..vtbl, ptr %49, align 8
  invoke void @Res.constructor(ptr %48, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %50 unwind label %52

50:                                               ; preds = %1
  store ptr %48, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %51 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %51, align 4
  invoke void @__cxa_throw(ptr %51, ptr @_ZTIi, ptr null)
          to label %114 unwind label %52

52:                                               ; preds = %50, %1
  %53 = landingpad { ptr, i32 }
          catch ptr null
  %54 = extractvalue { ptr, i32 } %53, 0
  %55 = call ptr @__cxa_begin_catch(ptr %54)
  %56 = load i1, ptr %2, align 1
  br i1 %56, label %57, label %61

57:                                               ; preds = %52
  store i1 false, ptr %2, align 1
  %58 = load ptr, ptr %3, align 8
  %59 = getelementptr %Res, ptr %58, i32 0, i32 0
  invoke void @Res.dispose(ptr %58)
          to label %60 unwind label %endcatch.pad4

60:                                               ; preds = %57
  br label %61

61:                                               ; preds = %60, %52
  invoke void @__cxa_rethrow()
          to label %114 unwind label %endcatch.pad2

endcatch.pad4:                                    ; preds = %57
  %62 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

63:                                               ; No predecessors!
  %64 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %63
  %65 = phi { ptr, i32 } [ %64, %63 ], [ %62, %endcatch.pad4 ]
  %66 = extractvalue { ptr, i32 } %65, 0
  %67 = call ptr @__cxa_begin_catch(ptr %66)
  %68 = load i1, ptr %2, align 1
  br i1 %68, label %69, label %72

69:                                               ; preds = %.caught3
  store i1 false, ptr %2, align 1
  %70 = load ptr, ptr %3, align 8
  %71 = getelementptr %Res, ptr %70, i32 0, i32 0
  invoke void @Res.dispose(ptr %70)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %69
  br label %72

72:                                               ; preds = %.noexc, %.caught3
  invoke void @__cxa_rethrow()
          to label %114 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %61, %72
  %73 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

74:                                               ; No predecessors!
  %75 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %74
  %76 = phi { ptr, i32 } [ %75, %74 ], [ %73, %endcatch.pad2 ]
  %77 = extractvalue { ptr, i32 } %76, 1
  %78 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %79 = icmp eq i32 %77, %78
  br i1 %79, label %83, label %80

80:                                               ; preds = %.caught1
  %81 = extractvalue { ptr, i32 } %76, 0
  %82 = call ptr @__cxa_begin_catch(ptr %81)
  invoke void @__cxa_rethrow()
          to label %114 unwind label %endcatch.pad

83:                                               ; preds = %.caught1
  %84 = extractvalue { ptr, i32 } %76, 0
  %85 = call ptr @__cxa_begin_catch(ptr %84)
  %86 = load i32, ptr %85, align 4
  store i32 %86, ptr %5, align 4
  %87 = load ptr, ptr @trail, align 8
  %88 = icmp eq ptr %87, null
  %89 = select i1 %88, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %87
  %90 = invoke i64 @strlen(ptr %89)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %83
  %91 = add i64 1, %90
  %92 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8))
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %93 = add i64 %91, %92
  %94 = add i64 %93, 8
  %95 = invoke ptr @GC_malloc(i64 %94)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %96 = getelementptr i8, ptr %95, i64 8
  %97 = getelementptr i8, ptr %96, i64 0
  %98 = sub i64 %93, 0
  %99 = invoke ptr @__memcpy_chk(ptr %97, ptr %89, i64 %90, i64 %98)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %100 = add i64 0, %90
  %101 = getelementptr i8, ptr %96, i64 %100
  %102 = sub i64 %93, %100
  %103 = invoke ptr @__memcpy_chk(ptr %101, ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8), i64 %92, i64 %102)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %104 = add i64 %100, %92
  %105 = getelementptr i8, ptr %96, i64 %104
  store i8 0, ptr %105, align 1
  store ptr %96, ptr @trail, align 8
  call void @__cxa_end_catch()
  %106 = invoke i64 @churn()
          to label %107 unwind label %111

107:                                              ; preds = %.noexc9
  %108 = load ptr, ptr %6, align 8
  store ptr %108, ptr %8, align 8
  %109 = load ptr, ptr %8, align 8
  ret ptr %109

endcatch.pad:                                     ; preds = %80
  %110 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

111:                                              ; preds = %.noexc9
  %112 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %111
  %113 = phi { ptr, i32 } [ %112, %111 ], [ %110, %endcatch.pad ]
  resume { ptr, i32 } %113

114:                                              ; preds = %80, %61, %72, %50
  unreachable

endcatch.resume:                                  ; preds = %.noexc8, %.noexc7, %.noexc6, %.noexc5, %83, %69
  %115 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %115
}

; Function Attrs: mustprogress
define ptr @caughtFurtherOutSwitch(ptr %0, double %1) #3 personality ptr @__gxx_personality_v0 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca double, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %8, align 8
  store double %1, ptr %7, align 8
  %10 = load ptr, ptr %8, align 8
  %11 = icmp eq ptr %10, null
  %12 = select i1 %11, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %10
  %13 = call i64 @strlen(ptr %12)
  %14 = add i64 1, %13
  %15 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638132224974231446, i64 8))
  %16 = add i64 %14, %15
  %17 = add i64 %16, 8
  %18 = call ptr @GC_malloc(i64 %17)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr i8, ptr %19, i64 0
  %21 = sub i64 %16, 0
  %22 = call ptr @__memcpy_chk(ptr %20, ptr %12, i64 %13, i64 %21)
  %23 = add i64 0, %13
  %24 = getelementptr i8, ptr %19, i64 %23
  %25 = sub i64 %16, %23
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638132224974231446, i64 8), i64 %15, i64 %25)
  %27 = add i64 %23, %15
  %28 = getelementptr i8, ptr %19, i64 %27
  store i8 0, ptr %28, align 1
  store ptr %19, ptr %6, align 8
  %29 = load double, ptr %7, align 8
  store i1 false, ptr %3, align 1
  %30 = fptosi double %29 to i64
  %31 = trunc i64 %30 to i32
  %32 = icmp eq i32 %31, 1
  br i1 %32, label %33, label %39

33:                                               ; preds = %2
  %34 = call ptr @GC_malloc(i64 24)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr %Res, ptr %35, i32 0, i32 0
  store ptr @Res..vtbl, ptr %36, align 8
  invoke void @Res.constructor(ptr %35, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %37 unwind label %46

37:                                               ; preds = %33
  store ptr %35, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %38 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %38, align 4
  invoke void @__cxa_throw(ptr %38, ptr @_ZTIi, ptr null)
          to label %109 unwind label %46

39:                                               ; preds = %2
  %40 = load i1, ptr %3, align 1
  br i1 %40, label %41, label %45

41:                                               ; preds = %39
  store i1 false, ptr %3, align 1
  %42 = load ptr, ptr %4, align 8
  %43 = getelementptr %Res, ptr %42, i32 0, i32 0
  invoke void @Res.dispose(ptr %42)
          to label %44 unwind label %46

44:                                               ; preds = %41
  br label %45

45:                                               ; preds = %44, %39
  br label %100

46:                                               ; preds = %37, %33, %41
  %47 = landingpad { ptr, i32 }
          catch ptr null
  %48 = extractvalue { ptr, i32 } %47, 0
  %49 = call ptr @__cxa_begin_catch(ptr %48)
  %50 = load i1, ptr %3, align 1
  br i1 %50, label %51, label %55

51:                                               ; preds = %46
  store i1 false, ptr %3, align 1
  %52 = load ptr, ptr %4, align 8
  %53 = getelementptr %Res, ptr %52, i32 0, i32 0
  invoke void @Res.dispose(ptr %52)
          to label %54 unwind label %endcatch.pad4

54:                                               ; preds = %51
  br label %55

55:                                               ; preds = %54, %46
  invoke void @__cxa_rethrow()
          to label %109 unwind label %endcatch.pad2

endcatch.pad4:                                    ; preds = %51
  %56 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

57:                                               ; No predecessors!
  %58 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %57
  %59 = phi { ptr, i32 } [ %58, %57 ], [ %56, %endcatch.pad4 ]
  %60 = extractvalue { ptr, i32 } %59, 0
  %61 = call ptr @__cxa_begin_catch(ptr %60)
  %62 = load i1, ptr %3, align 1
  br i1 %62, label %63, label %66

63:                                               ; preds = %.caught3
  store i1 false, ptr %3, align 1
  %64 = load ptr, ptr %4, align 8
  %65 = getelementptr %Res, ptr %64, i32 0, i32 0
  invoke void @Res.dispose(ptr %64)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %63
  br label %66

66:                                               ; preds = %.noexc, %.caught3
  invoke void @__cxa_rethrow()
          to label %109 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %66, %55
  %67 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

68:                                               ; No predecessors!
  %69 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %68
  %70 = phi { ptr, i32 } [ %69, %68 ], [ %67, %endcatch.pad2 ]
  %71 = extractvalue { ptr, i32 } %70, 1
  %72 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %73 = icmp eq i32 %71, %72
  br i1 %73, label %77, label %74

74:                                               ; preds = %.caught1
  %75 = extractvalue { ptr, i32 } %70, 0
  %76 = call ptr @__cxa_begin_catch(ptr %75)
  invoke void @__cxa_rethrow()
          to label %109 unwind label %endcatch.pad

77:                                               ; preds = %.caught1
  %78 = extractvalue { ptr, i32 } %70, 0
  %79 = call ptr @__cxa_begin_catch(ptr %78)
  %80 = load i32, ptr %79, align 4
  store i32 %80, ptr %5, align 4
  %81 = load ptr, ptr @trail, align 8
  %82 = icmp eq ptr %81, null
  %83 = select i1 %82, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %81
  %84 = invoke i64 @strlen(ptr %83)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %77
  %85 = add i64 1, %84
  %86 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8))
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %87 = add i64 %85, %86
  %88 = add i64 %87, 8
  %89 = invoke ptr @GC_malloc(i64 %88)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %90 = getelementptr i8, ptr %89, i64 8
  %91 = getelementptr i8, ptr %90, i64 0
  %92 = sub i64 %87, 0
  %93 = invoke ptr @__memcpy_chk(ptr %91, ptr %83, i64 %84, i64 %92)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %94 = add i64 0, %84
  %95 = getelementptr i8, ptr %90, i64 %94
  %96 = sub i64 %87, %94
  %97 = invoke ptr @__memcpy_chk(ptr %95, ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8), i64 %86, i64 %96)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %98 = add i64 %94, %86
  %99 = getelementptr i8, ptr %90, i64 %98
  store i8 0, ptr %99, align 1
  store ptr %90, ptr @trail, align 8
  call void @__cxa_end_catch()
  br label %100

100:                                              ; preds = %.noexc9, %45
  %101 = invoke i64 @churn()
          to label %102 unwind label %106

102:                                              ; preds = %100
  %103 = load ptr, ptr %6, align 8
  store ptr %103, ptr %9, align 8
  %104 = load ptr, ptr %9, align 8
  ret ptr %104

endcatch.pad:                                     ; preds = %74
  %105 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

106:                                              ; preds = %100
  %107 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %106
  %108 = phi { ptr, i32 } [ %107, %106 ], [ %105, %endcatch.pad ]
  resume { ptr, i32 } %108

109:                                              ; preds = %74, %55, %66, %37
  unreachable

endcatch.resume:                                  ; preds = %.noexc8, %.noexc7, %.noexc6, %.noexc5, %77, %63
  %110 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %110
}

; Function Attrs: mustprogress
define i1 @expectThrow({ ptr, ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca i1, align 1
  store { ptr, ptr, ptr } %0, ptr %3, align 8
  %5 = load { ptr, ptr, ptr }, ptr %3, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %10(ptr %6)
          to label %11 unwind label %18

11:                                               ; preds = %9
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %13()
          to label %14 unwind label %18

14:                                               ; preds = %12
  br label %15

15:                                               ; preds = %11, %14
  br label %16

16:                                               ; preds = %15
  br label %17

17:                                               ; preds = %16
  store i1 false, ptr %4, align 1
  br label %30

18:                                               ; preds = %9, %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %23
  br label %32

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %2, align 4
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %30

30:                                               ; preds = %17, %26
  %31 = load i1, ptr %4, align 1
  ret i1 %31

32:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %23
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define void @check({ ptr, ptr, ptr } %0, ptr %1, ptr %2) #2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca { ptr, ptr, ptr }, align 8
  store { ptr, ptr, ptr } %0, ptr %6, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %3
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  call void %12(ptr %8)
  br label %15

13:                                               ; preds = %3
  %14 = extractvalue { ptr, ptr, ptr } %7, 0
  call void %14()
  br label %15

15:                                               ; preds = %11, %13
  br label %16

16:                                               ; preds = %15
  %17 = call i64 @churn()
  %18 = load ptr, ptr %4, align 8
  %19 = load ptr, ptr @trail, align 8
  %20 = icmp eq ptr %18, null
  %21 = select i1 %20, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %18
  %22 = icmp eq ptr %19, null
  %23 = select i1 %22, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %19
  %24 = call i64 @strlen(ptr %21)
  %25 = add i64 1, %24
  %26 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_584909432750204567, i64 8))
  %27 = add i64 %25, %26
  %28 = call i64 @strlen(ptr %23)
  %29 = add i64 %27, %28
  %30 = add i64 %29, 8
  %31 = call ptr @GC_malloc(i64 %30)
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr i8, ptr %32, i64 0
  %34 = sub i64 %29, 0
  %35 = call ptr @__memcpy_chk(ptr %33, ptr %21, i64 %24, i64 %34)
  %36 = add i64 0, %24
  %37 = getelementptr i8, ptr %32, i64 %36
  %38 = sub i64 %29, %36
  %39 = call ptr @__memcpy_chk(ptr %37, ptr getelementptr inbounds nuw (i8, ptr @s_584909432750204567, i64 8), i64 %26, i64 %38)
  %40 = add i64 %36, %26
  %41 = getelementptr i8, ptr %32, i64 %40
  %42 = sub i64 %29, %40
  %43 = call ptr @__memcpy_chk(ptr %41, ptr %23, i64 %28, i64 %42)
  %44 = add i64 %40, %28
  %45 = getelementptr i8, ptr %32, i64 %44
  store i8 0, ptr %45, align 1
  %46 = icmp eq ptr %32, null
  %47 = select i1 %46, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %32
  call void @puts(ptr %47)
  %48 = load ptr, ptr @trail, align 8
  %49 = load ptr, ptr %5, align 8
  %50 = ptrtoint ptr %48 to i64
  %51 = ptrtoint ptr %49 to i64
  %52 = icmp ne i64 %50, 0
  %53 = icmp ne i64 %51, 0
  %54 = zext i1 %52 to i32
  %55 = zext i1 %53 to i32
  %56 = and i32 %54, %55
  %57 = icmp ne i32 %56, 0
  br i1 %57, label %58, label %61

58:                                               ; preds = %16
  %59 = call i32 @strcmp(ptr %48, ptr %49)
  %60 = icmp eq i32 %59, 0
  br label %63

61:                                               ; preds = %16
  %62 = icmp eq i64 %50, %51
  br label %63

63:                                               ; preds = %58, %61
  %64 = phi i1 [ %62, %61 ], [ %60, %58 ]
  br label %65

65:                                               ; preds = %63
  %66 = load ptr, ptr %4, align 8
  br i1 %64, label %67, label %69

67:                                               ; preds = %65
  ret void

68:                                               ; preds = %69
  unreachable

69:                                               ; preds = %65
  %70 = call i32 @fflush(ptr null)
  %71 = icmp eq ptr %66, null
  %72 = select i1 %71, ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr %66
  call void @__assert_fail(ptr %72, ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 139, ptr null)
  br label %68
}

; Function Attrs: mustprogress
define void @.f_main..f_.afL151C11FH1005041793..afL151C36FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  call void @throwFromBlock(ptr %4)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL151C11FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %3, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..f_.afL151C11FH1005041793..afL151C36FH1005041793, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  %9 = call i1 @expectThrow({ ptr, ptr, ptr } %8)
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  ret void

11:                                               ; preds = %12
  unreachable

12:                                               ; preds = %1
  %13 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2432362340776716864, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 151, ptr null)
  br label %11
}

; Function Attrs: mustprogress
define void @.f_main..f_.afL152C11FH1005041793..afL152C36FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  call void @throwThroughPlainScope(ptr %4)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL152C11FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %3, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..f_.afL152C11FH1005041793..afL152C36FH1005041793, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  %9 = call i1 @expectThrow({ ptr, ptr, ptr } %8)
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  ret void

11:                                               ; preds = %12
  unreachable

12:                                               ; preds = %1
  %13 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2432362340776716864, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 152, ptr null)
  br label %11
}

; Function Attrs: mustprogress
define void @.f_main..f_.afL153C11FH1005041793..afL153C36FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call ptr @throwFromSwitchCase(ptr %4, double 1.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL153C11FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %3, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..f_.afL153C11FH1005041793..afL153C36FH1005041793, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  %9 = call i1 @expectThrow({ ptr, ptr, ptr } %8)
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  ret void

11:                                               ; preds = %12
  unreachable

12:                                               ; preds = %1
  %13 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2432362340776716864, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 153, ptr null)
  br label %11
}

; Function Attrs: mustprogress
define void @.f_main..f_.afL154C11FH1005041793..afL154C36FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  call void @throwFromLoopBody(ptr %4)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL154C11FH1005041793(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %3, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..f_.afL154C11FH1005041793..afL154C36FH1005041793, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  %9 = call i1 @expectThrow({ ptr, ptr, ptr } %8)
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  ret void

11:                                               ; preds = %12
  unreachable

12:                                               ; preds = %1
  %13 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2432362340776716864, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 154, ptr null)
  br label %11
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, { ptr } }>, align 16
  %4 = alloca <{ i64, { ptr } }>, align 16
  %5 = alloca <{ i64, { ptr } }>, align 16
  %6 = alloca <{ i64, { ptr } }>, align 16
  %7 = alloca ptr, align 8
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638205892253321583, i64 8), ptr %9, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %10 = load ptr, ptr %9, align 8
  %11 = call ptr @returnFromBlock(ptr %10)
  store ptr %11, ptr %7, align 8
  %12 = call i64 @churn()
  %13 = load ptr, ptr %7, align 8
  %14 = load ptr, ptr @trail, align 8
  %15 = icmp eq ptr %13, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %13
  %17 = icmp eq ptr %14, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_11862283273679587948, i64 8))
  %20 = add i64 1, %19
  %21 = call i64 @strlen(ptr %16)
  %22 = add i64 %20, %21
  %23 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %24 = add i64 %22, %23
  %25 = call i64 @strlen(ptr %18)
  %26 = add i64 %24, %25
  %27 = add i64 %26, 8
  %28 = call ptr @GC_malloc(i64 %27)
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr i8, ptr %29, i64 0
  %31 = sub i64 %26, 0
  %32 = call ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_11862283273679587948, i64 8), i64 %19, i64 %31)
  %33 = add i64 0, %19
  %34 = getelementptr i8, ptr %29, i64 %33
  %35 = sub i64 %26, %33
  %36 = call ptr @__memcpy_chk(ptr %34, ptr %16, i64 %21, i64 %35)
  %37 = add i64 %33, %21
  %38 = getelementptr i8, ptr %29, i64 %37
  %39 = sub i64 %26, %37
  %40 = call ptr @__memcpy_chk(ptr %38, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %23, i64 %39)
  %41 = add i64 %37, %23
  %42 = getelementptr i8, ptr %29, i64 %41
  %43 = sub i64 %26, %41
  %44 = call ptr @__memcpy_chk(ptr %42, ptr %18, i64 %25, i64 %43)
  %45 = add i64 %41, %25
  %46 = getelementptr i8, ptr %29, i64 %45
  store i8 0, ptr %46, align 1
  %47 = icmp eq ptr %29, null
  %48 = select i1 %47, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %29
  call void @puts(ptr %48)
  %49 = load ptr, ptr %7, align 8
  %50 = ptrtoint ptr %49 to i64
  %51 = icmp ne i64 %50, 0
  %52 = zext i1 %51 to i32
  %53 = and i32 %52, 1
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %55, label %58

55:                                               ; preds = %0
  %56 = call i32 @strcmp(ptr %49, ptr getelementptr inbounds nuw (i8, ptr @s_636580981707762362, i64 8))
  %57 = icmp eq i32 %56, 0
  br label %60

58:                                               ; preds = %0
  %59 = icmp eq i64 %50, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_636580981707762362, i64 8) to i64)
  br label %60

60:                                               ; preds = %55, %58
  %61 = phi i1 [ %59, %58 ], [ %57, %55 ]
  br label %62

62:                                               ; preds = %60
  br i1 %61, label %63, label %78

63:                                               ; preds = %62
  %64 = load ptr, ptr @trail, align 8
  %65 = ptrtoint ptr %64 to i64
  %66 = icmp ne i64 %65, 0
  %67 = zext i1 %66 to i32
  %68 = and i32 %67, 1
  %69 = icmp ne i32 %68, 0
  br i1 %69, label %70, label %73

70:                                               ; preds = %63
  %71 = call i32 @strcmp(ptr %64, ptr getelementptr inbounds nuw (i8, ptr @s_620385175427431157, i64 8))
  %72 = icmp eq i32 %71, 0
  br label %75

73:                                               ; preds = %63
  %74 = icmp eq i64 %65, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_620385175427431157, i64 8) to i64)
  br label %75

75:                                               ; preds = %70, %73
  %76 = phi i1 [ %74, %73 ], [ %72, %70 ]
  br label %77

77:                                               ; preds = %75
  br label %78

78:                                               ; preds = %77, %62
  %79 = phi i1 [ %76, %77 ], [ %61, %62 ]
  br i1 %79, label %80, label %238

80:                                               ; preds = %78
  store i64 -1, ptr %6, align 8
  %81 = getelementptr i8, ptr %6, i64 8
  store { ptr } zeroinitializer, ptr %81, align 8
  %82 = getelementptr { ptr }, ptr %81, i32 0, i32 0
  store ptr %9, ptr %82, align 8
  %83 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL151C11FH1005041793, ptr null, ptr null }, ptr %81, 1
  %84 = insertvalue { ptr, ptr, ptr } %83, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  call void @check({ ptr, ptr, ptr } %84, ptr getelementptr inbounds nuw (i8, ptr @s_14968365627901597948, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_1254165271164641062, i64 8))
  store i64 -1, ptr %5, align 8
  %85 = getelementptr i8, ptr %5, i64 8
  store { ptr } zeroinitializer, ptr %85, align 8
  %86 = getelementptr { ptr }, ptr %85, i32 0, i32 0
  store ptr %9, ptr %86, align 8
  %87 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL152C11FH1005041793, ptr null, ptr null }, ptr %85, 1
  %88 = insertvalue { ptr, ptr, ptr } %87, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  call void @check({ ptr, ptr, ptr } %88, ptr getelementptr inbounds nuw (i8, ptr @s_11046246514244802049, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14630895708312341009, i64 8))
  store i64 -1, ptr %4, align 8
  %89 = getelementptr i8, ptr %4, i64 8
  store { ptr } zeroinitializer, ptr %89, align 8
  %90 = getelementptr { ptr }, ptr %89, i32 0, i32 0
  store ptr %9, ptr %90, align 8
  %91 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL153C11FH1005041793, ptr null, ptr null }, ptr %89, 1
  %92 = insertvalue { ptr, ptr, ptr } %91, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  call void @check({ ptr, ptr, ptr } %92, ptr getelementptr inbounds nuw (i8, ptr @s_9925662178766230706, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_2521489369436186603, i64 8))
  store i64 -1, ptr %3, align 8
  %93 = getelementptr i8, ptr %3, i64 8
  store { ptr } zeroinitializer, ptr %93, align 8
  %94 = getelementptr { ptr }, ptr %93, i32 0, i32 0
  store ptr %9, ptr %94, align 8
  %95 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL154C11FH1005041793, ptr null, ptr null }, ptr %93, 1
  %96 = insertvalue { ptr, ptr, ptr } %95, ptr getelementptr inbounds nuw (i8, ptr @td_13071058665056268293_object, i64 40), 2
  call void @check({ ptr, ptr, ptr } %96, ptr getelementptr inbounds nuw (i8, ptr @s_15882658668006414508, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_12917172910269586205, i64 8))
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %97 = load ptr, ptr %9, align 8
  %98 = call ptr @caughtFurtherOut(ptr %97)
  store ptr %98, ptr %2, align 8
  %99 = load ptr, ptr %2, align 8
  %100 = load ptr, ptr @trail, align 8
  %101 = icmp eq ptr %99, null
  %102 = select i1 %101, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %99
  %103 = icmp eq ptr %100, null
  %104 = select i1 %103, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %100
  %105 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_2773407230609657009, i64 8))
  %106 = add i64 1, %105
  %107 = call i64 @strlen(ptr %102)
  %108 = add i64 %106, %107
  %109 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %110 = add i64 %108, %109
  %111 = call i64 @strlen(ptr %104)
  %112 = add i64 %110, %111
  %113 = add i64 %112, 8
  %114 = call ptr @GC_malloc(i64 %113)
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr i8, ptr %115, i64 0
  %117 = sub i64 %112, 0
  %118 = call ptr @__memcpy_chk(ptr %116, ptr getelementptr inbounds nuw (i8, ptr @s_2773407230609657009, i64 8), i64 %105, i64 %117)
  %119 = add i64 0, %105
  %120 = getelementptr i8, ptr %115, i64 %119
  %121 = sub i64 %112, %119
  %122 = call ptr @__memcpy_chk(ptr %120, ptr %102, i64 %107, i64 %121)
  %123 = add i64 %119, %107
  %124 = getelementptr i8, ptr %115, i64 %123
  %125 = sub i64 %112, %123
  %126 = call ptr @__memcpy_chk(ptr %124, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %109, i64 %125)
  %127 = add i64 %123, %109
  %128 = getelementptr i8, ptr %115, i64 %127
  %129 = sub i64 %112, %127
  %130 = call ptr @__memcpy_chk(ptr %128, ptr %104, i64 %111, i64 %129)
  %131 = add i64 %127, %111
  %132 = getelementptr i8, ptr %115, i64 %131
  store i8 0, ptr %132, align 1
  %133 = icmp eq ptr %115, null
  %134 = select i1 %133, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %115
  call void @puts(ptr %134)
  %135 = load ptr, ptr %2, align 8
  %136 = ptrtoint ptr %135 to i64
  %137 = icmp ne i64 %136, 0
  %138 = zext i1 %137 to i32
  %139 = and i32 %138, 1
  %140 = icmp ne i32 %139, 0
  br i1 %140, label %141, label %144

141:                                              ; preds = %80
  %142 = call i32 @strcmp(ptr %135, ptr getelementptr inbounds nuw (i8, ptr @s_636575484149621307, i64 8))
  %143 = icmp eq i32 %142, 0
  br label %146

144:                                              ; preds = %80
  %145 = icmp eq i64 %136, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_636575484149621307, i64 8) to i64)
  br label %146

146:                                              ; preds = %141, %144
  %147 = phi i1 [ %145, %144 ], [ %143, %141 ]
  br label %148

148:                                              ; preds = %146
  br i1 %147, label %149, label %164

149:                                              ; preds = %148
  %150 = load ptr, ptr @trail, align 8
  %151 = ptrtoint ptr %150 to i64
  %152 = icmp ne i64 %151, 0
  %153 = zext i1 %152 to i32
  %154 = and i32 %153, 1
  %155 = icmp ne i32 %154, 0
  br i1 %155, label %156, label %159

156:                                              ; preds = %149
  %157 = call i32 @strcmp(ptr %150, ptr getelementptr inbounds nuw (i8, ptr @s_3019407056386408938, i64 8))
  %158 = icmp eq i32 %157, 0
  br label %161

159:                                              ; preds = %149
  %160 = icmp eq i64 %151, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3019407056386408938, i64 8) to i64)
  br label %161

161:                                              ; preds = %156, %159
  %162 = phi i1 [ %160, %159 ], [ %158, %156 ]
  br label %163

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %163, %148
  %165 = phi i1 [ %162, %163 ], [ %147, %148 ]
  br i1 %165, label %166, label %240

166:                                              ; preds = %164
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %167 = load ptr, ptr %9, align 8
  %168 = call ptr @caughtFurtherOutSwitch(ptr %167, double 1.000000e+00)
  store ptr %168, ptr %1, align 8
  %169 = load ptr, ptr %1, align 8
  %170 = load ptr, ptr @trail, align 8
  %171 = icmp eq ptr %169, null
  %172 = select i1 %171, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %169
  %173 = icmp eq ptr %170, null
  %174 = select i1 %173, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %170
  %175 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_7824925360632822997, i64 8))
  %176 = add i64 1, %175
  %177 = call i64 @strlen(ptr %172)
  %178 = add i64 %176, %177
  %179 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %180 = add i64 %178, %179
  %181 = call i64 @strlen(ptr %174)
  %182 = add i64 %180, %181
  %183 = add i64 %182, 8
  %184 = call ptr @GC_malloc(i64 %183)
  %185 = getelementptr i8, ptr %184, i64 8
  %186 = getelementptr i8, ptr %185, i64 0
  %187 = sub i64 %182, 0
  %188 = call ptr @__memcpy_chk(ptr %186, ptr getelementptr inbounds nuw (i8, ptr @s_7824925360632822997, i64 8), i64 %175, i64 %187)
  %189 = add i64 0, %175
  %190 = getelementptr i8, ptr %185, i64 %189
  %191 = sub i64 %182, %189
  %192 = call ptr @__memcpy_chk(ptr %190, ptr %172, i64 %177, i64 %191)
  %193 = add i64 %189, %177
  %194 = getelementptr i8, ptr %185, i64 %193
  %195 = sub i64 %182, %193
  %196 = call ptr @__memcpy_chk(ptr %194, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %179, i64 %195)
  %197 = add i64 %193, %179
  %198 = getelementptr i8, ptr %185, i64 %197
  %199 = sub i64 %182, %197
  %200 = call ptr @__memcpy_chk(ptr %198, ptr %174, i64 %181, i64 %199)
  %201 = add i64 %197, %181
  %202 = getelementptr i8, ptr %185, i64 %201
  store i8 0, ptr %202, align 1
  %203 = icmp eq ptr %185, null
  %204 = select i1 %203, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %185
  call void @puts(ptr %204)
  %205 = load ptr, ptr %1, align 8
  %206 = ptrtoint ptr %205 to i64
  %207 = icmp ne i64 %206, 0
  %208 = zext i1 %207 to i32
  %209 = and i32 %208, 1
  %210 = icmp ne i32 %209, 0
  br i1 %210, label %211, label %214

211:                                              ; preds = %166
  %212 = call i32 @strcmp(ptr %205, ptr getelementptr inbounds nuw (i8, ptr @s_636574384637993096, i64 8))
  %213 = icmp eq i32 %212, 0
  br label %216

214:                                              ; preds = %166
  %215 = icmp eq i64 %206, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_636574384637993096, i64 8) to i64)
  br label %216

216:                                              ; preds = %211, %214
  %217 = phi i1 [ %215, %214 ], [ %213, %211 ]
  br label %218

218:                                              ; preds = %216
  br i1 %217, label %219, label %234

219:                                              ; preds = %218
  %220 = load ptr, ptr @trail, align 8
  %221 = ptrtoint ptr %220 to i64
  %222 = icmp ne i64 %221, 0
  %223 = zext i1 %222 to i32
  %224 = and i32 %223, 1
  %225 = icmp ne i32 %224, 0
  br i1 %225, label %226, label %229

226:                                              ; preds = %219
  %227 = call i32 @strcmp(ptr %220, ptr getelementptr inbounds nuw (i8, ptr @s_3019407056386408938, i64 8))
  %228 = icmp eq i32 %227, 0
  br label %231

229:                                              ; preds = %219
  %230 = icmp eq i64 %221, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3019407056386408938, i64 8) to i64)
  br label %231

231:                                              ; preds = %226, %229
  %232 = phi i1 [ %230, %229 ], [ %228, %226 ]
  br label %233

233:                                              ; preds = %231
  br label %234

234:                                              ; preds = %233, %218
  %235 = phi i1 [ %232, %233 ], [ %217, %218 ]
  br i1 %235, label %236, label %242

236:                                              ; preds = %234
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

237:                                              ; preds = %242, %240, %238
  unreachable

238:                                              ; preds = %78
  %239 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8074741577383239958, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 149, ptr null)
  br label %237

240:                                              ; preds = %164
  %241 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10267497238324938131, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 159, ptr null)
  br label %237

242:                                              ; preds = %234
  %243 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16285134945708457239, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12376869648902559890, i64 8), i32 164, ptr null)
  br label %237
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
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

