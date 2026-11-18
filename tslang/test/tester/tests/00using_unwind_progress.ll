; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }
%Boom = type { ptr }
%ThrowsOnDispose = type { ptr }

$Res..size = comdat any

$Boom..size = comdat any

$ThrowsOnDispose..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_17851658315048716010 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBoom\00", align 8
@s_1872836310818908366 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFThrowsOnDispose\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_11111286348956853952 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe control: a normal exit disposes once\00", align 8
@f_2662995175179872379 = internal constant [113 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00using_unwind_progress.ts\00", align 8
@m_12601458686632539999 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe control: the body ran\00", align 8
@m_16373880600585569923 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunreachable: the function throws\00", align 8
@m_2343790244925707336 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe cleanup disposes what the body stored\00", align 8
@m_15951051187737246770 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe body ran and the throw was caught\00", align 8
@m_13099800921096224375 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunreachable: the constructor throws\00", align 8
@m_6585847342210022013 = internal constant [60 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa constructor that throws leaves nothing to dispose\00", align 8
@m_7063307771149137198 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe body never ran, and the throw was caught\00", align 8
@m_18172175995809474191 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunreachable: the second constructor throws\00", align 8
@m_8081476958950538590 = internal constant [78 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe declaration that completed is disposed, the one that threw is not\00", align 8
@m_14032912357038811970 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunreachable: the disposal throws\00", align 8
@m_16490597097528467626 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa disposal that throws is not repeated by the cleanup\00", align 8
@m_13066997112432497201 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe body ran, and the disposal's own throw was caught\00", align 8
@m_6935274325732293108 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe same in a hand-written try body\00", align 8
@m_10009227236195275891 = internal constant [60 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe body never ran, and the try's own catch took it\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@disposed = private global i32 0
@steps = private global i32 0
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@Boom..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_17851658315048716010, i64 8)
@Boom..size = global i64 8, comdat
@_ZTIi = external constant ptr
@Boom..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Boom..instanceOf, ptr @Boom..new, ptr @Boom.dispose, ptr @Boom..rtti, ptr @Boom..size }
@ThrowsOnDispose..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1872836310818908366, i64 8)
@ThrowsOnDispose..size = global i64 8, comdat
@ThrowsOnDispose..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @ThrowsOnDispose..instanceOf, ptr @ThrowsOnDispose..new, ptr @ThrowsOnDispose.dispose, ptr @ThrowsOnDispose..rtti, ptr @ThrowsOnDispose..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Boom..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Boom..rtti, align 8
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
define ptr @Boom..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Boom, ptr %3, i32 0, i32 0
  store ptr @Boom..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Boom.constructor(ptr %0, i1 %1) #2 {
  %3 = alloca i1, align 1
  store i1 %1, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  br i1 %4, label %5, label %7

5:                                                ; preds = %2
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  call void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
  br label %8

7:                                                ; preds = %2
  ret void

8:                                                ; preds = %5
  unreachable
}

; Function Attrs: mustprogress
define void @Boom.dispose(ptr %0) #2 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @ThrowsOnDispose..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @ThrowsOnDispose..rtti, align 8
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
define ptr @ThrowsOnDispose..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %ThrowsOnDispose, ptr %3, i32 0, i32 0
  store ptr @ThrowsOnDispose..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @ThrowsOnDispose.dispose(ptr %0) #2 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %4, align 4
  call void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
  br label %5

5:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define void @normalExit() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  store i1 false, ptr %1, align 1
  %3 = call ptr @GC_malloc(i64 16)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 0
  store ptr @Res..vtbl, ptr %5, align 8
  store ptr %4, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %6 = load i32, ptr @steps, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr @steps, align 4
  %8 = load i1, ptr %1, align 1
  br i1 %8, label %9, label %13

9:                                                ; preds = %0
  store i1 false, ptr %1, align 1
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr %Res, ptr %10, i32 0, i32 0
  invoke void @Res.dispose(ptr %10)
          to label %12 unwind label %14

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %12, %0
  ret void

14:                                               ; preds = %9
  %15 = landingpad { ptr, i32 }
          cleanup
  %16 = load i1, ptr %1, align 1
  br i1 %16, label %17, label %20

17:                                               ; preds = %14
  store i1 false, ptr %1, align 1
  %18 = load ptr, ptr %2, align 8
  %19 = getelementptr %Res, ptr %18, i32 0, i32 0
  call void @Res.dispose(ptr %18)
  br label %20

20:                                               ; preds = %17, %14
  resume { ptr, i32 } %15
}

; Function Attrs: mustprogress
define void @throwsAfterDeclaration() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, %Res }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Res zeroinitializer, ptr %4, align 8
  store i1 false, ptr %1, align 1
  store %Res zeroinitializer, ptr %4, align 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 0
  store ptr @Res..vtbl, ptr %5, align 8
  store ptr %4, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %6 = load i32, ptr @steps, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr @steps, align 4
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %16 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          cleanup
  %11 = load i1, ptr %1, align 1
  br i1 %11, label %12, label %15

12:                                               ; preds = %9
  store i1 false, ptr %1, align 1
  %13 = load ptr, ptr %2, align 8
  %14 = getelementptr %Res, ptr %13, i32 0, i32 0
  call void @Res.dispose(ptr %13)
  br label %15

15:                                               ; preds = %12, %9
  resume { ptr, i32 } %10

16:                                               ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @constructorThrows() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  store i1 false, ptr %1, align 1
  %3 = call ptr @GC_malloc(i64 16)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %Boom, ptr %4, i32 0, i32 0
  store ptr @Boom..vtbl, ptr %5, align 8
  invoke void @Boom.constructor(ptr %4, i1 true)
          to label %6 unwind label %15

6:                                                ; preds = %0
  store ptr %4, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %7 = load i32, ptr @steps, align 4
  %8 = add i32 %7, 1
  store i32 %8, ptr @steps, align 4
  %9 = load i1, ptr %1, align 1
  br i1 %9, label %10, label %14

10:                                               ; preds = %6
  store i1 false, ptr %1, align 1
  %11 = load ptr, ptr %2, align 8
  %12 = getelementptr %Boom, ptr %11, i32 0, i32 0
  invoke void @Boom.dispose(ptr %11)
          to label %13 unwind label %15

13:                                               ; preds = %10
  br label %14

14:                                               ; preds = %13, %6
  ret void

15:                                               ; preds = %10, %0
  %16 = landingpad { ptr, i32 }
          cleanup
  %17 = load i1, ptr %1, align 1
  br i1 %17, label %18, label %21

18:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %19 = load ptr, ptr %2, align 8
  %20 = getelementptr %Boom, ptr %19, i32 0, i32 0
  call void @Boom.dispose(ptr %19)
  br label %21

21:                                               ; preds = %18, %15
  resume { ptr, i32 } %16
}

; Function Attrs: mustprogress
define void @secondConstructorThrows() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  store ptr %6, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Boom, ptr %9, i32 0, i32 0
  store ptr @Boom..vtbl, ptr %10, align 8
  invoke void @Boom.constructor(ptr %9, i1 true)
          to label %11 unwind label %26

11:                                               ; preds = %0
  store ptr %9, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %12 = load i32, ptr @steps, align 4
  %13 = add i32 %12, 1
  store i32 %13, ptr @steps, align 4
  %14 = load i1, ptr %1, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Boom, ptr %16, i32 0, i32 0
  invoke void @Boom.dispose(ptr %16)
          to label %18 unwind label %26

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %11
  %20 = load i1, ptr %3, align 1
  br i1 %20, label %21, label %25

21:                                               ; preds = %19
  store i1 false, ptr %3, align 1
  %22 = load ptr, ptr %4, align 8
  %23 = getelementptr %Res, ptr %22, i32 0, i32 0
  invoke void @Res.dispose(ptr %22)
          to label %24 unwind label %26

24:                                               ; preds = %21
  br label %25

25:                                               ; preds = %24, %19
  ret void

26:                                               ; preds = %21, %15, %0
  %27 = landingpad { ptr, i32 }
          cleanup
  %28 = load i1, ptr %1, align 1
  br i1 %28, label %29, label %32

29:                                               ; preds = %26
  store i1 false, ptr %1, align 1
  %30 = load ptr, ptr %2, align 8
  %31 = getelementptr %Boom, ptr %30, i32 0, i32 0
  call void @Boom.dispose(ptr %30)
  br label %32

32:                                               ; preds = %29, %26
  %33 = load i1, ptr %3, align 1
  br i1 %33, label %34, label %37

34:                                               ; preds = %32
  store i1 false, ptr %3, align 1
  %35 = load ptr, ptr %4, align 8
  %36 = getelementptr %Res, ptr %35, i32 0, i32 0
  call void @Res.dispose(ptr %35)
  br label %37

37:                                               ; preds = %34, %32
  resume { ptr, i32 } %27
}

; Function Attrs: mustprogress
define void @disposalThrows() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  store i1 false, ptr %1, align 1
  %3 = call ptr @GC_malloc(i64 16)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %ThrowsOnDispose, ptr %4, i32 0, i32 0
  store ptr @ThrowsOnDispose..vtbl, ptr %5, align 8
  store ptr %4, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %6 = load i32, ptr @steps, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr @steps, align 4
  %8 = load i1, ptr %1, align 1
  br i1 %8, label %9, label %13

9:                                                ; preds = %0
  store i1 false, ptr %1, align 1
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr %ThrowsOnDispose, ptr %10, i32 0, i32 0
  invoke void @ThrowsOnDispose.dispose(ptr %10)
          to label %12 unwind label %14

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %12, %0
  ret void

14:                                               ; preds = %9
  %15 = landingpad { ptr, i32 }
          cleanup
  %16 = load i1, ptr %1, align 1
  br i1 %16, label %17, label %20

17:                                               ; preds = %14
  store i1 false, ptr %1, align 1
  %18 = load ptr, ptr %2, align 8
  %19 = getelementptr %ThrowsOnDispose, ptr %18, i32 0, i32 0
  call void @ThrowsOnDispose.dispose(ptr %18)
  br label %20

20:                                               ; preds = %17, %14
  resume { ptr, i32 } %15
}

; Function Attrs: mustprogress
define void @constructorThrowsInTryBody() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Boom, ptr %5, i32 0, i32 0
  store ptr @Boom..vtbl, ptr %6, align 8
  invoke void @Boom.constructor(ptr %5, i1 true)
          to label %7 unwind label %16

7:                                                ; preds = %0
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %8 = load i32, ptr @steps, align 4
  %9 = add i32 %8, 1
  store i32 %9, ptr @steps, align 4
  %10 = load i1, ptr %1, align 1
  br i1 %10, label %11, label %15

11:                                               ; preds = %7
  store i1 false, ptr %1, align 1
  %12 = load ptr, ptr %2, align 8
  %13 = getelementptr %Boom, ptr %12, i32 0, i32 0
  invoke void @Boom.dispose(ptr %12)
          to label %14 unwind label %16

14:                                               ; preds = %11
  br label %15

15:                                               ; preds = %14, %7
  br label %53

16:                                               ; preds = %11, %0
  %17 = landingpad { ptr, i32 }
          catch ptr null
  %18 = extractvalue { ptr, i32 } %17, 0
  %19 = call ptr @__cxa_begin_catch(ptr %18)
  %20 = load i1, ptr %1, align 1
  br i1 %20, label %21, label %25

21:                                               ; preds = %16
  store i1 false, ptr %1, align 1
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr %Boom, ptr %22, i32 0, i32 0
  invoke void @Boom.dispose(ptr %22)
          to label %24 unwind label %endcatch.pad2

24:                                               ; preds = %21
  br label %25

25:                                               ; preds = %24, %16
  invoke void @__cxa_rethrow()
          to label %54 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %21
  %26 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

27:                                               ; No predecessors!
  %28 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %27
  %29 = phi { ptr, i32 } [ %28, %27 ], [ %26, %endcatch.pad2 ]
  %30 = extractvalue { ptr, i32 } %29, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  %32 = load i1, ptr %1, align 1
  br i1 %32, label %33, label %36

33:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %34 = load ptr, ptr %2, align 8
  %35 = getelementptr %Boom, ptr %34, i32 0, i32 0
  invoke void @Boom.dispose(ptr %34)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %33
  br label %36

36:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %54 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %36, %25
  %37 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

38:                                               ; No predecessors!
  %39 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %38
  %40 = phi { ptr, i32 } [ %39, %38 ], [ %37, %endcatch.pad ]
  %41 = extractvalue { ptr, i32 } %40, 1
  %42 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %43 = icmp eq i32 %41, %42
  br i1 %43, label %47, label %44

44:                                               ; preds = %.caught
  %45 = extractvalue { ptr, i32 } %40, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  invoke void @__cxa_rethrow()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %44
  br label %54

47:                                               ; preds = %.caught
  %48 = extractvalue { ptr, i32 } %40, 0
  %49 = call ptr @__cxa_begin_catch(ptr %48)
  %50 = load i32, ptr %49, align 4
  store i32 %50, ptr %3, align 4
  %51 = load i32, ptr @steps, align 4
  %52 = add i32 %51, 10
  store i32 %52, ptr @steps, align 4
  call void @__cxa_end_catch()
  br label %53

53:                                               ; preds = %15, %47
  ret void

54:                                               ; preds = %.noexc3, %25, %36
  unreachable

endcatch.resume:                                  ; preds = %44, %33
  %55 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %55
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @steps, align 4
  call void @normalExit()
  %5 = load i32, ptr @disposed, align 4
  %6 = icmp eq i32 %5, 1
  br i1 %6, label %7, label %110

7:                                                ; preds = %0
  %8 = load i32, ptr @steps, align 4
  %9 = icmp eq i32 %8, 1
  br i1 %9, label %10, label %112

10:                                               ; preds = %7
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @steps, align 4
  invoke void @throwsAfterDeclaration()
          to label %11 unwind label %13

11:                                               ; preds = %10
  br i1 false, label %12, label %114

12:                                               ; preds = %11
  br label %27

13:                                               ; preds = %10
  %14 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %15 = extractvalue { ptr, i32 } %14, 1
  %16 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %17 = icmp eq i32 %15, %16
  br i1 %17, label %21, label %18

18:                                               ; preds = %13
  %19 = extractvalue { ptr, i32 } %14, 0
  %20 = call ptr @__cxa_begin_catch(ptr %19)
  invoke void @__cxa_rethrow()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %18
  br label %109

21:                                               ; preds = %13
  %22 = extractvalue { ptr, i32 } %14, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = load i32, ptr %23, align 4
  store i32 %24, ptr %4, align 4
  %25 = load i32, ptr @steps, align 4
  %26 = add i32 %25, 1
  store i32 %26, ptr @steps, align 4
  call void @__cxa_end_catch()
  br label %27

27:                                               ; preds = %12, %21
  %28 = load i32, ptr @disposed, align 4
  %29 = icmp eq i32 %28, 1
  br i1 %29, label %30, label %116

30:                                               ; preds = %27
  %31 = load i32, ptr @steps, align 4
  %32 = icmp eq i32 %31, 2
  br i1 %32, label %33, label %118

33:                                               ; preds = %30
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @steps, align 4
  invoke void @constructorThrows()
          to label %34 unwind label %36

34:                                               ; preds = %33
  br i1 false, label %35, label %120

35:                                               ; preds = %34
  br label %50

36:                                               ; preds = %33
  %37 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %38 = extractvalue { ptr, i32 } %37, 1
  %39 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %40 = icmp eq i32 %38, %39
  br i1 %40, label %44, label %41

41:                                               ; preds = %36
  %42 = extractvalue { ptr, i32 } %37, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %41
  br label %109

44:                                               ; preds = %36
  %45 = extractvalue { ptr, i32 } %37, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  %47 = load i32, ptr %46, align 4
  store i32 %47, ptr %3, align 4
  %48 = load i32, ptr @steps, align 4
  %49 = add i32 %48, 1
  store i32 %49, ptr @steps, align 4
  call void @__cxa_end_catch()
  br label %50

50:                                               ; preds = %35, %44
  %51 = load i32, ptr @disposed, align 4
  %52 = icmp eq i32 %51, 0
  br i1 %52, label %53, label %122

53:                                               ; preds = %50
  %54 = load i32, ptr @steps, align 4
  %55 = icmp eq i32 %54, 1
  br i1 %55, label %56, label %124

56:                                               ; preds = %53
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @steps, align 4
  invoke void @secondConstructorThrows()
          to label %57 unwind label %59

57:                                               ; preds = %56
  br i1 false, label %58, label %126

58:                                               ; preds = %57
  br label %73

59:                                               ; preds = %56
  %60 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %61 = extractvalue { ptr, i32 } %60, 1
  %62 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %63 = icmp eq i32 %61, %62
  br i1 %63, label %67, label %64

64:                                               ; preds = %59
  %65 = extractvalue { ptr, i32 } %60, 0
  %66 = call ptr @__cxa_begin_catch(ptr %65)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %64
  br label %109

67:                                               ; preds = %59
  %68 = extractvalue { ptr, i32 } %60, 0
  %69 = call ptr @__cxa_begin_catch(ptr %68)
  %70 = load i32, ptr %69, align 4
  store i32 %70, ptr %2, align 4
  %71 = load i32, ptr @steps, align 4
  %72 = add i32 %71, 1
  store i32 %72, ptr @steps, align 4
  call void @__cxa_end_catch()
  br label %73

73:                                               ; preds = %58, %67
  %74 = load i32, ptr @disposed, align 4
  %75 = icmp eq i32 %74, 1
  br i1 %75, label %76, label %128

76:                                               ; preds = %73
  %77 = load i32, ptr @steps, align 4
  %78 = icmp eq i32 %77, 1
  br i1 %78, label %79, label %130

79:                                               ; preds = %76
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @steps, align 4
  invoke void @disposalThrows()
          to label %80 unwind label %82

80:                                               ; preds = %79
  br i1 false, label %81, label %132

81:                                               ; preds = %80
  br label %96

82:                                               ; preds = %79
  %83 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %84 = extractvalue { ptr, i32 } %83, 1
  %85 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %86 = icmp eq i32 %84, %85
  br i1 %86, label %90, label %87

87:                                               ; preds = %82
  %88 = extractvalue { ptr, i32 } %83, 0
  %89 = call ptr @__cxa_begin_catch(ptr %88)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %87
  br label %109

90:                                               ; preds = %82
  %91 = extractvalue { ptr, i32 } %83, 0
  %92 = call ptr @__cxa_begin_catch(ptr %91)
  %93 = load i32, ptr %92, align 4
  store i32 %93, ptr %1, align 4
  %94 = load i32, ptr @steps, align 4
  %95 = add i32 %94, 1
  store i32 %95, ptr @steps, align 4
  call void @__cxa_end_catch()
  br label %96

96:                                               ; preds = %81, %90
  %97 = load i32, ptr @disposed, align 4
  %98 = icmp eq i32 %97, 1
  br i1 %98, label %99, label %134

99:                                               ; preds = %96
  %100 = load i32, ptr @steps, align 4
  %101 = icmp eq i32 %100, 2
  br i1 %101, label %102, label %136

102:                                              ; preds = %99
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @steps, align 4
  call void @constructorThrowsInTryBody()
  %103 = load i32, ptr @disposed, align 4
  %104 = icmp eq i32 %103, 0
  br i1 %104, label %105, label %138

105:                                              ; preds = %102
  %106 = load i32, ptr @steps, align 4
  %107 = icmp eq i32 %106, 10
  br i1 %107, label %108, label %140

108:                                              ; preds = %105
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

109:                                              ; preds = %140, %138, %136, %134, %132, %.noexc, %130, %128, %126, %.noexc1, %124, %122, %120, %.noexc2, %118, %116, %114, %.noexc3, %112, %110
  unreachable

110:                                              ; preds = %0
  %111 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11111286348956853952, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 102, ptr null)
  br label %109

112:                                              ; preds = %7
  %113 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12601458686632539999, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 103, ptr null)
  br label %109

114:                                              ; preds = %11
  %115 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16373880600585569923, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 109, ptr null)
  br label %109

116:                                              ; preds = %27
  %117 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2343790244925707336, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 114, ptr null)
  br label %109

118:                                              ; preds = %30
  %119 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15951051187737246770, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 115, ptr null)
  br label %109

120:                                              ; preds = %34
  %121 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13099800921096224375, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 121, ptr null)
  br label %109

122:                                              ; preds = %50
  %123 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6585847342210022013, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 126, ptr null)
  br label %109

124:                                              ; preds = %53
  %125 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7063307771149137198, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 127, ptr null)
  br label %109

126:                                              ; preds = %57
  %127 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18172175995809474191, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 133, ptr null)
  br label %109

128:                                              ; preds = %73
  %129 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8081476958950538590, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 138, ptr null)
  br label %109

130:                                              ; preds = %76
  %131 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7063307771149137198, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 139, ptr null)
  br label %109

132:                                              ; preds = %80
  %133 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14032912357038811970, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 145, ptr null)
  br label %109

134:                                              ; preds = %96
  %135 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16490597097528467626, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 150, ptr null)
  br label %109

136:                                              ; preds = %99
  %137 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13066997112432497201, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 151, ptr null)
  br label %109

138:                                              ; preds = %102
  %139 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6935274325732293108, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 156, ptr null)
  br label %109

140:                                              ; preds = %105
  %141 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10009227236195275891, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2662995175179872379, i64 8), i32 157, ptr null)
  br label %109

endcatch.resume:                                  ; preds = %18, %41, %64, %87
  %142 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %142
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

