; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%TempFile = type { ptr, ptr, double }

$TempFile..size = comdat any

@s_1480153656757206009 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFTempFile\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_2461554088447426271 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.some_temp_file\00", align 8
@m_695755664849308805 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFDisposable is not called\00", align 8
@f_2200968981965328159 = internal constant [102 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00disposable.ts\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_17112020470516500550 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispose\00", align 8
@called = private global i1 false
@TempFile..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1480153656757206009, i64 8)
@TempFile..size = global i64 24, comdat
@TempFile.Disposable..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_7869123793284976990_class, i64 40), ptr @TempFile..instanceOf, ptr @TempFile.dispose }
@td_7869123793284976990_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_7869123793284976990, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@TempFile..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @TempFile..instanceOf, ptr @TempFile.Disposable..vtbl, ptr @TempFile..new, ptr @TempFile.dispose, ptr @TempFile..rtti, ptr @TempFile..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @__gxx_personality_v0(...)

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

define internal void @tsrel_7869123793284976990(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %TempFile, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #1 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  call void @doSomeWork1()
  %3 = load i1, ptr @called, align 1
  br i1 %3, label %4, label %25

4:                                                ; preds = %0
  store i1 false, ptr @called, align 1
  store i1 false, ptr %1, align 1
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %TempFile, ptr %6, i32 0, i32 0
  store ptr @TempFile..vtbl, ptr %7, align 8
  invoke void @TempFile.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_2461554088447426271, i64 8))
          to label %8 unwind label %17

8:                                                ; preds = %4
  store ptr %6, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %9 = load i1, ptr %1, align 1
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  store i1 false, ptr %1, align 1
  %11 = load ptr, ptr %2, align 8
  %12 = getelementptr %TempFile, ptr %11, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %11)
          to label %13 unwind label %17

13:                                               ; preds = %10
  br label %14

14:                                               ; preds = %13, %8
  %15 = load i1, ptr @called, align 1
  br i1 %15, label %16, label %27

16:                                               ; preds = %14
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

17:                                               ; preds = %10, %4
  %18 = landingpad { ptr, i32 }
          cleanup
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %23

20:                                               ; preds = %17
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %TempFile, ptr %21, i32 0, i32 0
  call void @TempFile.dispose(ptr %21)
  br label %23

23:                                               ; preds = %20, %17
  resume { ptr, i32 } %18

24:                                               ; preds = %27, %25
  unreachable

25:                                               ; preds = %0
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_695755664849308805, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2200968981965328159, i64 8), i32 6, ptr null)
  br label %24

27:                                               ; preds = %14
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_695755664849308805, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2200968981965328159, i64 8), i32 14, ptr null)
  br label %24
}

; Function Attrs: mustprogress
define void @doSomeWork1() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca <{ i64, %TempFile }>, align 16
  store i64 -1, ptr %1, align 8
  %2 = getelementptr i8, ptr %1, i64 8
  store %TempFile zeroinitializer, ptr %2, align 8
  store %TempFile zeroinitializer, ptr %2, align 8
  %3 = getelementptr %TempFile, ptr %2, i32 0, i32 0
  store ptr @TempFile..vtbl, ptr %3, align 8
  call void @TempFile.constructor(ptr %2, ptr getelementptr inbounds nuw (i8, ptr @s_2461554088447426271, i64 8))
  %4 = getelementptr %TempFile, ptr %2, i32 0, i32 0
  call void @TempFile.dispose(ptr %2)
  ret void
}

; Function Attrs: mustprogress
define i1 @TempFile..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @TempFile..rtti, align 8
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
define ptr @TempFile..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %TempFile, ptr %3, i32 0, i32 0
  store ptr @TempFile..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @TempFile.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %TempFile, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = getelementptr %TempFile, ptr %8, i32 0, i32 2
  store double 1.000000e+00, ptr %9, align 8
  ret void
}

; Function Attrs: mustprogress
define void @TempFile.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %TempFile, ptr %3, i32 0, i32 2
  store double 0.000000e+00, ptr %4, align 8
  store i1 true, ptr @called, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_17112020470516500550, i64 8))
  ret void
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
attributes #1 = { mustprogress "personality" }
attributes #2 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

