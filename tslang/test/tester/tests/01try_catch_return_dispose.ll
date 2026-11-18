; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%TempFile = type { ptr, ptr, double }

$TempFile..size = comdat any

@s_1480153656757206009 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFTempFile\00", align 8
@s_17112020470516500550 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispose\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_14894915187058291560 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFinally\00", align 8
@s_849505297048112493 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcont\00", align 8
@s_7859873275976691790 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFTry\00", align 8
@s_2461554088447426271 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.some_temp_file\00", align 8
@s_3019566071980655109 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_8540387491017555995 = internal constant [116 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\01try_catch_return_dispose.ts\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@glb1 = private global i32 0
@dispose_called = private global i1 false
@TempFile..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1480153656757206009, i64 8)
@TempFile..size = global i64 24, comdat
@TempFile..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @TempFile..instanceOf, ptr @TempFile..new, ptr @TempFile.dispose, ptr @TempFile..rtti, ptr @TempFile..size }
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @__cxa_end_catch()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @puts(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @TempFile..instanceOf(ptr %0, ptr %1) #1 {
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
define ptr @TempFile..new() #1 {
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
define void @TempFile.constructor(ptr %0, ptr %1) #1 {
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
define void @TempFile.dispose(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %TempFile, ptr %3, i32 0, i32 2
  store double 0.000000e+00, ptr %4, align 8
  store i1 true, ptr @dispose_called, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_17112020470516500550, i64 8))
  ret void
}

; Function Attrs: mustprogress
define void @try2() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 10, ptr %3, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %TempFile, ptr %5, i32 0, i32 0
  store ptr @TempFile..vtbl, ptr %6, align 8
  invoke void @TempFile.constructor(ptr %5, ptr getelementptr inbounds nuw (i8, ptr @s_2461554088447426271, i64 8))
          to label %7 unwind label %23

7:                                                ; preds = %0
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %8 = load i32, ptr @glb1, align 4
  %9 = add i32 %8, 1
  store i32 %9, ptr @glb1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_7859873275976691790, i64 8))
  %10 = load i32, ptr %3, align 4
  %11 = icmp sgt i32 %10, 0
  br i1 %11, label %12, label %14

12:                                               ; preds = %7
  %13 = call ptr @__cxa_allocate_exception(i64 8)
  store double 1.000000e+00, ptr %13, align 8
  invoke void @__cxa_throw(ptr %13, ptr @_ZTId, ptr null)
          to label %50 unwind label %23

14:                                               ; preds = %7
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_849505297048112493, i64 8))
  %15 = load i1, ptr %1, align 1
  br i1 %15, label %16, label %20

16:                                               ; preds = %14
  store i1 false, ptr %1, align 1
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr %TempFile, ptr %17, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %17)
          to label %19 unwind label %23

19:                                               ; preds = %16
  br label %20

20:                                               ; preds = %19, %14
  %21 = load i32, ptr @glb1, align 4
  %22 = add i32 %21, 1
  store i32 %22, ptr @glb1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14894915187058291560, i64 8))
  ret void

23:                                               ; preds = %12, %16, %0
  %24 = landingpad { ptr, i32 }
          catch ptr null
  %25 = extractvalue { ptr, i32 } %24, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = load i1, ptr %1, align 1
  br i1 %27, label %28, label %32

28:                                               ; preds = %23
  store i1 false, ptr %1, align 1
  %29 = load ptr, ptr %2, align 8
  %30 = getelementptr %TempFile, ptr %29, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %29)
          to label %31 unwind label %endcatch.pad2

31:                                               ; preds = %28
  br label %32

32:                                               ; preds = %31, %23
  invoke void @__cxa_rethrow()
          to label %50 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %28
  %33 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %33, %endcatch.pad2 ]
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  %39 = load i1, ptr %1, align 1
  br i1 %39, label %40, label %43

40:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %41 = load ptr, ptr %2, align 8
  %42 = getelementptr %TempFile, ptr %41, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %41)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %40
  br label %43

43:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %50 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %32, %43
  %44 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

45:                                               ; No predecessors!
  %46 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %45
  %47 = phi { ptr, i32 } [ %46, %45 ], [ %44, %endcatch.pad ]
  %48 = load i32, ptr @glb1, align 4
  %49 = add i32 %48, 1
  store i32 %49, ptr @glb1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14894915187058291560, i64 8))
  resume { ptr, i32 } %47

50:                                               ; preds = %32, %43, %12
  unreachable

endcatch.resume:                                  ; preds = %40
  %51 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %51
}

; Function Attrs: mustprogress
define void @main2() #2 personality ptr @__gxx_personality_v0 {
  store i32 0, ptr @glb1, align 4
  invoke void @try2()
          to label %1 unwind label %2

1:                                                ; preds = %0
  br label %8

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          catch ptr null
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = call ptr @__cxa_begin_catch(ptr %4)
  %6 = load i32, ptr @glb1, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr @glb1, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %2
  call void @__cxa_end_catch()
  br label %8

8:                                                ; preds = %1, %.noexc
  %9 = load i32, ptr @glb1, align 4
  %10 = icmp eq i32 %9, 3
  br i1 %10, label %11, label %15

11:                                               ; preds = %8
  %12 = load i1, ptr @dispose_called, align 1
  br i1 %12, label %13, label %17

13:                                               ; preds = %11
  ret void

14:                                               ; preds = %17, %15
  unreachable

15:                                               ; preds = %8
  %16 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8540387491017555995, i64 8), i32 49, ptr null)
  br label %14

17:                                               ; preds = %11
  %18 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8540387491017555995, i64 8), i32 50, ptr null)
  br label %14

endcatch.resume:                                  ; preds = %2
  %19 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %19
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @main2()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0
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

