; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%TempFile = type { ptr, ptr, double }

$TempFile..size = comdat any

@s_1480153656757206009 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFTempFile\00", align 8
@s_17112020470516500550 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispose\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_14046730643414667274 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFend\00", align 8
@s_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@s_13957468038051399532 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch\00", align 8
@s_10816067483634310915 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFIn try\00", align 8
@s_2461554088447426271 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.some_temp_file\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_14788516130519408866 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally is not called\00", align 8
@f_14050134112324815179 = internal constant [116 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch_return_dispose.ts\00", align 8
@m_13476757536784520328 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispose is not called\00", align 8
@called = private global i1 false
@dispose_called = private global i1 false
@_ZTIi = external constant ptr
@TempFile..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1480153656757206009, i64 8)
@TempFile..size = global i64 24, comdat
@TempFile..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @TempFile..instanceOf, ptr @TempFile..new, ptr @TempFile.dispose, ptr @TempFile..rtti, ptr @TempFile..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

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

declare void @puts(ptr)

declare i32 @strcmp(ptr, ptr)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: mustprogress
define void @may_throw({ i32, i1 } %0) #2 {
  %2 = alloca i32, align 4
  %3 = extractvalue { i32, i1 } %0, 1
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  %5 = extractvalue { i32, i1 } %0, 0
  br label %7

6:                                                ; preds = %1
  br label %7

7:                                                ; preds = %4, %6
  %8 = phi i32 [ 0, %6 ], [ %5, %4 ]
  store i32 %8, ptr %2, align 4
  %9 = load i32, ptr %2, align 4
  %10 = icmp sgt i32 %9, 100
  br i1 %10, label %11, label %13

11:                                               ; preds = %7
  %12 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %12, align 4
  call void @__cxa_throw(ptr %12, ptr @_ZTIi, ptr null)
  br label %14

13:                                               ; preds = %7
  ret void

14:                                               ; preds = %11
  unreachable
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
  store i1 true, ptr @dispose_called, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_17112020470516500550, i64 8))
  ret void
}

; Function Attrs: mustprogress
define void @func1() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %TempFile, ptr %5, i32 0, i32 0
  store ptr @TempFile..vtbl, ptr %6, align 8
  invoke void @TempFile.constructor(ptr %5, ptr getelementptr inbounds nuw (i8, ptr @s_2461554088447426271, i64 8))
          to label %7 unwind label %15

7:                                                ; preds = %0
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10816067483634310915, i64 8))
  invoke void @may_throw({ i32, i1 } { i32 1000, i1 true })
          to label %8 unwind label %15

8:                                                ; preds = %7
  %9 = load i1, ptr %1, align 1
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  store i1 false, ptr %1, align 1
  %11 = load ptr, ptr %2, align 8
  %12 = getelementptr %TempFile, ptr %11, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %11)
          to label %13 unwind label %15

13:                                               ; preds = %10
  br label %14

14:                                               ; preds = %13, %8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  store i1 false, ptr @called, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14046730643414667274, i64 8))
  br label %54

15:                                               ; preds = %10, %7, %0
  %16 = landingpad { ptr, i32 }
          catch ptr null
  %17 = extractvalue { ptr, i32 } %16, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %24

20:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %TempFile, ptr %21, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %21)
          to label %23 unwind label %endcatch.pad2

23:                                               ; preds = %20
  br label %24

24:                                               ; preds = %23, %15
  invoke void @__cxa_rethrow()
          to label %55 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %20
  %25 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

26:                                               ; No predecessors!
  %27 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %26
  %28 = phi { ptr, i32 } [ %27, %26 ], [ %25, %endcatch.pad2 ]
  %29 = extractvalue { ptr, i32 } %28, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  %31 = load i1, ptr %1, align 1
  br i1 %31, label %32, label %35

32:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %33 = load ptr, ptr %2, align 8
  %34 = getelementptr %TempFile, ptr %33, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %33)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %32
  br label %35

35:                                               ; preds = %.noexc5, %.caught1
  invoke void @__cxa_rethrow()
          to label %55 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %35, %24
  %36 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

37:                                               ; No predecessors!
  %38 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %37
  %39 = phi { ptr, i32 } [ %38, %37 ], [ %36, %endcatch.pad ]
  %40 = extractvalue { ptr, i32 } %39, 1
  %41 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %42 = icmp eq i32 %40, %41
  br i1 %42, label %46, label %43

43:                                               ; preds = %.caught
  %44 = extractvalue { ptr, i32 } %39, 0
  %45 = call ptr @__cxa_begin_catch(ptr %44)
  invoke void @__cxa_rethrow()
          to label %55 unwind label %endcatch.pad4

46:                                               ; preds = %.caught
  %47 = extractvalue { ptr, i32 } %39, 0
  %48 = call ptr @__cxa_begin_catch(ptr %47)
  %49 = load i32, ptr %48, align 4
  store i32 %49, ptr %3, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_13957468038051399532, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %46
  call void @__cxa_end_catch()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  br label %54

endcatch.pad4:                                    ; preds = %43
  %50 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

51:                                               ; No predecessors!
  %52 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %51
  %53 = phi { ptr, i32 } [ %52, %51 ], [ %50, %endcatch.pad4 ]
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  resume { ptr, i32 } %53

54:                                               ; preds = %14, %.noexc
  ret void

55:                                               ; preds = %43, %24, %35
  unreachable

endcatch.resume:                                  ; preds = %32, %46
  %56 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %56
}

; Function Attrs: mustprogress
define void @func2() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %TempFile, ptr %5, i32 0, i32 0
  store ptr @TempFile..vtbl, ptr %6, align 8
  invoke void @TempFile.constructor(ptr %5, ptr getelementptr inbounds nuw (i8, ptr @s_2461554088447426271, i64 8))
          to label %7 unwind label %15

7:                                                ; preds = %0
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10816067483634310915, i64 8))
  invoke void @may_throw({ i32, i1 } { i32 1, i1 true })
          to label %8 unwind label %15

8:                                                ; preds = %7
  %9 = load i1, ptr %1, align 1
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  store i1 false, ptr %1, align 1
  %11 = load ptr, ptr %2, align 8
  %12 = getelementptr %TempFile, ptr %11, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %11)
          to label %13 unwind label %15

13:                                               ; preds = %10
  br label %14

14:                                               ; preds = %13, %8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14046730643414667274, i64 8))
  br label %54

15:                                               ; preds = %10, %7, %0
  %16 = landingpad { ptr, i32 }
          catch ptr null
  %17 = extractvalue { ptr, i32 } %16, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %24

20:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %TempFile, ptr %21, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %21)
          to label %23 unwind label %endcatch.pad2

23:                                               ; preds = %20
  br label %24

24:                                               ; preds = %23, %15
  invoke void @__cxa_rethrow()
          to label %55 unwind label %endcatch.pad4

endcatch.pad2:                                    ; preds = %20
  %25 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

26:                                               ; No predecessors!
  %27 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %26
  %28 = phi { ptr, i32 } [ %27, %26 ], [ %25, %endcatch.pad2 ]
  %29 = extractvalue { ptr, i32 } %28, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  %31 = load i1, ptr %1, align 1
  br i1 %31, label %32, label %35

32:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %33 = load ptr, ptr %2, align 8
  %34 = getelementptr %TempFile, ptr %33, i32 0, i32 0
  invoke void @TempFile.dispose(ptr %33)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %32
  br label %35

35:                                               ; preds = %.noexc5, %.caught1
  invoke void @__cxa_rethrow()
          to label %55 unwind label %endcatch.pad4

endcatch.pad4:                                    ; preds = %35, %24
  %36 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

37:                                               ; No predecessors!
  %38 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %37
  %39 = phi { ptr, i32 } [ %38, %37 ], [ %36, %endcatch.pad4 ]
  %40 = extractvalue { ptr, i32 } %39, 1
  %41 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %42 = icmp eq i32 %40, %41
  br i1 %42, label %46, label %43

43:                                               ; preds = %.caught3
  %44 = extractvalue { ptr, i32 } %39, 0
  %45 = call ptr @__cxa_begin_catch(ptr %44)
  invoke void @__cxa_rethrow()
          to label %55 unwind label %endcatch.pad

46:                                               ; preds = %.caught3
  %47 = extractvalue { ptr, i32 } %39, 0
  %48 = call ptr @__cxa_begin_catch(ptr %47)
  %49 = load i32, ptr %48, align 4
  store i32 %49, ptr %3, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_13957468038051399532, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %46
  call void @__cxa_end_catch()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  br label %54

endcatch.pad:                                     ; preds = %43
  %50 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

51:                                               ; No predecessors!
  %52 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %51
  %53 = phi { ptr, i32 } [ %52, %51 ], [ %50, %endcatch.pad ]
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  resume { ptr, i32 } %53

54:                                               ; preds = %14, %.noexc
  ret void

55:                                               ; preds = %43, %24, %35
  unreachable

endcatch.resume:                                  ; preds = %32, %46
  %56 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %56
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @func1()
  %1 = load i1, ptr @called, align 1
  br i1 %1, label %2, label %10

2:                                                ; preds = %0
  %3 = load i1, ptr @dispose_called, align 1
  br i1 %3, label %4, label %12

4:                                                ; preds = %2
  store i1 false, ptr @called, align 1
  store i1 false, ptr @dispose_called, align 1
  call void @func2()
  %5 = load i1, ptr @called, align 1
  br i1 %5, label %6, label %14

6:                                                ; preds = %4
  %7 = load i1, ptr @dispose_called, align 1
  br i1 %7, label %8, label %16

8:                                                ; preds = %6
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

9:                                                ; preds = %16, %14, %12, %10
  unreachable

10:                                               ; preds = %0
  %11 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14788516130519408866, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14050134112324815179, i64 8), i32 65, ptr null)
  br label %9

12:                                               ; preds = %2
  %13 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13476757536784520328, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14050134112324815179, i64 8), i32 66, ptr null)
  br label %9

14:                                               ; preds = %4
  %15 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14788516130519408866, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14050134112324815179, i64 8), i32 70, ptr null)
  br label %9

16:                                               ; preds = %6
  %17 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13476757536784520328, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14050134112324815179, i64 8), i32 71, ptr null)
  br label %9
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

