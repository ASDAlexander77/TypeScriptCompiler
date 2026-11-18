; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_3019566071980655109 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught\00", align 8
@s_2378669269878832019 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnum\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_10466072324029831577 = internal constant [64 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using in a try body disposes once when its catch runs\00", align 8
@f_16817579972225123052 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_using_catch.ts\00", align 8
@m_4290199297188172532 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe same with a number-typed catch\00", align 8
@disposed = private global i32 0
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIi = external constant ptr
@_ZTId = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__cxa_current_exception_type()

declare void @__cxa_end_catch()

declare void @puts(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

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
define void @twoCallsInCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %6, align 8
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIi, ptr null)
          to label %43 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = load i1, ptr %1, align 1
  br i1 %12, label %13, label %17

13:                                               ; preds = %8
  store i1 false, ptr %1, align 1
  %14 = load ptr, ptr %2, align 8
  %15 = getelementptr %Res, ptr %14, i32 0, i32 0
  invoke void @Res.dispose(ptr %14)
          to label %16 unwind label %endcatch.pad

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %16, %8
  invoke void @__cxa_rethrow()
          to label %43 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %13
  %18 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

19:                                               ; No predecessors!
  %20 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %19
  %21 = phi { ptr, i32 } [ %20, %19 ], [ %18, %endcatch.pad ]
  %22 = extractvalue { ptr, i32 } %21, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = load i1, ptr %1, align 1
  br i1 %24, label %25, label %28

25:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %26 = load ptr, ptr %2, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %25
  br label %28

28:                                               ; preds = %.noexc5, %.caught
  invoke void @__cxa_rethrow()
          to label %43 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %17, %28
  %29 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught1

30:                                               ; No predecessors!
  %31 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %30
  %32 = phi { ptr, i32 } [ %31, %30 ], [ %29, %endcatch.pad2 ]
  %33 = extractvalue { ptr, i32 } %32, 1
  %34 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %35 = icmp eq i32 %33, %34
  br i1 %35, label %39, label %36

36:                                               ; preds = %.caught1
  %37 = extractvalue { ptr, i32 } %32, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  invoke void @__cxa_rethrow()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %36
  br label %43

39:                                               ; preds = %.caught1
  %40 = extractvalue { ptr, i32 } %32, 0
  %41 = call ptr @__cxa_begin_catch(ptr %40)
  %42 = load i32, ptr %41, align 4
  store i32 %42, ptr %3, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %39
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc
  call void @__cxa_end_catch()
  ret void

43:                                               ; preds = %.noexc4, %17, %28, %0
  unreachable

endcatch.resume:                                  ; preds = %25, %36, %.noexc, %39
  %44 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %44
}

; Function Attrs: mustprogress
define i32 @callerHoldsValueAcrossCall() #2 {
  %1 = alloca i32, align 4
  store i32 0, ptr @disposed, align 4
  call void @twoCallsInCatch()
  %2 = load i32, ptr @disposed, align 4
  store i32 %2, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  ret i32 %3
}

; Function Attrs: mustprogress
define void @catchesANumber() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %6, align 8
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %7 = call ptr @__cxa_allocate_exception(i64 8)
  store double 1.500000e+00, ptr %7, align 8
  invoke void @__cxa_throw(ptr %7, ptr @_ZTId, ptr null)
          to label %80 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = load i1, ptr %1, align 1
  br i1 %12, label %13, label %17

13:                                               ; preds = %8
  store i1 false, ptr %1, align 1
  %14 = load ptr, ptr %2, align 8
  %15 = getelementptr %Res, ptr %14, i32 0, i32 0
  invoke void @Res.dispose(ptr %14)
          to label %16 unwind label %endcatch.pad2

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %16, %8
  invoke void @__cxa_rethrow()
          to label %80 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %13
  %18 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

19:                                               ; No predecessors!
  %20 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %19
  %21 = phi { ptr, i32 } [ %20, %19 ], [ %18, %endcatch.pad2 ]
  %22 = extractvalue { ptr, i32 } %21, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = load i1, ptr %1, align 1
  br i1 %24, label %25, label %28

25:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %26 = load ptr, ptr %2, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %25
  br label %28

28:                                               ; preds = %.noexc6, %.caught1
  invoke void @__cxa_rethrow()
          to label %80 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %17, %28
  %29 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  call void @__cxa_end_catch()
  br label %.caught

30:                                               ; No predecessors!
  %31 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %30
  %32 = phi { ptr, i32 } [ %31, %30 ], [ %29, %endcatch.pad ]
  %33 = extractvalue { ptr, i32 } %32, 1
  %34 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %35 = icmp eq i32 %33, %34
  %36 = extractvalue { ptr, i32 } %32, 1
  %37 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %38 = icmp eq i32 %36, %37
  %39 = extractvalue { ptr, i32 } %32, 1
  %40 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %41 = icmp eq i32 %39, %40
  %42 = extractvalue { ptr, i32 } %32, 1
  %43 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %44 = icmp eq i32 %42, %43
  %45 = or i1 %35, %38
  %46 = or i1 %45, %41
  %47 = or i1 %46, %44
  br i1 %47, label %51, label %48

48:                                               ; preds = %.caught
  %49 = extractvalue { ptr, i32 } %32, 0
  %50 = call ptr @__cxa_begin_catch(ptr %49)
  invoke void @__cxa_rethrow()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %48
  br label %80

51:                                               ; preds = %.caught
  %52 = extractvalue { ptr, i32 } %32, 0
  %53 = call ptr @__cxa_begin_catch(ptr %52)
  %54 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %51
  %55 = icmp eq ptr %54, @_ZTIi
  br i1 %55, label %56, label %59

56:                                               ; preds = %.noexc4
  %57 = load i32, ptr %53, align 4
  %58 = sitofp i32 %57 to double
  br label %77

59:                                               ; preds = %.noexc4
  %60 = icmp eq ptr %54, @_ZTIj
  br i1 %60, label %61, label %64

61:                                               ; preds = %59
  %62 = load i32, ptr %53, align 4
  %63 = uitofp i32 %62 to double
  br label %74

64:                                               ; preds = %59
  %65 = icmp eq ptr %54, @_ZTIf
  br i1 %65, label %66, label %69

66:                                               ; preds = %64
  %67 = load float, ptr %53, align 4
  %68 = fpext float %67 to double
  br label %71

69:                                               ; preds = %64
  %70 = load double, ptr %53, align 8
  br label %71

71:                                               ; preds = %66, %69
  %72 = phi double [ %70, %69 ], [ %68, %66 ]
  br label %73

73:                                               ; preds = %71
  br label %74

74:                                               ; preds = %61, %73
  %75 = phi double [ %72, %73 ], [ %63, %61 ]
  br label %76

76:                                               ; preds = %74
  br label %77

77:                                               ; preds = %56, %76
  %78 = phi double [ %75, %76 ], [ %58, %56 ]
  br label %79

79:                                               ; preds = %77
  store double %78, ptr %3, align 8
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_2378669269878832019, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %79
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc
  call void @__cxa_end_catch()
  ret void

80:                                               ; preds = %.noexc5, %17, %28, %0
  unreachable

endcatch.resume:                                  ; preds = %25, %48, %51, %.noexc, %79
  %81 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %81
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i32 @callerHoldsValueAcrossCall()
  %2 = icmp eq i32 %1, 1
  br i1 %2, label %3, label %8

3:                                                ; preds = %0
  store i32 0, ptr @disposed, align 4
  call void @catchesANumber()
  %4 = load i32, ptr @disposed, align 4
  %5 = icmp eq i32 %4, 1
  br i1 %5, label %6, label %10

6:                                                ; preds = %3
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

7:                                                ; preds = %10, %8
  unreachable

8:                                                ; preds = %0
  %9 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10466072324029831577, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16817579972225123052, i64 8), i32 57, ptr null)
  br label %7

10:                                               ; preds = %3
  %11 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4290199297188172532, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16817579972225123052, i64 8), i32 61, ptr null)
  br label %7
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

