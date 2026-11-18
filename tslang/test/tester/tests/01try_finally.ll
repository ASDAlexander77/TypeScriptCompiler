; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@s_6272037745333642222 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_16881732518263451103 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch outer\00", align 8
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: mustprogress
define void @mayThrow(double %0) #1 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = fcmp ogt double %3, 1.000000e+01
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call ptr @__cxa_allocate_exception(i64 8)
  store double 1.000000e+00, ptr %6, align 8
  call void @__cxa_throw(ptr %6, ptr @_ZTId, ptr null)
  br label %8

7:                                                ; preds = %1
  ret void

8:                                                ; preds = %5
  unreachable
}

; Function Attrs: mustprogress
define void @main4() #2 personality ptr @__gxx_personality_v0 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6272037745333642222, i64 8))
  invoke void @mayThrow(double 1.000000e+02)
          to label %1 unwind label %2

1:                                                ; preds = %0
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  ret void

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          cleanup
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  resume { ptr, i32 } %3
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca double, align 8
  invoke void @main4()
          to label %2 unwind label %3

2:                                                ; preds = %0
  br label %52

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %5 = extractvalue { ptr, i32 } %4, 1
  %6 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %7 = icmp eq i32 %5, %6
  %8 = extractvalue { ptr, i32 } %4, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  %11 = extractvalue { ptr, i32 } %4, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %13 = icmp eq i32 %11, %12
  %14 = extractvalue { ptr, i32 } %4, 1
  %15 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %16 = icmp eq i32 %14, %15
  %17 = or i1 %7, %10
  %18 = or i1 %17, %13
  %19 = or i1 %18, %16
  br i1 %19, label %23, label %20

20:                                               ; preds = %3
  %21 = extractvalue { ptr, i32 } %4, 0
  %22 = call ptr @__cxa_begin_catch(ptr %21)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %20
  br label %53

23:                                               ; preds = %3
  %24 = extractvalue { ptr, i32 } %4, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  %26 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %23
  %27 = icmp eq ptr %26, @_ZTIi
  br i1 %27, label %28, label %31

28:                                               ; preds = %.noexc1
  %29 = load i32, ptr %25, align 4
  %30 = sitofp i32 %29 to double
  br label %49

31:                                               ; preds = %.noexc1
  %32 = icmp eq ptr %26, @_ZTIj
  br i1 %32, label %33, label %36

33:                                               ; preds = %31
  %34 = load i32, ptr %25, align 4
  %35 = uitofp i32 %34 to double
  br label %46

36:                                               ; preds = %31
  %37 = icmp eq ptr %26, @_ZTIf
  br i1 %37, label %38, label %41

38:                                               ; preds = %36
  %39 = load float, ptr %25, align 4
  %40 = fpext float %39 to double
  br label %43

41:                                               ; preds = %36
  %42 = load double, ptr %25, align 8
  br label %43

43:                                               ; preds = %38, %41
  %44 = phi double [ %42, %41 ], [ %40, %38 ]
  br label %45

45:                                               ; preds = %43
  br label %46

46:                                               ; preds = %33, %45
  %47 = phi double [ %44, %45 ], [ %35, %33 ]
  br label %48

48:                                               ; preds = %46
  br label %49

49:                                               ; preds = %28, %48
  %50 = phi double [ %47, %48 ], [ %30, %28 ]
  br label %51

51:                                               ; preds = %49
  store double %50, ptr %1, align 8
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_16881732518263451103, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %51
  call void @__cxa_end_catch()
  br label %52

52:                                               ; preds = %2, %.noexc
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

53:                                               ; preds = %.noexc2
  unreachable

endcatch.resume:                                  ; preds = %20, %23, %51
  %54 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %54
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

attributes #0 = { nounwind memory(none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

