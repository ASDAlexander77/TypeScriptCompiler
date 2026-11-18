; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_14046730643414667274 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFend\00", align 8
@s_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@s_13957468038051399532 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch\00", align 8
@s_10816067483634310915 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFIn try\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_14788516130519408866 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally is not called\00", align 8
@f_5126867784565821602 = internal constant [108 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch_return.ts\00", align 8
@called = private global i1 false
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

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
define void @may_throw({ i32, i1 } %0) #1 {
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
define void @func1() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10816067483634310915, i64 8))
  invoke void @may_throw({ i32, i1 } { i32 1000, i1 true })
          to label %2 unwind label %3

2:                                                ; preds = %0
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  store i1 false, ptr @called, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14046730643414667274, i64 8))
  br label %19

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %5 = extractvalue { ptr, i32 } %4, 1
  %6 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %11, label %8

8:                                                ; preds = %3
  %9 = extractvalue { ptr, i32 } %4, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  invoke void @__cxa_rethrow()
          to label %20 unwind label %endcatch.pad

11:                                               ; preds = %3
  %12 = extractvalue { ptr, i32 } %4, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i32, ptr %13, align 4
  store i32 %14, ptr %1, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_13957468038051399532, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %11
  call void @__cxa_end_catch()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  br label %19

endcatch.pad:                                     ; preds = %8
  %15 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

16:                                               ; No predecessors!
  %17 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %16
  %18 = phi { ptr, i32 } [ %17, %16 ], [ %15, %endcatch.pad ]
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  store i1 true, ptr @called, align 1
  resume { ptr, i32 } %18

19:                                               ; preds = %2, %.noexc
  ret void

20:                                               ; preds = %8
  unreachable

endcatch.resume:                                  ; preds = %11
  %21 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %21
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @func1()
  %1 = load i1, ptr @called, align 1
  br i1 %1, label %2, label %4

2:                                                ; preds = %0
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

3:                                                ; preds = %4
  unreachable

4:                                                ; preds = %0
  %5 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14788516130519408866, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5126867784565821602, i64 8), i32 28, ptr null)
  br label %3
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

