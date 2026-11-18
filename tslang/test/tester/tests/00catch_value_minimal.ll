; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_14130353033048551820 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa catch clause binds the value that was thrown\00", align 8
@f_10494543871068754886 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00catch_value_minimal.ts\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@t = private global i32 0
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__cxa_rethrow()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define i32 @main() #1 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %2, align 4
  invoke void @__cxa_throw(ptr %2, ptr @_ZTIi, ptr null)
          to label %19 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %5 = extractvalue { ptr, i32 } %4, 1
  %6 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %11, label %8

8:                                                ; preds = %3
  %9 = extractvalue { ptr, i32 } %4, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %8
  br label %19

11:                                               ; preds = %3
  %12 = extractvalue { ptr, i32 } %4, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i32, ptr %13, align 4
  store i32 %14, ptr %1, align 4
  %15 = load i32, ptr %1, align 4
  store i32 %15, ptr @t, align 4
  call void @__cxa_end_catch()
  %16 = load i32, ptr @t, align 4
  %17 = icmp eq i32 %16, 2
  br i1 %17, label %18, label %20

18:                                               ; preds = %11
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

19:                                               ; preds = %20, %.noexc, %0
  unreachable

20:                                               ; preds = %11
  %21 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14130353033048551820, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10494543871068754886, i64 8), i32 30, ptr null)
  br label %19

endcatch.resume:                                  ; preds = %8
  %22 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %22
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
attributes #1 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

