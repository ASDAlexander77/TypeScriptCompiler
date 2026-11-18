; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_3815361247288411516 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnamespace exceptions {\0A@dllimport\0Afunction run();\0A\0A}\0A\00", align 8
@s_618458831055183935 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFgc\00", align 8
@s_15902901984413996407 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfoo\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_3154112327090001444 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtest exn\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_15810290828860591424 = internal constant [101 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\01try_catch.ts\00", align 8
@_ZTIPc = external constant ptr
@__decls_01try_catch_8663496863349540679 = global ptr getelementptr inbounds nuw (i8, ptr @s_3815361247288411516, i64 8)
@__tsmm_gc_01try_catch_8663496863349540679 = global ptr getelementptr inbounds nuw (i8, ptr @s_618458831055183935, i64 8)

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define i32 @exceptions.test4({ ptr, ptr, ptr } %0) #0 personality ptr @__gxx_personality_v0 {
  %2 = alloca { ptr, ptr, ptr }, align 8
  %3 = alloca i32, align 4
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  %4 = load { ptr, ptr, ptr }, ptr %2, align 8
  %5 = extractvalue { ptr, ptr, ptr } %4, 1
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %1
  %9 = extractvalue { ptr, ptr, ptr } %4, 0
  invoke void %9(ptr %5)
          to label %10 unwind label %17

10:                                               ; preds = %8
  br label %14

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %4, 0
  invoke void %12()
          to label %13 unwind label %17

13:                                               ; preds = %11
  br label %14

14:                                               ; preds = %10, %13
  br label %15

15:                                               ; preds = %14
  br label %16

16:                                               ; preds = %15
  store i32 10, ptr %3, align 4
  br label %21

17:                                               ; preds = %8, %11
  %18 = landingpad { ptr, i32 }
          catch ptr null
  %19 = extractvalue { ptr, i32 } %18, 0
  %20 = call ptr @__cxa_begin_catch(ptr %19)
  store i32 20, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %21

21:                                               ; preds = %16, %17
  %22 = load i32, ptr %3, align 4
  ret i32 %22
}

; Function Attrs: mustprogress
define void @exceptions..f_run..afL15C22FH4238935879() #1 {
  ret void
}

; Function Attrs: mustprogress
define void @exceptions..f_run..afL16C22FH4238935879() #1 {
  %1 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_15902901984413996407, i64 8), ptr %1, align 8
  call void @__cxa_throw(ptr %1, ptr @_ZTIPc, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @exceptions.run() #1 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3154112327090001444, i64 8))
  %1 = call i32 @exceptions.test4({ ptr, ptr, ptr } { ptr @exceptions..f_run..afL15C22FH4238935879, ptr null, ptr null })
  %2 = icmp eq i32 %1, 10
  br i1 %2, label %3, label %8

3:                                                ; preds = %0
  %4 = call i32 @exceptions.test4({ ptr, ptr, ptr } { ptr @exceptions..f_run..afL16C22FH4238935879, ptr null, ptr null })
  %5 = icmp eq i32 %4, 20
  br i1 %5, label %6, label %10

6:                                                ; preds = %3
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret void

7:                                                ; preds = %10, %8
  unreachable

8:                                                ; preds = %0
  %9 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15810290828860591424, i64 8), i32 15, ptr null)
  br label %7

10:                                               ; preds = %3
  %11 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15810290828860591424, i64 8), i32 16, ptr null)
  br label %7
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @exceptions.run()
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

attributes #0 = { mustprogress "personality" }
attributes #1 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

