; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_15311389968113750460 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFWRONGLY CAUGHT BY INNER\00", align 8
@s_14785268767687447306 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboom\00", align 8
@s_46537556930606345 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcorrectly rethrown and caught outer: \00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_17450617205034952454 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcorrectly rethrown and caught outer: boom\00", align 8
@m_8270522013899041519 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmismatched catch must rethrow, not swallow\00", align 8
@f_15064315143554972420 = internal constant [118 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch_mismatch_rethrow.ts\00", align 8
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @strcmp(ptr, ptr)

declare void @puts(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define ptr @inner() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8), ptr %3, align 8
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIPc, ptr null)
          to label %17 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %9
  br label %17

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %1, align 4
  store ptr getelementptr inbounds nuw (i8, ptr @s_15311389968113750460, i64 8), ptr %2, align 8
  call void @__cxa_end_catch()
  %16 = load ptr, ptr %2, align 8
  ret ptr %16

17:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %9
  %18 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %18
}

; Function Attrs: mustprogress
define ptr @testMismatchRethrow() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = invoke ptr @inner()
          to label %4 unwind label %5

4:                                                ; preds = %0
  store ptr %3, ptr %2, align 8
  br label %35

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %10
  br label %37

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  store ptr %15, ptr %1, align 8
  %16 = load ptr, ptr %1, align 8
  %17 = icmp eq ptr %16, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %16
  %19 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_46537556930606345, i64 8))
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %13
  %20 = add i64 1, %19
  %21 = invoke i64 @strlen(ptr %18)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %22 = add i64 %20, %21
  %23 = add i64 %22, 8
  %24 = invoke ptr @GC_malloc(i64 %23)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %22, 0
  %28 = invoke ptr @__memcpy_chk(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_46537556930606345, i64 8), i64 %19, i64 %27)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %22, %29
  %32 = invoke ptr @__memcpy_chk(ptr %30, ptr %18, i64 %21, i64 %31)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  store i8 0, ptr %34, align 1
  store ptr %25, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %35

35:                                               ; preds = %4, %.noexc5
  %36 = load ptr, ptr %2, align 8
  ret ptr %36

37:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %.noexc4, %.noexc3, %.noexc2, %.noexc1, %13, %10
  %38 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %38
}

; Function Attrs: mustprogress
define i32 @main() #3 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = call ptr @testMismatchRethrow()
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = icmp eq ptr %3, null
  %5 = select i1 %4, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  call void @puts(ptr %5)
  %6 = load ptr, ptr %1, align 8
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  %9 = zext i1 %8 to i32
  %10 = and i32 %9, 1
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %0
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_17450617205034952454, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %0
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17450617205034952454, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %22

20:                                               ; preds = %19
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

21:                                               ; preds = %22
  unreachable

22:                                               ; preds = %19
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8270522013899041519, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15064315143554972420, i64 8), i32 30, ptr null)
  br label %21
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
attributes #2 = { mustprogress "personality" }
attributes #3 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

