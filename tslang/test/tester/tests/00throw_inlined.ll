; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_14113715647558732790 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan inlined helper's throw must survive the inliner\00", align 8
@f_2451893981936320601 = internal constant [105 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_inlined.ts\00", align 8
@m_10375091444836837428 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFand the rest of the inlined body must still run\00", align 8
@m_12094041792633930749 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan inlined throw inside a catch clause must escape it\00", align 8
@m_11121776228053187021 = internal constant [45 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa conditional throw must survive too\00", align 8
@m_9273226343044821041 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFand the returning path must still return\00", align 8
@m_12270282982231979649 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe returning path must have run its body\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@steps = private global i32 0
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: mustprogress
define void @thrower() #1 {
  %1 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 5, ptr %1, align 4
  call void @__cxa_throw(ptr %1, ptr @_ZTIi, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @callsIt() #1 {
  %1 = load i32, ptr @steps, align 4
  %2 = add i32 %1, 1
  store i32 %2, ptr @steps, align 4
  call void @thrower()
  ret void
}

; Function Attrs: mustprogress
define void @throwsFromCatch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %2, align 4
  invoke void @__cxa_throw(ptr %2, ptr @_ZTIi, ptr null)
          to label %15 unwind label %3

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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %8
  br label %15

11:                                               ; preds = %3
  %12 = extractvalue { ptr, i32 } %4, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i32, ptr %13, align 4
  store i32 %14, ptr %1, align 4
  invoke void @thrower()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %11
  call void @__cxa_end_catch()
  ret void

15:                                               ; preds = %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %8, %11
  %16 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %16
}

; Function Attrs: mustprogress
define void @throwsOnlyWhenAsked(i1 %0) #1 {
  %2 = alloca i1, align 1
  store i1 %0, ptr %2, align 1
  %3 = load i1, ptr %2, align 1
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 3, ptr %5, align 4
  call void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
  br label %9

6:                                                ; preds = %1
  %7 = load i32, ptr @steps, align 4
  %8 = add i32 %7, 10
  store i32 %8, ptr @steps, align 4
  ret void

9:                                                ; preds = %4
  unreachable
}

; Function Attrs: mustprogress
define i1 @caught({ ptr, ptr, ptr } %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca i1, align 1
  store { ptr, ptr, ptr } %0, ptr %3, align 8
  %5 = load { ptr, ptr, ptr }, ptr %3, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %10(ptr %6)
          to label %11 unwind label %18

11:                                               ; preds = %9
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %13()
          to label %14 unwind label %18

14:                                               ; preds = %12
  br label %15

15:                                               ; preds = %11, %14
  br label %16

16:                                               ; preds = %15
  br label %17

17:                                               ; preds = %16
  store i1 false, ptr %4, align 1
  br label %30

18:                                               ; preds = %9, %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %23
  br label %32

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %2, align 4
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %30

30:                                               ; preds = %17, %26
  %31 = load i1, ptr %4, align 1
  ret i1 %31

32:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %23
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define void @.f_main..afL66C19FH1577969496() #1 {
  call void @callsIt()
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL69C19FH1577969496() #1 {
  call void @throwsFromCatch()
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL72C19FH1577969496() #1 {
  call void @throwsOnlyWhenAsked(i1 true)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL73C20FH1577969496() #1 {
  call void @throwsOnlyWhenAsked(i1 false)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  store i32 0, ptr @steps, align 4
  %1 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL66C19FH1577969496, ptr null, ptr null })
  br i1 %1, label %2, label %17

2:                                                ; preds = %0
  %3 = load i32, ptr @steps, align 4
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %5, label %19

5:                                                ; preds = %2
  %6 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL69C19FH1577969496, ptr null, ptr null })
  br i1 %6, label %7, label %21

7:                                                ; preds = %5
  store i32 0, ptr @steps, align 4
  %8 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL72C19FH1577969496, ptr null, ptr null })
  br i1 %8, label %9, label %23

9:                                                ; preds = %7
  %10 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL73C20FH1577969496, ptr null, ptr null })
  %11 = xor i1 %10, true
  br i1 %11, label %12, label %25

12:                                               ; preds = %9
  %13 = load i32, ptr @steps, align 4
  %14 = icmp eq i32 %13, 10
  br i1 %14, label %15, label %27

15:                                               ; preds = %12
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

16:                                               ; preds = %27, %25, %23, %21, %19, %17
  unreachable

17:                                               ; preds = %0
  %18 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14113715647558732790, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2451893981936320601, i64 8), i32 66, ptr null)
  br label %16

19:                                               ; preds = %2
  %20 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10375091444836837428, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2451893981936320601, i64 8), i32 67, ptr null)
  br label %16

21:                                               ; preds = %5
  %22 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12094041792633930749, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2451893981936320601, i64 8), i32 69, ptr null)
  br label %16

23:                                               ; preds = %7
  %24 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11121776228053187021, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2451893981936320601, i64 8), i32 72, ptr null)
  br label %16

25:                                               ; preds = %9
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9273226343044821041, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2451893981936320601, i64 8), i32 73, ptr null)
  br label %16

27:                                               ; preds = %12
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12270282982231979649, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2451893981936320601, i64 8), i32 74, ptr null)
  br label %16
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

