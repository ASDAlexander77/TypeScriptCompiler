; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_2733085097509461336 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa throw from a catch must reach the caller\00", align 8
@f_11711661141449983451 = internal constant [106 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_in_catch.ts\00", align 8
@m_4874075906784006237 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa rethrow from a catch must reach the caller\00", align 8
@m_7113881072353842081 = internal constant [76 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe outer handler must take it, and the inner one must not re-catch\00", align 8
@m_1427375041123764199 = internal constant [65 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa catch that does not throw still runs and ends normally\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define void @throwsALiteralFromCatch() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %2, align 4
  invoke void @__cxa_throw(ptr %2, ptr @_ZTIi, ptr null)
          to label %16 unwind label %3

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
  br label %16

11:                                               ; preds = %3
  %12 = extractvalue { ptr, i32 } %4, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i32, ptr %13, align 4
  store i32 %14, ptr %1, align 4
  call void @__cxa_end_catch()
  %15 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %15, align 4
  call void @__cxa_throw(ptr %15, ptr @_ZTIi, ptr null)
  br label %16

16:                                               ; preds = %11, %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %8
  %17 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %17
}

; Function Attrs: mustprogress
define void @rethrows() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 7, ptr %2, align 4
  invoke void @__cxa_throw(ptr %2, ptr @_ZTIi, ptr null)
          to label %17 unwind label %3

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
  br label %17

11:                                               ; preds = %3
  %12 = extractvalue { ptr, i32 } %4, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i32, ptr %13, align 4
  store i32 %14, ptr %1, align 4
  %15 = load i32, ptr %1, align 4
  call void @__cxa_end_catch()
  %16 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 %15, ptr %16, align 4
  call void @__cxa_throw(ptr %16, ptr @_ZTIi, ptr null)
  br label %17

17:                                               ; preds = %11, %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %8
  %18 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %18
}

; Function Attrs: mustprogress
define i32 @nestedThrowFromCatch() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %39 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %39 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %1, align 4
  %18 = load i32, ptr %3, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %3, align 4
  call void @__cxa_end_catch()
  %20 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %20, align 4
  invoke void @__cxa_throw(ptr %20, ptr @_ZTIi, ptr null)
          to label %39 unwind label %22

endcatch.pad:                                     ; preds = %11
  %21 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

22:                                               ; preds = %14
  %23 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad ]
  %25 = extractvalue { ptr, i32 } %24, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %27 = icmp eq i32 %25, %26
  br i1 %27, label %31, label %28

28:                                               ; preds = %.caught
  %29 = extractvalue { ptr, i32 } %24, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  br label %39

31:                                               ; preds = %.caught
  %32 = extractvalue { ptr, i32 } %24, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  %34 = load i32, ptr %33, align 4
  store i32 %34, ptr %2, align 4
  %35 = load i32, ptr %3, align 4
  %36 = add i32 %35, 10
  store i32 %36, ptr %3, align 4
  call void @__cxa_end_catch()
  %37 = load i32, ptr %3, align 4
  store i32 %37, ptr %4, align 4
  %38 = load i32, ptr %4, align 4
  ret i32 %38

39:                                               ; preds = %.noexc, %14, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %28
  %40 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %40
}

; Function Attrs: mustprogress
define i32 @plainCatch() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %19 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %10
  br label %19

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i32, ptr %15, align 4
  store i32 %16, ptr %1, align 4
  store i32 1, ptr %2, align 4
  call void @__cxa_end_catch()
  %17 = load i32, ptr %2, align 4
  store i32 %17, ptr %3, align 4
  %18 = load i32, ptr %3, align 4
  ret i32 %18

19:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %20 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %20
}

; Function Attrs: mustprogress
define i1 @caught({ ptr, ptr, ptr } %0) #1 personality ptr @__gxx_personality_v0 {
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
define void @.f_main..afL84C19FH760567015() #2 {
  call void @throwsALiteralFromCatch()
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL85C19FH760567015() #2 {
  call void @rethrows()
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL84C19FH760567015, ptr null, ptr null })
  br i1 %1, label %2, label %12

2:                                                ; preds = %0
  %3 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL85C19FH760567015, ptr null, ptr null })
  br i1 %3, label %4, label %14

4:                                                ; preds = %2
  %5 = call i32 @nestedThrowFromCatch()
  %6 = icmp eq i32 %5, 11
  br i1 %6, label %7, label %16

7:                                                ; preds = %4
  %8 = call i32 @plainCatch()
  %9 = icmp eq i32 %8, 1
  br i1 %9, label %10, label %18

10:                                               ; preds = %7
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

11:                                               ; preds = %18, %16, %14, %12
  unreachable

12:                                               ; preds = %0
  %13 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2733085097509461336, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11711661141449983451, i64 8), i32 84, ptr null)
  br label %11

14:                                               ; preds = %2
  %15 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4874075906784006237, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11711661141449983451, i64 8), i32 85, ptr null)
  br label %11

16:                                               ; preds = %4
  %17 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7113881072353842081, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11711661141449983451, i64 8), i32 86, ptr null)
  br label %11

18:                                               ; preds = %7
  %19 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1427375041123764199, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11711661141449983451, i64 8), i32 87, ptr null)
  br label %11
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
attributes #2 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

