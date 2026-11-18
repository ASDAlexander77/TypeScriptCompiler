; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_12638206991764949794 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs\00", align 8
@s_12638201494206808739 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFt\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_15843314615243799497 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa matching nested catch inside a catch with a finally\00", align 8
@f_2841482110013058845 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_in_catch_with_finally.ts\00", align 8
@m_1112154543766939056 = internal constant [74 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested mismatch runs the outer finally, then reaches the caller\00", align 8
@m_2079452496248405301 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboth finally blocks run, innermost first\00", align 8
@m_17924869790678909399 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa throw after a nested try still runs the finally\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@trail = private global i32 0
@_ZTIi = external constant ptr
@_ZTIPc = external constant ptr

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
define void @step(double %0) #1 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load i32, ptr @trail, align 4
  %4 = mul i32 %3, 10
  %5 = load double, ptr %2, align 8
  %6 = sitofp i32 %4 to double
  %7 = fadd double %6, %5
  %8 = fptosi double %7 to i64
  %9 = trunc i64 %8 to i32
  store i32 %9, ptr @trail, align 4
  ret void
}

; Function Attrs: mustprogress
define void @nestedMatches() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %37 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad1

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %2, align 4
  invoke void @step(double 1.000000e+00)
          to label %16 unwind label %endcatch.pad1

16:                                               ; preds = %12
  %17 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  store i32 2, ptr %17, align 4
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIi, ptr null)
          to label %37 unwind label %18

18:                                               ; preds = %.noexc
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %1, align 4
  invoke void @step(double 9.000000e+00)
          to label %30 unwind label %endcatch.pad

30:                                               ; preds = %26
  call void @__cxa_end_catch()
  invoke void @step(double 8.000000e+00)
          to label %31 unwind label %endcatch.pad1

31:                                               ; preds = %30
  call void @__cxa_end_catch()
  call void @step(double 2.000000e+00)
  ret void

endcatch.pad:                                     ; preds = %23, %26
  %32 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

endcatch.pad1:                                    ; preds = %9, %12, %30
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad1, %endcatch.pad, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %32, %endcatch.pad ], [ %33, %endcatch.pad1 ]
  call void @step(double 2.000000e+00)
  resume { ptr, i32 } %36

37:                                               ; preds = %23, %.noexc, %9, %0
  unreachable

endcatch.resume:                                  ; preds = %16
  %38 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %38
}

; Function Attrs: mustprogress
define void @nestedMismatchRunsOuterFinally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %37 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad1

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %2, align 4
  invoke void @step(double 1.000000e+00)
          to label %16 unwind label %endcatch.pad1

16:                                               ; preds = %12
  %17 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8), ptr %17, align 8
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIPc, ptr null)
          to label %37 unwind label %18

18:                                               ; preds = %.noexc
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %1, align 4
  invoke void @step(double 9.000000e+00)
          to label %30 unwind label %endcatch.pad

30:                                               ; preds = %26
  call void @__cxa_end_catch()
  invoke void @step(double 8.000000e+00)
          to label %31 unwind label %endcatch.pad1

31:                                               ; preds = %30
  call void @__cxa_end_catch()
  call void @step(double 2.000000e+00)
  ret void

endcatch.pad:                                     ; preds = %26, %23
  %32 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

endcatch.pad1:                                    ; preds = %9, %30, %12
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad1, %endcatch.pad, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %32, %endcatch.pad ], [ %33, %endcatch.pad1 ]
  call void @step(double 2.000000e+00)
  resume { ptr, i32 } %36

37:                                               ; preds = %23, %.noexc, %9, %0
  unreachable

endcatch.resume:                                  ; preds = %16
  %38 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %38
}

; Function Attrs: mustprogress
define void @bothFinallysRunInnermostFirst() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %45 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %45 unwind label %endcatch.pad1

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %2, align 4
  invoke void @step(double 1.000000e+00)
          to label %16 unwind label %endcatch.pad1

16:                                               ; preds = %12
  %17 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8), ptr %17, align 8
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIPc, ptr null)
          to label %45 unwind label %18

18:                                               ; preds = %.noexc
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %45 unwind label %endcatch.pad3

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %1, align 4
  invoke void @step(double 9.000000e+00)
          to label %30 unwind label %endcatch.pad3

30:                                               ; preds = %26
  call void @__cxa_end_catch()
  invoke void @step(double 5.000000e+00)
          to label %38 unwind label %endcatch.pad1

endcatch.pad3:                                    ; preds = %26, %23
  %31 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught2

32:                                               ; No predecessors!
  %33 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught2

.caught2:                                         ; preds = %endcatch.pad3, %32
  %34 = phi { ptr, i32 } [ %33, %32 ], [ %31, %endcatch.pad3 ]
  %35 = extractvalue { ptr, i32 } %34, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  invoke void @step(double 5.000000e+00)
          to label %37 unwind label %endcatch.pad

37:                                               ; preds = %.caught2
  invoke void @__cxa_rethrow()
          to label %45 unwind label %endcatch.pad

38:                                               ; preds = %30
  invoke void @step(double 8.000000e+00)
          to label %39 unwind label %endcatch.pad1

39:                                               ; preds = %38
  call void @__cxa_end_catch()
  call void @step(double 2.000000e+00)
  ret void

endcatch.pad:                                     ; preds = %.caught2, %37
  %40 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

endcatch.pad1:                                    ; preds = %30, %9, %12, %38
  %41 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

42:                                               ; No predecessors!
  %43 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad1, %endcatch.pad, %42
  %44 = phi { ptr, i32 } [ %43, %42 ], [ %40, %endcatch.pad ], [ %41, %endcatch.pad1 ]
  call void @step(double 2.000000e+00)
  resume { ptr, i32 } %44

45:                                               ; preds = %37, %23, %.noexc, %9, %0
  unreachable

endcatch.resume:                                  ; preds = %16
  %46 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %46
}

; Function Attrs: mustprogress
define void @throwAfterNestedTry() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %37 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad1

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %2, align 4
  invoke void @step(double 1.000000e+00)
          to label %16 unwind label %endcatch.pad1

16:                                               ; preds = %12
  %17 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  store i32 2, ptr %17, align 4
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIi, ptr null)
          to label %37 unwind label %18

18:                                               ; preds = %.noexc
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %1, align 4
  invoke void @step(double 9.000000e+00)
          to label %30 unwind label %endcatch.pad

30:                                               ; preds = %26
  call void @__cxa_end_catch()
  %31 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %30
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638201494206808739, i64 8), ptr %31, align 8
  invoke void @__cxa_throw(ptr %31, ptr @_ZTIPc, ptr null)
          to label %37 unwind label %endcatch.pad1

endcatch.pad:                                     ; preds = %26, %23
  %32 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

endcatch.pad1:                                    ; preds = %9, %12, %.noexc2
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad1, %endcatch.pad, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %32, %endcatch.pad ], [ %33, %endcatch.pad1 ]
  call void @step(double 2.000000e+00)
  resume { ptr, i32 } %36

37:                                               ; preds = %.noexc2, %23, %.noexc, %9, %0
  unreachable

endcatch.resume:                                  ; preds = %30, %16
  %38 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %38
}

; Function Attrs: mustprogress
define i32 @run({ ptr, ptr, ptr } %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca i32, align 4
  store { ptr, ptr, ptr } %0, ptr %3, align 8
  store i32 0, ptr @trail, align 4
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
  br label %29

18:                                               ; preds = %9, %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
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
  store ptr %28, ptr %2, align 8
  invoke void @step(double 3.000000e+00)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %26
  call void @__cxa_end_catch()
  br label %29

29:                                               ; preds = %17, %.noexc1
  %30 = load i32, ptr @trail, align 4
  store i32 %30, ptr %4, align 4
  %31 = load i32, ptr %4, align 4
  ret i32 %31

32:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %26, %23
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i32 @run({ ptr, ptr, ptr } { ptr @nestedMatches, ptr null, ptr null })
  %2 = icmp eq i32 %1, 1982
  br i1 %2, label %3, label %14

3:                                                ; preds = %0
  %4 = call i32 @run({ ptr, ptr, ptr } { ptr @nestedMismatchRunsOuterFinally, ptr null, ptr null })
  %5 = icmp eq i32 %4, 123
  br i1 %5, label %6, label %16

6:                                                ; preds = %3
  %7 = call i32 @run({ ptr, ptr, ptr } { ptr @bothFinallysRunInnermostFirst, ptr null, ptr null })
  %8 = icmp eq i32 %7, 1523
  br i1 %8, label %9, label %18

9:                                                ; preds = %6
  %10 = call i32 @run({ ptr, ptr, ptr } { ptr @throwAfterNestedTry, ptr null, ptr null })
  %11 = icmp eq i32 %10, 1923
  br i1 %11, label %12, label %20

12:                                               ; preds = %9
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

13:                                               ; preds = %20, %18, %16, %14
  unreachable

14:                                               ; preds = %0
  %15 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15843314615243799497, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2841482110013058845, i64 8), i32 100, ptr null)
  br label %13

16:                                               ; preds = %3
  %17 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1112154543766939056, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2841482110013058845, i64 8), i32 101, ptr null)
  br label %13

18:                                               ; preds = %6
  %19 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2079452496248405301, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2841482110013058845, i64 8), i32 102, ptr null)
  br label %13

20:                                               ; preds = %9
  %21 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17924869790678909399, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2841482110013058845, i64 8), i32 103, ptr null)
  br label %13
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

