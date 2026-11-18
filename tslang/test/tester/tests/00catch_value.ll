; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_14967900074057001445 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFpayload\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_16072557159647594798 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan int catch binds the thrown value\00", align 8
@f_11319436400589141251 = internal constant [103 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00catch_value.ts\00", align 8
@m_7511134177193727212 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa number catch binds the thrown value\00", align 8
@m_2423834359136791228 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa string catch binds the thrown value\00", align 8
@m_5586682363366401738 = internal constant [60 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthree catches of one type each bind their own value\00", align 8
@m_15098273051152703464 = internal constant [65 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe int catch of a two-type function binds its own value\00", align 8
@m_4346451784906322496 = internal constant [68 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe number catch of a two-type function binds its own value\00", align 8
@m_17790822031473819596 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe bound value outlives the clause\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIi = external constant ptr
@_ZTId = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIPc = external constant ptr
@twoTypesInt = private global i32 0
@twoTypesNumber = private global double 0.000000e+00

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare i32 @strcmp(ptr, ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__cxa_current_exception_type()

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
define i32 @caughtInt() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %20 unwind label %5

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
  br label %20

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i32, ptr %15, align 4
  store i32 %16, ptr %1, align 4
  %17 = load i32, ptr %1, align 4
  store i32 %17, ptr %2, align 4
  call void @__cxa_end_catch()
  %18 = load i32, ptr %2, align 4
  store i32 %18, ptr %3, align 4
  %19 = load i32, ptr %3, align 4
  ret i32 %19

20:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %21 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %21
}

; Function Attrs: mustprogress
define double @caughtNumber() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store double 2.500000e+00, ptr %4, align 8
  invoke void @__cxa_throw(ptr %4, ptr @_ZTId, ptr null)
          to label %57 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %9 = icmp eq i32 %7, %8
  %10 = extractvalue { ptr, i32 } %6, 1
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %12 = icmp eq i32 %10, %11
  %13 = extractvalue { ptr, i32 } %6, 1
  %14 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %15 = icmp eq i32 %13, %14
  %16 = extractvalue { ptr, i32 } %6, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %18 = icmp eq i32 %16, %17
  %19 = or i1 %9, %12
  %20 = or i1 %19, %15
  %21 = or i1 %20, %18
  br i1 %21, label %25, label %22

22:                                               ; preds = %5
  %23 = extractvalue { ptr, i32 } %6, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %22
  br label %57

25:                                               ; preds = %5
  %26 = extractvalue { ptr, i32 } %6, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  %28 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %25
  %29 = icmp eq ptr %28, @_ZTIi
  br i1 %29, label %30, label %33

30:                                               ; preds = %.noexc
  %31 = load i32, ptr %27, align 4
  %32 = sitofp i32 %31 to double
  br label %51

33:                                               ; preds = %.noexc
  %34 = icmp eq ptr %28, @_ZTIj
  br i1 %34, label %35, label %38

35:                                               ; preds = %33
  %36 = load i32, ptr %27, align 4
  %37 = uitofp i32 %36 to double
  br label %48

38:                                               ; preds = %33
  %39 = icmp eq ptr %28, @_ZTIf
  br i1 %39, label %40, label %43

40:                                               ; preds = %38
  %41 = load float, ptr %27, align 4
  %42 = fpext float %41 to double
  br label %45

43:                                               ; preds = %38
  %44 = load double, ptr %27, align 8
  br label %45

45:                                               ; preds = %40, %43
  %46 = phi double [ %44, %43 ], [ %42, %40 ]
  br label %47

47:                                               ; preds = %45
  br label %48

48:                                               ; preds = %35, %47
  %49 = phi double [ %46, %47 ], [ %37, %35 ]
  br label %50

50:                                               ; preds = %48
  br label %51

51:                                               ; preds = %30, %50
  %52 = phi double [ %49, %50 ], [ %32, %30 ]
  br label %53

53:                                               ; preds = %51
  store double %52, ptr %1, align 8
  %54 = load double, ptr %1, align 8
  store double %54, ptr %2, align 8
  call void @__cxa_end_catch()
  %55 = load double, ptr %2, align 8
  store double %55, ptr %3, align 8
  %56 = load double, ptr %3, align 8
  ret double %56

57:                                               ; preds = %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %22, %25
  %58 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %58
}

; Function Attrs: mustprogress
define ptr @caughtString() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_14967900074057001445, i64 8), ptr %4, align 8
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIPc, ptr null)
          to label %23 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %23 unwind label %endcatch.pad

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  store ptr %15, ptr %1, align 8
  %16 = load ptr, ptr %1, align 8
  store ptr %16, ptr %2, align 8
  call void @__cxa_end_catch()
  %17 = load ptr, ptr %2, align 8
  store ptr %17, ptr %3, align 8
  %18 = load ptr, ptr %3, align 8
  ret ptr %18

endcatch.pad:                                     ; preds = %10
  %19 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

20:                                               ; No predecessors!
  %21 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %20
  %22 = phi { ptr, i32 } [ %21, %20 ], [ %19, %endcatch.pad ]
  resume { ptr, i32 } %22

23:                                               ; preds = %10, %0
  unreachable
}

; Function Attrs: mustprogress
define i32 @threeInARow() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 0, ptr %6, align 4
  store i32 0, ptr %5, align 4
  store i32 0, ptr %4, align 4
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 7, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %58 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %11 = extractvalue { ptr, i32 } %10, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %17, label %14

14:                                               ; preds = %9
  %15 = extractvalue { ptr, i32 } %10, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %14
  br label %58

17:                                               ; preds = %9
  %18 = extractvalue { ptr, i32 } %10, 0
  %19 = call ptr @__cxa_begin_catch(ptr %18)
  %20 = load i32, ptr %19, align 4
  store i32 %20, ptr %3, align 4
  %21 = load i32, ptr %3, align 4
  store i32 %21, ptr %6, align 4
  call void @__cxa_end_catch()
  %22 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 8, ptr %22, align 4
  invoke void @__cxa_throw(ptr %22, ptr @_ZTIi, ptr null)
          to label %58 unwind label %23

23:                                               ; preds = %17
  %24 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %25 = extractvalue { ptr, i32 } %24, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %27 = icmp eq i32 %25, %26
  br i1 %27, label %31, label %28

28:                                               ; preds = %23
  %29 = extractvalue { ptr, i32 } %24, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  br label %58

31:                                               ; preds = %23
  %32 = extractvalue { ptr, i32 } %24, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  %34 = load i32, ptr %33, align 4
  store i32 %34, ptr %2, align 4
  %35 = load i32, ptr %2, align 4
  store i32 %35, ptr %5, align 4
  call void @__cxa_end_catch()
  %36 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 9, ptr %36, align 4
  invoke void @__cxa_throw(ptr %36, ptr @_ZTIi, ptr null)
          to label %58 unwind label %37

37:                                               ; preds = %31
  %38 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %39 = extractvalue { ptr, i32 } %38, 1
  %40 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %41 = icmp eq i32 %39, %40
  br i1 %41, label %45, label %42

42:                                               ; preds = %37
  %43 = extractvalue { ptr, i32 } %38, 0
  %44 = call ptr @__cxa_begin_catch(ptr %43)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %42
  br label %58

45:                                               ; preds = %37
  %46 = extractvalue { ptr, i32 } %38, 0
  %47 = call ptr @__cxa_begin_catch(ptr %46)
  %48 = load i32, ptr %47, align 4
  store i32 %48, ptr %1, align 4
  %49 = load i32, ptr %1, align 4
  store i32 %49, ptr %4, align 4
  call void @__cxa_end_catch()
  %50 = load i32, ptr %6, align 4
  %51 = mul i32 %50, 100
  %52 = load i32, ptr %5, align 4
  %53 = mul i32 %52, 10
  %54 = add i32 %51, %53
  %55 = load i32, ptr %4, align 4
  %56 = add i32 %54, %55
  store i32 %56, ptr %7, align 4
  %57 = load i32, ptr %7, align 4
  ret i32 %57

58:                                               ; preds = %.noexc1, %31, %.noexc, %17, %.noexc2, %0
  unreachable

endcatch.resume:                                  ; preds = %14, %42, %28
  %59 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %59
}

; Function Attrs: mustprogress
define void @twoTypes() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca i32, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %68 unwind label %4

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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %9
  br label %68

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %2, align 4
  %16 = load i32, ptr %2, align 4
  store i32 %16, ptr @twoTypesInt, align 4
  call void @__cxa_end_catch()
  %17 = call ptr @__cxa_allocate_exception(i64 8)
  store double 3.500000e+00, ptr %17, align 8
  invoke void @__cxa_throw(ptr %17, ptr @_ZTId, ptr null)
          to label %68 unwind label %18

18:                                               ; preds = %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %22 = icmp eq i32 %20, %21
  %23 = extractvalue { ptr, i32 } %19, 1
  %24 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %25 = icmp eq i32 %23, %24
  %26 = extractvalue { ptr, i32 } %19, 1
  %27 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %28 = icmp eq i32 %26, %27
  %29 = extractvalue { ptr, i32 } %19, 1
  %30 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %31 = icmp eq i32 %29, %30
  %32 = or i1 %22, %25
  %33 = or i1 %32, %28
  %34 = or i1 %33, %31
  br i1 %34, label %38, label %35

35:                                               ; preds = %18
  %36 = extractvalue { ptr, i32 } %19, 0
  %37 = call ptr @__cxa_begin_catch(ptr %36)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %35
  br label %68

38:                                               ; preds = %18
  %39 = extractvalue { ptr, i32 } %19, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  %41 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %38
  %42 = icmp eq ptr %41, @_ZTIi
  br i1 %42, label %43, label %46

43:                                               ; preds = %.noexc
  %44 = load i32, ptr %40, align 4
  %45 = sitofp i32 %44 to double
  br label %64

46:                                               ; preds = %.noexc
  %47 = icmp eq ptr %41, @_ZTIj
  br i1 %47, label %48, label %51

48:                                               ; preds = %46
  %49 = load i32, ptr %40, align 4
  %50 = uitofp i32 %49 to double
  br label %61

51:                                               ; preds = %46
  %52 = icmp eq ptr %41, @_ZTIf
  br i1 %52, label %53, label %56

53:                                               ; preds = %51
  %54 = load float, ptr %40, align 4
  %55 = fpext float %54 to double
  br label %58

56:                                               ; preds = %51
  %57 = load double, ptr %40, align 8
  br label %58

58:                                               ; preds = %53, %56
  %59 = phi double [ %57, %56 ], [ %55, %53 ]
  br label %60

60:                                               ; preds = %58
  br label %61

61:                                               ; preds = %48, %60
  %62 = phi double [ %59, %60 ], [ %50, %48 ]
  br label %63

63:                                               ; preds = %61
  br label %64

64:                                               ; preds = %43, %63
  %65 = phi double [ %62, %63 ], [ %45, %43 ]
  br label %66

66:                                               ; preds = %64
  store double %65, ptr %1, align 8
  %67 = load double, ptr %1, align 8
  store double %67, ptr @twoTypesNumber, align 8
  call void @__cxa_end_catch()
  ret void

68:                                               ; preds = %.noexc1, %12, %.noexc2, %0
  unreachable

endcatch.resume:                                  ; preds = %9, %35, %38
  %69 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %69
}

; Function Attrs: mustprogress
define i32 @readAfterTheClause() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 41, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %22 unwind label %5

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
  br label %22

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i32, ptr %15, align 4
  store i32 %16, ptr %1, align 4
  %17 = load i32, ptr %1, align 4
  store i32 %17, ptr %2, align 4
  call void @__cxa_end_catch()
  %18 = load i32, ptr %2, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %2, align 4
  %20 = load i32, ptr %2, align 4
  store i32 %20, ptr %3, align 4
  %21 = load i32, ptr %3, align 4
  ret i32 %21

22:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %23 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %23
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i32 @caughtInt()
  %2 = icmp eq i32 %1, 2
  br i1 %2, label %3, label %35

3:                                                ; preds = %0
  %4 = call double @caughtNumber()
  %5 = fcmp oeq double %4, 2.500000e+00
  br i1 %5, label %6, label %37

6:                                                ; preds = %3
  %7 = call ptr @caughtString()
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %6
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_14967900074057001445, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %6
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14967900074057001445, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %39

21:                                               ; preds = %20
  %22 = call i32 @threeInARow()
  %23 = icmp eq i32 %22, 789
  br i1 %23, label %24, label %41

24:                                               ; preds = %21
  call void @twoTypes()
  %25 = load i32, ptr @twoTypesInt, align 4
  %26 = icmp eq i32 %25, 2
  br i1 %26, label %27, label %43

27:                                               ; preds = %24
  %28 = load double, ptr @twoTypesNumber, align 8
  %29 = fcmp oeq double %28, 3.500000e+00
  br i1 %29, label %30, label %45

30:                                               ; preds = %27
  %31 = call i32 @readAfterTheClause()
  %32 = icmp eq i32 %31, 42
  br i1 %32, label %33, label %47

33:                                               ; preds = %30
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

34:                                               ; preds = %47, %45, %43, %41, %39, %37, %35
  unreachable

35:                                               ; preds = %0
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16072557159647594798, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 77, ptr null)
  br label %34

37:                                               ; preds = %3
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7511134177193727212, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 78, ptr null)
  br label %34

39:                                               ; preds = %20
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2423834359136791228, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 79, ptr null)
  br label %34

41:                                               ; preds = %21
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5586682363366401738, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 80, ptr null)
  br label %34

43:                                               ; preds = %24
  %44 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15098273051152703464, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 82, ptr null)
  br label %34

45:                                               ; preds = %27
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4346451784906322496, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 83, ptr null)
  br label %34

47:                                               ; preds = %30
  %48 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17790822031473819596, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11319436400589141251, i64 8), i32 84, ptr null)
  br label %34
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

