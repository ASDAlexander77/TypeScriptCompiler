; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@a_3_17496507242642067975 = internal constant <3 x i32> <i32 0, i32 1, i32 2>
@s_3815361247288411516 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnamespace exceptions {\0A@dllimport\0Afunction run();\0A\0A}\0A\00", align 8
@s_618458831055183935 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFgc\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_11350717838621562508 = internal constant [102 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\51exceptions.ts\00", align 8
@s_15902901984413996407 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfoo\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_3511218394696192748 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtest exn done\00", align 8
@s_3154112327090001444 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtest exn\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@glb1 = private global i32 0
@x = private global i32 0
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIPc = external constant ptr
@__decls_51exceptions_1949474134872955442 = global ptr getelementptr inbounds nuw (i8, ptr @s_3815361247288411516, i64 8)
@__tsmm_gc_51exceptions_1949474134872955442 = global ptr getelementptr inbounds nuw (i8, ptr @s_618458831055183935, i64 8)
@td_6046455916167272971_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_6046455916167272971, ptr @tsretcb_6046455916167272971, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

define internal void @__tslang_inc_ref(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %13

3:                                                ; preds = %1
  %4 = getelementptr i8, ptr %0, i64 -8
  %5 = load i64, ptr %4, align 8
  %6 = icmp ne i64 %5, -1
  %7 = icmp sge i64 %5, 0
  br i1 %7, label %11, label %8

8:                                                ; preds = %3
  br i1 %6, label %9, label %13

9:                                                ; preds = %8
  %10 = atomicrmw add ptr %4, i64 1 monotonic, align 8
  br label %13

11:                                               ; preds = %3
  %12 = add i64 %5, 1
  store i64 %12, ptr %4, align 8
  br label %13

13:                                               ; preds = %11, %9, %8, %1
  ret void
}

define internal void @tsretcb_6046455916167272971(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal i1 @__tslang_dec_ref(ptr %0) {
  %2 = getelementptr i8, ptr %0, i64 -8
  %3 = load i64, ptr %2, align 8
  %4 = icmp ne i64 %3, -1
  %5 = icmp sge i64 %3, 0
  br i1 %5, label %11, label %6

6:                                                ; preds = %1
  br i1 %4, label %7, label %15

7:                                                ; preds = %6
  %8 = atomicrmw sub ptr %2, i64 1 release, align 8
  %9 = and i64 %8, 9223372036854775807
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %14, label %15

11:                                               ; preds = %1
  %12 = sub i64 %3, 1
  store i64 %12, ptr %2, align 8
  %13 = icmp eq i64 %12, 0
  ret i1 %13

14:                                               ; preds = %7
  fence acquire
  ret i1 true

15:                                               ; preds = %7, %6
  ret i1 false
}

declare void @GC_free(ptr)

define internal void @__tslang_free_block(ptr %0) {
  %2 = icmp eq ptr %0, null
  %3 = getelementptr i8, ptr %0, i64 -8
  %4 = select i1 %2, ptr null, ptr %3
  call void @GC_free(ptr %4)
  ret void
}

define internal void @tsrelcb_6046455916167272971(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  call void @__tslang_free_block(ptr %2)
  br label %7

7:                                                ; preds = %6, %4, %1
  ret void
}

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #2

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define void @pause(double %0) #3 {
  ret void
}

; Function Attrs: mustprogress
define void @exceptions.immediate(double %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %3, align 8
  invoke void @pause(double 1.000000e+00)
          to label %4 unwind label %14

4:                                                ; preds = %1
  %5 = load double, ptr %3, align 8
  %6 = fcmp ogt double %5, 0.000000e+00
  br i1 %6, label %7, label %10

7:                                                ; preds = %4
  %8 = load double, ptr %3, align 8
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store double %8, ptr %9, align 8
  invoke void @__cxa_throw(ptr %9, ptr @_ZTId, ptr null)
          to label %86 unwind label %14

10:                                               ; preds = %4
  invoke void @pause(double 1.000000e+00)
          to label %11 unwind label %14

11:                                               ; preds = %10
  %12 = load i32, ptr @glb1, align 4
  %13 = add i32 %12, 1
  store i32 %13, ptr @glb1, align 4
  br label %82

14:                                               ; preds = %7, %10, %1
  %15 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %16 = extractvalue { ptr, i32 } %15, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %18 = icmp eq i32 %16, %17
  %19 = extractvalue { ptr, i32 } %15, 1
  %20 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %21 = icmp eq i32 %19, %20
  %22 = extractvalue { ptr, i32 } %15, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %24 = icmp eq i32 %22, %23
  %25 = extractvalue { ptr, i32 } %15, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %27 = icmp eq i32 %25, %26
  %28 = or i1 %18, %21
  %29 = or i1 %28, %24
  %30 = or i1 %29, %27
  br i1 %30, label %34, label %31

31:                                               ; preds = %14
  %32 = extractvalue { ptr, i32 } %15, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  invoke void @__cxa_rethrow()
          to label %86 unwind label %endcatch.pad

34:                                               ; preds = %14
  %35 = extractvalue { ptr, i32 } %15, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  %37 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %34
  %38 = icmp eq ptr %37, @_ZTIi
  br i1 %38, label %39, label %42

39:                                               ; preds = %.noexc3
  %40 = load i32, ptr %36, align 4
  %41 = sitofp i32 %40 to double
  br label %60

42:                                               ; preds = %.noexc3
  %43 = icmp eq ptr %37, @_ZTIj
  br i1 %43, label %44, label %47

44:                                               ; preds = %42
  %45 = load i32, ptr %36, align 4
  %46 = uitofp i32 %45 to double
  br label %57

47:                                               ; preds = %42
  %48 = icmp eq ptr %37, @_ZTIf
  br i1 %48, label %49, label %52

49:                                               ; preds = %47
  %50 = load float, ptr %36, align 4
  %51 = fpext float %50 to double
  br label %54

52:                                               ; preds = %47
  %53 = load double, ptr %36, align 8
  br label %54

54:                                               ; preds = %49, %52
  %55 = phi double [ %53, %52 ], [ %51, %49 ]
  br label %56

56:                                               ; preds = %54
  br label %57

57:                                               ; preds = %44, %56
  %58 = phi double [ %55, %56 ], [ %46, %44 ]
  br label %59

59:                                               ; preds = %57
  br label %60

60:                                               ; preds = %39, %59
  %61 = phi double [ %58, %59 ], [ %41, %39 ]
  br label %62

62:                                               ; preds = %60
  store double %61, ptr %2, align 8
  %63 = load double, ptr %2, align 8
  %64 = load double, ptr %3, align 8
  %65 = fcmp oeq double %63, %64
  br i1 %65, label %66, label %87

66:                                               ; preds = %62
  %67 = load i32, ptr @glb1, align 4
  %68 = add i32 %67, 10
  store i32 %68, ptr @glb1, align 4
  %69 = load double, ptr %3, align 8
  %70 = fcmp oge double %69, 1.000000e+01
  br i1 %70, label %71, label %74

71:                                               ; preds = %66
  %72 = load double, ptr %2, align 8
  %73 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %71
  store double %72, ptr %73, align 8
  invoke void @__cxa_throw(ptr %73, ptr @_ZTId, ptr null)
          to label %86 unwind label %endcatch.pad

74:                                               ; preds = %66
  call void @__cxa_end_catch()
  br label %82

endcatch.pad:                                     ; preds = %31, %.noexc2
  %75 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

76:                                               ; No predecessors!
  %77 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %76
  %78 = phi { ptr, i32 } [ %77, %76 ], [ %75, %endcatch.pad ]
  %79 = load i32, ptr @x, align 4
  %80 = load i32, ptr @glb1, align 4
  %81 = add i32 %79, %80
  store i32 %81, ptr @x, align 4
  resume { ptr, i32 } %78

82:                                               ; preds = %74, %11
  %83 = load i32, ptr @x, align 4
  %84 = load i32, ptr @glb1, align 4
  %85 = add i32 %83, %84
  store i32 %85, ptr @x, align 4
  ret void

86:                                               ; preds = %.noexc2, %.noexc1, %31, %7
  unreachable

87:                                               ; preds = %62
  %88 = invoke i32 @fflush(ptr null)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %87
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 15, ptr null)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.noexc
  br label %86

endcatch.resume:                                  ; preds = %34, %71, %.noexc, %87
  %89 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %89
}

; Function Attrs: mustprogress
define void @exceptions.throwVal(double %0) #3 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  call void @pause(double 1.000000e+00)
  %3 = load double, ptr %2, align 8
  %4 = fcmp ogt double %3, 0.000000e+00
  br i1 %4, label %5, label %8

5:                                                ; preds = %1
  %6 = load double, ptr %2, align 8
  %7 = call ptr @__cxa_allocate_exception(i64 8)
  store double %6, ptr %7, align 8
  call void @__cxa_throw(ptr %7, ptr @_ZTId, ptr null)
  br label %9

8:                                                ; preds = %1
  call void @pause(double 1.000000e+00)
  ret void

9:                                                ; preds = %5
  unreachable
}

; Function Attrs: mustprogress
define void @exceptions..f_higherorder..afL35C14FH2615810610(ptr %0) #3 {
  %2 = getelementptr { double }, ptr %0, i32 0, i32 0
  %3 = load double, ptr %2, align 8
  call void @exceptions.throwVal(double %3)
  ret void
}

; Function Attrs: mustprogress
define void @exceptions.higherorder(double %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %3, align 8
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { double }, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @exceptions..f_higherorder..afL35C14FH2615810610, ptr null, ptr null }, ptr %5, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_6046455916167272971_object, i64 40), 2
  invoke void @exceptions..f_higherorder..afL35C14FH2615810610(ptr %5)
          to label %10 unwind label %13

10:                                               ; preds = %1
  %11 = load i32, ptr @glb1, align 4
  %12 = add i32 %11, 1
  store i32 %12, ptr @glb1, align 4
  br label %81

13:                                               ; preds = %1
  %14 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %15 = extractvalue { ptr, i32 } %14, 1
  %16 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %17 = icmp eq i32 %15, %16
  %18 = extractvalue { ptr, i32 } %14, 1
  %19 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %20 = icmp eq i32 %18, %19
  %21 = extractvalue { ptr, i32 } %14, 1
  %22 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %23 = icmp eq i32 %21, %22
  %24 = extractvalue { ptr, i32 } %14, 1
  %25 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %26 = icmp eq i32 %24, %25
  %27 = or i1 %17, %20
  %28 = or i1 %27, %23
  %29 = or i1 %28, %26
  br i1 %29, label %33, label %30

30:                                               ; preds = %13
  %31 = extractvalue { ptr, i32 } %14, 0
  %32 = call ptr @__cxa_begin_catch(ptr %31)
  invoke void @__cxa_rethrow()
          to label %85 unwind label %endcatch.pad

33:                                               ; preds = %13
  %34 = extractvalue { ptr, i32 } %14, 0
  %35 = call ptr @__cxa_begin_catch(ptr %34)
  %36 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %33
  %37 = icmp eq ptr %36, @_ZTIi
  br i1 %37, label %38, label %41

38:                                               ; preds = %.noexc
  %39 = load i32, ptr %35, align 4
  %40 = sitofp i32 %39 to double
  br label %59

41:                                               ; preds = %.noexc
  %42 = icmp eq ptr %36, @_ZTIj
  br i1 %42, label %43, label %46

43:                                               ; preds = %41
  %44 = load i32, ptr %35, align 4
  %45 = uitofp i32 %44 to double
  br label %56

46:                                               ; preds = %41
  %47 = icmp eq ptr %36, @_ZTIf
  br i1 %47, label %48, label %51

48:                                               ; preds = %46
  %49 = load float, ptr %35, align 4
  %50 = fpext float %49 to double
  br label %53

51:                                               ; preds = %46
  %52 = load double, ptr %35, align 8
  br label %53

53:                                               ; preds = %48, %51
  %54 = phi double [ %52, %51 ], [ %50, %48 ]
  br label %55

55:                                               ; preds = %53
  br label %56

56:                                               ; preds = %43, %55
  %57 = phi double [ %54, %55 ], [ %45, %43 ]
  br label %58

58:                                               ; preds = %56
  br label %59

59:                                               ; preds = %38, %58
  %60 = phi double [ %57, %58 ], [ %40, %38 ]
  br label %61

61:                                               ; preds = %59
  store double %60, ptr %2, align 8
  %62 = load double, ptr %2, align 8
  %63 = load double, ptr %3, align 8
  %64 = fcmp oeq double %62, %63
  br i1 %64, label %65, label %86

65:                                               ; preds = %61
  %66 = load i32, ptr @glb1, align 4
  %67 = add i32 %66, 10
  store i32 %67, ptr @glb1, align 4
  %68 = load double, ptr %3, align 8
  %69 = fcmp oge double %68, 1.000000e+01
  br i1 %69, label %70, label %73

70:                                               ; preds = %65
  %71 = load double, ptr %2, align 8
  %72 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %70
  store double %71, ptr %72, align 8
  invoke void @__cxa_throw(ptr %72, ptr @_ZTId, ptr null)
          to label %85 unwind label %endcatch.pad

73:                                               ; preds = %65
  call void @__cxa_end_catch()
  br label %81

endcatch.pad:                                     ; preds = %30, %.noexc1
  %74 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

75:                                               ; No predecessors!
  %76 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %75
  %77 = phi { ptr, i32 } [ %76, %75 ], [ %74, %endcatch.pad ]
  %78 = load i32, ptr @x, align 4
  %79 = load i32, ptr @glb1, align 4
  %80 = add i32 %78, %79
  store i32 %80, ptr @x, align 4
  resume { ptr, i32 } %77

81:                                               ; preds = %10, %73
  %82 = load i32, ptr @x, align 4
  %83 = load i32, ptr @glb1, align 4
  %84 = add i32 %82, %83
  store i32 %84, ptr @x, align 4
  ret void

85:                                               ; preds = %.noexc1, %.noexc3, %30
  unreachable

86:                                               ; preds = %61
  %87 = invoke i32 @fflush(ptr null)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %86
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 38, ptr null)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  br label %85

endcatch.resume:                                  ; preds = %.noexc2, %86, %70, %33
  %88 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %88
}

; Function Attrs: mustprogress
define void @exceptions..f_lambda.inner(ptr %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = getelementptr { double }, ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8
  invoke void @exceptions.throwVal(double %4)
          to label %5 unwind label %8

5:                                                ; preds = %1
  %6 = load i32, ptr @glb1, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr @glb1, align 4
  br label %74

8:                                                ; preds = %1
  %9 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 1
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %12 = icmp eq i32 %10, %11
  %13 = extractvalue { ptr, i32 } %9, 1
  %14 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %15 = icmp eq i32 %13, %14
  %16 = extractvalue { ptr, i32 } %9, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %18 = icmp eq i32 %16, %17
  %19 = extractvalue { ptr, i32 } %9, 1
  %20 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %21 = icmp eq i32 %19, %20
  %22 = or i1 %12, %15
  %23 = or i1 %22, %18
  %24 = or i1 %23, %21
  br i1 %24, label %28, label %25

25:                                               ; preds = %8
  %26 = extractvalue { ptr, i32 } %9, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  invoke void @__cxa_rethrow()
          to label %78 unwind label %endcatch.pad

28:                                               ; preds = %8
  %29 = extractvalue { ptr, i32 } %9, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  %31 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %28
  %32 = icmp eq ptr %31, @_ZTIi
  br i1 %32, label %33, label %36

33:                                               ; preds = %.noexc1
  %34 = load i32, ptr %30, align 4
  %35 = sitofp i32 %34 to double
  br label %54

36:                                               ; preds = %.noexc1
  %37 = icmp eq ptr %31, @_ZTIj
  br i1 %37, label %38, label %41

38:                                               ; preds = %36
  %39 = load i32, ptr %30, align 4
  %40 = uitofp i32 %39 to double
  br label %51

41:                                               ; preds = %36
  %42 = icmp eq ptr %31, @_ZTIf
  br i1 %42, label %43, label %46

43:                                               ; preds = %41
  %44 = load float, ptr %30, align 4
  %45 = fpext float %44 to double
  br label %48

46:                                               ; preds = %41
  %47 = load double, ptr %30, align 8
  br label %48

48:                                               ; preds = %43, %46
  %49 = phi double [ %47, %46 ], [ %45, %43 ]
  br label %50

50:                                               ; preds = %48
  br label %51

51:                                               ; preds = %38, %50
  %52 = phi double [ %49, %50 ], [ %40, %38 ]
  br label %53

53:                                               ; preds = %51
  br label %54

54:                                               ; preds = %33, %53
  %55 = phi double [ %52, %53 ], [ %35, %33 ]
  br label %56

56:                                               ; preds = %54
  store double %55, ptr %2, align 8
  %57 = load double, ptr %2, align 8
  %58 = fcmp oeq double %57, %4
  br i1 %58, label %59, label %79

59:                                               ; preds = %56
  %60 = load i32, ptr @glb1, align 4
  %61 = add i32 %60, 10
  store i32 %61, ptr @glb1, align 4
  %62 = fcmp oge double %4, 1.000000e+01
  br i1 %62, label %63, label %66

63:                                               ; preds = %59
  %64 = load double, ptr %2, align 8
  %65 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %63
  store double %64, ptr %65, align 8
  invoke void @__cxa_throw(ptr %65, ptr @_ZTId, ptr null)
          to label %78 unwind label %endcatch.pad

66:                                               ; preds = %59
  call void @__cxa_end_catch()
  br label %74

endcatch.pad:                                     ; preds = %25, %.noexc
  %67 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

68:                                               ; No predecessors!
  %69 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %68
  %70 = phi { ptr, i32 } [ %69, %68 ], [ %67, %endcatch.pad ]
  %71 = load i32, ptr @x, align 4
  %72 = load i32, ptr @glb1, align 4
  %73 = add i32 %71, %72
  store i32 %73, ptr @x, align 4
  resume { ptr, i32 } %70

74:                                               ; preds = %5, %66
  %75 = load i32, ptr @x, align 4
  %76 = load i32, ptr @glb1, align 4
  %77 = add i32 %75, %76
  store i32 %77, ptr @x, align 4
  ret void

78:                                               ; preds = %.noexc, %.noexc3, %25
  unreachable

79:                                               ; preds = %56
  %80 = invoke i32 @fflush(ptr null)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %79
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 53, ptr null)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  br label %78

endcatch.resume:                                  ; preds = %.noexc2, %79, %28, %63
  %81 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %81
}

; Function Attrs: mustprogress
define void @exceptions.lambda(double %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca <{ i64, { double } }>, align 16
  %3 = alloca double, align 8
  store double %0, ptr %3, align 8
  store i64 -1, ptr %2, align 8
  %4 = getelementptr i8, ptr %2, i64 8
  store { double } zeroinitializer, ptr %4, align 8
  %5 = getelementptr { double }, ptr %4, i32 0, i32 0
  %6 = load double, ptr %3, align 8
  store double %6, ptr %5, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @exceptions..f_lambda.inner, ptr null, ptr null }, ptr %4, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_6046455916167272971_object, i64 40), 2
  call void @exceptions..f_lambda.inner(ptr %4)
  ret void
}

; Function Attrs: mustprogress
define void @exceptions.callingThrowVal(double %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %3, align 8
  invoke void @pause(double 1.000000e+00)
          to label %4 unwind label %10

4:                                                ; preds = %1
  %5 = load double, ptr %3, align 8
  invoke void @exceptions.throwVal(double %5)
          to label %6 unwind label %10

6:                                                ; preds = %4
  invoke void @pause(double 1.000000e+00)
          to label %7 unwind label %10

7:                                                ; preds = %6
  %8 = load i32, ptr @glb1, align 4
  %9 = add i32 %8, 1
  store i32 %9, ptr @glb1, align 4
  br label %78

10:                                               ; preds = %6, %4, %1
  %11 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %14 = icmp eq i32 %12, %13
  %15 = extractvalue { ptr, i32 } %11, 1
  %16 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %17 = icmp eq i32 %15, %16
  %18 = extractvalue { ptr, i32 } %11, 1
  %19 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %20 = icmp eq i32 %18, %19
  %21 = extractvalue { ptr, i32 } %11, 1
  %22 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %23 = icmp eq i32 %21, %22
  %24 = or i1 %14, %17
  %25 = or i1 %24, %20
  %26 = or i1 %25, %23
  br i1 %26, label %30, label %27

27:                                               ; preds = %10
  %28 = extractvalue { ptr, i32 } %11, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  invoke void @__cxa_rethrow()
          to label %82 unwind label %endcatch.pad

30:                                               ; preds = %10
  %31 = extractvalue { ptr, i32 } %11, 0
  %32 = call ptr @__cxa_begin_catch(ptr %31)
  %33 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %30
  %34 = icmp eq ptr %33, @_ZTIi
  br i1 %34, label %35, label %38

35:                                               ; preds = %.noexc1
  %36 = load i32, ptr %32, align 4
  %37 = sitofp i32 %36 to double
  br label %56

38:                                               ; preds = %.noexc1
  %39 = icmp eq ptr %33, @_ZTIj
  br i1 %39, label %40, label %43

40:                                               ; preds = %38
  %41 = load i32, ptr %32, align 4
  %42 = uitofp i32 %41 to double
  br label %53

43:                                               ; preds = %38
  %44 = icmp eq ptr %33, @_ZTIf
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  %46 = load float, ptr %32, align 4
  %47 = fpext float %46 to double
  br label %50

48:                                               ; preds = %43
  %49 = load double, ptr %32, align 8
  br label %50

50:                                               ; preds = %45, %48
  %51 = phi double [ %49, %48 ], [ %47, %45 ]
  br label %52

52:                                               ; preds = %50
  br label %53

53:                                               ; preds = %40, %52
  %54 = phi double [ %51, %52 ], [ %42, %40 ]
  br label %55

55:                                               ; preds = %53
  br label %56

56:                                               ; preds = %35, %55
  %57 = phi double [ %54, %55 ], [ %37, %35 ]
  br label %58

58:                                               ; preds = %56
  store double %57, ptr %2, align 8
  %59 = load double, ptr %2, align 8
  %60 = load double, ptr %3, align 8
  %61 = fcmp oeq double %59, %60
  br i1 %61, label %62, label %83

62:                                               ; preds = %58
  %63 = load i32, ptr @glb1, align 4
  %64 = add i32 %63, 10
  store i32 %64, ptr @glb1, align 4
  %65 = load double, ptr %3, align 8
  %66 = fcmp oge double %65, 1.000000e+01
  br i1 %66, label %67, label %70

67:                                               ; preds = %62
  %68 = load double, ptr %2, align 8
  %69 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %67
  store double %68, ptr %69, align 8
  invoke void @__cxa_throw(ptr %69, ptr @_ZTId, ptr null)
          to label %82 unwind label %endcatch.pad

70:                                               ; preds = %62
  call void @__cxa_end_catch()
  br label %78

endcatch.pad:                                     ; preds = %27, %.noexc
  %71 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

72:                                               ; No predecessors!
  %73 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %72
  %74 = phi { ptr, i32 } [ %73, %72 ], [ %71, %endcatch.pad ]
  %75 = load i32, ptr @x, align 4
  %76 = load i32, ptr @glb1, align 4
  %77 = add i32 %75, %76
  store i32 %77, ptr @x, align 4
  resume { ptr, i32 } %74

78:                                               ; preds = %7, %70
  %79 = load i32, ptr @x, align 4
  %80 = load i32, ptr @glb1, align 4
  %81 = add i32 %79, %80
  store i32 %81, ptr @x, align 4
  ret void

82:                                               ; preds = %.noexc, %.noexc3, %27
  unreachable

83:                                               ; preds = %58
  %84 = invoke i32 @fflush(ptr null)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %83
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 71, ptr null)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  br label %82

endcatch.resume:                                  ; preds = %.noexc2, %83, %30, %67
  %85 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %85
}

; Function Attrs: mustprogress
define void @exceptions.nested() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  invoke void @exceptions.callingThrowVal(double 1.000000e+01)
          to label %3 unwind label %4

3:                                                ; preds = %0
  br label %119

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %8 = icmp eq i32 %6, %7
  %9 = extractvalue { ptr, i32 } %5, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %11 = icmp eq i32 %9, %10
  %12 = extractvalue { ptr, i32 } %5, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %14 = icmp eq i32 %12, %13
  %15 = extractvalue { ptr, i32 } %5, 1
  %16 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %17 = icmp eq i32 %15, %16
  %18 = or i1 %8, %11
  %19 = or i1 %18, %14
  %20 = or i1 %19, %17
  br i1 %20, label %24, label %21

21:                                               ; preds = %4
  %22 = extractvalue { ptr, i32 } %5, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  invoke void @__cxa_rethrow()
          to label %120 unwind label %endcatch.pad

24:                                               ; preds = %4
  %25 = extractvalue { ptr, i32 } %5, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %24
  %28 = icmp eq ptr %27, @_ZTIi
  br i1 %28, label %29, label %32

29:                                               ; preds = %.noexc
  %30 = load i32, ptr %26, align 4
  %31 = sitofp i32 %30 to double
  br label %50

32:                                               ; preds = %.noexc
  %33 = icmp eq ptr %27, @_ZTIj
  br i1 %33, label %34, label %37

34:                                               ; preds = %32
  %35 = load i32, ptr %26, align 4
  %36 = uitofp i32 %35 to double
  br label %47

37:                                               ; preds = %32
  %38 = icmp eq ptr %27, @_ZTIf
  br i1 %38, label %39, label %42

39:                                               ; preds = %37
  %40 = load float, ptr %26, align 4
  %41 = fpext float %40 to double
  br label %44

42:                                               ; preds = %37
  %43 = load double, ptr %26, align 8
  br label %44

44:                                               ; preds = %39, %42
  %45 = phi double [ %43, %42 ], [ %41, %39 ]
  br label %46

46:                                               ; preds = %44
  br label %47

47:                                               ; preds = %34, %46
  %48 = phi double [ %45, %46 ], [ %36, %34 ]
  br label %49

49:                                               ; preds = %47
  br label %50

50:                                               ; preds = %29, %49
  %51 = phi double [ %48, %49 ], [ %31, %29 ]
  br label %52

52:                                               ; preds = %50
  store double %51, ptr %1, align 8
  %53 = load i32, ptr @glb1, align 4
  %54 = icmp eq i32 %53, 10
  br i1 %54, label %55, label %58

55:                                               ; preds = %52
  %56 = load i32, ptr @x, align 4
  %57 = icmp eq i32 %56, 10
  br label %58

58:                                               ; preds = %55, %52
  %59 = phi i1 [ %57, %55 ], [ %54, %52 ]
  br i1 %59, label %60, label %121

60:                                               ; preds = %58
  %61 = load i32, ptr @glb1, align 4
  %62 = add i32 %61, 1
  store i32 %62, ptr @glb1, align 4
  %63 = load double, ptr %1, align 8
  call void @__cxa_end_catch()
  %64 = call ptr @__cxa_allocate_exception(i64 8)
  store double %63, ptr %64, align 8
  invoke void @__cxa_throw(ptr %64, ptr @_ZTId, ptr null)
          to label %120 unwind label %66

endcatch.pad:                                     ; preds = %21
  %65 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  call void @__cxa_end_catch()
  br label %.caught

66:                                               ; preds = %60
  %67 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %66
  %68 = phi { ptr, i32 } [ %67, %66 ], [ %65, %endcatch.pad ]
  %69 = extractvalue { ptr, i32 } %68, 1
  %70 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %71 = icmp eq i32 %69, %70
  %72 = extractvalue { ptr, i32 } %68, 1
  %73 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %74 = icmp eq i32 %72, %73
  %75 = extractvalue { ptr, i32 } %68, 1
  %76 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %77 = icmp eq i32 %75, %76
  %78 = extractvalue { ptr, i32 } %68, 1
  %79 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %80 = icmp eq i32 %78, %79
  %81 = or i1 %71, %74
  %82 = or i1 %81, %77
  %83 = or i1 %82, %80
  br i1 %83, label %87, label %84

84:                                               ; preds = %.caught
  %85 = extractvalue { ptr, i32 } %68, 0
  %86 = call ptr @__cxa_begin_catch(ptr %85)
  invoke void @__cxa_rethrow()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %84
  br label %120

87:                                               ; preds = %.caught
  %88 = extractvalue { ptr, i32 } %68, 0
  %89 = call ptr @__cxa_begin_catch(ptr %88)
  %90 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %87
  %91 = icmp eq ptr %90, @_ZTIi
  br i1 %91, label %92, label %95

92:                                               ; preds = %.noexc6
  %93 = load i32, ptr %89, align 4
  %94 = sitofp i32 %93 to double
  br label %113

95:                                               ; preds = %.noexc6
  %96 = icmp eq ptr %90, @_ZTIj
  br i1 %96, label %97, label %100

97:                                               ; preds = %95
  %98 = load i32, ptr %89, align 4
  %99 = uitofp i32 %98 to double
  br label %110

100:                                              ; preds = %95
  %101 = icmp eq ptr %90, @_ZTIf
  br i1 %101, label %102, label %105

102:                                              ; preds = %100
  %103 = load float, ptr %89, align 4
  %104 = fpext float %103 to double
  br label %107

105:                                              ; preds = %100
  %106 = load double, ptr %89, align 8
  br label %107

107:                                              ; preds = %102, %105
  %108 = phi double [ %106, %105 ], [ %104, %102 ]
  br label %109

109:                                              ; preds = %107
  br label %110

110:                                              ; preds = %97, %109
  %111 = phi double [ %108, %109 ], [ %99, %97 ]
  br label %112

112:                                              ; preds = %110
  br label %113

113:                                              ; preds = %92, %112
  %114 = phi double [ %111, %112 ], [ %94, %92 ]
  br label %115

115:                                              ; preds = %113
  store double %114, ptr %2, align 8
  %116 = load i32, ptr @glb1, align 4
  %117 = icmp eq i32 %116, 11
  br i1 %117, label %118, label %123

118:                                              ; preds = %115
  call void @__cxa_end_catch()
  br label %119

119:                                              ; preds = %3, %118
  ret void

120:                                              ; preds = %.noexc4, %.noexc5, %60, %.noexc2, %21
  unreachable

121:                                              ; preds = %58
  %122 = invoke i32 @fflush(ptr null)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %121
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 85, ptr null)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  br label %120

123:                                              ; preds = %115
  %124 = invoke i32 @fflush(ptr null)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %123
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 90, ptr null)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  br label %120

endcatch.resume:                                  ; preds = %87, %84, %.noexc3, %123, %.noexc1, %121, %24
  %125 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %125
}

; Function Attrs: mustprogress
define void @exceptions.test3({ ptr, ptr, ptr } %0) #3 {
  %2 = alloca { ptr, ptr, ptr }, align 8
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  store i32 0, ptr @x, align 4
  store i32 0, ptr @glb1, align 4
  %3 = load { ptr, ptr, ptr }, ptr %2, align 8
  %4 = extractvalue { ptr, ptr, ptr } %3, 1
  %5 = ptrtoint ptr %4 to i64
  %6 = icmp ne i64 %5, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %1
  %8 = extractvalue { ptr, ptr, ptr } %3, 0
  call void %8(ptr %4, double 1.000000e+00)
  br label %11

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %3, 0
  call void %10(double 1.000000e+00)
  br label %11

11:                                               ; preds = %7, %9
  br label %12

12:                                               ; preds = %11
  %13 = load i32, ptr @glb1, align 4
  %14 = icmp eq i32 %13, 10
  br i1 %14, label %15, label %18

15:                                               ; preds = %12
  %16 = load i32, ptr @x, align 4
  %17 = icmp eq i32 %16, 10
  br label %18

18:                                               ; preds = %15, %12
  %19 = phi i1 [ %17, %15 ], [ %14, %12 ]
  br i1 %19, label %20, label %58

20:                                               ; preds = %18
  %21 = load { ptr, ptr, ptr }, ptr %2, align 8
  %22 = extractvalue { ptr, ptr, ptr } %21, 1
  %23 = ptrtoint ptr %22 to i64
  %24 = icmp ne i64 %23, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %20
  %26 = extractvalue { ptr, ptr, ptr } %21, 0
  call void %26(ptr %22, double 0.000000e+00)
  br label %29

27:                                               ; preds = %20
  %28 = extractvalue { ptr, ptr, ptr } %21, 0
  call void %28(double 0.000000e+00)
  br label %29

29:                                               ; preds = %25, %27
  br label %30

30:                                               ; preds = %29
  %31 = load i32, ptr @glb1, align 4
  %32 = icmp eq i32 %31, 11
  br i1 %32, label %33, label %36

33:                                               ; preds = %30
  %34 = load i32, ptr @x, align 4
  %35 = icmp eq i32 %34, 21
  br label %36

36:                                               ; preds = %33, %30
  %37 = phi i1 [ %35, %33 ], [ %32, %30 ]
  br i1 %37, label %38, label %60

38:                                               ; preds = %36
  %39 = load { ptr, ptr, ptr }, ptr %2, align 8
  %40 = extractvalue { ptr, ptr, ptr } %39, 1
  %41 = ptrtoint ptr %40 to i64
  %42 = icmp ne i64 %41, 0
  br i1 %42, label %43, label %45

43:                                               ; preds = %38
  %44 = extractvalue { ptr, ptr, ptr } %39, 0
  call void %44(ptr %40, double 3.000000e+00)
  br label %47

45:                                               ; preds = %38
  %46 = extractvalue { ptr, ptr, ptr } %39, 0
  call void %46(double 3.000000e+00)
  br label %47

47:                                               ; preds = %43, %45
  br label %48

48:                                               ; preds = %47
  %49 = load i32, ptr @glb1, align 4
  %50 = icmp eq i32 %49, 21
  br i1 %50, label %51, label %54

51:                                               ; preds = %48
  %52 = load i32, ptr @x, align 4
  %53 = icmp eq i32 %52, 42
  br label %54

54:                                               ; preds = %51, %48
  %55 = phi i1 [ %53, %51 ], [ %50, %48 ]
  br i1 %55, label %56, label %62

56:                                               ; preds = %54
  ret void

57:                                               ; preds = %62, %60, %58
  unreachable

58:                                               ; preds = %18
  %59 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 97, ptr null)
  br label %57

60:                                               ; preds = %36
  %61 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 99, ptr null)
  br label %57

62:                                               ; preds = %54
  %63 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 101, ptr null)
  br label %57
}

; Function Attrs: mustprogress
define i32 @exceptions.test4({ ptr, ptr, ptr } %0) #4 personality ptr @__gxx_personality_v0 {
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
define void @exceptions.test5() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 20)
  %6 = getelementptr i8, ptr %5, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @a_3_17496507242642067975, i64 12, i1 false)
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  store ptr %6, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 1
  store i64 3, ptr %10, align 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 2
  store i64 3, ptr %11, align 8
  store ptr %8, ptr %2, align 8
  br label %12

12:                                               ; preds = %34, %0
  %13 = load i32, ptr %3, align 4
  %14 = load ptr, ptr %2, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr @__ts_empty_array, ptr %14
  %17 = getelementptr { ptr, i64, i64 }, ptr %16, i32 0, i32 1
  %18 = load i64, ptr %17, align 8
  %19 = sext i32 %13 to i64
  %20 = icmp slt i64 %19, %18
  br i1 %20, label %21, label %37

21:                                               ; preds = %12
  %22 = load i32, ptr %3, align 4
  %23 = load ptr, ptr %2, align 8
  %24 = icmp eq ptr %23, null
  %25 = select i1 %24, ptr @__ts_empty_array, ptr %23
  %26 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr i32, ptr %27, i32 %22
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %1, align 4
  %30 = load i32, ptr %4, align 4
  %31 = add i32 %30, 1
  store i32 %31, ptr %4, align 4
  %32 = load i32, ptr %1, align 4
  %33 = icmp eq i32 %32, 1
  br i1 %33, label %37, label %34

34:                                               ; preds = %21
  %35 = load i32, ptr %3, align 4
  %36 = add i32 %35, 1
  store i32 %36, ptr %3, align 4
  br label %12

37:                                               ; preds = %21, %12
  %38 = load i32, ptr %4, align 4
  %39 = icmp eq i32 %38, 2
  br i1 %39, label %40, label %42

40:                                               ; preds = %37
  ret void

41:                                               ; preds = %42
  unreachable

42:                                               ; preds = %37
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 128, ptr null)
  br label %41
}

; Function Attrs: mustprogress
define void @exceptions..f_run..afL150C22FH2615810610() #3 {
  ret void
}

; Function Attrs: mustprogress
define void @exceptions..f_run..afL151C22FH2615810610() #3 {
  %1 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_15902901984413996407, i64 8), ptr %1, align 8
  call void @__cxa_throw(ptr %1, ptr @_ZTIPc, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @exceptions.run() #3 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3154112327090001444, i64 8))
  store i32 0, ptr @x, align 4
  store i32 0, ptr @glb1, align 4
  call void @exceptions.callingThrowVal(double 1.000000e+00)
  %1 = load i32, ptr @glb1, align 4
  %2 = icmp eq i32 %1, 10
  br i1 %2, label %3, label %6

3:                                                ; preds = %0
  %4 = load i32, ptr @x, align 4
  %5 = icmp eq i32 %4, 10
  br label %6

6:                                                ; preds = %3, %0
  %7 = phi i1 [ %5, %3 ], [ %2, %0 ]
  br i1 %7, label %8, label %35

8:                                                ; preds = %6
  call void @exceptions.callingThrowVal(double 0.000000e+00)
  %9 = load i32, ptr @glb1, align 4
  %10 = icmp eq i32 %9, 11
  br i1 %10, label %11, label %14

11:                                               ; preds = %8
  %12 = load i32, ptr @x, align 4
  %13 = icmp eq i32 %12, 21
  br label %14

14:                                               ; preds = %11, %8
  %15 = phi i1 [ %13, %11 ], [ %10, %8 ]
  br i1 %15, label %16, label %37

16:                                               ; preds = %14
  call void @exceptions.callingThrowVal(double 3.000000e+00)
  %17 = load i32, ptr @glb1, align 4
  %18 = icmp eq i32 %17, 21
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = load i32, ptr @x, align 4
  %21 = icmp eq i32 %20, 42
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi i1 [ %21, %19 ], [ %18, %16 ]
  br i1 %23, label %24, label %39

24:                                               ; preds = %22
  call void @exceptions.test3({ ptr, ptr, ptr } { ptr @exceptions.callingThrowVal, ptr null, ptr null })
  call void @exceptions.test3({ ptr, ptr, ptr } { ptr @exceptions.immediate, ptr null, ptr null })
  call void @exceptions.test3({ ptr, ptr, ptr } { ptr @exceptions.higherorder, ptr null, ptr null })
  call void @exceptions.test3({ ptr, ptr, ptr } { ptr @exceptions.lambda, ptr null, ptr null })
  store i32 0, ptr @x, align 4
  store i32 0, ptr @glb1, align 4
  call void @exceptions.nested()
  %25 = load i32, ptr @glb1, align 4
  %26 = icmp eq i32 %25, 11
  br i1 %26, label %27, label %41

27:                                               ; preds = %24
  %28 = call i32 @exceptions.test4({ ptr, ptr, ptr } { ptr @exceptions..f_run..afL150C22FH2615810610, ptr null, ptr null })
  %29 = icmp eq i32 %28, 10
  br i1 %29, label %30, label %43

30:                                               ; preds = %27
  %31 = call i32 @exceptions.test4({ ptr, ptr, ptr } { ptr @exceptions..f_run..afL151C22FH2615810610, ptr null, ptr null })
  %32 = icmp eq i32 %31, 20
  br i1 %32, label %33, label %45

33:                                               ; preds = %30
  call void @exceptions.test5()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3511218394696192748, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret void

34:                                               ; preds = %45, %43, %41, %39, %37, %35
  unreachable

35:                                               ; preds = %6
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 135, ptr null)
  br label %34

37:                                               ; preds = %14
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 137, ptr null)
  br label %34

39:                                               ; preds = %22
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 139, ptr null)
  br label %34

41:                                               ; preds = %24
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 148, ptr null)
  br label %34

43:                                               ; preds = %27
  %44 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 150, ptr null)
  br label %34

45:                                               ; preds = %30
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11350717838621562508, i64 8), i32 151, ptr null)
  br label %34
}

; Function Attrs: mustprogress
define i32 @main() #3 {
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

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { nounwind memory(none) }
attributes #3 = { mustprogress }
attributes #4 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

