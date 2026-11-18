; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@s_6272037745333642222 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_6207888721374582755 = internal constant [103 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_finally.ts\00", align 8
@s_14894915187058291560 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFinally\00", align 8
@s_849505297048112493 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcont\00", align 8
@s_7859873275976691790 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFTry\00", align 8
@s_3019566071980655109 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught\00", align 8
@s_7962741442650003760 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFexcept\00", align 8
@s_10387589392458792520 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFasd3\00", align 8
@s_10387591591482048942 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFasd1\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@glb1 = private global i32 0
@x = private global i32 0
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIPc = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare ptr @__cxa_current_exception_type()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define void @main1() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr %1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6272037745333642222, i64 8))
  %4 = load i32, ptr %1, align 4
  %5 = add i32 %4, 1
  store i32 %5, ptr %1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
  %6 = load i32, ptr %1, align 4
  %7 = icmp eq i32 %6, 2
  br i1 %7, label %8, label %10

8:                                                ; preds = %0
  ret void

9:                                                ; preds = %10
  unreachable

10:                                               ; preds = %0
  %11 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6207888721374582755, i64 8), i32 16, ptr null)
  br label %9
}

; Function Attrs: mustprogress
define void @try2() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  store i32 10, ptr %1, align 4
  %2 = load i32, ptr @glb1, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @glb1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_7859873275976691790, i64 8))
  %4 = load i32, ptr %1, align 4
  %5 = icmp sgt i32 %4, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %0
  %7 = call ptr @__cxa_allocate_exception(i64 8)
  store double 1.000000e+00, ptr %7, align 8
  invoke void @__cxa_throw(ptr %7, ptr @_ZTId, ptr null)
          to label %15 unwind label %11

8:                                                ; preds = %0
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_849505297048112493, i64 8))
  %9 = load i32, ptr @glb1, align 4
  %10 = add i32 %9, 1
  store i32 %10, ptr @glb1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14894915187058291560, i64 8))
  ret void

11:                                               ; preds = %6
  %12 = landingpad { ptr, i32 }
          cleanup
  %13 = load i32, ptr @glb1, align 4
  %14 = add i32 %13, 1
  store i32 %14, ptr @glb1, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14894915187058291560, i64 8))
  resume { ptr, i32 } %12

15:                                               ; preds = %6
  unreachable
}

; Function Attrs: mustprogress
define void @main2() #1 personality ptr @__gxx_personality_v0 {
  store i32 0, ptr @glb1, align 4
  invoke void @try2()
          to label %1 unwind label %2

1:                                                ; preds = %0
  br label %8

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          catch ptr null
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = call ptr @__cxa_begin_catch(ptr %4)
  %6 = load i32, ptr @glb1, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr @glb1, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %2
  call void @__cxa_end_catch()
  br label %8

8:                                                ; preds = %1, %.noexc
  %9 = load i32, ptr @glb1, align 4
  %10 = icmp eq i32 %9, 3
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  ret void

12:                                               ; preds = %13
  unreachable

13:                                               ; preds = %8
  %14 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6207888721374582755, i64 8), i32 45, ptr null)
  br label %12

endcatch.resume:                                  ; preds = %2
  %15 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %15
}

; Function Attrs: mustprogress
define void @main3() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = add i32 %3, 1
  store i32 %4, ptr %2, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6272037745333642222, i64 8))
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_7962741442650003760, i64 8), ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIPc, ptr null)
          to label %30 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = load i32, ptr %2, align 4
  %11 = add i32 %10, 1
  store i32 %11, ptr %2, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %6
  invoke void @__cxa_rethrow()
          to label %30 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %.noexc1
  %12 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  call void @__cxa_end_catch()
  br label %.caught

13:                                               ; No predecessors!
  %14 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %13
  %15 = phi { ptr, i32 } [ %14, %13 ], [ %12, %endcatch.pad ]
  %16 = extractvalue { ptr, i32 } %15, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %18 = icmp eq i32 %16, %17
  br i1 %18, label %22, label %19

19:                                               ; preds = %.caught
  %20 = extractvalue { ptr, i32 } %15, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %19
  br label %30

22:                                               ; preds = %.caught
  %23 = extractvalue { ptr, i32 } %15, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  store ptr %24, ptr %1, align 8
  %25 = load i32, ptr %2, align 4
  %26 = add i32 %25, 1
  store i32 %26, ptr %2, align 4
  call void @__cxa_end_catch()
  %27 = load i32, ptr %2, align 4
  %28 = icmp eq i32 %27, 3
  br i1 %28, label %29, label %31

29:                                               ; preds = %22
  ret void

30:                                               ; preds = %31, %.noexc, %.noexc1, %0
  unreachable

31:                                               ; preds = %22
  %32 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6207888721374582755, i64 8), i32 69, ptr null)
  br label %30

endcatch.resume:                                  ; preds = %6, %19
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define void @main4() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store double 1.000000e+00, ptr %4, align 8
  invoke void @__cxa_throw(ptr %4, ptr @_ZTId, ptr null)
          to label %119 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
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
          to label %119 unwind label %endcatch.pad2

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
  %54 = load i32, ptr %3, align 4
  %55 = add i32 %54, 1
  store i32 %55, ptr %3, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10387591591482048942, i64 8))
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %53
  %56 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  store double 2.000000e+00, ptr %56, align 8
  invoke void @__cxa_throw(ptr %56, ptr @_ZTId, ptr null)
          to label %119 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %22, %.noexc5
  %57 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

58:                                               ; No predecessors!
  %59 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %58
  %60 = phi { ptr, i32 } [ %59, %58 ], [ %57, %endcatch.pad2 ]
  %61 = extractvalue { ptr, i32 } %60, 0
  %62 = call ptr @__cxa_begin_catch(ptr %61)
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_18175057562723240456, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.caught1
  invoke void @__cxa_rethrow()
          to label %119 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %.noexc3
  %63 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  call void @__cxa_end_catch()
  br label %.caught

64:                                               ; No predecessors!
  %65 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %64
  %66 = phi { ptr, i32 } [ %65, %64 ], [ %63, %endcatch.pad ]
  %67 = extractvalue { ptr, i32 } %66, 1
  %68 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %69 = icmp eq i32 %67, %68
  %70 = extractvalue { ptr, i32 } %66, 1
  %71 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %72 = icmp eq i32 %70, %71
  %73 = extractvalue { ptr, i32 } %66, 1
  %74 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %75 = icmp eq i32 %73, %74
  %76 = extractvalue { ptr, i32 } %66, 1
  %77 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %78 = icmp eq i32 %76, %77
  %79 = or i1 %69, %72
  %80 = or i1 %79, %75
  %81 = or i1 %80, %78
  br i1 %81, label %85, label %82

82:                                               ; preds = %.caught
  %83 = extractvalue { ptr, i32 } %66, 0
  %84 = call ptr @__cxa_begin_catch(ptr %83)
  invoke void @__cxa_rethrow()
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %82
  br label %119

85:                                               ; preds = %.caught
  %86 = extractvalue { ptr, i32 } %66, 0
  %87 = call ptr @__cxa_begin_catch(ptr %86)
  %88 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %85
  %89 = icmp eq ptr %88, @_ZTIi
  br i1 %89, label %90, label %93

90:                                               ; preds = %.noexc8
  %91 = load i32, ptr %87, align 4
  %92 = sitofp i32 %91 to double
  br label %111

93:                                               ; preds = %.noexc8
  %94 = icmp eq ptr %88, @_ZTIj
  br i1 %94, label %95, label %98

95:                                               ; preds = %93
  %96 = load i32, ptr %87, align 4
  %97 = uitofp i32 %96 to double
  br label %108

98:                                               ; preds = %93
  %99 = icmp eq ptr %88, @_ZTIf
  br i1 %99, label %100, label %103

100:                                              ; preds = %98
  %101 = load float, ptr %87, align 4
  %102 = fpext float %101 to double
  br label %105

103:                                              ; preds = %98
  %104 = load double, ptr %87, align 8
  br label %105

105:                                              ; preds = %100, %103
  %106 = phi double [ %104, %103 ], [ %102, %100 ]
  br label %107

107:                                              ; preds = %105
  br label %108

108:                                              ; preds = %95, %107
  %109 = phi double [ %106, %107 ], [ %97, %95 ]
  br label %110

110:                                              ; preds = %108
  br label %111

111:                                              ; preds = %90, %110
  %112 = phi double [ %109, %110 ], [ %92, %90 ]
  br label %113

113:                                              ; preds = %111
  store double %112, ptr %2, align 8
  %114 = load i32, ptr %3, align 4
  %115 = add i32 %114, 1
  store i32 %115, ptr %3, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10387589392458792520, i64 8))
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %113
  call void @__cxa_end_catch()
  %116 = load i32, ptr %3, align 4
  %117 = icmp eq i32 %116, 2
  br i1 %117, label %118, label %120

118:                                              ; preds = %.noexc7
  ret void

119:                                              ; preds = %120, %.noexc6, %.noexc3, %.noexc5, %22, %0
  unreachable

120:                                              ; preds = %.noexc7
  %121 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6207888721374582755, i64 8), i32 94, ptr null)
  br label %119

endcatch.resume:                                  ; preds = %85, %113, %82, %.noexc4, %53, %.caught1, %25
  %122 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %122
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @main1()
  call void @main2()
  call void @main3()
  call void @main4()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
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

attributes #0 = { nounwind memory(none) }
attributes #1 = { mustprogress "personality" }
attributes #2 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

