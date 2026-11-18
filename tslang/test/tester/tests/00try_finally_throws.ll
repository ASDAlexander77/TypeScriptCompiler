; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_620445648566982762 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFab\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_14793735007222968981 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody\00", align 8
@m_13407432819789831890 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa call in the finally replaces the exception\00", align 8
@f_14741384667565128915 = internal constant [110 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_finally_throws.ts\00", align 8
@m_2193947250942521861 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa throw in the finally replaces it\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_9956518632497476131 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa finally that throws on the normal path\00", align 8
@m_16267537631902925577 = internal constant [66 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa finally whose call does not throw rethrows the original\00", align 8
@m_11842483425202219940 = internal constant [61 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan outer finally runs on the new exception's way out\00", align 8
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare i32 @strcmp(ptr, ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: mustprogress
define void @thrower(double %0) #2 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
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
  ret void

9:                                                ; preds = %5
  unreachable
}

; Function Attrs: mustprogress
define i1 @rethrowsOriginal() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca i1, align 1
  %4 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  store double 0.000000e+00, ptr %2, align 8
  invoke void @thrower(double 1.000000e+00)
          to label %5 unwind label %6

5:                                                ; preds = %0
  store i1 true, ptr %3, align 1
  invoke void @thrower(double 0.000000e+00)
          to label %11 unwind label %13

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  store i1 true, ptr %3, align 1
  invoke void @thrower(double 0.000000e+00)
          to label %10 unwind label %endcatch.pad

10:                                               ; preds = %6
  invoke void @__cxa_rethrow()
          to label %72 unwind label %endcatch.pad

11:                                               ; preds = %5
  br label %64

endcatch.pad:                                     ; preds = %10, %6
  %12 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  call void @__cxa_end_catch()
  br label %.caught

13:                                               ; preds = %5
  %14 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %13
  %15 = phi { ptr, i32 } [ %14, %13 ], [ %12, %endcatch.pad ]
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

31:                                               ; preds = %.caught
  %32 = extractvalue { ptr, i32 } %15, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %31
  br label %72

34:                                               ; preds = %.caught
  %35 = extractvalue { ptr, i32 } %15, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  %37 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %34
  %38 = icmp eq ptr %37, @_ZTIi
  br i1 %38, label %39, label %42

39:                                               ; preds = %.noexc1
  %40 = load i32, ptr %36, align 4
  %41 = sitofp i32 %40 to double
  br label %60

42:                                               ; preds = %.noexc1
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
  store double %61, ptr %1, align 8
  %63 = load double, ptr %1, align 8
  store double %63, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %64

64:                                               ; preds = %11, %62
  %65 = load i1, ptr %3, align 1
  br i1 %65, label %66, label %69

66:                                               ; preds = %64
  %67 = load double, ptr %2, align 8
  %68 = fcmp oeq double %67, 1.000000e+00
  br label %69

69:                                               ; preds = %66, %64
  %70 = phi i1 [ %68, %66 ], [ %65, %64 ]
  store i1 %70, ptr %4, align 1
  %71 = load i1, ptr %4, align 1
  ret i1 %71

72:                                               ; preds = %.noexc, %10
  unreachable

endcatch.resume:                                  ; preds = %34, %31
  %73 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %73
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store double 0.000000e+00, ptr %5, align 8
  invoke void @thrower(double 1.000000e+00)
          to label %7 unwind label %8

7:                                                ; preds = %0
  invoke void @thrower(double 2.000000e+00)
          to label %13 unwind label %15

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @thrower(double 2.000000e+00)
          to label %12 unwind label %endcatch.pad4

12:                                               ; preds = %8
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad4

13:                                               ; preds = %7
  br label %66

endcatch.pad4:                                    ; preds = %12, %8
  %14 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

15:                                               ; preds = %7
  %16 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %15
  %17 = phi { ptr, i32 } [ %16, %15 ], [ %14, %endcatch.pad4 ]
  %18 = extractvalue { ptr, i32 } %17, 1
  %19 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %20 = icmp eq i32 %18, %19
  %21 = extractvalue { ptr, i32 } %17, 1
  %22 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %23 = icmp eq i32 %21, %22
  %24 = extractvalue { ptr, i32 } %17, 1
  %25 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %26 = icmp eq i32 %24, %25
  %27 = extractvalue { ptr, i32 } %17, 1
  %28 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %29 = icmp eq i32 %27, %28
  %30 = or i1 %20, %23
  %31 = or i1 %30, %26
  %32 = or i1 %31, %29
  br i1 %32, label %36, label %33

33:                                               ; preds = %.caught3
  %34 = extractvalue { ptr, i32 } %17, 0
  %35 = call ptr @__cxa_begin_catch(ptr %34)
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad

36:                                               ; preds = %.caught3
  %37 = extractvalue { ptr, i32 } %17, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  %39 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %36
  %40 = icmp eq ptr %39, @_ZTIi
  br i1 %40, label %41, label %44

41:                                               ; preds = %.noexc
  %42 = load i32, ptr %38, align 4
  %43 = sitofp i32 %42 to double
  br label %62

44:                                               ; preds = %.noexc
  %45 = icmp eq ptr %39, @_ZTIj
  br i1 %45, label %46, label %49

46:                                               ; preds = %44
  %47 = load i32, ptr %38, align 4
  %48 = uitofp i32 %47 to double
  br label %59

49:                                               ; preds = %44
  %50 = icmp eq ptr %39, @_ZTIf
  br i1 %50, label %51, label %54

51:                                               ; preds = %49
  %52 = load float, ptr %38, align 4
  %53 = fpext float %52 to double
  br label %56

54:                                               ; preds = %49
  %55 = load double, ptr %38, align 8
  br label %56

56:                                               ; preds = %51, %54
  %57 = phi double [ %55, %54 ], [ %53, %51 ]
  br label %58

58:                                               ; preds = %56
  br label %59

59:                                               ; preds = %46, %58
  %60 = phi double [ %57, %58 ], [ %48, %46 ]
  br label %61

61:                                               ; preds = %59
  br label %62

62:                                               ; preds = %41, %61
  %63 = phi double [ %60, %61 ], [ %43, %41 ]
  br label %64

64:                                               ; preds = %62
  store double %63, ptr %4, align 8
  %65 = load double, ptr %4, align 8
  store double %65, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %66

66:                                               ; preds = %13, %64
  %67 = load double, ptr %5, align 8
  %68 = fcmp oeq double %67, 2.000000e+00
  br i1 %68, label %69, label %357

69:                                               ; preds = %66
  store double 0.000000e+00, ptr %5, align 8
  invoke void @thrower(double 1.000000e+00)
          to label %70 unwind label %72

70:                                               ; preds = %69
  %71 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 3, ptr %71, align 4
  invoke void @__cxa_throw(ptr %71, ptr @_ZTIi, ptr null)
          to label %356 unwind label %78

72:                                               ; preds = %69
  %73 = landingpad { ptr, i32 }
          catch ptr null
  %74 = extractvalue { ptr, i32 } %73, 0
  %75 = call ptr @__cxa_begin_catch(ptr %74)
  %76 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %72
  store i32 3, ptr %76, align 4
  invoke void @__cxa_throw(ptr %76, ptr @_ZTIi, ptr null)
          to label %356 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %.noexc9
  %77 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

78:                                               ; preds = %70
  %79 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %78
  %80 = phi { ptr, i32 } [ %79, %78 ], [ %77, %endcatch.pad2 ]
  %81 = extractvalue { ptr, i32 } %80, 1
  %82 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %83 = icmp eq i32 %81, %82
  %84 = extractvalue { ptr, i32 } %80, 1
  %85 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %86 = icmp eq i32 %84, %85
  %87 = extractvalue { ptr, i32 } %80, 1
  %88 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %89 = icmp eq i32 %87, %88
  %90 = extractvalue { ptr, i32 } %80, 1
  %91 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %92 = icmp eq i32 %90, %91
  %93 = or i1 %83, %86
  %94 = or i1 %93, %89
  %95 = or i1 %94, %92
  br i1 %95, label %99, label %96

96:                                               ; preds = %.caught1
  %97 = extractvalue { ptr, i32 } %80, 0
  %98 = call ptr @__cxa_begin_catch(ptr %97)
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad

99:                                               ; preds = %.caught1
  %100 = extractvalue { ptr, i32 } %80, 0
  %101 = call ptr @__cxa_begin_catch(ptr %100)
  %102 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %99
  %103 = icmp eq ptr %102, @_ZTIi
  br i1 %103, label %104, label %107

104:                                              ; preds = %.noexc10
  %105 = load i32, ptr %101, align 4
  %106 = sitofp i32 %105 to double
  br label %125

107:                                              ; preds = %.noexc10
  %108 = icmp eq ptr %102, @_ZTIj
  br i1 %108, label %109, label %112

109:                                              ; preds = %107
  %110 = load i32, ptr %101, align 4
  %111 = uitofp i32 %110 to double
  br label %122

112:                                              ; preds = %107
  %113 = icmp eq ptr %102, @_ZTIf
  br i1 %113, label %114, label %117

114:                                              ; preds = %112
  %115 = load float, ptr %101, align 4
  %116 = fpext float %115 to double
  br label %119

117:                                              ; preds = %112
  %118 = load double, ptr %101, align 8
  br label %119

119:                                              ; preds = %114, %117
  %120 = phi double [ %118, %117 ], [ %116, %114 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %109, %121
  %123 = phi double [ %120, %121 ], [ %111, %109 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %104, %124
  %126 = phi double [ %123, %124 ], [ %106, %104 ]
  br label %127

127:                                              ; preds = %125
  store double %126, ptr %3, align 8
  %128 = load double, ptr %3, align 8
  store double %128, ptr %5, align 8
  call void @__cxa_end_catch()
  %129 = load double, ptr %5, align 8
  %130 = fcmp oeq double %129, 3.000000e+00
  br i1 %130, label %131, label %359

131:                                              ; preds = %127
  store double 0.000000e+00, ptr %5, align 8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14793735007222968981, i64 8))
  invoke void @thrower(double 4.000000e+00)
          to label %132 unwind label %133

132:                                              ; preds = %131
  br label %183

133:                                              ; preds = %131
  %134 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %135 = extractvalue { ptr, i32 } %134, 1
  %136 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %137 = icmp eq i32 %135, %136
  %138 = extractvalue { ptr, i32 } %134, 1
  %139 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %140 = icmp eq i32 %138, %139
  %141 = extractvalue { ptr, i32 } %134, 1
  %142 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %143 = icmp eq i32 %141, %142
  %144 = extractvalue { ptr, i32 } %134, 1
  %145 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %146 = icmp eq i32 %144, %145
  %147 = or i1 %137, %140
  %148 = or i1 %147, %143
  %149 = or i1 %148, %146
  br i1 %149, label %153, label %150

150:                                              ; preds = %133
  %151 = extractvalue { ptr, i32 } %134, 0
  %152 = call ptr @__cxa_begin_catch(ptr %151)
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad

153:                                              ; preds = %133
  %154 = extractvalue { ptr, i32 } %134, 0
  %155 = call ptr @__cxa_begin_catch(ptr %154)
  %156 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %153
  %157 = icmp eq ptr %156, @_ZTIi
  br i1 %157, label %158, label %161

158:                                              ; preds = %.noexc11
  %159 = load i32, ptr %155, align 4
  %160 = sitofp i32 %159 to double
  br label %179

161:                                              ; preds = %.noexc11
  %162 = icmp eq ptr %156, @_ZTIj
  br i1 %162, label %163, label %166

163:                                              ; preds = %161
  %164 = load i32, ptr %155, align 4
  %165 = uitofp i32 %164 to double
  br label %176

166:                                              ; preds = %161
  %167 = icmp eq ptr %156, @_ZTIf
  br i1 %167, label %168, label %171

168:                                              ; preds = %166
  %169 = load float, ptr %155, align 4
  %170 = fpext float %169 to double
  br label %173

171:                                              ; preds = %166
  %172 = load double, ptr %155, align 8
  br label %173

173:                                              ; preds = %168, %171
  %174 = phi double [ %172, %171 ], [ %170, %168 ]
  br label %175

175:                                              ; preds = %173
  br label %176

176:                                              ; preds = %163, %175
  %177 = phi double [ %174, %175 ], [ %165, %163 ]
  br label %178

178:                                              ; preds = %176
  br label %179

179:                                              ; preds = %158, %178
  %180 = phi double [ %177, %178 ], [ %160, %158 ]
  br label %181

181:                                              ; preds = %179
  store double %180, ptr %2, align 8
  %182 = load double, ptr %2, align 8
  store double %182, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %183

183:                                              ; preds = %132, %181
  %184 = load double, ptr %5, align 8
  %185 = fcmp oeq double %184, 4.000000e+00
  br i1 %185, label %186, label %361

186:                                              ; preds = %183
  %187 = invoke i1 @rethrowsOriginal()
          to label %188 unwind label %353

188:                                              ; preds = %186
  br i1 %187, label %189, label %363

189:                                              ; preds = %188
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  store double 0.000000e+00, ptr %5, align 8
  invoke void @thrower(double 1.000000e+00)
          to label %190 unwind label %210

190:                                              ; preds = %189
  %191 = load ptr, ptr %6, align 8
  %192 = icmp eq ptr %191, null
  %193 = select i1 %192, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %191
  %194 = call i64 @strlen(ptr %193)
  %195 = add i64 1, %194
  %196 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %197 = add i64 %195, %196
  %198 = add i64 %197, 8
  %199 = call ptr @GC_malloc(i64 %198)
  %200 = getelementptr i8, ptr %199, i64 8
  %201 = getelementptr i8, ptr %200, i64 0
  %202 = sub i64 %197, 0
  %203 = call ptr @__memcpy_chk(ptr %201, ptr %193, i64 %194, i64 %202)
  %204 = add i64 0, %194
  %205 = getelementptr i8, ptr %200, i64 %204
  %206 = sub i64 %197, %204
  %207 = call ptr @__memcpy_chk(ptr %205, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i64 %196, i64 %206)
  %208 = add i64 %204, %196
  %209 = getelementptr i8, ptr %200, i64 %208
  store i8 0, ptr %209, align 1
  store ptr %200, ptr %6, align 8
  invoke void @thrower(double 5.000000e+00)
          to label %234 unwind label %255

210:                                              ; preds = %189
  %211 = landingpad { ptr, i32 }
          catch ptr null
  %212 = extractvalue { ptr, i32 } %211, 0
  %213 = call ptr @__cxa_begin_catch(ptr %212)
  %214 = load ptr, ptr %6, align 8
  %215 = icmp eq ptr %214, null
  %216 = select i1 %215, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %214
  %217 = invoke i64 @strlen(ptr %216)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %210
  %218 = add i64 1, %217
  %219 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %.noexc13
  %220 = add i64 %218, %219
  %221 = add i64 %220, 8
  %222 = invoke ptr @GC_malloc(i64 %221)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %.noexc14
  %223 = getelementptr i8, ptr %222, i64 8
  %224 = getelementptr i8, ptr %223, i64 0
  %225 = sub i64 %220, 0
  %226 = invoke ptr @__memcpy_chk(ptr %224, ptr %216, i64 %217, i64 %225)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %.noexc15
  %227 = add i64 0, %217
  %228 = getelementptr i8, ptr %223, i64 %227
  %229 = sub i64 %220, %227
  %230 = invoke ptr @__memcpy_chk(ptr %228, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i64 %219, i64 %229)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %.noexc16
  %231 = add i64 %227, %219
  %232 = getelementptr i8, ptr %223, i64 %231
  store i8 0, ptr %232, align 1
  store ptr %223, ptr %6, align 8
  invoke void @thrower(double 5.000000e+00)
          to label %233 unwind label %endcatch.pad6

233:                                              ; preds = %.noexc17
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad6

234:                                              ; preds = %190
  %235 = load ptr, ptr %6, align 8
  %236 = icmp eq ptr %235, null
  %237 = select i1 %236, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %235
  %238 = call i64 @strlen(ptr %237)
  %239 = add i64 1, %238
  %240 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
  %241 = add i64 %239, %240
  %242 = add i64 %241, 8
  %243 = call ptr @GC_malloc(i64 %242)
  %244 = getelementptr i8, ptr %243, i64 8
  %245 = getelementptr i8, ptr %244, i64 0
  %246 = sub i64 %241, 0
  %247 = call ptr @__memcpy_chk(ptr %245, ptr %237, i64 %238, i64 %246)
  %248 = add i64 0, %238
  %249 = getelementptr i8, ptr %244, i64 %248
  %250 = sub i64 %241, %248
  %251 = call ptr @__memcpy_chk(ptr %249, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8), i64 %240, i64 %250)
  %252 = add i64 %248, %240
  %253 = getelementptr i8, ptr %244, i64 %252
  store i8 0, ptr %253, align 1
  store ptr %244, ptr %6, align 8
  br label %331

endcatch.pad6:                                    ; preds = %233, %.noexc17
  %254 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught5

255:                                              ; preds = %190
  %256 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught5

.caught5:                                         ; preds = %endcatch.pad6, %255
  %257 = phi { ptr, i32 } [ %256, %255 ], [ %254, %endcatch.pad6 ]
  %258 = extractvalue { ptr, i32 } %257, 0
  %259 = call ptr @__cxa_begin_catch(ptr %258)
  %260 = load ptr, ptr %6, align 8
  %261 = icmp eq ptr %260, null
  %262 = select i1 %261, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %260
  %263 = invoke i64 @strlen(ptr %262)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %.caught5
  %264 = add i64 1, %263
  %265 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %.noexc18
  %266 = add i64 %264, %265
  %267 = add i64 %266, 8
  %268 = invoke ptr @GC_malloc(i64 %267)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %.noexc19
  %269 = getelementptr i8, ptr %268, i64 8
  %270 = getelementptr i8, ptr %269, i64 0
  %271 = sub i64 %266, 0
  %272 = invoke ptr @__memcpy_chk(ptr %270, ptr %262, i64 %263, i64 %271)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %.noexc20
  %273 = add i64 0, %263
  %274 = getelementptr i8, ptr %269, i64 %273
  %275 = sub i64 %266, %273
  %276 = invoke ptr @__memcpy_chk(ptr %274, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8), i64 %265, i64 %275)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %.noexc21
  %277 = add i64 %273, %265
  %278 = getelementptr i8, ptr %269, i64 %277
  store i8 0, ptr %278, align 1
  store ptr %269, ptr %6, align 8
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad8

endcatch.pad8:                                    ; preds = %.noexc22
  %279 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught7

280:                                              ; No predecessors!
  %281 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught7

.caught7:                                         ; preds = %endcatch.pad8, %280
  %282 = phi { ptr, i32 } [ %281, %280 ], [ %279, %endcatch.pad8 ]
  %283 = extractvalue { ptr, i32 } %282, 1
  %284 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %285 = icmp eq i32 %283, %284
  %286 = extractvalue { ptr, i32 } %282, 1
  %287 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %288 = icmp eq i32 %286, %287
  %289 = extractvalue { ptr, i32 } %282, 1
  %290 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %291 = icmp eq i32 %289, %290
  %292 = extractvalue { ptr, i32 } %282, 1
  %293 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %294 = icmp eq i32 %292, %293
  %295 = or i1 %285, %288
  %296 = or i1 %295, %291
  %297 = or i1 %296, %294
  br i1 %297, label %301, label %298

298:                                              ; preds = %.caught7
  %299 = extractvalue { ptr, i32 } %282, 0
  %300 = call ptr @__cxa_begin_catch(ptr %299)
  invoke void @__cxa_rethrow()
          to label %356 unwind label %endcatch.pad

301:                                              ; preds = %.caught7
  %302 = extractvalue { ptr, i32 } %282, 0
  %303 = call ptr @__cxa_begin_catch(ptr %302)
  %304 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %301
  %305 = icmp eq ptr %304, @_ZTIi
  br i1 %305, label %306, label %309

306:                                              ; preds = %.noexc12
  %307 = load i32, ptr %303, align 4
  %308 = sitofp i32 %307 to double
  br label %327

309:                                              ; preds = %.noexc12
  %310 = icmp eq ptr %304, @_ZTIj
  br i1 %310, label %311, label %314

311:                                              ; preds = %309
  %312 = load i32, ptr %303, align 4
  %313 = uitofp i32 %312 to double
  br label %324

314:                                              ; preds = %309
  %315 = icmp eq ptr %304, @_ZTIf
  br i1 %315, label %316, label %319

316:                                              ; preds = %314
  %317 = load float, ptr %303, align 4
  %318 = fpext float %317 to double
  br label %321

319:                                              ; preds = %314
  %320 = load double, ptr %303, align 8
  br label %321

321:                                              ; preds = %316, %319
  %322 = phi double [ %320, %319 ], [ %318, %316 ]
  br label %323

323:                                              ; preds = %321
  br label %324

324:                                              ; preds = %311, %323
  %325 = phi double [ %322, %323 ], [ %313, %311 ]
  br label %326

326:                                              ; preds = %324
  br label %327

327:                                              ; preds = %306, %326
  %328 = phi double [ %325, %326 ], [ %308, %306 ]
  br label %329

329:                                              ; preds = %327
  store double %328, ptr %1, align 8
  %330 = load double, ptr %1, align 8
  store double %330, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %331

331:                                              ; preds = %234, %329
  %332 = load double, ptr %5, align 8
  %333 = fcmp oeq double %332, 5.000000e+00
  br i1 %333, label %334, label %349

334:                                              ; preds = %331
  %335 = load ptr, ptr %6, align 8
  %336 = ptrtoint ptr %335 to i64
  %337 = icmp ne i64 %336, 0
  %338 = zext i1 %337 to i32
  %339 = and i32 %338, 1
  %340 = icmp ne i32 %339, 0
  br i1 %340, label %341, label %344

341:                                              ; preds = %334
  %342 = call i32 @strcmp(ptr %335, ptr getelementptr inbounds nuw (i8, ptr @s_620445648566982762, i64 8))
  %343 = icmp eq i32 %342, 0
  br label %346

344:                                              ; preds = %334
  %345 = icmp eq i64 %336, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_620445648566982762, i64 8) to i64)
  br label %346

346:                                              ; preds = %341, %344
  %347 = phi i1 [ %345, %344 ], [ %343, %341 ]
  br label %348

348:                                              ; preds = %346
  br label %349

349:                                              ; preds = %348, %331
  %350 = phi i1 [ %347, %348 ], [ %333, %331 ]
  br i1 %350, label %351, label %365

351:                                              ; preds = %349
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

endcatch.pad:                                     ; preds = %150, %33, %298, %96
  %352 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

353:                                              ; preds = %186
  %354 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %353
  %355 = phi { ptr, i32 } [ %354, %353 ], [ %352, %endcatch.pad ]
  resume { ptr, i32 } %355

356:                                              ; preds = %365, %298, %.noexc22, %233, %363, %361, %150, %359, %96, %70, %.noexc9, %357, %33, %12
  unreachable

357:                                              ; preds = %66
  %358 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13407432819789831890, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14741384667565128915, i64 8), i32 31, ptr null)
  br label %356

359:                                              ; preds = %127
  %360 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2193947250942521861, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14741384667565128915, i64 8), i32 39, ptr null)
  br label %356

361:                                              ; preds = %183
  %362 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9956518632497476131, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14741384667565128915, i64 8), i32 47, ptr null)
  br label %356

363:                                              ; preds = %188
  %364 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16267537631902925577, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14741384667565128915, i64 8), i32 49, ptr null)
  br label %356

365:                                              ; preds = %349
  %366 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11842483425202219940, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_14741384667565128915, i64 8), i32 62, ptr null)
  br label %356

endcatch.resume:                                  ; preds = %.noexc21, %.noexc20, %.noexc19, %.noexc18, %.caught5, %.noexc16, %.noexc15, %.noexc14, %.noexc13, %210, %301, %153, %99, %72, %36
  %367 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %367
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
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

