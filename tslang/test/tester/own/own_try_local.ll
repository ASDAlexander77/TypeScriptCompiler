; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_12453060767899089178 = internal constant [101 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_try_local.ts\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

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

define internal void @__tslang_retain_block_slot(ptr %0) {
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

define internal void @tsrel_1752011451674293319(ptr %0) {
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
declare noalias ptr @GC_malloc(i64) #0

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: mustprogress
define i64 @f(double %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %4 = alloca <{ i64, [2 x double] }>, align 16
  %5 = alloca double, align 8
  %6 = alloca i64, align 8
  store double %0, ptr %5, align 8
  %7 = load double, ptr %5, align 8
  %8 = load double, ptr %5, align 8
  store i64 -1, ptr %4, align 8
  %9 = getelementptr i8, ptr %4, i64 8
  store [2 x double] zeroinitializer, ptr %9, align 8
  %10 = getelementptr double, ptr %9, i64 0
  store double %7, ptr %10, align 8
  %11 = getelementptr double, ptr %9, i64 1
  store double %8, ptr %11, align 8
  store i64 -1, ptr %3, align 8
  %12 = getelementptr i8, ptr %3, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 0
  store ptr %9, ptr %13, align 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 1
  store i64 2, ptr %14, align 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 2
  store i64 2, ptr %15, align 8
  store ptr %12, ptr %2, align 8
  %16 = load double, ptr %5, align 8
  %17 = frem double %16, 2.000000e+00
  %18 = fcmp oeq double %17, 0.000000e+00
  br i1 %18, label %19, label %22

19:                                               ; preds = %1
  %20 = load double, ptr %5, align 8
  %21 = call ptr @__cxa_allocate_exception(i64 8)
  store double %20, ptr %21, align 8
  call void @__cxa_throw(ptr %21, ptr @_ZTId, ptr null)
  br label %29

22:                                               ; preds = %1
  %23 = load ptr, ptr %2, align 8
  %24 = icmp eq ptr %23, null
  %25 = select i1 %24, ptr @__ts_empty_array, ptr %23
  %26 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 1
  %27 = load i64, ptr %26, align 8
  store i64 %27, ptr %6, align 8
  %28 = load i64, ptr %6, align 8
  ret i64 %28

29:                                               ; preds = %19
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  br label %5

5:                                                ; preds = %154, %0
  %6 = load i32, ptr %3, align 4
  %7 = icmp slt i32 %6, 100000
  br i1 %7, label %8, label %157

8:                                                ; preds = %5
  %9 = load i32, ptr %3, align 4
  %10 = sitofp i32 %9 to double
  %11 = invoke i64 @f(double %10)
          to label %12 unwind label %13

12:                                               ; preds = %8
  br label %154

13:                                               ; preds = %8
  %14 = landingpad { ptr, i32 }
          catch ptr null
  %15 = extractvalue { ptr, i32 } %14, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %13
  %18 = icmp eq ptr %17, @_ZTIi
  br i1 %18, label %19, label %26

19:                                               ; preds = %.noexc
  %20 = load i32, ptr %16, align 4
  %21 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %19
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 0
  store i64 4, ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 2
  store i32 %20, ptr %25, align 4
  br label %149

26:                                               ; preds = %.noexc
  %27 = icmp eq ptr %17, @_ZTIj
  br i1 %27, label %28, label %35

28:                                               ; preds = %26
  %29 = load i32, ptr %16, align 4
  %30 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %28
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 0
  store i64 4, ptr %32, align 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %33, align 8
  %34 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 2
  store i32 %29, ptr %34, align 4
  br label %146

35:                                               ; preds = %26
  %36 = icmp eq ptr %17, @_ZTId
  br i1 %36, label %37, label %44

37:                                               ; preds = %35
  %38 = load double, ptr %16, align 8
  %39 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %37
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 0
  store i64 8, ptr %41, align 8
  %42 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %42, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 2
  store double %38, ptr %43, align 8
  br label %143

44:                                               ; preds = %35
  %45 = icmp eq ptr %17, @_ZTIf
  br i1 %45, label %46, label %53

46:                                               ; preds = %44
  %47 = load float, ptr %16, align 4
  %48 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %46
  %49 = getelementptr i8, ptr %48, i64 8
  %50 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 0
  store i64 4, ptr %50, align 8
  %51 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %51, align 8
  %52 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 2
  store float %47, ptr %52, align 4
  br label %140

53:                                               ; preds = %44
  %54 = icmp eq ptr %17, @_ZTIb
  br i1 %54, label %55, label %62

55:                                               ; preds = %53
  %56 = load i1, ptr %16, align 1
  %57 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %55
  %58 = getelementptr i8, ptr %57, i64 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 0
  store i64 1, ptr %59, align 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %60, align 8
  %61 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 2
  store i1 %56, ptr %61, align 1
  br label %137

62:                                               ; preds = %53
  %63 = icmp eq ptr %17, @_ZTIx
  br i1 %63, label %64, label %71

64:                                               ; preds = %62
  %65 = load i64, ptr %16, align 8
  %66 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %64
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 0
  store i64 8, ptr %68, align 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %69, align 8
  %70 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 2
  store i64 %65, ptr %70, align 8
  br label %134

71:                                               ; preds = %62
  %72 = icmp eq ptr %17, @_ZTIPc
  br i1 %72, label %73, label %80

73:                                               ; preds = %71
  %74 = load ptr, ptr %16, align 8
  %75 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %73
  %76 = getelementptr i8, ptr %75, i64 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 0
  store i64 8, ptr %77, align 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %78, align 8
  %79 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 2
  store ptr %74, ptr %79, align 8
  br label %131

80:                                               ; preds = %71
  %81 = icmp eq ptr %17, @_ZTIPv
  br i1 %81, label %82, label %84

82:                                               ; preds = %80
  %83 = load ptr, ptr %16, align 8
  br label %128

84:                                               ; preds = %80
  %85 = load ptr, ptr %17, align 8
  %86 = icmp eq ptr %85, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %86, label %87, label %118

87:                                               ; preds = %84
  %88 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %17, i32 0, i32 2
  %89 = load i32, ptr %88, align 4
  %90 = and i32 %89, 1073741824
  %91 = icmp ne i32 %90, 0
  br i1 %91, label %92, label %108

92:                                               ; preds = %87
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %17, i32 0, i32 4
  %94 = load ptr, ptr %93, align 8
  %95 = icmp ne ptr %94, null
  br i1 %95, label %96, label %98

96:                                               ; preds = %92
  invoke void %94(ptr %1, ptr %16)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %96
  %97 = load ptr, ptr %1, align 8
  br label %105

98:                                               ; preds = %92
  %99 = load ptr, ptr %16, align 8
  %100 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %98
  %101 = getelementptr i8, ptr %100, i64 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 0
  store i64 8, ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %103, align 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 2
  store ptr %99, ptr %104, align 8
  br label %105

105:                                              ; preds = %.noexc7, %.noexc9
  %106 = phi ptr [ %101, %.noexc9 ], [ %97, %.noexc7 ]
  br label %107

107:                                              ; preds = %105
  br label %115

108:                                              ; preds = %87
  %109 = load ptr, ptr %16, align 8
  %110 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %108
  %111 = getelementptr i8, ptr %110, i64 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 0
  store i64 8, ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 2
  store ptr %109, ptr %114, align 8
  br label %115

115:                                              ; preds = %107, %.noexc8
  %116 = phi ptr [ %111, %.noexc8 ], [ %106, %107 ]
  br label %117

117:                                              ; preds = %115
  br label %125

118:                                              ; preds = %84
  %119 = load ptr, ptr %16, align 8
  %120 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %118
  %121 = getelementptr i8, ptr %120, i64 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 0
  store i64 8, ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %123, align 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 2
  store ptr %119, ptr %124, align 8
  br label %125

125:                                              ; preds = %117, %.noexc6
  %126 = phi ptr [ %121, %.noexc6 ], [ %116, %117 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %82, %127
  %129 = phi ptr [ %126, %127 ], [ %83, %82 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc1, %130
  %132 = phi ptr [ %129, %130 ], [ %76, %.noexc1 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc3, %133
  %135 = phi ptr [ %132, %133 ], [ %67, %.noexc3 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc5, %136
  %138 = phi ptr [ %135, %136 ], [ %58, %.noexc5 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc4, %139
  %141 = phi ptr [ %138, %139 ], [ %49, %.noexc4 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc11, %142
  %144 = phi ptr [ %141, %142 ], [ %40, %.noexc11 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc10, %145
  %147 = phi ptr [ %144, %145 ], [ %31, %.noexc10 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc2, %148
  %150 = phi ptr [ %147, %148 ], [ %22, %.noexc2 ]
  br label %151

151:                                              ; preds = %149
  store ptr %150, ptr %2, align 8
  %152 = load i32, ptr %4, align 4
  %153 = add i32 %152, 1
  store i32 %153, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %154

154:                                              ; preds = %12, %151
  %155 = load i32, ptr %3, align 4
  %156 = add i32 %155, 1
  store i32 %156, ptr %3, align 4
  br label %5

157:                                              ; preds = %5
  %158 = load i32, ptr %4, align 4
  %159 = icmp eq i32 %158, 50000
  br i1 %159, label %160, label %162

160:                                              ; preds = %157
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

161:                                              ; preds = %162
  unreachable

162:                                              ; preds = %157
  %163 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12453060767899089178, i64 8), i32 13, ptr null)
  br label %161

endcatch.resume:                                  ; preds = %37, %28, %98, %108, %96, %118, %55, %46, %64, %19, %73, %13
  %164 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %164
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
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

