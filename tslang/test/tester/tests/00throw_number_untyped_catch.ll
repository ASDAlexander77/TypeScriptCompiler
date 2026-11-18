; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_12252883629495545558 = internal constant [80 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch must catch a thrown int and control flow must continue\00", align 8
@f_13393505935389037822 = internal constant [118 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_number_untyped_catch.ts\00", align 8
@m_10632935273526443851 = internal constant [82 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa `catch (e: any)` must catch a thrown int and control flow must continue\00", align 8
@m_8565447958558989591 = internal constant [82 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch must catch a thrown float and control flow must continue\00", align 8
@m_7133933383724466346 = internal constant [84 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa `catch (e: any)` must catch a thrown float and control flow must continue\00", align 8
@m_6705484713491263742 = internal constant [94 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch must catch a thrown `number` variable and control flow must continue\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define i32 @untypedCatchesInt() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 42, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc4, %.noexc1
  %99 = phi ptr [ %94, %.noexc1 ], [ %90, %.noexc4 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc5
  %109 = phi ptr [ %104, %.noexc5 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc6
  %119 = phi ptr [ %114, %.noexc6 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc8, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc8 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc11, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc11 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc10, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc10 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc7, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc7 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc3, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc3 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc9, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc9 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc2, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc2 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  %145 = load i32, ptr %3, align 4
  %146 = add i32 %145, 10
  store i32 %146, ptr %3, align 4
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %57, %48, %21, %66, %39, %111, %101, %89, %30, %12, %91, %6
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i32 @anyCatchesInt() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 42, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc7
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc7
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc, %.noexc3
  %99 = phi ptr [ %94, %.noexc3 ], [ %90, %.noexc ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc11
  %109 = phi ptr [ %104, %.noexc11 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc5
  %119 = phi ptr [ %114, %.noexc5 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc4, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc4 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc2, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc2 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc1, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc1 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc9, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc9 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc8, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc8 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc10, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc10 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc6, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc6 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  %145 = load i32, ptr %3, align 4
  %146 = add i32 %145, 10
  store i32 %146, ptr %3, align 4
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %101, %21, %39, %30, %6, %12, %111, %66, %91, %57, %48, %89
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i32 @untypedCatchesFloat() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store double 2.500000e+00, ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTId, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc2
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc2
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc, %.noexc8
  %99 = phi ptr [ %94, %.noexc8 ], [ %90, %.noexc ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc5
  %109 = phi ptr [ %104, %.noexc5 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc3
  %119 = phi ptr [ %114, %.noexc3 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc9, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc9 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc6, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc6 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc7, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc7 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc11, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc11 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc1, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc1 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc4, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc4 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc10, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc10 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  %145 = load i32, ptr %3, align 4
  %146 = add i32 %145, 10
  store i32 %146, ptr %3, align 4
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %39, %12, %66, %91, %48, %57, %101, %21, %111, %6, %30, %89
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i32 @anyCatchesFloat() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store double 2.500000e+00, ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTId, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc1
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc1
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc4, %.noexc2
  %99 = phi ptr [ %94, %.noexc2 ], [ %90, %.noexc4 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc
  %109 = phi ptr [ %104, %.noexc ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc10
  %119 = phi ptr [ %114, %.noexc10 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc9, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc9 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc8, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc8 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc7, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc7 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc3, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc3 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc11, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc11 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc5, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc5 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc6, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc6 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  %145 = load i32, ptr %3, align 4
  %146 = add i32 %145, 10
  store i32 %146, ptr %3, align 4
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %30, %111, %66, %57, %48, %12, %21, %89, %39, %91, %6, %101
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i32 @untypedCatchesNumberVar() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca i32, align 4
  store double 4.200000e+01, ptr %4, align 8
  store i32 0, ptr %3, align 4
  %6 = load double, ptr %4, align 8
  %7 = call ptr @__cxa_allocate_exception(i64 8)
  store double %6, ptr %7, align 8
  invoke void @__cxa_throw(ptr %7, ptr @_ZTId, ptr null)
          to label %151 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %8
  %13 = icmp eq ptr %12, @_ZTIi
  br i1 %13, label %14, label %21

14:                                               ; preds = %.noexc9
  %15 = load i32, ptr %11, align 4
  %16 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %14
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 0
  store i64 4, ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 2
  store i32 %15, ptr %20, align 4
  br label %144

21:                                               ; preds = %.noexc9
  %22 = icmp eq ptr %12, @_ZTIj
  br i1 %22, label %23, label %30

23:                                               ; preds = %21
  %24 = load i32, ptr %11, align 4
  %25 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %23
  %26 = getelementptr i8, ptr %25, i64 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %26, i32 0, i32 0
  store i64 4, ptr %27, align 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %26, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %26, i32 0, i32 2
  store i32 %24, ptr %29, align 4
  br label %141

30:                                               ; preds = %21
  %31 = icmp eq ptr %12, @_ZTId
  br i1 %31, label %32, label %39

32:                                               ; preds = %30
  %33 = load double, ptr %11, align 8
  %34 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %32
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr { i64, ptr, double }, ptr %35, i32 0, i32 0
  store i64 8, ptr %36, align 8
  %37 = getelementptr { i64, ptr, double }, ptr %35, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %35, i32 0, i32 2
  store double %33, ptr %38, align 8
  br label %138

39:                                               ; preds = %30
  %40 = icmp eq ptr %12, @_ZTIf
  br i1 %40, label %41, label %48

41:                                               ; preds = %39
  %42 = load float, ptr %11, align 4
  %43 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %41
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = getelementptr { i64, ptr, float }, ptr %44, i32 0, i32 0
  store i64 4, ptr %45, align 8
  %46 = getelementptr { i64, ptr, float }, ptr %44, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %44, i32 0, i32 2
  store float %42, ptr %47, align 4
  br label %135

48:                                               ; preds = %39
  %49 = icmp eq ptr %12, @_ZTIb
  br i1 %49, label %50, label %57

50:                                               ; preds = %48
  %51 = load i1, ptr %11, align 1
  %52 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %50
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %53, i32 0, i32 0
  store i64 1, ptr %54, align 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %53, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %53, i32 0, i32 2
  store i1 %51, ptr %56, align 1
  br label %132

57:                                               ; preds = %48
  %58 = icmp eq ptr %12, @_ZTIx
  br i1 %58, label %59, label %66

59:                                               ; preds = %57
  %60 = load i64, ptr %11, align 8
  %61 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %59
  %62 = getelementptr i8, ptr %61, i64 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %62, i32 0, i32 0
  store i64 8, ptr %63, align 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %62, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %62, i32 0, i32 2
  store i64 %60, ptr %65, align 8
  br label %129

66:                                               ; preds = %57
  %67 = icmp eq ptr %12, @_ZTIPc
  br i1 %67, label %68, label %75

68:                                               ; preds = %66
  %69 = load ptr, ptr %11, align 8
  %70 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %68
  %71 = getelementptr i8, ptr %70, i64 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %71, i32 0, i32 0
  store i64 8, ptr %72, align 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %71, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %71, i32 0, i32 2
  store ptr %69, ptr %74, align 8
  br label %126

75:                                               ; preds = %66
  %76 = icmp eq ptr %12, @_ZTIPv
  br i1 %76, label %77, label %79

77:                                               ; preds = %75
  %78 = load ptr, ptr %11, align 8
  br label %123

79:                                               ; preds = %75
  %80 = load ptr, ptr %12, align 8
  %81 = icmp eq ptr %80, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %81, label %82, label %113

82:                                               ; preds = %79
  %83 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %12, i32 0, i32 2
  %84 = load i32, ptr %83, align 4
  %85 = and i32 %84, 1073741824
  %86 = icmp ne i32 %85, 0
  br i1 %86, label %87, label %103

87:                                               ; preds = %82
  %88 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %12, i32 0, i32 4
  %89 = load ptr, ptr %88, align 8
  %90 = icmp ne ptr %89, null
  br i1 %90, label %91, label %93

91:                                               ; preds = %87
  invoke void %89(ptr %1, ptr %11)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %91
  %92 = load ptr, ptr %1, align 8
  br label %100

93:                                               ; preds = %87
  %94 = load ptr, ptr %11, align 8
  %95 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %93
  %96 = getelementptr i8, ptr %95, i64 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 0
  store i64 8, ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 2
  store ptr %94, ptr %99, align 8
  br label %100

100:                                              ; preds = %.noexc1, %.noexc5
  %101 = phi ptr [ %96, %.noexc5 ], [ %92, %.noexc1 ]
  br label %102

102:                                              ; preds = %100
  br label %110

103:                                              ; preds = %82
  %104 = load ptr, ptr %11, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %102, %.noexc7
  %111 = phi ptr [ %106, %.noexc7 ], [ %101, %102 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %79
  %114 = load ptr, ptr %11, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc11
  %121 = phi ptr [ %116, %.noexc11 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %123

123:                                              ; preds = %77, %122
  %124 = phi ptr [ %121, %122 ], [ %78, %77 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %.noexc, %125
  %127 = phi ptr [ %124, %125 ], [ %71, %.noexc ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc10, %128
  %130 = phi ptr [ %127, %128 ], [ %62, %.noexc10 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc6, %131
  %133 = phi ptr [ %130, %131 ], [ %53, %.noexc6 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc2, %134
  %136 = phi ptr [ %133, %134 ], [ %44, %.noexc2 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc8, %137
  %139 = phi ptr [ %136, %137 ], [ %35, %.noexc8 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc4, %140
  %142 = phi ptr [ %139, %140 ], [ %26, %.noexc4 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc3, %143
  %145 = phi ptr [ %142, %143 ], [ %17, %.noexc3 ]
  br label %146

146:                                              ; preds = %144
  store ptr %145, ptr %2, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  %147 = load i32, ptr %3, align 4
  %148 = add i32 %147, 10
  store i32 %148, ptr %3, align 4
  %149 = load i32, ptr %3, align 4
  store i32 %149, ptr %5, align 4
  %150 = load i32, ptr %5, align 4
  ret i32 %150

151:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %113, %59, %8, %32, %103, %50, %93, %23, %14, %41, %91, %68
  %152 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %152
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i32 @untypedCatchesInt()
  %2 = icmp eq i32 %1, 11
  br i1 %2, label %3, label %17

3:                                                ; preds = %0
  %4 = call i32 @anyCatchesInt()
  %5 = icmp eq i32 %4, 11
  br i1 %5, label %6, label %19

6:                                                ; preds = %3
  %7 = call i32 @untypedCatchesFloat()
  %8 = icmp eq i32 %7, 11
  br i1 %8, label %9, label %21

9:                                                ; preds = %6
  %10 = call i32 @anyCatchesFloat()
  %11 = icmp eq i32 %10, 11
  br i1 %11, label %12, label %23

12:                                               ; preds = %9
  %13 = call i32 @untypedCatchesNumberVar()
  %14 = icmp eq i32 %13, 11
  br i1 %14, label %15, label %25

15:                                               ; preds = %12
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

16:                                               ; preds = %25, %23, %21, %19, %17
  unreachable

17:                                               ; preds = %0
  %18 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12252883629495545558, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13393505935389037822, i64 8), i32 81, ptr null)
  br label %16

19:                                               ; preds = %3
  %20 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10632935273526443851, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13393505935389037822, i64 8), i32 82, ptr null)
  br label %16

21:                                               ; preds = %6
  %22 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8565447958558989591, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13393505935389037822, i64 8), i32 83, ptr null)
  br label %16

23:                                               ; preds = %9
  %24 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7133933383724466346, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13393505935389037822, i64 8), i32 84, ptr null)
  br label %16

25:                                               ; preds = %12
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6705484713491263742, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13393505935389037822, i64 8), i32 85, ptr null)
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

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress "personality" }
attributes #2 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

