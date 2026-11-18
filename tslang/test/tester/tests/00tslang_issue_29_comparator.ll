; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_3585166406363658812 = internal constant [135 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa closure or a bound method can't be used as a plain function pointer: the pointer carries no captured variables and no 'this'\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_9888404765695350776 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa plain function\00", align 8
@f_4529517530979974559 = internal constant [118 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00tslang_issue_29_comparator.ts\00", align 8
@m_4268721992646063776 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa closure called directly\00", align 8
@m_10058088038608025772 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa closure as a C comparator throws\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
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
@td_17419097105591164174_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_17419097105591164174, ptr @tsretcb_17419097105591164174, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

define internal void @tsretcb_17419097105591164174(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_17419097105591164174(ptr %0) {
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
declare void @qsort(ptr, i64, i64, ptr) #1

; Function Attrs: mustprogress
define i32 @byValue(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load i32, ptr %6, align 4
  %8 = load ptr, ptr %3, align 8
  %9 = load i32, ptr %8, align 4
  %10 = sub i32 %7, %9
  store i32 %10, ptr %5, align 4
  %11 = load i32, ptr %5, align 4
  ret i32 %11
}

; Function Attrs: mustprogress
define i1 @sortWith(ptr %0, { ptr, ptr, ptr } %1) #2 personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca { ptr, ptr, ptr }, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i1, align 1
  store ptr %0, ptr %6, align 8
  store { ptr, ptr, ptr } %1, ptr %5, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr @__ts_empty_array, ptr %8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr i32, ptr %12, i32 0
  %14 = load ptr, ptr %6, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr @__ts_empty_array, ptr %14
  %17 = getelementptr { ptr, i64, i64 }, ptr %16, i32 0, i32 1
  %18 = load i64, ptr %17, align 8
  %19 = load { ptr, ptr, ptr }, ptr %5, align 8
  %20 = extractvalue { ptr, ptr, ptr } %19, 1
  %21 = ptrtoint ptr %20 to i64
  %22 = icmp ne i64 %21, 0
  br i1 %22, label %23, label %25

23:                                               ; preds = %2
  %24 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_3585166406363658812, i64 8), ptr %24, align 8
  invoke void @__cxa_throw(ptr %24, ptr @_ZTIPc, ptr null)
          to label %169 unwind label %28

25:                                               ; preds = %2
  %26 = extractvalue { ptr, ptr, ptr } %19, 0
  invoke void @qsort(ptr %13, i64 %18, i64 4, ptr %26)
          to label %27 unwind label %28

27:                                               ; preds = %25
  store i1 true, ptr %7, align 1
  br label %167

28:                                               ; preds = %23, %25
  %29 = landingpad { ptr, i32 }
          catch ptr null
  %30 = extractvalue { ptr, i32 } %29, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  %32 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %28
  %33 = icmp eq ptr %32, @_ZTIi
  br i1 %33, label %34, label %41

34:                                               ; preds = %.noexc10
  %35 = load i32, ptr %31, align 4
  %36 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %34
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 0
  store i64 4, ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %39, align 8
  %40 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 2
  store i32 %35, ptr %40, align 4
  br label %164

41:                                               ; preds = %.noexc10
  %42 = icmp eq ptr %32, @_ZTIj
  br i1 %42, label %43, label %50

43:                                               ; preds = %41
  %44 = load i32, ptr %31, align 4
  %45 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %43
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr { i64, ptr, i32 }, ptr %46, i32 0, i32 0
  store i64 4, ptr %47, align 8
  %48 = getelementptr { i64, ptr, i32 }, ptr %46, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %48, align 8
  %49 = getelementptr { i64, ptr, i32 }, ptr %46, i32 0, i32 2
  store i32 %44, ptr %49, align 4
  br label %161

50:                                               ; preds = %41
  %51 = icmp eq ptr %32, @_ZTId
  br i1 %51, label %52, label %59

52:                                               ; preds = %50
  %53 = load double, ptr %31, align 8
  %54 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = getelementptr { i64, ptr, double }, ptr %55, i32 0, i32 0
  store i64 8, ptr %56, align 8
  %57 = getelementptr { i64, ptr, double }, ptr %55, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %57, align 8
  %58 = getelementptr { i64, ptr, double }, ptr %55, i32 0, i32 2
  store double %53, ptr %58, align 8
  br label %158

59:                                               ; preds = %50
  %60 = icmp eq ptr %32, @_ZTIf
  br i1 %60, label %61, label %68

61:                                               ; preds = %59
  %62 = load float, ptr %31, align 4
  %63 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %61
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr { i64, ptr, float }, ptr %64, i32 0, i32 0
  store i64 4, ptr %65, align 8
  %66 = getelementptr { i64, ptr, float }, ptr %64, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %66, align 8
  %67 = getelementptr { i64, ptr, float }, ptr %64, i32 0, i32 2
  store float %62, ptr %67, align 4
  br label %155

68:                                               ; preds = %59
  %69 = icmp eq ptr %32, @_ZTIb
  br i1 %69, label %70, label %77

70:                                               ; preds = %68
  %71 = load i1, ptr %31, align 1
  %72 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %70
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr { i64, ptr, i1 }, ptr %73, i32 0, i32 0
  store i64 1, ptr %74, align 8
  %75 = getelementptr { i64, ptr, i1 }, ptr %73, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %75, align 8
  %76 = getelementptr { i64, ptr, i1 }, ptr %73, i32 0, i32 2
  store i1 %71, ptr %76, align 1
  br label %152

77:                                               ; preds = %68
  %78 = icmp eq ptr %32, @_ZTIx
  br i1 %78, label %79, label %86

79:                                               ; preds = %77
  %80 = load i64, ptr %31, align 8
  %81 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %79
  %82 = getelementptr i8, ptr %81, i64 8
  %83 = getelementptr { i64, ptr, i64 }, ptr %82, i32 0, i32 0
  store i64 8, ptr %83, align 8
  %84 = getelementptr { i64, ptr, i64 }, ptr %82, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %84, align 8
  %85 = getelementptr { i64, ptr, i64 }, ptr %82, i32 0, i32 2
  store i64 %80, ptr %85, align 8
  br label %149

86:                                               ; preds = %77
  %87 = icmp eq ptr %32, @_ZTIPc
  br i1 %87, label %88, label %95

88:                                               ; preds = %86
  %89 = load ptr, ptr %31, align 8
  %90 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %88
  %91 = getelementptr i8, ptr %90, i64 8
  %92 = getelementptr { i64, ptr, ptr }, ptr %91, i32 0, i32 0
  store i64 8, ptr %92, align 8
  %93 = getelementptr { i64, ptr, ptr }, ptr %91, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %93, align 8
  %94 = getelementptr { i64, ptr, ptr }, ptr %91, i32 0, i32 2
  store ptr %89, ptr %94, align 8
  br label %146

95:                                               ; preds = %86
  %96 = icmp eq ptr %32, @_ZTIPv
  br i1 %96, label %97, label %99

97:                                               ; preds = %95
  %98 = load ptr, ptr %31, align 8
  br label %143

99:                                               ; preds = %95
  %100 = load ptr, ptr %32, align 8
  %101 = icmp eq ptr %100, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %101, label %102, label %133

102:                                              ; preds = %99
  %103 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %32, i32 0, i32 2
  %104 = load i32, ptr %103, align 4
  %105 = and i32 %104, 1073741824
  %106 = icmp ne i32 %105, 0
  br i1 %106, label %107, label %123

107:                                              ; preds = %102
  %108 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %32, i32 0, i32 4
  %109 = load ptr, ptr %108, align 8
  %110 = icmp ne ptr %109, null
  br i1 %110, label %111, label %113

111:                                              ; preds = %107
  invoke void %109(ptr %3, ptr %31)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %111
  %112 = load ptr, ptr %3, align 8
  br label %120

113:                                              ; preds = %107
  %114 = load ptr, ptr %31, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %.noexc6, %.noexc7
  %121 = phi ptr [ %116, %.noexc7 ], [ %112, %.noexc6 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %102
  %124 = load ptr, ptr %31, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc4
  %131 = phi ptr [ %126, %.noexc4 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %140

133:                                              ; preds = %99
  %134 = load ptr, ptr %31, align 8
  %135 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %133
  %136 = getelementptr i8, ptr %135, i64 8
  %137 = getelementptr { i64, ptr, ptr }, ptr %136, i32 0, i32 0
  store i64 8, ptr %137, align 8
  %138 = getelementptr { i64, ptr, ptr }, ptr %136, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %138, align 8
  %139 = getelementptr { i64, ptr, ptr }, ptr %136, i32 0, i32 2
  store ptr %134, ptr %139, align 8
  br label %140

140:                                              ; preds = %132, %.noexc2
  %141 = phi ptr [ %136, %.noexc2 ], [ %131, %132 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %97, %142
  %144 = phi ptr [ %141, %142 ], [ %98, %97 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc, %145
  %147 = phi ptr [ %144, %145 ], [ %91, %.noexc ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc8, %148
  %150 = phi ptr [ %147, %148 ], [ %82, %.noexc8 ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc9, %151
  %153 = phi ptr [ %150, %151 ], [ %73, %.noexc9 ]
  br label %154

154:                                              ; preds = %152
  br label %155

155:                                              ; preds = %.noexc5, %154
  %156 = phi ptr [ %153, %154 ], [ %64, %.noexc5 ]
  br label %157

157:                                              ; preds = %155
  br label %158

158:                                              ; preds = %.noexc11, %157
  %159 = phi ptr [ %156, %157 ], [ %55, %.noexc11 ]
  br label %160

160:                                              ; preds = %158
  br label %161

161:                                              ; preds = %.noexc1, %160
  %162 = phi ptr [ %159, %160 ], [ %46, %.noexc1 ]
  br label %163

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %.noexc3, %163
  %165 = phi ptr [ %162, %163 ], [ %37, %.noexc3 ]
  br label %166

166:                                              ; preds = %164
  store ptr %165, ptr %4, align 8
  store i1 false, ptr %7, align 1
  call void @__cxa_end_catch()
  br label %167

167:                                              ; preds = %166, %27
  %168 = load i1, ptr %7, align 1
  ret i1 %168

169:                                              ; preds = %23
  unreachable

endcatch.resume:                                  ; preds = %52, %28, %70, %79, %113, %111, %61, %123, %34, %133, %43, %88
  %170 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %170
}

; Function Attrs: mustprogress
define i32 @.f_main..afL17C28FH2360552237(ptr %0, ptr %1, ptr %2) #1 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  %7 = getelementptr { i32 }, ptr %0, i32 0, i32 0
  %8 = load i32, ptr %7, align 4
  %9 = load ptr, ptr %4, align 8
  %10 = load ptr, ptr %5, align 8
  %11 = call i32 @byValue(ptr %9, ptr %10)
  %12 = add i32 %11, %8
  store i32 %12, ptr %6, align 4
  %13 = load i32, ptr %6, align 4
  ret i32 %13
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 20)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr i32, ptr %3, i64 0
  store i32 3, ptr %4, align 4
  %5 = getelementptr i32, ptr %3, i64 1
  store i32 1, ptr %5, align 4
  %6 = getelementptr i32, ptr %3, i64 2
  store i32 2, ptr %6, align 4
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  store ptr %3, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 1
  store i64 3, ptr %10, align 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 2
  store i64 3, ptr %11, align 8
  store ptr %8, ptr %1, align 8
  %12 = load ptr, ptr %1, align 8
  %13 = call i1 @sortWith(ptr %12, { ptr, ptr, ptr } { ptr @byValue, ptr null, ptr null })
  br i1 %13, label %14, label %23

14:                                               ; preds = %0
  %15 = load ptr, ptr %1, align 8
  %16 = icmp eq ptr %15, null
  %17 = select i1 %16, ptr @__ts_empty_array, ptr %15
  %18 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 0
  %19 = load ptr, ptr %18, align 8
  %20 = getelementptr i32, ptr %19, i32 0
  %21 = load i32, ptr %20, align 4
  %22 = icmp eq i32 %21, 1
  br label %23

23:                                               ; preds = %14, %0
  %24 = phi i1 [ %22, %14 ], [ %13, %0 ]
  br i1 %24, label %25, label %34

25:                                               ; preds = %23
  %26 = load ptr, ptr %1, align 8
  %27 = icmp eq ptr %26, null
  %28 = select i1 %27, ptr @__ts_empty_array, ptr %26
  %29 = getelementptr { ptr, i64, i64 }, ptr %28, i32 0, i32 0
  %30 = load ptr, ptr %29, align 8
  %31 = getelementptr i32, ptr %30, i32 1
  %32 = load i32, ptr %31, align 4
  %33 = icmp eq i32 %32, 2
  br label %34

34:                                               ; preds = %25, %23
  %35 = phi i1 [ %33, %25 ], [ %24, %23 ]
  br i1 %35, label %36, label %45

36:                                               ; preds = %34
  %37 = load ptr, ptr %1, align 8
  %38 = icmp eq ptr %37, null
  %39 = select i1 %38, ptr @__ts_empty_array, ptr %37
  %40 = getelementptr { ptr, i64, i64 }, ptr %39, i32 0, i32 0
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr i32, ptr %41, i32 2
  %43 = load i32, ptr %42, align 4
  %44 = icmp eq i32 %43, 3
  br label %45

45:                                               ; preds = %36, %34
  %46 = phi i1 [ %44, %36 ], [ %35, %34 ]
  br i1 %46, label %47, label %84

47:                                               ; preds = %45
  %48 = call ptr @GC_malloc(i64 12)
  %49 = getelementptr i8, ptr %48, i64 8
  %50 = getelementptr { i32 }, ptr %49, i32 0, i32 0
  store i32 0, ptr %50, align 4
  %51 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL17C28FH2360552237, ptr null, ptr null }, ptr %49, 1
  %52 = insertvalue { ptr, ptr, ptr } %51, ptr getelementptr inbounds nuw (i8, ptr @td_17419097105591164174_object, i64 40), 2
  %53 = load ptr, ptr %1, align 8
  %54 = icmp eq ptr %53, null
  %55 = select i1 %54, ptr @__ts_empty_array, ptr %53
  %56 = getelementptr { ptr, i64, i64 }, ptr %55, i32 0, i32 0
  %57 = load ptr, ptr %56, align 8
  %58 = getelementptr i32, ptr %57, i32 0
  %59 = load ptr, ptr %1, align 8
  %60 = icmp eq ptr %59, null
  %61 = select i1 %60, ptr @__ts_empty_array, ptr %59
  %62 = getelementptr { ptr, i64, i64 }, ptr %61, i32 0, i32 0
  %63 = load ptr, ptr %62, align 8
  %64 = getelementptr i32, ptr %63, i32 1
  %65 = extractvalue { ptr, ptr, ptr } %52, 1
  %66 = ptrtoint ptr %65 to i64
  %67 = icmp ne i64 %66, 0
  br i1 %67, label %68, label %71

68:                                               ; preds = %47
  %69 = extractvalue { ptr, ptr, ptr } %52, 0
  %70 = call i32 %69(ptr %65, ptr %58, ptr %64)
  br label %74

71:                                               ; preds = %47
  %72 = extractvalue { ptr, ptr, ptr } %52, 0
  %73 = call i32 %72(ptr %58, ptr %64)
  br label %74

74:                                               ; preds = %68, %71
  %75 = phi i32 [ %73, %71 ], [ %70, %68 ]
  br label %76

76:                                               ; preds = %74
  %77 = icmp sgt i32 %75, 0
  br i1 %77, label %78, label %86

78:                                               ; preds = %76
  %79 = load ptr, ptr %1, align 8
  %80 = call i1 @sortWith(ptr %79, { ptr, ptr, ptr } %52)
  %81 = xor i1 %80, true
  br i1 %81, label %82, label %88

82:                                               ; preds = %78
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

83:                                               ; preds = %88, %86, %84
  unreachable

84:                                               ; preds = %45
  %85 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9888404765695350776, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4529517530979974559, i64 8), i32 15, ptr null)
  br label %83

86:                                               ; preds = %76
  %87 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4268721992646063776, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4529517530979974559, i64 8), i32 18, ptr null)
  br label %83

88:                                               ; preds = %78
  %89 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10058088038608025772, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4529517530979974559, i64 8), i32 19, ptr null)
  br label %83
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

