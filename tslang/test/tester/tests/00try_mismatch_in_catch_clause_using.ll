; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_7464751892813498902 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwith a using\00", align 8
@f_15081112902223709724 = internal constant [126 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_mismatch_in_catch_clause_using.ts\00", align 8
@m_441435409882857575 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwith a using and a finally\00", align 8
@m_17909071614601602156 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFno try around the catch\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@disposed = private global i32 0
@finals = private global i32 0
@wrong = private global i32 0
@outer = private global i32 0
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
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

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @__cxa_rethrow()

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
declare noalias ptr @GC_malloc(i64) #1

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Res..rtti, align 8
  %7 = ptrtoint ptr %5 to i64
  %8 = ptrtoint ptr %6 to i64
  %9 = icmp ne i64 %7, 0
  %10 = icmp ne i64 %8, 0
  %11 = zext i1 %9 to i32
  %12 = zext i1 %10 to i32
  %13 = and i32 %11, %12
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %18

15:                                               ; preds = %2
  %16 = call i32 @strcmp(ptr %5, ptr %6)
  %17 = icmp eq i32 %16, 0
  br label %20

18:                                               ; preds = %2
  %19 = icmp eq i64 %7, %8
  br label %20

20:                                               ; preds = %15, %18
  %21 = phi i1 [ %19, %18 ], [ %17, %15 ]
  br label %22

22:                                               ; preds = %20
  store i1 %21, ptr %4, align 1
  %23 = load i1, ptr %4, align 1
  ret i1 %23
}

; Function Attrs: mustprogress
define ptr @Res..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  ret void
}

; Function Attrs: mustprogress
define void @reset() #2 {
  store i32 0, ptr @disposed, align 4
  store i32 0, ptr @finals, align 4
  store i32 0, ptr @wrong, align 4
  store i32 0, ptr @outer, align 4
  ret void
}

; Function Attrs: mustprogress
define void @withUsing() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %370 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc16
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc16
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %142

31:                                               ; preds = %22
  %32 = icmp eq ptr %13, @_ZTId
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load double, ptr %12, align 8
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 0
  store i64 8, ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 2
  store double %34, ptr %39, align 8
  br label %139

40:                                               ; preds = %31
  %41 = icmp eq ptr %13, @_ZTIf
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load float, ptr %12, align 4
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 0
  store i64 4, ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 2
  store float %43, ptr %48, align 4
  br label %136

49:                                               ; preds = %40
  %50 = icmp eq ptr %13, @_ZTIb
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load i1, ptr %12, align 1
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 0
  store i64 1, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 2
  store i1 %52, ptr %57, align 1
  br label %133

58:                                               ; preds = %49
  %59 = icmp eq ptr %13, @_ZTIx
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i64, ptr %12, align 8
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 0
  store i64 8, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 2
  store i64 %61, ptr %66, align 8
  br label %130

67:                                               ; preds = %58
  %68 = icmp eq ptr %13, @_ZTIPc
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load ptr, ptr %12, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 2
  store ptr %70, ptr %75, align 8
  br label %127

76:                                               ; preds = %67
  %77 = icmp eq ptr %13, @_ZTIPv
  br i1 %77, label %78, label %80

78:                                               ; preds = %76
  %79 = load ptr, ptr %12, align 8
  br label %124

80:                                               ; preds = %76
  %81 = load ptr, ptr %13, align 8
  %82 = icmp eq ptr %81, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %82, label %83, label %114

83:                                               ; preds = %80
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 2
  %85 = load i32, ptr %84, align 4
  %86 = and i32 %85, 1073741824
  %87 = icmp ne i32 %86, 0
  br i1 %87, label %88, label %104

88:                                               ; preds = %83
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 4
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %94

92:                                               ; preds = %88
  invoke void %90(ptr %5, ptr %12)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %92
  %93 = load ptr, ptr %5, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc28, %.noexc30
  %102 = phi ptr [ %97, %.noexc30 ], [ %93, %.noexc28 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc
  %112 = phi ptr [ %107, %.noexc ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc32
  %122 = phi ptr [ %117, %.noexc32 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc24, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc24 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc21, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc21 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc18, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc18 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc23, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc23 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc17, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc17 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc14, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc14 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc13, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc13 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %6, align 8
  store i1 false, ptr %2, align 1
  %148 = invoke ptr @GC_malloc(i64 16)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %147
  %149 = getelementptr i8, ptr %148, i64 8
  %150 = getelementptr %Res, ptr %149, i32 0, i32 0
  store ptr @Res..vtbl, ptr %150, align 8
  store ptr %149, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %151 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %.noexc11
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %151, align 8
  invoke void @__cxa_throw(ptr %151, ptr @_ZTIPc, ptr null)
          to label %370 unwind label %152

152:                                              ; preds = %.noexc12
  %153 = landingpad { ptr, i32 }
          catch ptr null
  %154 = extractvalue { ptr, i32 } %153, 0
  %155 = call ptr @__cxa_begin_catch(ptr %154)
  %156 = load i1, ptr %2, align 1
  br i1 %156, label %157, label %172

157:                                              ; preds = %152
  store i1 false, ptr %2, align 1
  %158 = load ptr, ptr %3, align 8
  %159 = getelementptr %Res, ptr %158, i32 0, i32 0
  invoke void @Res.dispose(ptr %158)
          to label %160 unwind label %endcatch.pad2

160:                                              ; preds = %157
  br label %172

endcatch.pad2:                                    ; preds = %157
  %161 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

162:                                              ; No predecessors!
  %163 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %162
  %164 = phi { ptr, i32 } [ %163, %162 ], [ %161, %endcatch.pad2 ]
  %165 = extractvalue { ptr, i32 } %164, 0
  %166 = call ptr @__cxa_begin_catch(ptr %165)
  %167 = load i1, ptr %2, align 1
  br i1 %167, label %168, label %171

168:                                              ; preds = %.caught1
  store i1 false, ptr %2, align 1
  %169 = load ptr, ptr %3, align 8
  %170 = getelementptr %Res, ptr %169, i32 0, i32 0
  invoke void @Res.dispose(ptr %169)
          to label %.noexc7 unwind label %endcatch.resume6

.noexc7:                                          ; preds = %168
  br label %171

171:                                              ; preds = %.noexc7, %.caught1
  invoke void @__cxa_rethrow()
          to label %370 unwind label %endcatch.pad

172:                                              ; preds = %160, %152
  invoke void @__cxa_rethrow()
          to label %370 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %171, %172
  %173 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

174:                                              ; No predecessors!
  %175 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %174
  %176 = phi { ptr, i32 } [ %175, %174 ], [ %173, %endcatch.pad ]
  %177 = extractvalue { ptr, i32 } %176, 1
  %178 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %179 = icmp eq i32 %177, %178
  %180 = extractvalue { ptr, i32 } %176, 1
  %181 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %182 = icmp eq i32 %180, %181
  %183 = extractvalue { ptr, i32 } %176, 1
  %184 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %185 = icmp eq i32 %183, %184
  %186 = extractvalue { ptr, i32 } %176, 1
  %187 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %188 = icmp eq i32 %186, %187
  %189 = or i1 %179, %182
  %190 = or i1 %189, %185
  %191 = or i1 %190, %188
  br i1 %191, label %195, label %192

192:                                              ; preds = %.caught
  %193 = extractvalue { ptr, i32 } %176, 0
  %194 = call ptr @__cxa_begin_catch(ptr %193)
  invoke void @__cxa_rethrow()
          to label %370 unwind label %endcatch.pad4

195:                                              ; preds = %.caught
  %196 = extractvalue { ptr, i32 } %176, 0
  %197 = call ptr @__cxa_begin_catch(ptr %196)
  %198 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc31 unwind label %endcatch.resume6

.noexc31:                                         ; preds = %195
  %199 = icmp eq ptr %198, @_ZTIi
  br i1 %199, label %200, label %203

200:                                              ; preds = %.noexc31
  %201 = load i32, ptr %197, align 4
  %202 = sitofp i32 %201 to double
  br label %221

203:                                              ; preds = %.noexc31
  %204 = icmp eq ptr %198, @_ZTIj
  br i1 %204, label %205, label %208

205:                                              ; preds = %203
  %206 = load i32, ptr %197, align 4
  %207 = uitofp i32 %206 to double
  br label %218

208:                                              ; preds = %203
  %209 = icmp eq ptr %198, @_ZTIf
  br i1 %209, label %210, label %213

210:                                              ; preds = %208
  %211 = load float, ptr %197, align 4
  %212 = fpext float %211 to double
  br label %215

213:                                              ; preds = %208
  %214 = load double, ptr %197, align 8
  br label %215

215:                                              ; preds = %210, %213
  %216 = phi double [ %214, %213 ], [ %212, %210 ]
  br label %217

217:                                              ; preds = %215
  br label %218

218:                                              ; preds = %205, %217
  %219 = phi double [ %216, %217 ], [ %207, %205 ]
  br label %220

220:                                              ; preds = %218
  br label %221

221:                                              ; preds = %200, %220
  %222 = phi double [ %219, %220 ], [ %202, %200 ]
  br label %223

223:                                              ; preds = %221
  store double %222, ptr %4, align 8
  %224 = load i32, ptr @wrong, align 4
  %225 = add i32 %224, 1
  store i32 %225, ptr @wrong, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %369

endcatch.pad4:                                    ; preds = %192
  %226 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught3

227:                                              ; No predecessors!
  %228 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %227
  %229 = phi { ptr, i32 } [ %228, %227 ], [ %226, %endcatch.pad4 ]
  %230 = extractvalue { ptr, i32 } %229, 0
  %231 = call ptr @__cxa_begin_catch(ptr %230)
  %232 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %.caught3
  %233 = icmp eq ptr %232, @_ZTIi
  br i1 %233, label %234, label %241

234:                                              ; preds = %.noexc20
  %235 = load i32, ptr %231, align 4
  %236 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %234
  %237 = getelementptr i8, ptr %236, i64 8
  %238 = getelementptr { i64, ptr, i32 }, ptr %237, i32 0, i32 0
  store i64 4, ptr %238, align 8
  %239 = getelementptr { i64, ptr, i32 }, ptr %237, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %239, align 8
  %240 = getelementptr { i64, ptr, i32 }, ptr %237, i32 0, i32 2
  store i32 %235, ptr %240, align 4
  br label %364

241:                                              ; preds = %.noexc20
  %242 = icmp eq ptr %232, @_ZTIj
  br i1 %242, label %243, label %250

243:                                              ; preds = %241
  %244 = load i32, ptr %231, align 4
  %245 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %243
  %246 = getelementptr i8, ptr %245, i64 8
  %247 = getelementptr { i64, ptr, i32 }, ptr %246, i32 0, i32 0
  store i64 4, ptr %247, align 8
  %248 = getelementptr { i64, ptr, i32 }, ptr %246, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %248, align 8
  %249 = getelementptr { i64, ptr, i32 }, ptr %246, i32 0, i32 2
  store i32 %244, ptr %249, align 4
  br label %361

250:                                              ; preds = %241
  %251 = icmp eq ptr %232, @_ZTId
  br i1 %251, label %252, label %259

252:                                              ; preds = %250
  %253 = load double, ptr %231, align 8
  %254 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %252
  %255 = getelementptr i8, ptr %254, i64 8
  %256 = getelementptr { i64, ptr, double }, ptr %255, i32 0, i32 0
  store i64 8, ptr %256, align 8
  %257 = getelementptr { i64, ptr, double }, ptr %255, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %257, align 8
  %258 = getelementptr { i64, ptr, double }, ptr %255, i32 0, i32 2
  store double %253, ptr %258, align 8
  br label %358

259:                                              ; preds = %250
  %260 = icmp eq ptr %232, @_ZTIf
  br i1 %260, label %261, label %268

261:                                              ; preds = %259
  %262 = load float, ptr %231, align 4
  %263 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %261
  %264 = getelementptr i8, ptr %263, i64 8
  %265 = getelementptr { i64, ptr, float }, ptr %264, i32 0, i32 0
  store i64 4, ptr %265, align 8
  %266 = getelementptr { i64, ptr, float }, ptr %264, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %266, align 8
  %267 = getelementptr { i64, ptr, float }, ptr %264, i32 0, i32 2
  store float %262, ptr %267, align 4
  br label %355

268:                                              ; preds = %259
  %269 = icmp eq ptr %232, @_ZTIb
  br i1 %269, label %270, label %277

270:                                              ; preds = %268
  %271 = load i1, ptr %231, align 1
  %272 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %270
  %273 = getelementptr i8, ptr %272, i64 8
  %274 = getelementptr { i64, ptr, i1 }, ptr %273, i32 0, i32 0
  store i64 1, ptr %274, align 8
  %275 = getelementptr { i64, ptr, i1 }, ptr %273, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %275, align 8
  %276 = getelementptr { i64, ptr, i1 }, ptr %273, i32 0, i32 2
  store i1 %271, ptr %276, align 1
  br label %352

277:                                              ; preds = %268
  %278 = icmp eq ptr %232, @_ZTIx
  br i1 %278, label %279, label %286

279:                                              ; preds = %277
  %280 = load i64, ptr %231, align 8
  %281 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %279
  %282 = getelementptr i8, ptr %281, i64 8
  %283 = getelementptr { i64, ptr, i64 }, ptr %282, i32 0, i32 0
  store i64 8, ptr %283, align 8
  %284 = getelementptr { i64, ptr, i64 }, ptr %282, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %284, align 8
  %285 = getelementptr { i64, ptr, i64 }, ptr %282, i32 0, i32 2
  store i64 %280, ptr %285, align 8
  br label %349

286:                                              ; preds = %277
  %287 = icmp eq ptr %232, @_ZTIPc
  br i1 %287, label %288, label %295

288:                                              ; preds = %286
  %289 = load ptr, ptr %231, align 8
  %290 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %288
  %291 = getelementptr i8, ptr %290, i64 8
  %292 = getelementptr { i64, ptr, ptr }, ptr %291, i32 0, i32 0
  store i64 8, ptr %292, align 8
  %293 = getelementptr { i64, ptr, ptr }, ptr %291, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %293, align 8
  %294 = getelementptr { i64, ptr, ptr }, ptr %291, i32 0, i32 2
  store ptr %289, ptr %294, align 8
  br label %346

295:                                              ; preds = %286
  %296 = icmp eq ptr %232, @_ZTIPv
  br i1 %296, label %297, label %299

297:                                              ; preds = %295
  %298 = load ptr, ptr %231, align 8
  br label %343

299:                                              ; preds = %295
  %300 = load ptr, ptr %232, align 8
  %301 = icmp eq ptr %300, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %301, label %302, label %333

302:                                              ; preds = %299
  %303 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %232, i32 0, i32 2
  %304 = load i32, ptr %303, align 4
  %305 = and i32 %304, 1073741824
  %306 = icmp ne i32 %305, 0
  br i1 %306, label %307, label %323

307:                                              ; preds = %302
  %308 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %232, i32 0, i32 4
  %309 = load ptr, ptr %308, align 8
  %310 = icmp ne ptr %309, null
  br i1 %310, label %311, label %313

311:                                              ; preds = %307
  invoke void %309(ptr %1, ptr %231)
          to label %.noexc29 unwind label %endcatch.resume

.noexc29:                                         ; preds = %311
  %312 = load ptr, ptr %1, align 8
  br label %320

313:                                              ; preds = %307
  %314 = load ptr, ptr %231, align 8
  %315 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %313
  %316 = getelementptr i8, ptr %315, i64 8
  %317 = getelementptr { i64, ptr, ptr }, ptr %316, i32 0, i32 0
  store i64 8, ptr %317, align 8
  %318 = getelementptr { i64, ptr, ptr }, ptr %316, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %318, align 8
  %319 = getelementptr { i64, ptr, ptr }, ptr %316, i32 0, i32 2
  store ptr %314, ptr %319, align 8
  br label %320

320:                                              ; preds = %.noexc29, %.noexc8
  %321 = phi ptr [ %316, %.noexc8 ], [ %312, %.noexc29 ]
  br label %322

322:                                              ; preds = %320
  br label %330

323:                                              ; preds = %302
  %324 = load ptr, ptr %231, align 8
  %325 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %323
  %326 = getelementptr i8, ptr %325, i64 8
  %327 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 0
  store i64 8, ptr %327, align 8
  %328 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %328, align 8
  %329 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 2
  store ptr %324, ptr %329, align 8
  br label %330

330:                                              ; preds = %322, %.noexc27
  %331 = phi ptr [ %326, %.noexc27 ], [ %321, %322 ]
  br label %332

332:                                              ; preds = %330
  br label %340

333:                                              ; preds = %299
  %334 = load ptr, ptr %231, align 8
  %335 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %333
  %336 = getelementptr i8, ptr %335, i64 8
  %337 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 0
  store i64 8, ptr %337, align 8
  %338 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %338, align 8
  %339 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 2
  store ptr %334, ptr %339, align 8
  br label %340

340:                                              ; preds = %332, %.noexc5
  %341 = phi ptr [ %336, %.noexc5 ], [ %331, %332 ]
  br label %342

342:                                              ; preds = %340
  br label %343

343:                                              ; preds = %297, %342
  %344 = phi ptr [ %341, %342 ], [ %298, %297 ]
  br label %345

345:                                              ; preds = %343
  br label %346

346:                                              ; preds = %.noexc22, %345
  %347 = phi ptr [ %344, %345 ], [ %291, %.noexc22 ]
  br label %348

348:                                              ; preds = %346
  br label %349

349:                                              ; preds = %.noexc15, %348
  %350 = phi ptr [ %347, %348 ], [ %282, %.noexc15 ]
  br label %351

351:                                              ; preds = %349
  br label %352

352:                                              ; preds = %.noexc19, %351
  %353 = phi ptr [ %350, %351 ], [ %273, %.noexc19 ]
  br label %354

354:                                              ; preds = %352
  br label %355

355:                                              ; preds = %.noexc10, %354
  %356 = phi ptr [ %353, %354 ], [ %264, %.noexc10 ]
  br label %357

357:                                              ; preds = %355
  br label %358

358:                                              ; preds = %.noexc26, %357
  %359 = phi ptr [ %356, %357 ], [ %255, %.noexc26 ]
  br label %360

360:                                              ; preds = %358
  br label %361

361:                                              ; preds = %.noexc25, %360
  %362 = phi ptr [ %359, %360 ], [ %246, %.noexc25 ]
  br label %363

363:                                              ; preds = %361
  br label %364

364:                                              ; preds = %.noexc9, %363
  %365 = phi ptr [ %362, %363 ], [ %237, %.noexc9 ]
  br label %366

366:                                              ; preds = %364
  store ptr %365, ptr %7, align 8
  %367 = load i32, ptr @outer, align 4
  %368 = add i32 %367, 1
  store i32 %368, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %369

369:                                              ; preds = %223, %366
  ret void

370:                                              ; preds = %192, %172, %171, %.noexc12, %0
  unreachable

endcatch.resume:                                  ; preds = %114, %94, %311, %92, %323, %252, %243, %69, %42, %288, %60, %.caught3, %270, %51, %33, %9, %279, %24, %15, %.noexc11, %147, %261, %234, %313, %333, %104
  %371 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %371

endcatch.resume6:                                 ; preds = %195, %168
  %372 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %372
}

; Function Attrs: mustprogress
define void @withUsingAndFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %380 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc13
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc13
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %142

31:                                               ; preds = %22
  %32 = icmp eq ptr %13, @_ZTId
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load double, ptr %12, align 8
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 0
  store i64 8, ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 2
  store double %34, ptr %39, align 8
  br label %139

40:                                               ; preds = %31
  %41 = icmp eq ptr %13, @_ZTIf
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load float, ptr %12, align 4
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 0
  store i64 4, ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 2
  store float %43, ptr %48, align 4
  br label %136

49:                                               ; preds = %40
  %50 = icmp eq ptr %13, @_ZTIb
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load i1, ptr %12, align 1
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 0
  store i64 1, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 2
  store i1 %52, ptr %57, align 1
  br label %133

58:                                               ; preds = %49
  %59 = icmp eq ptr %13, @_ZTIx
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i64, ptr %12, align 8
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 0
  store i64 8, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 2
  store i64 %61, ptr %66, align 8
  br label %130

67:                                               ; preds = %58
  %68 = icmp eq ptr %13, @_ZTIPc
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load ptr, ptr %12, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc29 unwind label %endcatch.resume

.noexc29:                                         ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 2
  store ptr %70, ptr %75, align 8
  br label %127

76:                                               ; preds = %67
  %77 = icmp eq ptr %13, @_ZTIPv
  br i1 %77, label %78, label %80

78:                                               ; preds = %76
  %79 = load ptr, ptr %12, align 8
  br label %124

80:                                               ; preds = %76
  %81 = load ptr, ptr %13, align 8
  %82 = icmp eq ptr %81, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %82, label %83, label %114

83:                                               ; preds = %80
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 2
  %85 = load i32, ptr %84, align 4
  %86 = and i32 %85, 1073741824
  %87 = icmp ne i32 %86, 0
  br i1 %87, label %88, label %104

88:                                               ; preds = %83
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 4
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %94

92:                                               ; preds = %88
  invoke void %90(ptr %5, ptr %12)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %92
  %93 = load ptr, ptr %5, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc32, %.noexc33
  %102 = phi ptr [ %97, %.noexc33 ], [ %93, %.noexc32 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc17
  %112 = phi ptr [ %107, %.noexc17 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc19
  %122 = phi ptr [ %117, %.noexc19 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc29, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc29 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc18, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc18 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc16, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc16 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc14, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc14 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc21, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc21 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc25, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc25 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc27, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc27 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %6, align 8
  store i1 false, ptr %2, align 1
  %148 = invoke ptr @GC_malloc(i64 16)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %147
  %149 = getelementptr i8, ptr %148, i64 8
  %150 = getelementptr %Res, ptr %149, i32 0, i32 0
  store ptr @Res..vtbl, ptr %150, align 8
  store ptr %149, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %151 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %151, align 8
  invoke void @__cxa_throw(ptr %151, ptr @_ZTIPc, ptr null)
          to label %380 unwind label %152

152:                                              ; preds = %.noexc8
  %153 = landingpad { ptr, i32 }
          catch ptr null
  %154 = extractvalue { ptr, i32 } %153, 0
  %155 = call ptr @__cxa_begin_catch(ptr %154)
  %156 = load i1, ptr %2, align 1
  br i1 %156, label %157, label %172

157:                                              ; preds = %152
  store i1 false, ptr %2, align 1
  %158 = load ptr, ptr %3, align 8
  %159 = getelementptr %Res, ptr %158, i32 0, i32 0
  invoke void @Res.dispose(ptr %158)
          to label %160 unwind label %endcatch.pad4

160:                                              ; preds = %157
  br label %172

endcatch.pad4:                                    ; preds = %157
  %161 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

162:                                              ; No predecessors!
  %163 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %162
  %164 = phi { ptr, i32 } [ %163, %162 ], [ %161, %endcatch.pad4 ]
  %165 = extractvalue { ptr, i32 } %164, 0
  %166 = call ptr @__cxa_begin_catch(ptr %165)
  %167 = load i1, ptr %2, align 1
  br i1 %167, label %168, label %171

168:                                              ; preds = %.caught3
  store i1 false, ptr %2, align 1
  %169 = load ptr, ptr %3, align 8
  %170 = getelementptr %Res, ptr %169, i32 0, i32 0
  invoke void @Res.dispose(ptr %169)
          to label %.noexc12 unwind label %endcatch.resume11

.noexc12:                                         ; preds = %168
  br label %171

171:                                              ; preds = %.noexc12, %.caught3
  invoke void @__cxa_rethrow()
          to label %380 unwind label %endcatch.pad2

172:                                              ; preds = %160, %152
  invoke void @__cxa_rethrow()
          to label %380 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %172, %171
  %173 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

174:                                              ; No predecessors!
  %175 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %174
  %176 = phi { ptr, i32 } [ %175, %174 ], [ %173, %endcatch.pad2 ]
  %177 = extractvalue { ptr, i32 } %176, 1
  %178 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %179 = icmp eq i32 %177, %178
  %180 = extractvalue { ptr, i32 } %176, 1
  %181 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %182 = icmp eq i32 %180, %181
  %183 = extractvalue { ptr, i32 } %176, 1
  %184 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %185 = icmp eq i32 %183, %184
  %186 = extractvalue { ptr, i32 } %176, 1
  %187 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %188 = icmp eq i32 %186, %187
  %189 = or i1 %179, %182
  %190 = or i1 %189, %185
  %191 = or i1 %190, %188
  br i1 %191, label %195, label %192

192:                                              ; preds = %.caught1
  %193 = extractvalue { ptr, i32 } %176, 0
  %194 = call ptr @__cxa_begin_catch(ptr %193)
  invoke void @__cxa_rethrow()
          to label %380 unwind label %endcatch.pad

195:                                              ; preds = %.caught1
  %196 = extractvalue { ptr, i32 } %176, 0
  %197 = call ptr @__cxa_begin_catch(ptr %196)
  %198 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc23 unwind label %endcatch.resume11

.noexc23:                                         ; preds = %195
  %199 = icmp eq ptr %198, @_ZTIi
  br i1 %199, label %200, label %203

200:                                              ; preds = %.noexc23
  %201 = load i32, ptr %197, align 4
  %202 = sitofp i32 %201 to double
  br label %221

203:                                              ; preds = %.noexc23
  %204 = icmp eq ptr %198, @_ZTIj
  br i1 %204, label %205, label %208

205:                                              ; preds = %203
  %206 = load i32, ptr %197, align 4
  %207 = uitofp i32 %206 to double
  br label %218

208:                                              ; preds = %203
  %209 = icmp eq ptr %198, @_ZTIf
  br i1 %209, label %210, label %213

210:                                              ; preds = %208
  %211 = load float, ptr %197, align 4
  %212 = fpext float %211 to double
  br label %215

213:                                              ; preds = %208
  %214 = load double, ptr %197, align 8
  br label %215

215:                                              ; preds = %210, %213
  %216 = phi double [ %214, %213 ], [ %212, %210 ]
  br label %217

217:                                              ; preds = %215
  br label %218

218:                                              ; preds = %205, %217
  %219 = phi double [ %216, %217 ], [ %207, %205 ]
  br label %220

220:                                              ; preds = %218
  br label %221

221:                                              ; preds = %200, %220
  %222 = phi double [ %219, %220 ], [ %202, %200 ]
  br label %223

223:                                              ; preds = %221
  store double %222, ptr %4, align 8
  %224 = load i32, ptr @wrong, align 4
  %225 = add i32 %224, 1
  store i32 %225, ptr @wrong, align 4
  call void @__cxa_end_catch()
  %226 = load i32, ptr @finals, align 4
  %227 = add i32 %226, 1
  store i32 %227, ptr @finals, align 4
  call void @__cxa_end_catch()
  br label %379

endcatch.pad:                                     ; preds = %192
  %228 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

229:                                              ; No predecessors!
  %230 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %229
  %231 = phi { ptr, i32 } [ %230, %229 ], [ %228, %endcatch.pad ]
  %232 = extractvalue { ptr, i32 } %231, 0
  %233 = call ptr @__cxa_begin_catch(ptr %232)
  %234 = load i32, ptr @finals, align 4
  %235 = add i32 %234, 1
  store i32 %235, ptr @finals, align 4
  invoke void @__cxa_rethrow()
          to label %380 unwind label %endcatch.pad6

endcatch.pad6:                                    ; preds = %.caught
  %236 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught5

237:                                              ; No predecessors!
  %238 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught5

.caught5:                                         ; preds = %endcatch.pad6, %237
  %239 = phi { ptr, i32 } [ %238, %237 ], [ %236, %endcatch.pad6 ]
  %240 = extractvalue { ptr, i32 } %239, 0
  %241 = call ptr @__cxa_begin_catch(ptr %240)
  %242 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %.caught5
  %243 = icmp eq ptr %242, @_ZTIi
  br i1 %243, label %244, label %251

244:                                              ; preds = %.noexc26
  %245 = load i32, ptr %241, align 4
  %246 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %244
  %247 = getelementptr i8, ptr %246, i64 8
  %248 = getelementptr { i64, ptr, i32 }, ptr %247, i32 0, i32 0
  store i64 4, ptr %248, align 8
  %249 = getelementptr { i64, ptr, i32 }, ptr %247, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %249, align 8
  %250 = getelementptr { i64, ptr, i32 }, ptr %247, i32 0, i32 2
  store i32 %245, ptr %250, align 4
  br label %374

251:                                              ; preds = %.noexc26
  %252 = icmp eq ptr %242, @_ZTIj
  br i1 %252, label %253, label %260

253:                                              ; preds = %251
  %254 = load i32, ptr %241, align 4
  %255 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %253
  %256 = getelementptr i8, ptr %255, i64 8
  %257 = getelementptr { i64, ptr, i32 }, ptr %256, i32 0, i32 0
  store i64 4, ptr %257, align 8
  %258 = getelementptr { i64, ptr, i32 }, ptr %256, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %258, align 8
  %259 = getelementptr { i64, ptr, i32 }, ptr %256, i32 0, i32 2
  store i32 %254, ptr %259, align 4
  br label %371

260:                                              ; preds = %251
  %261 = icmp eq ptr %242, @_ZTId
  br i1 %261, label %262, label %269

262:                                              ; preds = %260
  %263 = load double, ptr %241, align 8
  %264 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc34 unwind label %endcatch.resume

.noexc34:                                         ; preds = %262
  %265 = getelementptr i8, ptr %264, i64 8
  %266 = getelementptr { i64, ptr, double }, ptr %265, i32 0, i32 0
  store i64 8, ptr %266, align 8
  %267 = getelementptr { i64, ptr, double }, ptr %265, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %267, align 8
  %268 = getelementptr { i64, ptr, double }, ptr %265, i32 0, i32 2
  store double %263, ptr %268, align 8
  br label %368

269:                                              ; preds = %260
  %270 = icmp eq ptr %242, @_ZTIf
  br i1 %270, label %271, label %278

271:                                              ; preds = %269
  %272 = load float, ptr %241, align 4
  %273 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc31 unwind label %endcatch.resume

.noexc31:                                         ; preds = %271
  %274 = getelementptr i8, ptr %273, i64 8
  %275 = getelementptr { i64, ptr, float }, ptr %274, i32 0, i32 0
  store i64 4, ptr %275, align 8
  %276 = getelementptr { i64, ptr, float }, ptr %274, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %276, align 8
  %277 = getelementptr { i64, ptr, float }, ptr %274, i32 0, i32 2
  store float %272, ptr %277, align 4
  br label %365

278:                                              ; preds = %269
  %279 = icmp eq ptr %242, @_ZTIb
  br i1 %279, label %280, label %287

280:                                              ; preds = %278
  %281 = load i1, ptr %241, align 1
  %282 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %280
  %283 = getelementptr i8, ptr %282, i64 8
  %284 = getelementptr { i64, ptr, i1 }, ptr %283, i32 0, i32 0
  store i64 1, ptr %284, align 8
  %285 = getelementptr { i64, ptr, i1 }, ptr %283, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %285, align 8
  %286 = getelementptr { i64, ptr, i1 }, ptr %283, i32 0, i32 2
  store i1 %281, ptr %286, align 1
  br label %362

287:                                              ; preds = %278
  %288 = icmp eq ptr %242, @_ZTIx
  br i1 %288, label %289, label %296

289:                                              ; preds = %287
  %290 = load i64, ptr %241, align 8
  %291 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %289
  %292 = getelementptr i8, ptr %291, i64 8
  %293 = getelementptr { i64, ptr, i64 }, ptr %292, i32 0, i32 0
  store i64 8, ptr %293, align 8
  %294 = getelementptr { i64, ptr, i64 }, ptr %292, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %294, align 8
  %295 = getelementptr { i64, ptr, i64 }, ptr %292, i32 0, i32 2
  store i64 %290, ptr %295, align 8
  br label %359

296:                                              ; preds = %287
  %297 = icmp eq ptr %242, @_ZTIPc
  br i1 %297, label %298, label %305

298:                                              ; preds = %296
  %299 = load ptr, ptr %241, align 8
  %300 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %298
  %301 = getelementptr i8, ptr %300, i64 8
  %302 = getelementptr { i64, ptr, ptr }, ptr %301, i32 0, i32 0
  store i64 8, ptr %302, align 8
  %303 = getelementptr { i64, ptr, ptr }, ptr %301, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %303, align 8
  %304 = getelementptr { i64, ptr, ptr }, ptr %301, i32 0, i32 2
  store ptr %299, ptr %304, align 8
  br label %356

305:                                              ; preds = %296
  %306 = icmp eq ptr %242, @_ZTIPv
  br i1 %306, label %307, label %309

307:                                              ; preds = %305
  %308 = load ptr, ptr %241, align 8
  br label %353

309:                                              ; preds = %305
  %310 = load ptr, ptr %242, align 8
  %311 = icmp eq ptr %310, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %311, label %312, label %343

312:                                              ; preds = %309
  %313 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %242, i32 0, i32 2
  %314 = load i32, ptr %313, align 4
  %315 = and i32 %314, 1073741824
  %316 = icmp ne i32 %315, 0
  br i1 %316, label %317, label %333

317:                                              ; preds = %312
  %318 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %242, i32 0, i32 4
  %319 = load ptr, ptr %318, align 8
  %320 = icmp ne ptr %319, null
  br i1 %320, label %321, label %323

321:                                              ; preds = %317
  invoke void %319(ptr %1, ptr %241)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %321
  %322 = load ptr, ptr %1, align 8
  br label %330

323:                                              ; preds = %317
  %324 = load ptr, ptr %241, align 8
  %325 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %323
  %326 = getelementptr i8, ptr %325, i64 8
  %327 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 0
  store i64 8, ptr %327, align 8
  %328 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %328, align 8
  %329 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 2
  store ptr %324, ptr %329, align 8
  br label %330

330:                                              ; preds = %.noexc20, %.noexc22
  %331 = phi ptr [ %326, %.noexc22 ], [ %322, %.noexc20 ]
  br label %332

332:                                              ; preds = %330
  br label %340

333:                                              ; preds = %312
  %334 = load ptr, ptr %241, align 8
  %335 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %333
  %336 = getelementptr i8, ptr %335, i64 8
  %337 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 0
  store i64 8, ptr %337, align 8
  %338 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %338, align 8
  %339 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 2
  store ptr %334, ptr %339, align 8
  br label %340

340:                                              ; preds = %332, %.noexc
  %341 = phi ptr [ %336, %.noexc ], [ %331, %332 ]
  br label %342

342:                                              ; preds = %340
  br label %350

343:                                              ; preds = %309
  %344 = load ptr, ptr %241, align 8
  %345 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %343
  %346 = getelementptr i8, ptr %345, i64 8
  %347 = getelementptr { i64, ptr, ptr }, ptr %346, i32 0, i32 0
  store i64 8, ptr %347, align 8
  %348 = getelementptr { i64, ptr, ptr }, ptr %346, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %348, align 8
  %349 = getelementptr { i64, ptr, ptr }, ptr %346, i32 0, i32 2
  store ptr %344, ptr %349, align 8
  br label %350

350:                                              ; preds = %342, %.noexc10
  %351 = phi ptr [ %346, %.noexc10 ], [ %341, %342 ]
  br label %352

352:                                              ; preds = %350
  br label %353

353:                                              ; preds = %307, %352
  %354 = phi ptr [ %351, %352 ], [ %308, %307 ]
  br label %355

355:                                              ; preds = %353
  br label %356

356:                                              ; preds = %.noexc28, %355
  %357 = phi ptr [ %354, %355 ], [ %301, %.noexc28 ]
  br label %358

358:                                              ; preds = %356
  br label %359

359:                                              ; preds = %.noexc15, %358
  %360 = phi ptr [ %357, %358 ], [ %292, %.noexc15 ]
  br label %361

361:                                              ; preds = %359
  br label %362

362:                                              ; preds = %.noexc9, %361
  %363 = phi ptr [ %360, %361 ], [ %283, %.noexc9 ]
  br label %364

364:                                              ; preds = %362
  br label %365

365:                                              ; preds = %.noexc31, %364
  %366 = phi ptr [ %363, %364 ], [ %274, %.noexc31 ]
  br label %367

367:                                              ; preds = %365
  br label %368

368:                                              ; preds = %.noexc34, %367
  %369 = phi ptr [ %366, %367 ], [ %265, %.noexc34 ]
  br label %370

370:                                              ; preds = %368
  br label %371

371:                                              ; preds = %.noexc24, %370
  %372 = phi ptr [ %369, %370 ], [ %256, %.noexc24 ]
  br label %373

373:                                              ; preds = %371
  br label %374

374:                                              ; preds = %.noexc30, %373
  %375 = phi ptr [ %372, %373 ], [ %247, %.noexc30 ]
  br label %376

376:                                              ; preds = %374
  store ptr %375, ptr %7, align 8
  %377 = load i32, ptr @outer, align 4
  %378 = add i32 %377, 1
  store i32 %378, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %379

379:                                              ; preds = %223, %376
  ret void

380:                                              ; preds = %.caught, %192, %172, %171, %.noexc8, %0
  unreachable

endcatch.resume:                                  ; preds = %262, %94, %92, %271, %244, %69, %298, %15, %.caught5, %24, %253, %323, %33, %321, %114, %60, %104, %51, %289, %42, %9, %343, %280, %.noexc7, %147, %333
  %381 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %381

endcatch.resume11:                                ; preds = %195, %168
  %382 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %382
}

; Function Attrs: mustprogress
define void @noTryAround() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %231 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  %11 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %7
  %12 = icmp eq ptr %11, @_ZTIi
  br i1 %12, label %13, label %20

13:                                               ; preds = %.noexc17
  %14 = load i32, ptr %10, align 4
  %15 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %13
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 0
  store i64 4, ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 2
  store i32 %14, ptr %19, align 4
  br label %143

20:                                               ; preds = %.noexc17
  %21 = icmp eq ptr %11, @_ZTIj
  br i1 %21, label %22, label %29

22:                                               ; preds = %20
  %23 = load i32, ptr %10, align 4
  %24 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %22
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 0
  store i64 4, ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %27, align 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 2
  store i32 %23, ptr %28, align 4
  br label %140

29:                                               ; preds = %20
  %30 = icmp eq ptr %11, @_ZTId
  br i1 %30, label %31, label %38

31:                                               ; preds = %29
  %32 = load double, ptr %10, align 8
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 0
  store i64 8, ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 2
  store double %32, ptr %37, align 8
  br label %137

38:                                               ; preds = %29
  %39 = icmp eq ptr %11, @_ZTIf
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load float, ptr %10, align 4
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 0
  store i64 4, ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 2
  store float %41, ptr %46, align 4
  br label %134

47:                                               ; preds = %38
  %48 = icmp eq ptr %11, @_ZTIb
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load i1, ptr %10, align 1
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 0
  store i64 1, ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 2
  store i1 %50, ptr %55, align 1
  br label %131

56:                                               ; preds = %47
  %57 = icmp eq ptr %11, @_ZTIx
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load i64, ptr %10, align 8
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 0
  store i64 8, ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 2
  store i64 %59, ptr %64, align 8
  br label %128

65:                                               ; preds = %56
  %66 = icmp eq ptr %11, @_ZTIPc
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load ptr, ptr %10, align 8
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 0
  store i64 8, ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 2
  store ptr %68, ptr %73, align 8
  br label %125

74:                                               ; preds = %65
  %75 = icmp eq ptr %11, @_ZTIPv
  br i1 %75, label %76, label %78

76:                                               ; preds = %74
  %77 = load ptr, ptr %10, align 8
  br label %122

78:                                               ; preds = %74
  %79 = load ptr, ptr %11, align 8
  %80 = icmp eq ptr %79, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %80, label %81, label %112

81:                                               ; preds = %78
  %82 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %11, i32 0, i32 2
  %83 = load i32, ptr %82, align 4
  %84 = and i32 %83, 1073741824
  %85 = icmp ne i32 %84, 0
  br i1 %85, label %86, label %102

86:                                               ; preds = %81
  %87 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %11, i32 0, i32 4
  %88 = load ptr, ptr %87, align 8
  %89 = icmp ne ptr %88, null
  br i1 %89, label %90, label %92

90:                                               ; preds = %86
  invoke void %88(ptr %4, ptr %10)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %90
  %91 = load ptr, ptr %4, align 8
  br label %99

92:                                               ; preds = %86
  %93 = load ptr, ptr %10, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %99

99:                                               ; preds = %.noexc16, %.noexc3
  %100 = phi ptr [ %95, %.noexc3 ], [ %91, %.noexc16 ]
  br label %101

101:                                              ; preds = %99
  br label %109

102:                                              ; preds = %81
  %103 = load ptr, ptr %10, align 8
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 0
  store i64 8, ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 2
  store ptr %103, ptr %108, align 8
  br label %109

109:                                              ; preds = %101, %.noexc13
  %110 = phi ptr [ %105, %.noexc13 ], [ %100, %101 ]
  br label %111

111:                                              ; preds = %109
  br label %119

112:                                              ; preds = %78
  %113 = load ptr, ptr %10, align 8
  %114 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %112
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 0
  store i64 8, ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 2
  store ptr %113, ptr %118, align 8
  br label %119

119:                                              ; preds = %111, %.noexc9
  %120 = phi ptr [ %115, %.noexc9 ], [ %110, %111 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %76, %121
  %123 = phi ptr [ %120, %121 ], [ %77, %76 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc12, %124
  %126 = phi ptr [ %123, %124 ], [ %70, %.noexc12 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc11, %127
  %129 = phi ptr [ %126, %127 ], [ %61, %.noexc11 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc5, %130
  %132 = phi ptr [ %129, %130 ], [ %52, %.noexc5 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc, %133
  %135 = phi ptr [ %132, %133 ], [ %43, %.noexc ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc18, %136
  %138 = phi ptr [ %135, %136 ], [ %34, %.noexc18 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc15, %139
  %141 = phi ptr [ %138, %139 ], [ %25, %.noexc15 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc4, %142
  %144 = phi ptr [ %141, %142 ], [ %16, %.noexc4 ]
  br label %145

145:                                              ; preds = %143
  store ptr %144, ptr %5, align 8
  store i1 false, ptr %1, align 1
  %146 = invoke ptr @GC_malloc(i64 16)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %145
  %147 = getelementptr i8, ptr %146, i64 8
  %148 = getelementptr %Res, ptr %147, i32 0, i32 0
  store ptr @Res..vtbl, ptr %148, align 8
  store ptr %147, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %149 = load i32, ptr @finals, align 4
  %150 = add i32 %149, 1
  store i32 %150, ptr @finals, align 4
  %151 = load i1, ptr %1, align 1
  br i1 %151, label %152, label %230

152:                                              ; preds = %.noexc8
  store i1 false, ptr %1, align 1
  %153 = load ptr, ptr %2, align 8
  %154 = getelementptr %Res, ptr %153, i32 0, i32 0
  invoke void @Res.dispose(ptr %153)
          to label %155 unwind label %156

155:                                              ; preds = %152
  br label %230

156:                                              ; preds = %152
  %157 = landingpad { ptr, i32 }
          catch ptr null
  %158 = extractvalue { ptr, i32 } %157, 0
  %159 = call ptr @__cxa_begin_catch(ptr %158)
  %160 = load i1, ptr %1, align 1
  br i1 %160, label %161, label %176

161:                                              ; preds = %156
  store i1 false, ptr %1, align 1
  %162 = load ptr, ptr %2, align 8
  %163 = getelementptr %Res, ptr %162, i32 0, i32 0
  invoke void @Res.dispose(ptr %162)
          to label %164 unwind label %endcatch.pad

164:                                              ; preds = %161
  br label %176

endcatch.pad:                                     ; preds = %161
  %165 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

166:                                              ; No predecessors!
  %167 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %166
  %168 = phi { ptr, i32 } [ %167, %166 ], [ %165, %endcatch.pad ]
  %169 = extractvalue { ptr, i32 } %168, 0
  %170 = call ptr @__cxa_begin_catch(ptr %169)
  %171 = load i1, ptr %1, align 1
  br i1 %171, label %172, label %175

172:                                              ; preds = %.caught
  store i1 false, ptr %1, align 1
  %173 = load ptr, ptr %2, align 8
  %174 = getelementptr %Res, ptr %173, i32 0, i32 0
  invoke void @Res.dispose(ptr %173)
          to label %.noexc14 unwind label %endcatch.resume6

.noexc14:                                         ; preds = %172
  br label %175

175:                                              ; preds = %.noexc14, %.caught
  invoke void @__cxa_rethrow()
          to label %231 unwind label %endcatch.pad2

176:                                              ; preds = %164, %156
  invoke void @__cxa_rethrow()
          to label %231 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %175, %176
  %177 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  call void @__cxa_end_catch()
  br label %.caught1

178:                                              ; No predecessors!
  %179 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %178
  %180 = phi { ptr, i32 } [ %179, %178 ], [ %177, %endcatch.pad2 ]
  %181 = extractvalue { ptr, i32 } %180, 1
  %182 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %183 = icmp eq i32 %181, %182
  %184 = extractvalue { ptr, i32 } %180, 1
  %185 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %186 = icmp eq i32 %184, %185
  %187 = extractvalue { ptr, i32 } %180, 1
  %188 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %189 = icmp eq i32 %187, %188
  %190 = extractvalue { ptr, i32 } %180, 1
  %191 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %192 = icmp eq i32 %190, %191
  %193 = or i1 %183, %186
  %194 = or i1 %193, %189
  %195 = or i1 %194, %192
  br i1 %195, label %199, label %196

196:                                              ; preds = %.caught1
  %197 = extractvalue { ptr, i32 } %180, 0
  %198 = call ptr @__cxa_begin_catch(ptr %197)
  invoke void @__cxa_rethrow()
          to label %.noexc7 unwind label %endcatch.resume6

.noexc7:                                          ; preds = %196
  br label %231

199:                                              ; preds = %.caught1
  %200 = extractvalue { ptr, i32 } %180, 0
  %201 = call ptr @__cxa_begin_catch(ptr %200)
  %202 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume6

.noexc10:                                         ; preds = %199
  %203 = icmp eq ptr %202, @_ZTIi
  br i1 %203, label %204, label %207

204:                                              ; preds = %.noexc10
  %205 = load i32, ptr %201, align 4
  %206 = sitofp i32 %205 to double
  br label %225

207:                                              ; preds = %.noexc10
  %208 = icmp eq ptr %202, @_ZTIj
  br i1 %208, label %209, label %212

209:                                              ; preds = %207
  %210 = load i32, ptr %201, align 4
  %211 = uitofp i32 %210 to double
  br label %222

212:                                              ; preds = %207
  %213 = icmp eq ptr %202, @_ZTIf
  br i1 %213, label %214, label %217

214:                                              ; preds = %212
  %215 = load float, ptr %201, align 4
  %216 = fpext float %215 to double
  br label %219

217:                                              ; preds = %212
  %218 = load double, ptr %201, align 8
  br label %219

219:                                              ; preds = %214, %217
  %220 = phi double [ %218, %217 ], [ %216, %214 ]
  br label %221

221:                                              ; preds = %219
  br label %222

222:                                              ; preds = %209, %221
  %223 = phi double [ %220, %221 ], [ %211, %209 ]
  br label %224

224:                                              ; preds = %222
  br label %225

225:                                              ; preds = %204, %224
  %226 = phi double [ %223, %224 ], [ %206, %204 ]
  br label %227

227:                                              ; preds = %225
  store double %226, ptr %3, align 8
  %228 = load i32, ptr @wrong, align 4
  %229 = add i32 %228, 1
  store i32 %229, ptr @wrong, align 4
  call void @__cxa_end_catch()
  br label %230

230:                                              ; preds = %155, %227, %.noexc8
  call void @__cxa_end_catch()
  ret void

231:                                              ; preds = %.noexc7, %176, %175, %0
  unreachable

endcatch.resume:                                  ; preds = %31, %7, %90, %22, %102, %67, %58, %112, %145, %49, %13, %92, %40
  %232 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %232

endcatch.resume6:                                 ; preds = %172, %199, %196
  %233 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %233
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @reset()
  call void @withUsing()
  %1 = load i32, ptr @disposed, align 4
  %2 = icmp eq i32 %1, 1
  br i1 %2, label %3, label %6

3:                                                ; preds = %0
  %4 = load i32, ptr @wrong, align 4
  %5 = icmp eq i32 %4, 0
  br label %6

6:                                                ; preds = %3, %0
  %7 = phi i1 [ %5, %3 ], [ %2, %0 ]
  br i1 %7, label %8, label %11

8:                                                ; preds = %6
  %9 = load i32, ptr @outer, align 4
  %10 = icmp eq i32 %9, 1
  br label %11

11:                                               ; preds = %8, %6
  %12 = phi i1 [ %10, %8 ], [ %7, %6 ]
  br i1 %12, label %13, label %46

13:                                               ; preds = %11
  call void @reset()
  call void @withUsingAndFinally()
  %14 = load i32, ptr @disposed, align 4
  %15 = icmp eq i32 %14, 1
  br i1 %15, label %16, label %19

16:                                               ; preds = %13
  %17 = load i32, ptr @finals, align 4
  %18 = icmp eq i32 %17, 1
  br label %19

19:                                               ; preds = %16, %13
  %20 = phi i1 [ %18, %16 ], [ %15, %13 ]
  br i1 %20, label %21, label %24

21:                                               ; preds = %19
  %22 = load i32, ptr @wrong, align 4
  %23 = icmp eq i32 %22, 0
  br label %24

24:                                               ; preds = %21, %19
  %25 = phi i1 [ %23, %21 ], [ %20, %19 ]
  br i1 %25, label %26, label %29

26:                                               ; preds = %24
  %27 = load i32, ptr @outer, align 4
  %28 = icmp eq i32 %27, 1
  br label %29

29:                                               ; preds = %26, %24
  %30 = phi i1 [ %28, %26 ], [ %25, %24 ]
  br i1 %30, label %31, label %48

31:                                               ; preds = %29
  call void @reset()
  call void @noTryAround()
  %32 = load i32, ptr @disposed, align 4
  %33 = icmp eq i32 %32, 1
  br i1 %33, label %34, label %37

34:                                               ; preds = %31
  %35 = load i32, ptr @finals, align 4
  %36 = icmp eq i32 %35, 1
  br label %37

37:                                               ; preds = %34, %31
  %38 = phi i1 [ %36, %34 ], [ %33, %31 ]
  br i1 %38, label %39, label %42

39:                                               ; preds = %37
  %40 = load i32, ptr @wrong, align 4
  %41 = icmp eq i32 %40, 0
  br label %42

42:                                               ; preds = %39, %37
  %43 = phi i1 [ %41, %39 ], [ %38, %37 ]
  br i1 %43, label %44, label %50

44:                                               ; preds = %42
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

45:                                               ; preds = %50, %48, %46
  unreachable

46:                                               ; preds = %11
  %47 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7464751892813498902, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15081112902223709724, i64 8), i32 80, ptr null)
  br label %45

48:                                               ; preds = %29
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_441435409882857575, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15081112902223709724, i64 8), i32 84, ptr null)
  br label %45

50:                                               ; preds = %42
  %51 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17909071614601602156, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15081112902223709724, i64 8), i32 88, ptr null)
  br label %45
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
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

