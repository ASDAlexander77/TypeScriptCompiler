; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_15619039406129020236 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin an untyped catch clause\00", align 8
@f_5027590903802435426 = internal constant [120 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_mismatch_in_catch_clause.ts\00", align 8
@m_4788699618267080347 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin a typed catch clause\00", align 8
@m_2596540853700611608 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwith its own finally\00", align 8
@m_4837289883065086718 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin the parent's finally\00", align 8
@m_7059949095878146050 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo levels up, through a finally\00", align 8
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

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

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
define void @inUntypedCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %342 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  %11 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %7
  %12 = icmp eq ptr %11, @_ZTIi
  br i1 %12, label %13, label %20

13:                                               ; preds = %.noexc12
  %14 = load i32, ptr %10, align 4
  %15 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %13
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 0
  store i64 4, ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 2
  store i32 %14, ptr %19, align 4
  br label %143

20:                                               ; preds = %.noexc12
  %21 = icmp eq ptr %11, @_ZTIj
  br i1 %21, label %22, label %29

22:                                               ; preds = %20
  %23 = load i32, ptr %10, align 4
  %24 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %22
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %31
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
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %40
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
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %49
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
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %58
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
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %67
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
  invoke void %88(ptr %3, ptr %10)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %90
  %91 = load ptr, ptr %3, align 8
  br label %99

92:                                               ; preds = %86
  %93 = load ptr, ptr %10, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %99

99:                                               ; preds = %.noexc26, %.noexc14
  %100 = phi ptr [ %95, %.noexc14 ], [ %91, %.noexc26 ]
  br label %101

101:                                              ; preds = %99
  br label %109

102:                                              ; preds = %81
  %103 = load ptr, ptr %10, align 8
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 0
  store i64 8, ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 2
  store ptr %103, ptr %108, align 8
  br label %109

109:                                              ; preds = %101, %.noexc24
  %110 = phi ptr [ %105, %.noexc24 ], [ %100, %101 ]
  br label %111

111:                                              ; preds = %109
  br label %119

112:                                              ; preds = %78
  %113 = load ptr, ptr %10, align 8
  %114 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %112
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 0
  store i64 8, ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 2
  store ptr %113, ptr %118, align 8
  br label %119

119:                                              ; preds = %111, %.noexc17
  %120 = phi ptr [ %115, %.noexc17 ], [ %110, %111 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %76, %121
  %123 = phi ptr [ %120, %121 ], [ %77, %76 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc19, %124
  %126 = phi ptr [ %123, %124 ], [ %70, %.noexc19 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc21, %127
  %129 = phi ptr [ %126, %127 ], [ %61, %.noexc21 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc15, %130
  %132 = phi ptr [ %129, %130 ], [ %52, %.noexc15 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc13, %133
  %135 = phi ptr [ %132, %133 ], [ %43, %.noexc13 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc7, %136
  %138 = phi ptr [ %135, %136 ], [ %34, %.noexc7 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc4, %139
  %141 = phi ptr [ %138, %139 ], [ %25, %.noexc4 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc10, %142
  %144 = phi ptr [ %141, %142 ], [ %16, %.noexc10 ]
  br label %145

145:                                              ; preds = %143
  store ptr %144, ptr %4, align 8
  %146 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %145
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %146, align 8
  invoke void @__cxa_throw(ptr %146, ptr @_ZTIPc, ptr null)
          to label %342 unwind label %147

147:                                              ; preds = %.noexc3
  %148 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %149 = extractvalue { ptr, i32 } %148, 1
  %150 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %151 = icmp eq i32 %149, %150
  %152 = extractvalue { ptr, i32 } %148, 1
  %153 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %154 = icmp eq i32 %152, %153
  %155 = extractvalue { ptr, i32 } %148, 1
  %156 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %157 = icmp eq i32 %155, %156
  %158 = extractvalue { ptr, i32 } %148, 1
  %159 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %160 = icmp eq i32 %158, %159
  %161 = or i1 %151, %154
  %162 = or i1 %161, %157
  %163 = or i1 %162, %160
  br i1 %163, label %167, label %164

164:                                              ; preds = %147
  %165 = extractvalue { ptr, i32 } %148, 0
  %166 = call ptr @__cxa_begin_catch(ptr %165)
  invoke void @__cxa_rethrow()
          to label %342 unwind label %endcatch.pad

167:                                              ; preds = %147
  %168 = extractvalue { ptr, i32 } %148, 0
  %169 = call ptr @__cxa_begin_catch(ptr %168)
  %170 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume8

.noexc9:                                          ; preds = %167
  %171 = icmp eq ptr %170, @_ZTIi
  br i1 %171, label %172, label %175

172:                                              ; preds = %.noexc9
  %173 = load i32, ptr %169, align 4
  %174 = sitofp i32 %173 to double
  br label %193

175:                                              ; preds = %.noexc9
  %176 = icmp eq ptr %170, @_ZTIj
  br i1 %176, label %177, label %180

177:                                              ; preds = %175
  %178 = load i32, ptr %169, align 4
  %179 = uitofp i32 %178 to double
  br label %190

180:                                              ; preds = %175
  %181 = icmp eq ptr %170, @_ZTIf
  br i1 %181, label %182, label %185

182:                                              ; preds = %180
  %183 = load float, ptr %169, align 4
  %184 = fpext float %183 to double
  br label %187

185:                                              ; preds = %180
  %186 = load double, ptr %169, align 8
  br label %187

187:                                              ; preds = %182, %185
  %188 = phi double [ %186, %185 ], [ %184, %182 ]
  br label %189

189:                                              ; preds = %187
  br label %190

190:                                              ; preds = %177, %189
  %191 = phi double [ %188, %189 ], [ %179, %177 ]
  br label %192

192:                                              ; preds = %190
  br label %193

193:                                              ; preds = %172, %192
  %194 = phi double [ %191, %192 ], [ %174, %172 ]
  br label %195

195:                                              ; preds = %193
  store double %194, ptr %2, align 8
  %196 = load i32, ptr @wrong, align 4
  %197 = add i32 %196, 1
  store i32 %197, ptr @wrong, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %341

endcatch.pad:                                     ; preds = %164
  %198 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

199:                                              ; No predecessors!
  %200 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %199
  %201 = phi { ptr, i32 } [ %200, %199 ], [ %198, %endcatch.pad ]
  %202 = extractvalue { ptr, i32 } %201, 0
  %203 = call ptr @__cxa_begin_catch(ptr %202)
  %204 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %.caught
  %205 = icmp eq ptr %204, @_ZTIi
  br i1 %205, label %206, label %213

206:                                              ; preds = %.noexc20
  %207 = load i32, ptr %203, align 4
  %208 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %206
  %209 = getelementptr i8, ptr %208, i64 8
  %210 = getelementptr { i64, ptr, i32 }, ptr %209, i32 0, i32 0
  store i64 4, ptr %210, align 8
  %211 = getelementptr { i64, ptr, i32 }, ptr %209, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %211, align 8
  %212 = getelementptr { i64, ptr, i32 }, ptr %209, i32 0, i32 2
  store i32 %207, ptr %212, align 4
  br label %336

213:                                              ; preds = %.noexc20
  %214 = icmp eq ptr %204, @_ZTIj
  br i1 %214, label %215, label %222

215:                                              ; preds = %213
  %216 = load i32, ptr %203, align 4
  %217 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %215
  %218 = getelementptr i8, ptr %217, i64 8
  %219 = getelementptr { i64, ptr, i32 }, ptr %218, i32 0, i32 0
  store i64 4, ptr %219, align 8
  %220 = getelementptr { i64, ptr, i32 }, ptr %218, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %220, align 8
  %221 = getelementptr { i64, ptr, i32 }, ptr %218, i32 0, i32 2
  store i32 %216, ptr %221, align 4
  br label %333

222:                                              ; preds = %213
  %223 = icmp eq ptr %204, @_ZTId
  br i1 %223, label %224, label %231

224:                                              ; preds = %222
  %225 = load double, ptr %203, align 8
  %226 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %224
  %227 = getelementptr i8, ptr %226, i64 8
  %228 = getelementptr { i64, ptr, double }, ptr %227, i32 0, i32 0
  store i64 8, ptr %228, align 8
  %229 = getelementptr { i64, ptr, double }, ptr %227, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %229, align 8
  %230 = getelementptr { i64, ptr, double }, ptr %227, i32 0, i32 2
  store double %225, ptr %230, align 8
  br label %330

231:                                              ; preds = %222
  %232 = icmp eq ptr %204, @_ZTIf
  br i1 %232, label %233, label %240

233:                                              ; preds = %231
  %234 = load float, ptr %203, align 4
  %235 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %233
  %236 = getelementptr i8, ptr %235, i64 8
  %237 = getelementptr { i64, ptr, float }, ptr %236, i32 0, i32 0
  store i64 4, ptr %237, align 8
  %238 = getelementptr { i64, ptr, float }, ptr %236, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %238, align 8
  %239 = getelementptr { i64, ptr, float }, ptr %236, i32 0, i32 2
  store float %234, ptr %239, align 4
  br label %327

240:                                              ; preds = %231
  %241 = icmp eq ptr %204, @_ZTIb
  br i1 %241, label %242, label %249

242:                                              ; preds = %240
  %243 = load i1, ptr %203, align 1
  %244 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %242
  %245 = getelementptr i8, ptr %244, i64 8
  %246 = getelementptr { i64, ptr, i1 }, ptr %245, i32 0, i32 0
  store i64 1, ptr %246, align 8
  %247 = getelementptr { i64, ptr, i1 }, ptr %245, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %247, align 8
  %248 = getelementptr { i64, ptr, i1 }, ptr %245, i32 0, i32 2
  store i1 %243, ptr %248, align 1
  br label %324

249:                                              ; preds = %240
  %250 = icmp eq ptr %204, @_ZTIx
  br i1 %250, label %251, label %258

251:                                              ; preds = %249
  %252 = load i64, ptr %203, align 8
  %253 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %251
  %254 = getelementptr i8, ptr %253, i64 8
  %255 = getelementptr { i64, ptr, i64 }, ptr %254, i32 0, i32 0
  store i64 8, ptr %255, align 8
  %256 = getelementptr { i64, ptr, i64 }, ptr %254, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %256, align 8
  %257 = getelementptr { i64, ptr, i64 }, ptr %254, i32 0, i32 2
  store i64 %252, ptr %257, align 8
  br label %321

258:                                              ; preds = %249
  %259 = icmp eq ptr %204, @_ZTIPc
  br i1 %259, label %260, label %267

260:                                              ; preds = %258
  %261 = load ptr, ptr %203, align 8
  %262 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %260
  %263 = getelementptr i8, ptr %262, i64 8
  %264 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 0
  store i64 8, ptr %264, align 8
  %265 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %265, align 8
  %266 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 2
  store ptr %261, ptr %266, align 8
  br label %318

267:                                              ; preds = %258
  %268 = icmp eq ptr %204, @_ZTIPv
  br i1 %268, label %269, label %271

269:                                              ; preds = %267
  %270 = load ptr, ptr %203, align 8
  br label %315

271:                                              ; preds = %267
  %272 = load ptr, ptr %204, align 8
  %273 = icmp eq ptr %272, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %273, label %274, label %305

274:                                              ; preds = %271
  %275 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %204, i32 0, i32 2
  %276 = load i32, ptr %275, align 4
  %277 = and i32 %276, 1073741824
  %278 = icmp ne i32 %277, 0
  br i1 %278, label %279, label %295

279:                                              ; preds = %274
  %280 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %204, i32 0, i32 4
  %281 = load ptr, ptr %280, align 8
  %282 = icmp ne ptr %281, null
  br i1 %282, label %283, label %285

283:                                              ; preds = %279
  invoke void %281(ptr %1, ptr %203)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %283
  %284 = load ptr, ptr %1, align 8
  br label %292

285:                                              ; preds = %279
  %286 = load ptr, ptr %203, align 8
  %287 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %285
  %288 = getelementptr i8, ptr %287, i64 8
  %289 = getelementptr { i64, ptr, ptr }, ptr %288, i32 0, i32 0
  store i64 8, ptr %289, align 8
  %290 = getelementptr { i64, ptr, ptr }, ptr %288, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %290, align 8
  %291 = getelementptr { i64, ptr, ptr }, ptr %288, i32 0, i32 2
  store ptr %286, ptr %291, align 8
  br label %292

292:                                              ; preds = %.noexc1, %.noexc6
  %293 = phi ptr [ %288, %.noexc6 ], [ %284, %.noexc1 ]
  br label %294

294:                                              ; preds = %292
  br label %302

295:                                              ; preds = %274
  %296 = load ptr, ptr %203, align 8
  %297 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %295
  %298 = getelementptr i8, ptr %297, i64 8
  %299 = getelementptr { i64, ptr, ptr }, ptr %298, i32 0, i32 0
  store i64 8, ptr %299, align 8
  %300 = getelementptr { i64, ptr, ptr }, ptr %298, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %300, align 8
  %301 = getelementptr { i64, ptr, ptr }, ptr %298, i32 0, i32 2
  store ptr %296, ptr %301, align 8
  br label %302

302:                                              ; preds = %294, %.noexc22
  %303 = phi ptr [ %298, %.noexc22 ], [ %293, %294 ]
  br label %304

304:                                              ; preds = %302
  br label %312

305:                                              ; preds = %271
  %306 = load ptr, ptr %203, align 8
  %307 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %305
  %308 = getelementptr i8, ptr %307, i64 8
  %309 = getelementptr { i64, ptr, ptr }, ptr %308, i32 0, i32 0
  store i64 8, ptr %309, align 8
  %310 = getelementptr { i64, ptr, ptr }, ptr %308, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %310, align 8
  %311 = getelementptr { i64, ptr, ptr }, ptr %308, i32 0, i32 2
  store ptr %306, ptr %311, align 8
  br label %312

312:                                              ; preds = %304, %.noexc18
  %313 = phi ptr [ %308, %.noexc18 ], [ %303, %304 ]
  br label %314

314:                                              ; preds = %312
  br label %315

315:                                              ; preds = %269, %314
  %316 = phi ptr [ %313, %314 ], [ %270, %269 ]
  br label %317

317:                                              ; preds = %315
  br label %318

318:                                              ; preds = %.noexc2, %317
  %319 = phi ptr [ %316, %317 ], [ %263, %.noexc2 ]
  br label %320

320:                                              ; preds = %318
  br label %321

321:                                              ; preds = %.noexc11, %320
  %322 = phi ptr [ %319, %320 ], [ %254, %.noexc11 ]
  br label %323

323:                                              ; preds = %321
  br label %324

324:                                              ; preds = %.noexc, %323
  %325 = phi ptr [ %322, %323 ], [ %245, %.noexc ]
  br label %326

326:                                              ; preds = %324
  br label %327

327:                                              ; preds = %.noexc25, %326
  %328 = phi ptr [ %325, %326 ], [ %236, %.noexc25 ]
  br label %329

329:                                              ; preds = %327
  br label %330

330:                                              ; preds = %.noexc5, %329
  %331 = phi ptr [ %328, %329 ], [ %227, %.noexc5 ]
  br label %332

332:                                              ; preds = %330
  br label %333

333:                                              ; preds = %.noexc23, %332
  %334 = phi ptr [ %331, %332 ], [ %218, %.noexc23 ]
  br label %335

335:                                              ; preds = %333
  br label %336

336:                                              ; preds = %.noexc16, %335
  %337 = phi ptr [ %334, %335 ], [ %209, %.noexc16 ]
  br label %338

338:                                              ; preds = %336
  store ptr %337, ptr %5, align 8
  %339 = load i32, ptr @outer, align 4
  %340 = add i32 %339, 1
  store i32 %340, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %341

341:                                              ; preds = %195, %338
  ret void

342:                                              ; preds = %164, %.noexc3, %0
  unreachable

endcatch.resume:                                  ; preds = %90, %233, %102, %215, %295, %58, %.caught, %67, %305, %112, %206, %49, %92, %40, %7, %251, %13, %31, %285, %224, %22, %145, %260, %283, %242
  %343 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %343

endcatch.resume8:                                 ; preds = %167
  %344 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %344
}

; Function Attrs: mustprogress
define void @inTypedCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %252 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %10 = icmp eq i32 %8, %9
  %11 = extractvalue { ptr, i32 } %7, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %13 = icmp eq i32 %11, %12
  %14 = extractvalue { ptr, i32 } %7, 1
  %15 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %16 = icmp eq i32 %14, %15
  %17 = extractvalue { ptr, i32 } %7, 1
  %18 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %19 = icmp eq i32 %17, %18
  %20 = or i1 %10, %13
  %21 = or i1 %20, %16
  %22 = or i1 %21, %19
  br i1 %22, label %26, label %23

23:                                               ; preds = %6
  %24 = extractvalue { ptr, i32 } %7, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %252 unwind label %endcatch.pad1

26:                                               ; preds = %6
  %27 = extractvalue { ptr, i32 } %7, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %26
  %30 = icmp eq ptr %29, @_ZTIi
  br i1 %30, label %31, label %34

31:                                               ; preds = %.noexc7
  %32 = load i32, ptr %28, align 4
  %33 = sitofp i32 %32 to double
  br label %52

34:                                               ; preds = %.noexc7
  %35 = icmp eq ptr %29, @_ZTIj
  br i1 %35, label %36, label %39

36:                                               ; preds = %34
  %37 = load i32, ptr %28, align 4
  %38 = uitofp i32 %37 to double
  br label %49

39:                                               ; preds = %34
  %40 = icmp eq ptr %29, @_ZTIf
  br i1 %40, label %41, label %44

41:                                               ; preds = %39
  %42 = load float, ptr %28, align 4
  %43 = fpext float %42 to double
  br label %46

44:                                               ; preds = %39
  %45 = load double, ptr %28, align 8
  br label %46

46:                                               ; preds = %41, %44
  %47 = phi double [ %45, %44 ], [ %43, %41 ]
  br label %48

48:                                               ; preds = %46
  br label %49

49:                                               ; preds = %36, %48
  %50 = phi double [ %47, %48 ], [ %38, %36 ]
  br label %51

51:                                               ; preds = %49
  br label %52

52:                                               ; preds = %31, %51
  %53 = phi double [ %50, %51 ], [ %33, %31 ]
  br label %54

54:                                               ; preds = %52
  store double %53, ptr %3, align 8
  %55 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %54
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %55, align 8
  invoke void @__cxa_throw(ptr %55, ptr @_ZTIPc, ptr null)
          to label %252 unwind label %56

56:                                               ; preds = %.noexc12
  %57 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %58 = extractvalue { ptr, i32 } %57, 1
  %59 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %60 = icmp eq i32 %58, %59
  %61 = extractvalue { ptr, i32 } %57, 1
  %62 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %63 = icmp eq i32 %61, %62
  %64 = extractvalue { ptr, i32 } %57, 1
  %65 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %66 = icmp eq i32 %64, %65
  %67 = extractvalue { ptr, i32 } %57, 1
  %68 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %69 = icmp eq i32 %67, %68
  %70 = or i1 %60, %63
  %71 = or i1 %70, %66
  %72 = or i1 %71, %69
  br i1 %72, label %76, label %73

73:                                               ; preds = %56
  %74 = extractvalue { ptr, i32 } %57, 0
  %75 = call ptr @__cxa_begin_catch(ptr %74)
  invoke void @__cxa_rethrow()
          to label %252 unwind label %endcatch.pad

76:                                               ; preds = %56
  %77 = extractvalue { ptr, i32 } %57, 0
  %78 = call ptr @__cxa_begin_catch(ptr %77)
  %79 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc6 unwind label %endcatch.resume5

.noexc6:                                          ; preds = %76
  %80 = icmp eq ptr %79, @_ZTIi
  br i1 %80, label %81, label %84

81:                                               ; preds = %.noexc6
  %82 = load i32, ptr %78, align 4
  %83 = sitofp i32 %82 to double
  br label %102

84:                                               ; preds = %.noexc6
  %85 = icmp eq ptr %79, @_ZTIj
  br i1 %85, label %86, label %89

86:                                               ; preds = %84
  %87 = load i32, ptr %78, align 4
  %88 = uitofp i32 %87 to double
  br label %99

89:                                               ; preds = %84
  %90 = icmp eq ptr %79, @_ZTIf
  br i1 %90, label %91, label %94

91:                                               ; preds = %89
  %92 = load float, ptr %78, align 4
  %93 = fpext float %92 to double
  br label %96

94:                                               ; preds = %89
  %95 = load double, ptr %78, align 8
  br label %96

96:                                               ; preds = %91, %94
  %97 = phi double [ %95, %94 ], [ %93, %91 ]
  br label %98

98:                                               ; preds = %96
  br label %99

99:                                               ; preds = %86, %98
  %100 = phi double [ %97, %98 ], [ %88, %86 ]
  br label %101

101:                                              ; preds = %99
  br label %102

102:                                              ; preds = %81, %101
  %103 = phi double [ %100, %101 ], [ %83, %81 ]
  br label %104

104:                                              ; preds = %102
  store double %103, ptr %2, align 8
  %105 = load i32, ptr @wrong, align 4
  %106 = add i32 %105, 1
  store i32 %106, ptr @wrong, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %251

endcatch.pad:                                     ; preds = %73
  %107 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

endcatch.pad1:                                    ; preds = %23
  %108 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

109:                                              ; No predecessors!
  %110 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad1, %endcatch.pad, %109
  %111 = phi { ptr, i32 } [ %110, %109 ], [ %107, %endcatch.pad ], [ %108, %endcatch.pad1 ]
  %112 = extractvalue { ptr, i32 } %111, 0
  %113 = call ptr @__cxa_begin_catch(ptr %112)
  %114 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.caught
  %115 = icmp eq ptr %114, @_ZTIi
  br i1 %115, label %116, label %123

116:                                              ; preds = %.noexc4
  %117 = load i32, ptr %113, align 4
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, i32 }, ptr %119, i32 0, i32 0
  store i64 4, ptr %120, align 8
  %121 = getelementptr { i64, ptr, i32 }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, i32 }, ptr %119, i32 0, i32 2
  store i32 %117, ptr %122, align 4
  br label %246

123:                                              ; preds = %.noexc4
  %124 = icmp eq ptr %114, @_ZTIj
  br i1 %124, label %125, label %132

125:                                              ; preds = %123
  %126 = load i32, ptr %113, align 4
  %127 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %125
  %128 = getelementptr i8, ptr %127, i64 8
  %129 = getelementptr { i64, ptr, i32 }, ptr %128, i32 0, i32 0
  store i64 4, ptr %129, align 8
  %130 = getelementptr { i64, ptr, i32 }, ptr %128, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %130, align 8
  %131 = getelementptr { i64, ptr, i32 }, ptr %128, i32 0, i32 2
  store i32 %126, ptr %131, align 4
  br label %243

132:                                              ; preds = %123
  %133 = icmp eq ptr %114, @_ZTId
  br i1 %133, label %134, label %141

134:                                              ; preds = %132
  %135 = load double, ptr %113, align 8
  %136 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %134
  %137 = getelementptr i8, ptr %136, i64 8
  %138 = getelementptr { i64, ptr, double }, ptr %137, i32 0, i32 0
  store i64 8, ptr %138, align 8
  %139 = getelementptr { i64, ptr, double }, ptr %137, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %139, align 8
  %140 = getelementptr { i64, ptr, double }, ptr %137, i32 0, i32 2
  store double %135, ptr %140, align 8
  br label %240

141:                                              ; preds = %132
  %142 = icmp eq ptr %114, @_ZTIf
  br i1 %142, label %143, label %150

143:                                              ; preds = %141
  %144 = load float, ptr %113, align 4
  %145 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %143
  %146 = getelementptr i8, ptr %145, i64 8
  %147 = getelementptr { i64, ptr, float }, ptr %146, i32 0, i32 0
  store i64 4, ptr %147, align 8
  %148 = getelementptr { i64, ptr, float }, ptr %146, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %148, align 8
  %149 = getelementptr { i64, ptr, float }, ptr %146, i32 0, i32 2
  store float %144, ptr %149, align 4
  br label %237

150:                                              ; preds = %141
  %151 = icmp eq ptr %114, @_ZTIb
  br i1 %151, label %152, label %159

152:                                              ; preds = %150
  %153 = load i1, ptr %113, align 1
  %154 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %152
  %155 = getelementptr i8, ptr %154, i64 8
  %156 = getelementptr { i64, ptr, i1 }, ptr %155, i32 0, i32 0
  store i64 1, ptr %156, align 8
  %157 = getelementptr { i64, ptr, i1 }, ptr %155, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %157, align 8
  %158 = getelementptr { i64, ptr, i1 }, ptr %155, i32 0, i32 2
  store i1 %153, ptr %158, align 1
  br label %234

159:                                              ; preds = %150
  %160 = icmp eq ptr %114, @_ZTIx
  br i1 %160, label %161, label %168

161:                                              ; preds = %159
  %162 = load i64, ptr %113, align 8
  %163 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %161
  %164 = getelementptr i8, ptr %163, i64 8
  %165 = getelementptr { i64, ptr, i64 }, ptr %164, i32 0, i32 0
  store i64 8, ptr %165, align 8
  %166 = getelementptr { i64, ptr, i64 }, ptr %164, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %166, align 8
  %167 = getelementptr { i64, ptr, i64 }, ptr %164, i32 0, i32 2
  store i64 %162, ptr %167, align 8
  br label %231

168:                                              ; preds = %159
  %169 = icmp eq ptr %114, @_ZTIPc
  br i1 %169, label %170, label %177

170:                                              ; preds = %168
  %171 = load ptr, ptr %113, align 8
  %172 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %170
  %173 = getelementptr i8, ptr %172, i64 8
  %174 = getelementptr { i64, ptr, ptr }, ptr %173, i32 0, i32 0
  store i64 8, ptr %174, align 8
  %175 = getelementptr { i64, ptr, ptr }, ptr %173, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %175, align 8
  %176 = getelementptr { i64, ptr, ptr }, ptr %173, i32 0, i32 2
  store ptr %171, ptr %176, align 8
  br label %228

177:                                              ; preds = %168
  %178 = icmp eq ptr %114, @_ZTIPv
  br i1 %178, label %179, label %181

179:                                              ; preds = %177
  %180 = load ptr, ptr %113, align 8
  br label %225

181:                                              ; preds = %177
  %182 = load ptr, ptr %114, align 8
  %183 = icmp eq ptr %182, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %183, label %184, label %215

184:                                              ; preds = %181
  %185 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %114, i32 0, i32 2
  %186 = load i32, ptr %185, align 4
  %187 = and i32 %186, 1073741824
  %188 = icmp ne i32 %187, 0
  br i1 %188, label %189, label %205

189:                                              ; preds = %184
  %190 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %114, i32 0, i32 4
  %191 = load ptr, ptr %190, align 8
  %192 = icmp ne ptr %191, null
  br i1 %192, label %193, label %195

193:                                              ; preds = %189
  invoke void %191(ptr %1, ptr %113)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %193
  %194 = load ptr, ptr %1, align 8
  br label %202

195:                                              ; preds = %189
  %196 = load ptr, ptr %113, align 8
  %197 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %195
  %198 = getelementptr i8, ptr %197, i64 8
  %199 = getelementptr { i64, ptr, ptr }, ptr %198, i32 0, i32 0
  store i64 8, ptr %199, align 8
  %200 = getelementptr { i64, ptr, ptr }, ptr %198, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %200, align 8
  %201 = getelementptr { i64, ptr, ptr }, ptr %198, i32 0, i32 2
  store ptr %196, ptr %201, align 8
  br label %202

202:                                              ; preds = %.noexc10, %.noexc9
  %203 = phi ptr [ %198, %.noexc9 ], [ %194, %.noexc10 ]
  br label %204

204:                                              ; preds = %202
  br label %212

205:                                              ; preds = %184
  %206 = load ptr, ptr %113, align 8
  %207 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %205
  %208 = getelementptr i8, ptr %207, i64 8
  %209 = getelementptr { i64, ptr, ptr }, ptr %208, i32 0, i32 0
  store i64 8, ptr %209, align 8
  %210 = getelementptr { i64, ptr, ptr }, ptr %208, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %210, align 8
  %211 = getelementptr { i64, ptr, ptr }, ptr %208, i32 0, i32 2
  store ptr %206, ptr %211, align 8
  br label %212

212:                                              ; preds = %204, %.noexc14
  %213 = phi ptr [ %208, %.noexc14 ], [ %203, %204 ]
  br label %214

214:                                              ; preds = %212
  br label %222

215:                                              ; preds = %181
  %216 = load ptr, ptr %113, align 8
  %217 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %215
  %218 = getelementptr i8, ptr %217, i64 8
  %219 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 0
  store i64 8, ptr %219, align 8
  %220 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %220, align 8
  %221 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 2
  store ptr %216, ptr %221, align 8
  br label %222

222:                                              ; preds = %214, %.noexc8
  %223 = phi ptr [ %218, %.noexc8 ], [ %213, %214 ]
  br label %224

224:                                              ; preds = %222
  br label %225

225:                                              ; preds = %179, %224
  %226 = phi ptr [ %223, %224 ], [ %180, %179 ]
  br label %227

227:                                              ; preds = %225
  br label %228

228:                                              ; preds = %.noexc13, %227
  %229 = phi ptr [ %226, %227 ], [ %173, %.noexc13 ]
  br label %230

230:                                              ; preds = %228
  br label %231

231:                                              ; preds = %.noexc16, %230
  %232 = phi ptr [ %229, %230 ], [ %164, %.noexc16 ]
  br label %233

233:                                              ; preds = %231
  br label %234

234:                                              ; preds = %.noexc2, %233
  %235 = phi ptr [ %232, %233 ], [ %155, %.noexc2 ]
  br label %236

236:                                              ; preds = %234
  br label %237

237:                                              ; preds = %.noexc, %236
  %238 = phi ptr [ %235, %236 ], [ %146, %.noexc ]
  br label %239

239:                                              ; preds = %237
  br label %240

240:                                              ; preds = %.noexc11, %239
  %241 = phi ptr [ %238, %239 ], [ %137, %.noexc11 ]
  br label %242

242:                                              ; preds = %240
  br label %243

243:                                              ; preds = %.noexc3, %242
  %244 = phi ptr [ %241, %242 ], [ %128, %.noexc3 ]
  br label %245

245:                                              ; preds = %243
  br label %246

246:                                              ; preds = %.noexc15, %245
  %247 = phi ptr [ %244, %245 ], [ %119, %.noexc15 ]
  br label %248

248:                                              ; preds = %246
  store ptr %247, ptr %4, align 8
  %249 = load i32, ptr @outer, align 4
  %250 = add i32 %249, 1
  store i32 %250, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %251

251:                                              ; preds = %104, %248
  ret void

252:                                              ; preds = %73, %.noexc12, %23, %0
  unreachable

endcatch.resume:                                  ; preds = %161, %116, %205, %170, %54, %134, %193, %195, %215, %26, %.caught, %125, %152, %143
  %253 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %253

endcatch.resume5:                                 ; preds = %76
  %254 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %254
}

; Function Attrs: mustprogress
define void @withFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %352 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  %11 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %7
  %12 = icmp eq ptr %11, @_ZTIi
  br i1 %12, label %13, label %20

13:                                               ; preds = %.noexc5
  %14 = load i32, ptr %10, align 4
  %15 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %13
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 0
  store i64 4, ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 2
  store i32 %14, ptr %19, align 4
  br label %143

20:                                               ; preds = %.noexc5
  %21 = icmp eq ptr %11, @_ZTIj
  br i1 %21, label %22, label %29

22:                                               ; preds = %20
  %23 = load i32, ptr %10, align 4
  %24 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %22
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
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %40
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %49
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
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %58
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
  invoke void %88(ptr %3, ptr %10)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %90
  %91 = load ptr, ptr %3, align 8
  br label %99

92:                                               ; preds = %86
  %93 = load ptr, ptr %10, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %99

99:                                               ; preds = %.noexc15, %.noexc14
  %100 = phi ptr [ %95, %.noexc14 ], [ %91, %.noexc15 ]
  br label %101

101:                                              ; preds = %99
  br label %109

102:                                              ; preds = %81
  %103 = load ptr, ptr %10, align 8
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 0
  store i64 8, ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 2
  store ptr %103, ptr %108, align 8
  br label %109

109:                                              ; preds = %101, %.noexc
  %110 = phi ptr [ %105, %.noexc ], [ %100, %101 ]
  br label %111

111:                                              ; preds = %109
  br label %119

112:                                              ; preds = %78
  %113 = load ptr, ptr %10, align 8
  %114 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %112
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 0
  store i64 8, ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 2
  store ptr %113, ptr %118, align 8
  br label %119

119:                                              ; preds = %111, %.noexc3
  %120 = phi ptr [ %115, %.noexc3 ], [ %110, %111 ]
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

128:                                              ; preds = %.noexc16, %127
  %129 = phi ptr [ %126, %127 ], [ %61, %.noexc16 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc9, %130
  %132 = phi ptr [ %129, %130 ], [ %52, %.noexc9 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc22, %133
  %135 = phi ptr [ %132, %133 ], [ %43, %.noexc22 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc18, %136
  %138 = phi ptr [ %135, %136 ], [ %34, %.noexc18 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc17, %139
  %141 = phi ptr [ %138, %139 ], [ %25, %.noexc17 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc19, %142
  %144 = phi ptr [ %141, %142 ], [ %16, %.noexc19 ]
  br label %145

145:                                              ; preds = %143
  store ptr %144, ptr %4, align 8
  %146 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %145
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %146, align 8
  invoke void @__cxa_throw(ptr %146, ptr @_ZTIPc, ptr null)
          to label %352 unwind label %147

147:                                              ; preds = %.noexc13
  %148 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %149 = extractvalue { ptr, i32 } %148, 1
  %150 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %151 = icmp eq i32 %149, %150
  %152 = extractvalue { ptr, i32 } %148, 1
  %153 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %154 = icmp eq i32 %152, %153
  %155 = extractvalue { ptr, i32 } %148, 1
  %156 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %157 = icmp eq i32 %155, %156
  %158 = extractvalue { ptr, i32 } %148, 1
  %159 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %160 = icmp eq i32 %158, %159
  %161 = or i1 %151, %154
  %162 = or i1 %161, %157
  %163 = or i1 %162, %160
  br i1 %163, label %167, label %164

164:                                              ; preds = %147
  %165 = extractvalue { ptr, i32 } %148, 0
  %166 = call ptr @__cxa_begin_catch(ptr %165)
  invoke void @__cxa_rethrow()
          to label %352 unwind label %endcatch.pad2

167:                                              ; preds = %147
  %168 = extractvalue { ptr, i32 } %148, 0
  %169 = call ptr @__cxa_begin_catch(ptr %168)
  %170 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume10

.noexc11:                                         ; preds = %167
  %171 = icmp eq ptr %170, @_ZTIi
  br i1 %171, label %172, label %175

172:                                              ; preds = %.noexc11
  %173 = load i32, ptr %169, align 4
  %174 = sitofp i32 %173 to double
  br label %193

175:                                              ; preds = %.noexc11
  %176 = icmp eq ptr %170, @_ZTIj
  br i1 %176, label %177, label %180

177:                                              ; preds = %175
  %178 = load i32, ptr %169, align 4
  %179 = uitofp i32 %178 to double
  br label %190

180:                                              ; preds = %175
  %181 = icmp eq ptr %170, @_ZTIf
  br i1 %181, label %182, label %185

182:                                              ; preds = %180
  %183 = load float, ptr %169, align 4
  %184 = fpext float %183 to double
  br label %187

185:                                              ; preds = %180
  %186 = load double, ptr %169, align 8
  br label %187

187:                                              ; preds = %182, %185
  %188 = phi double [ %186, %185 ], [ %184, %182 ]
  br label %189

189:                                              ; preds = %187
  br label %190

190:                                              ; preds = %177, %189
  %191 = phi double [ %188, %189 ], [ %179, %177 ]
  br label %192

192:                                              ; preds = %190
  br label %193

193:                                              ; preds = %172, %192
  %194 = phi double [ %191, %192 ], [ %174, %172 ]
  br label %195

195:                                              ; preds = %193
  store double %194, ptr %2, align 8
  %196 = load i32, ptr @wrong, align 4
  %197 = add i32 %196, 1
  store i32 %197, ptr @wrong, align 4
  call void @__cxa_end_catch()
  %198 = load i32, ptr @finals, align 4
  %199 = add i32 %198, 1
  store i32 %199, ptr @finals, align 4
  call void @__cxa_end_catch()
  br label %351

endcatch.pad2:                                    ; preds = %164
  %200 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

201:                                              ; No predecessors!
  %202 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %201
  %203 = phi { ptr, i32 } [ %202, %201 ], [ %200, %endcatch.pad2 ]
  %204 = extractvalue { ptr, i32 } %203, 0
  %205 = call ptr @__cxa_begin_catch(ptr %204)
  %206 = load i32, ptr @finals, align 4
  %207 = add i32 %206, 1
  store i32 %207, ptr @finals, align 4
  invoke void @__cxa_rethrow()
          to label %352 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %.caught1
  %208 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

209:                                              ; No predecessors!
  %210 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %209
  %211 = phi { ptr, i32 } [ %210, %209 ], [ %208, %endcatch.pad ]
  %212 = extractvalue { ptr, i32 } %211, 0
  %213 = call ptr @__cxa_begin_catch(ptr %212)
  %214 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.caught
  %215 = icmp eq ptr %214, @_ZTIi
  br i1 %215, label %216, label %223

216:                                              ; preds = %.noexc6
  %217 = load i32, ptr %213, align 4
  %218 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %216
  %219 = getelementptr i8, ptr %218, i64 8
  %220 = getelementptr { i64, ptr, i32 }, ptr %219, i32 0, i32 0
  store i64 4, ptr %220, align 8
  %221 = getelementptr { i64, ptr, i32 }, ptr %219, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %221, align 8
  %222 = getelementptr { i64, ptr, i32 }, ptr %219, i32 0, i32 2
  store i32 %217, ptr %222, align 4
  br label %346

223:                                              ; preds = %.noexc6
  %224 = icmp eq ptr %214, @_ZTIj
  br i1 %224, label %225, label %232

225:                                              ; preds = %223
  %226 = load i32, ptr %213, align 4
  %227 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %225
  %228 = getelementptr i8, ptr %227, i64 8
  %229 = getelementptr { i64, ptr, i32 }, ptr %228, i32 0, i32 0
  store i64 4, ptr %229, align 8
  %230 = getelementptr { i64, ptr, i32 }, ptr %228, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %230, align 8
  %231 = getelementptr { i64, ptr, i32 }, ptr %228, i32 0, i32 2
  store i32 %226, ptr %231, align 4
  br label %343

232:                                              ; preds = %223
  %233 = icmp eq ptr %214, @_ZTId
  br i1 %233, label %234, label %241

234:                                              ; preds = %232
  %235 = load double, ptr %213, align 8
  %236 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %234
  %237 = getelementptr i8, ptr %236, i64 8
  %238 = getelementptr { i64, ptr, double }, ptr %237, i32 0, i32 0
  store i64 8, ptr %238, align 8
  %239 = getelementptr { i64, ptr, double }, ptr %237, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %239, align 8
  %240 = getelementptr { i64, ptr, double }, ptr %237, i32 0, i32 2
  store double %235, ptr %240, align 8
  br label %340

241:                                              ; preds = %232
  %242 = icmp eq ptr %214, @_ZTIf
  br i1 %242, label %243, label %250

243:                                              ; preds = %241
  %244 = load float, ptr %213, align 4
  %245 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %243
  %246 = getelementptr i8, ptr %245, i64 8
  %247 = getelementptr { i64, ptr, float }, ptr %246, i32 0, i32 0
  store i64 4, ptr %247, align 8
  %248 = getelementptr { i64, ptr, float }, ptr %246, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %248, align 8
  %249 = getelementptr { i64, ptr, float }, ptr %246, i32 0, i32 2
  store float %244, ptr %249, align 4
  br label %337

250:                                              ; preds = %241
  %251 = icmp eq ptr %214, @_ZTIb
  br i1 %251, label %252, label %259

252:                                              ; preds = %250
  %253 = load i1, ptr %213, align 1
  %254 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %252
  %255 = getelementptr i8, ptr %254, i64 8
  %256 = getelementptr { i64, ptr, i1 }, ptr %255, i32 0, i32 0
  store i64 1, ptr %256, align 8
  %257 = getelementptr { i64, ptr, i1 }, ptr %255, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %257, align 8
  %258 = getelementptr { i64, ptr, i1 }, ptr %255, i32 0, i32 2
  store i1 %253, ptr %258, align 1
  br label %334

259:                                              ; preds = %250
  %260 = icmp eq ptr %214, @_ZTIx
  br i1 %260, label %261, label %268

261:                                              ; preds = %259
  %262 = load i64, ptr %213, align 8
  %263 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %261
  %264 = getelementptr i8, ptr %263, i64 8
  %265 = getelementptr { i64, ptr, i64 }, ptr %264, i32 0, i32 0
  store i64 8, ptr %265, align 8
  %266 = getelementptr { i64, ptr, i64 }, ptr %264, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %266, align 8
  %267 = getelementptr { i64, ptr, i64 }, ptr %264, i32 0, i32 2
  store i64 %262, ptr %267, align 8
  br label %331

268:                                              ; preds = %259
  %269 = icmp eq ptr %214, @_ZTIPc
  br i1 %269, label %270, label %277

270:                                              ; preds = %268
  %271 = load ptr, ptr %213, align 8
  %272 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %270
  %273 = getelementptr i8, ptr %272, i64 8
  %274 = getelementptr { i64, ptr, ptr }, ptr %273, i32 0, i32 0
  store i64 8, ptr %274, align 8
  %275 = getelementptr { i64, ptr, ptr }, ptr %273, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %275, align 8
  %276 = getelementptr { i64, ptr, ptr }, ptr %273, i32 0, i32 2
  store ptr %271, ptr %276, align 8
  br label %328

277:                                              ; preds = %268
  %278 = icmp eq ptr %214, @_ZTIPv
  br i1 %278, label %279, label %281

279:                                              ; preds = %277
  %280 = load ptr, ptr %213, align 8
  br label %325

281:                                              ; preds = %277
  %282 = load ptr, ptr %214, align 8
  %283 = icmp eq ptr %282, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %283, label %284, label %315

284:                                              ; preds = %281
  %285 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %214, i32 0, i32 2
  %286 = load i32, ptr %285, align 4
  %287 = and i32 %286, 1073741824
  %288 = icmp ne i32 %287, 0
  br i1 %288, label %289, label %305

289:                                              ; preds = %284
  %290 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %214, i32 0, i32 4
  %291 = load ptr, ptr %290, align 8
  %292 = icmp ne ptr %291, null
  br i1 %292, label %293, label %295

293:                                              ; preds = %289
  invoke void %291(ptr %1, ptr %213)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %293
  %294 = load ptr, ptr %1, align 8
  br label %302

295:                                              ; preds = %289
  %296 = load ptr, ptr %213, align 8
  %297 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %295
  %298 = getelementptr i8, ptr %297, i64 8
  %299 = getelementptr { i64, ptr, ptr }, ptr %298, i32 0, i32 0
  store i64 8, ptr %299, align 8
  %300 = getelementptr { i64, ptr, ptr }, ptr %298, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %300, align 8
  %301 = getelementptr { i64, ptr, ptr }, ptr %298, i32 0, i32 2
  store ptr %296, ptr %301, align 8
  br label %302

302:                                              ; preds = %.noexc8, %.noexc4
  %303 = phi ptr [ %298, %.noexc4 ], [ %294, %.noexc8 ]
  br label %304

304:                                              ; preds = %302
  br label %312

305:                                              ; preds = %284
  %306 = load ptr, ptr %213, align 8
  %307 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %305
  %308 = getelementptr i8, ptr %307, i64 8
  %309 = getelementptr { i64, ptr, ptr }, ptr %308, i32 0, i32 0
  store i64 8, ptr %309, align 8
  %310 = getelementptr { i64, ptr, ptr }, ptr %308, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %310, align 8
  %311 = getelementptr { i64, ptr, ptr }, ptr %308, i32 0, i32 2
  store ptr %306, ptr %311, align 8
  br label %312

312:                                              ; preds = %304, %.noexc23
  %313 = phi ptr [ %308, %.noexc23 ], [ %303, %304 ]
  br label %314

314:                                              ; preds = %312
  br label %322

315:                                              ; preds = %281
  %316 = load ptr, ptr %213, align 8
  %317 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %315
  %318 = getelementptr i8, ptr %317, i64 8
  %319 = getelementptr { i64, ptr, ptr }, ptr %318, i32 0, i32 0
  store i64 8, ptr %319, align 8
  %320 = getelementptr { i64, ptr, ptr }, ptr %318, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %320, align 8
  %321 = getelementptr { i64, ptr, ptr }, ptr %318, i32 0, i32 2
  store ptr %316, ptr %321, align 8
  br label %322

322:                                              ; preds = %314, %.noexc27
  %323 = phi ptr [ %318, %.noexc27 ], [ %313, %314 ]
  br label %324

324:                                              ; preds = %322
  br label %325

325:                                              ; preds = %279, %324
  %326 = phi ptr [ %323, %324 ], [ %280, %279 ]
  br label %327

327:                                              ; preds = %325
  br label %328

328:                                              ; preds = %.noexc7, %327
  %329 = phi ptr [ %326, %327 ], [ %273, %.noexc7 ]
  br label %330

330:                                              ; preds = %328
  br label %331

331:                                              ; preds = %.noexc20, %330
  %332 = phi ptr [ %329, %330 ], [ %264, %.noexc20 ]
  br label %333

333:                                              ; preds = %331
  br label %334

334:                                              ; preds = %.noexc21, %333
  %335 = phi ptr [ %332, %333 ], [ %255, %.noexc21 ]
  br label %336

336:                                              ; preds = %334
  br label %337

337:                                              ; preds = %.noexc24, %336
  %338 = phi ptr [ %335, %336 ], [ %246, %.noexc24 ]
  br label %339

339:                                              ; preds = %337
  br label %340

340:                                              ; preds = %.noexc26, %339
  %341 = phi ptr [ %338, %339 ], [ %237, %.noexc26 ]
  br label %342

342:                                              ; preds = %340
  br label %343

343:                                              ; preds = %.noexc25, %342
  %344 = phi ptr [ %341, %342 ], [ %228, %.noexc25 ]
  br label %345

345:                                              ; preds = %343
  br label %346

346:                                              ; preds = %.noexc28, %345
  %347 = phi ptr [ %344, %345 ], [ %219, %.noexc28 ]
  br label %348

348:                                              ; preds = %346
  store ptr %347, ptr %5, align 8
  %349 = load i32, ptr @outer, align 4
  %350 = add i32 %349, 1
  store i32 %350, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %351

351:                                              ; preds = %195, %348
  ret void

352:                                              ; preds = %.caught1, %164, %.noexc13, %0
  unreachable

endcatch.resume:                                  ; preds = %216, %315, %234, %225, %243, %305, %40, %252, %261, %13, %31, %22, %58, %90, %92, %145, %67, %49, %293, %270, %.caught, %7, %295, %112, %102
  %353 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %353

endcatch.resume10:                                ; preds = %167
  %354 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %354
}

; Function Attrs: mustprogress
define void @inParentFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = load i32, ptr @finals, align 4
  %7 = add i32 %6, 10
  store i32 %7, ptr @finals, align 4
  store i1 false, ptr %2, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  store ptr %9, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %11 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %11, align 8
  invoke void @__cxa_throw(ptr %11, ptr @_ZTIPc, ptr null)
          to label %240 unwind label %12

12:                                               ; preds = %0
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i1, ptr %2, align 1
  br i1 %16, label %17, label %32

17:                                               ; preds = %12
  store i1 false, ptr %2, align 1
  %18 = load ptr, ptr %3, align 8
  %19 = getelementptr %Res, ptr %18, i32 0, i32 0
  invoke void @Res.dispose(ptr %18)
          to label %20 unwind label %endcatch.pad2

20:                                               ; preds = %17
  br label %32

endcatch.pad2:                                    ; preds = %17
  %21 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

22:                                               ; No predecessors!
  %23 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad2 ]
  %25 = extractvalue { ptr, i32 } %24, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = load i1, ptr %2, align 1
  br i1 %27, label %28, label %31

28:                                               ; preds = %.caught1
  store i1 false, ptr %2, align 1
  %29 = load ptr, ptr %3, align 8
  %30 = getelementptr %Res, ptr %29, i32 0, i32 0
  invoke void @Res.dispose(ptr %29)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %28
  br label %31

31:                                               ; preds = %.noexc8, %.caught1
  invoke void @__cxa_rethrow()
          to label %240 unwind label %endcatch.pad

32:                                               ; preds = %20, %12
  invoke void @__cxa_rethrow()
          to label %240 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %32, %31
  %33 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %33, %endcatch.pad ]
  %37 = extractvalue { ptr, i32 } %36, 1
  %38 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %39 = icmp eq i32 %37, %38
  %40 = extractvalue { ptr, i32 } %36, 1
  %41 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %42 = icmp eq i32 %40, %41
  %43 = extractvalue { ptr, i32 } %36, 1
  %44 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %45 = icmp eq i32 %43, %44
  %46 = extractvalue { ptr, i32 } %36, 1
  %47 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %48 = icmp eq i32 %46, %47
  %49 = or i1 %39, %42
  %50 = or i1 %49, %45
  %51 = or i1 %50, %48
  br i1 %51, label %55, label %52

52:                                               ; preds = %.caught
  %53 = extractvalue { ptr, i32 } %36, 0
  %54 = call ptr @__cxa_begin_catch(ptr %53)
  invoke void @__cxa_rethrow()
          to label %240 unwind label %endcatch.pad6

55:                                               ; preds = %.caught
  %56 = extractvalue { ptr, i32 } %36, 0
  %57 = call ptr @__cxa_begin_catch(ptr %56)
  %58 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %55
  %59 = icmp eq ptr %58, @_ZTIi
  br i1 %59, label %60, label %63

60:                                               ; preds = %.noexc
  %61 = load i32, ptr %57, align 4
  %62 = sitofp i32 %61 to double
  br label %81

63:                                               ; preds = %.noexc
  %64 = icmp eq ptr %58, @_ZTIj
  br i1 %64, label %65, label %68

65:                                               ; preds = %63
  %66 = load i32, ptr %57, align 4
  %67 = uitofp i32 %66 to double
  br label %78

68:                                               ; preds = %63
  %69 = icmp eq ptr %58, @_ZTIf
  br i1 %69, label %70, label %73

70:                                               ; preds = %68
  %71 = load float, ptr %57, align 4
  %72 = fpext float %71 to double
  br label %75

73:                                               ; preds = %68
  %74 = load double, ptr %57, align 8
  br label %75

75:                                               ; preds = %70, %73
  %76 = phi double [ %74, %73 ], [ %72, %70 ]
  br label %77

77:                                               ; preds = %75
  br label %78

78:                                               ; preds = %65, %77
  %79 = phi double [ %76, %77 ], [ %67, %65 ]
  br label %80

80:                                               ; preds = %78
  br label %81

81:                                               ; preds = %60, %80
  %82 = phi double [ %79, %80 ], [ %62, %60 ]
  br label %83

83:                                               ; preds = %81
  store double %82, ptr %4, align 8
  %84 = load i32, ptr @wrong, align 4
  %85 = add i32 %84, 1
  store i32 %85, ptr @wrong, align 4
  call void @__cxa_end_catch()
  %86 = load i32, ptr @finals, align 4
  %87 = add i32 %86, 1
  store i32 %87, ptr @finals, align 4
  br label %239

endcatch.pad6:                                    ; preds = %52
  %88 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught5

89:                                               ; No predecessors!
  %90 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught5

.caught5:                                         ; preds = %endcatch.pad6, %89
  %91 = phi { ptr, i32 } [ %90, %89 ], [ %88, %endcatch.pad6 ]
  %92 = extractvalue { ptr, i32 } %91, 0
  %93 = call ptr @__cxa_begin_catch(ptr %92)
  %94 = load i32, ptr @finals, align 4
  %95 = add i32 %94, 1
  store i32 %95, ptr @finals, align 4
  invoke void @__cxa_rethrow()
          to label %240 unwind label %endcatch.pad4

endcatch.pad4:                                    ; preds = %.caught5
  %96 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

97:                                               ; No predecessors!
  %98 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %97
  %99 = phi { ptr, i32 } [ %98, %97 ], [ %96, %endcatch.pad4 ]
  %100 = extractvalue { ptr, i32 } %99, 0
  %101 = call ptr @__cxa_begin_catch(ptr %100)
  %102 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %.caught3
  %103 = icmp eq ptr %102, @_ZTIi
  br i1 %103, label %104, label %111

104:                                              ; preds = %.noexc15
  %105 = load i32, ptr %101, align 4
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, i32 }, ptr %107, i32 0, i32 0
  store i64 4, ptr %108, align 8
  %109 = getelementptr { i64, ptr, i32 }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, i32 }, ptr %107, i32 0, i32 2
  store i32 %105, ptr %110, align 4
  br label %234

111:                                              ; preds = %.noexc15
  %112 = icmp eq ptr %102, @_ZTIj
  br i1 %112, label %113, label %120

113:                                              ; preds = %111
  %114 = load i32, ptr %101, align 4
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, i32 }, ptr %116, i32 0, i32 0
  store i64 4, ptr %117, align 8
  %118 = getelementptr { i64, ptr, i32 }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, i32 }, ptr %116, i32 0, i32 2
  store i32 %114, ptr %119, align 4
  br label %231

120:                                              ; preds = %111
  %121 = icmp eq ptr %102, @_ZTId
  br i1 %121, label %122, label %129

122:                                              ; preds = %120
  %123 = load double, ptr %101, align 8
  %124 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %122
  %125 = getelementptr i8, ptr %124, i64 8
  %126 = getelementptr { i64, ptr, double }, ptr %125, i32 0, i32 0
  store i64 8, ptr %126, align 8
  %127 = getelementptr { i64, ptr, double }, ptr %125, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %127, align 8
  %128 = getelementptr { i64, ptr, double }, ptr %125, i32 0, i32 2
  store double %123, ptr %128, align 8
  br label %228

129:                                              ; preds = %120
  %130 = icmp eq ptr %102, @_ZTIf
  br i1 %130, label %131, label %138

131:                                              ; preds = %129
  %132 = load float, ptr %101, align 4
  %133 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %131
  %134 = getelementptr i8, ptr %133, i64 8
  %135 = getelementptr { i64, ptr, float }, ptr %134, i32 0, i32 0
  store i64 4, ptr %135, align 8
  %136 = getelementptr { i64, ptr, float }, ptr %134, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %136, align 8
  %137 = getelementptr { i64, ptr, float }, ptr %134, i32 0, i32 2
  store float %132, ptr %137, align 4
  br label %225

138:                                              ; preds = %129
  %139 = icmp eq ptr %102, @_ZTIb
  br i1 %139, label %140, label %147

140:                                              ; preds = %138
  %141 = load i1, ptr %101, align 1
  %142 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %140
  %143 = getelementptr i8, ptr %142, i64 8
  %144 = getelementptr { i64, ptr, i1 }, ptr %143, i32 0, i32 0
  store i64 1, ptr %144, align 8
  %145 = getelementptr { i64, ptr, i1 }, ptr %143, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %145, align 8
  %146 = getelementptr { i64, ptr, i1 }, ptr %143, i32 0, i32 2
  store i1 %141, ptr %146, align 1
  br label %222

147:                                              ; preds = %138
  %148 = icmp eq ptr %102, @_ZTIx
  br i1 %148, label %149, label %156

149:                                              ; preds = %147
  %150 = load i64, ptr %101, align 8
  %151 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %149
  %152 = getelementptr i8, ptr %151, i64 8
  %153 = getelementptr { i64, ptr, i64 }, ptr %152, i32 0, i32 0
  store i64 8, ptr %153, align 8
  %154 = getelementptr { i64, ptr, i64 }, ptr %152, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %154, align 8
  %155 = getelementptr { i64, ptr, i64 }, ptr %152, i32 0, i32 2
  store i64 %150, ptr %155, align 8
  br label %219

156:                                              ; preds = %147
  %157 = icmp eq ptr %102, @_ZTIPc
  br i1 %157, label %158, label %165

158:                                              ; preds = %156
  %159 = load ptr, ptr %101, align 8
  %160 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %158
  %161 = getelementptr i8, ptr %160, i64 8
  %162 = getelementptr { i64, ptr, ptr }, ptr %161, i32 0, i32 0
  store i64 8, ptr %162, align 8
  %163 = getelementptr { i64, ptr, ptr }, ptr %161, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %163, align 8
  %164 = getelementptr { i64, ptr, ptr }, ptr %161, i32 0, i32 2
  store ptr %159, ptr %164, align 8
  br label %216

165:                                              ; preds = %156
  %166 = icmp eq ptr %102, @_ZTIPv
  br i1 %166, label %167, label %169

167:                                              ; preds = %165
  %168 = load ptr, ptr %101, align 8
  br label %213

169:                                              ; preds = %165
  %170 = load ptr, ptr %102, align 8
  %171 = icmp eq ptr %170, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %171, label %172, label %203

172:                                              ; preds = %169
  %173 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %102, i32 0, i32 2
  %174 = load i32, ptr %173, align 4
  %175 = and i32 %174, 1073741824
  %176 = icmp ne i32 %175, 0
  br i1 %176, label %177, label %193

177:                                              ; preds = %172
  %178 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %102, i32 0, i32 4
  %179 = load ptr, ptr %178, align 8
  %180 = icmp ne ptr %179, null
  br i1 %180, label %181, label %183

181:                                              ; preds = %177
  invoke void %179(ptr %1, ptr %101)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %181
  %182 = load ptr, ptr %1, align 8
  br label %190

183:                                              ; preds = %177
  %184 = load ptr, ptr %101, align 8
  %185 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %183
  %186 = getelementptr i8, ptr %185, i64 8
  %187 = getelementptr { i64, ptr, ptr }, ptr %186, i32 0, i32 0
  store i64 8, ptr %187, align 8
  %188 = getelementptr { i64, ptr, ptr }, ptr %186, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %188, align 8
  %189 = getelementptr { i64, ptr, ptr }, ptr %186, i32 0, i32 2
  store ptr %184, ptr %189, align 8
  br label %190

190:                                              ; preds = %.noexc12, %.noexc7
  %191 = phi ptr [ %186, %.noexc7 ], [ %182, %.noexc12 ]
  br label %192

192:                                              ; preds = %190
  br label %200

193:                                              ; preds = %172
  %194 = load ptr, ptr %101, align 8
  %195 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %193
  %196 = getelementptr i8, ptr %195, i64 8
  %197 = getelementptr { i64, ptr, ptr }, ptr %196, i32 0, i32 0
  store i64 8, ptr %197, align 8
  %198 = getelementptr { i64, ptr, ptr }, ptr %196, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %198, align 8
  %199 = getelementptr { i64, ptr, ptr }, ptr %196, i32 0, i32 2
  store ptr %194, ptr %199, align 8
  br label %200

200:                                              ; preds = %192, %.noexc19
  %201 = phi ptr [ %196, %.noexc19 ], [ %191, %192 ]
  br label %202

202:                                              ; preds = %200
  br label %210

203:                                              ; preds = %169
  %204 = load ptr, ptr %101, align 8
  %205 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %203
  %206 = getelementptr i8, ptr %205, i64 8
  %207 = getelementptr { i64, ptr, ptr }, ptr %206, i32 0, i32 0
  store i64 8, ptr %207, align 8
  %208 = getelementptr { i64, ptr, ptr }, ptr %206, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %208, align 8
  %209 = getelementptr { i64, ptr, ptr }, ptr %206, i32 0, i32 2
  store ptr %204, ptr %209, align 8
  br label %210

210:                                              ; preds = %202, %.noexc9
  %211 = phi ptr [ %206, %.noexc9 ], [ %201, %202 ]
  br label %212

212:                                              ; preds = %210
  br label %213

213:                                              ; preds = %167, %212
  %214 = phi ptr [ %211, %212 ], [ %168, %167 ]
  br label %215

215:                                              ; preds = %213
  br label %216

216:                                              ; preds = %.noexc17, %215
  %217 = phi ptr [ %214, %215 ], [ %161, %.noexc17 ]
  br label %218

218:                                              ; preds = %216
  br label %219

219:                                              ; preds = %.noexc16, %218
  %220 = phi ptr [ %217, %218 ], [ %152, %.noexc16 ]
  br label %221

221:                                              ; preds = %219
  br label %222

222:                                              ; preds = %.noexc18, %221
  %223 = phi ptr [ %220, %221 ], [ %143, %.noexc18 ]
  br label %224

224:                                              ; preds = %222
  br label %225

225:                                              ; preds = %.noexc13, %224
  %226 = phi ptr [ %223, %224 ], [ %134, %.noexc13 ]
  br label %227

227:                                              ; preds = %225
  br label %228

228:                                              ; preds = %.noexc11, %227
  %229 = phi ptr [ %226, %227 ], [ %125, %.noexc11 ]
  br label %230

230:                                              ; preds = %228
  br label %231

231:                                              ; preds = %.noexc10, %230
  %232 = phi ptr [ %229, %230 ], [ %116, %.noexc10 ]
  br label %233

233:                                              ; preds = %231
  br label %234

234:                                              ; preds = %.noexc14, %233
  %235 = phi ptr [ %232, %233 ], [ %107, %.noexc14 ]
  br label %236

236:                                              ; preds = %234
  store ptr %235, ptr %5, align 8
  %237 = load i32, ptr @outer, align 4
  %238 = add i32 %237, 1
  store i32 %238, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %239

239:                                              ; preds = %83, %236
  ret void

240:                                              ; preds = %.caught5, %52, %32, %31, %0
  unreachable

endcatch.resume:                                  ; preds = %193, %140, %158, %149, %.caught3, %104, %131, %181, %122, %113, %203, %28, %183, %55
  %241 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %241
}

; Function Attrs: mustprogress
define void @twoLevelsUp() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %494 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc9
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc9
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %24
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
          to label %.noexc35 unwind label %endcatch.resume

.noexc35:                                         ; preds = %33
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
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %42
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %51
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
          to label %.noexc39 unwind label %endcatch.resume

.noexc39:                                         ; preds = %60
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
          to label %.noexc41 unwind label %endcatch.resume

.noexc41:                                         ; preds = %92
  %93 = load ptr, ptr %5, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc34 unwind label %endcatch.resume

.noexc34:                                         ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc41, %.noexc34
  %102 = phi ptr [ %97, %.noexc34 ], [ %93, %.noexc41 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc15
  %112 = phi ptr [ %107, %.noexc15 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc31 unwind label %endcatch.resume

.noexc31:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc31
  %122 = phi ptr [ %117, %.noexc31 ], [ %112, %113 ]
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

130:                                              ; preds = %.noexc39, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc39 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc11, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc11 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc24, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc24 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc35, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc35 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc22, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc22 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %6, align 8
  %148 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %147
  store i32 2, ptr %148, align 4
  invoke void @__cxa_throw(ptr %148, ptr @_ZTIi, ptr null)
          to label %494 unwind label %149

149:                                              ; preds = %.noexc18
  %150 = landingpad { ptr, i32 }
          catch ptr null
  %151 = extractvalue { ptr, i32 } %150, 0
  %152 = call ptr @__cxa_begin_catch(ptr %151)
  %153 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc30 unwind label %endcatch.resume3

.noexc30:                                         ; preds = %149
  %154 = icmp eq ptr %153, @_ZTIi
  br i1 %154, label %155, label %162

155:                                              ; preds = %.noexc30
  %156 = load i32, ptr %152, align 4
  %157 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume3

.noexc27:                                         ; preds = %155
  %158 = getelementptr i8, ptr %157, i64 8
  %159 = getelementptr { i64, ptr, i32 }, ptr %158, i32 0, i32 0
  store i64 4, ptr %159, align 8
  %160 = getelementptr { i64, ptr, i32 }, ptr %158, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %160, align 8
  %161 = getelementptr { i64, ptr, i32 }, ptr %158, i32 0, i32 2
  store i32 %156, ptr %161, align 4
  br label %285

162:                                              ; preds = %.noexc30
  %163 = icmp eq ptr %153, @_ZTIj
  br i1 %163, label %164, label %171

164:                                              ; preds = %162
  %165 = load i32, ptr %152, align 4
  %166 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume3

.noexc26:                                         ; preds = %164
  %167 = getelementptr i8, ptr %166, i64 8
  %168 = getelementptr { i64, ptr, i32 }, ptr %167, i32 0, i32 0
  store i64 4, ptr %168, align 8
  %169 = getelementptr { i64, ptr, i32 }, ptr %167, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %169, align 8
  %170 = getelementptr { i64, ptr, i32 }, ptr %167, i32 0, i32 2
  store i32 %165, ptr %170, align 4
  br label %282

171:                                              ; preds = %162
  %172 = icmp eq ptr %153, @_ZTId
  br i1 %172, label %173, label %180

173:                                              ; preds = %171
  %174 = load double, ptr %152, align 8
  %175 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume3

.noexc20:                                         ; preds = %173
  %176 = getelementptr i8, ptr %175, i64 8
  %177 = getelementptr { i64, ptr, double }, ptr %176, i32 0, i32 0
  store i64 8, ptr %177, align 8
  %178 = getelementptr { i64, ptr, double }, ptr %176, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %178, align 8
  %179 = getelementptr { i64, ptr, double }, ptr %176, i32 0, i32 2
  store double %174, ptr %179, align 8
  br label %279

180:                                              ; preds = %171
  %181 = icmp eq ptr %153, @_ZTIf
  br i1 %181, label %182, label %189

182:                                              ; preds = %180
  %183 = load float, ptr %152, align 4
  %184 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume3

.noexc8:                                          ; preds = %182
  %185 = getelementptr i8, ptr %184, i64 8
  %186 = getelementptr { i64, ptr, float }, ptr %185, i32 0, i32 0
  store i64 4, ptr %186, align 8
  %187 = getelementptr { i64, ptr, float }, ptr %185, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %187, align 8
  %188 = getelementptr { i64, ptr, float }, ptr %185, i32 0, i32 2
  store float %183, ptr %188, align 4
  br label %276

189:                                              ; preds = %180
  %190 = icmp eq ptr %153, @_ZTIb
  br i1 %190, label %191, label %198

191:                                              ; preds = %189
  %192 = load i1, ptr %152, align 1
  %193 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume3

.noexc12:                                         ; preds = %191
  %194 = getelementptr i8, ptr %193, i64 8
  %195 = getelementptr { i64, ptr, i1 }, ptr %194, i32 0, i32 0
  store i64 1, ptr %195, align 8
  %196 = getelementptr { i64, ptr, i1 }, ptr %194, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %196, align 8
  %197 = getelementptr { i64, ptr, i1 }, ptr %194, i32 0, i32 2
  store i1 %192, ptr %197, align 1
  br label %273

198:                                              ; preds = %189
  %199 = icmp eq ptr %153, @_ZTIx
  br i1 %199, label %200, label %207

200:                                              ; preds = %198
  %201 = load i64, ptr %152, align 8
  %202 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume3

.noexc16:                                         ; preds = %200
  %203 = getelementptr i8, ptr %202, i64 8
  %204 = getelementptr { i64, ptr, i64 }, ptr %203, i32 0, i32 0
  store i64 8, ptr %204, align 8
  %205 = getelementptr { i64, ptr, i64 }, ptr %203, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %205, align 8
  %206 = getelementptr { i64, ptr, i64 }, ptr %203, i32 0, i32 2
  store i64 %201, ptr %206, align 8
  br label %270

207:                                              ; preds = %198
  %208 = icmp eq ptr %153, @_ZTIPc
  br i1 %208, label %209, label %216

209:                                              ; preds = %207
  %210 = load ptr, ptr %152, align 8
  %211 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume3

.noexc4:                                          ; preds = %209
  %212 = getelementptr i8, ptr %211, i64 8
  %213 = getelementptr { i64, ptr, ptr }, ptr %212, i32 0, i32 0
  store i64 8, ptr %213, align 8
  %214 = getelementptr { i64, ptr, ptr }, ptr %212, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %214, align 8
  %215 = getelementptr { i64, ptr, ptr }, ptr %212, i32 0, i32 2
  store ptr %210, ptr %215, align 8
  br label %267

216:                                              ; preds = %207
  %217 = icmp eq ptr %153, @_ZTIPv
  br i1 %217, label %218, label %220

218:                                              ; preds = %216
  %219 = load ptr, ptr %152, align 8
  br label %264

220:                                              ; preds = %216
  %221 = load ptr, ptr %153, align 8
  %222 = icmp eq ptr %221, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %222, label %223, label %254

223:                                              ; preds = %220
  %224 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %153, i32 0, i32 2
  %225 = load i32, ptr %224, align 4
  %226 = and i32 %225, 1073741824
  %227 = icmp ne i32 %226, 0
  br i1 %227, label %228, label %244

228:                                              ; preds = %223
  %229 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %153, i32 0, i32 4
  %230 = load ptr, ptr %229, align 8
  %231 = icmp ne ptr %230, null
  br i1 %231, label %232, label %234

232:                                              ; preds = %228
  invoke void %230(ptr %3, ptr %152)
          to label %.noexc7 unwind label %endcatch.resume3

.noexc7:                                          ; preds = %232
  %233 = load ptr, ptr %3, align 8
  br label %241

234:                                              ; preds = %228
  %235 = load ptr, ptr %152, align 8
  %236 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume3

.noexc19:                                         ; preds = %234
  %237 = getelementptr i8, ptr %236, i64 8
  %238 = getelementptr { i64, ptr, ptr }, ptr %237, i32 0, i32 0
  store i64 8, ptr %238, align 8
  %239 = getelementptr { i64, ptr, ptr }, ptr %237, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %239, align 8
  %240 = getelementptr { i64, ptr, ptr }, ptr %237, i32 0, i32 2
  store ptr %235, ptr %240, align 8
  br label %241

241:                                              ; preds = %.noexc7, %.noexc19
  %242 = phi ptr [ %237, %.noexc19 ], [ %233, %.noexc7 ]
  br label %243

243:                                              ; preds = %241
  br label %251

244:                                              ; preds = %223
  %245 = load ptr, ptr %152, align 8
  %246 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume3

.noexc14:                                         ; preds = %244
  %247 = getelementptr i8, ptr %246, i64 8
  %248 = getelementptr { i64, ptr, ptr }, ptr %247, i32 0, i32 0
  store i64 8, ptr %248, align 8
  %249 = getelementptr { i64, ptr, ptr }, ptr %247, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %249, align 8
  %250 = getelementptr { i64, ptr, ptr }, ptr %247, i32 0, i32 2
  store ptr %245, ptr %250, align 8
  br label %251

251:                                              ; preds = %243, %.noexc14
  %252 = phi ptr [ %247, %.noexc14 ], [ %242, %243 ]
  br label %253

253:                                              ; preds = %251
  br label %261

254:                                              ; preds = %220
  %255 = load ptr, ptr %152, align 8
  %256 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume3

.noexc6:                                          ; preds = %254
  %257 = getelementptr i8, ptr %256, i64 8
  %258 = getelementptr { i64, ptr, ptr }, ptr %257, i32 0, i32 0
  store i64 8, ptr %258, align 8
  %259 = getelementptr { i64, ptr, ptr }, ptr %257, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %259, align 8
  %260 = getelementptr { i64, ptr, ptr }, ptr %257, i32 0, i32 2
  store ptr %255, ptr %260, align 8
  br label %261

261:                                              ; preds = %253, %.noexc6
  %262 = phi ptr [ %257, %.noexc6 ], [ %252, %253 ]
  br label %263

263:                                              ; preds = %261
  br label %264

264:                                              ; preds = %218, %263
  %265 = phi ptr [ %262, %263 ], [ %219, %218 ]
  br label %266

266:                                              ; preds = %264
  br label %267

267:                                              ; preds = %.noexc4, %266
  %268 = phi ptr [ %265, %266 ], [ %212, %.noexc4 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %.noexc16, %269
  %271 = phi ptr [ %268, %269 ], [ %203, %.noexc16 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc12, %272
  %274 = phi ptr [ %271, %272 ], [ %194, %.noexc12 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc8, %275
  %277 = phi ptr [ %274, %275 ], [ %185, %.noexc8 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc20, %278
  %280 = phi ptr [ %277, %278 ], [ %176, %.noexc20 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc26, %281
  %283 = phi ptr [ %280, %281 ], [ %167, %.noexc26 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc27, %284
  %286 = phi ptr [ %283, %284 ], [ %158, %.noexc27 ]
  br label %287

287:                                              ; preds = %285
  store ptr %286, ptr %4, align 8
  %288 = invoke ptr @__cxa_allocate_exception(i64 8)
          to label %.noexc21 unwind label %endcatch.resume3

.noexc21:                                         ; preds = %287
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %288, align 8
  invoke void @__cxa_throw(ptr %288, ptr @_ZTIPc, ptr null)
          to label %494 unwind label %289

289:                                              ; preds = %.noexc21
  %290 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %291 = extractvalue { ptr, i32 } %290, 1
  %292 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %293 = icmp eq i32 %291, %292
  %294 = extractvalue { ptr, i32 } %290, 1
  %295 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %296 = icmp eq i32 %294, %295
  %297 = extractvalue { ptr, i32 } %290, 1
  %298 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %299 = icmp eq i32 %297, %298
  %300 = extractvalue { ptr, i32 } %290, 1
  %301 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %302 = icmp eq i32 %300, %301
  %303 = or i1 %293, %296
  %304 = or i1 %303, %299
  %305 = or i1 %304, %302
  br i1 %305, label %309, label %306

306:                                              ; preds = %289
  %307 = extractvalue { ptr, i32 } %290, 0
  %308 = call ptr @__cxa_begin_catch(ptr %307)
  invoke void @__cxa_rethrow()
          to label %494 unwind label %endcatch.pad2

309:                                              ; preds = %289
  %310 = extractvalue { ptr, i32 } %290, 0
  %311 = call ptr @__cxa_begin_catch(ptr %310)
  %312 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc37 unwind label %endcatch.resume36

.noexc37:                                         ; preds = %309
  %313 = icmp eq ptr %312, @_ZTIi
  br i1 %313, label %314, label %317

314:                                              ; preds = %.noexc37
  %315 = load i32, ptr %311, align 4
  %316 = sitofp i32 %315 to double
  br label %335

317:                                              ; preds = %.noexc37
  %318 = icmp eq ptr %312, @_ZTIj
  br i1 %318, label %319, label %322

319:                                              ; preds = %317
  %320 = load i32, ptr %311, align 4
  %321 = uitofp i32 %320 to double
  br label %332

322:                                              ; preds = %317
  %323 = icmp eq ptr %312, @_ZTIf
  br i1 %323, label %324, label %327

324:                                              ; preds = %322
  %325 = load float, ptr %311, align 4
  %326 = fpext float %325 to double
  br label %329

327:                                              ; preds = %322
  %328 = load double, ptr %311, align 8
  br label %329

329:                                              ; preds = %324, %327
  %330 = phi double [ %328, %327 ], [ %326, %324 ]
  br label %331

331:                                              ; preds = %329
  br label %332

332:                                              ; preds = %319, %331
  %333 = phi double [ %330, %331 ], [ %321, %319 ]
  br label %334

334:                                              ; preds = %332
  br label %335

335:                                              ; preds = %314, %334
  %336 = phi double [ %333, %334 ], [ %316, %314 ]
  br label %337

337:                                              ; preds = %335
  store double %336, ptr %2, align 8
  %338 = load i32, ptr @wrong, align 4
  %339 = add i32 %338, 1
  store i32 %339, ptr @wrong, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  %340 = load i32, ptr @finals, align 4
  %341 = add i32 %340, 1
  store i32 %341, ptr @finals, align 4
  br label %493

endcatch.pad2:                                    ; preds = %306
  %342 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught1

343:                                              ; No predecessors!
  %344 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %343
  %345 = phi { ptr, i32 } [ %344, %343 ], [ %342, %endcatch.pad2 ]
  %346 = extractvalue { ptr, i32 } %345, 0
  %347 = call ptr @__cxa_begin_catch(ptr %346)
  %348 = load i32, ptr @finals, align 4
  %349 = add i32 %348, 1
  store i32 %349, ptr @finals, align 4
  invoke void @__cxa_rethrow()
          to label %494 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %.caught1
  %350 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

351:                                              ; No predecessors!
  %352 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %351
  %353 = phi { ptr, i32 } [ %352, %351 ], [ %350, %endcatch.pad ]
  %354 = extractvalue { ptr, i32 } %353, 0
  %355 = call ptr @__cxa_begin_catch(ptr %354)
  %356 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc38 unwind label %endcatch.resume

.noexc38:                                         ; preds = %.caught
  %357 = icmp eq ptr %356, @_ZTIi
  br i1 %357, label %358, label %365

358:                                              ; preds = %.noexc38
  %359 = load i32, ptr %355, align 4
  %360 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %358
  %361 = getelementptr i8, ptr %360, i64 8
  %362 = getelementptr { i64, ptr, i32 }, ptr %361, i32 0, i32 0
  store i64 4, ptr %362, align 8
  %363 = getelementptr { i64, ptr, i32 }, ptr %361, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %363, align 8
  %364 = getelementptr { i64, ptr, i32 }, ptr %361, i32 0, i32 2
  store i32 %359, ptr %364, align 4
  br label %488

365:                                              ; preds = %.noexc38
  %366 = icmp eq ptr %356, @_ZTIj
  br i1 %366, label %367, label %374

367:                                              ; preds = %365
  %368 = load i32, ptr %355, align 4
  %369 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %367
  %370 = getelementptr i8, ptr %369, i64 8
  %371 = getelementptr { i64, ptr, i32 }, ptr %370, i32 0, i32 0
  store i64 4, ptr %371, align 8
  %372 = getelementptr { i64, ptr, i32 }, ptr %370, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %372, align 8
  %373 = getelementptr { i64, ptr, i32 }, ptr %370, i32 0, i32 2
  store i32 %368, ptr %373, align 4
  br label %485

374:                                              ; preds = %365
  %375 = icmp eq ptr %356, @_ZTId
  br i1 %375, label %376, label %383

376:                                              ; preds = %374
  %377 = load double, ptr %355, align 8
  %378 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc42 unwind label %endcatch.resume

.noexc42:                                         ; preds = %376
  %379 = getelementptr i8, ptr %378, i64 8
  %380 = getelementptr { i64, ptr, double }, ptr %379, i32 0, i32 0
  store i64 8, ptr %380, align 8
  %381 = getelementptr { i64, ptr, double }, ptr %379, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %381, align 8
  %382 = getelementptr { i64, ptr, double }, ptr %379, i32 0, i32 2
  store double %377, ptr %382, align 8
  br label %482

383:                                              ; preds = %374
  %384 = icmp eq ptr %356, @_ZTIf
  br i1 %384, label %385, label %392

385:                                              ; preds = %383
  %386 = load float, ptr %355, align 4
  %387 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc40 unwind label %endcatch.resume

.noexc40:                                         ; preds = %385
  %388 = getelementptr i8, ptr %387, i64 8
  %389 = getelementptr { i64, ptr, float }, ptr %388, i32 0, i32 0
  store i64 4, ptr %389, align 8
  %390 = getelementptr { i64, ptr, float }, ptr %388, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %390, align 8
  %391 = getelementptr { i64, ptr, float }, ptr %388, i32 0, i32 2
  store float %386, ptr %391, align 4
  br label %479

392:                                              ; preds = %383
  %393 = icmp eq ptr %356, @_ZTIb
  br i1 %393, label %394, label %401

394:                                              ; preds = %392
  %395 = load i1, ptr %355, align 1
  %396 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %394
  %397 = getelementptr i8, ptr %396, i64 8
  %398 = getelementptr { i64, ptr, i1 }, ptr %397, i32 0, i32 0
  store i64 1, ptr %398, align 8
  %399 = getelementptr { i64, ptr, i1 }, ptr %397, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %399, align 8
  %400 = getelementptr { i64, ptr, i1 }, ptr %397, i32 0, i32 2
  store i1 %395, ptr %400, align 1
  br label %476

401:                                              ; preds = %392
  %402 = icmp eq ptr %356, @_ZTIx
  br i1 %402, label %403, label %410

403:                                              ; preds = %401
  %404 = load i64, ptr %355, align 8
  %405 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %403
  %406 = getelementptr i8, ptr %405, i64 8
  %407 = getelementptr { i64, ptr, i64 }, ptr %406, i32 0, i32 0
  store i64 8, ptr %407, align 8
  %408 = getelementptr { i64, ptr, i64 }, ptr %406, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %408, align 8
  %409 = getelementptr { i64, ptr, i64 }, ptr %406, i32 0, i32 2
  store i64 %404, ptr %409, align 8
  br label %473

410:                                              ; preds = %401
  %411 = icmp eq ptr %356, @_ZTIPc
  br i1 %411, label %412, label %419

412:                                              ; preds = %410
  %413 = load ptr, ptr %355, align 8
  %414 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %412
  %415 = getelementptr i8, ptr %414, i64 8
  %416 = getelementptr { i64, ptr, ptr }, ptr %415, i32 0, i32 0
  store i64 8, ptr %416, align 8
  %417 = getelementptr { i64, ptr, ptr }, ptr %415, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %417, align 8
  %418 = getelementptr { i64, ptr, ptr }, ptr %415, i32 0, i32 2
  store ptr %413, ptr %418, align 8
  br label %470

419:                                              ; preds = %410
  %420 = icmp eq ptr %356, @_ZTIPv
  br i1 %420, label %421, label %423

421:                                              ; preds = %419
  %422 = load ptr, ptr %355, align 8
  br label %467

423:                                              ; preds = %419
  %424 = load ptr, ptr %356, align 8
  %425 = icmp eq ptr %424, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %425, label %426, label %457

426:                                              ; preds = %423
  %427 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %356, i32 0, i32 2
  %428 = load i32, ptr %427, align 4
  %429 = and i32 %428, 1073741824
  %430 = icmp ne i32 %429, 0
  br i1 %430, label %431, label %447

431:                                              ; preds = %426
  %432 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %356, i32 0, i32 4
  %433 = load ptr, ptr %432, align 8
  %434 = icmp ne ptr %433, null
  br i1 %434, label %435, label %437

435:                                              ; preds = %431
  invoke void %433(ptr %1, ptr %355)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %435
  %436 = load ptr, ptr %1, align 8
  br label %444

437:                                              ; preds = %431
  %438 = load ptr, ptr %355, align 8
  %439 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %437
  %440 = getelementptr i8, ptr %439, i64 8
  %441 = getelementptr { i64, ptr, ptr }, ptr %440, i32 0, i32 0
  store i64 8, ptr %441, align 8
  %442 = getelementptr { i64, ptr, ptr }, ptr %440, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %442, align 8
  %443 = getelementptr { i64, ptr, ptr }, ptr %440, i32 0, i32 2
  store ptr %438, ptr %443, align 8
  br label %444

444:                                              ; preds = %.noexc10, %.noexc5
  %445 = phi ptr [ %440, %.noexc5 ], [ %436, %.noexc10 ]
  br label %446

446:                                              ; preds = %444
  br label %454

447:                                              ; preds = %426
  %448 = load ptr, ptr %355, align 8
  %449 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %447
  %450 = getelementptr i8, ptr %449, i64 8
  %451 = getelementptr { i64, ptr, ptr }, ptr %450, i32 0, i32 0
  store i64 8, ptr %451, align 8
  %452 = getelementptr { i64, ptr, ptr }, ptr %450, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %452, align 8
  %453 = getelementptr { i64, ptr, ptr }, ptr %450, i32 0, i32 2
  store ptr %448, ptr %453, align 8
  br label %454

454:                                              ; preds = %446, %.noexc28
  %455 = phi ptr [ %450, %.noexc28 ], [ %445, %446 ]
  br label %456

456:                                              ; preds = %454
  br label %464

457:                                              ; preds = %423
  %458 = load ptr, ptr %355, align 8
  %459 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %457
  %460 = getelementptr i8, ptr %459, i64 8
  %461 = getelementptr { i64, ptr, ptr }, ptr %460, i32 0, i32 0
  store i64 8, ptr %461, align 8
  %462 = getelementptr { i64, ptr, ptr }, ptr %460, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %462, align 8
  %463 = getelementptr { i64, ptr, ptr }, ptr %460, i32 0, i32 2
  store ptr %458, ptr %463, align 8
  br label %464

464:                                              ; preds = %456, %.noexc23
  %465 = phi ptr [ %460, %.noexc23 ], [ %455, %456 ]
  br label %466

466:                                              ; preds = %464
  br label %467

467:                                              ; preds = %421, %466
  %468 = phi ptr [ %465, %466 ], [ %422, %421 ]
  br label %469

469:                                              ; preds = %467
  br label %470

470:                                              ; preds = %.noexc17, %469
  %471 = phi ptr [ %468, %469 ], [ %415, %.noexc17 ]
  br label %472

472:                                              ; preds = %470
  br label %473

473:                                              ; preds = %.noexc33, %472
  %474 = phi ptr [ %471, %472 ], [ %406, %.noexc33 ]
  br label %475

475:                                              ; preds = %473
  br label %476

476:                                              ; preds = %.noexc25, %475
  %477 = phi ptr [ %474, %475 ], [ %397, %.noexc25 ]
  br label %478

478:                                              ; preds = %476
  br label %479

479:                                              ; preds = %.noexc40, %478
  %480 = phi ptr [ %477, %478 ], [ %388, %.noexc40 ]
  br label %481

481:                                              ; preds = %479
  br label %482

482:                                              ; preds = %.noexc42, %481
  %483 = phi ptr [ %480, %481 ], [ %379, %.noexc42 ]
  br label %484

484:                                              ; preds = %482
  br label %485

485:                                              ; preds = %.noexc32, %484
  %486 = phi ptr [ %483, %484 ], [ %370, %.noexc32 ]
  br label %487

487:                                              ; preds = %485
  br label %488

488:                                              ; preds = %.noexc13, %487
  %489 = phi ptr [ %486, %487 ], [ %361, %.noexc13 ]
  br label %490

490:                                              ; preds = %488
  store ptr %489, ptr %7, align 8
  %491 = load i32, ptr @outer, align 4
  %492 = add i32 %491, 1
  store i32 %492, ptr @outer, align 4
  call void @__cxa_end_catch()
  br label %493

493:                                              ; preds = %337, %490
  ret void

494:                                              ; preds = %.caught1, %306, %.noexc21, %.noexc18, %0
  unreachable

endcatch.resume:                                  ; preds = %376, %92, %385, %60, %.caught, %33, %94, %403, %367, %114, %69, %447, %394, %42, %457, %24, %147, %412, %104, %358, %51, %435, %9, %437, %15
  %495 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %495

endcatch.resume3:                                 ; preds = %149, %155, %164, %287, %173, %234, %200, %244, %191, %182, %232, %254, %209
  %496 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %496

endcatch.resume36:                                ; preds = %309
  %497 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %497
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @reset()
  call void @inUntypedCatch()
  %1 = load i32, ptr @wrong, align 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %6

3:                                                ; preds = %0
  %4 = load i32, ptr @outer, align 4
  %5 = icmp eq i32 %4, 1
  br label %6

6:                                                ; preds = %3, %0
  %7 = phi i1 [ %5, %3 ], [ %2, %0 ]
  br i1 %7, label %8, label %62

8:                                                ; preds = %6
  call void @reset()
  call void @inTypedCatch()
  %9 = load i32, ptr @wrong, align 4
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %8
  %12 = load i32, ptr @outer, align 4
  %13 = icmp eq i32 %12, 1
  br label %14

14:                                               ; preds = %11, %8
  %15 = phi i1 [ %13, %11 ], [ %10, %8 ]
  br i1 %15, label %16, label %64

16:                                               ; preds = %14
  call void @reset()
  call void @withFinally()
  %17 = load i32, ptr @wrong, align 4
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = load i32, ptr @finals, align 4
  %21 = icmp eq i32 %20, 1
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi i1 [ %21, %19 ], [ %18, %16 ]
  br i1 %23, label %24, label %27

24:                                               ; preds = %22
  %25 = load i32, ptr @outer, align 4
  %26 = icmp eq i32 %25, 1
  br label %27

27:                                               ; preds = %24, %22
  %28 = phi i1 [ %26, %24 ], [ %23, %22 ]
  br i1 %28, label %29, label %66

29:                                               ; preds = %27
  call void @reset()
  call void @inParentFinally()
  %30 = load i32, ptr @disposed, align 4
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %35

32:                                               ; preds = %29
  %33 = load i32, ptr @finals, align 4
  %34 = icmp eq i32 %33, 11
  br label %35

35:                                               ; preds = %32, %29
  %36 = phi i1 [ %34, %32 ], [ %31, %29 ]
  br i1 %36, label %37, label %40

37:                                               ; preds = %35
  %38 = load i32, ptr @wrong, align 4
  %39 = icmp eq i32 %38, 0
  br label %40

40:                                               ; preds = %37, %35
  %41 = phi i1 [ %39, %37 ], [ %36, %35 ]
  br i1 %41, label %42, label %45

42:                                               ; preds = %40
  %43 = load i32, ptr @outer, align 4
  %44 = icmp eq i32 %43, 1
  br label %45

45:                                               ; preds = %42, %40
  %46 = phi i1 [ %44, %42 ], [ %41, %40 ]
  br i1 %46, label %47, label %68

47:                                               ; preds = %45
  call void @reset()
  call void @twoLevelsUp()
  %48 = load i32, ptr @wrong, align 4
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %50, label %53

50:                                               ; preds = %47
  %51 = load i32, ptr @finals, align 4
  %52 = icmp eq i32 %51, 1
  br label %53

53:                                               ; preds = %50, %47
  %54 = phi i1 [ %52, %50 ], [ %49, %47 ]
  br i1 %54, label %55, label %58

55:                                               ; preds = %53
  %56 = load i32, ptr @outer, align 4
  %57 = icmp eq i32 %56, 1
  br label %58

58:                                               ; preds = %55, %53
  %59 = phi i1 [ %57, %55 ], [ %54, %53 ]
  br i1 %59, label %60, label %70

60:                                               ; preds = %58
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

61:                                               ; preds = %70, %68, %66, %64, %62
  unreachable

62:                                               ; preds = %6
  %63 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15619039406129020236, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5027590903802435426, i64 8), i32 122, ptr null)
  br label %61

64:                                               ; preds = %14
  %65 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4788699618267080347, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5027590903802435426, i64 8), i32 126, ptr null)
  br label %61

66:                                               ; preds = %27
  %67 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2596540853700611608, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5027590903802435426, i64 8), i32 130, ptr null)
  br label %61

68:                                               ; preds = %45
  %69 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4837289883065086718, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5027590903802435426, i64 8), i32 134, ptr null)
  br label %61

70:                                               ; preds = %58
  %71 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7059949095878146050, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5027590903802435426, i64 8), i32 138, ptr null)
  br label %61
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

