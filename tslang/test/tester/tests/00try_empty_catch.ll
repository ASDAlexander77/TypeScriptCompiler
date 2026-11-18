; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Node = type { ptr, i32 }
%C = type { ptr }

$Node..size = comdat any

$C..size = comdat any

@s_7403105002735449741 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNode\00", align 8
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_3298874795566462716 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe finally after an empty catch runs\00", align 8
@f_8917624681063520109 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_empty_catch.ts\00", align 8
@m_15814985938145335932 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan inner empty catch lets the outer try go on\00", align 8
@m_15846919584820021358 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan empty catch in a loop\00", align 8
@m_6517057680562749814 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa try that does not throw\00", align 8
@m_15770550493454631089 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan empty catch in a method\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Node..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7403105002735449741, i64 8)
@Node..size = global i64 16, comdat
@Node..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Node..instanceOf, ptr @Node..new, ptr @Node..rtti, ptr @Node..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 8, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C.run, ptr @C..rtti, ptr @C..size }
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

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define void @Node.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 1
  store i32 0, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Node..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Node..rtti, align 8
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
define ptr @Node..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 0
  store ptr @Node..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @h() #1 {
  %1 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %1, align 4
  call void @__cxa_throw(ptr %1, ptr @_ZTIi, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @emptyCatch() #2 personality ptr @__gxx_personality_v0 {
  invoke void @h()
          to label %1 unwind label %2

1:                                                ; preds = %0
  br label %6

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          catch ptr null
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = call ptr @__cxa_begin_catch(ptr %4)
  call void @__cxa_end_catch()
  br label %6

6:                                                ; preds = %1, %2
  ret void
}

; Function Attrs: mustprogress
define void @emptyCatchWithVariable() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  invoke void @h()
          to label %3 unwind label %4

3:                                                ; preds = %0
  br label %143

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = call ptr @__cxa_begin_catch(ptr %6)
  %8 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %4
  %9 = icmp eq ptr %8, @_ZTIi
  br i1 %9, label %10, label %17

10:                                               ; preds = %.noexc9
  %11 = load i32, ptr %7, align 4
  %12 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %10
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr { i64, ptr, i32 }, ptr %13, i32 0, i32 0
  store i64 4, ptr %14, align 8
  %15 = getelementptr { i64, ptr, i32 }, ptr %13, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %15, align 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %13, i32 0, i32 2
  store i32 %11, ptr %16, align 4
  br label %140

17:                                               ; preds = %.noexc9
  %18 = icmp eq ptr %8, @_ZTIj
  br i1 %18, label %19, label %26

19:                                               ; preds = %17
  %20 = load i32, ptr %7, align 4
  %21 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %19
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 0
  store i64 4, ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 2
  store i32 %20, ptr %25, align 4
  br label %137

26:                                               ; preds = %17
  %27 = icmp eq ptr %8, @_ZTId
  br i1 %27, label %28, label %35

28:                                               ; preds = %26
  %29 = load double, ptr %7, align 8
  %30 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr { i64, ptr, double }, ptr %31, i32 0, i32 0
  store i64 8, ptr %32, align 8
  %33 = getelementptr { i64, ptr, double }, ptr %31, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %33, align 8
  %34 = getelementptr { i64, ptr, double }, ptr %31, i32 0, i32 2
  store double %29, ptr %34, align 8
  br label %134

35:                                               ; preds = %26
  %36 = icmp eq ptr %8, @_ZTIf
  br i1 %36, label %37, label %44

37:                                               ; preds = %35
  %38 = load float, ptr %7, align 4
  %39 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %37
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = getelementptr { i64, ptr, float }, ptr %40, i32 0, i32 0
  store i64 4, ptr %41, align 8
  %42 = getelementptr { i64, ptr, float }, ptr %40, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %42, align 8
  %43 = getelementptr { i64, ptr, float }, ptr %40, i32 0, i32 2
  store float %38, ptr %43, align 4
  br label %131

44:                                               ; preds = %35
  %45 = icmp eq ptr %8, @_ZTIb
  br i1 %45, label %46, label %53

46:                                               ; preds = %44
  %47 = load i1, ptr %7, align 1
  %48 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %46
  %49 = getelementptr i8, ptr %48, i64 8
  %50 = getelementptr { i64, ptr, i1 }, ptr %49, i32 0, i32 0
  store i64 1, ptr %50, align 8
  %51 = getelementptr { i64, ptr, i1 }, ptr %49, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %51, align 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %49, i32 0, i32 2
  store i1 %47, ptr %52, align 1
  br label %128

53:                                               ; preds = %44
  %54 = icmp eq ptr %8, @_ZTIx
  br i1 %54, label %55, label %62

55:                                               ; preds = %53
  %56 = load i64, ptr %7, align 8
  %57 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %55
  %58 = getelementptr i8, ptr %57, i64 8
  %59 = getelementptr { i64, ptr, i64 }, ptr %58, i32 0, i32 0
  store i64 8, ptr %59, align 8
  %60 = getelementptr { i64, ptr, i64 }, ptr %58, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %60, align 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %58, i32 0, i32 2
  store i64 %56, ptr %61, align 8
  br label %125

62:                                               ; preds = %53
  %63 = icmp eq ptr %8, @_ZTIPc
  br i1 %63, label %64, label %71

64:                                               ; preds = %62
  %65 = load ptr, ptr %7, align 8
  %66 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %64
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr { i64, ptr, ptr }, ptr %67, i32 0, i32 0
  store i64 8, ptr %68, align 8
  %69 = getelementptr { i64, ptr, ptr }, ptr %67, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %69, align 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %67, i32 0, i32 2
  store ptr %65, ptr %70, align 8
  br label %122

71:                                               ; preds = %62
  %72 = icmp eq ptr %8, @_ZTIPv
  br i1 %72, label %73, label %75

73:                                               ; preds = %71
  %74 = load ptr, ptr %7, align 8
  br label %119

75:                                               ; preds = %71
  %76 = load ptr, ptr %8, align 8
  %77 = icmp eq ptr %76, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %77, label %78, label %109

78:                                               ; preds = %75
  %79 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %8, i32 0, i32 2
  %80 = load i32, ptr %79, align 4
  %81 = and i32 %80, 1073741824
  %82 = icmp ne i32 %81, 0
  br i1 %82, label %83, label %99

83:                                               ; preds = %78
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %8, i32 0, i32 4
  %85 = load ptr, ptr %84, align 8
  %86 = icmp ne ptr %85, null
  br i1 %86, label %87, label %89

87:                                               ; preds = %83
  invoke void %85(ptr %1, ptr %7)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %87
  %88 = load ptr, ptr %1, align 8
  br label %96

89:                                               ; preds = %83
  %90 = load ptr, ptr %7, align 8
  %91 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %89
  %92 = getelementptr i8, ptr %91, i64 8
  %93 = getelementptr { i64, ptr, ptr }, ptr %92, i32 0, i32 0
  store i64 8, ptr %93, align 8
  %94 = getelementptr { i64, ptr, ptr }, ptr %92, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %94, align 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %92, i32 0, i32 2
  store ptr %90, ptr %95, align 8
  br label %96

96:                                               ; preds = %.noexc1, %.noexc5
  %97 = phi ptr [ %92, %.noexc5 ], [ %88, %.noexc1 ]
  br label %98

98:                                               ; preds = %96
  br label %106

99:                                               ; preds = %78
  %100 = load ptr, ptr %7, align 8
  %101 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %99
  %102 = getelementptr i8, ptr %101, i64 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 0
  store i64 8, ptr %103, align 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %104, align 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 2
  store ptr %100, ptr %105, align 8
  br label %106

106:                                              ; preds = %98, %.noexc7
  %107 = phi ptr [ %102, %.noexc7 ], [ %97, %98 ]
  br label %108

108:                                              ; preds = %106
  br label %116

109:                                              ; preds = %75
  %110 = load ptr, ptr %7, align 8
  %111 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %109
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 0
  store i64 8, ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 2
  store ptr %110, ptr %115, align 8
  br label %116

116:                                              ; preds = %108, %.noexc10
  %117 = phi ptr [ %112, %.noexc10 ], [ %107, %108 ]
  br label %118

118:                                              ; preds = %116
  br label %119

119:                                              ; preds = %73, %118
  %120 = phi ptr [ %117, %118 ], [ %74, %73 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %.noexc8, %121
  %123 = phi ptr [ %120, %121 ], [ %67, %.noexc8 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc2, %124
  %126 = phi ptr [ %123, %124 ], [ %58, %.noexc2 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc11, %127
  %129 = phi ptr [ %126, %127 ], [ %49, %.noexc11 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc4, %130
  %132 = phi ptr [ %129, %130 ], [ %40, %.noexc4 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc, %133
  %135 = phi ptr [ %132, %133 ], [ %31, %.noexc ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc3, %136
  %138 = phi ptr [ %135, %136 ], [ %22, %.noexc3 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc6, %139
  %141 = phi ptr [ %138, %139 ], [ %13, %.noexc6 ]
  br label %142

142:                                              ; preds = %140
  store ptr %141, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %143

143:                                              ; preds = %3, %142
  ret void

endcatch.resume:                                  ; preds = %46, %109, %4, %64, %99, %10, %89, %37, %19, %55, %87, %28
  %144 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %144
}

; Function Attrs: mustprogress
define void @emptyStatementInCatch() #2 personality ptr @__gxx_personality_v0 {
  invoke void @h()
          to label %1 unwind label %2

1:                                                ; preds = %0
  br label %6

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          catch ptr null
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = call ptr @__cxa_begin_catch(ptr %4)
  call void @__cxa_end_catch()
  br label %6

6:                                                ; preds = %1, %2
  ret void
}

; Function Attrs: mustprogress
define i1 @emptyCatchThenFinally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca i1, align 1
  store i1 false, ptr %1, align 1
  invoke void @h()
          to label %3 unwind label %4

3:                                                ; preds = %0
  br label %8

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = call ptr @__cxa_begin_catch(ptr %6)
  call void @__cxa_end_catch()
  br label %8

8:                                                ; preds = %3, %4
  store i1 true, ptr %1, align 1
  %9 = load i1, ptr %1, align 1
  store i1 %9, ptr %2, align 1
  %10 = load i1, ptr %2, align 1
  ret i1 %10
}

; Function Attrs: mustprogress
define void @emptyCatchWithOwnedLocal() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 0
  store ptr @Node..vtbl, ptr %4, align 8
  invoke void @Node.constructor(ptr %3)
          to label %5 unwind label %9

5:                                                ; preds = %0
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  %7 = getelementptr %Node, ptr %6, i32 0, i32 1
  store i32 1, ptr %7, align 4
  invoke void @h()
          to label %8 unwind label %9

8:                                                ; preds = %5
  br label %13

9:                                                ; preds = %5, %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  call void @__cxa_end_catch()
  br label %13

13:                                               ; preds = %8, %9
  ret void
}

; Function Attrs: mustprogress
define i1 @nestedEmptyCatches() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca i1, align 1
  store i1 false, ptr %1, align 1
  invoke void @h()
          to label %3 unwind label %4

3:                                                ; preds = %0
  br label %8

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = call ptr @__cxa_begin_catch(ptr %6)
  call void @__cxa_end_catch()
  br label %8

8:                                                ; preds = %3, %4
  store i1 true, ptr %1, align 1
  invoke void @h()
          to label %9 unwind label %10

9:                                                ; preds = %8
  br label %14

10:                                               ; preds = %8
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  call void @__cxa_end_catch()
  br label %14

14:                                               ; preds = %9, %10
  %15 = load i1, ptr %1, align 1
  store i1 %15, ptr %2, align 1
  %16 = load i1, ptr %2, align 1
  ret i1 %16
}

; Function Attrs: mustprogress
define i32 @emptyCatchInLoop() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %13, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 5
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  invoke void @h()
          to label %8 unwind label %9

8:                                                ; preds = %7
  br label %13

9:                                                ; preds = %7
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  call void @__cxa_end_catch()
  br label %13

13:                                               ; preds = %8, %9
  %14 = load i32, ptr %2, align 4
  %15 = add i32 %14, 1
  store i32 %15, ptr %2, align 4
  %16 = load i32, ptr %1, align 4
  %17 = add i32 %16, 1
  store i32 %17, ptr %1, align 4
  br label %4

18:                                               ; preds = %4
  %19 = load i32, ptr %2, align 4
  store i32 %19, ptr %3, align 4
  %20 = load i32, ptr %3, align 4
  ret i32 %20
}

; Function Attrs: mustprogress
define i32 @noThrowEmptyCatch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 7, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  store i32 %3, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  ret i32 %4
}

; Function Attrs: mustprogress
define i1 @C..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @C..rtti, align 8
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
define ptr @C..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %C, ptr %3, i32 0, i32 0
  store ptr @C..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i1 @C.run(ptr %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  invoke void @h()
          to label %3 unwind label %4

3:                                                ; preds = %1
  br label %8

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = call ptr @__cxa_begin_catch(ptr %6)
  call void @__cxa_end_catch()
  br label %8

8:                                                ; preds = %3, %4
  store i1 true, ptr %2, align 1
  %9 = load i1, ptr %2, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca <{ i64, %C }>, align 16
  store i64 -1, ptr %1, align 8
  %2 = getelementptr i8, ptr %1, i64 8
  store %C zeroinitializer, ptr %2, align 8
  call void @emptyCatch()
  call void @emptyCatchWithVariable()
  call void @emptyStatementInCatch()
  %3 = call i1 @emptyCatchThenFinally()
  br i1 %3, label %4, label %18

4:                                                ; preds = %0
  call void @emptyCatchWithOwnedLocal()
  %5 = call i1 @nestedEmptyCatches()
  br i1 %5, label %6, label %20

6:                                                ; preds = %4
  %7 = call i32 @emptyCatchInLoop()
  %8 = icmp eq i32 %7, 5
  br i1 %8, label %9, label %22

9:                                                ; preds = %6
  %10 = call i32 @noThrowEmptyCatch()
  %11 = icmp eq i32 %10, 7
  br i1 %11, label %12, label %24

12:                                               ; preds = %9
  store %C zeroinitializer, ptr %2, align 8
  %13 = getelementptr %C, ptr %2, i32 0, i32 0
  store ptr @C..vtbl, ptr %13, align 8
  %14 = getelementptr %C, ptr %2, i32 0, i32 0
  %15 = call i1 @C.run(ptr %2)
  br i1 %15, label %16, label %26

16:                                               ; preds = %12
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

17:                                               ; preds = %26, %24, %22, %20, %18
  unreachable

18:                                               ; preds = %0
  %19 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3298874795566462716, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8917624681063520109, i64 8), i32 110, ptr null)
  br label %17

20:                                               ; preds = %4
  %21 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15814985938145335932, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8917624681063520109, i64 8), i32 112, ptr null)
  br label %17

22:                                               ; preds = %6
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15846919584820021358, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8917624681063520109, i64 8), i32 113, ptr null)
  br label %17

24:                                               ; preds = %9
  %25 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6517057680562749814, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8917624681063520109, i64 8), i32 114, ptr null)
  br label %17

26:                                               ; preds = %12
  %27 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15770550493454631089, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8917624681063520109, i64 8), i32 115, ptr null)
  br label %17
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

