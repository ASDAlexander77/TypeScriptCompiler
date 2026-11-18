; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Node = type { ptr, i32, ptr }

$Node..size = comdat any

@s_7403105002735449741 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNode\00", align 8
@s_12638185001532385574 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFg\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_2770870231077248403 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrown\00", align 8
@s_16713252134513893189 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreplaced\00", align 8
@m_16525322597876987486 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/catch\00", align 8
@f_10815677319114142208 = internal constant [120 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_shared_global_argument_throw.ts\00", align 8
@m_291859784901123456 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe hold given back before the catch runs\00", align 8
@m_4113272673747287148 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe hold given back before the finally runs\00", align 8
@m_707516295030849314 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/finally\00", align 8
@s_2638862777294684392 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFlocal\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_17819851308403393800 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa block with a local in a try body\00", align 8
@s_9932668635479743809 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfirst\00", align 8
@m_11395844481985125100 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa call in a catch clause with a finally\00", align 8
@m_937078698315841043 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa try in a loop\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_2393915098112578145 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFneg\00", align 8
@m_17268163596755564902 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe normal path\00", align 8
@Node..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7403105002735449741, i64 8)
@Node..size = global i64 24, comdat
@Node..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Node..instanceOf, ptr @Node..new, ptr @Node..rtti, ptr @Node..size }
@g = private global ptr undef
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
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @g__cctor, ptr null }]

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__cxa_rethrow()

declare void @puts(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

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

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

define internal void @g__cctor() {
  %1 = call ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_12638185001532385574, i64 8), double 1.000000e+00)
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  store ptr %1, ptr %3, align 8
  store ptr %3, ptr @g, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Node.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 1
  store i32 0, ptr %4, align 4
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Node, ptr %5, i32 0, i32 2
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
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
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 0
  store ptr @Node..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i1 @countIs(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  br i1 true, label %10, label %7

7:                                                ; preds = %2
  %8 = load double, ptr %3, align 8
  %9 = fcmp oeq double -1.000000e+00, %8
  br label %10

10:                                               ; preds = %7, %2
  %11 = phi i1 [ %9, %7 ], [ true, %2 ]
  store i1 %11, ptr %5, align 1
  %12 = load i1, ptr %5, align 1
  ret i1 %12
}

; Function Attrs: mustprogress
define ptr @mk(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = call ptr @GC_malloc(i64 32)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Node, ptr %7, i32 0, i32 0
  store ptr @Node..vtbl, ptr %8, align 8
  call void @Node.constructor(ptr %7)
  %9 = getelementptr %Node, ptr %7, i32 0, i32 2
  %10 = load ptr, ptr %4, align 8
  store ptr %10, ptr %9, align 8
  %11 = getelementptr %Node, ptr %7, i32 0, i32 1
  %12 = load double, ptr %3, align 8
  %13 = fptosi double %12 to i64
  %14 = trunc i64 %13 to i32
  store i32 %14, ptr %11, align 4
  store ptr %7, ptr %5, align 8
  %15 = load ptr, ptr %5, align 8
  ret ptr %15
}

; Function Attrs: mustprogress
define i32 @maybeAssign(ptr %0, i1 %1) #1 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i1 %1, ptr %3, align 1
  %6 = load i1, ptr %3, align 1
  br i1 %6, label %7, label %11

7:                                                ; preds = %2
  %8 = call ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8), double 2.000000e+00)
  %9 = call ptr @GC_malloc(i64 16)
  %10 = getelementptr i8, ptr %9, i64 8
  store ptr %8, ptr %10, align 8
  store ptr %10, ptr @g, align 8
  br label %11

11:                                               ; preds = %7, %2
  %12 = load ptr, ptr %4, align 8
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %Node, ptr %13, i32 0, i32 1
  %15 = load i32, ptr %14, align 4
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %19

17:                                               ; preds = %11
  %18 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2770870231077248403, i64 8), ptr %18, align 8
  call void @__cxa_throw(ptr %18, ptr @_ZTIPc, ptr null)
  br label %25

19:                                               ; preds = %11
  %20 = load ptr, ptr %4, align 8
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr %Node, ptr %21, i32 0, i32 1
  %23 = load i32, ptr %22, align 4
  store i32 %23, ptr %5, align 4
  %24 = load i32, ptr %5, align 4
  ret i32 %24

25:                                               ; preds = %17
  unreachable
}

; Function Attrs: mustprogress
define void @inTryCatch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = load ptr, ptr @g, align 8
  %4 = invoke i32 @maybeAssign(ptr %3, i1 false)
          to label %5 unwind label %6

5:                                                ; preds = %0
  br label %148

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc10
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc10
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %21
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %30
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
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %66
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
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc14, %.noexc4
  %99 = phi ptr [ %94, %.noexc4 ], [ %90, %.noexc14 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc8
  %109 = phi ptr [ %104, %.noexc8 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc9
  %119 = phi ptr [ %114, %.noexc9 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc12, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc12 ]
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

133:                                              ; preds = %.noexc3, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc3 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc11, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc11 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc7, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc7 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr @g, align 8
  %146 = invoke i1 @countIs(ptr %145, double 1.000000e+00)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %144
  br i1 %146, label %147, label %155

147:                                              ; preds = %.noexc13
  call void @__cxa_end_catch()
  br label %148

148:                                              ; preds = %5, %147
  %149 = load ptr, ptr @g, align 8
  %150 = call i1 @countIs(ptr %149, double 1.000000e+00)
  br i1 %150, label %151, label %153

151:                                              ; preds = %148
  ret void

152:                                              ; preds = %153, %.noexc6
  unreachable

153:                                              ; preds = %148
  %154 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16525322597876987486, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 45, ptr null)
  br label %152

155:                                              ; preds = %.noexc13
  %156 = invoke i32 @fflush(ptr null)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %155
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_291859784901123456, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 42, ptr null)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  br label %152

endcatch.resume:                                  ; preds = %89, %144, %66, %21, %6, %111, %101, %12, %.noexc5, %155, %91, %39, %57, %48, %30
  %157 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %157
}

; Function Attrs: mustprogress
define void @inTryFinally() #2 personality ptr @__gxx_personality_v0 {
  %1 = load ptr, ptr @g, align 8
  %2 = invoke i32 @maybeAssign(ptr %1, i1 false)
          to label %3 unwind label %7

3:                                                ; preds = %0
  %4 = load ptr, ptr @g, align 8
  %5 = call i1 @countIs(ptr %4, double 1.000000e+00)
  br i1 %5, label %6, label %13

6:                                                ; preds = %3
  ret void

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          cleanup
  %9 = load ptr, ptr @g, align 8
  %10 = call i1 @countIs(ptr %9, double 1.000000e+00)
  br i1 %10, label %11, label %15

11:                                               ; preds = %7
  resume { ptr, i32 } %8

12:                                               ; preds = %13, %15
  unreachable

13:                                               ; preds = %3
  %14 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4113272673747287148, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 52, ptr null)
  br label %12

15:                                               ; preds = %7
  %16 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4113272673747287148, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 52, ptr null)
  br label %12
}

; Function Attrs: mustprogress
define void @throughFinally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  invoke void @inTryFinally()
          to label %3 unwind label %4

3:                                                ; preds = %0
  br label %143

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = call ptr @__cxa_begin_catch(ptr %6)
  %8 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %4
  %9 = icmp eq ptr %8, @_ZTIi
  br i1 %9, label %10, label %17

10:                                               ; preds = %.noexc2
  %11 = load i32, ptr %7, align 4
  %12 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %10
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr { i64, ptr, i32 }, ptr %13, i32 0, i32 0
  store i64 4, ptr %14, align 8
  %15 = getelementptr { i64, ptr, i32 }, ptr %13, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %15, align 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %13, i32 0, i32 2
  store i32 %11, ptr %16, align 4
  br label %140

17:                                               ; preds = %.noexc2
  %18 = icmp eq ptr %8, @_ZTIj
  br i1 %18, label %19, label %26

19:                                               ; preds = %17
  %20 = load i32, ptr %7, align 4
  %21 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %19
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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %28
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %37
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
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %46
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %55
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %64
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %87
  %88 = load ptr, ptr %1, align 8
  br label %96

89:                                               ; preds = %83
  %90 = load ptr, ptr %7, align 8
  %91 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %89
  %92 = getelementptr i8, ptr %91, i64 8
  %93 = getelementptr { i64, ptr, ptr }, ptr %92, i32 0, i32 0
  store i64 8, ptr %93, align 8
  %94 = getelementptr { i64, ptr, ptr }, ptr %92, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %94, align 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %92, i32 0, i32 2
  store ptr %90, ptr %95, align 8
  br label %96

96:                                               ; preds = %.noexc7, %.noexc
  %97 = phi ptr [ %92, %.noexc ], [ %88, %.noexc7 ]
  br label %98

98:                                               ; preds = %96
  br label %106

99:                                               ; preds = %78
  %100 = load ptr, ptr %7, align 8
  %101 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %99
  %102 = getelementptr i8, ptr %101, i64 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 0
  store i64 8, ptr %103, align 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %104, align 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 2
  store ptr %100, ptr %105, align 8
  br label %106

106:                                              ; preds = %98, %.noexc8
  %107 = phi ptr [ %102, %.noexc8 ], [ %97, %98 ]
  br label %108

108:                                              ; preds = %106
  br label %116

109:                                              ; preds = %75
  %110 = load ptr, ptr %7, align 8
  %111 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %109
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 0
  store i64 8, ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 2
  store ptr %110, ptr %115, align 8
  br label %116

116:                                              ; preds = %108, %.noexc4
  %117 = phi ptr [ %112, %.noexc4 ], [ %107, %108 ]
  br label %118

118:                                              ; preds = %116
  br label %119

119:                                              ; preds = %73, %118
  %120 = phi ptr [ %117, %118 ], [ %74, %73 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %.noexc6, %121
  %123 = phi ptr [ %120, %121 ], [ %67, %.noexc6 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc9, %124
  %126 = phi ptr [ %123, %124 ], [ %58, %.noexc9 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc3, %127
  %129 = phi ptr [ %126, %127 ], [ %49, %.noexc3 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc11, %130
  %132 = phi ptr [ %129, %130 ], [ %40, %.noexc11 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc1, %133
  %135 = phi ptr [ %132, %133 ], [ %31, %.noexc1 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc5, %136
  %138 = phi ptr [ %135, %136 ], [ %22, %.noexc5 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc10, %139
  %141 = phi ptr [ %138, %139 ], [ %13, %.noexc10 ]
  br label %142

142:                                              ; preds = %140
  store ptr %141, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %143

143:                                              ; preds = %3, %142
  %144 = load ptr, ptr @g, align 8
  %145 = call i1 @countIs(ptr %144, double 1.000000e+00)
  br i1 %145, label %146, label %148

146:                                              ; preds = %143
  ret void

147:                                              ; preds = %148
  unreachable

148:                                              ; preds = %143
  %149 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_707516295030849314, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 62, ptr null)
  br label %147

endcatch.resume:                                  ; preds = %37, %10, %55, %99, %87, %64, %19, %109, %46, %4, %28, %89
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define void @inBlockWithLocal() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = invoke ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_2638862777294684392, i64 8), double 3.000000e+00)
          to label %4 unwind label %15

4:                                                ; preds = %0
  %5 = load ptr, ptr @g, align 8
  %6 = invoke i32 @maybeAssign(ptr %5, i1 false)
          to label %7 unwind label %15

7:                                                ; preds = %4
  %8 = getelementptr %Node, ptr %3, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = call ptr @GC_malloc_atomic(i64 58)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %11, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %9)
  %13 = icmp eq ptr %11, null
  %14 = select i1 %13, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %11
  call void @puts(ptr %14)
  br label %154

15:                                               ; preds = %4, %0
  %16 = landingpad { ptr, i32 }
          catch ptr null
  %17 = extractvalue { ptr, i32 } %16, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %15
  %20 = icmp eq ptr %19, @_ZTIi
  br i1 %20, label %21, label %28

21:                                               ; preds = %.noexc2
  %22 = load i32, ptr %18, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %151

28:                                               ; preds = %.noexc2
  %29 = icmp eq ptr %19, @_ZTIj
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load i32, ptr %18, align 4
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, i32 }, ptr %33, i32 0, i32 0
  store i64 4, ptr %34, align 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, i32 }, ptr %33, i32 0, i32 2
  store i32 %31, ptr %36, align 4
  br label %148

37:                                               ; preds = %28
  %38 = icmp eq ptr %19, @_ZTId
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load double, ptr %18, align 8
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 0
  store i64 8, ptr %43, align 8
  %44 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 2
  store double %40, ptr %45, align 8
  br label %145

46:                                               ; preds = %37
  %47 = icmp eq ptr %19, @_ZTIf
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load float, ptr %18, align 4
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, float }, ptr %51, i32 0, i32 0
  store i64 4, ptr %52, align 8
  %53 = getelementptr { i64, ptr, float }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, float }, ptr %51, i32 0, i32 2
  store float %49, ptr %54, align 4
  br label %142

55:                                               ; preds = %46
  %56 = icmp eq ptr %19, @_ZTIb
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i1, ptr %18, align 1
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i1 }, ptr %60, i32 0, i32 0
  store i64 1, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i1 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i1 }, ptr %60, i32 0, i32 2
  store i1 %58, ptr %63, align 1
  br label %139

64:                                               ; preds = %55
  %65 = icmp eq ptr %19, @_ZTIx
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load i64, ptr %18, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, i64 }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, i64 }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, i64 }, ptr %69, i32 0, i32 2
  store i64 %67, ptr %72, align 8
  br label %136

73:                                               ; preds = %64
  %74 = icmp eq ptr %19, @_ZTIPc
  br i1 %74, label %75, label %82

75:                                               ; preds = %73
  %76 = load ptr, ptr %18, align 8
  %77 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %75
  %78 = getelementptr i8, ptr %77, i64 8
  %79 = getelementptr { i64, ptr, ptr }, ptr %78, i32 0, i32 0
  store i64 8, ptr %79, align 8
  %80 = getelementptr { i64, ptr, ptr }, ptr %78, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %80, align 8
  %81 = getelementptr { i64, ptr, ptr }, ptr %78, i32 0, i32 2
  store ptr %76, ptr %81, align 8
  br label %133

82:                                               ; preds = %73
  %83 = icmp eq ptr %19, @_ZTIPv
  br i1 %83, label %84, label %86

84:                                               ; preds = %82
  %85 = load ptr, ptr %18, align 8
  br label %130

86:                                               ; preds = %82
  %87 = load ptr, ptr %19, align 8
  %88 = icmp eq ptr %87, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %88, label %89, label %120

89:                                               ; preds = %86
  %90 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %19, i32 0, i32 2
  %91 = load i32, ptr %90, align 4
  %92 = and i32 %91, 1073741824
  %93 = icmp ne i32 %92, 0
  br i1 %93, label %94, label %110

94:                                               ; preds = %89
  %95 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %19, i32 0, i32 4
  %96 = load ptr, ptr %95, align 8
  %97 = icmp ne ptr %96, null
  br i1 %97, label %98, label %100

98:                                               ; preds = %94
  invoke void %96(ptr %1, ptr %18)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %98
  %99 = load ptr, ptr %1, align 8
  br label %107

100:                                              ; preds = %94
  %101 = load ptr, ptr %18, align 8
  %102 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %100
  %103 = getelementptr i8, ptr %102, i64 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 0
  store i64 8, ptr %104, align 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 2
  store ptr %101, ptr %106, align 8
  br label %107

107:                                              ; preds = %.noexc11, %.noexc3
  %108 = phi ptr [ %103, %.noexc3 ], [ %99, %.noexc11 ]
  br label %109

109:                                              ; preds = %107
  br label %117

110:                                              ; preds = %89
  %111 = load ptr, ptr %18, align 8
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 0
  store i64 8, ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 2
  store ptr %111, ptr %116, align 8
  br label %117

117:                                              ; preds = %109, %.noexc
  %118 = phi ptr [ %113, %.noexc ], [ %108, %109 ]
  br label %119

119:                                              ; preds = %117
  br label %127

120:                                              ; preds = %86
  %121 = load ptr, ptr %18, align 8
  %122 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %120
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 0
  store i64 8, ptr %124, align 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %125, align 8
  %126 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 2
  store ptr %121, ptr %126, align 8
  br label %127

127:                                              ; preds = %119, %.noexc8
  %128 = phi ptr [ %123, %.noexc8 ], [ %118, %119 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %84, %129
  %131 = phi ptr [ %128, %129 ], [ %85, %84 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc4, %132
  %134 = phi ptr [ %131, %132 ], [ %78, %.noexc4 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc7, %135
  %137 = phi ptr [ %134, %135 ], [ %69, %.noexc7 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc5, %138
  %140 = phi ptr [ %137, %138 ], [ %60, %.noexc5 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc9, %141
  %143 = phi ptr [ %140, %141 ], [ %51, %.noexc9 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc1, %144
  %146 = phi ptr [ %143, %144 ], [ %42, %.noexc1 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc6, %147
  %149 = phi ptr [ %146, %147 ], [ %33, %.noexc6 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc10, %150
  %152 = phi ptr [ %149, %150 ], [ %24, %.noexc10 ]
  br label %153

153:                                              ; preds = %151
  store ptr %152, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %154

154:                                              ; preds = %7, %153
  %155 = load ptr, ptr @g, align 8
  %156 = call i1 @countIs(ptr %155, double 1.000000e+00)
  br i1 %156, label %157, label %159

157:                                              ; preds = %154
  ret void

158:                                              ; preds = %159
  unreachable

159:                                              ; preds = %154
  %160 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17819851308403393800, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 75, ptr null)
  br label %158

endcatch.resume:                                  ; preds = %98, %21, %48, %120, %66, %30, %57, %75, %100, %15, %39, %110
  %161 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %161
}

; Function Attrs: mustprogress
define void @inCatchWithFinally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %6, align 8
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIPc, ptr null)
          to label %311 unwind label %7

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
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %13
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
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %22
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %31
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
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %40
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %49
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %67
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
  invoke void %88(ptr %2, ptr %10)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %90
  %91 = load ptr, ptr %2, align 8
  br label %99

92:                                               ; preds = %86
  %93 = load ptr, ptr %10, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %99

99:                                               ; preds = %.noexc13, %.noexc19
  %100 = phi ptr [ %95, %.noexc19 ], [ %91, %.noexc13 ]
  br label %101

101:                                              ; preds = %99
  br label %109

102:                                              ; preds = %81
  %103 = load ptr, ptr %10, align 8
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 0
  store i64 8, ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 2
  store ptr %103, ptr %108, align 8
  br label %109

109:                                              ; preds = %101, %.noexc12
  %110 = phi ptr [ %105, %.noexc12 ], [ %100, %101 ]
  br label %111

111:                                              ; preds = %109
  br label %119

112:                                              ; preds = %78
  %113 = load ptr, ptr %10, align 8
  %114 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %112
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 0
  store i64 8, ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 2
  store ptr %113, ptr %118, align 8
  br label %119

119:                                              ; preds = %111, %.noexc15
  %120 = phi ptr [ %115, %.noexc15 ], [ %110, %111 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %76, %121
  %123 = phi ptr [ %120, %121 ], [ %77, %76 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc6, %124
  %126 = phi ptr [ %123, %124 ], [ %70, %.noexc6 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc21, %127
  %129 = phi ptr [ %126, %127 ], [ %61, %.noexc21 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc8, %130
  %132 = phi ptr [ %129, %130 ], [ %52, %.noexc8 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc16, %133
  %135 = phi ptr [ %132, %133 ], [ %43, %.noexc16 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc11, %136
  %138 = phi ptr [ %135, %136 ], [ %34, %.noexc11 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc14, %139
  %141 = phi ptr [ %138, %139 ], [ %25, %.noexc14 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc18, %142
  %144 = phi ptr [ %141, %142 ], [ %16, %.noexc18 ]
  br label %145

145:                                              ; preds = %143
  store ptr %144, ptr %3, align 8
  %146 = load ptr, ptr @g, align 8
  %147 = invoke i32 @maybeAssign(ptr %146, i1 false)
          to label %148 unwind label %endcatch.pad2

148:                                              ; preds = %145
  call void @__cxa_end_catch()
  %149 = load i32, ptr %5, align 4
  %150 = add i32 %149, 1
  store i32 %150, ptr %5, align 4
  br label %302

endcatch.pad2:                                    ; preds = %145
  %151 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

152:                                              ; No predecessors!
  %153 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %152
  %154 = phi { ptr, i32 } [ %153, %152 ], [ %151, %endcatch.pad2 ]
  %155 = extractvalue { ptr, i32 } %154, 0
  %156 = call ptr @__cxa_begin_catch(ptr %155)
  %157 = load i32, ptr %5, align 4
  %158 = add i32 %157, 1
  store i32 %158, ptr %5, align 4
  invoke void @__cxa_rethrow()
          to label %311 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %.caught1
  %159 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

160:                                              ; No predecessors!
  %161 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %160
  %162 = phi { ptr, i32 } [ %161, %160 ], [ %159, %endcatch.pad ]
  %163 = extractvalue { ptr, i32 } %162, 0
  %164 = call ptr @__cxa_begin_catch(ptr %163)
  %165 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %.caught
  %166 = icmp eq ptr %165, @_ZTIi
  br i1 %166, label %167, label %174

167:                                              ; preds = %.noexc17
  %168 = load i32, ptr %164, align 4
  %169 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %167
  %170 = getelementptr i8, ptr %169, i64 8
  %171 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 0
  store i64 4, ptr %171, align 8
  %172 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %172, align 8
  %173 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 2
  store i32 %168, ptr %173, align 4
  br label %297

174:                                              ; preds = %.noexc17
  %175 = icmp eq ptr %165, @_ZTIj
  br i1 %175, label %176, label %183

176:                                              ; preds = %174
  %177 = load i32, ptr %164, align 4
  %178 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %176
  %179 = getelementptr i8, ptr %178, i64 8
  %180 = getelementptr { i64, ptr, i32 }, ptr %179, i32 0, i32 0
  store i64 4, ptr %180, align 8
  %181 = getelementptr { i64, ptr, i32 }, ptr %179, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %181, align 8
  %182 = getelementptr { i64, ptr, i32 }, ptr %179, i32 0, i32 2
  store i32 %177, ptr %182, align 4
  br label %294

183:                                              ; preds = %174
  %184 = icmp eq ptr %165, @_ZTId
  br i1 %184, label %185, label %192

185:                                              ; preds = %183
  %186 = load double, ptr %164, align 8
  %187 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %185
  %188 = getelementptr i8, ptr %187, i64 8
  %189 = getelementptr { i64, ptr, double }, ptr %188, i32 0, i32 0
  store i64 8, ptr %189, align 8
  %190 = getelementptr { i64, ptr, double }, ptr %188, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %190, align 8
  %191 = getelementptr { i64, ptr, double }, ptr %188, i32 0, i32 2
  store double %186, ptr %191, align 8
  br label %291

192:                                              ; preds = %183
  %193 = icmp eq ptr %165, @_ZTIf
  br i1 %193, label %194, label %201

194:                                              ; preds = %192
  %195 = load float, ptr %164, align 4
  %196 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %194
  %197 = getelementptr i8, ptr %196, i64 8
  %198 = getelementptr { i64, ptr, float }, ptr %197, i32 0, i32 0
  store i64 4, ptr %198, align 8
  %199 = getelementptr { i64, ptr, float }, ptr %197, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %199, align 8
  %200 = getelementptr { i64, ptr, float }, ptr %197, i32 0, i32 2
  store float %195, ptr %200, align 4
  br label %288

201:                                              ; preds = %192
  %202 = icmp eq ptr %165, @_ZTIb
  br i1 %202, label %203, label %210

203:                                              ; preds = %201
  %204 = load i1, ptr %164, align 1
  %205 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %203
  %206 = getelementptr i8, ptr %205, i64 8
  %207 = getelementptr { i64, ptr, i1 }, ptr %206, i32 0, i32 0
  store i64 1, ptr %207, align 8
  %208 = getelementptr { i64, ptr, i1 }, ptr %206, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %208, align 8
  %209 = getelementptr { i64, ptr, i1 }, ptr %206, i32 0, i32 2
  store i1 %204, ptr %209, align 1
  br label %285

210:                                              ; preds = %201
  %211 = icmp eq ptr %165, @_ZTIx
  br i1 %211, label %212, label %219

212:                                              ; preds = %210
  %213 = load i64, ptr %164, align 8
  %214 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %212
  %215 = getelementptr i8, ptr %214, i64 8
  %216 = getelementptr { i64, ptr, i64 }, ptr %215, i32 0, i32 0
  store i64 8, ptr %216, align 8
  %217 = getelementptr { i64, ptr, i64 }, ptr %215, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %217, align 8
  %218 = getelementptr { i64, ptr, i64 }, ptr %215, i32 0, i32 2
  store i64 %213, ptr %218, align 8
  br label %282

219:                                              ; preds = %210
  %220 = icmp eq ptr %165, @_ZTIPc
  br i1 %220, label %221, label %228

221:                                              ; preds = %219
  %222 = load ptr, ptr %164, align 8
  %223 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %221
  %224 = getelementptr i8, ptr %223, i64 8
  %225 = getelementptr { i64, ptr, ptr }, ptr %224, i32 0, i32 0
  store i64 8, ptr %225, align 8
  %226 = getelementptr { i64, ptr, ptr }, ptr %224, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %226, align 8
  %227 = getelementptr { i64, ptr, ptr }, ptr %224, i32 0, i32 2
  store ptr %222, ptr %227, align 8
  br label %279

228:                                              ; preds = %219
  %229 = icmp eq ptr %165, @_ZTIPv
  br i1 %229, label %230, label %232

230:                                              ; preds = %228
  %231 = load ptr, ptr %164, align 8
  br label %276

232:                                              ; preds = %228
  %233 = load ptr, ptr %165, align 8
  %234 = icmp eq ptr %233, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %234, label %235, label %266

235:                                              ; preds = %232
  %236 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %165, i32 0, i32 2
  %237 = load i32, ptr %236, align 4
  %238 = and i32 %237, 1073741824
  %239 = icmp ne i32 %238, 0
  br i1 %239, label %240, label %256

240:                                              ; preds = %235
  %241 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %165, i32 0, i32 4
  %242 = load ptr, ptr %241, align 8
  %243 = icmp ne ptr %242, null
  br i1 %243, label %244, label %246

244:                                              ; preds = %240
  invoke void %242(ptr %1, ptr %164)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %244
  %245 = load ptr, ptr %1, align 8
  br label %253

246:                                              ; preds = %240
  %247 = load ptr, ptr %164, align 8
  %248 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %246
  %249 = getelementptr i8, ptr %248, i64 8
  %250 = getelementptr { i64, ptr, ptr }, ptr %249, i32 0, i32 0
  store i64 8, ptr %250, align 8
  %251 = getelementptr { i64, ptr, ptr }, ptr %249, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %251, align 8
  %252 = getelementptr { i64, ptr, ptr }, ptr %249, i32 0, i32 2
  store ptr %247, ptr %252, align 8
  br label %253

253:                                              ; preds = %.noexc, %.noexc4
  %254 = phi ptr [ %249, %.noexc4 ], [ %245, %.noexc ]
  br label %255

255:                                              ; preds = %253
  br label %263

256:                                              ; preds = %235
  %257 = load ptr, ptr %164, align 8
  %258 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %256
  %259 = getelementptr i8, ptr %258, i64 8
  %260 = getelementptr { i64, ptr, ptr }, ptr %259, i32 0, i32 0
  store i64 8, ptr %260, align 8
  %261 = getelementptr { i64, ptr, ptr }, ptr %259, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %261, align 8
  %262 = getelementptr { i64, ptr, ptr }, ptr %259, i32 0, i32 2
  store ptr %257, ptr %262, align 8
  br label %263

263:                                              ; preds = %255, %.noexc7
  %264 = phi ptr [ %259, %.noexc7 ], [ %254, %255 ]
  br label %265

265:                                              ; preds = %263
  br label %273

266:                                              ; preds = %232
  %267 = load ptr, ptr %164, align 8
  %268 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %266
  %269 = getelementptr i8, ptr %268, i64 8
  %270 = getelementptr { i64, ptr, ptr }, ptr %269, i32 0, i32 0
  store i64 8, ptr %270, align 8
  %271 = getelementptr { i64, ptr, ptr }, ptr %269, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %271, align 8
  %272 = getelementptr { i64, ptr, ptr }, ptr %269, i32 0, i32 2
  store ptr %267, ptr %272, align 8
  br label %273

273:                                              ; preds = %265, %.noexc22
  %274 = phi ptr [ %269, %.noexc22 ], [ %264, %265 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %230, %275
  %277 = phi ptr [ %274, %275 ], [ %231, %230 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc20, %278
  %280 = phi ptr [ %277, %278 ], [ %224, %.noexc20 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc10, %281
  %283 = phi ptr [ %280, %281 ], [ %215, %.noexc10 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc25, %284
  %286 = phi ptr [ %283, %284 ], [ %206, %.noexc25 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc23, %287
  %289 = phi ptr [ %286, %287 ], [ %197, %.noexc23 ]
  br label %290

290:                                              ; preds = %288
  br label %291

291:                                              ; preds = %.noexc9, %290
  %292 = phi ptr [ %289, %290 ], [ %188, %.noexc9 ]
  br label %293

293:                                              ; preds = %291
  br label %294

294:                                              ; preds = %.noexc3, %293
  %295 = phi ptr [ %292, %293 ], [ %179, %.noexc3 ]
  br label %296

296:                                              ; preds = %294
  br label %297

297:                                              ; preds = %.noexc24, %296
  %298 = phi ptr [ %295, %296 ], [ %170, %.noexc24 ]
  br label %299

299:                                              ; preds = %297
  store ptr %298, ptr %4, align 8
  %300 = load i32, ptr %5, align 4
  %301 = add i32 %300, 1
  store i32 %301, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %302

302:                                              ; preds = %148, %299
  %303 = load i32, ptr %5, align 4
  %304 = icmp eq i32 %303, 2
  br i1 %304, label %305, label %308

305:                                              ; preds = %302
  %306 = load ptr, ptr @g, align 8
  %307 = call i1 @countIs(ptr %306, double 1.000000e+00)
  br label %308

308:                                              ; preds = %305, %302
  %309 = phi i1 [ %307, %305 ], [ %304, %302 ]
  br i1 %309, label %310, label %312

310:                                              ; preds = %308
  ret void

311:                                              ; preds = %312, %.caught1, %0
  unreachable

312:                                              ; preds = %308
  %313 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11395844481985125100, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 92, ptr null)
  br label %311

endcatch.resume:                                  ; preds = %203, %167, %194, %266, %58, %221, %92, %13, %.caught, %40, %112, %22, %90, %102, %31, %212, %185, %49, %256, %67, %7, %246, %176, %244
  %314 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %314
}

; Function Attrs: mustprogress
define void @inLoop() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %150, %0
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %153

7:                                                ; preds = %4
  %8 = load ptr, ptr @g, align 8
  %9 = invoke i32 @maybeAssign(ptr %8, i1 false)
          to label %10 unwind label %11

10:                                               ; preds = %7
  br label %150

11:                                               ; preds = %7
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc
  %18 = load i32, ptr %14, align 4
  %19 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %17
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 0
  store i64 4, ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 2
  store i32 %18, ptr %23, align 4
  br label %147

24:                                               ; preds = %.noexc
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %26
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 0
  store i64 4, ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 2
  store i32 %27, ptr %32, align 4
  br label %144

33:                                               ; preds = %24
  %34 = icmp eq ptr %15, @_ZTId
  br i1 %34, label %35, label %42

35:                                               ; preds = %33
  %36 = load double, ptr %14, align 8
  %37 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %35
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 0
  store i64 8, ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 2
  store double %36, ptr %41, align 8
  br label %141

42:                                               ; preds = %33
  %43 = icmp eq ptr %15, @_ZTIf
  br i1 %43, label %44, label %51

44:                                               ; preds = %42
  %45 = load float, ptr %14, align 4
  %46 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %44
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 0
  store i64 4, ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 2
  store float %45, ptr %50, align 4
  br label %138

51:                                               ; preds = %42
  %52 = icmp eq ptr %15, @_ZTIb
  br i1 %52, label %53, label %60

53:                                               ; preds = %51
  %54 = load i1, ptr %14, align 1
  %55 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %53
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 0
  store i64 1, ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 2
  store i1 %54, ptr %59, align 1
  br label %135

60:                                               ; preds = %51
  %61 = icmp eq ptr %15, @_ZTIx
  br i1 %61, label %62, label %69

62:                                               ; preds = %60
  %63 = load i64, ptr %14, align 8
  %64 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %62
  %65 = getelementptr i8, ptr %64, i64 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 0
  store i64 8, ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 2
  store i64 %63, ptr %68, align 8
  br label %132

69:                                               ; preds = %60
  %70 = icmp eq ptr %15, @_ZTIPc
  br i1 %70, label %71, label %78

71:                                               ; preds = %69
  %72 = load ptr, ptr %14, align 8
  %73 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %71
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 0
  store i64 8, ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 2
  store ptr %72, ptr %77, align 8
  br label %129

78:                                               ; preds = %69
  %79 = icmp eq ptr %15, @_ZTIPv
  br i1 %79, label %80, label %82

80:                                               ; preds = %78
  %81 = load ptr, ptr %14, align 8
  br label %126

82:                                               ; preds = %78
  %83 = load ptr, ptr %15, align 8
  %84 = icmp eq ptr %83, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %84, label %85, label %116

85:                                               ; preds = %82
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 2
  %87 = load i32, ptr %86, align 4
  %88 = and i32 %87, 1073741824
  %89 = icmp ne i32 %88, 0
  br i1 %89, label %90, label %106

90:                                               ; preds = %85
  %91 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 4
  %92 = load ptr, ptr %91, align 8
  %93 = icmp ne ptr %92, null
  br i1 %93, label %94, label %96

94:                                               ; preds = %90
  invoke void %92(ptr %1, ptr %14)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %94
  %95 = load ptr, ptr %1, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc4, %.noexc7
  %104 = phi ptr [ %99, %.noexc7 ], [ %95, %.noexc4 ]
  br label %105

105:                                              ; preds = %103
  br label %113

106:                                              ; preds = %85
  %107 = load ptr, ptr %14, align 8
  %108 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %106
  %109 = getelementptr i8, ptr %108, i64 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 0
  store i64 8, ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 2
  store ptr %107, ptr %112, align 8
  br label %113

113:                                              ; preds = %105, %.noexc1
  %114 = phi ptr [ %109, %.noexc1 ], [ %104, %105 ]
  br label %115

115:                                              ; preds = %113
  br label %123

116:                                              ; preds = %82
  %117 = load ptr, ptr %14, align 8
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc10
  %124 = phi ptr [ %119, %.noexc10 ], [ %114, %115 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %80, %125
  %127 = phi ptr [ %124, %125 ], [ %81, %80 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc3, %128
  %130 = phi ptr [ %127, %128 ], [ %74, %.noexc3 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc6, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc6 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc5, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc5 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc9, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc9 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc8, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc8 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc2, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc2 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc11, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc11 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %150

150:                                              ; preds = %10, %149
  %151 = load i32, ptr %3, align 4
  %152 = add i32 %151, 1
  store i32 %152, ptr %3, align 4
  br label %4

153:                                              ; preds = %4
  %154 = load ptr, ptr @g, align 8
  %155 = call i1 @countIs(ptr %154, double 1.000000e+00)
  br i1 %155, label %156, label %158

156:                                              ; preds = %153
  ret void

157:                                              ; preds = %158
  unreachable

158:                                              ; preds = %153
  %159 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_937078698315841043, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 103, ptr null)
  br label %157

endcatch.resume:                                  ; preds = %17, %116, %44, %35, %96, %62, %53, %94, %71, %26, %106, %11
  %160 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %160
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  call void @inTryCatch()
  call void @throughFinally()
  call void @inBlockWithLocal()
  call void @inCatchWithFinally()
  call void @inLoop()
  store i1 false, ptr %3, align 1
  %4 = invoke ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_2393915098112578145, i64 8), double -1.000000e+00)
          to label %5 unwind label %12

5:                                                ; preds = %0
  %6 = call ptr @GC_malloc(i64 16)
  %7 = getelementptr i8, ptr %6, i64 8
  store ptr %4, ptr %7, align 8
  store ptr %7, ptr @g, align 8
  %8 = load ptr, ptr @g, align 8
  %9 = invoke i32 @maybeAssign(ptr %8, i1 true)
          to label %10 unwind label %12

10:                                               ; preds = %5
  %11 = icmp eq i32 %9, -1
  store i1 %11, ptr %3, align 1
  br label %151

12:                                               ; preds = %5, %0
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %12
  %17 = icmp eq ptr %16, @_ZTIi
  br i1 %17, label %18, label %25

18:                                               ; preds = %.noexc5
  %19 = load i32, ptr %15, align 4
  %20 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %18
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 0
  store i64 4, ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 2
  store i32 %19, ptr %24, align 4
  br label %148

25:                                               ; preds = %.noexc5
  %26 = icmp eq ptr %16, @_ZTIj
  br i1 %26, label %27, label %34

27:                                               ; preds = %25
  %28 = load i32, ptr %15, align 4
  %29 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %27
  %30 = getelementptr i8, ptr %29, i64 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 0
  store i64 4, ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %32, align 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 2
  store i32 %28, ptr %33, align 4
  br label %145

34:                                               ; preds = %25
  %35 = icmp eq ptr %16, @_ZTId
  br i1 %35, label %36, label %43

36:                                               ; preds = %34
  %37 = load double, ptr %15, align 8
  %38 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %36
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 0
  store i64 8, ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %41, align 8
  %42 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 2
  store double %37, ptr %42, align 8
  br label %142

43:                                               ; preds = %34
  %44 = icmp eq ptr %16, @_ZTIf
  br i1 %44, label %45, label %52

45:                                               ; preds = %43
  %46 = load float, ptr %15, align 4
  %47 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %45
  %48 = getelementptr i8, ptr %47, i64 8
  %49 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 0
  store i64 4, ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %50, align 8
  %51 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 2
  store float %46, ptr %51, align 4
  br label %139

52:                                               ; preds = %43
  %53 = icmp eq ptr %16, @_ZTIb
  br i1 %53, label %54, label %61

54:                                               ; preds = %52
  %55 = load i1, ptr %15, align 1
  %56 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %54
  %57 = getelementptr i8, ptr %56, i64 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 0
  store i64 1, ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %59, align 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 2
  store i1 %55, ptr %60, align 1
  br label %136

61:                                               ; preds = %52
  %62 = icmp eq ptr %16, @_ZTIx
  br i1 %62, label %63, label %70

63:                                               ; preds = %61
  %64 = load i64, ptr %15, align 8
  %65 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %63
  %66 = getelementptr i8, ptr %65, i64 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 0
  store i64 8, ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %68, align 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 2
  store i64 %64, ptr %69, align 8
  br label %133

70:                                               ; preds = %61
  %71 = icmp eq ptr %16, @_ZTIPc
  br i1 %71, label %72, label %79

72:                                               ; preds = %70
  %73 = load ptr, ptr %15, align 8
  %74 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %72
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 0
  store i64 8, ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %77, align 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 2
  store ptr %73, ptr %78, align 8
  br label %130

79:                                               ; preds = %70
  %80 = icmp eq ptr %16, @_ZTIPv
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = load ptr, ptr %15, align 8
  br label %127

83:                                               ; preds = %79
  %84 = load ptr, ptr %16, align 8
  %85 = icmp eq ptr %84, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %85, label %86, label %117

86:                                               ; preds = %83
  %87 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %16, i32 0, i32 2
  %88 = load i32, ptr %87, align 4
  %89 = and i32 %88, 1073741824
  %90 = icmp ne i32 %89, 0
  br i1 %90, label %91, label %107

91:                                               ; preds = %86
  %92 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %16, i32 0, i32 4
  %93 = load ptr, ptr %92, align 8
  %94 = icmp ne ptr %93, null
  br i1 %94, label %95, label %97

95:                                               ; preds = %91
  invoke void %93(ptr %1, ptr %15)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %95
  %96 = load ptr, ptr %1, align 8
  br label %104

97:                                               ; preds = %91
  %98 = load ptr, ptr %15, align 8
  %99 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %97
  %100 = getelementptr i8, ptr %99, i64 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 0
  store i64 8, ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 2
  store ptr %98, ptr %103, align 8
  br label %104

104:                                              ; preds = %.noexc9, %.noexc11
  %105 = phi ptr [ %100, %.noexc11 ], [ %96, %.noexc9 ]
  br label %106

106:                                              ; preds = %104
  br label %114

107:                                              ; preds = %86
  %108 = load ptr, ptr %15, align 8
  %109 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %107
  %110 = getelementptr i8, ptr %109, i64 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 0
  store i64 8, ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 2
  store ptr %108, ptr %113, align 8
  br label %114

114:                                              ; preds = %106, %.noexc
  %115 = phi ptr [ %110, %.noexc ], [ %105, %106 ]
  br label %116

116:                                              ; preds = %114
  br label %124

117:                                              ; preds = %83
  %118 = load ptr, ptr %15, align 8
  %119 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %117
  %120 = getelementptr i8, ptr %119, i64 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 0
  store i64 8, ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 2
  store ptr %118, ptr %123, align 8
  br label %124

124:                                              ; preds = %116, %.noexc1
  %125 = phi ptr [ %120, %.noexc1 ], [ %115, %116 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %81, %126
  %128 = phi ptr [ %125, %126 ], [ %82, %81 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc3, %129
  %131 = phi ptr [ %128, %129 ], [ %75, %.noexc3 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc2, %132
  %134 = phi ptr [ %131, %132 ], [ %66, %.noexc2 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc7, %135
  %137 = phi ptr [ %134, %135 ], [ %57, %.noexc7 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc10, %138
  %140 = phi ptr [ %137, %138 ], [ %48, %.noexc10 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc4, %141
  %143 = phi ptr [ %140, %141 ], [ %39, %.noexc4 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc8, %144
  %146 = phi ptr [ %143, %144 ], [ %30, %.noexc8 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc6, %147
  %149 = phi ptr [ %146, %147 ], [ %21, %.noexc6 ]
  br label %150

150:                                              ; preds = %148
  store ptr %149, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %151

151:                                              ; preds = %10, %150
  %152 = load i1, ptr %3, align 1
  br i1 %152, label %153, label %156

153:                                              ; preds = %151
  %154 = load ptr, ptr @g, align 8
  %155 = call i1 @countIs(ptr %154, double 1.000000e+00)
  br label %156

156:                                              ; preds = %153, %151
  %157 = phi i1 [ %155, %153 ], [ %152, %151 ]
  br i1 %157, label %158, label %176

158:                                              ; preds = %156
  %159 = load ptr, ptr @g, align 8
  %160 = load ptr, ptr %159, align 8
  %161 = getelementptr %Node, ptr %160, i32 0, i32 2
  %162 = load ptr, ptr %161, align 8
  %163 = ptrtoint ptr %162 to i64
  %164 = icmp ne i64 %163, 0
  %165 = zext i1 %164 to i32
  %166 = and i32 %165, 1
  %167 = icmp ne i32 %166, 0
  br i1 %167, label %168, label %171

168:                                              ; preds = %158
  %169 = call i32 @strcmp(ptr %162, ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8))
  %170 = icmp eq i32 %169, 0
  br label %173

171:                                              ; preds = %158
  %172 = icmp eq i64 %163, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8) to i64)
  br label %173

173:                                              ; preds = %168, %171
  %174 = phi i1 [ %172, %171 ], [ %170, %168 ]
  br label %175

175:                                              ; preds = %173
  br label %176

176:                                              ; preds = %175, %156
  %177 = phi i1 [ %174, %175 ], [ %157, %156 ]
  br i1 %177, label %178, label %180

178:                                              ; preds = %176
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

179:                                              ; preds = %180
  unreachable

180:                                              ; preds = %176
  %181 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17268163596755564902, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10815677319114142208, i64 8), i32 121, ptr null)
  br label %179

endcatch.resume:                                  ; preds = %97, %45, %95, %27, %54, %18, %12, %36, %72, %63, %117, %107
  %182 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %182
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

