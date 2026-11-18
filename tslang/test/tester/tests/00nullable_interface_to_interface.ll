; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Square = type { %Rect, ptr }
%Rect = type { ptr, double, ptr }
%Box = type { ptr, double, double }

$Box..size = comdat any

$Rect..size = comdat any

$Square..size = comdat any

@ts_vtable_map_17875616299094573122 = internal constant [3 x i32] [i32 0, i32 1, i32 3]
@ts_remap_vtable_list = internal global ptr null
@s_1609559789082440018 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBox\00", align 8
@s_8364433741114752039 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRect\00", align 8
@s_7368384151532411686 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFSquare\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_4745878797808296967 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field read through the other interface\00", align 8
@f_2658979206283382412 = internal constant [123 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00nullable_interface_to_interface.ts\00", align 8
@m_1039812570711373850 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field that moved\00", align 8
@m_14662280579041655297 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa method that moved\00", align 8
@m_564923817551714954 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa returned value seen through the other interface\00", align 8
@m_12204294225069353136 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull seen through the other interface\00", align 8
@m_14778402311747416871 = internal constant [69 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull seen through an interface with members in another order\00", align 8
@m_14728058011860429015 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFits method throws\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Box..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1609559789082440018, i64 8)
@Box..size = global i64 24, comdat
@Box..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Box..instanceOf, ptr @Box..new, ptr @Box..rtti, ptr @Box..size }
@Rect..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_8364433741114752039, i64 8)
@Rect..size = global i64 24, comdat
@Rect.Shape..vtbl = private global { ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_5856831998602112496_class, i64 40), ptr @Rect..instanceOf, ptr @Rect.area, ptr inttoptr (i64 8 to ptr) }
@td_5856831998602112496_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_5856831998602112496, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Rect..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Rect..instanceOf, ptr @Rect.Shape..vtbl, ptr @Rect..new, ptr @Rect.area, ptr @Rect..rtti, ptr @Rect..size }
@Square..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7368384151532411686, i64 8)
@Square..size = global i64 32, comdat
@Square.Shape..vtbl = private global { ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_1571629035032042005_class, i64 40), ptr @Square..instanceOf, ptr @Rect.area, ptr inttoptr (i64 8 to ptr) }
@td_1571629035032042005_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_1571629035032042005, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Square..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Square..instanceOf, ptr @Square.Shape..vtbl, ptr @Square..new, ptr @Rect.area, ptr @Rect..rtti, ptr @Rect..size, ptr @Square..rtti, ptr @Square..size }
@_ZTIPc = external constant ptr
@Shape.null..vtbl = private global { ptr, ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @Shape.null.area, ptr null }
@HasId.null..vtbl = private global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr null }
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

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

define internal ptr @ts_remap_vtable(ptr %0, ptr %1, i64 %2) {
  %4 = icmp eq ptr %0, null
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  ret ptr null

6:                                                ; preds = %3
  %7 = load atomic ptr, ptr @ts_remap_vtable_list acquire, align 8
  br label %8

8:                                                ; preds = %34, %6
  %9 = phi ptr [ %35, %34 ], [ %7, %6 ]
  %10 = icmp eq ptr %9, null
  br i1 %10, label %36, label %11

11:                                               ; preds = %8
  %12 = getelementptr i64, ptr %9, i64 1
  %13 = load i64, ptr %12, align 8
  %14 = icmp eq i64 %13, %2
  br i1 %14, label %15, label %34

15:                                               ; preds = %18, %11
  %16 = phi i64 [ %31, %18 ], [ 0, %11 ]
  %17 = icmp eq i64 %16, %2
  br i1 %17, label %32, label %18

18:                                               ; preds = %15
  %19 = getelementptr i64, ptr %9, i64 2
  %20 = getelementptr i64, ptr %19, i64 %16
  %21 = load i64, ptr %20, align 8
  %22 = getelementptr i32, ptr %1, i64 %16
  %23 = load i32, ptr %22, align 4
  %24 = sext i32 %23 to i64
  %25 = icmp slt i64 %24, 0
  %26 = select i1 %25, i64 0, i64 %24
  %27 = getelementptr i64, ptr %0, i64 %26
  %28 = load i64, ptr %27, align 8
  %29 = select i1 %25, i64 -1, i64 %28
  %30 = icmp eq i64 %21, %29
  %31 = add i64 %16, 1
  br i1 %30, label %15, label %34

32:                                               ; preds = %15
  %33 = getelementptr i64, ptr %9, i64 2
  ret ptr %33

34:                                               ; preds = %18, %11
  %35 = load ptr, ptr %9, align 8
  br label %8

36:                                               ; preds = %8
  %37 = add i64 %2, 2
  %38 = mul i64 %37, 8
  %39 = add i64 %38, 8
  %40 = call ptr @GC_malloc(i64 %39)
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr i64, ptr %41, i64 1
  store i64 %2, ptr %42, align 8
  br label %43

43:                                               ; preds = %46, %36
  %44 = phi i64 [ %57, %46 ], [ 0, %36 ]
  %45 = icmp eq i64 %44, %2
  br i1 %45, label %58, label %46

46:                                               ; preds = %43
  %47 = getelementptr i64, ptr %41, i64 2
  %48 = getelementptr i64, ptr %47, i64 %44
  %49 = getelementptr i32, ptr %1, i64 %44
  %50 = load i32, ptr %49, align 4
  %51 = sext i32 %50 to i64
  %52 = icmp slt i64 %51, 0
  %53 = select i1 %52, i64 0, i64 %51
  %54 = getelementptr i64, ptr %0, i64 %53
  %55 = load i64, ptr %54, align 8
  %56 = select i1 %52, i64 -1, i64 %55
  store i64 %56, ptr %48, align 8
  %57 = add i64 %44, 1
  br label %43

58:                                               ; preds = %43
  %59 = load atomic ptr, ptr @ts_remap_vtable_list monotonic, align 8
  br label %60

60:                                               ; preds = %60, %58
  %61 = phi ptr [ %63, %60 ], [ %59, %58 ]
  store ptr %61, ptr %41, align 8
  %62 = cmpxchg ptr @ts_remap_vtable_list, ptr %61, ptr %41 acq_rel monotonic, align 8
  %63 = extractvalue { ptr, i1 } %62, 0
  %64 = extractvalue { ptr, i1 } %62, 1
  br i1 %64, label %65, label %60

65:                                               ; preds = %60
  %66 = getelementptr i64, ptr %41, i64 2
  ret ptr %66
}

declare void @__cxa_end_catch()

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

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_1571629035032042005(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %9

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = getelementptr %Square, ptr %2, i32 0, i32 0
  call void @tsrel_5516768905696647638(ptr %7)
  %8 = getelementptr %Square, ptr %2, i32 0, i32 1
  call void @tsrel_17239760223775721853(ptr %8)
  call void @__tslang_free_block(ptr %2)
  br label %9

9:                                                ; preds = %6, %4, %1
  ret void
}

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

define internal void @tsrel_17239760223775721853(ptr %0) {
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

define internal void @tsrel_5516768905696647638(ptr %0) {
  %2 = getelementptr %Rect, ptr %0, i32 0, i32 2
  call void @tsrel_17239760223775721853(ptr %2)
  ret void
}

define internal void @tsrel_5856831998602112496(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %15

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %15

6:                                                ; preds = %4
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, @Square..vtbl
  br i1 %8, label %9, label %12

9:                                                ; preds = %6
  %10 = getelementptr %Square, ptr %2, i32 0, i32 0
  call void @tsrel_5516768905696647638(ptr %10)
  %11 = getelementptr %Square, ptr %2, i32 0, i32 1
  call void @tsrel_17239760223775721853(ptr %11)
  br label %14

12:                                               ; preds = %6
  %13 = getelementptr %Rect, ptr %2, i32 0, i32 2
  call void @tsrel_17239760223775721853(ptr %13)
  br label %14

14:                                               ; preds = %9, %12
  call void @__tslang_free_block(ptr %2)
  br label %15

15:                                               ; preds = %14, %4, %1
  ret void
}

; Function Attrs: mustprogress
define i1 @Box..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Box..rtti, align 8
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
define ptr @Box..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Box, ptr %3, i32 0, i32 0
  store ptr @Box..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Box.constructor(ptr %0, double %1, double %2) #1 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 1
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %Box, ptr %10, i32 0, i32 2
  %12 = load double, ptr %4, align 8
  store double %12, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Rect..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Rect..rtti, align 8
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
define ptr @Rect..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Rect, ptr %3, i32 0, i32 0
  store ptr @Rect..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Rect.constructor(ptr %0, double %1, ptr %2) #1 {
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %Rect, ptr %7, i32 0, i32 1
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %Rect, ptr %10, i32 0, i32 2
  %12 = load ptr, ptr %4, align 8
  store ptr %12, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Rect.area(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Rect, ptr %4, i32 0, i32 2
  %6 = load ptr, ptr %5, align 8
  %7 = getelementptr %Box, ptr %6, i32 0, i32 1
  %8 = load double, ptr %7, align 8
  %9 = load ptr, ptr %2, align 8
  %10 = getelementptr %Rect, ptr %9, i32 0, i32 2
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr %Box, ptr %11, i32 0, i32 2
  %13 = load double, ptr %12, align 8
  %14 = fmul double %8, %13
  store double %14, ptr %3, align 8
  %15 = load double, ptr %3, align 8
  ret double %15
}

; Function Attrs: mustprogress
define i1 @Square..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @Square..rtti, align 8
  %8 = ptrtoint ptr %6 to i64
  %9 = ptrtoint ptr %7 to i64
  %10 = icmp ne i64 %8, 0
  %11 = icmp ne i64 %9, 0
  %12 = zext i1 %10 to i32
  %13 = zext i1 %11 to i32
  %14 = and i32 %12, %13
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %2
  %17 = call i32 @strcmp(ptr %6, ptr %7)
  %18 = icmp eq i32 %17, 0
  br label %21

19:                                               ; preds = %2
  %20 = icmp eq i64 %8, %9
  br label %21

21:                                               ; preds = %16, %19
  %22 = phi i1 [ %20, %19 ], [ %18, %16 ]
  br label %23

23:                                               ; preds = %21
  br i1 %22, label %29, label %24

24:                                               ; preds = %23
  %25 = load ptr, ptr %4, align 8
  %26 = getelementptr %Square, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %3, align 8
  %28 = call i1 @Rect..instanceOf(ptr %26, ptr %27)
  br label %29

29:                                               ; preds = %24, %23
  %30 = phi i1 [ %28, %24 ], [ %22, %23 ]
  store i1 %30, ptr %5, align 1
  %31 = load i1, ptr %5, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define ptr @Square..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Square, ptr %3, i32 0, i32 0
  %5 = getelementptr %Rect, ptr %4, i32 0, i32 0
  store ptr @Square..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define void @Square.constructor(ptr %0, double %1, double %2, ptr %3) #1 {
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %8, align 8
  store double %1, ptr %7, align 8
  store double %2, ptr %6, align 8
  store ptr %3, ptr %5, align 8
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr %Square, ptr %9, i32 0, i32 0
  %11 = load double, ptr %7, align 8
  %12 = call ptr @GC_malloc(i64 32)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %Box, ptr %13, i32 0, i32 0
  store ptr @Box..vtbl, ptr %14, align 8
  %15 = load double, ptr %6, align 8
  %16 = load double, ptr %6, align 8
  call void @Box.constructor(ptr %13, double %15, double %16)
  call void @Rect.constructor(ptr %10, double %11, ptr %13)
  %17 = load ptr, ptr %8, align 8
  %18 = getelementptr %Square, ptr %17, i32 0, i32 1
  %19 = load ptr, ptr %5, align 8
  store ptr %19, ptr %18, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Shape.null.area(ptr %0) #2 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress noinline
define { ptr, ptr } @mkSquare(double %0, double %1) #3 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca { ptr, ptr }, align 8
  store double %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = call ptr @GC_malloc(i64 40)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Square, ptr %7, i32 0, i32 0
  %9 = getelementptr %Rect, ptr %8, i32 0, i32 0
  store ptr @Square..vtbl, ptr %9, align 8
  %10 = load double, ptr %4, align 8
  %11 = load double, ptr %3, align 8
  %12 = call ptr @GC_malloc(i64 32)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %Box, ptr %13, i32 0, i32 0
  store ptr @Box..vtbl, ptr %14, align 8
  %15 = load double, ptr %4, align 8
  %16 = load double, ptr %4, align 8
  call void @Box.constructor(ptr %13, double %15, double %16)
  call void @Square.constructor(ptr %7, double %10, double %11, ptr %13)
  %17 = ptrtoint ptr %7 to i64
  %18 = icmp ne i64 %17, 0
  br i1 %18, label %19, label %20

19:                                               ; preds = %2
  br label %21

20:                                               ; preds = %2
  br label %21

21:                                               ; preds = %19, %20
  %22 = phi ptr [ @Shape.null..vtbl, %20 ], [ @Square.Shape..vtbl, %19 ]
  %23 = insertvalue { ptr, ptr } undef, ptr %22, 0
  %24 = insertvalue { ptr, ptr } %23, ptr %7, 1
  store { ptr, ptr } %24, ptr %5, align 8
  %25 = load { ptr, ptr }, ptr %5, align 8
  ret { ptr, ptr } %25
}

; Function Attrs: mustprogress noinline
define { ptr, ptr } @mkShape(i1 %0) #3 {
  %2 = alloca i1, align 1
  %3 = alloca { ptr, ptr }, align 8
  store i1 %0, ptr %2, align 1
  %4 = load i1, ptr %2, align 1
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call { ptr, ptr } @mkSquare(double 5.000000e+00, double 2.000000e+00)
  br label %8

7:                                                ; preds = %1
  br label %8

8:                                                ; preds = %5, %7
  %9 = phi { ptr, ptr } [ { ptr @Shape.null..vtbl, ptr null }, %7 ], [ %6, %5 ]
  store { ptr, ptr } %9, ptr %3, align 8
  %10 = load { ptr, ptr }, ptr %3, align 8
  ret { ptr, ptr } %10
}

; Function Attrs: mustprogress noinline
define { ptr, ptr } @mkNull() #3 {
  %1 = alloca { ptr, ptr }, align 8
  store { ptr, ptr } { ptr @HasId.null..vtbl, ptr null }, ptr %1, align 8
  %2 = load { ptr, ptr }, ptr %1, align 8
  ret { ptr, ptr } %2
}

; Function Attrs: mustprogress
define double @call({ ptr, ptr } %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca { ptr, ptr }, align 8
  %5 = alloca double, align 8
  store { ptr, ptr } %0, ptr %4, align 8
  %6 = load { ptr, ptr }, ptr %4, align 8
  %7 = extractvalue { ptr, ptr } %6, 0
  %8 = extractvalue { ptr, ptr } %6, 1
  %9 = getelementptr ptr, ptr %7, i32 2
  %10 = load ptr, ptr %9, align 8
  %11 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %10, 0
  %12 = insertvalue { ptr, ptr, ptr } %11, ptr %8, 1
  %13 = insertvalue { ptr, ptr, ptr } %12, ptr null, 2
  %14 = extractvalue { ptr, ptr, ptr } %13, 1
  %15 = extractvalue { ptr, ptr, ptr } %13, 0
  %16 = invoke double %15(ptr %14)
          to label %17 unwind label %18

17:                                               ; preds = %1
  store double %16, ptr %5, align 8
  br label %157

18:                                               ; preds = %1
  %19 = landingpad { ptr, i32 }
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  %22 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %18
  %23 = icmp eq ptr %22, @_ZTIi
  br i1 %23, label %24, label %31

24:                                               ; preds = %.noexc11
  %25 = load i32, ptr %21, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %154

31:                                               ; preds = %.noexc11
  %32 = icmp eq ptr %22, @_ZTIj
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load i32, ptr %21, align 4
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 0
  store i64 4, ptr %37, align 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 2
  store i32 %34, ptr %39, align 4
  br label %151

40:                                               ; preds = %31
  %41 = icmp eq ptr %22, @_ZTId
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load double, ptr %21, align 8
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 0
  store i64 8, ptr %46, align 8
  %47 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 2
  store double %43, ptr %48, align 8
  br label %148

49:                                               ; preds = %40
  %50 = icmp eq ptr %22, @_ZTIf
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load float, ptr %21, align 4
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 0
  store i64 4, ptr %55, align 8
  %56 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 2
  store float %52, ptr %57, align 4
  br label %145

58:                                               ; preds = %49
  %59 = icmp eq ptr %22, @_ZTIb
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i1, ptr %21, align 1
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 0
  store i64 1, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 2
  store i1 %61, ptr %66, align 1
  br label %142

67:                                               ; preds = %58
  %68 = icmp eq ptr %22, @_ZTIx
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load i64, ptr %21, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 2
  store i64 %70, ptr %75, align 8
  br label %139

76:                                               ; preds = %67
  %77 = icmp eq ptr %22, @_ZTIPc
  br i1 %77, label %78, label %85

78:                                               ; preds = %76
  %79 = load ptr, ptr %21, align 8
  %80 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %78
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 0
  store i64 8, ptr %82, align 8
  %83 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %83, align 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 2
  store ptr %79, ptr %84, align 8
  br label %136

85:                                               ; preds = %76
  %86 = icmp eq ptr %22, @_ZTIPv
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = load ptr, ptr %21, align 8
  br label %133

89:                                               ; preds = %85
  %90 = load ptr, ptr %22, align 8
  %91 = icmp eq ptr %90, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %91, label %92, label %123

92:                                               ; preds = %89
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 2
  %94 = load i32, ptr %93, align 4
  %95 = and i32 %94, 1073741824
  %96 = icmp ne i32 %95, 0
  br i1 %96, label %97, label %113

97:                                               ; preds = %92
  %98 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 4
  %99 = load ptr, ptr %98, align 8
  %100 = icmp ne ptr %99, null
  br i1 %100, label %101, label %103

101:                                              ; preds = %97
  invoke void %99(ptr %2, ptr %21)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %101
  %102 = load ptr, ptr %2, align 8
  br label %110

103:                                              ; preds = %97
  %104 = load ptr, ptr %21, align 8
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

110:                                              ; preds = %.noexc10, %.noexc7
  %111 = phi ptr [ %106, %.noexc7 ], [ %102, %.noexc10 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %92
  %114 = load ptr, ptr %21, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc4
  %121 = phi ptr [ %116, %.noexc4 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %89
  %124 = load ptr, ptr %21, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc9
  %131 = phi ptr [ %126, %.noexc9 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %87, %132
  %134 = phi ptr [ %131, %132 ], [ %88, %87 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc, %135
  %137 = phi ptr [ %134, %135 ], [ %81, %.noexc ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc2, %138
  %140 = phi ptr [ %137, %138 ], [ %72, %.noexc2 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc1, %141
  %143 = phi ptr [ %140, %141 ], [ %63, %.noexc1 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc6, %144
  %146 = phi ptr [ %143, %144 ], [ %54, %.noexc6 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc5, %147
  %149 = phi ptr [ %146, %147 ], [ %45, %.noexc5 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc3, %150
  %152 = phi ptr [ %149, %150 ], [ %36, %.noexc3 ]
  br label %153

153:                                              ; preds = %151
  br label %154

154:                                              ; preds = %.noexc8, %153
  %155 = phi ptr [ %152, %153 ], [ %27, %.noexc8 ]
  br label %156

156:                                              ; preds = %154
  store ptr %155, ptr %3, align 8
  store double -2.000000e+00, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %157

157:                                              ; preds = %17, %156
  %158 = load double, ptr %5, align 8
  ret double %158

endcatch.resume:                                  ; preds = %18, %101, %123, %24, %103, %51, %42, %113, %33, %69, %60, %78
  %159 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %159
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca { ptr, ptr }, align 8
  %2 = call { ptr, ptr } @mkSquare(double 7.000000e+00, double 3.000000e+00)
  %3 = call { ptr, ptr } @mkNull()
  store { ptr, ptr } %3, ptr %1, align 8
  %4 = extractvalue { ptr, ptr } %2, 0
  %5 = call ptr @ts_remap_vtable(ptr %4, ptr @ts_vtable_map_17875616299094573122, i64 3)
  %6 = insertvalue { ptr, ptr } %2, ptr %5, 0
  store { ptr, ptr } %6, ptr %1, align 8
  %7 = load { ptr, ptr }, ptr %1, align 8
  %8 = extractvalue { ptr, ptr } %7, 0
  %9 = extractvalue { ptr, ptr } %7, 1
  %10 = getelementptr ptr, ptr %8, i32 2
  %11 = load ptr, ptr %10, align 8
  %12 = ptrtoint ptr %9 to i64
  %13 = ptrtoint ptr %11 to i64
  %14 = add i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = extractvalue { ptr, ptr } %7, 0
  %17 = extractvalue { ptr, ptr } %7, 1
  %18 = getelementptr ptr, ptr %16, i32 2
  %19 = load ptr, ptr %18, align 8
  %20 = ptrtoint ptr %19 to i64
  %21 = and i64 %20, -9223372036854775808
  %22 = icmp ne i64 %21, 0
  br i1 %22, label %23, label %29

23:                                               ; preds = %0
  %24 = sub i64 %20, -9223372036854775808
  %25 = inttoptr i64 %24 to ptr
  %26 = getelementptr { ptr, ptr }, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %26, align 8
  %28 = call double %27(ptr %17)
  br label %31

29:                                               ; preds = %0
  %30 = load double, ptr %15, align 8
  br label %31

31:                                               ; preds = %23, %29
  %32 = phi double [ %30, %29 ], [ %28, %23 ]
  br label %33

33:                                               ; preds = %31
  %34 = fcmp oeq double %32, 7.000000e+00
  br i1 %34, label %35, label %115

35:                                               ; preds = %33
  %36 = extractvalue { ptr, ptr } %2, 0
  %37 = extractvalue { ptr, ptr } %2, 1
  %38 = getelementptr ptr, ptr %36, i32 3
  %39 = load ptr, ptr %38, align 8
  %40 = ptrtoint ptr %37 to i64
  %41 = ptrtoint ptr %39 to i64
  %42 = add i64 %40, %41
  %43 = inttoptr i64 %42 to ptr
  %44 = extractvalue { ptr, ptr } %2, 0
  %45 = extractvalue { ptr, ptr } %2, 1
  %46 = getelementptr ptr, ptr %44, i32 3
  %47 = load ptr, ptr %46, align 8
  %48 = ptrtoint ptr %47 to i64
  %49 = and i64 %48, -9223372036854775808
  %50 = icmp ne i64 %49, 0
  br i1 %50, label %51, label %57

51:                                               ; preds = %35
  %52 = sub i64 %48, -9223372036854775808
  %53 = inttoptr i64 %52 to ptr
  %54 = getelementptr { ptr, ptr }, ptr %53, i32 0, i32 0
  %55 = load ptr, ptr %54, align 8
  %56 = call double %55(ptr %45)
  br label %59

57:                                               ; preds = %35
  %58 = load double, ptr %43, align 8
  br label %59

59:                                               ; preds = %51, %57
  %60 = phi double [ %58, %57 ], [ %56, %51 ]
  br label %61

61:                                               ; preds = %59
  %62 = fcmp oeq double %60, 7.000000e+00
  br i1 %62, label %63, label %117

63:                                               ; preds = %61
  %64 = call double @call({ ptr, ptr } %2)
  %65 = fcmp oeq double %64, 9.000000e+00
  br i1 %65, label %66, label %119

66:                                               ; preds = %63
  %67 = call { ptr, ptr } @mkShape(i1 true)
  %68 = extractvalue { ptr, ptr } %67, 0
  %69 = call ptr @ts_remap_vtable(ptr %68, ptr @ts_vtable_map_17875616299094573122, i64 3)
  %70 = insertvalue { ptr, ptr } %67, ptr %69, 0
  %71 = extractvalue { ptr, ptr } %70, 0
  %72 = extractvalue { ptr, ptr } %70, 1
  %73 = getelementptr ptr, ptr %71, i32 2
  %74 = load ptr, ptr %73, align 8
  %75 = ptrtoint ptr %72 to i64
  %76 = ptrtoint ptr %74 to i64
  %77 = add i64 %75, %76
  %78 = inttoptr i64 %77 to ptr
  %79 = extractvalue { ptr, ptr } %70, 0
  %80 = extractvalue { ptr, ptr } %70, 1
  %81 = getelementptr ptr, ptr %79, i32 2
  %82 = load ptr, ptr %81, align 8
  %83 = ptrtoint ptr %82 to i64
  %84 = and i64 %83, -9223372036854775808
  %85 = icmp ne i64 %84, 0
  br i1 %85, label %86, label %92

86:                                               ; preds = %66
  %87 = sub i64 %83, -9223372036854775808
  %88 = inttoptr i64 %87 to ptr
  %89 = getelementptr { ptr, ptr }, ptr %88, i32 0, i32 0
  %90 = load ptr, ptr %89, align 8
  %91 = call double %90(ptr %80)
  br label %94

92:                                               ; preds = %66
  %93 = load double, ptr %78, align 8
  br label %94

94:                                               ; preds = %86, %92
  %95 = phi double [ %93, %92 ], [ %91, %86 ]
  br label %96

96:                                               ; preds = %94
  %97 = fcmp oeq double %95, 5.000000e+00
  br i1 %97, label %98, label %121

98:                                               ; preds = %96
  %99 = call { ptr, ptr } @mkShape(i1 false)
  %100 = extractvalue { ptr, ptr } %99, 0
  %101 = call ptr @ts_remap_vtable(ptr %100, ptr @ts_vtable_map_17875616299094573122, i64 3)
  %102 = insertvalue { ptr, ptr } %99, ptr %101, 0
  %103 = extractvalue { ptr, ptr } %102, 1
  %104 = ptrtoint ptr %103 to i64
  %105 = icmp eq i64 %104, 0
  br i1 %105, label %106, label %123

106:                                              ; preds = %98
  %107 = extractvalue { ptr, ptr } %99, 1
  %108 = ptrtoint ptr %107 to i64
  %109 = icmp eq i64 %108, 0
  br i1 %109, label %110, label %125

110:                                              ; preds = %106
  %111 = call double @call({ ptr, ptr } %99)
  %112 = fcmp oeq double %111, -2.000000e+00
  br i1 %112, label %113, label %127

113:                                              ; preds = %110
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

114:                                              ; preds = %127, %125, %123, %121, %119, %117, %115
  unreachable

115:                                              ; preds = %33
  %116 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4745878797808296967, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 63, ptr null)
  br label %114

117:                                              ; preds = %61
  %118 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1039812570711373850, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 67, ptr null)
  br label %114

119:                                              ; preds = %63
  %120 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14662280579041655297, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 68, ptr null)
  br label %114

121:                                              ; preds = %96
  %122 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_564923817551714954, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 72, ptr null)
  br label %114

123:                                              ; preds = %98
  %124 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12204294225069353136, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 77, ptr null)
  br label %114

125:                                              ; preds = %106
  %126 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14778402311747416871, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 79, ptr null)
  br label %114

127:                                              ; preds = %110
  %128 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14728058011860429015, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2658979206283382412, i64 8), i32 80, ptr null)
  br label %114
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
attributes #2 = { mustprogress "__null_interface_vtable" }
attributes #3 = { mustprogress noinline }
attributes #4 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

