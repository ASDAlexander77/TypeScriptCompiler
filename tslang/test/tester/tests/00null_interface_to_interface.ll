; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%E = type { ptr, double, double }

$E..size = comdat any

@ts_vtable_map_17321853624197266297 = internal constant [4 x i32] [i32 0, i32 1, i32 2, i32 4]
@ts_remap_vtable_list = internal global ptr null
@s_12638217986881231904 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFE\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_13245662913593092714 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field read first\00", align 8
@f_4676464794216407311 = internal constant [119 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00null_interface_to_interface.ts\00", align 8
@m_3415245582621781851 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class object seen through another interface\00", align 8
@m_7681166024898334302 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null class object seen through another interface\00", align 8
@m_14728058011860429015 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFits method throws\00", align 8
@m_12719377034891008438 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull seen through another interface\00", align 8
@m_13623463018656580609 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFits method throws too\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@E..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638217986881231904, i64 8)
@E..size = global i64 24, comdat
@E..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @E..instanceOf, ptr @E..new, ptr @E.get, ptr @E..rtti, ptr @E..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@E.Hit..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_7632308648875476676_class, i64 40), ptr @E.Hit..vtbl..instanceOf.dispatch, ptr @E.Hit..vtbl.get.dispatch, ptr inttoptr (i64 8 to ptr), ptr inttoptr (i64 16 to ptr) }
@td_7632308648875476676_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_7632308648875476676, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Hit.null..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @Hit.null.get, ptr null, ptr null }
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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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

declare i32 @strcmp(ptr, ptr)

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

define internal void @tsrel_7632308648875476676(ptr %0) {
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

; Function Attrs: mustprogress
define i1 @E..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @E..rtti, align 8
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
define ptr @E..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %E, ptr %3, i32 0, i32 0
  store ptr @E..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @E.constructor(ptr %0, double %1, double %2) #1 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %E, ptr %7, i32 0, i32 1
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %E, ptr %10, i32 0, i32 2
  %12 = load double, ptr %4, align 8
  store double %12, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress
define double @E.get(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %E, ptr %4, i32 0, i32 2
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, 1.000000e+01
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress noinline
define ptr @mkE(i1 %0, double %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca i1, align 1
  %7 = alloca ptr, align 8
  store i1 %0, ptr %6, align 1
  store double %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %8 = load i1, ptr %6, align 1
  br i1 %8, label %9, label %15

9:                                                ; preds = %3
  %10 = call ptr @GC_malloc(i64 32)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %E, ptr %11, i32 0, i32 0
  store ptr @E..vtbl, ptr %12, align 8
  %13 = load double, ptr %5, align 8
  %14 = load double, ptr %4, align 8
  call void @E.constructor(ptr %11, double %13, double %14)
  br label %16

15:                                               ; preds = %3
  br label %16

16:                                               ; preds = %9, %15
  %17 = phi ptr [ null, %15 ], [ %11, %9 ]
  store ptr %17, ptr %7, align 8
  %18 = load ptr, ptr %7, align 8
  ret ptr %18
}

; Function Attrs: mustprogress
define double @call({ ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
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
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %18
  %23 = icmp eq ptr %22, @_ZTIi
  br i1 %23, label %24, label %31

24:                                               ; preds = %.noexc3
  %25 = load i32, ptr %21, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %154

31:                                               ; preds = %.noexc3
  %32 = icmp eq ptr %22, @_ZTIj
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load i32, ptr %21, align 4
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %33
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %42
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %51
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %60
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %69
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %78
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %101
  %102 = load ptr, ptr %2, align 8
  br label %110

103:                                              ; preds = %97
  %104 = load ptr, ptr %21, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %.noexc, %.noexc6
  %111 = phi ptr [ %106, %.noexc6 ], [ %102, %.noexc ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %92
  %114 = load ptr, ptr %21, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc9
  %121 = phi ptr [ %116, %.noexc9 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %89
  %124 = load ptr, ptr %21, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc2
  %131 = phi ptr [ %126, %.noexc2 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %87, %132
  %134 = phi ptr [ %131, %132 ], [ %88, %87 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc5, %135
  %137 = phi ptr [ %134, %135 ], [ %81, %.noexc5 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc7, %138
  %140 = phi ptr [ %137, %138 ], [ %72, %.noexc7 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc10, %141
  %143 = phi ptr [ %140, %141 ], [ %63, %.noexc10 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc8, %144
  %146 = phi ptr [ %143, %144 ], [ %54, %.noexc8 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc11, %147
  %149 = phi ptr [ %146, %147 ], [ %45, %.noexc11 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc1, %150
  %152 = phi ptr [ %149, %150 ], [ %36, %.noexc1 ]
  br label %153

153:                                              ; preds = %151
  br label %154

154:                                              ; preds = %.noexc4, %153
  %155 = phi ptr [ %152, %153 ], [ %27, %.noexc4 ]
  br label %156

156:                                              ; preds = %154
  store ptr %155, ptr %3, align 8
  store double -2.000000e+00, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %157

157:                                              ; preds = %17, %156
  %158 = load double, ptr %5, align 8
  ret double %158

endcatch.resume:                                  ; preds = %42, %60, %113, %51, %69, %103, %78, %24, %18, %123, %33, %101
  %159 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %159
}

; Function Attrs: mustprogress
define internal i1 @E.Hit..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  %4 = getelementptr %E, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = call i1 %7(ptr %0, ptr %1)
  store i1 %8, ptr %3, align 1
  %9 = load i1, ptr %3, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define internal double @E.Hit..vtbl.get.dispatch(ptr %0) #1 {
  %2 = alloca double, align 8
  %3 = getelementptr %E, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr ptr, ptr %4, i32 2
  %6 = load ptr, ptr %5, align 8
  %7 = call double %6(ptr %0)
  store double %7, ptr %2, align 8
  %8 = load double, ptr %2, align 8
  ret double %8
}

; Function Attrs: mustprogress
define double @Hit.null.get(ptr %0) #4 {
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

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call ptr @mkE(i1 true, double 1.000000e+00, double 2.000000e+00)
  %2 = ptrtoint ptr %1 to i64
  %3 = icmp ne i64 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  br label %6

5:                                                ; preds = %0
  br label %6

6:                                                ; preds = %4, %5
  %7 = phi ptr [ @Hit.null..vtbl, %5 ], [ @E.Hit..vtbl, %4 ]
  %8 = insertvalue { ptr, ptr } undef, ptr %7, 0
  %9 = insertvalue { ptr, ptr } %8, ptr %1, 1
  %10 = extractvalue { ptr, ptr } %9, 0
  %11 = call ptr @ts_remap_vtable(ptr %10, ptr @ts_vtable_map_17321853624197266297, i64 4)
  %12 = insertvalue { ptr, ptr } %9, ptr %11, 0
  %13 = extractvalue { ptr, ptr } %12, 0
  %14 = extractvalue { ptr, ptr } %12, 1
  %15 = getelementptr ptr, ptr %13, i32 3
  %16 = load ptr, ptr %15, align 8
  %17 = ptrtoint ptr %14 to i64
  %18 = ptrtoint ptr %16 to i64
  %19 = add i64 %17, %18
  %20 = inttoptr i64 %19 to ptr
  %21 = extractvalue { ptr, ptr } %12, 0
  %22 = extractvalue { ptr, ptr } %12, 1
  %23 = getelementptr ptr, ptr %21, i32 3
  %24 = load ptr, ptr %23, align 8
  %25 = ptrtoint ptr %24 to i64
  %26 = and i64 %25, -9223372036854775808
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %34

28:                                               ; preds = %6
  %29 = sub i64 %25, -9223372036854775808
  %30 = inttoptr i64 %29 to ptr
  %31 = getelementptr { ptr, ptr }, ptr %30, i32 0, i32 0
  %32 = load ptr, ptr %31, align 8
  %33 = call double %32(ptr %22)
  br label %36

34:                                               ; preds = %6
  %35 = load double, ptr %20, align 8
  br label %36

36:                                               ; preds = %28, %34
  %37 = phi double [ %35, %34 ], [ %33, %28 ]
  br label %38

38:                                               ; preds = %36
  %39 = fcmp oeq double %37, 2.000000e+00
  br i1 %39, label %40, label %73

40:                                               ; preds = %38
  %41 = call double @call({ ptr, ptr } %12)
  %42 = fcmp oeq double %41, 2.000000e+01
  br i1 %42, label %43, label %75

43:                                               ; preds = %40
  %44 = call ptr @mkE(i1 false, double 1.000000e+00, double 2.000000e+00)
  %45 = ptrtoint ptr %44 to i64
  %46 = icmp ne i64 %45, 0
  br i1 %46, label %47, label %48

47:                                               ; preds = %43
  br label %49

48:                                               ; preds = %43
  br label %49

49:                                               ; preds = %47, %48
  %50 = phi ptr [ @Hit.null..vtbl, %48 ], [ @E.Hit..vtbl, %47 ]
  %51 = insertvalue { ptr, ptr } undef, ptr %50, 0
  %52 = insertvalue { ptr, ptr } %51, ptr %44, 1
  %53 = extractvalue { ptr, ptr } %52, 0
  %54 = call ptr @ts_remap_vtable(ptr %53, ptr @ts_vtable_map_17321853624197266297, i64 4)
  %55 = insertvalue { ptr, ptr } %52, ptr %54, 0
  %56 = extractvalue { ptr, ptr } %55, 1
  %57 = ptrtoint ptr %56 to i64
  %58 = icmp eq i64 %57, 0
  br i1 %58, label %59, label %77

59:                                               ; preds = %49
  %60 = call double @call({ ptr, ptr } %55)
  %61 = fcmp oeq double %60, -2.000000e+00
  br i1 %61, label %62, label %79

62:                                               ; preds = %59
  %63 = call ptr @ts_remap_vtable(ptr @Hit.null..vtbl, ptr @ts_vtable_map_17321853624197266297, i64 4)
  %64 = insertvalue { ptr, ptr } { ptr @Hit.null..vtbl, ptr null }, ptr %63, 0
  %65 = extractvalue { ptr, ptr } %64, 1
  %66 = ptrtoint ptr %65 to i64
  %67 = icmp eq i64 %66, 0
  br i1 %67, label %68, label %81

68:                                               ; preds = %62
  %69 = call double @call({ ptr, ptr } %64)
  %70 = fcmp oeq double %69, -2.000000e+00
  br i1 %70, label %71, label %83

71:                                               ; preds = %68
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

72:                                               ; preds = %83, %81, %79, %77, %75, %73
  unreachable

73:                                               ; preds = %38
  %74 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13245662913593092714, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4676464794216407311, i64 8), i32 41, ptr null)
  br label %72

75:                                               ; preds = %40
  %76 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3415245582621781851, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4676464794216407311, i64 8), i32 42, ptr null)
  br label %72

77:                                               ; preds = %49
  %78 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7681166024898334302, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4676464794216407311, i64 8), i32 46, ptr null)
  br label %72

79:                                               ; preds = %59
  %80 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14728058011860429015, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4676464794216407311, i64 8), i32 47, ptr null)
  br label %72

81:                                               ; preds = %62
  %82 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12719377034891008438, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4676464794216407311, i64 8), i32 51, ptr null)
  br label %72

83:                                               ; preds = %68
  %84 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13623463018656580609, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4676464794216407311, i64 8), i32 52, ptr null)
  br label %72
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
attributes #2 = { mustprogress noinline }
attributes #3 = { mustprogress "personality" }
attributes #4 = { mustprogress "__null_interface_vtable" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

