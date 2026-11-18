; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, ptr, double }

$C..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_12265111460882272040 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_err_borrow_catch_local.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_3978046374080941754 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNaN\00", align 8
@s_865306855574198937 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInfinity\00", align 8
@s_5514277459305441004 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-Infinity\00", align 8
@s_555417232352566889 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s\00", align 8
@s_12638135523509116079 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF0\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_12638121229857949336 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-\00", align 8
@s_13533418361890864287 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%.*e\00", align 8
@s_565227158805101677 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%c%.*s\00", align 8
@s_3707881812392243519 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF000000000000000000000\00", align 8
@s_14597275362824401209 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%s%.*s\00", align 8
@s_2324455066069071285 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%.*s.%s\00", align 8
@s_3663401783476796495 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s0.%.*s%s\00", align 8
@s_12638124528392833969 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.\00", align 8
@s_17791929768168046296 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%c%s%se%+d\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@frmt_1189076927660576410 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-lli\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 24, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C.m, ptr @C..rtti, ptr @C..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_17033125819989208466 = internal constant [0 x double] undef
@a_0_2926441800300339944 = internal constant [0 x ptr] undef
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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

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

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare void @puts(ptr)

define internal void @__ts_number_to_string(ptr %0, double %1) {
  %3 = alloca i8, i32 32, align 1
  %4 = alloca i8, i32 32, align 1
  %5 = fcmp olt double %1, 0.000000e+00
  %6 = fcmp uno double %1, %1
  br i1 %6, label %7, label %9

7:                                                ; preds = %2
  %8 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_3978046374080941754, i64 8))
  ret void

9:                                                ; preds = %2
  %10 = call double @llvm.fabs.f64(double %1)
  %11 = fcmp oeq double %10, 0x7FF0000000000000
  br i1 %11, label %12, label %15

12:                                               ; preds = %9
  %13 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_5514277459305441004, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_865306855574198937, i64 8)
  %14 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_555417232352566889, i64 8), ptr %13)
  ret void

15:                                               ; preds = %9
  %16 = fcmp oeq double %1, 0.000000e+00
  br i1 %16, label %17, label %19

17:                                               ; preds = %15
  %18 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_12638135523509116079, i64 8))
  ret void

19:                                               ; preds = %15
  %20 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_12638121229857949336, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
  br label %21

21:                                               ; preds = %21, %19
  %22 = phi i32 [ %28, %21 ], [ 0, %19 ]
  %23 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %3, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_13533418361890864287, i64 8), i32 %22, double %10)
  %24 = call double @strtod(ptr %3, ptr null)
  %25 = fcmp oeq double %24, %10
  %26 = icmp sge i32 %22, 16
  %27 = or i1 %25, %26
  %28 = add i32 %22, 1
  br i1 %27, label %29, label %21

29:                                               ; preds = %21
  %30 = add i32 %22, 1
  %31 = icmp eq i32 %22, 0
  %32 = add i32 %22, 3
  %33 = select i1 %31, i32 2, i32 %32
  %34 = getelementptr i8, ptr %3, i32 %33
  %35 = call double @strtod(ptr %34, ptr null)
  %36 = fptosi double %35 to i32
  %37 = add i32 %36, 1
  %38 = load i8, ptr %3, align 1
  %39 = sext i8 %38 to i32
  %40 = getelementptr i8, ptr %3, i32 2
  %41 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %4, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_565227158805101677, i64 8), i32 %39, i32 %22, ptr %40)
  %42 = icmp sle i32 %37, 21
  %43 = icmp sle i32 %30, %37
  %44 = and i1 %43, %42
  br i1 %44, label %45, label %48

45:                                               ; preds = %29
  %46 = sub i32 %37, %30
  %47 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_14597275362824401209, i64 8), ptr %20, ptr %4, i32 %46, ptr getelementptr inbounds nuw (i8, ptr @s_3707881812392243519, i64 8))
  ret void

48:                                               ; preds = %29
  %49 = icmp sgt i32 %37, 0
  %50 = and i1 %49, %42
  br i1 %50, label %51, label %54

51:                                               ; preds = %48
  %52 = getelementptr i8, ptr %4, i32 %37
  %53 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_2324455066069071285, i64 8), ptr %20, i32 %37, ptr %4, ptr %52)
  ret void

54:                                               ; preds = %48
  %55 = icmp sle i32 %37, 0
  %56 = icmp sgt i32 %37, -6
  %57 = and i1 %56, %55
  br i1 %57, label %58, label %61

58:                                               ; preds = %54
  %59 = sub i32 0, %37
  %60 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_3663401783476796495, i64 8), ptr %20, i32 %59, ptr getelementptr inbounds nuw (i8, ptr @s_3707881812392243519, i64 8), ptr %4)
  ret void

61:                                               ; preds = %54
  %62 = icmp sgt i32 %30, 1
  %63 = select i1 %62, ptr getelementptr inbounds nuw (i8, ptr @s_12638124528392833969, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
  %64 = getelementptr i8, ptr %4, i32 1
  %65 = sub i32 %37, 1
  %66 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_17791929768168046296, i64 8), ptr %20, i32 %39, ptr %63, ptr %64, i32 %65)
  ret void
}

declare double @strtod(ptr, ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @C..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @C..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %C, ptr %3, i32 0, i32 0
  store ptr @C..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @C.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %C, ptr %5, i32 0, i32 1
  %7 = call ptr @GC_malloc(i64 8)
  %8 = getelementptr i8, ptr %7, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %8, ptr @a_0_17033125819989208466, i64 0, i1 false)
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  store ptr %8, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  store i64 0, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 2
  store i64 0, ptr %13, align 8
  store ptr %10, ptr %6, align 8
  %14 = load ptr, ptr %4, align 8
  %15 = getelementptr %C, ptr %14, i32 0, i32 2
  %16 = load double, ptr %3, align 8
  store double %16, ptr %15, align 8
  ret void
}

; Function Attrs: mustprogress
define double @C.m(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %C, ptr %4, i32 0, i32 2
  %6 = load double, ptr %5, align 8
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 8)
  %4 = getelementptr i8, ptr %3, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_2926441800300339944, i64 0, i1 false)
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 0
  store ptr %4, ptr %7, align 8
  %8 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 1
  store i64 0, ptr %8, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 2
  store i64 0, ptr %9, align 8
  store ptr %6, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %10

10:                                               ; preds = %108, %0
  %11 = load i32, ptr %1, align 4
  %12 = icmp slt i32 %11, 64
  br i1 %12, label %13, label %112

13:                                               ; preds = %10
  %14 = call ptr @GC_malloc(i64 32)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %C, ptr %15, i32 0, i32 0
  store ptr @C..vtbl, ptr %16, align 8
  call void @C.constructor(ptr %15, double 9.990000e+02)
  %17 = getelementptr %C, ptr %15, i32 0, i32 1
  %18 = load ptr, ptr %17, align 8
  %19 = icmp eq ptr %18, null
  br i1 %19, label %20, label %23

20:                                               ; preds = %13
  %21 = call ptr @GC_malloc(i64 32)
  %22 = getelementptr i8, ptr %21, i64 8
  store ptr %22, ptr %17, align 8
  br label %24

23:                                               ; preds = %13
  br label %24

24:                                               ; preds = %20, %23
  %25 = phi ptr [ %18, %23 ], [ %22, %20 ]
  br label %26

26:                                               ; preds = %24
  %27 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 1
  %28 = load i64, ptr %27, align 8
  %29 = add i64 %28, 1
  %30 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 0
  %31 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 2
  %32 = load ptr, ptr %30, align 8
  %33 = load i64, ptr %31, align 8
  %34 = icmp ugt i64 %29, %33
  br i1 %34, label %35, label %59

35:                                               ; preds = %26
  %36 = mul i64 %33, 2
  %37 = icmp uge i64 %36, %29
  %38 = select i1 %37, i64 %36, i64 %29
  %39 = icmp ult i64 %38, 4
  %40 = select i1 %39, i64 4, i64 %38
  %41 = icmp ule i64 %29, 4294967295
  br i1 %41, label %42, label %114

42:                                               ; preds = %35
  %43 = icmp ule i64 %40, 2305843009213693951
  br i1 %43, label %44, label %116

44:                                               ; preds = %42
  %45 = mul i64 8, %40
  %46 = add i64 %45, 8
  %47 = icmp eq ptr %32, null
  %48 = getelementptr i8, ptr %32, i64 -8
  %49 = select i1 %47, ptr null, ptr %48
  %50 = call ptr @GC_realloc(ptr %49, i64 %46)
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = icmp ne ptr %51, null
  %53 = icmp eq i64 %45, 0
  %54 = or i1 %52, %53
  br i1 %54, label %55, label %118

55:                                               ; preds = %44
  %56 = mul i64 8, %33
  %57 = getelementptr i8, ptr %51, i64 %56
  %58 = sub i64 %45, %56
  call void @llvm.memset.p0.i64(ptr %57, i8 0, i64 %58, i1 false)
  store ptr %51, ptr %30, align 8
  store i64 %40, ptr %31, align 8
  br label %60

59:                                               ; preds = %26
  br label %60

60:                                               ; preds = %55, %59
  %61 = phi ptr [ %32, %59 ], [ %51, %55 ]
  br label %62

62:                                               ; preds = %60
  %63 = getelementptr double, ptr %61, i64 %28
  store double 9.900000e+01, ptr %63, align 8
  store i64 %29, ptr %27, align 8
  %64 = load ptr, ptr %2, align 8
  %65 = icmp eq ptr %64, null
  br i1 %65, label %66, label %69

66:                                               ; preds = %62
  %67 = call ptr @GC_malloc(i64 32)
  %68 = getelementptr i8, ptr %67, i64 8
  store ptr %68, ptr %2, align 8
  br label %70

69:                                               ; preds = %62
  br label %70

70:                                               ; preds = %66, %69
  %71 = phi ptr [ %64, %69 ], [ %68, %66 ]
  br label %72

72:                                               ; preds = %70
  %73 = getelementptr { ptr, i64, i64 }, ptr %71, i32 0, i32 1
  %74 = load i64, ptr %73, align 8
  %75 = add i64 %74, 1
  %76 = getelementptr { ptr, i64, i64 }, ptr %71, i32 0, i32 0
  %77 = getelementptr { ptr, i64, i64 }, ptr %71, i32 0, i32 2
  %78 = load ptr, ptr %76, align 8
  %79 = load i64, ptr %77, align 8
  %80 = icmp ugt i64 %75, %79
  br i1 %80, label %81, label %105

81:                                               ; preds = %72
  %82 = mul i64 %79, 2
  %83 = icmp uge i64 %82, %75
  %84 = select i1 %83, i64 %82, i64 %75
  %85 = icmp ult i64 %84, 4
  %86 = select i1 %85, i64 4, i64 %84
  %87 = icmp ule i64 %75, 4294967295
  br i1 %87, label %88, label %120

88:                                               ; preds = %81
  %89 = icmp ule i64 %86, 2305843009213693951
  br i1 %89, label %90, label %122

90:                                               ; preds = %88
  %91 = mul i64 8, %86
  %92 = add i64 %91, 8
  %93 = icmp eq ptr %78, null
  %94 = getelementptr i8, ptr %78, i64 -8
  %95 = select i1 %93, ptr null, ptr %94
  %96 = call ptr @GC_realloc(ptr %95, i64 %92)
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = icmp ne ptr %97, null
  %99 = icmp eq i64 %91, 0
  %100 = or i1 %98, %99
  br i1 %100, label %101, label %124

101:                                              ; preds = %90
  %102 = mul i64 8, %79
  %103 = getelementptr i8, ptr %97, i64 %102
  %104 = sub i64 %91, %102
  call void @llvm.memset.p0.i64(ptr %103, i8 0, i64 %104, i1 false)
  store ptr %97, ptr %76, align 8
  store i64 %86, ptr %77, align 8
  br label %106

105:                                              ; preds = %72
  br label %106

106:                                              ; preds = %101, %105
  %107 = phi ptr [ %78, %105 ], [ %97, %101 ]
  br label %108

108:                                              ; preds = %106
  %109 = getelementptr ptr, ptr %107, i64 %74
  store ptr %15, ptr %109, align 8
  store i64 %75, ptr %73, align 8
  %110 = load i32, ptr %1, align 4
  %111 = add i32 %110, 1
  store i32 %111, ptr %1, align 4
  br label %10

112:                                              ; preds = %10
  ret void

113:                                              ; preds = %124, %122, %120, %118, %116, %114
  unreachable

114:                                              ; preds = %35
  %115 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 14, ptr null)
  br label %113

116:                                              ; preds = %42
  %117 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 14, ptr null)
  br label %113

118:                                              ; preds = %44
  %119 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 14, ptr null)
  br label %113

120:                                              ; preds = %81
  %121 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 15, ptr null)
  br label %113

122:                                              ; preds = %88
  %123 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 15, ptr null)
  br label %113

124:                                              ; preds = %90
  %125 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 15, ptr null)
  br label %113
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %C, ptr %8, i32 0, i32 0
  store ptr @C..vtbl, ptr %9, align 8
  call void @C.constructor(ptr %8, double 1.000000e+00)
  store ptr %8, ptr %6, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %C, ptr %10, i32 0, i32 1
  %12 = load ptr, ptr %11, align 8
  %13 = icmp eq ptr %12, null
  br i1 %13, label %14, label %17

14:                                               ; preds = %0
  %15 = call ptr @GC_malloc(i64 32)
  %16 = getelementptr i8, ptr %15, i64 8
  store ptr %16, ptr %11, align 8
  br label %18

17:                                               ; preds = %0
  br label %18

18:                                               ; preds = %14, %17
  %19 = phi ptr [ %12, %17 ], [ %16, %14 ]
  br label %20

20:                                               ; preds = %18
  %21 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 1
  %22 = load i64, ptr %21, align 8
  %23 = add i64 %22, 1
  %24 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 0
  %25 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 2
  %26 = load ptr, ptr %24, align 8
  %27 = load i64, ptr %25, align 8
  %28 = icmp ugt i64 %23, %27
  br i1 %28, label %29, label %53

29:                                               ; preds = %20
  %30 = mul i64 %27, 2
  %31 = icmp uge i64 %30, %23
  %32 = select i1 %31, i64 %30, i64 %23
  %33 = icmp ult i64 %32, 4
  %34 = select i1 %33, i64 4, i64 %32
  %35 = icmp ule i64 %23, 4294967295
  br i1 %35, label %36, label %280

36:                                               ; preds = %29
  %37 = icmp ule i64 %34, 2305843009213693951
  br i1 %37, label %38, label %282

38:                                               ; preds = %36
  %39 = mul i64 8, %34
  %40 = add i64 %39, 8
  %41 = icmp eq ptr %26, null
  %42 = getelementptr i8, ptr %26, i64 -8
  %43 = select i1 %41, ptr null, ptr %42
  %44 = call ptr @GC_realloc(ptr %43, i64 %40)
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = icmp ne ptr %45, null
  %47 = icmp eq i64 %39, 0
  %48 = or i1 %46, %47
  br i1 %48, label %49, label %284

49:                                               ; preds = %38
  %50 = mul i64 8, %27
  %51 = getelementptr i8, ptr %45, i64 %50
  %52 = sub i64 %39, %50
  call void @llvm.memset.p0.i64(ptr %51, i8 0, i64 %52, i1 false)
  store ptr %45, ptr %24, align 8
  store i64 %34, ptr %25, align 8
  br label %54

53:                                               ; preds = %20
  br label %54

54:                                               ; preds = %49, %53
  %55 = phi ptr [ %26, %53 ], [ %45, %49 ]
  br label %56

56:                                               ; preds = %54
  %57 = getelementptr double, ptr %55, i64 %22
  store double 1.000000e+00, ptr %57, align 8
  store i64 %23, ptr %21, align 8
  %58 = load ptr, ptr %6, align 8
  store ptr %58, ptr %5, align 8
  %59 = load ptr, ptr %6, align 8
  store ptr %59, ptr %4, align 8
  %60 = load ptr, ptr %6, align 8
  %61 = getelementptr %C, ptr %60, i32 0, i32 2
  %62 = load double, ptr %61, align 8
  %63 = load ptr, ptr %4, align 8
  %64 = getelementptr %C, ptr %63, i32 0, i32 2
  %65 = load double, ptr %64, align 8
  %66 = call ptr @GC_malloc_atomic(i64 40)
  %67 = getelementptr i8, ptr %66, i64 8
  call void @__ts_number_to_string(ptr %67, double %62)
  %68 = call ptr @GC_malloc_atomic(i64 40)
  %69 = getelementptr i8, ptr %68, i64 8
  call void @__ts_number_to_string(ptr %69, double %65)
  %70 = call ptr @llvm.stacksave.p0()
  %71 = icmp eq ptr %67, null
  %72 = select i1 %71, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %67
  %73 = icmp eq ptr %69, null
  %74 = select i1 %73, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %69
  %75 = call i64 @strlen(ptr %72)
  %76 = add i64 1, %75
  %77 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %78 = add i64 %76, %77
  %79 = call i64 @strlen(ptr %74)
  %80 = add i64 %78, %79
  %81 = alloca i8, i64 %80, align 1
  %82 = getelementptr i8, ptr %81, i64 0
  %83 = sub i64 %80, 0
  %84 = call ptr @__memcpy_chk(ptr %82, ptr %72, i64 %75, i64 %83)
  %85 = add i64 0, %75
  %86 = getelementptr i8, ptr %81, i64 %85
  %87 = sub i64 %80, %85
  %88 = call ptr @__memcpy_chk(ptr %86, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %77, i64 %87)
  %89 = add i64 %85, %77
  %90 = getelementptr i8, ptr %81, i64 %89
  %91 = sub i64 %80, %89
  %92 = call ptr @__memcpy_chk(ptr %90, ptr %74, i64 %79, i64 %91)
  %93 = add i64 %89, %79
  %94 = getelementptr i8, ptr %81, i64 %93
  store i8 0, ptr %94, align 1
  call void @puts(ptr %81)
  call void @llvm.stackrestore.p0(ptr %70)
  %95 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %95, align 4
  invoke void @__cxa_throw(ptr %95, ptr @_ZTIi, ptr null)
          to label %279 unwind label %96

96:                                               ; preds = %56
  %97 = landingpad { ptr, i32 }
          catch ptr null
  %98 = extractvalue { ptr, i32 } %97, 0
  %99 = call ptr @__cxa_begin_catch(ptr %98)
  %100 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %96
  %101 = icmp eq ptr %100, @_ZTIi
  br i1 %101, label %102, label %109

102:                                              ; preds = %.noexc2
  %103 = load i32, ptr %99, align 4
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, i32 }, ptr %105, i32 0, i32 0
  store i64 4, ptr %106, align 8
  %107 = getelementptr { i64, ptr, i32 }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, i32 }, ptr %105, i32 0, i32 2
  store i32 %103, ptr %108, align 4
  br label %232

109:                                              ; preds = %.noexc2
  %110 = icmp eq ptr %100, @_ZTIj
  br i1 %110, label %111, label %118

111:                                              ; preds = %109
  %112 = load i32, ptr %99, align 4
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, i32 }, ptr %114, i32 0, i32 0
  store i64 4, ptr %115, align 8
  %116 = getelementptr { i64, ptr, i32 }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, i32 }, ptr %114, i32 0, i32 2
  store i32 %112, ptr %117, align 4
  br label %229

118:                                              ; preds = %109
  %119 = icmp eq ptr %100, @_ZTId
  br i1 %119, label %120, label %127

120:                                              ; preds = %118
  %121 = load double, ptr %99, align 8
  %122 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %120
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr { i64, ptr, double }, ptr %123, i32 0, i32 0
  store i64 8, ptr %124, align 8
  %125 = getelementptr { i64, ptr, double }, ptr %123, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %125, align 8
  %126 = getelementptr { i64, ptr, double }, ptr %123, i32 0, i32 2
  store double %121, ptr %126, align 8
  br label %226

127:                                              ; preds = %118
  %128 = icmp eq ptr %100, @_ZTIf
  br i1 %128, label %129, label %136

129:                                              ; preds = %127
  %130 = load float, ptr %99, align 4
  %131 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %129
  %132 = getelementptr i8, ptr %131, i64 8
  %133 = getelementptr { i64, ptr, float }, ptr %132, i32 0, i32 0
  store i64 4, ptr %133, align 8
  %134 = getelementptr { i64, ptr, float }, ptr %132, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %134, align 8
  %135 = getelementptr { i64, ptr, float }, ptr %132, i32 0, i32 2
  store float %130, ptr %135, align 4
  br label %223

136:                                              ; preds = %127
  %137 = icmp eq ptr %100, @_ZTIb
  br i1 %137, label %138, label %145

138:                                              ; preds = %136
  %139 = load i1, ptr %99, align 1
  %140 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %138
  %141 = getelementptr i8, ptr %140, i64 8
  %142 = getelementptr { i64, ptr, i1 }, ptr %141, i32 0, i32 0
  store i64 1, ptr %142, align 8
  %143 = getelementptr { i64, ptr, i1 }, ptr %141, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %143, align 8
  %144 = getelementptr { i64, ptr, i1 }, ptr %141, i32 0, i32 2
  store i1 %139, ptr %144, align 1
  br label %220

145:                                              ; preds = %136
  %146 = icmp eq ptr %100, @_ZTIx
  br i1 %146, label %147, label %154

147:                                              ; preds = %145
  %148 = load i64, ptr %99, align 8
  %149 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %147
  %150 = getelementptr i8, ptr %149, i64 8
  %151 = getelementptr { i64, ptr, i64 }, ptr %150, i32 0, i32 0
  store i64 8, ptr %151, align 8
  %152 = getelementptr { i64, ptr, i64 }, ptr %150, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %152, align 8
  %153 = getelementptr { i64, ptr, i64 }, ptr %150, i32 0, i32 2
  store i64 %148, ptr %153, align 8
  br label %217

154:                                              ; preds = %145
  %155 = icmp eq ptr %100, @_ZTIPc
  br i1 %155, label %156, label %163

156:                                              ; preds = %154
  %157 = load ptr, ptr %99, align 8
  %158 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %156
  %159 = getelementptr i8, ptr %158, i64 8
  %160 = getelementptr { i64, ptr, ptr }, ptr %159, i32 0, i32 0
  store i64 8, ptr %160, align 8
  %161 = getelementptr { i64, ptr, ptr }, ptr %159, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %161, align 8
  %162 = getelementptr { i64, ptr, ptr }, ptr %159, i32 0, i32 2
  store ptr %157, ptr %162, align 8
  br label %214

163:                                              ; preds = %154
  %164 = icmp eq ptr %100, @_ZTIPv
  br i1 %164, label %165, label %167

165:                                              ; preds = %163
  %166 = load ptr, ptr %99, align 8
  br label %211

167:                                              ; preds = %163
  %168 = load ptr, ptr %100, align 8
  %169 = icmp eq ptr %168, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %169, label %170, label %201

170:                                              ; preds = %167
  %171 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %100, i32 0, i32 2
  %172 = load i32, ptr %171, align 4
  %173 = and i32 %172, 1073741824
  %174 = icmp ne i32 %173, 0
  br i1 %174, label %175, label %191

175:                                              ; preds = %170
  %176 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %100, i32 0, i32 4
  %177 = load ptr, ptr %176, align 8
  %178 = icmp ne ptr %177, null
  br i1 %178, label %179, label %181

179:                                              ; preds = %175
  invoke void %177(ptr %1, ptr %99)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %179
  %180 = load ptr, ptr %1, align 8
  br label %188

181:                                              ; preds = %175
  %182 = load ptr, ptr %99, align 8
  %183 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %181
  %184 = getelementptr i8, ptr %183, i64 8
  %185 = getelementptr { i64, ptr, ptr }, ptr %184, i32 0, i32 0
  store i64 8, ptr %185, align 8
  %186 = getelementptr { i64, ptr, ptr }, ptr %184, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %186, align 8
  %187 = getelementptr { i64, ptr, ptr }, ptr %184, i32 0, i32 2
  store ptr %182, ptr %187, align 8
  br label %188

188:                                              ; preds = %.noexc20, %.noexc22
  %189 = phi ptr [ %184, %.noexc22 ], [ %180, %.noexc20 ]
  br label %190

190:                                              ; preds = %188
  br label %198

191:                                              ; preds = %170
  %192 = load ptr, ptr %99, align 8
  %193 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %191
  %194 = getelementptr i8, ptr %193, i64 8
  %195 = getelementptr { i64, ptr, ptr }, ptr %194, i32 0, i32 0
  store i64 8, ptr %195, align 8
  %196 = getelementptr { i64, ptr, ptr }, ptr %194, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %196, align 8
  %197 = getelementptr { i64, ptr, ptr }, ptr %194, i32 0, i32 2
  store ptr %192, ptr %197, align 8
  br label %198

198:                                              ; preds = %190, %.noexc4
  %199 = phi ptr [ %194, %.noexc4 ], [ %189, %190 ]
  br label %200

200:                                              ; preds = %198
  br label %208

201:                                              ; preds = %167
  %202 = load ptr, ptr %99, align 8
  %203 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %201
  %204 = getelementptr i8, ptr %203, i64 8
  %205 = getelementptr { i64, ptr, ptr }, ptr %204, i32 0, i32 0
  store i64 8, ptr %205, align 8
  %206 = getelementptr { i64, ptr, ptr }, ptr %204, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %206, align 8
  %207 = getelementptr { i64, ptr, ptr }, ptr %204, i32 0, i32 2
  store ptr %202, ptr %207, align 8
  br label %208

208:                                              ; preds = %200, %.noexc25
  %209 = phi ptr [ %204, %.noexc25 ], [ %199, %200 ]
  br label %210

210:                                              ; preds = %208
  br label %211

211:                                              ; preds = %165, %210
  %212 = phi ptr [ %209, %210 ], [ %166, %165 ]
  br label %213

213:                                              ; preds = %211
  br label %214

214:                                              ; preds = %.noexc3, %213
  %215 = phi ptr [ %212, %213 ], [ %159, %.noexc3 ]
  br label %216

216:                                              ; preds = %214
  br label %217

217:                                              ; preds = %.noexc, %216
  %218 = phi ptr [ %215, %216 ], [ %150, %.noexc ]
  br label %219

219:                                              ; preds = %217
  br label %220

220:                                              ; preds = %.noexc23, %219
  %221 = phi ptr [ %218, %219 ], [ %141, %.noexc23 ]
  br label %222

222:                                              ; preds = %220
  br label %223

223:                                              ; preds = %.noexc24, %222
  %224 = phi ptr [ %221, %222 ], [ %132, %.noexc24 ]
  br label %225

225:                                              ; preds = %223
  br label %226

226:                                              ; preds = %.noexc5, %225
  %227 = phi ptr [ %224, %225 ], [ %123, %.noexc5 ]
  br label %228

228:                                              ; preds = %226
  br label %229

229:                                              ; preds = %.noexc21, %228
  %230 = phi ptr [ %227, %228 ], [ %114, %.noexc21 ]
  br label %231

231:                                              ; preds = %229
  br label %232

232:                                              ; preds = %.noexc1, %231
  %233 = phi ptr [ %230, %231 ], [ %105, %.noexc1 ]
  br label %234

234:                                              ; preds = %232
  store ptr %233, ptr %3, align 8
  %235 = load ptr, ptr %5, align 8
  store ptr %235, ptr %2, align 8
  %236 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %234
  %237 = getelementptr i8, ptr %236, i64 8
  %238 = getelementptr %C, ptr %237, i32 0, i32 0
  store ptr @C..vtbl, ptr %238, align 8
  invoke void @C.constructor(ptr %237, double 2.000000e+00)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  store ptr %237, ptr %6, align 8
  invoke void @churn()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %239 = load ptr, ptr %2, align 8
  %240 = getelementptr %C, ptr %239, i32 0, i32 2
  %241 = load double, ptr %240, align 8
  %242 = load ptr, ptr %2, align 8
  %243 = getelementptr %C, ptr %242, i32 0, i32 1
  %244 = load ptr, ptr %243, align 8
  %245 = icmp eq ptr %244, null
  %246 = select i1 %245, ptr @__ts_empty_array, ptr %244
  %247 = getelementptr { ptr, i64, i64 }, ptr %246, i32 0, i32 1
  %248 = load i64, ptr %247, align 8
  %249 = invoke ptr @GC_malloc_atomic(i64 40)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %250 = getelementptr i8, ptr %249, i64 8
  invoke void @__ts_number_to_string(ptr %250, double %241)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %.noexc9
  %251 = invoke ptr @GC_malloc_atomic(i64 58)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  %252 = getelementptr i8, ptr %251, i64 8
  %253 = invoke i32 (ptr, i64, ptr, ...) @snprintf(ptr %252, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %248)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %.noexc11
  %254 = call ptr @llvm.stacksave.p0()
  %255 = icmp eq ptr %250, null
  %256 = select i1 %255, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %250
  %257 = icmp eq ptr %252, null
  %258 = select i1 %257, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %252
  %259 = invoke i64 @strlen(ptr %256)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %.noexc12
  %260 = add i64 1, %259
  %261 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %.noexc13
  %262 = add i64 %260, %261
  %263 = invoke i64 @strlen(ptr %258)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %.noexc14
  %264 = add i64 %262, %263
  %265 = alloca i8, i64 %264, align 1
  %266 = getelementptr i8, ptr %265, i64 0
  %267 = sub i64 %264, 0
  %268 = invoke ptr @__memcpy_chk(ptr %266, ptr %256, i64 %259, i64 %267)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %.noexc15
  %269 = add i64 0, %259
  %270 = getelementptr i8, ptr %265, i64 %269
  %271 = sub i64 %264, %269
  %272 = invoke ptr @__memcpy_chk(ptr %270, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %261, i64 %271)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %.noexc16
  %273 = add i64 %269, %261
  %274 = getelementptr i8, ptr %265, i64 %273
  %275 = sub i64 %264, %273
  %276 = invoke ptr @__memcpy_chk(ptr %274, ptr %258, i64 %263, i64 %275)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %.noexc17
  %277 = add i64 %273, %263
  %278 = getelementptr i8, ptr %265, i64 %277
  store i8 0, ptr %278, align 1
  invoke void @puts(ptr %265)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %.noexc18
  call void @llvm.stackrestore.p0(ptr %254)
  call void @__cxa_end_catch()
  ret i32 0

279:                                              ; preds = %56, %284, %282, %280
  unreachable

280:                                              ; preds = %29
  %281 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 21, ptr null)
  br label %279

282:                                              ; preds = %36
  %283 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 21, ptr null)
  br label %279

284:                                              ; preds = %38
  %285 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12265111460882272040, i64 8), i32 21, ptr null)
  br label %279

endcatch.resume:                                  ; preds = %201, %129, %138, %181, %111, %179, %.noexc18, %.noexc17, %.noexc16, %.noexc15, %.noexc14, %.noexc13, %.noexc12, %.noexc11, %.noexc10, %.noexc9, %.noexc8, %.noexc7, %.noexc6, %234, %120, %191, %156, %96, %102, %147
  %286 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %286
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #6

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #6

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

