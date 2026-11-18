; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, double }
%K = type { ptr }

$"___array_reduce<!ts.class<V, !ts.class_storage<V>>,!ts.class<V, !ts.class_storage<V>>>" = comdat any

$V..size = comdat any

$K..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638233380044026858 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFK\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_9349456191639824251 = internal constant [98 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_reduce.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_12682538616232704602 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnegative\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
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
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 16, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V.plus, ptr @V..rtti, ptr @V..size }
@K..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638233380044026858, i64 8)
@K..size = global i64 8, comdat
@K.zero = global ptr null
@K..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @K..instanceOf, ptr @K..new, ptr @K..rtti, ptr @K..size, ptr @K.zero }
@finallies = private global double 0.000000e+00
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_6198119186829423739 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @K.static_constructor, ptr null }]

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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

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
define i1 @V..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @V..rtti, align 8
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
define ptr @V..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %V, ptr %3, i32 0, i32 0
  store ptr @V..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @V.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %V, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @V.plus(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %V, ptr %7, i32 0, i32 0
  store ptr @V..vtbl, ptr %8, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr %V, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = load ptr, ptr %3, align 8
  %13 = getelementptr %V, ptr %12, i32 0, i32 1
  %14 = load double, ptr %13, align 8
  %15 = fadd double %11, %14
  call void @V.constructor(ptr %7, double %15)
  store ptr %7, ptr %5, align 8
  %16 = load ptr, ptr %5, align 8
  ret ptr %16
}

; Function Attrs: mustprogress
define i1 @K..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @K..rtti, align 8
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
define ptr @K..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %K, ptr %3, i32 0, i32 0
  store ptr @K..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @K.static_constructor() #2 {
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %V, ptr %2, i32 0, i32 0
  store ptr @V..vtbl, ptr %3, align 8
  call void @V.constructor(ptr %2, double 0.000000e+00)
  store ptr %2, ptr @K.zero, align 8
  ret void
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 8)
  %4 = getelementptr i8, ptr %3, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_6198119186829423739, i64 0, i1 false)
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

10:                                               ; preds = %61, %0
  %11 = load i32, ptr %1, align 4
  %12 = icmp slt i32 %11, 64
  br i1 %12, label %13, label %65

13:                                               ; preds = %10
  %14 = call ptr @GC_malloc(i64 24)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %V, ptr %15, i32 0, i32 0
  store ptr @V..vtbl, ptr %16, align 8
  call void @V.constructor(ptr %15, double 9.990000e+02)
  %17 = load ptr, ptr %2, align 8
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %22

19:                                               ; preds = %13
  %20 = call ptr @GC_malloc(i64 32)
  %21 = getelementptr i8, ptr %20, i64 8
  store ptr %21, ptr %2, align 8
  br label %23

22:                                               ; preds = %13
  br label %23

23:                                               ; preds = %19, %22
  %24 = phi ptr [ %17, %22 ], [ %21, %19 ]
  br label %25

25:                                               ; preds = %23
  %26 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 1
  %27 = load i64, ptr %26, align 8
  %28 = add i64 %27, 1
  %29 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 0
  %30 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 2
  %31 = load ptr, ptr %29, align 8
  %32 = load i64, ptr %30, align 8
  %33 = icmp ugt i64 %28, %32
  br i1 %33, label %34, label %58

34:                                               ; preds = %25
  %35 = mul i64 %32, 2
  %36 = icmp uge i64 %35, %28
  %37 = select i1 %36, i64 %35, i64 %28
  %38 = icmp ult i64 %37, 4
  %39 = select i1 %38, i64 4, i64 %37
  %40 = icmp ule i64 %28, 4294967295
  br i1 %40, label %41, label %67

41:                                               ; preds = %34
  %42 = icmp ule i64 %39, 2305843009213693951
  br i1 %42, label %43, label %69

43:                                               ; preds = %41
  %44 = mul i64 8, %39
  %45 = add i64 %44, 8
  %46 = icmp eq ptr %31, null
  %47 = getelementptr i8, ptr %31, i64 -8
  %48 = select i1 %46, ptr null, ptr %47
  %49 = call ptr @GC_realloc(ptr %48, i64 %45)
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = icmp ne ptr %50, null
  %52 = icmp eq i64 %44, 0
  %53 = or i1 %51, %52
  br i1 %53, label %54, label %71

54:                                               ; preds = %43
  %55 = mul i64 8, %32
  %56 = getelementptr i8, ptr %50, i64 %55
  %57 = sub i64 %44, %55
  call void @llvm.memset.p0.i64(ptr %56, i8 0, i64 %57, i1 false)
  store ptr %50, ptr %29, align 8
  store i64 %39, ptr %30, align 8
  br label %59

58:                                               ; preds = %25
  br label %59

59:                                               ; preds = %54, %58
  %60 = phi ptr [ %31, %58 ], [ %50, %54 ]
  br label %61

61:                                               ; preds = %59
  %62 = getelementptr ptr, ptr %60, i64 %27
  store ptr %15, ptr %62, align 8
  store i64 %28, ptr %26, align 8
  %63 = load i32, ptr %1, align 4
  %64 = add i32 %63, 1
  store i32 %64, ptr %1, align 4
  br label %10

65:                                               ; preds = %10
  ret void

66:                                               ; preds = %71, %69, %67
  unreachable

67:                                               ; preds = %34
  %68 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9349456191639824251, i64 8), i32 22, ptr null)
  br label %66

69:                                               ; preds = %41
  %70 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9349456191639824251, i64 8), i32 22, ptr null)
  br label %66

71:                                               ; preds = %43
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9349456191639824251, i64 8), i32 22, ptr null)
  br label %66
}

; Function Attrs: mustprogress
define ptr @pick(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = getelementptr %V, ptr %6, i32 0, i32 1
  %8 = load double, ptr %7, align 8
  %9 = fcmp ogt double %8, 2.000000e+00
  br i1 %9, label %10, label %12

10:                                               ; preds = %2
  %11 = load ptr, ptr %4, align 8
  br label %16

12:                                               ; preds = %2
  %13 = load ptr, ptr %4, align 8
  %14 = load ptr, ptr %3, align 8
  %15 = call ptr @V.plus(ptr %13, ptr %14)
  br label %16

16:                                               ; preds = %10, %12
  %17 = phi ptr [ %15, %12 ], [ %11, %10 ]
  store ptr %17, ptr %5, align 8
  %18 = load ptr, ptr %5, align 8
  ret ptr %18
}

; Function Attrs: mustprogress
define ptr @add(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = call ptr @V.plus(ptr %6, ptr %7)
  store ptr %8, ptr %5, align 8
  %9 = load ptr, ptr %5, align 8
  ret ptr %9
}

; Function Attrs: mustprogress
define ptr @early(ptr %0, double %1) #2 {
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store double %1, ptr %4, align 8
  %7 = load double, ptr %4, align 8
  %8 = fcmp ogt double %7, 5.000000e+00
  br i1 %8, label %9, label %11

9:                                                ; preds = %2
  %10 = load ptr, ptr %5, align 8
  store ptr %10, ptr %6, align 8
  br label %32

11:                                               ; preds = %2
  store i32 0, ptr %3, align 4
  br label %12

12:                                               ; preds = %20, %11
  %13 = load i32, ptr %3, align 4
  %14 = load double, ptr %4, align 8
  %15 = sitofp i32 %13 to double
  %16 = fcmp olt double %15, %14
  br i1 %16, label %17, label %23

17:                                               ; preds = %12
  %18 = load i32, ptr %3, align 4
  %19 = icmp eq i32 %18, 3
  br i1 %19, label %23, label %20

20:                                               ; preds = %17
  %21 = load i32, ptr %3, align 4
  %22 = add i32 %21, 1
  store i32 %22, ptr %3, align 4
  br label %12

23:                                               ; preds = %17, %12
  %24 = call ptr @GC_malloc(i64 24)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr %V, ptr %25, i32 0, i32 0
  store ptr @V..vtbl, ptr %26, align 8
  %27 = load ptr, ptr %5, align 8
  %28 = getelementptr %V, ptr %27, i32 0, i32 1
  %29 = load double, ptr %28, align 8
  %30 = load double, ptr %4, align 8
  %31 = fadd double %29, %30
  call void @V.constructor(ptr %25, double %31)
  store ptr %25, ptr %6, align 8
  br label %32

32:                                               ; preds = %9, %23
  %33 = load ptr, ptr %6, align 8
  ret ptr %33
}

; Function Attrs: mustprogress
define ptr @fromLoop(ptr %0, double %1) #2 {
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8
  store double %1, ptr %4, align 8
  store i32 0, ptr %3, align 4
  br label %7

7:                                                ; preds = %17, %2
  %8 = load i32, ptr %3, align 4
  %9 = load double, ptr %4, align 8
  %10 = sitofp i32 %8 to double
  %11 = fcmp olt double %10, %9
  br i1 %11, label %12, label %20

12:                                               ; preds = %7
  %13 = load i32, ptr %3, align 4
  %14 = icmp eq i32 %13, 2
  br i1 %14, label %15, label %17

15:                                               ; preds = %12
  %16 = load ptr, ptr %5, align 8
  store ptr %16, ptr %6, align 8
  br label %28

17:                                               ; preds = %12
  %18 = load i32, ptr %3, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %3, align 4
  br label %7

20:                                               ; preds = %7
  %21 = call ptr @GC_malloc(i64 24)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr %V, ptr %22, i32 0, i32 0
  store ptr @V..vtbl, ptr %23, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr %V, ptr %24, i32 0, i32 1
  %26 = load double, ptr %25, align 8
  %27 = fmul double %26, 1.000000e+01
  call void @V.constructor(ptr %22, double %27)
  store ptr %22, ptr %6, align 8
  br label %28

28:                                               ; preds = %15, %20
  %29 = load ptr, ptr %6, align 8
  ret ptr %29
}

; Function Attrs: mustprogress
define ptr @guarded(ptr %0, double %1) #3 personality ptr @__gxx_personality_v0 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  %7 = fcmp ogt double %6, 1.000000e+00
  br i1 %7, label %8, label %12

8:                                                ; preds = %2
  %9 = load ptr, ptr %4, align 8
  store ptr %9, ptr %5, align 8
  %10 = load double, ptr @finallies, align 8
  %11 = fadd double %10, 1.000000e+00
  store double %11, ptr @finallies, align 8
  br label %22

12:                                               ; preds = %2
  %13 = load double, ptr @finallies, align 8
  %14 = fadd double %13, 1.000000e+00
  store double %14, ptr @finallies, align 8
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %V, ptr %16, i32 0, i32 0
  store ptr @V..vtbl, ptr %17, align 8
  %18 = load ptr, ptr %4, align 8
  %19 = getelementptr %V, ptr %18, i32 0, i32 1
  %20 = load double, ptr %19, align 8
  %21 = fadd double %20, 1.000000e+02
  call void @V.constructor(ptr %16, double %21)
  store ptr %16, ptr %5, align 8
  br label %22

22:                                               ; preds = %8, %12
  %23 = load ptr, ptr %5, align 8
  ret ptr %23
}

; Function Attrs: mustprogress
define ptr @thrower(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  %7 = fcmp olt double %6, 0.000000e+00
  br i1 %7, label %8, label %10

8:                                                ; preds = %2
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12682538616232704602, i64 8), ptr %9, align 8
  call void @__cxa_throw(ptr %9, ptr @_ZTIPc, ptr null)
  br label %25

10:                                               ; preds = %2
  %11 = load double, ptr %3, align 8
  %12 = fcmp oeq double %11, 0.000000e+00
  br i1 %12, label %13, label %15

13:                                               ; preds = %10
  %14 = load ptr, ptr %4, align 8
  store ptr %14, ptr %5, align 8
  br label %23

15:                                               ; preds = %10
  %16 = call ptr @GC_malloc(i64 24)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr %V, ptr %17, i32 0, i32 0
  store ptr @V..vtbl, ptr %18, align 8
  %19 = load ptr, ptr %4, align 8
  %20 = getelementptr %V, ptr %19, i32 0, i32 1
  %21 = load double, ptr %20, align 8
  %22 = fadd double %21, 1.000000e+00
  call void @V.constructor(ptr %17, double %22)
  store ptr %17, ptr %5, align 8
  br label %23

23:                                               ; preds = %13, %15
  %24 = load ptr, ptr %5, align 8
  ret ptr %24

25:                                               ; preds = %8
  unreachable
}

; Function Attrs: mustprogress
define ptr @.f_main..afL88C24FH3492898027(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = call ptr @V.plus(ptr %6, ptr %7)
  store ptr %8, ptr %5, align 8
  %9 = load ptr, ptr %5, align 8
  ret ptr %9
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"___array_reduce<!ts.class<V, !ts.class_storage<V>>,!ts.class<V, !ts.class_storage<V>>>"(ptr %0, { ptr, ptr, ptr } %1, ptr %2) #2 comdat {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca { ptr, ptr, ptr }, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %8, align 8
  store { ptr, ptr, ptr } %1, ptr %7, align 8
  store ptr %2, ptr %6, align 8
  %10 = load ptr, ptr %6, align 8
  store ptr %10, ptr %5, align 8
  store i32 0, ptr %4, align 4
  br label %11

11:                                               ; preds = %42, %3
  %12 = load i32, ptr %4, align 4
  %13 = load ptr, ptr %8, align 8
  %14 = icmp eq ptr %13, null
  %15 = select i1 %14, ptr @__ts_empty_array, ptr %13
  %16 = getelementptr { ptr, i64, i64 }, ptr %15, i32 0, i32 1
  %17 = load i64, ptr %16, align 8
  %18 = sext i32 %12 to i64
  %19 = icmp slt i64 %18, %17
  br i1 %19, label %20, label %45

20:                                               ; preds = %11
  %21 = load i32, ptr %4, align 4
  %22 = load ptr, ptr %8, align 8
  %23 = icmp eq ptr %22, null
  %24 = select i1 %23, ptr @__ts_empty_array, ptr %22
  %25 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 0
  %26 = load ptr, ptr %25, align 8
  %27 = getelementptr ptr, ptr %26, i32 %21
  %28 = load ptr, ptr %27, align 8
  %29 = load { ptr, ptr, ptr }, ptr %7, align 8
  %30 = load ptr, ptr %5, align 8
  %31 = extractvalue { ptr, ptr, ptr } %29, 1
  %32 = ptrtoint ptr %31 to i64
  %33 = icmp ne i64 %32, 0
  br i1 %33, label %34, label %37

34:                                               ; preds = %20
  %35 = extractvalue { ptr, ptr, ptr } %29, 0
  %36 = call ptr %35(ptr %31, ptr %30, ptr %28)
  br label %40

37:                                               ; preds = %20
  %38 = extractvalue { ptr, ptr, ptr } %29, 0
  %39 = call ptr %38(ptr %30, ptr %28)
  br label %40

40:                                               ; preds = %34, %37
  %41 = phi ptr [ %39, %37 ], [ %36, %34 ]
  br label %42

42:                                               ; preds = %40
  store ptr %41, ptr %5, align 8
  %43 = load i32, ptr %4, align 4
  %44 = add i32 %43, 1
  store i32 %44, ptr %4, align 4
  br label %11

45:                                               ; preds = %11
  %46 = load ptr, ptr %5, align 8
  store ptr %46, ptr %9, align 8
  %47 = load ptr, ptr %9, align 8
  ret ptr %47
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca double, align 8
  %7 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %8 = alloca <{ i64, [2 x ptr] }>, align 16
  %9 = alloca ptr, align 8
  %10 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %11 = alloca <{ i64, [3 x ptr] }>, align 16
  %12 = alloca <{ i64, %V }>, align 16
  %13 = alloca <{ i64, %V }>, align 16
  store i64 -1, ptr %13, align 8
  %14 = getelementptr i8, ptr %13, i64 8
  store %V zeroinitializer, ptr %14, align 8
  store i64 -1, ptr %12, align 8
  %15 = getelementptr i8, ptr %12, i64 8
  store %V zeroinitializer, ptr %15, align 8
  %16 = call ptr @GC_malloc(i64 24)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr %V, ptr %17, i32 0, i32 0
  store ptr @V..vtbl, ptr %18, align 8
  call void @V.constructor(ptr %17, double 1.000000e+00)
  %19 = call ptr @GC_malloc(i64 24)
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr %V, ptr %20, i32 0, i32 0
  store ptr @V..vtbl, ptr %21, align 8
  call void @V.constructor(ptr %20, double 2.000000e+00)
  %22 = call ptr @GC_malloc(i64 24)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr %V, ptr %23, i32 0, i32 0
  store ptr @V..vtbl, ptr %24, align 8
  call void @V.constructor(ptr %23, double 3.000000e+00)
  store i64 -1, ptr %11, align 8
  %25 = getelementptr i8, ptr %11, i64 8
  store [3 x ptr] zeroinitializer, ptr %25, align 8
  %26 = getelementptr ptr, ptr %25, i64 0
  store ptr %17, ptr %26, align 8
  %27 = getelementptr ptr, ptr %25, i64 1
  store ptr %20, ptr %27, align 8
  %28 = getelementptr ptr, ptr %25, i64 2
  store ptr %23, ptr %28, align 8
  store i64 -1, ptr %10, align 8
  %29 = getelementptr i8, ptr %10, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %29, align 8
  %30 = getelementptr { ptr, i64, i64 }, ptr %29, i32 0, i32 0
  store ptr %25, ptr %30, align 8
  %31 = getelementptr { ptr, i64, i64 }, ptr %29, i32 0, i32 1
  store i64 3, ptr %31, align 8
  %32 = getelementptr { ptr, i64, i64 }, ptr %29, i32 0, i32 2
  store i64 3, ptr %32, align 8
  store ptr %29, ptr %9, align 8
  %33 = load ptr, ptr %9, align 8
  %34 = call ptr @GC_malloc(i64 24)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr %V, ptr %35, i32 0, i32 0
  store ptr @V..vtbl, ptr %36, align 8
  call void @V.constructor(ptr %35, double 0.000000e+00)
  %37 = call ptr @"___array_reduce<!ts.class<V, !ts.class_storage<V>>,!ts.class<V, !ts.class_storage<V>>>"(ptr %33, { ptr, ptr, ptr } { ptr @.f_main..afL88C24FH3492898027, ptr null, ptr null }, ptr %35)
  %38 = load ptr, ptr %9, align 8
  %39 = load ptr, ptr @K.zero, align 8
  %40 = call ptr @"___array_reduce<!ts.class<V, !ts.class_storage<V>>,!ts.class<V, !ts.class_storage<V>>>"(ptr %38, { ptr, ptr, ptr } { ptr @pick, ptr null, ptr null }, ptr %39)
  call void @churn()
  %41 = getelementptr %V, ptr %37, i32 0, i32 1
  %42 = load double, ptr %41, align 8
  %43 = getelementptr %V, ptr %40, i32 0, i32 1
  %44 = load double, ptr %43, align 8
  %45 = call ptr @GC_malloc_atomic(i64 40)
  %46 = getelementptr i8, ptr %45, i64 8
  call void @__ts_number_to_string(ptr %46, double %42)
  %47 = call ptr @GC_malloc_atomic(i64 40)
  %48 = getelementptr i8, ptr %47, i64 8
  call void @__ts_number_to_string(ptr %48, double %44)
  %49 = call ptr @llvm.stacksave.p0()
  %50 = icmp eq ptr %46, null
  %51 = select i1 %50, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %46
  %52 = icmp eq ptr %48, null
  %53 = select i1 %52, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %48
  %54 = call i64 @strlen(ptr %51)
  %55 = add i64 1, %54
  %56 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %57 = add i64 %55, %56
  %58 = call i64 @strlen(ptr %53)
  %59 = add i64 %57, %58
  %60 = alloca i8, i64 %59, align 1
  %61 = getelementptr i8, ptr %60, i64 0
  %62 = sub i64 %59, 0
  %63 = call ptr @__memcpy_chk(ptr %61, ptr %51, i64 %54, i64 %62)
  %64 = add i64 0, %54
  %65 = getelementptr i8, ptr %60, i64 %64
  %66 = sub i64 %59, %64
  %67 = call ptr @__memcpy_chk(ptr %65, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %56, i64 %66)
  %68 = add i64 %64, %56
  %69 = getelementptr i8, ptr %60, i64 %68
  %70 = sub i64 %59, %68
  %71 = call ptr @__memcpy_chk(ptr %69, ptr %53, i64 %58, i64 %70)
  %72 = add i64 %68, %58
  %73 = getelementptr i8, ptr %60, i64 %72
  store i8 0, ptr %73, align 1
  call void @puts(ptr %60)
  call void @llvm.stackrestore.p0(ptr %49)
  %74 = call ptr @GC_malloc(i64 24)
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = getelementptr %V, ptr %75, i32 0, i32 0
  store ptr @V..vtbl, ptr %76, align 8
  call void @V.constructor(ptr %75, double 5.000000e+00)
  %77 = call ptr @GC_malloc(i64 24)
  %78 = getelementptr i8, ptr %77, i64 8
  %79 = getelementptr %V, ptr %78, i32 0, i32 0
  store ptr @V..vtbl, ptr %79, align 8
  call void @V.constructor(ptr %78, double 7.000000e+00)
  store i64 -1, ptr %8, align 8
  %80 = getelementptr i8, ptr %8, i64 8
  store [2 x ptr] zeroinitializer, ptr %80, align 8
  %81 = getelementptr ptr, ptr %80, i64 0
  store ptr %75, ptr %81, align 8
  %82 = getelementptr ptr, ptr %80, i64 1
  store ptr %78, ptr %82, align 8
  store i64 -1, ptr %7, align 8
  %83 = getelementptr i8, ptr %7, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %83, align 8
  %84 = getelementptr { ptr, i64, i64 }, ptr %83, i32 0, i32 0
  store ptr %80, ptr %84, align 8
  %85 = getelementptr { ptr, i64, i64 }, ptr %83, i32 0, i32 1
  store i64 2, ptr %85, align 8
  %86 = getelementptr { ptr, i64, i64 }, ptr %83, i32 0, i32 2
  store i64 2, ptr %86, align 8
  %87 = load ptr, ptr @K.zero, align 8
  %88 = call ptr @"___array_reduce<!ts.class<V, !ts.class_storage<V>>,!ts.class<V, !ts.class_storage<V>>>"(ptr %83, { ptr, ptr, ptr } { ptr @pick, ptr null, ptr null }, ptr %87)
  call void @churn()
  %89 = getelementptr %V, ptr %88, i32 0, i32 1
  %90 = load double, ptr %89, align 8
  %91 = load ptr, ptr @K.zero, align 8
  %92 = getelementptr %V, ptr %91, i32 0, i32 1
  %93 = load double, ptr %92, align 8
  %94 = call ptr @GC_malloc_atomic(i64 40)
  %95 = getelementptr i8, ptr %94, i64 8
  call void @__ts_number_to_string(ptr %95, double %90)
  %96 = call ptr @GC_malloc_atomic(i64 40)
  %97 = getelementptr i8, ptr %96, i64 8
  call void @__ts_number_to_string(ptr %97, double %93)
  %98 = call ptr @llvm.stacksave.p0()
  %99 = icmp eq ptr %95, null
  %100 = select i1 %99, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %95
  %101 = icmp eq ptr %97, null
  %102 = select i1 %101, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %97
  %103 = call i64 @strlen(ptr %100)
  %104 = add i64 1, %103
  %105 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %106 = add i64 %104, %105
  %107 = call i64 @strlen(ptr %102)
  %108 = add i64 %106, %107
  %109 = alloca i8, i64 %108, align 1
  %110 = getelementptr i8, ptr %109, i64 0
  %111 = sub i64 %108, 0
  %112 = call ptr @__memcpy_chk(ptr %110, ptr %100, i64 %103, i64 %111)
  %113 = add i64 0, %103
  %114 = getelementptr i8, ptr %109, i64 %113
  %115 = sub i64 %108, %113
  %116 = call ptr @__memcpy_chk(ptr %114, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %105, i64 %115)
  %117 = add i64 %113, %105
  %118 = getelementptr i8, ptr %109, i64 %117
  %119 = sub i64 %108, %117
  %120 = call ptr @__memcpy_chk(ptr %118, ptr %102, i64 %107, i64 %119)
  %121 = add i64 %117, %107
  %122 = getelementptr i8, ptr %109, i64 %121
  store i8 0, ptr %122, align 1
  call void @puts(ptr %109)
  call void @llvm.stackrestore.p0(ptr %98)
  %123 = load ptr, ptr %9, align 8
  %124 = call ptr @GC_malloc(i64 24)
  %125 = getelementptr i8, ptr %124, i64 8
  %126 = getelementptr %V, ptr %125, i32 0, i32 0
  store ptr @V..vtbl, ptr %126, align 8
  call void @V.constructor(ptr %125, double 1.000000e+01)
  %127 = call ptr @"___array_reduce<!ts.class<V, !ts.class_storage<V>>,!ts.class<V, !ts.class_storage<V>>>"(ptr %123, { ptr, ptr, ptr } { ptr @add, ptr null, ptr null }, ptr %125)
  call void @churn()
  %128 = getelementptr %V, ptr %127, i32 0, i32 1
  %129 = load double, ptr %128, align 8
  %130 = call ptr @GC_malloc_atomic(i64 40)
  %131 = getelementptr i8, ptr %130, i64 8
  call void @__ts_number_to_string(ptr %131, double %129)
  %132 = icmp eq ptr %131, null
  %133 = select i1 %132, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %131
  call void @puts(ptr %133)
  store double 0.000000e+00, ptr %6, align 8
  store i32 0, ptr %5, align 4
  store i32 0, ptr %4, align 4
  br label %134

134:                                              ; preds = %313, %0
  %135 = load i32, ptr %4, align 4
  %136 = icmp slt i32 %135, 20000
  br i1 %136, label %137, label %331

137:                                              ; preds = %134
  %138 = load i32, ptr %4, align 4
  %139 = srem i32 %138, 9
  %140 = call ptr @GC_malloc(i64 24)
  %141 = getelementptr i8, ptr %140, i64 8
  %142 = getelementptr %V, ptr %141, i32 0, i32 0
  store ptr @V..vtbl, ptr %142, align 8
  %143 = load i32, ptr %4, align 4
  %144 = srem i32 %143, 10
  %145 = sitofp i32 %144 to double
  call void @V.constructor(ptr %141, double %145)
  %146 = sitofp i32 %139 to double
  %147 = call ptr @early(ptr %141, double %146)
  %148 = call ptr @GC_malloc(i64 24)
  %149 = getelementptr i8, ptr %148, i64 8
  %150 = getelementptr %V, ptr %149, i32 0, i32 0
  store ptr @V..vtbl, ptr %150, align 8
  call void @V.constructor(ptr %149, double 1.000000e+00)
  %151 = sitofp i32 %139 to double
  %152 = call ptr @fromLoop(ptr %149, double %151)
  %153 = call ptr @GC_malloc(i64 24)
  %154 = getelementptr i8, ptr %153, i64 8
  %155 = getelementptr %V, ptr %154, i32 0, i32 0
  store ptr @V..vtbl, ptr %155, align 8
  call void @V.constructor(ptr %154, double 2.000000e+00)
  %156 = sitofp i32 %139 to double
  %157 = call ptr @guarded(ptr %154, double %156)
  store %V zeroinitializer, ptr %15, align 8
  %158 = getelementptr %V, ptr %15, i32 0, i32 0
  store ptr @V..vtbl, ptr %158, align 8
  %159 = sitofp i32 %139 to double
  call void @V.constructor(ptr %15, double %159)
  store %V zeroinitializer, ptr %14, align 8
  %160 = getelementptr %V, ptr %14, i32 0, i32 0
  store ptr @V..vtbl, ptr %160, align 8
  call void @V.constructor(ptr %14, double 1.000000e+00)
  %161 = call ptr @add(ptr %15, ptr %14)
  store double 0.000000e+00, ptr %3, align 8
  %162 = call ptr @GC_malloc(i64 24)
  %163 = getelementptr i8, ptr %162, i64 8
  %164 = getelementptr %V, ptr %163, i32 0, i32 0
  store ptr @V..vtbl, ptr %164, align 8
  invoke void @V.constructor(ptr %163, double 3.000000e+00)
          to label %165 unwind label %172

165:                                              ; preds = %137
  %166 = sub i32 %139, 1
  %167 = sitofp i32 %166 to double
  %168 = invoke ptr @thrower(ptr %163, double %167)
          to label %169 unwind label %172

169:                                              ; preds = %165
  %170 = getelementptr %V, ptr %168, i32 0, i32 1
  %171 = load double, ptr %170, align 8
  store double %171, ptr %3, align 8
  br label %313

172:                                              ; preds = %165, %137
  %173 = landingpad { ptr, i32 }
          catch ptr null
  %174 = extractvalue { ptr, i32 } %173, 0
  %175 = call ptr @__cxa_begin_catch(ptr %174)
  %176 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %172
  %177 = icmp eq ptr %176, @_ZTIi
  br i1 %177, label %178, label %185

178:                                              ; preds = %.noexc9
  %179 = load i32, ptr %175, align 4
  %180 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %178
  %181 = getelementptr i8, ptr %180, i64 8
  %182 = getelementptr { i64, ptr, i32 }, ptr %181, i32 0, i32 0
  store i64 4, ptr %182, align 8
  %183 = getelementptr { i64, ptr, i32 }, ptr %181, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %183, align 8
  %184 = getelementptr { i64, ptr, i32 }, ptr %181, i32 0, i32 2
  store i32 %179, ptr %184, align 4
  br label %308

185:                                              ; preds = %.noexc9
  %186 = icmp eq ptr %176, @_ZTIj
  br i1 %186, label %187, label %194

187:                                              ; preds = %185
  %188 = load i32, ptr %175, align 4
  %189 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %187
  %190 = getelementptr i8, ptr %189, i64 8
  %191 = getelementptr { i64, ptr, i32 }, ptr %190, i32 0, i32 0
  store i64 4, ptr %191, align 8
  %192 = getelementptr { i64, ptr, i32 }, ptr %190, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %192, align 8
  %193 = getelementptr { i64, ptr, i32 }, ptr %190, i32 0, i32 2
  store i32 %188, ptr %193, align 4
  br label %305

194:                                              ; preds = %185
  %195 = icmp eq ptr %176, @_ZTId
  br i1 %195, label %196, label %203

196:                                              ; preds = %194
  %197 = load double, ptr %175, align 8
  %198 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %196
  %199 = getelementptr i8, ptr %198, i64 8
  %200 = getelementptr { i64, ptr, double }, ptr %199, i32 0, i32 0
  store i64 8, ptr %200, align 8
  %201 = getelementptr { i64, ptr, double }, ptr %199, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %201, align 8
  %202 = getelementptr { i64, ptr, double }, ptr %199, i32 0, i32 2
  store double %197, ptr %202, align 8
  br label %302

203:                                              ; preds = %194
  %204 = icmp eq ptr %176, @_ZTIf
  br i1 %204, label %205, label %212

205:                                              ; preds = %203
  %206 = load float, ptr %175, align 4
  %207 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %205
  %208 = getelementptr i8, ptr %207, i64 8
  %209 = getelementptr { i64, ptr, float }, ptr %208, i32 0, i32 0
  store i64 4, ptr %209, align 8
  %210 = getelementptr { i64, ptr, float }, ptr %208, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %210, align 8
  %211 = getelementptr { i64, ptr, float }, ptr %208, i32 0, i32 2
  store float %206, ptr %211, align 4
  br label %299

212:                                              ; preds = %203
  %213 = icmp eq ptr %176, @_ZTIb
  br i1 %213, label %214, label %221

214:                                              ; preds = %212
  %215 = load i1, ptr %175, align 1
  %216 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %214
  %217 = getelementptr i8, ptr %216, i64 8
  %218 = getelementptr { i64, ptr, i1 }, ptr %217, i32 0, i32 0
  store i64 1, ptr %218, align 8
  %219 = getelementptr { i64, ptr, i1 }, ptr %217, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %219, align 8
  %220 = getelementptr { i64, ptr, i1 }, ptr %217, i32 0, i32 2
  store i1 %215, ptr %220, align 1
  br label %296

221:                                              ; preds = %212
  %222 = icmp eq ptr %176, @_ZTIx
  br i1 %222, label %223, label %230

223:                                              ; preds = %221
  %224 = load i64, ptr %175, align 8
  %225 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %223
  %226 = getelementptr i8, ptr %225, i64 8
  %227 = getelementptr { i64, ptr, i64 }, ptr %226, i32 0, i32 0
  store i64 8, ptr %227, align 8
  %228 = getelementptr { i64, ptr, i64 }, ptr %226, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %228, align 8
  %229 = getelementptr { i64, ptr, i64 }, ptr %226, i32 0, i32 2
  store i64 %224, ptr %229, align 8
  br label %293

230:                                              ; preds = %221
  %231 = icmp eq ptr %176, @_ZTIPc
  br i1 %231, label %232, label %239

232:                                              ; preds = %230
  %233 = load ptr, ptr %175, align 8
  %234 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %232
  %235 = getelementptr i8, ptr %234, i64 8
  %236 = getelementptr { i64, ptr, ptr }, ptr %235, i32 0, i32 0
  store i64 8, ptr %236, align 8
  %237 = getelementptr { i64, ptr, ptr }, ptr %235, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %237, align 8
  %238 = getelementptr { i64, ptr, ptr }, ptr %235, i32 0, i32 2
  store ptr %233, ptr %238, align 8
  br label %290

239:                                              ; preds = %230
  %240 = icmp eq ptr %176, @_ZTIPv
  br i1 %240, label %241, label %243

241:                                              ; preds = %239
  %242 = load ptr, ptr %175, align 8
  br label %287

243:                                              ; preds = %239
  %244 = load ptr, ptr %176, align 8
  %245 = icmp eq ptr %244, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %245, label %246, label %277

246:                                              ; preds = %243
  %247 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %176, i32 0, i32 2
  %248 = load i32, ptr %247, align 4
  %249 = and i32 %248, 1073741824
  %250 = icmp ne i32 %249, 0
  br i1 %250, label %251, label %267

251:                                              ; preds = %246
  %252 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %176, i32 0, i32 4
  %253 = load ptr, ptr %252, align 8
  %254 = icmp ne ptr %253, null
  br i1 %254, label %255, label %257

255:                                              ; preds = %251
  invoke void %253(ptr %1, ptr %175)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %255
  %256 = load ptr, ptr %1, align 8
  br label %264

257:                                              ; preds = %251
  %258 = load ptr, ptr %175, align 8
  %259 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %257
  %260 = getelementptr i8, ptr %259, i64 8
  %261 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 0
  store i64 8, ptr %261, align 8
  %262 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %262, align 8
  %263 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 2
  store ptr %258, ptr %263, align 8
  br label %264

264:                                              ; preds = %.noexc8, %.noexc3
  %265 = phi ptr [ %260, %.noexc3 ], [ %256, %.noexc8 ]
  br label %266

266:                                              ; preds = %264
  br label %274

267:                                              ; preds = %246
  %268 = load ptr, ptr %175, align 8
  %269 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %267
  %270 = getelementptr i8, ptr %269, i64 8
  %271 = getelementptr { i64, ptr, ptr }, ptr %270, i32 0, i32 0
  store i64 8, ptr %271, align 8
  %272 = getelementptr { i64, ptr, ptr }, ptr %270, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %272, align 8
  %273 = getelementptr { i64, ptr, ptr }, ptr %270, i32 0, i32 2
  store ptr %268, ptr %273, align 8
  br label %274

274:                                              ; preds = %266, %.noexc5
  %275 = phi ptr [ %270, %.noexc5 ], [ %265, %266 ]
  br label %276

276:                                              ; preds = %274
  br label %284

277:                                              ; preds = %243
  %278 = load ptr, ptr %175, align 8
  %279 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %277
  %280 = getelementptr i8, ptr %279, i64 8
  %281 = getelementptr { i64, ptr, ptr }, ptr %280, i32 0, i32 0
  store i64 8, ptr %281, align 8
  %282 = getelementptr { i64, ptr, ptr }, ptr %280, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %282, align 8
  %283 = getelementptr { i64, ptr, ptr }, ptr %280, i32 0, i32 2
  store ptr %278, ptr %283, align 8
  br label %284

284:                                              ; preds = %276, %.noexc2
  %285 = phi ptr [ %280, %.noexc2 ], [ %275, %276 ]
  br label %286

286:                                              ; preds = %284
  br label %287

287:                                              ; preds = %241, %286
  %288 = phi ptr [ %285, %286 ], [ %242, %241 ]
  br label %289

289:                                              ; preds = %287
  br label %290

290:                                              ; preds = %.noexc11, %289
  %291 = phi ptr [ %288, %289 ], [ %235, %.noexc11 ]
  br label %292

292:                                              ; preds = %290
  br label %293

293:                                              ; preds = %.noexc10, %292
  %294 = phi ptr [ %291, %292 ], [ %226, %.noexc10 ]
  br label %295

295:                                              ; preds = %293
  br label %296

296:                                              ; preds = %.noexc7, %295
  %297 = phi ptr [ %294, %295 ], [ %217, %.noexc7 ]
  br label %298

298:                                              ; preds = %296
  br label %299

299:                                              ; preds = %.noexc6, %298
  %300 = phi ptr [ %297, %298 ], [ %208, %.noexc6 ]
  br label %301

301:                                              ; preds = %299
  br label %302

302:                                              ; preds = %.noexc, %301
  %303 = phi ptr [ %300, %301 ], [ %199, %.noexc ]
  br label %304

304:                                              ; preds = %302
  br label %305

305:                                              ; preds = %.noexc4, %304
  %306 = phi ptr [ %303, %304 ], [ %190, %.noexc4 ]
  br label %307

307:                                              ; preds = %305
  br label %308

308:                                              ; preds = %.noexc1, %307
  %309 = phi ptr [ %306, %307 ], [ %181, %.noexc1 ]
  br label %310

310:                                              ; preds = %308
  store ptr %309, ptr %2, align 8
  %311 = load i32, ptr %5, align 4
  %312 = add i32 %311, 1
  store i32 %312, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %313

313:                                              ; preds = %169, %310
  call void @churn()
  %314 = load double, ptr %6, align 8
  %315 = getelementptr %V, ptr %147, i32 0, i32 1
  %316 = load double, ptr %315, align 8
  %317 = getelementptr %V, ptr %152, i32 0, i32 1
  %318 = load double, ptr %317, align 8
  %319 = fadd double %316, %318
  %320 = getelementptr %V, ptr %157, i32 0, i32 1
  %321 = load double, ptr %320, align 8
  %322 = fadd double %319, %321
  %323 = load double, ptr %3, align 8
  %324 = fadd double %322, %323
  %325 = getelementptr %V, ptr %161, i32 0, i32 1
  %326 = load double, ptr %325, align 8
  %327 = fadd double %324, %326
  %328 = fadd double %314, %327
  store double %328, ptr %6, align 8
  %329 = load i32, ptr %4, align 4
  %330 = add i32 %329, 1
  store i32 %330, ptr %4, align 4
  br label %134

331:                                              ; preds = %134
  %332 = load double, ptr %6, align 8
  %333 = load i32, ptr %5, align 4
  %334 = load double, ptr @finallies, align 8
  %335 = call ptr @GC_malloc_atomic(i64 40)
  %336 = getelementptr i8, ptr %335, i64 8
  call void @__ts_number_to_string(ptr %336, double %332)
  %337 = call ptr @GC_malloc_atomic(i64 58)
  %338 = getelementptr i8, ptr %337, i64 8
  %339 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %338, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %333)
  %340 = call ptr @GC_malloc_atomic(i64 40)
  %341 = getelementptr i8, ptr %340, i64 8
  call void @__ts_number_to_string(ptr %341, double %334)
  %342 = call ptr @llvm.stacksave.p0()
  %343 = icmp eq ptr %336, null
  %344 = select i1 %343, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %336
  %345 = icmp eq ptr %338, null
  %346 = select i1 %345, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %338
  %347 = icmp eq ptr %341, null
  %348 = select i1 %347, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %341
  %349 = call i64 @strlen(ptr %344)
  %350 = add i64 1, %349
  %351 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %352 = add i64 %350, %351
  %353 = call i64 @strlen(ptr %346)
  %354 = add i64 %352, %353
  %355 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %356 = add i64 %354, %355
  %357 = call i64 @strlen(ptr %348)
  %358 = add i64 %356, %357
  %359 = alloca i8, i64 %358, align 1
  %360 = getelementptr i8, ptr %359, i64 0
  %361 = sub i64 %358, 0
  %362 = call ptr @__memcpy_chk(ptr %360, ptr %344, i64 %349, i64 %361)
  %363 = add i64 0, %349
  %364 = getelementptr i8, ptr %359, i64 %363
  %365 = sub i64 %358, %363
  %366 = call ptr @__memcpy_chk(ptr %364, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %351, i64 %365)
  %367 = add i64 %363, %351
  %368 = getelementptr i8, ptr %359, i64 %367
  %369 = sub i64 %358, %367
  %370 = call ptr @__memcpy_chk(ptr %368, ptr %346, i64 %353, i64 %369)
  %371 = add i64 %367, %353
  %372 = getelementptr i8, ptr %359, i64 %371
  %373 = sub i64 %358, %371
  %374 = call ptr @__memcpy_chk(ptr %372, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %355, i64 %373)
  %375 = add i64 %371, %355
  %376 = getelementptr i8, ptr %359, i64 %375
  %377 = sub i64 %358, %375
  %378 = call ptr @__memcpy_chk(ptr %376, ptr %348, i64 %357, i64 %377)
  %379 = add i64 %375, %357
  %380 = getelementptr i8, ptr %359, i64 %379
  store i8 0, ptr %380, align 1
  call void @puts(ptr %359)
  call void @llvm.stackrestore.p0(ptr %342)
  %381 = getelementptr %V, ptr %37, i32 0, i32 1
  %382 = load double, ptr %381, align 8
  %383 = fcmp oeq double %382, 6.000000e+00
  br i1 %383, label %384, label %388

384:                                              ; preds = %331
  %385 = getelementptr %V, ptr %40, i32 0, i32 1
  %386 = load double, ptr %385, align 8
  %387 = fcmp oeq double %386, 3.000000e+00
  br label %388

388:                                              ; preds = %384, %331
  %389 = phi i1 [ %387, %384 ], [ %383, %331 ]
  br i1 %389, label %390, label %394

390:                                              ; preds = %388
  %391 = getelementptr %V, ptr %88, i32 0, i32 1
  %392 = load double, ptr %391, align 8
  %393 = fcmp oeq double %392, 0.000000e+00
  br label %394

394:                                              ; preds = %390, %388
  %395 = phi i1 [ %393, %390 ], [ %389, %388 ]
  br i1 %395, label %396, label %401

396:                                              ; preds = %394
  %397 = load ptr, ptr @K.zero, align 8
  %398 = getelementptr %V, ptr %397, i32 0, i32 1
  %399 = load double, ptr %398, align 8
  %400 = fcmp oeq double %399, 0.000000e+00
  br label %401

401:                                              ; preds = %396, %394
  %402 = phi i1 [ %400, %396 ], [ %395, %394 ]
  br i1 %402, label %403, label %407

403:                                              ; preds = %401
  %404 = getelementptr %V, ptr %127, i32 0, i32 1
  %405 = load double, ptr %404, align 8
  %406 = fcmp oeq double %405, 1.600000e+01
  br label %407

407:                                              ; preds = %403, %401
  %408 = phi i1 [ %406, %403 ], [ %402, %401 ]
  br i1 %408, label %409, label %411

409:                                              ; preds = %407
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

410:                                              ; preds = %411
  unreachable

411:                                              ; preds = %407
  %412 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9349456191639824251, i64 8), i32 123, ptr null)
  br label %410

endcatch.resume:                                  ; preds = %232, %223, %172, %255, %214, %205, %267, %187, %257, %277, %178, %196
  %413 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %413
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

