; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, ptr, double }
%N = type { ptr, ptr }

$V..size = comdat any

$N..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638230081509142225 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFN\00", align 8
@s_12638203693230065161 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_3978046374080941754 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNaN\00", align 8
@s_865306855574198937 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInfinity\00", align 8
@s_5514277459305441004 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-Infinity\00", align 8
@s_555417232352566889 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s\00", align 8
@s_12638135523509116079 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF0\00", align 8
@s_12638121229857949336 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-\00", align 8
@s_13533418361890864287 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%.*e\00", align 8
@s_565227158805101677 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%c%.*s\00", align 8
@s_3707881812392243519 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF000000000000000000000\00", align 8
@s_14597275362824401209 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%s%.*s\00", align 8
@s_2324455066069071285 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%.*s.%s\00", align 8
@s_3663401783476796495 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s0.%.*s%s\00", align 8
@s_12638124528392833969 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.\00", align 8
@s_17791929768168046296 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%c%s%se%+d\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_1484228504673423653 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00array_argument_held.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_634741498754143809 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv2\00", align 8
@s_634738200219259176 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv1\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_2950142705226678939 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan alias of a new array, in a loop\00", align 8
@m_16306909777058545598 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan alias of a new array, in a try\00", align 8
@m_11939981217937750340 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa new array whose parameter is assigned\00", align 8
@m_7736067829917520921 = internal constant [66 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa new array given on to a call that assigns its parameter\00", align 8
@m_856888895201963235 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa constant array, copied for the call\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 24, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V..rtti, ptr @V..size }
@N..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638230081509142225, i64 8)
@N..size = global i64 16, comdat
@N..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @N..instanceOf, ptr @N..new, ptr @N..rtti, ptr @N..size }
@a_0_14389619779565911031 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

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
  %2 = call ptr @GC_malloc(i64 32)
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
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %V, ptr %7, i32 0, i32 2
  %9 = load double, ptr %3, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr %V, ptr %10, i32 0, i32 1
  %12 = load double, ptr %3, align 8
  %13 = call ptr @GC_malloc_atomic(i64 40)
  %14 = getelementptr i8, ptr %13, i64 8
  call void @__ts_number_to_string(ptr %14, double %12)
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr %16)
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %20, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr %16, i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  store ptr %23, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @N..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @N..rtti, align 8
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
define ptr @N..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %N, ptr %3, i32 0, i32 0
  store ptr @N..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @N.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 1
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %V, ptr %8, i32 0, i32 0
  store ptr @V..vtbl, ptr %9, align 8
  %10 = load double, ptr %3, align 8
  call void @V.constructor(ptr %8, double %10)
  store ptr %8, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @mk(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 0
  store ptr @N..vtbl, ptr %6, align 8
  %7 = load double, ptr %2, align 8
  call void @N.constructor(ptr %5, double %7)
  store ptr %5, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 8)
  %4 = getelementptr i8, ptr %3, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_14389619779565911031, i64 0, i1 false)
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
  %16 = getelementptr %N, ptr %15, i32 0, i32 0
  store ptr @N..vtbl, ptr %16, align 8
  call void @N.constructor(ptr %15, double 9.990000e+02)
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 27, ptr null)
  br label %66

69:                                               ; preds = %41
  %70 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 27, ptr null)
  br label %66

71:                                               ; preds = %43
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 27, ptr null)
  br label %66
}

; Function Attrs: mustprogress
define i1 @aliasInLoop(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store i32 0, ptr %3, align 4
  br label %6

6:                                                ; preds = %9, %1
  %7 = load i32, ptr %3, align 4
  %8 = icmp slt i32 %7, 3
  br i1 %8, label %9, label %13

9:                                                ; preds = %6
  %10 = load ptr, ptr %4, align 8
  store ptr %10, ptr %2, align 8
  call void @churn()
  %11 = load i32, ptr %3, align 4
  %12 = add i32 %11, 1
  store i32 %12, ptr %3, align 4
  br label %6

13:                                               ; preds = %6
  %14 = load ptr, ptr %4, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr @__ts_empty_array, ptr %14
  %17 = getelementptr { ptr, i64, i64 }, ptr %16, i32 0, i32 0
  %18 = load ptr, ptr %17, align 8
  %19 = getelementptr ptr, ptr %18, i32 0
  %20 = load ptr, ptr %19, align 8
  %21 = getelementptr %N, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %21, align 8
  %23 = getelementptr %V, ptr %22, i32 0, i32 2
  %24 = load double, ptr %23, align 8
  %25 = fcmp oeq double %24, 1.000000e+00
  br i1 %25, label %26, label %51

26:                                               ; preds = %13
  %27 = load ptr, ptr %4, align 8
  %28 = icmp eq ptr %27, null
  %29 = select i1 %28, ptr @__ts_empty_array, ptr %27
  %30 = getelementptr { ptr, i64, i64 }, ptr %29, i32 0, i32 0
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr ptr, ptr %31, i32 1
  %33 = load ptr, ptr %32, align 8
  %34 = getelementptr %N, ptr %33, i32 0, i32 1
  %35 = load ptr, ptr %34, align 8
  %36 = getelementptr %V, ptr %35, i32 0, i32 1
  %37 = load ptr, ptr %36, align 8
  %38 = ptrtoint ptr %37 to i64
  %39 = icmp ne i64 %38, 0
  %40 = zext i1 %39 to i32
  %41 = and i32 %40, 1
  %42 = icmp ne i32 %41, 0
  br i1 %42, label %43, label %46

43:                                               ; preds = %26
  %44 = call i32 @strcmp(ptr %37, ptr getelementptr inbounds nuw (i8, ptr @s_634741498754143809, i64 8))
  %45 = icmp eq i32 %44, 0
  br label %48

46:                                               ; preds = %26
  %47 = icmp eq i64 %38, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_634741498754143809, i64 8) to i64)
  br label %48

48:                                               ; preds = %43, %46
  %49 = phi i1 [ %47, %46 ], [ %45, %43 ]
  br label %50

50:                                               ; preds = %48
  br label %51

51:                                               ; preds = %50, %13
  %52 = phi i1 [ %49, %50 ], [ %25, %13 ]
  store i1 %52, ptr %5, align 1
  %53 = load i1, ptr %5, align 1
  ret i1 %53
}

; Function Attrs: mustprogress
define i1 @aliasInTry(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %0, ptr %3, align 8
  store ptr null, ptr %2, align 8
  %5 = load ptr, ptr %3, align 8
  store ptr %5, ptr %2, align 8
  %6 = call ptr @mk(double 3.000000e+00)
  %7 = load ptr, ptr %3, align 8
  %8 = icmp eq ptr %7, null
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = call ptr @GC_malloc(i64 32)
  %11 = getelementptr i8, ptr %10, i64 8
  store ptr %11, ptr %3, align 8
  br label %13

12:                                               ; preds = %1
  br label %13

13:                                               ; preds = %9, %12
  %14 = phi ptr [ %7, %12 ], [ %11, %9 ]
  br label %15

15:                                               ; preds = %13
  %16 = getelementptr { ptr, i64, i64 }, ptr %14, i32 0, i32 1
  %17 = load i64, ptr %16, align 8
  %18 = add i64 %17, 1
  %19 = getelementptr { ptr, i64, i64 }, ptr %14, i32 0, i32 0
  %20 = getelementptr { ptr, i64, i64 }, ptr %14, i32 0, i32 2
  %21 = load ptr, ptr %19, align 8
  %22 = load i64, ptr %20, align 8
  %23 = icmp ugt i64 %18, %22
  br i1 %23, label %24, label %48

24:                                               ; preds = %15
  %25 = mul i64 %22, 2
  %26 = icmp uge i64 %25, %18
  %27 = select i1 %26, i64 %25, i64 %18
  %28 = icmp ult i64 %27, 4
  %29 = select i1 %28, i64 4, i64 %27
  %30 = icmp ule i64 %18, 4294967295
  br i1 %30, label %31, label %91

31:                                               ; preds = %24
  %32 = icmp ule i64 %29, 2305843009213693951
  br i1 %32, label %33, label %93

33:                                               ; preds = %31
  %34 = mul i64 8, %29
  %35 = add i64 %34, 8
  %36 = icmp eq ptr %21, null
  %37 = getelementptr i8, ptr %21, i64 -8
  %38 = select i1 %36, ptr null, ptr %37
  %39 = call ptr @GC_realloc(ptr %38, i64 %35)
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = icmp ne ptr %40, null
  %42 = icmp eq i64 %34, 0
  %43 = or i1 %41, %42
  br i1 %43, label %44, label %95

44:                                               ; preds = %33
  %45 = mul i64 8, %22
  %46 = getelementptr i8, ptr %40, i64 %45
  %47 = sub i64 %34, %45
  call void @llvm.memset.p0.i64(ptr %46, i8 0, i64 %47, i1 false)
  store ptr %40, ptr %19, align 8
  store i64 %29, ptr %20, align 8
  br label %49

48:                                               ; preds = %15
  br label %49

49:                                               ; preds = %44, %48
  %50 = phi ptr [ %21, %48 ], [ %40, %44 ]
  br label %51

51:                                               ; preds = %49
  %52 = getelementptr ptr, ptr %50, i64 %17
  store ptr %6, ptr %52, align 8
  store i64 %18, ptr %16, align 8
  call void @churn()
  %53 = load ptr, ptr %3, align 8
  %54 = icmp eq ptr %53, null
  %55 = select i1 %54, ptr @__ts_empty_array, ptr %53
  %56 = getelementptr { ptr, i64, i64 }, ptr %55, i32 0, i32 1
  %57 = load i64, ptr %56, align 8
  %58 = icmp eq i64 %57, 3
  br i1 %58, label %59, label %72

59:                                               ; preds = %51
  %60 = load ptr, ptr %3, align 8
  %61 = icmp eq ptr %60, null
  %62 = select i1 %61, ptr @__ts_empty_array, ptr %60
  %63 = getelementptr { ptr, i64, i64 }, ptr %62, i32 0, i32 0
  %64 = load ptr, ptr %63, align 8
  %65 = getelementptr ptr, ptr %64, i32 2
  %66 = load ptr, ptr %65, align 8
  %67 = getelementptr %N, ptr %66, i32 0, i32 1
  %68 = load ptr, ptr %67, align 8
  %69 = getelementptr %V, ptr %68, i32 0, i32 2
  %70 = load double, ptr %69, align 8
  %71 = fcmp oeq double %70, 3.000000e+00
  br label %72

72:                                               ; preds = %59, %51
  %73 = phi i1 [ %71, %59 ], [ %58, %51 ]
  br i1 %73, label %74, label %87

74:                                               ; preds = %72
  %75 = load ptr, ptr %3, align 8
  %76 = icmp eq ptr %75, null
  %77 = select i1 %76, ptr @__ts_empty_array, ptr %75
  %78 = getelementptr { ptr, i64, i64 }, ptr %77, i32 0, i32 0
  %79 = load ptr, ptr %78, align 8
  %80 = getelementptr ptr, ptr %79, i32 0
  %81 = load ptr, ptr %80, align 8
  %82 = getelementptr %N, ptr %81, i32 0, i32 1
  %83 = load ptr, ptr %82, align 8
  %84 = getelementptr %V, ptr %83, i32 0, i32 2
  %85 = load double, ptr %84, align 8
  %86 = fcmp oeq double %85, 1.000000e+00
  br label %87

87:                                               ; preds = %74, %72
  %88 = phi i1 [ %86, %74 ], [ %73, %72 ]
  store i1 %88, ptr %4, align 1
  %89 = load i1, ptr %4, align 1
  ret i1 %89

90:                                               ; preds = %95, %93, %91
  unreachable

91:                                               ; preds = %24
  %92 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 44, ptr null)
  br label %90

93:                                               ; preds = %31
  %94 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 44, ptr null)
  br label %90

95:                                               ; preds = %33
  %96 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 44, ptr null)
  br label %90
}

; Function Attrs: mustprogress
define i1 @assigned(ptr %0) #2 {
  %2 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %3 = alloca <{ i64, [1 x ptr] }>, align 16
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr @__ts_empty_array, ptr %6
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr ptr, ptr %10, i32 0
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr %N, ptr %12, i32 0, i32 1
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr %V, ptr %14, i32 0, i32 2
  %16 = load double, ptr %15, align 8
  %17 = call ptr @mk(double 7.000000e+00)
  store i64 -1, ptr %3, align 8
  %18 = getelementptr i8, ptr %3, i64 8
  store [1 x ptr] zeroinitializer, ptr %18, align 8
  %19 = getelementptr ptr, ptr %18, i64 0
  store ptr %17, ptr %19, align 8
  store i64 -1, ptr %2, align 8
  %20 = getelementptr i8, ptr %2, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %20, align 8
  %21 = getelementptr { ptr, i64, i64 }, ptr %20, i32 0, i32 0
  store ptr %18, ptr %21, align 8
  %22 = getelementptr { ptr, i64, i64 }, ptr %20, i32 0, i32 1
  store i64 1, ptr %22, align 8
  %23 = getelementptr { ptr, i64, i64 }, ptr %20, i32 0, i32 2
  store i64 1, ptr %23, align 8
  store ptr %20, ptr %4, align 8
  call void @churn()
  %24 = fcmp oeq double %16, 1.000000e+00
  br i1 %24, label %25, label %38

25:                                               ; preds = %1
  %26 = load ptr, ptr %4, align 8
  %27 = icmp eq ptr %26, null
  %28 = select i1 %27, ptr @__ts_empty_array, ptr %26
  %29 = getelementptr { ptr, i64, i64 }, ptr %28, i32 0, i32 0
  %30 = load ptr, ptr %29, align 8
  %31 = getelementptr ptr, ptr %30, i32 0
  %32 = load ptr, ptr %31, align 8
  %33 = getelementptr %N, ptr %32, i32 0, i32 1
  %34 = load ptr, ptr %33, align 8
  %35 = getelementptr %V, ptr %34, i32 0, i32 2
  %36 = load double, ptr %35, align 8
  %37 = fcmp oeq double %36, 7.000000e+00
  br label %38

38:                                               ; preds = %25, %1
  %39 = phi i1 [ %37, %25 ], [ %24, %1 ]
  store i1 %39, ptr %5, align 1
  %40 = load i1, ptr %5, align 1
  ret i1 %40
}

; Function Attrs: mustprogress
define i1 @passedOn(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call i1 @assigned(ptr %4)
  call void @churn()
  br i1 %5, label %6, label %19

6:                                                ; preds = %1
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, null
  %9 = select i1 %8, ptr @__ts_empty_array, ptr %7
  %10 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 0
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr ptr, ptr %11, i32 0
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %N, ptr %13, i32 0, i32 1
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr %V, ptr %15, i32 0, i32 2
  %17 = load double, ptr %16, align 8
  %18 = fcmp oeq double %17, 1.000000e+00
  br label %19

19:                                               ; preds = %6, %1
  %20 = phi i1 [ %18, %6 ], [ %5, %1 ]
  br i1 %20, label %21, label %46

21:                                               ; preds = %19
  %22 = load ptr, ptr %2, align 8
  %23 = icmp eq ptr %22, null
  %24 = select i1 %23, ptr @__ts_empty_array, ptr %22
  %25 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 0
  %26 = load ptr, ptr %25, align 8
  %27 = getelementptr ptr, ptr %26, i32 0
  %28 = load ptr, ptr %27, align 8
  %29 = getelementptr %N, ptr %28, i32 0, i32 1
  %30 = load ptr, ptr %29, align 8
  %31 = getelementptr %V, ptr %30, i32 0, i32 1
  %32 = load ptr, ptr %31, align 8
  %33 = ptrtoint ptr %32 to i64
  %34 = icmp ne i64 %33, 0
  %35 = zext i1 %34 to i32
  %36 = and i32 %35, 1
  %37 = icmp ne i32 %36, 0
  br i1 %37, label %38, label %41

38:                                               ; preds = %21
  %39 = call i32 @strcmp(ptr %32, ptr getelementptr inbounds nuw (i8, ptr @s_634738200219259176, i64 8))
  %40 = icmp eq i32 %39, 0
  br label %43

41:                                               ; preds = %21
  %42 = icmp eq i64 %33, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_634738200219259176, i64 8) to i64)
  br label %43

43:                                               ; preds = %38, %41
  %44 = phi i1 [ %42, %41 ], [ %40, %38 ]
  br label %45

45:                                               ; preds = %43
  br label %46

46:                                               ; preds = %45, %19
  %47 = phi i1 [ %44, %45 ], [ %20, %19 ]
  store i1 %47, ptr %3, align 1
  %48 = load i1, ptr %3, align 1
  ret i1 %48
}

; Function Attrs: mustprogress
define double @numbers(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store i32 0, ptr %3, align 4
  br label %6

6:                                                ; preds = %9, %1
  %7 = load i32, ptr %3, align 4
  %8 = icmp slt i32 %7, 3
  br i1 %8, label %9, label %13

9:                                                ; preds = %6
  %10 = load ptr, ptr %4, align 8
  store ptr %10, ptr %2, align 8
  call void @churn()
  %11 = load i32, ptr %3, align 4
  %12 = add i32 %11, 1
  store i32 %12, ptr %3, align 4
  br label %6

13:                                               ; preds = %6
  %14 = load ptr, ptr %4, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr @__ts_empty_array, ptr %14
  %17 = getelementptr { ptr, i64, i64 }, ptr %16, i32 0, i32 0
  %18 = load ptr, ptr %17, align 8
  %19 = getelementptr double, ptr %18, i32 0
  %20 = load double, ptr %19, align 8
  %21 = load ptr, ptr %4, align 8
  %22 = icmp eq ptr %21, null
  %23 = select i1 %22, ptr @__ts_empty_array, ptr %21
  %24 = getelementptr { ptr, i64, i64 }, ptr %23, i32 0, i32 0
  %25 = load ptr, ptr %24, align 8
  %26 = getelementptr double, ptr %25, i32 1
  %27 = load double, ptr %26, align 8
  %28 = fadd double %20, %27
  %29 = load ptr, ptr %4, align 8
  %30 = icmp eq ptr %29, null
  %31 = select i1 %30, ptr @__ts_empty_array, ptr %29
  %32 = getelementptr { ptr, i64, i64 }, ptr %31, i32 0, i32 0
  %33 = load ptr, ptr %32, align 8
  %34 = getelementptr double, ptr %33, i32 2
  %35 = load double, ptr %34, align 8
  %36 = fadd double %28, %35
  store double %36, ptr %5, align 8
  %37 = load double, ptr %5, align 8
  ret double %37
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %2 = alloca <{ i64, [3 x double] }>, align 16
  %3 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %4 = alloca <{ i64, [1 x ptr] }>, align 16
  %5 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %6 = alloca <{ i64, [1 x ptr] }>, align 16
  %7 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %8 = alloca <{ i64, [2 x ptr] }>, align 16
  %9 = alloca i32, align 4
  store i32 0, ptr %9, align 4
  br label %10

10:                                               ; preds = %66, %0
  %11 = load i32, ptr %9, align 4
  %12 = icmp slt i32 %11, 3
  br i1 %12, label %13, label %69

13:                                               ; preds = %10
  %14 = call ptr @mk(double 1.000000e+00)
  %15 = call ptr @mk(double 2.000000e+00)
  store i64 -1, ptr %8, align 8
  %16 = getelementptr i8, ptr %8, i64 8
  store [2 x ptr] zeroinitializer, ptr %16, align 8
  %17 = getelementptr ptr, ptr %16, i64 0
  store ptr %14, ptr %17, align 8
  %18 = getelementptr ptr, ptr %16, i64 1
  store ptr %15, ptr %18, align 8
  store i64 -1, ptr %7, align 8
  %19 = getelementptr i8, ptr %7, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %19, align 8
  %20 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 0
  store ptr %16, ptr %20, align 8
  %21 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 1
  store i64 2, ptr %21, align 8
  %22 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 2
  store i64 2, ptr %22, align 8
  %23 = call i1 @aliasInLoop(ptr %19)
  br i1 %23, label %24, label %71

24:                                               ; preds = %13
  %25 = call ptr @mk(double 1.000000e+00)
  %26 = call ptr @mk(double 2.000000e+00)
  %27 = call ptr @GC_malloc(i64 24)
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr ptr, ptr %28, i64 0
  store ptr %25, ptr %29, align 8
  %30 = getelementptr ptr, ptr %28, i64 1
  store ptr %26, ptr %30, align 8
  %31 = call ptr @GC_malloc(i64 32)
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr { ptr, i64, i64 }, ptr %32, i32 0, i32 0
  store ptr %28, ptr %33, align 8
  %34 = getelementptr { ptr, i64, i64 }, ptr %32, i32 0, i32 1
  store i64 2, ptr %34, align 8
  %35 = getelementptr { ptr, i64, i64 }, ptr %32, i32 0, i32 2
  store i64 2, ptr %35, align 8
  %36 = call i1 @aliasInTry(ptr %32)
  br i1 %36, label %37, label %73

37:                                               ; preds = %24
  %38 = call ptr @mk(double 1.000000e+00)
  store i64 -1, ptr %6, align 8
  %39 = getelementptr i8, ptr %6, i64 8
  store [1 x ptr] zeroinitializer, ptr %39, align 8
  %40 = getelementptr ptr, ptr %39, i64 0
  store ptr %38, ptr %40, align 8
  store i64 -1, ptr %5, align 8
  %41 = getelementptr i8, ptr %5, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %41, align 8
  %42 = getelementptr { ptr, i64, i64 }, ptr %41, i32 0, i32 0
  store ptr %39, ptr %42, align 8
  %43 = getelementptr { ptr, i64, i64 }, ptr %41, i32 0, i32 1
  store i64 1, ptr %43, align 8
  %44 = getelementptr { ptr, i64, i64 }, ptr %41, i32 0, i32 2
  store i64 1, ptr %44, align 8
  %45 = call i1 @assigned(ptr %41)
  br i1 %45, label %46, label %75

46:                                               ; preds = %37
  %47 = call ptr @mk(double 1.000000e+00)
  store i64 -1, ptr %4, align 8
  %48 = getelementptr i8, ptr %4, i64 8
  store [1 x ptr] zeroinitializer, ptr %48, align 8
  %49 = getelementptr ptr, ptr %48, i64 0
  store ptr %47, ptr %49, align 8
  store i64 -1, ptr %3, align 8
  %50 = getelementptr i8, ptr %3, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %50, align 8
  %51 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 0
  store ptr %48, ptr %51, align 8
  %52 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 1
  store i64 1, ptr %52, align 8
  %53 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 2
  store i64 1, ptr %53, align 8
  %54 = call i1 @passedOn(ptr %50)
  br i1 %54, label %55, label %77

55:                                               ; preds = %46
  store i64 -1, ptr %2, align 8
  %56 = getelementptr i8, ptr %2, i64 8
  store [3 x double] zeroinitializer, ptr %56, align 8
  %57 = getelementptr double, ptr %56, i64 0
  store double 1.000000e+00, ptr %57, align 8
  %58 = getelementptr double, ptr %56, i64 1
  store double 2.000000e+00, ptr %58, align 8
  %59 = getelementptr double, ptr %56, i64 2
  store double 3.000000e+00, ptr %59, align 8
  store i64 -1, ptr %1, align 8
  %60 = getelementptr i8, ptr %1, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %60, align 8
  %61 = getelementptr { ptr, i64, i64 }, ptr %60, i32 0, i32 0
  store ptr %56, ptr %61, align 8
  %62 = getelementptr { ptr, i64, i64 }, ptr %60, i32 0, i32 1
  store i64 3, ptr %62, align 8
  %63 = getelementptr { ptr, i64, i64 }, ptr %60, i32 0, i32 2
  store i64 3, ptr %63, align 8
  %64 = call double @numbers(ptr %60)
  %65 = fcmp oeq double %64, 6.000000e+00
  br i1 %65, label %66, label %79

66:                                               ; preds = %55
  %67 = load i32, ptr %9, align 4
  %68 = add i32 %67, 1
  store i32 %68, ptr %9, align 4
  br label %10

69:                                               ; preds = %10
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

70:                                               ; preds = %79, %77, %75, %73, %71
  unreachable

71:                                               ; preds = %13
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2950142705226678939, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 75, ptr null)
  br label %70

73:                                               ; preds = %24
  %74 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16306909777058545598, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 76, ptr null)
  br label %70

75:                                               ; preds = %37
  %76 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11939981217937750340, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 77, ptr null)
  br label %70

77:                                               ; preds = %46
  %78 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7736067829917520921, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 78, ptr null)
  br label %70

79:                                               ; preds = %55
  %80 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_856888895201963235, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1484228504673423653, i64 8), i32 79, ptr null)
  br label %70
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

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

