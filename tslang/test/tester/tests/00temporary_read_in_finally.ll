; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, ptr, double }

$V..size = comdat any

@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
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
@f_7030385148271370996 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00temporary_read_in_finally.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_773604880766989148 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFread in the finally of a return\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_4789556428579878665 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcontinue through a finally\00", align 8
@m_16534269621397713645 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak through a finally\00", align 8
@m_780545326882473505 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo continues through one finally\00", align 8
@m_11387707198342658965 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa labelled continue through two finallys\00", align 8
@m_12071978550225446582 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreturn through a finally\00", align 8
@m_16972213606948871319 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa call in the finally\00", align 8
@m_11265703241562357215 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa continue through a finally that breaks\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 24, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V.get, ptr @V..rtti, ptr @V..size }
@a_0_6198119186829423739 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @ts.internal.eh_try_pad(i32)

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
define double @V.get(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %V, ptr %4, i32 0, i32 2
  %6 = load double, ptr %5, align 8
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define ptr @mv(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %V, ptr %5, i32 0, i32 0
  store ptr @V..vtbl, ptr %6, align 8
  %7 = load double, ptr %2, align 8
  call void @V.constructor(ptr %5, double %7)
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
  %14 = call ptr @GC_malloc(i64 32)
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 24, ptr null)
  br label %66

69:                                               ; preds = %41
  %70 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 24, ptr null)
  br label %66

71:                                               ; preds = %43
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 24, ptr null)
  br label %66
}

; Function Attrs: mustprogress
define double @viaContinue() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %31, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 4
  br i1 %6, label %7, label %34

7:                                                ; preds = %4
  %8 = load i32, ptr %1, align 4
  %9 = sitofp i32 %8 to double
  %10 = call ptr @mv(double %9)
  invoke void @churn()
          to label %11 unwind label %15

11:                                               ; preds = %7
  %12 = load i32, ptr %1, align 4
  %13 = srem i32 %12, 2
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %21, label %26

15:                                               ; preds = %7
  %16 = landingpad { ptr, i32 }
          cleanup
  call void @churn()
  %17 = load double, ptr %2, align 8
  %18 = getelementptr %V, ptr %10, i32 0, i32 0
  %19 = call double @V.get(ptr %10)
  %20 = fadd double %17, %19
  store double %20, ptr %2, align 8
  resume { ptr, i32 } %16

21:                                               ; preds = %11
  call void @churn()
  %22 = load double, ptr %2, align 8
  %23 = getelementptr %V, ptr %10, i32 0, i32 0
  %24 = call double @V.get(ptr %10)
  %25 = fadd double %22, %24
  store double %25, ptr %2, align 8
  br label %31

26:                                               ; preds = %11
  call void @churn()
  %27 = load double, ptr %2, align 8
  %28 = getelementptr %V, ptr %10, i32 0, i32 0
  %29 = call double @V.get(ptr %10)
  %30 = fadd double %27, %29
  store double %30, ptr %2, align 8
  br label %31

31:                                               ; preds = %21, %26
  %32 = load i32, ptr %1, align 4
  %33 = add i32 %32, 1
  store i32 %33, ptr %1, align 4
  br label %4

34:                                               ; preds = %4
  %35 = load double, ptr %2, align 8
  store double %35, ptr %3, align 8
  %36 = load double, ptr %3, align 8
  ret double %36
}

; Function Attrs: mustprogress
define double @viaBreak() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %19, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 4
  br i1 %6, label %7, label %26

7:                                                ; preds = %4
  %8 = load i32, ptr %1, align 4
  %9 = add i32 %8, 10
  %10 = sitofp i32 %9 to double
  %11 = call ptr @mv(double %10)
  %12 = load i32, ptr %1, align 4
  %13 = icmp eq i32 %12, 1
  br i1 %13, label %14, label %19

14:                                               ; preds = %7
  call void @churn()
  %15 = load double, ptr %2, align 8
  %16 = getelementptr %V, ptr %11, i32 0, i32 2
  %17 = load double, ptr %16, align 8
  %18 = fadd double %15, %17
  store double %18, ptr %2, align 8
  br label %26

19:                                               ; preds = %7
  call void @churn()
  %20 = load double, ptr %2, align 8
  %21 = getelementptr %V, ptr %11, i32 0, i32 2
  %22 = load double, ptr %21, align 8
  %23 = fadd double %20, %22
  store double %23, ptr %2, align 8
  %24 = load i32, ptr %1, align 4
  %25 = add i32 %24, 1
  store i32 %25, ptr %1, align 4
  br label %4

26:                                               ; preds = %14, %4
  %27 = load double, ptr %2, align 8
  store double %27, ptr %3, align 8
  %28 = load double, ptr %3, align 8
  ret double %28
}

; Function Attrs: mustprogress
define double @viaTwoJumps() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %28, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 6
  br i1 %6, label %7, label %31

7:                                                ; preds = %4
  %8 = load i32, ptr %1, align 4
  %9 = sitofp i32 %8 to double
  %10 = call ptr @mv(double %9)
  %11 = load i32, ptr %1, align 4
  %12 = srem i32 %11, 3
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %18, label %14

14:                                               ; preds = %7
  %15 = load i32, ptr %1, align 4
  %16 = srem i32 %15, 3
  %17 = icmp eq i32 %16, 1
  br i1 %17, label %18, label %23

18:                                               ; preds = %14, %7
  call void @churn()
  %19 = load double, ptr %2, align 8
  %20 = getelementptr %V, ptr %10, i32 0, i32 0
  %21 = call double @V.get(ptr %10)
  %22 = fadd double %19, %21
  store double %22, ptr %2, align 8
  br label %28

23:                                               ; preds = %14
  call void @churn()
  %24 = load double, ptr %2, align 8
  %25 = getelementptr %V, ptr %10, i32 0, i32 0
  %26 = call double @V.get(ptr %10)
  %27 = fadd double %24, %26
  store double %27, ptr %2, align 8
  br label %28

28:                                               ; preds = %18, %23
  %29 = load i32, ptr %1, align 4
  %30 = add i32 %29, 1
  store i32 %30, ptr %1, align 4
  br label %4

31:                                               ; preds = %4
  %32 = load double, ptr %2, align 8
  store double %32, ptr %3, align 8
  %33 = load double, ptr %3, align 8
  ret double %33
}

; Function Attrs: mustprogress
define double @viaLabelledContinue() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double 0.000000e+00, ptr %3, align 8
  store i32 0, ptr %2, align 4
  br label %5

5:                                                ; preds = %53, %0
  %6 = load i32, ptr %2, align 4
  %7 = icmp slt i32 %6, 3
  br i1 %7, label %8, label %56

8:                                                ; preds = %5
  %9 = load i32, ptr %2, align 4
  %10 = add i32 %9, 1
  %11 = sitofp i32 %10 to double
  %12 = call ptr @mv(double %11)
  store i32 0, ptr %1, align 4
  br label %13

13:                                               ; preds = %36, %8
  %14 = load i32, ptr %1, align 4
  %15 = icmp slt i32 %14, 3
  br i1 %15, label %16, label %52

16:                                               ; preds = %13
  %17 = load i32, ptr %1, align 4
  %18 = icmp eq i32 %17, 1
  br i1 %18, label %19, label %31

19:                                               ; preds = %16
  invoke void @churn()
          to label %20 unwind label %45

20:                                               ; preds = %19
  %21 = load double, ptr %3, align 8
  %22 = getelementptr %V, ptr %12, i32 0, i32 0
  %23 = invoke double @V.get(ptr %12)
          to label %24 unwind label %45

24:                                               ; preds = %20
  %25 = fadd double %21, %23
  store double %25, ptr %3, align 8
  call void @churn()
  %26 = load double, ptr %3, align 8
  %27 = getelementptr %V, ptr %12, i32 0, i32 2
  %28 = load double, ptr %27, align 8
  %29 = fmul double %28, 1.000000e+02
  %30 = fadd double %26, %29
  store double %30, ptr %3, align 8
  br label %53

31:                                               ; preds = %16
  invoke void @churn()
          to label %32 unwind label %45

32:                                               ; preds = %31
  %33 = load double, ptr %3, align 8
  %34 = getelementptr %V, ptr %12, i32 0, i32 0
  %35 = invoke double @V.get(ptr %12)
          to label %36 unwind label %45

36:                                               ; preds = %32
  %37 = fadd double %33, %35
  store double %37, ptr %3, align 8
  call void @churn()
  %38 = load double, ptr %3, align 8
  %39 = getelementptr %V, ptr %12, i32 0, i32 2
  %40 = load double, ptr %39, align 8
  %41 = fmul double %40, 1.000000e+02
  %42 = fadd double %38, %41
  store double %42, ptr %3, align 8
  %43 = load i32, ptr %1, align 4
  %44 = add i32 %43, 1
  store i32 %44, ptr %1, align 4
  br label %13

45:                                               ; preds = %20, %19, %32, %31
  %46 = landingpad { ptr, i32 }
          cleanup
  call void @churn()
  %47 = load double, ptr %3, align 8
  %48 = getelementptr %V, ptr %12, i32 0, i32 2
  %49 = load double, ptr %48, align 8
  %50 = fmul double %49, 1.000000e+02
  %51 = fadd double %47, %50
  store double %51, ptr %3, align 8
  resume { ptr, i32 } %46

52:                                               ; preds = %13
  br label %53

53:                                               ; preds = %24, %52
  %54 = load i32, ptr %2, align 4
  %55 = add i32 %54, 1
  store i32 %55, ptr %2, align 4
  br label %5

56:                                               ; preds = %5
  %57 = load double, ptr %3, align 8
  store double %57, ptr %4, align 8
  %58 = load double, ptr %4, align 8
  ret double %58
}

; Function Attrs: mustprogress
define double @viaReturn(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = call ptr @mv(double %4)
  %6 = load double, ptr %2, align 8
  %7 = fcmp ogt double %6, 0.000000e+00
  br i1 %7, label %8, label %57

8:                                                ; preds = %1
  %9 = load double, ptr %2, align 8
  store double %9, ptr %3, align 8
  call void @churn()
  %10 = getelementptr %V, ptr %5, i32 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = load double, ptr %2, align 8
  %13 = fcmp oeq double %11, %12
  br i1 %13, label %14, label %54

14:                                               ; preds = %8
  %15 = getelementptr %V, ptr %5, i32 0, i32 1
  %16 = load ptr, ptr %15, align 8
  %17 = load double, ptr %2, align 8
  %18 = call ptr @GC_malloc_atomic(i64 40)
  %19 = getelementptr i8, ptr %18, i64 8
  call void @__ts_number_to_string(ptr %19, double %17)
  %20 = icmp eq ptr %19, null
  %21 = select i1 %20, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %19
  %22 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %23 = add i64 1, %22
  %24 = call i64 @strlen(ptr %21)
  %25 = add i64 %23, %24
  %26 = add i64 %25, 8
  %27 = call ptr @GC_malloc(i64 %26)
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr i8, ptr %28, i64 0
  %30 = sub i64 %25, 0
  %31 = call ptr @__memcpy_chk(ptr %29, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %22, i64 %30)
  %32 = add i64 0, %22
  %33 = getelementptr i8, ptr %28, i64 %32
  %34 = sub i64 %25, %32
  %35 = call ptr @__memcpy_chk(ptr %33, ptr %21, i64 %24, i64 %34)
  %36 = add i64 %32, %24
  %37 = getelementptr i8, ptr %28, i64 %36
  store i8 0, ptr %37, align 1
  %38 = ptrtoint ptr %16 to i64
  %39 = ptrtoint ptr %28 to i64
  %40 = icmp ne i64 %38, 0
  %41 = icmp ne i64 %39, 0
  %42 = zext i1 %40 to i32
  %43 = zext i1 %41 to i32
  %44 = and i32 %42, %43
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %46, label %49

46:                                               ; preds = %14
  %47 = call i32 @strcmp(ptr %16, ptr %28)
  %48 = icmp eq i32 %47, 0
  br label %51

49:                                               ; preds = %14
  %50 = icmp eq i64 %38, %39
  br label %51

51:                                               ; preds = %46, %49
  %52 = phi i1 [ %50, %49 ], [ %48, %46 ]
  br label %53

53:                                               ; preds = %51
  br label %54

54:                                               ; preds = %53, %8
  %55 = phi i1 [ %52, %53 ], [ %13, %8 ]
  br i1 %55, label %56, label %108

56:                                               ; preds = %54
  br label %105

57:                                               ; preds = %1
  call void @churn()
  %58 = getelementptr %V, ptr %5, i32 0, i32 2
  %59 = load double, ptr %58, align 8
  %60 = load double, ptr %2, align 8
  %61 = fcmp oeq double %59, %60
  br i1 %61, label %62, label %102

62:                                               ; preds = %57
  %63 = getelementptr %V, ptr %5, i32 0, i32 1
  %64 = load ptr, ptr %63, align 8
  %65 = load double, ptr %2, align 8
  %66 = call ptr @GC_malloc_atomic(i64 40)
  %67 = getelementptr i8, ptr %66, i64 8
  call void @__ts_number_to_string(ptr %67, double %65)
  %68 = icmp eq ptr %67, null
  %69 = select i1 %68, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %67
  %70 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %71 = add i64 1, %70
  %72 = call i64 @strlen(ptr %69)
  %73 = add i64 %71, %72
  %74 = add i64 %73, 8
  %75 = call ptr @GC_malloc(i64 %74)
  %76 = getelementptr i8, ptr %75, i64 8
  %77 = getelementptr i8, ptr %76, i64 0
  %78 = sub i64 %73, 0
  %79 = call ptr @__memcpy_chk(ptr %77, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %70, i64 %78)
  %80 = add i64 0, %70
  %81 = getelementptr i8, ptr %76, i64 %80
  %82 = sub i64 %73, %80
  %83 = call ptr @__memcpy_chk(ptr %81, ptr %69, i64 %72, i64 %82)
  %84 = add i64 %80, %72
  %85 = getelementptr i8, ptr %76, i64 %84
  store i8 0, ptr %85, align 1
  %86 = ptrtoint ptr %64 to i64
  %87 = ptrtoint ptr %76 to i64
  %88 = icmp ne i64 %86, 0
  %89 = icmp ne i64 %87, 0
  %90 = zext i1 %88 to i32
  %91 = zext i1 %89 to i32
  %92 = and i32 %90, %91
  %93 = icmp ne i32 %92, 0
  br i1 %93, label %94, label %97

94:                                               ; preds = %62
  %95 = call i32 @strcmp(ptr %64, ptr %76)
  %96 = icmp eq i32 %95, 0
  br label %99

97:                                               ; preds = %62
  %98 = icmp eq i64 %86, %87
  br label %99

99:                                               ; preds = %94, %97
  %100 = phi i1 [ %98, %97 ], [ %96, %94 ]
  br label %101

101:                                              ; preds = %99
  br label %102

102:                                              ; preds = %101, %57
  %103 = phi i1 [ %100, %101 ], [ %61, %57 ]
  br i1 %103, label %104, label %110

104:                                              ; preds = %102
  store double -1.000000e+00, ptr %3, align 8
  br label %105

105:                                              ; preds = %56, %104
  %106 = load double, ptr %3, align 8
  ret double %106

107:                                              ; preds = %108, %110
  unreachable

108:                                              ; preds = %54
  %109 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_773604880766989148, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 116, ptr null)
  br label %107

110:                                              ; preds = %102
  %111 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_773604880766989148, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 116, ptr null)
  br label %107
}

; Function Attrs: mustprogress
define double @jumpInFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %28, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 4
  br i1 %6, label %7, label %31

7:                                                ; preds = %4
  %8 = load i32, ptr %1, align 4
  %9 = sitofp i32 %8 to double
  %10 = call ptr @mv(double %9)
  %11 = load i32, ptr %1, align 4
  %12 = srem i32 %11, 2
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %19

14:                                               ; preds = %7
  call void @churn()
  %15 = load i32, ptr %1, align 4
  %16 = icmp eq i32 %15, 2
  br i1 %16, label %17, label %18

17:                                               ; preds = %14
  br label %31

18:                                               ; preds = %14
  br label %28

19:                                               ; preds = %7
  call void @churn()
  %20 = load i32, ptr %1, align 4
  %21 = icmp eq i32 %20, 2
  br i1 %21, label %22, label %23

22:                                               ; preds = %19
  br label %31

23:                                               ; preds = %19
  %24 = load double, ptr %2, align 8
  %25 = getelementptr %V, ptr %10, i32 0, i32 2
  %26 = load double, ptr %25, align 8
  %27 = fadd double %24, %26
  store double %27, ptr %2, align 8
  br label %28

28:                                               ; preds = %18, %23
  %29 = load i32, ptr %1, align 4
  %30 = add i32 %29, 1
  store i32 %30, ptr %1, align 4
  br label %4

31:                                               ; preds = %17, %22, %4
  call void @churn()
  %32 = load double, ptr %2, align 8
  store double %32, ptr %3, align 8
  %33 = load double, ptr %3, align 8
  ret double %33
}

; Function Attrs: mustprogress
define double @callInFinally(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  call void @churn()
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %V, ptr %4, i32 0, i32 0
  %6 = call double @V.get(ptr %4)
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define double @viaCallInFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %23, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %26

7:                                                ; preds = %4
  %8 = load i32, ptr %1, align 4
  %9 = add i32 %8, 5
  %10 = sitofp i32 %9 to double
  %11 = call ptr @mv(double %10)
  %12 = getelementptr %V, ptr %11, i32 0, i32 2
  %13 = load double, ptr %12, align 8
  %14 = fcmp ogt double %13, 2.000000e+00
  br i1 %14, label %15, label %19

15:                                               ; preds = %7
  %16 = load double, ptr %2, align 8
  %17 = call double @callInFinally(ptr %11)
  %18 = fadd double %16, %17
  store double %18, ptr %2, align 8
  br label %23

19:                                               ; preds = %7
  %20 = load double, ptr %2, align 8
  %21 = call double @callInFinally(ptr %11)
  %22 = fadd double %20, %21
  store double %22, ptr %2, align 8
  br label %23

23:                                               ; preds = %15, %19
  %24 = load i32, ptr %1, align 4
  %25 = add i32 %24, 1
  store i32 %25, ptr %1, align 4
  br label %4

26:                                               ; preds = %4
  %27 = load double, ptr %2, align 8
  store double %27, ptr %3, align 8
  %28 = load double, ptr %3, align 8
  ret double %28
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call double @viaContinue()
  %2 = fcmp oeq double %1, 6.000000e+00
  br i1 %2, label %3, label %23

3:                                                ; preds = %0
  %4 = call double @viaBreak()
  %5 = fcmp oeq double %4, 2.100000e+01
  br i1 %5, label %6, label %25

6:                                                ; preds = %3
  %7 = call double @viaTwoJumps()
  %8 = fcmp oeq double %7, 1.500000e+01
  br i1 %8, label %9, label %27

9:                                                ; preds = %6
  %10 = call double @viaLabelledContinue()
  %11 = fcmp oeq double %10, 1.212000e+03
  br i1 %11, label %12, label %29

12:                                               ; preds = %9
  %13 = call double @viaReturn(double 3.000000e+00)
  %14 = fcmp oeq double %13, 3.000000e+00
  br i1 %14, label %15, label %31

15:                                               ; preds = %12
  %16 = call double @viaCallInFinally()
  %17 = fcmp oeq double %16, 1.800000e+01
  br i1 %17, label %18, label %33

18:                                               ; preds = %15
  %19 = call double @jumpInFinally()
  %20 = fcmp oeq double %19, 1.000000e+00
  br i1 %20, label %21, label %35

21:                                               ; preds = %18
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

22:                                               ; preds = %35, %33, %31, %29, %27, %25, %23
  unreachable

23:                                               ; preds = %0
  %24 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4789556428579878665, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 169, ptr null)
  br label %22

25:                                               ; preds = %3
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16534269621397713645, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 170, ptr null)
  br label %22

27:                                               ; preds = %6
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_780545326882473505, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 171, ptr null)
  br label %22

29:                                               ; preds = %9
  %30 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11387707198342658965, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 172, ptr null)
  br label %22

31:                                               ; preds = %12
  %32 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12071978550225446582, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 173, ptr null)
  br label %22

33:                                               ; preds = %15
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16972213606948871319, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 174, ptr null)
  br label %22

35:                                               ; preds = %18
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11265703241562357215, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7030385148271370996, i64 8), i32 175, ptr null)
  br label %22
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

