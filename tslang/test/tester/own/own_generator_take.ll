; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, double }
%.f_made..objL19C1FH2070732902 = type { i32, i32, ptr, ptr }
%.f_rows..objL23C1FH2070732902 = type { i32, i32, ptr, ptr }

$C..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_12638206991764949794 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs\00", align 8
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
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_5745265928196723405 = internal constant [106 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_generator_take.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_637536457312488496 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs2\00", align 8
@s_637539755847373129 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs1\00", align 8
@m_626102635892993007 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFok\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 16, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C..rtti, ptr @C..size }
@a_0_2926441800300339944 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

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
  %2 = call ptr @GC_malloc(i64 24)
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
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @label(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = call ptr @GC_malloc_atomic(i64 40)
  %6 = getelementptr i8, ptr %5, i64 8
  call void @__ts_number_to_string(ptr %6, double %4)
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %9 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8))
  %10 = add i64 1, %9
  %11 = call i64 @strlen(ptr %8)
  %12 = add i64 %10, %11
  %13 = add i64 %12, 8
  %14 = call ptr @GC_malloc(i64 %13)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i8, ptr %15, i64 0
  %17 = sub i64 %12, 0
  %18 = call ptr @__memcpy_chk(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8), i64 %9, i64 %17)
  %19 = add i64 0, %9
  %20 = getelementptr i8, ptr %15, i64 %19
  %21 = sub i64 %12, %19
  %22 = call ptr @__memcpy_chk(ptr %20, ptr %8, i64 %11, i64 %21)
  %23 = add i64 %19, %11
  %24 = getelementptr i8, ptr %15, i64 %23
  store i8 0, ptr %24, align 1
  store ptr %15, ptr %3, align 8
  %25 = load ptr, ptr %3, align 8
  ret ptr %25
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

10:                                               ; preds = %61, %0
  %11 = load i32, ptr %1, align 4
  %12 = icmp slt i32 %11, 64
  br i1 %12, label %13, label %65

13:                                               ; preds = %10
  %14 = call ptr @GC_malloc(i64 24)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %C, ptr %15, i32 0, i32 0
  store ptr @C..vtbl, ptr %16, align 8
  call void @C.constructor(ptr %15, double 9.990000e+02)
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 16, ptr null)
  br label %66

69:                                               ; preds = %41
  %70 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 16, ptr null)
  br label %66

71:                                               ; preds = %43
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 16, ptr null)
  br label %66
}

; Function Attrs: mustprogress
define { ptr, i1 } @.f_made..f_made..objL19C1FH2070732902.next(ptr %0) #2 {
  %2 = alloca { ptr, i1 }, align 8
  %3 = alloca { ptr, i1 }, align 8
  %4 = alloca { ptr, i1 }, align 8
  %5 = alloca ptr, align 8
  %6 = alloca { ptr, i1 }, align 8
  store ptr %0, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = getelementptr %.f_made..objL19C1FH2070732902, ptr %7, i32 0, i32 3
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr { i32 }, ptr %9, i32 0, i32 0
  %11 = load i32, ptr %10, align 4
  %12 = load ptr, ptr %5, align 8
  %13 = getelementptr %.f_made..objL19C1FH2070732902, ptr %12, i32 0, i32 0
  %14 = load i32, ptr %13, align 4
  %15 = icmp slt i32 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %1
  store { ptr, i1 } { ptr null, i1 true }, ptr %4, align 8
  %17 = getelementptr { ptr, i1 }, ptr %4, i32 0, i32 0
  store ptr null, ptr %17, align 8
  %18 = load { ptr, i1 }, ptr %4, align 8
  store { ptr, i1 } %18, ptr %6, align 8
  br label %45

19:                                               ; preds = %1
  %20 = load ptr, ptr %5, align 8
  %21 = getelementptr %.f_made..objL19C1FH2070732902, ptr %20, i32 0, i32 0
  %22 = load i32, ptr %21, align 4
  %23 = load ptr, ptr %5, align 8
  %24 = getelementptr %.f_made..objL19C1FH2070732902, ptr %23, i32 0, i32 1
  switch i32 %22, label %45 [
    i32 0, label %25
    i32 1, label %39
  ]

25:                                               ; preds = %19
  store i32 0, ptr %24, align 4
  br label %26

26:                                               ; preds = %25, %39
  %27 = load i32, ptr %24, align 4
  %28 = icmp slt i32 %27, %11
  br i1 %28, label %29, label %42

29:                                               ; preds = %26
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr %.f_made..objL19C1FH2070732902, ptr %30, i32 0, i32 0
  store i32 1, ptr %31, align 4
  %32 = call ptr @GC_malloc(i64 24)
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr %C, ptr %33, i32 0, i32 0
  store ptr @C..vtbl, ptr %34, align 8
  %35 = load i32, ptr %24, align 4
  %36 = sitofp i32 %35 to double
  call void @C.constructor(ptr %33, double %36)
  store { ptr, i1 } zeroinitializer, ptr %3, align 8
  %37 = getelementptr { ptr, i1 }, ptr %3, i32 0, i32 0
  store ptr %33, ptr %37, align 8
  %38 = load { ptr, i1 }, ptr %3, align 8
  store { ptr, i1 } %38, ptr %6, align 8
  br label %45

39:                                               ; preds = %19
  %40 = load i32, ptr %24, align 4
  %41 = add i32 %40, 1
  store i32 %41, ptr %24, align 4
  br label %26

42:                                               ; preds = %26
  store { ptr, i1 } { ptr null, i1 true }, ptr %2, align 8
  %43 = getelementptr { ptr, i1 }, ptr %2, i32 0, i32 0
  store ptr null, ptr %43, align 8
  %44 = load { ptr, i1 }, ptr %2, align 8
  store { ptr, i1 } %44, ptr %6, align 8
  br label %45

45:                                               ; preds = %16, %29, %42, %19
  %46 = load { ptr, i1 }, ptr %6, align 8
  ret { ptr, i1 } %46
}

; Function Attrs: mustprogress
define ptr @made(i32 %0) #2 {
  %2 = alloca { i32, i32, ptr, ptr }, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 12)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr { i32 }, ptr %6, i32 0, i32 0
  %8 = load i32, ptr %3, align 4
  store i32 %8, ptr %7, align 4
  store { i32, i32, ptr, ptr } { i32 0, i32 0, ptr @.f_made..f_made..objL19C1FH2070732902.next, ptr null }, ptr %2, align 8
  %9 = getelementptr { i32, i32, ptr, ptr }, ptr %2, i32 0, i32 3
  store ptr %6, ptr %9, align 8
  %10 = load { i32, i32, ptr, ptr }, ptr %2, align 8
  %11 = call ptr @GC_malloc(i64 32)
  %12 = getelementptr i8, ptr %11, i64 8
  store { i32, i32, ptr, ptr } %10, ptr %12, align 8
  store ptr %12, ptr %4, align 8
  %13 = load ptr, ptr %4, align 8
  ret ptr %13
}

; Function Attrs: mustprogress
define { ptr, i1 } @.f_rows..f_rows..objL23C1FH2070732902.next(ptr %0) #2 {
  %2 = alloca { ptr, i1 }, align 8
  %3 = alloca { ptr, i1 }, align 8
  %4 = alloca { ptr, i1 }, align 8
  %5 = alloca ptr, align 8
  %6 = alloca { ptr, i1 }, align 8
  store ptr %0, ptr %5, align 8
  %7 = load ptr, ptr %5, align 8
  %8 = getelementptr %.f_rows..objL23C1FH2070732902, ptr %7, i32 0, i32 3
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr { i32 }, ptr %9, i32 0, i32 0
  %11 = load i32, ptr %10, align 4
  %12 = load ptr, ptr %5, align 8
  %13 = getelementptr %.f_rows..objL23C1FH2070732902, ptr %12, i32 0, i32 0
  %14 = load i32, ptr %13, align 4
  %15 = icmp slt i32 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %1
  store { ptr, i1 } { ptr null, i1 true }, ptr %4, align 8
  %17 = getelementptr { ptr, i1 }, ptr %4, i32 0, i32 0
  store ptr null, ptr %17, align 8
  %18 = load { ptr, i1 }, ptr %4, align 8
  store { ptr, i1 } %18, ptr %6, align 8
  br label %56

19:                                               ; preds = %1
  %20 = load ptr, ptr %5, align 8
  %21 = getelementptr %.f_rows..objL23C1FH2070732902, ptr %20, i32 0, i32 0
  %22 = load i32, ptr %21, align 4
  %23 = load ptr, ptr %5, align 8
  %24 = getelementptr %.f_rows..objL23C1FH2070732902, ptr %23, i32 0, i32 1
  switch i32 %22, label %56 [
    i32 0, label %25
    i32 1, label %50
  ]

25:                                               ; preds = %19
  store i32 0, ptr %24, align 4
  br label %26

26:                                               ; preds = %25, %50
  %27 = load i32, ptr %24, align 4
  %28 = icmp slt i32 %27, %11
  br i1 %28, label %29, label %53

29:                                               ; preds = %26
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr %.f_rows..objL23C1FH2070732902, ptr %30, i32 0, i32 0
  store i32 1, ptr %31, align 4
  %32 = load i32, ptr %24, align 4
  %33 = sitofp i32 %32 to double
  %34 = call ptr @label(double %33)
  %35 = load i32, ptr %24, align 4
  %36 = add i32 %35, 1
  %37 = sitofp i32 %36 to double
  %38 = call ptr @label(double %37)
  %39 = call ptr @GC_malloc(i64 24)
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = getelementptr ptr, ptr %40, i64 0
  store ptr %34, ptr %41, align 8
  %42 = getelementptr ptr, ptr %40, i64 1
  store ptr %38, ptr %42, align 8
  %43 = call ptr @GC_malloc(i64 32)
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 0
  store ptr %40, ptr %45, align 8
  %46 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 1
  store i64 2, ptr %46, align 8
  %47 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 2
  store i64 2, ptr %47, align 8
  store { ptr, i1 } zeroinitializer, ptr %3, align 8
  %48 = getelementptr { ptr, i1 }, ptr %3, i32 0, i32 0
  store ptr %44, ptr %48, align 8
  %49 = load { ptr, i1 }, ptr %3, align 8
  store { ptr, i1 } %49, ptr %6, align 8
  br label %56

50:                                               ; preds = %19
  %51 = load i32, ptr %24, align 4
  %52 = add i32 %51, 1
  store i32 %52, ptr %24, align 4
  br label %26

53:                                               ; preds = %26
  store { ptr, i1 } { ptr null, i1 true }, ptr %2, align 8
  %54 = getelementptr { ptr, i1 }, ptr %2, i32 0, i32 0
  store ptr null, ptr %54, align 8
  %55 = load { ptr, i1 }, ptr %2, align 8
  store { ptr, i1 } %55, ptr %6, align 8
  br label %56

56:                                               ; preds = %16, %29, %53, %19
  %57 = load { ptr, i1 }, ptr %6, align 8
  ret { ptr, i1 } %57
}

; Function Attrs: mustprogress
define ptr @rows(i32 %0) #2 {
  %2 = alloca { i32, i32, ptr, ptr }, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 12)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr { i32 }, ptr %6, i32 0, i32 0
  %8 = load i32, ptr %3, align 4
  store i32 %8, ptr %7, align 4
  store { i32, i32, ptr, ptr } { i32 0, i32 0, ptr @.f_rows..f_rows..objL23C1FH2070732902.next, ptr null }, ptr %2, align 8
  %9 = getelementptr { i32, i32, ptr, ptr }, ptr %2, i32 0, i32 3
  store ptr %6, ptr %9, align 8
  %10 = load { i32, i32, ptr, ptr }, ptr %2, align 8
  %11 = call ptr @GC_malloc(i64 32)
  %12 = getelementptr i8, ptr %11, i64 8
  store { i32, i32, ptr, ptr } %10, ptr %12, align 8
  store ptr %12, ptr %4, align 8
  %13 = load ptr, ptr %4, align 8
  ret ptr %13
}

; Function Attrs: mustprogress
define ptr @pushed(i32 %0) #2 {
  %2 = alloca { ptr, i1 }, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store ptr null, ptr %6, align 8
  store i32 %0, ptr %5, align 4
  %7 = call ptr @GC_malloc(i64 8)
  %8 = getelementptr i8, ptr %7, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %8, ptr @a_0_2926441800300339944, i64 0, i1 false)
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  store ptr %8, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  store i64 0, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 2
  store i64 0, ptr %13, align 8
  store ptr %10, ptr %4, align 8
  %14 = load i32, ptr %5, align 4
  %15 = call ptr @made(i32 %14)
  store ptr %15, ptr %3, align 8
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr { i32, i32, ptr, ptr }, ptr %16, i32 0, i32 2
  %18 = insertvalue { ptr, ptr } zeroinitializer, ptr %17, 0
  %19 = insertvalue { ptr, ptr } %18, ptr %16, 1
  %20 = extractvalue { ptr, ptr } %19, 1
  %21 = extractvalue { ptr, ptr } %19, 0
  %22 = load ptr, ptr %21, align 8
  %23 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %22, 0
  %24 = insertvalue { ptr, ptr, ptr } %23, ptr %20, 1
  %25 = insertvalue { ptr, ptr, ptr } %24, ptr null, 2
  %26 = extractvalue { ptr, ptr, ptr } %25, 1
  %27 = extractvalue { ptr, ptr, ptr } %25, 0
  %28 = call { ptr, i1 } %27(ptr %26)
  store { ptr, i1 } %28, ptr %2, align 8
  br label %29

29:                                               ; preds = %80, %1
  %30 = getelementptr { ptr, i1 }, ptr %2, i32 0, i32 1
  %31 = load i1, ptr %30, align 1
  %32 = xor i1 %31, true
  br i1 %32, label %33, label %95

33:                                               ; preds = %29
  %34 = getelementptr { ptr, i1 }, ptr %2, i32 0, i32 0
  %35 = load ptr, ptr %34, align 8
  %36 = load ptr, ptr %4, align 8
  %37 = icmp eq ptr %36, null
  br i1 %37, label %38, label %41

38:                                               ; preds = %33
  %39 = call ptr @GC_malloc(i64 32)
  %40 = getelementptr i8, ptr %39, i64 8
  store ptr %40, ptr %4, align 8
  br label %42

41:                                               ; preds = %33
  br label %42

42:                                               ; preds = %38, %41
  %43 = phi ptr [ %36, %41 ], [ %40, %38 ]
  br label %44

44:                                               ; preds = %42
  %45 = getelementptr { ptr, i64, i64 }, ptr %43, i32 0, i32 1
  %46 = load i64, ptr %45, align 8
  %47 = add i64 %46, 1
  %48 = getelementptr { ptr, i64, i64 }, ptr %43, i32 0, i32 0
  %49 = getelementptr { ptr, i64, i64 }, ptr %43, i32 0, i32 2
  %50 = load ptr, ptr %48, align 8
  %51 = load i64, ptr %49, align 8
  %52 = icmp ugt i64 %47, %51
  br i1 %52, label %53, label %77

53:                                               ; preds = %44
  %54 = mul i64 %51, 2
  %55 = icmp uge i64 %54, %47
  %56 = select i1 %55, i64 %54, i64 %47
  %57 = icmp ult i64 %56, 4
  %58 = select i1 %57, i64 4, i64 %56
  %59 = icmp ule i64 %47, 4294967295
  br i1 %59, label %60, label %99

60:                                               ; preds = %53
  %61 = icmp ule i64 %58, 2305843009213693951
  br i1 %61, label %62, label %101

62:                                               ; preds = %60
  %63 = mul i64 8, %58
  %64 = add i64 %63, 8
  %65 = icmp eq ptr %50, null
  %66 = getelementptr i8, ptr %50, i64 -8
  %67 = select i1 %65, ptr null, ptr %66
  %68 = call ptr @GC_realloc(ptr %67, i64 %64)
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = icmp ne ptr %69, null
  %71 = icmp eq i64 %63, 0
  %72 = or i1 %70, %71
  br i1 %72, label %73, label %103

73:                                               ; preds = %62
  %74 = mul i64 8, %51
  %75 = getelementptr i8, ptr %69, i64 %74
  %76 = sub i64 %63, %74
  call void @llvm.memset.p0.i64(ptr %75, i8 0, i64 %76, i1 false)
  store ptr %69, ptr %48, align 8
  store i64 %58, ptr %49, align 8
  br label %78

77:                                               ; preds = %44
  br label %78

78:                                               ; preds = %73, %77
  %79 = phi ptr [ %50, %77 ], [ %69, %73 ]
  br label %80

80:                                               ; preds = %78
  %81 = getelementptr ptr, ptr %79, i64 %46
  store ptr %35, ptr %81, align 8
  store i64 %47, ptr %45, align 8
  %82 = load ptr, ptr %3, align 8
  %83 = getelementptr { i32, i32, ptr, ptr }, ptr %82, i32 0, i32 2
  %84 = insertvalue { ptr, ptr } zeroinitializer, ptr %83, 0
  %85 = insertvalue { ptr, ptr } %84, ptr %82, 1
  %86 = extractvalue { ptr, ptr } %85, 1
  %87 = extractvalue { ptr, ptr } %85, 0
  %88 = load ptr, ptr %87, align 8
  %89 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %88, 0
  %90 = insertvalue { ptr, ptr, ptr } %89, ptr %86, 1
  %91 = insertvalue { ptr, ptr, ptr } %90, ptr null, 2
  %92 = extractvalue { ptr, ptr, ptr } %91, 1
  %93 = extractvalue { ptr, ptr, ptr } %91, 0
  %94 = call { ptr, i1 } %93(ptr %92)
  store { ptr, i1 } %94, ptr %2, align 8
  br label %29

95:                                               ; preds = %29
  %96 = load ptr, ptr %4, align 8
  store ptr %96, ptr %6, align 8
  %97 = load ptr, ptr %6, align 8
  ret ptr %97

98:                                               ; preds = %103, %101, %99
  unreachable

99:                                               ; preds = %53
  %100 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 29, ptr null)
  br label %98

101:                                              ; preds = %60
  %102 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 29, ptr null)
  br label %98

103:                                              ; preds = %62
  %104 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 29, ptr null)
  br label %98
}

; Function Attrs: mustprogress
define ptr @evens(i32 %0) #2 {
  %2 = alloca { ptr, i1 }, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  store ptr null, ptr %6, align 8
  store i32 %0, ptr %5, align 4
  %7 = call ptr @GC_malloc(i64 8)
  %8 = getelementptr i8, ptr %7, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %8, ptr @a_0_2926441800300339944, i64 0, i1 false)
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  store ptr %8, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  store i64 0, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 2
  store i64 0, ptr %13, align 8
  store ptr %10, ptr %4, align 8
  %14 = load i32, ptr %5, align 4
  %15 = call ptr @made(i32 %14)
  store ptr %15, ptr %3, align 8
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr { i32, i32, ptr, ptr }, ptr %16, i32 0, i32 2
  %18 = insertvalue { ptr, ptr } zeroinitializer, ptr %17, 0
  %19 = insertvalue { ptr, ptr } %18, ptr %16, 1
  %20 = extractvalue { ptr, ptr } %19, 1
  %21 = extractvalue { ptr, ptr } %19, 0
  %22 = load ptr, ptr %21, align 8
  %23 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %22, 0
  %24 = insertvalue { ptr, ptr, ptr } %23, ptr %20, 1
  %25 = insertvalue { ptr, ptr, ptr } %24, ptr null, 2
  %26 = extractvalue { ptr, ptr, ptr } %25, 1
  %27 = extractvalue { ptr, ptr, ptr } %25, 0
  %28 = call { ptr, i1 } %27(ptr %26)
  store { ptr, i1 } %28, ptr %2, align 8
  br label %29

29:                                               ; preds = %87, %1
  %30 = getelementptr { ptr, i1 }, ptr %2, i32 0, i32 1
  %31 = load i1, ptr %30, align 1
  %32 = xor i1 %31, true
  br i1 %32, label %33, label %101

33:                                               ; preds = %29
  %34 = getelementptr { ptr, i1 }, ptr %2, i32 0, i32 0
  %35 = load ptr, ptr %34, align 8
  %36 = getelementptr %C, ptr %35, i32 0, i32 1
  %37 = load double, ptr %36, align 8
  %38 = frem double %37, 2.000000e+00
  %39 = fcmp oeq double %38, 0.000000e+00
  br i1 %39, label %40, label %87

40:                                               ; preds = %33
  %41 = load ptr, ptr %4, align 8
  %42 = icmp eq ptr %41, null
  br i1 %42, label %43, label %46

43:                                               ; preds = %40
  %44 = call ptr @GC_malloc(i64 32)
  %45 = getelementptr i8, ptr %44, i64 8
  store ptr %45, ptr %4, align 8
  br label %47

46:                                               ; preds = %40
  br label %47

47:                                               ; preds = %43, %46
  %48 = phi ptr [ %41, %46 ], [ %45, %43 ]
  br label %49

49:                                               ; preds = %47
  %50 = getelementptr { ptr, i64, i64 }, ptr %48, i32 0, i32 1
  %51 = load i64, ptr %50, align 8
  %52 = add i64 %51, 1
  %53 = getelementptr { ptr, i64, i64 }, ptr %48, i32 0, i32 0
  %54 = getelementptr { ptr, i64, i64 }, ptr %48, i32 0, i32 2
  %55 = load ptr, ptr %53, align 8
  %56 = load i64, ptr %54, align 8
  %57 = icmp ugt i64 %52, %56
  br i1 %57, label %58, label %82

58:                                               ; preds = %49
  %59 = mul i64 %56, 2
  %60 = icmp uge i64 %59, %52
  %61 = select i1 %60, i64 %59, i64 %52
  %62 = icmp ult i64 %61, 4
  %63 = select i1 %62, i64 4, i64 %61
  %64 = icmp ule i64 %52, 4294967295
  br i1 %64, label %65, label %105

65:                                               ; preds = %58
  %66 = icmp ule i64 %63, 2305843009213693951
  br i1 %66, label %67, label %107

67:                                               ; preds = %65
  %68 = mul i64 8, %63
  %69 = add i64 %68, 8
  %70 = icmp eq ptr %55, null
  %71 = getelementptr i8, ptr %55, i64 -8
  %72 = select i1 %70, ptr null, ptr %71
  %73 = call ptr @GC_realloc(ptr %72, i64 %69)
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = icmp ne ptr %74, null
  %76 = icmp eq i64 %68, 0
  %77 = or i1 %75, %76
  br i1 %77, label %78, label %109

78:                                               ; preds = %67
  %79 = mul i64 8, %56
  %80 = getelementptr i8, ptr %74, i64 %79
  %81 = sub i64 %68, %79
  call void @llvm.memset.p0.i64(ptr %80, i8 0, i64 %81, i1 false)
  store ptr %74, ptr %53, align 8
  store i64 %63, ptr %54, align 8
  br label %83

82:                                               ; preds = %49
  br label %83

83:                                               ; preds = %78, %82
  %84 = phi ptr [ %55, %82 ], [ %74, %78 ]
  br label %85

85:                                               ; preds = %83
  %86 = getelementptr ptr, ptr %84, i64 %51
  store ptr %35, ptr %86, align 8
  store i64 %52, ptr %50, align 8
  br label %87

87:                                               ; preds = %85, %33
  %88 = load ptr, ptr %3, align 8
  %89 = getelementptr { i32, i32, ptr, ptr }, ptr %88, i32 0, i32 2
  %90 = insertvalue { ptr, ptr } zeroinitializer, ptr %89, 0
  %91 = insertvalue { ptr, ptr } %90, ptr %88, 1
  %92 = extractvalue { ptr, ptr } %91, 1
  %93 = extractvalue { ptr, ptr } %91, 0
  %94 = load ptr, ptr %93, align 8
  %95 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %94, 0
  %96 = insertvalue { ptr, ptr, ptr } %95, ptr %92, 1
  %97 = insertvalue { ptr, ptr, ptr } %96, ptr null, 2
  %98 = extractvalue { ptr, ptr, ptr } %97, 1
  %99 = extractvalue { ptr, ptr, ptr } %97, 0
  %100 = call { ptr, i1 } %99(ptr %98)
  store { ptr, i1 } %100, ptr %2, align 8
  br label %29

101:                                              ; preds = %29
  %102 = load ptr, ptr %4, align 8
  store ptr %102, ptr %6, align 8
  %103 = load ptr, ptr %6, align 8
  ret ptr %103

104:                                              ; preds = %109, %107, %105
  unreachable

105:                                              ; preds = %58
  %106 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 36, ptr null)
  br label %104

107:                                              ; preds = %65
  %108 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 36, ptr null)
  br label %104

109:                                              ; preds = %67
  %110 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 36, ptr null)
  br label %104
}

; Function Attrs: mustprogress
define ptr @spread(i32 %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, i1 }, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  store ptr null, ptr %7, align 8
  store i32 %0, ptr %6, align 4
  %8 = load i32, ptr %6, align 4
  %9 = call ptr @made(i32 %8)
  %10 = call ptr @GC_malloc(i64 8)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = call ptr @GC_malloc(i64 32)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 0
  store ptr %11, ptr %14, align 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 1
  store i64 0, ptr %15, align 8
  %16 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 2
  store i64 0, ptr %16, align 8
  store ptr %13, ptr %5, align 8
  store ptr %9, ptr %4, align 8
  %17 = load ptr, ptr %4, align 8
  %18 = getelementptr { i32, i32, ptr, ptr }, ptr %17, i32 0, i32 2
  %19 = insertvalue { ptr, ptr } zeroinitializer, ptr %18, 0
  %20 = insertvalue { ptr, ptr } %19, ptr %17, 1
  %21 = extractvalue { ptr, ptr } %20, 1
  %22 = extractvalue { ptr, ptr } %20, 0
  %23 = load ptr, ptr %22, align 8
  %24 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %23, 0
  %25 = insertvalue { ptr, ptr, ptr } %24, ptr %21, 1
  %26 = insertvalue { ptr, ptr, ptr } %25, ptr null, 2
  %27 = extractvalue { ptr, ptr, ptr } %26, 1
  %28 = extractvalue { ptr, ptr, ptr } %26, 0
  %29 = call { ptr, i1 } %28(ptr %27)
  store { ptr, i1 } %29, ptr %3, align 8
  br label %30

30:                                               ; preds = %81, %1
  %31 = getelementptr { ptr, i1 }, ptr %3, i32 0, i32 1
  %32 = load i1, ptr %31, align 1
  %33 = xor i1 %32, true
  br i1 %33, label %34, label %96

34:                                               ; preds = %30
  %35 = getelementptr { ptr, i1 }, ptr %3, i32 0, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = load ptr, ptr %5, align 8
  %38 = icmp eq ptr %37, null
  br i1 %38, label %39, label %42

39:                                               ; preds = %34
  %40 = call ptr @GC_malloc(i64 32)
  %41 = getelementptr i8, ptr %40, i64 8
  store ptr %41, ptr %5, align 8
  br label %43

42:                                               ; preds = %34
  br label %43

43:                                               ; preds = %39, %42
  %44 = phi ptr [ %37, %42 ], [ %41, %39 ]
  br label %45

45:                                               ; preds = %43
  %46 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 1
  %47 = load i64, ptr %46, align 8
  %48 = add i64 %47, 1
  %49 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 0
  %50 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 2
  %51 = load ptr, ptr %49, align 8
  %52 = load i64, ptr %50, align 8
  %53 = icmp ugt i64 %48, %52
  br i1 %53, label %54, label %78

54:                                               ; preds = %45
  %55 = mul i64 %52, 2
  %56 = icmp uge i64 %55, %48
  %57 = select i1 %56, i64 %55, i64 %48
  %58 = icmp ult i64 %57, 4
  %59 = select i1 %58, i64 4, i64 %57
  %60 = icmp ule i64 %48, 4294967295
  br i1 %60, label %61, label %101

61:                                               ; preds = %54
  %62 = icmp ule i64 %59, 2305843009213693951
  br i1 %62, label %63, label %103

63:                                               ; preds = %61
  %64 = mul i64 8, %59
  %65 = add i64 %64, 8
  %66 = icmp eq ptr %51, null
  %67 = getelementptr i8, ptr %51, i64 -8
  %68 = select i1 %66, ptr null, ptr %67
  %69 = call ptr @GC_realloc(ptr %68, i64 %65)
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = icmp ne ptr %70, null
  %72 = icmp eq i64 %64, 0
  %73 = or i1 %71, %72
  br i1 %73, label %74, label %105

74:                                               ; preds = %63
  %75 = mul i64 8, %52
  %76 = getelementptr i8, ptr %70, i64 %75
  %77 = sub i64 %64, %75
  call void @llvm.memset.p0.i64(ptr %76, i8 0, i64 %77, i1 false)
  store ptr %70, ptr %49, align 8
  store i64 %59, ptr %50, align 8
  br label %79

78:                                               ; preds = %45
  br label %79

79:                                               ; preds = %74, %78
  %80 = phi ptr [ %51, %78 ], [ %70, %74 ]
  br label %81

81:                                               ; preds = %79
  %82 = getelementptr ptr, ptr %80, i64 %47
  store ptr %36, ptr %82, align 8
  store i64 %48, ptr %46, align 8
  %83 = load ptr, ptr %4, align 8
  %84 = getelementptr { i32, i32, ptr, ptr }, ptr %83, i32 0, i32 2
  %85 = insertvalue { ptr, ptr } zeroinitializer, ptr %84, 0
  %86 = insertvalue { ptr, ptr } %85, ptr %83, 1
  %87 = extractvalue { ptr, ptr } %86, 1
  %88 = extractvalue { ptr, ptr } %86, 0
  %89 = load ptr, ptr %88, align 8
  %90 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %89, 0
  %91 = insertvalue { ptr, ptr, ptr } %90, ptr %87, 1
  %92 = insertvalue { ptr, ptr, ptr } %91, ptr null, 2
  %93 = extractvalue { ptr, ptr, ptr } %92, 1
  %94 = extractvalue { ptr, ptr, ptr } %92, 0
  %95 = call { ptr, i1 } %94(ptr %93)
  store { ptr, i1 } %95, ptr %3, align 8
  br label %30

96:                                               ; preds = %30
  %97 = load ptr, ptr %5, align 8
  store ptr %97, ptr %2, align 8
  %98 = load ptr, ptr %2, align 8
  store ptr %98, ptr %7, align 8
  %99 = load ptr, ptr %7, align 8
  ret ptr %99

100:                                              ; preds = %105, %103, %101
  unreachable

101:                                              ; preds = %54
  %102 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 43, ptr null)
  br label %100

103:                                              ; preds = %61
  %104 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 43, ptr null)
  br label %100

105:                                              ; preds = %63
  %106 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 43, ptr null)
  br label %100
}

; Function Attrs: mustprogress
define i32 @inTry(i32 %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca { ptr, i1 }, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  store i32 %0, ptr %9, align 4
  store i32 0, ptr %8, align 4
  store ptr null, ptr %5, align 8
  store ptr null, ptr %4, align 8
  %11 = load i32, ptr %9, align 4
  %12 = invoke ptr @rows(i32 %11)
          to label %13 unwind label %141

13:                                               ; preds = %1
  %14 = call ptr @GC_malloc(i64 8)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = call ptr @GC_malloc(i64 32)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 0
  store ptr %15, ptr %18, align 8
  %19 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 1
  store i64 0, ptr %19, align 8
  %20 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 2
  store i64 0, ptr %20, align 8
  store ptr %17, ptr %3, align 8
  store ptr %12, ptr %7, align 8
  %21 = load ptr, ptr %7, align 8
  %22 = getelementptr { i32, i32, ptr, ptr }, ptr %21, i32 0, i32 2
  %23 = insertvalue { ptr, ptr } zeroinitializer, ptr %22, 0
  %24 = insertvalue { ptr, ptr } %23, ptr %21, 1
  %25 = extractvalue { ptr, ptr } %24, 1
  %26 = extractvalue { ptr, ptr } %24, 0
  %27 = load ptr, ptr %26, align 8
  %28 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %27, 0
  %29 = insertvalue { ptr, ptr, ptr } %28, ptr %25, 1
  %30 = insertvalue { ptr, ptr, ptr } %29, ptr null, 2
  %31 = extractvalue { ptr, ptr, ptr } %30, 1
  %32 = extractvalue { ptr, ptr, ptr } %30, 0
  %33 = invoke { ptr, i1 } %32(ptr %31)
          to label %34 unwind label %141

34:                                               ; preds = %13
  store { ptr, i1 } %33, ptr %6, align 8
  br label %35

35:                                               ; preds = %102, %34
  %36 = getelementptr { ptr, i1 }, ptr %6, i32 0, i32 1
  %37 = load i1, ptr %36, align 1
  %38 = xor i1 %37, true
  br i1 %38, label %39, label %103

39:                                               ; preds = %35
  store ptr null, ptr %2, align 8
  %40 = getelementptr { ptr, i1 }, ptr %6, i32 0, i32 0
  %41 = load ptr, ptr %40, align 8
  store ptr %41, ptr %2, align 8
  %42 = load ptr, ptr %2, align 8
  %43 = load ptr, ptr %3, align 8
  %44 = icmp eq ptr %43, null
  br i1 %44, label %45, label %48

45:                                               ; preds = %39
  %46 = call ptr @GC_malloc(i64 32)
  %47 = getelementptr i8, ptr %46, i64 8
  store ptr %47, ptr %3, align 8
  br label %49

48:                                               ; preds = %39
  br label %49

49:                                               ; preds = %45, %48
  %50 = phi ptr [ %43, %48 ], [ %47, %45 ]
  br label %51

51:                                               ; preds = %49
  %52 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 1
  %53 = load i64, ptr %52, align 8
  %54 = add i64 %53, 1
  %55 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 0
  %56 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 2
  %57 = load ptr, ptr %55, align 8
  %58 = load i64, ptr %56, align 8
  %59 = icmp ugt i64 %54, %58
  br i1 %59, label %60, label %84

60:                                               ; preds = %51
  %61 = mul i64 %58, 2
  %62 = icmp uge i64 %61, %54
  %63 = select i1 %62, i64 %61, i64 %54
  %64 = icmp ult i64 %63, 4
  %65 = select i1 %64, i64 4, i64 %63
  %66 = icmp ule i64 %54, 4294967295
  br i1 %66, label %67, label %149

67:                                               ; preds = %60
  %68 = icmp ule i64 %65, 2305843009213693951
  br i1 %68, label %69, label %151

69:                                               ; preds = %67
  %70 = mul i64 8, %65
  %71 = add i64 %70, 8
  %72 = icmp eq ptr %57, null
  %73 = getelementptr i8, ptr %57, i64 -8
  %74 = select i1 %72, ptr null, ptr %73
  %75 = call ptr @GC_realloc(ptr %74, i64 %71)
  %76 = getelementptr i8, ptr %75, i64 8
  %77 = icmp ne ptr %76, null
  %78 = icmp eq i64 %70, 0
  %79 = or i1 %77, %78
  br i1 %79, label %80, label %153

80:                                               ; preds = %69
  %81 = mul i64 8, %58
  %82 = getelementptr i8, ptr %76, i64 %81
  %83 = sub i64 %70, %81
  call void @llvm.memset.p0.i64(ptr %82, i8 0, i64 %83, i1 false)
  store ptr %76, ptr %55, align 8
  store i64 %65, ptr %56, align 8
  br label %85

84:                                               ; preds = %51
  br label %85

85:                                               ; preds = %80, %84
  %86 = phi ptr [ %57, %84 ], [ %76, %80 ]
  br label %87

87:                                               ; preds = %85
  %88 = getelementptr ptr, ptr %86, i64 %53
  store ptr %42, ptr %88, align 8
  store i64 %54, ptr %52, align 8
  %89 = load ptr, ptr %7, align 8
  %90 = getelementptr { i32, i32, ptr, ptr }, ptr %89, i32 0, i32 2
  %91 = insertvalue { ptr, ptr } zeroinitializer, ptr %90, 0
  %92 = insertvalue { ptr, ptr } %91, ptr %89, 1
  %93 = extractvalue { ptr, ptr } %92, 1
  %94 = extractvalue { ptr, ptr } %92, 0
  %95 = load ptr, ptr %94, align 8
  %96 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %95, 0
  %97 = insertvalue { ptr, ptr, ptr } %96, ptr %93, 1
  %98 = insertvalue { ptr, ptr, ptr } %97, ptr null, 2
  %99 = extractvalue { ptr, ptr, ptr } %98, 1
  %100 = extractvalue { ptr, ptr, ptr } %98, 0
  %101 = invoke { ptr, i1 } %100(ptr %99)
          to label %102 unwind label %141

102:                                              ; preds = %87
  store { ptr, i1 } %101, ptr %6, align 8
  br label %35

103:                                              ; preds = %35
  %104 = load ptr, ptr %3, align 8
  store ptr %104, ptr %5, align 8
  invoke void @churn()
          to label %105 unwind label %141

105:                                              ; preds = %103
  %106 = load ptr, ptr %5, align 8
  %107 = icmp eq ptr %106, null
  %108 = select i1 %107, ptr @__ts_empty_array, ptr %106
  %109 = getelementptr { ptr, i64, i64 }, ptr %108, i32 0, i32 1
  %110 = load i64, ptr %109, align 8
  %111 = load ptr, ptr %5, align 8
  %112 = load i32, ptr %9, align 4
  %113 = sub i32 %112, 1
  %114 = icmp eq ptr %111, null
  %115 = select i1 %114, ptr @__ts_empty_array, ptr %111
  %116 = getelementptr { ptr, i64, i64 }, ptr %115, i32 0, i32 0
  %117 = load ptr, ptr %116, align 8
  %118 = getelementptr ptr, ptr %117, i32 %113
  %119 = load ptr, ptr %118, align 8
  %120 = icmp eq ptr %119, null
  %121 = select i1 %120, ptr @__ts_empty_array, ptr %119
  %122 = getelementptr { ptr, i64, i64 }, ptr %121, i32 0, i32 0
  %123 = load ptr, ptr %122, align 8
  %124 = getelementptr ptr, ptr %123, i32 1
  %125 = load ptr, ptr %124, align 8
  %126 = call i64 @strlen(ptr %125)
  %127 = add i64 %110, %126
  %128 = trunc i64 %127 to i32
  store i32 %128, ptr %8, align 4
  %129 = load i32, ptr %9, align 4
  %130 = invoke ptr @pushed(i32 %129)
          to label %131 unwind label %141

131:                                              ; preds = %105
  store ptr %130, ptr %4, align 8
  %132 = load i32, ptr %8, align 4
  %133 = load ptr, ptr %4, align 8
  %134 = icmp eq ptr %133, null
  %135 = select i1 %134, ptr @__ts_empty_array, ptr %133
  %136 = getelementptr { ptr, i64, i64 }, ptr %135, i32 0, i32 1
  %137 = load i64, ptr %136, align 8
  %138 = sext i32 %132 to i64
  %139 = add i64 %138, %137
  %140 = trunc i64 %139 to i32
  store i32 %140, ptr %8, align 4
  br label %145

141:                                              ; preds = %87, %105, %103, %13, %1
  %142 = landingpad { ptr, i32 }
          catch ptr null
  %143 = extractvalue { ptr, i32 } %142, 0
  %144 = call ptr @__cxa_begin_catch(ptr %143)
  store i32 -1, ptr %8, align 4
  call void @__cxa_end_catch()
  br label %145

145:                                              ; preds = %141, %131
  %146 = load i32, ptr %8, align 4
  store i32 %146, ptr %10, align 4
  %147 = load i32, ptr %10, align 4
  ret i32 %147

148:                                              ; preds = %153, %151, %149
  unreachable

149:                                              ; preds = %60
  %150 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 50, ptr null)
  br label %148

151:                                              ; preds = %67
  %152 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 50, ptr null)
  br label %148

153:                                              ; preds = %69
  %154 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 50, ptr null)
  br label %148
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, i1 }, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  store i32 0, ptr %10, align 4
  store i32 0, ptr %9, align 4
  br label %11

11:                                               ; preds = %241, %0
  %12 = load i32, ptr %9, align 4
  %13 = icmp slt i32 %12, 300
  br i1 %13, label %14, label %244

14:                                               ; preds = %11
  %15 = call ptr @pushed(i32 4)
  store ptr %15, ptr %8, align 8
  call void @churn()
  %16 = load ptr, ptr %8, align 8
  %17 = icmp eq ptr %16, null
  %18 = select i1 %17, ptr @__ts_empty_array, ptr %16
  %19 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 1
  %20 = load i64, ptr %19, align 8
  %21 = icmp eq i64 %20, 4
  br i1 %21, label %22, label %33

22:                                               ; preds = %14
  %23 = load ptr, ptr %8, align 8
  %24 = icmp eq ptr %23, null
  %25 = select i1 %24, ptr @__ts_empty_array, ptr %23
  %26 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr ptr, ptr %27, i32 3
  %29 = load ptr, ptr %28, align 8
  %30 = getelementptr %C, ptr %29, i32 0, i32 1
  %31 = load double, ptr %30, align 8
  %32 = fcmp oeq double %31, 3.000000e+00
  br label %33

33:                                               ; preds = %22, %14
  %34 = phi i1 [ %32, %22 ], [ %21, %14 ]
  br i1 %34, label %35, label %38

35:                                               ; preds = %33
  %36 = load i32, ptr %10, align 4
  %37 = add i32 %36, 1
  store i32 %37, ptr %10, align 4
  br label %38

38:                                               ; preds = %35, %33
  %39 = call ptr @evens(i32 5)
  store ptr %39, ptr %7, align 8
  call void @churn()
  %40 = load ptr, ptr %7, align 8
  %41 = icmp eq ptr %40, null
  %42 = select i1 %41, ptr @__ts_empty_array, ptr %40
  %43 = getelementptr { ptr, i64, i64 }, ptr %42, i32 0, i32 1
  %44 = load i64, ptr %43, align 8
  %45 = icmp eq i64 %44, 3
  br i1 %45, label %46, label %57

46:                                               ; preds = %38
  %47 = load ptr, ptr %7, align 8
  %48 = icmp eq ptr %47, null
  %49 = select i1 %48, ptr @__ts_empty_array, ptr %47
  %50 = getelementptr { ptr, i64, i64 }, ptr %49, i32 0, i32 0
  %51 = load ptr, ptr %50, align 8
  %52 = getelementptr ptr, ptr %51, i32 2
  %53 = load ptr, ptr %52, align 8
  %54 = getelementptr %C, ptr %53, i32 0, i32 1
  %55 = load double, ptr %54, align 8
  %56 = fcmp oeq double %55, 4.000000e+00
  br label %57

57:                                               ; preds = %46, %38
  %58 = phi i1 [ %56, %46 ], [ %45, %38 ]
  br i1 %58, label %59, label %62

59:                                               ; preds = %57
  %60 = load i32, ptr %10, align 4
  %61 = add i32 %60, 1
  store i32 %61, ptr %10, align 4
  br label %62

62:                                               ; preds = %59, %57
  %63 = call ptr @spread(i32 3)
  store ptr %63, ptr %6, align 8
  call void @churn()
  %64 = load ptr, ptr %6, align 8
  %65 = icmp eq ptr %64, null
  %66 = select i1 %65, ptr @__ts_empty_array, ptr %64
  %67 = getelementptr { ptr, i64, i64 }, ptr %66, i32 0, i32 1
  %68 = load i64, ptr %67, align 8
  %69 = icmp eq i64 %68, 3
  br i1 %69, label %70, label %81

70:                                               ; preds = %62
  %71 = load ptr, ptr %6, align 8
  %72 = icmp eq ptr %71, null
  %73 = select i1 %72, ptr @__ts_empty_array, ptr %71
  %74 = getelementptr { ptr, i64, i64 }, ptr %73, i32 0, i32 0
  %75 = load ptr, ptr %74, align 8
  %76 = getelementptr ptr, ptr %75, i32 1
  %77 = load ptr, ptr %76, align 8
  %78 = getelementptr %C, ptr %77, i32 0, i32 1
  %79 = load double, ptr %78, align 8
  %80 = fcmp oeq double %79, 1.000000e+00
  br label %81

81:                                               ; preds = %70, %62
  %82 = phi i1 [ %80, %70 ], [ %69, %62 ]
  br i1 %82, label %83, label %86

83:                                               ; preds = %81
  %84 = load i32, ptr %10, align 4
  %85 = add i32 %84, 1
  store i32 %85, ptr %10, align 4
  br label %86

86:                                               ; preds = %83, %81
  %87 = call ptr @rows(i32 2)
  %88 = call ptr @GC_malloc(i64 8)
  %89 = getelementptr i8, ptr %88, i64 8
  %90 = call ptr @GC_malloc(i64 32)
  %91 = getelementptr i8, ptr %90, i64 8
  %92 = getelementptr { ptr, i64, i64 }, ptr %91, i32 0, i32 0
  store ptr %89, ptr %92, align 8
  %93 = getelementptr { ptr, i64, i64 }, ptr %91, i32 0, i32 1
  store i64 0, ptr %93, align 8
  %94 = getelementptr { ptr, i64, i64 }, ptr %91, i32 0, i32 2
  store i64 0, ptr %94, align 8
  store ptr %91, ptr %5, align 8
  store ptr %87, ptr %4, align 8
  %95 = load ptr, ptr %4, align 8
  %96 = getelementptr { i32, i32, ptr, ptr }, ptr %95, i32 0, i32 2
  %97 = insertvalue { ptr, ptr } zeroinitializer, ptr %96, 0
  %98 = insertvalue { ptr, ptr } %97, ptr %95, 1
  %99 = extractvalue { ptr, ptr } %98, 1
  %100 = extractvalue { ptr, ptr } %98, 0
  %101 = load ptr, ptr %100, align 8
  %102 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %101, 0
  %103 = insertvalue { ptr, ptr, ptr } %102, ptr %99, 1
  %104 = insertvalue { ptr, ptr, ptr } %103, ptr null, 2
  %105 = extractvalue { ptr, ptr, ptr } %104, 1
  %106 = extractvalue { ptr, ptr, ptr } %104, 0
  %107 = call { ptr, i1 } %106(ptr %105)
  store { ptr, i1 } %107, ptr %3, align 8
  br label %108

108:                                              ; preds = %160, %86
  %109 = getelementptr { ptr, i1 }, ptr %3, i32 0, i32 1
  %110 = load i1, ptr %109, align 1
  %111 = xor i1 %110, true
  br i1 %111, label %112, label %175

112:                                              ; preds = %108
  %113 = getelementptr { ptr, i1 }, ptr %3, i32 0, i32 0
  %114 = load ptr, ptr %113, align 8
  store ptr %114, ptr %2, align 8
  %115 = load ptr, ptr %2, align 8
  %116 = load ptr, ptr %5, align 8
  %117 = icmp eq ptr %116, null
  br i1 %117, label %118, label %121

118:                                              ; preds = %112
  %119 = call ptr @GC_malloc(i64 32)
  %120 = getelementptr i8, ptr %119, i64 8
  store ptr %120, ptr %5, align 8
  br label %122

121:                                              ; preds = %112
  br label %122

122:                                              ; preds = %118, %121
  %123 = phi ptr [ %116, %121 ], [ %120, %118 ]
  br label %124

124:                                              ; preds = %122
  %125 = getelementptr { ptr, i64, i64 }, ptr %123, i32 0, i32 1
  %126 = load i64, ptr %125, align 8
  %127 = add i64 %126, 1
  %128 = getelementptr { ptr, i64, i64 }, ptr %123, i32 0, i32 0
  %129 = getelementptr { ptr, i64, i64 }, ptr %123, i32 0, i32 2
  %130 = load ptr, ptr %128, align 8
  %131 = load i64, ptr %129, align 8
  %132 = icmp ugt i64 %127, %131
  br i1 %132, label %133, label %157

133:                                              ; preds = %124
  %134 = mul i64 %131, 2
  %135 = icmp uge i64 %134, %127
  %136 = select i1 %135, i64 %134, i64 %127
  %137 = icmp ult i64 %136, 4
  %138 = select i1 %137, i64 4, i64 %136
  %139 = icmp ule i64 %127, 4294967295
  br i1 %139, label %140, label %249

140:                                              ; preds = %133
  %141 = icmp ule i64 %138, 2305843009213693951
  br i1 %141, label %142, label %251

142:                                              ; preds = %140
  %143 = mul i64 8, %138
  %144 = add i64 %143, 8
  %145 = icmp eq ptr %130, null
  %146 = getelementptr i8, ptr %130, i64 -8
  %147 = select i1 %145, ptr null, ptr %146
  %148 = call ptr @GC_realloc(ptr %147, i64 %144)
  %149 = getelementptr i8, ptr %148, i64 8
  %150 = icmp ne ptr %149, null
  %151 = icmp eq i64 %143, 0
  %152 = or i1 %150, %151
  br i1 %152, label %153, label %253

153:                                              ; preds = %142
  %154 = mul i64 8, %131
  %155 = getelementptr i8, ptr %149, i64 %154
  %156 = sub i64 %143, %154
  call void @llvm.memset.p0.i64(ptr %155, i8 0, i64 %156, i1 false)
  store ptr %149, ptr %128, align 8
  store i64 %138, ptr %129, align 8
  br label %158

157:                                              ; preds = %124
  br label %158

158:                                              ; preds = %153, %157
  %159 = phi ptr [ %130, %157 ], [ %149, %153 ]
  br label %160

160:                                              ; preds = %158
  %161 = getelementptr ptr, ptr %159, i64 %126
  store ptr %115, ptr %161, align 8
  store i64 %127, ptr %125, align 8
  %162 = load ptr, ptr %4, align 8
  %163 = getelementptr { i32, i32, ptr, ptr }, ptr %162, i32 0, i32 2
  %164 = insertvalue { ptr, ptr } zeroinitializer, ptr %163, 0
  %165 = insertvalue { ptr, ptr } %164, ptr %162, 1
  %166 = extractvalue { ptr, ptr } %165, 1
  %167 = extractvalue { ptr, ptr } %165, 0
  %168 = load ptr, ptr %167, align 8
  %169 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %168, 0
  %170 = insertvalue { ptr, ptr, ptr } %169, ptr %166, 1
  %171 = insertvalue { ptr, ptr, ptr } %170, ptr null, 2
  %172 = extractvalue { ptr, ptr, ptr } %171, 1
  %173 = extractvalue { ptr, ptr, ptr } %171, 0
  %174 = call { ptr, i1 } %173(ptr %172)
  store { ptr, i1 } %174, ptr %3, align 8
  br label %108

175:                                              ; preds = %108
  %176 = load ptr, ptr %5, align 8
  store ptr %176, ptr %1, align 8
  call void @churn()
  %177 = load ptr, ptr %1, align 8
  %178 = icmp eq ptr %177, null
  %179 = select i1 %178, ptr @__ts_empty_array, ptr %177
  %180 = getelementptr { ptr, i64, i64 }, ptr %179, i32 0, i32 0
  %181 = load ptr, ptr %180, align 8
  %182 = getelementptr ptr, ptr %181, i32 1
  %183 = load ptr, ptr %182, align 8
  %184 = icmp eq ptr %183, null
  %185 = select i1 %184, ptr @__ts_empty_array, ptr %183
  %186 = getelementptr { ptr, i64, i64 }, ptr %185, i32 0, i32 0
  %187 = load ptr, ptr %186, align 8
  %188 = getelementptr ptr, ptr %187, i32 0
  %189 = load ptr, ptr %188, align 8
  %190 = ptrtoint ptr %189 to i64
  %191 = icmp ne i64 %190, 0
  %192 = zext i1 %191 to i32
  %193 = and i32 %192, 1
  %194 = icmp ne i32 %193, 0
  br i1 %194, label %195, label %198

195:                                              ; preds = %175
  %196 = call i32 @strcmp(ptr %189, ptr getelementptr inbounds nuw (i8, ptr @s_637539755847373129, i64 8))
  %197 = icmp eq i32 %196, 0
  br label %200

198:                                              ; preds = %175
  %199 = icmp eq i64 %190, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637539755847373129, i64 8) to i64)
  br label %200

200:                                              ; preds = %195, %198
  %201 = phi i1 [ %199, %198 ], [ %197, %195 ]
  br label %202

202:                                              ; preds = %200
  br i1 %201, label %203, label %230

203:                                              ; preds = %202
  %204 = load ptr, ptr %1, align 8
  %205 = icmp eq ptr %204, null
  %206 = select i1 %205, ptr @__ts_empty_array, ptr %204
  %207 = getelementptr { ptr, i64, i64 }, ptr %206, i32 0, i32 0
  %208 = load ptr, ptr %207, align 8
  %209 = getelementptr ptr, ptr %208, i32 1
  %210 = load ptr, ptr %209, align 8
  %211 = icmp eq ptr %210, null
  %212 = select i1 %211, ptr @__ts_empty_array, ptr %210
  %213 = getelementptr { ptr, i64, i64 }, ptr %212, i32 0, i32 0
  %214 = load ptr, ptr %213, align 8
  %215 = getelementptr ptr, ptr %214, i32 1
  %216 = load ptr, ptr %215, align 8
  %217 = ptrtoint ptr %216 to i64
  %218 = icmp ne i64 %217, 0
  %219 = zext i1 %218 to i32
  %220 = and i32 %219, 1
  %221 = icmp ne i32 %220, 0
  br i1 %221, label %222, label %225

222:                                              ; preds = %203
  %223 = call i32 @strcmp(ptr %216, ptr getelementptr inbounds nuw (i8, ptr @s_637536457312488496, i64 8))
  %224 = icmp eq i32 %223, 0
  br label %227

225:                                              ; preds = %203
  %226 = icmp eq i64 %217, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637536457312488496, i64 8) to i64)
  br label %227

227:                                              ; preds = %222, %225
  %228 = phi i1 [ %226, %225 ], [ %224, %222 ]
  br label %229

229:                                              ; preds = %227
  br label %230

230:                                              ; preds = %229, %202
  %231 = phi i1 [ %228, %229 ], [ %201, %202 ]
  br i1 %231, label %232, label %235

232:                                              ; preds = %230
  %233 = load i32, ptr %10, align 4
  %234 = add i32 %233, 1
  store i32 %234, ptr %10, align 4
  br label %235

235:                                              ; preds = %232, %230
  %236 = call i32 @inTry(i32 3)
  %237 = icmp eq i32 %236, 8
  br i1 %237, label %238, label %241

238:                                              ; preds = %235
  %239 = load i32, ptr %10, align 4
  %240 = add i32 %239, 1
  store i32 %240, ptr %10, align 4
  br label %241

241:                                              ; preds = %238, %235
  %242 = load i32, ptr %9, align 4
  %243 = add i32 %242, 1
  store i32 %243, ptr %9, align 4
  br label %11

244:                                              ; preds = %11
  %245 = load i32, ptr %10, align 4
  %246 = icmp eq i32 %245, 1500
  br i1 %246, label %247, label %255

247:                                              ; preds = %244
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

248:                                              ; preds = %253, %251, %249, %255
  unreachable

249:                                              ; preds = %133
  %250 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 74, ptr null)
  br label %248

251:                                              ; preds = %140
  %252 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 74, ptr null)
  br label %248

253:                                              ; preds = %142
  %254 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 74, ptr null)
  br label %248

255:                                              ; preds = %244
  %256 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_626102635892993007, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5745265928196723405, i64 8), i32 80, ptr null)
  br label %248
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

