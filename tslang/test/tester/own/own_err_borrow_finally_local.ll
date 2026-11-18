; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, ptr, double }

$C..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_8779641955872412825 = internal constant [116 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_err_borrow_finally_local.ts\00", align 8
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
@a_0_17033125819989208466 = internal constant [0 x double] undef
@a_0_2926441800300339944 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 13, ptr null)
  br label %113

116:                                              ; preds = %42
  %117 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 13, ptr null)
  br label %113

118:                                              ; preds = %44
  %119 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 13, ptr null)
  br label %113

120:                                              ; preds = %81
  %121 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 14, ptr null)
  br label %113

122:                                              ; preds = %88
  %123 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 14, ptr null)
  br label %113

124:                                              ; preds = %90
  %125 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 14, ptr null)
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
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %C, ptr %6, i32 0, i32 0
  store ptr @C..vtbl, ptr %7, align 8
  call void @C.constructor(ptr %6, double 1.000000e+00)
  store ptr %6, ptr %4, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = getelementptr %C, ptr %8, i32 0, i32 1
  %10 = load ptr, ptr %9, align 8
  %11 = icmp eq ptr %10, null
  br i1 %11, label %12, label %15

12:                                               ; preds = %0
  %13 = call ptr @GC_malloc(i64 32)
  %14 = getelementptr i8, ptr %13, i64 8
  store ptr %14, ptr %9, align 8
  br label %16

15:                                               ; preds = %0
  br label %16

16:                                               ; preds = %12, %15
  %17 = phi ptr [ %10, %15 ], [ %14, %12 ]
  br label %18

18:                                               ; preds = %16
  %19 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 1
  %20 = load i64, ptr %19, align 8
  %21 = add i64 %20, 1
  %22 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 0
  %23 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 2
  %24 = load ptr, ptr %22, align 8
  %25 = load i64, ptr %23, align 8
  %26 = icmp ugt i64 %21, %25
  br i1 %26, label %27, label %51

27:                                               ; preds = %18
  %28 = mul i64 %25, 2
  %29 = icmp uge i64 %28, %21
  %30 = select i1 %29, i64 %28, i64 %21
  %31 = icmp ult i64 %30, 4
  %32 = select i1 %31, i64 4, i64 %30
  %33 = icmp ule i64 %21, 4294967295
  br i1 %33, label %34, label %145

34:                                               ; preds = %27
  %35 = icmp ule i64 %32, 2305843009213693951
  br i1 %35, label %36, label %147

36:                                               ; preds = %34
  %37 = mul i64 8, %32
  %38 = add i64 %37, 8
  %39 = icmp eq ptr %24, null
  %40 = getelementptr i8, ptr %24, i64 -8
  %41 = select i1 %39, ptr null, ptr %40
  %42 = call ptr @GC_realloc(ptr %41, i64 %38)
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = icmp ne ptr %43, null
  %45 = icmp eq i64 %37, 0
  %46 = or i1 %44, %45
  br i1 %46, label %47, label %149

47:                                               ; preds = %36
  %48 = mul i64 8, %25
  %49 = getelementptr i8, ptr %43, i64 %48
  %50 = sub i64 %37, %48
  call void @llvm.memset.p0.i64(ptr %49, i8 0, i64 %50, i1 false)
  store ptr %43, ptr %22, align 8
  store i64 %32, ptr %23, align 8
  br label %52

51:                                               ; preds = %18
  br label %52

52:                                               ; preds = %47, %51
  %53 = phi ptr [ %24, %51 ], [ %43, %47 ]
  br label %54

54:                                               ; preds = %52
  %55 = getelementptr double, ptr %53, i64 %20
  store double 1.000000e+00, ptr %55, align 8
  store i64 %21, ptr %19, align 8
  %56 = load ptr, ptr %4, align 8
  store ptr %56, ptr %3, align 8
  %57 = load ptr, ptr %4, align 8
  store ptr %57, ptr %2, align 8
  %58 = load ptr, ptr %4, align 8
  %59 = getelementptr %C, ptr %58, i32 0, i32 2
  %60 = load double, ptr %59, align 8
  %61 = load ptr, ptr %2, align 8
  %62 = getelementptr %C, ptr %61, i32 0, i32 2
  %63 = load double, ptr %62, align 8
  %64 = call ptr @GC_malloc_atomic(i64 40)
  %65 = getelementptr i8, ptr %64, i64 8
  call void @__ts_number_to_string(ptr %65, double %60)
  %66 = call ptr @GC_malloc_atomic(i64 40)
  %67 = getelementptr i8, ptr %66, i64 8
  call void @__ts_number_to_string(ptr %67, double %63)
  %68 = call ptr @llvm.stacksave.p0()
  %69 = icmp eq ptr %65, null
  %70 = select i1 %69, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %65
  %71 = icmp eq ptr %67, null
  %72 = select i1 %71, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %67
  %73 = call i64 @strlen(ptr %70)
  %74 = add i64 1, %73
  %75 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %76 = add i64 %74, %75
  %77 = call i64 @strlen(ptr %72)
  %78 = add i64 %76, %77
  %79 = alloca i8, i64 %78, align 1
  %80 = getelementptr i8, ptr %79, i64 0
  %81 = sub i64 %78, 0
  %82 = call ptr @__memcpy_chk(ptr %80, ptr %70, i64 %73, i64 %81)
  %83 = add i64 0, %73
  %84 = getelementptr i8, ptr %79, i64 %83
  %85 = sub i64 %78, %83
  %86 = call ptr @__memcpy_chk(ptr %84, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %75, i64 %85)
  %87 = add i64 %83, %75
  %88 = getelementptr i8, ptr %79, i64 %87
  %89 = sub i64 %78, %87
  %90 = call ptr @__memcpy_chk(ptr %88, ptr %72, i64 %77, i64 %89)
  %91 = add i64 %87, %77
  %92 = getelementptr i8, ptr %79, i64 %91
  store i8 0, ptr %92, align 1
  call void @puts(ptr %79)
  call void @llvm.stackrestore.p0(ptr %68)
  %93 = load ptr, ptr %3, align 8
  %94 = getelementptr %C, ptr %93, i32 0, i32 2
  %95 = load double, ptr %94, align 8
  %96 = call ptr @GC_malloc_atomic(i64 40)
  %97 = getelementptr i8, ptr %96, i64 8
  call void @__ts_number_to_string(ptr %97, double %95)
  %98 = icmp eq ptr %97, null
  %99 = select i1 %98, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %97
  call void @puts(ptr %99)
  %100 = load ptr, ptr %3, align 8
  store ptr %100, ptr %1, align 8
  %101 = call ptr @GC_malloc(i64 32)
  %102 = getelementptr i8, ptr %101, i64 8
  %103 = getelementptr %C, ptr %102, i32 0, i32 0
  store ptr @C..vtbl, ptr %103, align 8
  call void @C.constructor(ptr %102, double 2.000000e+00)
  store ptr %102, ptr %4, align 8
  call void @churn()
  %104 = load ptr, ptr %1, align 8
  %105 = getelementptr %C, ptr %104, i32 0, i32 2
  %106 = load double, ptr %105, align 8
  %107 = load ptr, ptr %1, align 8
  %108 = getelementptr %C, ptr %107, i32 0, i32 1
  %109 = load ptr, ptr %108, align 8
  %110 = icmp eq ptr %109, null
  %111 = select i1 %110, ptr @__ts_empty_array, ptr %109
  %112 = getelementptr { ptr, i64, i64 }, ptr %111, i32 0, i32 1
  %113 = load i64, ptr %112, align 8
  %114 = call ptr @GC_malloc_atomic(i64 40)
  %115 = getelementptr i8, ptr %114, i64 8
  call void @__ts_number_to_string(ptr %115, double %106)
  %116 = call ptr @GC_malloc_atomic(i64 58)
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %117, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %113)
  %119 = call ptr @llvm.stacksave.p0()
  %120 = icmp eq ptr %115, null
  %121 = select i1 %120, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %115
  %122 = icmp eq ptr %117, null
  %123 = select i1 %122, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %117
  %124 = call i64 @strlen(ptr %121)
  %125 = add i64 1, %124
  %126 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %127 = add i64 %125, %126
  %128 = call i64 @strlen(ptr %123)
  %129 = add i64 %127, %128
  %130 = alloca i8, i64 %129, align 1
  %131 = getelementptr i8, ptr %130, i64 0
  %132 = sub i64 %129, 0
  %133 = call ptr @__memcpy_chk(ptr %131, ptr %121, i64 %124, i64 %132)
  %134 = add i64 0, %124
  %135 = getelementptr i8, ptr %130, i64 %134
  %136 = sub i64 %129, %134
  %137 = call ptr @__memcpy_chk(ptr %135, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %126, i64 %136)
  %138 = add i64 %134, %126
  %139 = getelementptr i8, ptr %130, i64 %138
  %140 = sub i64 %129, %138
  %141 = call ptr @__memcpy_chk(ptr %139, ptr %123, i64 %128, i64 %140)
  %142 = add i64 %138, %128
  %143 = getelementptr i8, ptr %130, i64 %142
  store i8 0, ptr %143, align 1
  call void @puts(ptr %130)
  call void @llvm.stackrestore.p0(ptr %119)
  ret i32 0

144:                                              ; preds = %149, %147, %145
  unreachable

145:                                              ; preds = %27
  %146 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 20, ptr null)
  br label %144

147:                                              ; preds = %34
  %148 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 20, ptr null)
  br label %144

149:                                              ; preds = %36
  %150 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8779641955872412825, i64 8), i32 20, ptr null)
  br label %144
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

