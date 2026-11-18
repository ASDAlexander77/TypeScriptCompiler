; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, double }
%H = type { ptr, ptr, ptr }
%Junk = type { ptr, i32, i32 }

$C..size = comdat any

$H..size = comdat any

$Junk..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@s_1560590634776335115 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFJunk\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_15715255546628866100 = internal constant [108 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00const_field_read.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_16176887329329086193 = internal constant [64 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe read survives the field's overwrite through finally\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_14602836883454343452 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field read\00", align 8
@m_15398452451596293419 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nullable field read\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 16, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C..rtti, ptr @C..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 24, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@Junk..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1560590634776335115, i64 8)
@Junk..size = global i64 16, comdat
@Junk..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Junk..instanceOf, ptr @Junk..new, ptr @Junk..rtti, ptr @Junk..size }
@a_0_6706924739130738156 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

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
define void @H.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %C, ptr %6, i32 0, i32 0
  store ptr @C..vtbl, ptr %7, align 8
  call void @C.constructor(ptr %6, double 1.000000e+00)
  store ptr %6, ptr %4, align 8
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr %H, ptr %8, i32 0, i32 2
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %C, ptr %11, i32 0, i32 0
  store ptr @C..vtbl, ptr %12, align 8
  call void @C.constructor(ptr %11, double 2.000000e+00)
  store ptr %11, ptr %9, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @H..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @H..rtti, align 8
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
define ptr @H..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 0
  store ptr @H..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Junk.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Junk, ptr %3, i32 0, i32 1
  store i32 -1, ptr %4, align 4
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Junk, ptr %5, i32 0, i32 2
  store i32 -2, ptr %6, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Junk..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Junk..rtti, align 8
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
define ptr @Junk..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Junk, ptr %3, i32 0, i32 0
  store ptr @Junk..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i64 @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = call ptr @GC_malloc(i64 8)
  %5 = getelementptr i8, ptr %4, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %5, ptr @a_0_6706924739130738156, i64 0, i1 false)
  %6 = call ptr @GC_malloc(i64 32)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 0
  store ptr %5, ptr %8, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 1
  store i64 0, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 2
  store i64 0, ptr %10, align 8
  store ptr %7, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %11

11:                                               ; preds = %62, %0
  %12 = load i32, ptr %1, align 4
  %13 = icmp slt i32 %12, 1000
  br i1 %13, label %14, label %66

14:                                               ; preds = %11
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %Junk, ptr %16, i32 0, i32 0
  store ptr @Junk..vtbl, ptr %17, align 8
  call void @Junk.constructor(ptr %16)
  %18 = load ptr, ptr %2, align 8
  %19 = icmp eq ptr %18, null
  br i1 %19, label %20, label %23

20:                                               ; preds = %14
  %21 = call ptr @GC_malloc(i64 32)
  %22 = getelementptr i8, ptr %21, i64 8
  store ptr %22, ptr %2, align 8
  br label %24

23:                                               ; preds = %14
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
  br i1 %41, label %42, label %74

42:                                               ; preds = %35
  %43 = icmp ule i64 %40, 2305843009213693951
  br i1 %43, label %44, label %76

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
  br i1 %54, label %55, label %78

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
  %63 = getelementptr ptr, ptr %61, i64 %28
  store ptr %16, ptr %63, align 8
  store i64 %29, ptr %27, align 8
  %64 = load i32, ptr %1, align 4
  %65 = add i32 %64, 1
  store i32 %65, ptr %1, align 4
  br label %11

66:                                               ; preds = %11
  %67 = load ptr, ptr %2, align 8
  %68 = icmp eq ptr %67, null
  %69 = select i1 %68, ptr @__ts_empty_array, ptr %67
  %70 = getelementptr { ptr, i64, i64 }, ptr %69, i32 0, i32 1
  %71 = load i64, ptr %70, align 8
  store i64 %71, ptr %3, align 8
  %72 = load i64, ptr %3, align 8
  ret i64 %72

73:                                               ; preds = %78, %76, %74
  unreachable

74:                                               ; preds = %35
  %75 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 21, ptr null)
  br label %73

76:                                               ; preds = %42
  %77 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 21, ptr null)
  br label %73

78:                                               ; preds = %44
  %79 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 21, ptr null)
  br label %73
}

; Function Attrs: mustprogress
define i1 @field(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca i1, align 1
  %4 = alloca <{ i64, %H }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %H zeroinitializer, ptr %5, align 8
  store double %0, ptr %2, align 8
  store %H zeroinitializer, ptr %5, align 8
  %6 = getelementptr %H, ptr %5, i32 0, i32 0
  store ptr @H..vtbl, ptr %6, align 8
  call void @H.constructor(ptr %5)
  %7 = getelementptr %H, ptr %5, i32 0, i32 1
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %C, ptr %9, i32 0, i32 0
  store ptr @C..vtbl, ptr %10, align 8
  %11 = load double, ptr %2, align 8
  call void @C.constructor(ptr %9, double %11)
  store ptr %9, ptr %7, align 8
  %12 = getelementptr %H, ptr %5, i32 0, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %H, ptr %5, i32 0, i32 1
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %C, ptr %16, i32 0, i32 0
  store ptr @C..vtbl, ptr %17, align 8
  call void @C.constructor(ptr %16, double 9.000000e+00)
  store ptr %16, ptr %14, align 8
  %18 = call i64 @churn()
  %19 = getelementptr %C, ptr %13, i32 0, i32 1
  %20 = load double, ptr %19, align 8
  %21 = load double, ptr %2, align 8
  %22 = fcmp oeq double %20, %21
  br i1 %22, label %23, label %29

23:                                               ; preds = %1
  %24 = getelementptr %H, ptr %5, i32 0, i32 1
  %25 = load ptr, ptr %24, align 8
  %26 = getelementptr %C, ptr %25, i32 0, i32 1
  %27 = load double, ptr %26, align 8
  %28 = fcmp oeq double %27, 9.000000e+00
  br label %29

29:                                               ; preds = %23, %1
  %30 = phi i1 [ %28, %23 ], [ %22, %1 ]
  store i1 %30, ptr %3, align 1
  %31 = load i1, ptr %3, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define i1 @nullableField(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca i1, align 1
  %4 = alloca <{ i64, %H }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %H zeroinitializer, ptr %5, align 8
  store double %0, ptr %2, align 8
  store %H zeroinitializer, ptr %5, align 8
  %6 = getelementptr %H, ptr %5, i32 0, i32 0
  store ptr @H..vtbl, ptr %6, align 8
  call void @H.constructor(ptr %5)
  %7 = getelementptr %H, ptr %5, i32 0, i32 2
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %C, ptr %9, i32 0, i32 0
  store ptr @C..vtbl, ptr %10, align 8
  %11 = load double, ptr %2, align 8
  call void @C.constructor(ptr %9, double %11)
  store ptr %9, ptr %7, align 8
  %12 = getelementptr %H, ptr %5, i32 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %H, ptr %5, i32 0, i32 2
  store ptr null, ptr %14, align 8
  %15 = call i64 @churn()
  %16 = ptrtoint ptr %13 to i64
  %17 = icmp ne i64 %16, 0
  br i1 %17, label %18, label %23

18:                                               ; preds = %1
  %19 = getelementptr %C, ptr %13, i32 0, i32 1
  %20 = load double, ptr %19, align 8
  %21 = load double, ptr %2, align 8
  %22 = fcmp oeq double %20, %21
  br label %23

23:                                               ; preds = %18, %1
  %24 = phi i1 [ %22, %18 ], [ %17, %1 ]
  store i1 %24, ptr %3, align 1
  %25 = load i1, ptr %3, align 1
  ret i1 %25
}

; Function Attrs: mustprogress
define i32 @readInFinally(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr %H, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  store i32 0, ptr %2, align 4
  %8 = load ptr, ptr %3, align 8
  %9 = getelementptr %H, ptr %8, i32 0, i32 1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %C, ptr %11, i32 0, i32 0
  store ptr @C..vtbl, ptr %12, align 8
  invoke void @C.constructor(ptr %11, double 9.000000e+00)
          to label %13 unwind label %24

13:                                               ; preds = %1
  store ptr %11, ptr %9, align 8
  %14 = load i32, ptr %2, align 4
  store i32 %14, ptr %4, align 4
  %15 = call i64 @churn()
  %16 = getelementptr %C, ptr %7, i32 0, i32 1
  %17 = load double, ptr %16, align 8
  %18 = fptosi double %17 to i64
  %19 = trunc i64 %18 to i32
  store i32 %19, ptr %2, align 4
  %20 = load i32, ptr %2, align 4
  %21 = icmp eq i32 %20, 1
  br i1 %21, label %22, label %35

22:                                               ; preds = %13
  %23 = load i32, ptr %4, align 4
  ret i32 %23

24:                                               ; preds = %1
  %25 = landingpad { ptr, i32 }
          cleanup
  %26 = call i64 @churn()
  %27 = getelementptr %C, ptr %7, i32 0, i32 1
  %28 = load double, ptr %27, align 8
  %29 = fptosi double %28 to i64
  %30 = trunc i64 %29 to i32
  store i32 %30, ptr %2, align 4
  %31 = load i32, ptr %2, align 4
  %32 = icmp eq i32 %31, 1
  br i1 %32, label %33, label %37

33:                                               ; preds = %24
  resume { ptr, i32 } %25

34:                                               ; preds = %35, %37
  unreachable

35:                                               ; preds = %13
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16176887329329086193, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 52, ptr null)
  br label %34

37:                                               ; preds = %24
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16176887329329086193, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 52, ptr null)
  br label %34
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca <{ i64, %H }>, align 16
  store i64 -1, ptr %2, align 8
  %3 = getelementptr i8, ptr %2, i64 8
  store %H zeroinitializer, ptr %3, align 8
  store i32 0, ptr %1, align 4
  br label %4

4:                                                ; preds = %15, %0
  %5 = load i32, ptr %1, align 4
  %6 = icmp slt i32 %5, 20
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  %8 = load i32, ptr %1, align 4
  %9 = sitofp i32 %8 to double
  %10 = call i1 @field(double %9)
  br i1 %10, label %11, label %22

11:                                               ; preds = %7
  %12 = load i32, ptr %1, align 4
  %13 = sitofp i32 %12 to double
  %14 = call i1 @nullableField(double %13)
  br i1 %14, label %15, label %24

15:                                               ; preds = %11
  %16 = load i32, ptr %1, align 4
  %17 = add i32 %16, 1
  store i32 %17, ptr %1, align 4
  br label %4

18:                                               ; preds = %4
  store %H zeroinitializer, ptr %3, align 8
  %19 = getelementptr %H, ptr %3, i32 0, i32 0
  store ptr @H..vtbl, ptr %19, align 8
  call void @H.constructor(ptr %3)
  %20 = call i32 @readInFinally(ptr %3)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

21:                                               ; preds = %24, %22
  unreachable

22:                                               ; preds = %7
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14602836883454343452, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 58, ptr null)
  br label %21

24:                                               ; preds = %11
  %25 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15398452451596293419, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15715255546628866100, i64 8), i32 59, ptr null)
  br label %21
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

