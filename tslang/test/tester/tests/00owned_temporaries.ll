; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Vec = type { ptr, double }

$Vec..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_8913785957651686373 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFVec\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_13523762721940311722 = internal constant [109 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00owned_temporaries.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_10081758968786619921 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested call arguments survive their statement\00", align 8
@m_16524708675501408318 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan intermediate built from intermediates survives\00", align 8
@m_8935485834063972330 = internal constant [69 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa loop body's temporaries do not disturb what it carries out\00", align 8
@m_5300301882725252942 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa temporary the callee keeps is not freed under it\00", align 8
@m_8632710368751004317 = internal constant [64 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa value used as an argument is still owned by its local\00", align 8
@m_324297260880672661 = internal constant [80 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa temporary argument survives a callee that allocates before reading it\00", align 8
@m_5488436750520871312 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdiscarding a result does not disturb anything live\00", align 8
@m_5884349861029923589 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa temporary read by a callee survives the call\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Vec..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_8913785957651686373, i64 8)
@Vec..size = global i64 16, comdat
@Vec..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Vec..instanceOf, ptr @Vec..new, ptr @Vec..rtti, ptr @Vec..size }
@a_0_1382183400266196230 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Vec..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Vec..rtti, align 8
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
define ptr @Vec..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Vec, ptr %3, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Vec.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Vec, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @times(double %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  store double %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Vec, ptr %7, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %8, align 8
  %9 = load double, ptr %4, align 8
  %10 = load ptr, ptr %3, align 8
  %11 = getelementptr %Vec, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  %13 = fmul double %9, %12
  call void @Vec.constructor(ptr %7, double %13)
  store ptr %7, ptr %5, align 8
  %14 = load ptr, ptr %5, align 8
  ret ptr %14
}

; Function Attrs: mustprogress
define ptr @plus(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Vec, ptr %7, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %8, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr %Vec, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = load ptr, ptr %3, align 8
  %13 = getelementptr %Vec, ptr %12, i32 0, i32 1
  %14 = load double, ptr %13, align 8
  %15 = fadd double %11, %14
  call void @Vec.constructor(ptr %7, double %15)
  store ptr %7, ptr %5, align 8
  %16 = load ptr, ptr %5, align 8
  ret ptr %16
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %4, align 8
  store i32 0, ptr %2, align 4
  br label %5

5:                                                ; preds = %8, %0
  %6 = load i32, ptr %2, align 4
  %7 = icmp slt i32 %6, 64
  br i1 %7, label %8, label %12

8:                                                ; preds = %5
  store %Vec zeroinitializer, ptr %4, align 8
  %9 = getelementptr %Vec, ptr %4, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %9, align 8
  call void @Vec.constructor(ptr %4, double 9.990000e+02)
  store ptr %4, ptr %1, align 8
  %10 = load i32, ptr %2, align 4
  %11 = add i32 %10, 1
  store i32 %11, ptr %2, align 4
  br label %5

12:                                               ; preds = %5
  ret void
}

; Function Attrs: mustprogress
define double @nestedCallArguments() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca <{ i64, %Vec }>, align 16
  %4 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Vec zeroinitializer, ptr %5, align 8
  store i64 -1, ptr %3, align 8
  %6 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %6, align 8
  store %Vec zeroinitializer, ptr %6, align 8
  %7 = getelementptr %Vec, ptr %6, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %7, align 8
  call void @Vec.constructor(ptr %6, double 3.000000e+00)
  %8 = call ptr @times(double 2.000000e+00, ptr %6)
  store %Vec zeroinitializer, ptr %5, align 8
  %9 = getelementptr %Vec, ptr %5, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %9, align 8
  call void @Vec.constructor(ptr %5, double 1.000000e+00)
  %10 = call ptr @plus(ptr %8, ptr %5)
  store ptr %10, ptr %1, align 8
  call void @churn()
  %11 = load ptr, ptr %1, align 8
  %12 = getelementptr %Vec, ptr %11, i32 0, i32 1
  %13 = load double, ptr %12, align 8
  store double %13, ptr %2, align 8
  %14 = load double, ptr %2, align 8
  ret double %14
}

; Function Attrs: mustprogress
define double @deeperNesting() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca <{ i64, %Vec }>, align 16
  %4 = alloca <{ i64, %Vec }>, align 16
  %5 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Vec zeroinitializer, ptr %6, align 8
  store i64 -1, ptr %4, align 8
  %7 = getelementptr i8, ptr %4, i64 8
  store %Vec zeroinitializer, ptr %7, align 8
  store i64 -1, ptr %3, align 8
  %8 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %8, align 8
  store %Vec zeroinitializer, ptr %8, align 8
  %9 = getelementptr %Vec, ptr %8, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %9, align 8
  call void @Vec.constructor(ptr %8, double 1.000000e+00)
  %10 = call ptr @times(double 2.000000e+00, ptr %8)
  store %Vec zeroinitializer, ptr %7, align 8
  %11 = getelementptr %Vec, ptr %7, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %11, align 8
  call void @Vec.constructor(ptr %7, double 1.000000e+00)
  %12 = call ptr @times(double 3.000000e+00, ptr %7)
  %13 = call ptr @plus(ptr %10, ptr %12)
  store %Vec zeroinitializer, ptr %6, align 8
  %14 = getelementptr %Vec, ptr %6, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %14, align 8
  call void @Vec.constructor(ptr %6, double 4.000000e+00)
  %15 = call ptr @plus(ptr %13, ptr %6)
  store ptr %15, ptr %1, align 8
  call void @churn()
  %16 = load ptr, ptr %1, align 8
  %17 = getelementptr %Vec, ptr %16, i32 0, i32 1
  %18 = load double, ptr %17, align 8
  store double %18, ptr %2, align 8
  %19 = load double, ptr %2, align 8
  ret double %19
}

; Function Attrs: mustprogress
define double @temporariesInALoop() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca <{ i64, %Vec }>, align 16
  %6 = alloca <{ i64, %Vec }>, align 16
  %7 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %7, align 8
  %8 = getelementptr i8, ptr %7, i64 8
  store %Vec zeroinitializer, ptr %8, align 8
  store i64 -1, ptr %6, align 8
  %9 = getelementptr i8, ptr %6, i64 8
  store %Vec zeroinitializer, ptr %9, align 8
  store i64 -1, ptr %5, align 8
  %10 = getelementptr i8, ptr %5, i64 8
  store %Vec zeroinitializer, ptr %10, align 8
  store i32 0, ptr %3, align 4
  store %Vec zeroinitializer, ptr %10, align 8
  %11 = getelementptr %Vec, ptr %10, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %11, align 8
  call void @Vec.constructor(ptr %10, double 0.000000e+00)
  store ptr %10, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %12

12:                                               ; preds = %15, %0
  %13 = load i32, ptr %1, align 4
  %14 = icmp slt i32 %13, 8
  br i1 %14, label %15, label %32

15:                                               ; preds = %12
  store %Vec zeroinitializer, ptr %9, align 8
  %16 = getelementptr %Vec, ptr %9, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %16, align 8
  %17 = load i32, ptr %1, align 4
  %18 = sitofp i32 %17 to double
  call void @Vec.constructor(ptr %9, double %18)
  %19 = call ptr @times(double 2.000000e+00, ptr %9)
  store %Vec zeroinitializer, ptr %8, align 8
  %20 = getelementptr %Vec, ptr %8, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %20, align 8
  call void @Vec.constructor(ptr %8, double 1.000000e+00)
  %21 = call ptr @plus(ptr %19, ptr %8)
  store ptr %21, ptr %2, align 8
  %22 = load i32, ptr %3, align 4
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %Vec, ptr %23, i32 0, i32 1
  %25 = load double, ptr %24, align 8
  %26 = sitofp i32 %22 to double
  %27 = fadd double %26, %25
  %28 = fptosi double %27 to i64
  %29 = trunc i64 %28 to i32
  store i32 %29, ptr %3, align 4
  %30 = load i32, ptr %1, align 4
  %31 = add i32 %30, 1
  store i32 %31, ptr %1, align 4
  br label %12

32:                                               ; preds = %12
  call void @churn()
  %33 = load i32, ptr %3, align 4
  %34 = load ptr, ptr %2, align 8
  %35 = getelementptr %Vec, ptr %34, i32 0, i32 1
  %36 = load double, ptr %35, align 8
  %37 = sitofp i32 %33 to double
  %38 = fadd double %37, %36
  store double %38, ptr %4, align 8
  %39 = load double, ptr %4, align 8
  ret double %39
}

; Function Attrs: mustprogress
define double @temporaryKeptByCallee() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %4, align 8
  %5 = call ptr @GC_malloc(i64 8)
  %6 = getelementptr i8, ptr %5, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @a_0_1382183400266196230, i64 0, i1 false)
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 0
  store ptr %6, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 1
  store i64 0, ptr %10, align 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %8, i32 0, i32 2
  store i64 0, ptr %11, align 8
  store ptr %8, ptr %1, align 8
  store %Vec zeroinitializer, ptr %4, align 8
  %12 = getelementptr %Vec, ptr %4, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %12, align 8
  call void @Vec.constructor(ptr %4, double 5.000000e+00)
  %13 = call ptr @times(double 2.000000e+00, ptr %4)
  %14 = load ptr, ptr %1, align 8
  %15 = icmp eq ptr %14, null
  br i1 %15, label %16, label %19

16:                                               ; preds = %0
  %17 = call ptr @GC_malloc(i64 32)
  %18 = getelementptr i8, ptr %17, i64 8
  store ptr %18, ptr %1, align 8
  br label %20

19:                                               ; preds = %0
  br label %20

20:                                               ; preds = %16, %19
  %21 = phi ptr [ %14, %19 ], [ %18, %16 ]
  br label %22

22:                                               ; preds = %20
  %23 = getelementptr { ptr, i64, i64 }, ptr %21, i32 0, i32 1
  %24 = load i64, ptr %23, align 8
  %25 = add i64 %24, 1
  %26 = getelementptr { ptr, i64, i64 }, ptr %21, i32 0, i32 0
  %27 = getelementptr { ptr, i64, i64 }, ptr %21, i32 0, i32 2
  %28 = load ptr, ptr %26, align 8
  %29 = load i64, ptr %27, align 8
  %30 = icmp ugt i64 %25, %29
  br i1 %30, label %31, label %55

31:                                               ; preds = %22
  %32 = mul i64 %29, 2
  %33 = icmp uge i64 %32, %25
  %34 = select i1 %33, i64 %32, i64 %25
  %35 = icmp ult i64 %34, 4
  %36 = select i1 %35, i64 4, i64 %34
  %37 = icmp ule i64 %25, 4294967295
  br i1 %37, label %38, label %130

38:                                               ; preds = %31
  %39 = icmp ule i64 %36, 2305843009213693951
  br i1 %39, label %40, label %132

40:                                               ; preds = %38
  %41 = mul i64 8, %36
  %42 = add i64 %41, 8
  %43 = icmp eq ptr %28, null
  %44 = getelementptr i8, ptr %28, i64 -8
  %45 = select i1 %43, ptr null, ptr %44
  %46 = call ptr @GC_realloc(ptr %45, i64 %42)
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = icmp ne ptr %47, null
  %49 = icmp eq i64 %41, 0
  %50 = or i1 %48, %49
  br i1 %50, label %51, label %134

51:                                               ; preds = %40
  %52 = mul i64 8, %29
  %53 = getelementptr i8, ptr %47, i64 %52
  %54 = sub i64 %41, %52
  call void @llvm.memset.p0.i64(ptr %53, i8 0, i64 %54, i1 false)
  store ptr %47, ptr %26, align 8
  store i64 %36, ptr %27, align 8
  br label %56

55:                                               ; preds = %22
  br label %56

56:                                               ; preds = %51, %55
  %57 = phi ptr [ %28, %55 ], [ %47, %51 ]
  br label %58

58:                                               ; preds = %56
  %59 = getelementptr ptr, ptr %57, i64 %24
  store ptr %13, ptr %59, align 8
  store i64 %25, ptr %23, align 8
  %60 = call ptr @GC_malloc(i64 24)
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr %Vec, ptr %61, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %62, align 8
  call void @Vec.constructor(ptr %61, double 7.000000e+00)
  %63 = load ptr, ptr %1, align 8
  %64 = icmp eq ptr %63, null
  br i1 %64, label %65, label %68

65:                                               ; preds = %58
  %66 = call ptr @GC_malloc(i64 32)
  %67 = getelementptr i8, ptr %66, i64 8
  store ptr %67, ptr %1, align 8
  br label %69

68:                                               ; preds = %58
  br label %69

69:                                               ; preds = %65, %68
  %70 = phi ptr [ %63, %68 ], [ %67, %65 ]
  br label %71

71:                                               ; preds = %69
  %72 = getelementptr { ptr, i64, i64 }, ptr %70, i32 0, i32 1
  %73 = load i64, ptr %72, align 8
  %74 = add i64 %73, 1
  %75 = getelementptr { ptr, i64, i64 }, ptr %70, i32 0, i32 0
  %76 = getelementptr { ptr, i64, i64 }, ptr %70, i32 0, i32 2
  %77 = load ptr, ptr %75, align 8
  %78 = load i64, ptr %76, align 8
  %79 = icmp ugt i64 %74, %78
  br i1 %79, label %80, label %104

80:                                               ; preds = %71
  %81 = mul i64 %78, 2
  %82 = icmp uge i64 %81, %74
  %83 = select i1 %82, i64 %81, i64 %74
  %84 = icmp ult i64 %83, 4
  %85 = select i1 %84, i64 4, i64 %83
  %86 = icmp ule i64 %74, 4294967295
  br i1 %86, label %87, label %136

87:                                               ; preds = %80
  %88 = icmp ule i64 %85, 2305843009213693951
  br i1 %88, label %89, label %138

89:                                               ; preds = %87
  %90 = mul i64 8, %85
  %91 = add i64 %90, 8
  %92 = icmp eq ptr %77, null
  %93 = getelementptr i8, ptr %77, i64 -8
  %94 = select i1 %92, ptr null, ptr %93
  %95 = call ptr @GC_realloc(ptr %94, i64 %91)
  %96 = getelementptr i8, ptr %95, i64 8
  %97 = icmp ne ptr %96, null
  %98 = icmp eq i64 %90, 0
  %99 = or i1 %97, %98
  br i1 %99, label %100, label %140

100:                                              ; preds = %89
  %101 = mul i64 8, %78
  %102 = getelementptr i8, ptr %96, i64 %101
  %103 = sub i64 %90, %101
  call void @llvm.memset.p0.i64(ptr %102, i8 0, i64 %103, i1 false)
  store ptr %96, ptr %75, align 8
  store i64 %85, ptr %76, align 8
  br label %105

104:                                              ; preds = %71
  br label %105

105:                                              ; preds = %100, %104
  %106 = phi ptr [ %77, %104 ], [ %96, %100 ]
  br label %107

107:                                              ; preds = %105
  %108 = getelementptr ptr, ptr %106, i64 %73
  store ptr %61, ptr %108, align 8
  store i64 %74, ptr %72, align 8
  call void @churn()
  %109 = load ptr, ptr %1, align 8
  %110 = icmp eq ptr %109, null
  %111 = select i1 %110, ptr @__ts_empty_array, ptr %109
  %112 = getelementptr { ptr, i64, i64 }, ptr %111, i32 0, i32 0
  %113 = load ptr, ptr %112, align 8
  %114 = getelementptr ptr, ptr %113, i32 0
  %115 = load ptr, ptr %114, align 8
  %116 = getelementptr %Vec, ptr %115, i32 0, i32 1
  %117 = load double, ptr %116, align 8
  %118 = load ptr, ptr %1, align 8
  %119 = icmp eq ptr %118, null
  %120 = select i1 %119, ptr @__ts_empty_array, ptr %118
  %121 = getelementptr { ptr, i64, i64 }, ptr %120, i32 0, i32 0
  %122 = load ptr, ptr %121, align 8
  %123 = getelementptr ptr, ptr %122, i32 1
  %124 = load ptr, ptr %123, align 8
  %125 = getelementptr %Vec, ptr %124, i32 0, i32 1
  %126 = load double, ptr %125, align 8
  %127 = fadd double %117, %126
  store double %127, ptr %2, align 8
  %128 = load double, ptr %2, align 8
  ret double %128

129:                                              ; preds = %140, %138, %136, %134, %132, %130
  unreachable

130:                                              ; preds = %31
  %131 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 73, ptr null)
  br label %129

132:                                              ; preds = %38
  %133 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 73, ptr null)
  br label %129

134:                                              ; preds = %40
  %135 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 73, ptr null)
  br label %129

136:                                              ; preds = %80
  %137 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 74, ptr null)
  br label %129

138:                                              ; preds = %87
  %139 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 74, ptr null)
  br label %129

140:                                              ; preds = %89
  %141 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 74, ptr null)
  br label %129
}

; Function Attrs: mustprogress
define double @usedAsArgumentAndBound() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca <{ i64, %Vec }>, align 16
  %5 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Vec zeroinitializer, ptr %6, align 8
  store i64 -1, ptr %4, align 8
  %7 = getelementptr i8, ptr %4, i64 8
  store %Vec zeroinitializer, ptr %7, align 8
  store %Vec zeroinitializer, ptr %7, align 8
  %8 = getelementptr %Vec, ptr %7, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %8, align 8
  call void @Vec.constructor(ptr %7, double 6.000000e+00)
  store ptr %7, ptr %2, align 8
  %9 = load ptr, ptr %2, align 8
  store %Vec zeroinitializer, ptr %6, align 8
  %10 = getelementptr %Vec, ptr %6, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %10, align 8
  call void @Vec.constructor(ptr %6, double 1.000000e+00)
  %11 = call ptr @plus(ptr %9, ptr %6)
  store ptr %11, ptr %1, align 8
  call void @churn()
  %12 = load ptr, ptr %2, align 8
  %13 = getelementptr %Vec, ptr %12, i32 0, i32 1
  %14 = load double, ptr %13, align 8
  %15 = load ptr, ptr %1, align 8
  %16 = getelementptr %Vec, ptr %15, i32 0, i32 1
  %17 = load double, ptr %16, align 8
  %18 = fadd double %14, %17
  store double %18, ptr %3, align 8
  %19 = load double, ptr %3, align 8
  ret double %19
}

; Function Attrs: mustprogress
define ptr @plusAfterChurn(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  call void @churn()
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Vec, ptr %7, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %8, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr %Vec, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = load ptr, ptr %3, align 8
  %13 = getelementptr %Vec, ptr %12, i32 0, i32 1
  %14 = load double, ptr %13, align 8
  %15 = fadd double %11, %14
  call void @Vec.constructor(ptr %7, double %15)
  store ptr %7, ptr %5, align 8
  %16 = load ptr, ptr %5, align 8
  ret ptr %16
}

; Function Attrs: mustprogress
define double @argumentReadAfterCalleeAllocates() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca <{ i64, %Vec }>, align 16
  %4 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Vec zeroinitializer, ptr %5, align 8
  store i64 -1, ptr %3, align 8
  %6 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %6, align 8
  store %Vec zeroinitializer, ptr %6, align 8
  %7 = getelementptr %Vec, ptr %6, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %7, align 8
  call void @Vec.constructor(ptr %6, double 2.000000e+01)
  store %Vec zeroinitializer, ptr %5, align 8
  %8 = getelementptr %Vec, ptr %5, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %8, align 8
  call void @Vec.constructor(ptr %5, double 3.000000e+00)
  %9 = call ptr @plusAfterChurn(ptr %6, ptr %5)
  store ptr %9, ptr %1, align 8
  call void @churn()
  %10 = load ptr, ptr %1, align 8
  %11 = getelementptr %Vec, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  store double %12, ptr %2, align 8
  %13 = load double, ptr %2, align 8
  ret double %13
}

; Function Attrs: mustprogress
define double @discardedResult() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca <{ i64, %Vec }>, align 16
  %4 = alloca <{ i64, %Vec }>, align 16
  %5 = alloca <{ i64, %Vec }>, align 16
  %6 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %6, align 8
  %7 = getelementptr i8, ptr %6, i64 8
  store %Vec zeroinitializer, ptr %7, align 8
  store i64 -1, ptr %5, align 8
  %8 = getelementptr i8, ptr %5, i64 8
  store %Vec zeroinitializer, ptr %8, align 8
  store i64 -1, ptr %4, align 8
  %9 = getelementptr i8, ptr %4, i64 8
  store %Vec zeroinitializer, ptr %9, align 8
  store i64 -1, ptr %3, align 8
  %10 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %10, align 8
  store %Vec zeroinitializer, ptr %10, align 8
  %11 = getelementptr %Vec, ptr %10, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %11, align 8
  call void @Vec.constructor(ptr %10, double 1.200000e+01)
  store ptr %10, ptr %1, align 8
  store %Vec zeroinitializer, ptr %9, align 8
  %12 = getelementptr %Vec, ptr %9, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %12, align 8
  call void @Vec.constructor(ptr %9, double 4.000000e+00)
  %13 = call ptr @times(double 2.000000e+00, ptr %9)
  store %Vec zeroinitializer, ptr %8, align 8
  %14 = getelementptr %Vec, ptr %8, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %14, align 8
  call void @Vec.constructor(ptr %8, double 1.000000e+00)
  store %Vec zeroinitializer, ptr %7, align 8
  %15 = getelementptr %Vec, ptr %7, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %15, align 8
  call void @Vec.constructor(ptr %7, double 2.000000e+00)
  %16 = call ptr @plus(ptr %8, ptr %7)
  call void @churn()
  %17 = load ptr, ptr %1, align 8
  %18 = getelementptr %Vec, ptr %17, i32 0, i32 1
  %19 = load double, ptr %18, align 8
  store double %19, ptr %2, align 8
  %20 = load double, ptr %2, align 8
  ret double %20
}

; Function Attrs: mustprogress
define double @temporaryReadByCallee() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca <{ i64, %Vec }>, align 16
  %4 = alloca <{ i64, %Vec }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Vec zeroinitializer, ptr %5, align 8
  store i64 -1, ptr %3, align 8
  %6 = getelementptr i8, ptr %3, i64 8
  store %Vec zeroinitializer, ptr %6, align 8
  store %Vec zeroinitializer, ptr %6, align 8
  %7 = getelementptr %Vec, ptr %6, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %7, align 8
  call void @Vec.constructor(ptr %6, double 2.000000e+00)
  store %Vec zeroinitializer, ptr %5, align 8
  %8 = getelementptr %Vec, ptr %5, i32 0, i32 0
  store ptr @Vec..vtbl, ptr %8, align 8
  call void @Vec.constructor(ptr %5, double 2.000000e+00)
  %9 = call ptr @plus(ptr %6, ptr %5)
  %10 = call ptr @times(double 3.000000e+00, ptr %9)
  store ptr %10, ptr %1, align 8
  call void @churn()
  %11 = load ptr, ptr %1, align 8
  %12 = getelementptr %Vec, ptr %11, i32 0, i32 1
  %13 = load double, ptr %12, align 8
  store double %13, ptr %2, align 8
  %14 = load double, ptr %2, align 8
  ret double %14
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call double @nestedCallArguments()
  %2 = fcmp oeq double %1, 7.000000e+00
  br i1 %2, label %3, label %26

3:                                                ; preds = %0
  %4 = call double @deeperNesting()
  %5 = fcmp oeq double %4, 9.000000e+00
  br i1 %5, label %6, label %28

6:                                                ; preds = %3
  %7 = call double @temporariesInALoop()
  %8 = fcmp oeq double %7, 7.900000e+01
  br i1 %8, label %9, label %30

9:                                                ; preds = %6
  %10 = call double @temporaryKeptByCallee()
  %11 = fcmp oeq double %10, 1.700000e+01
  br i1 %11, label %12, label %32

12:                                               ; preds = %9
  %13 = call double @usedAsArgumentAndBound()
  %14 = fcmp oeq double %13, 1.300000e+01
  br i1 %14, label %15, label %34

15:                                               ; preds = %12
  %16 = call double @argumentReadAfterCalleeAllocates()
  %17 = fcmp oeq double %16, 2.300000e+01
  br i1 %17, label %18, label %36

18:                                               ; preds = %15
  %19 = call double @discardedResult()
  %20 = fcmp oeq double %19, 1.200000e+01
  br i1 %20, label %21, label %38

21:                                               ; preds = %18
  %22 = call double @temporaryReadByCallee()
  %23 = fcmp oeq double %22, 1.200000e+01
  br i1 %23, label %24, label %40

24:                                               ; preds = %21
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

25:                                               ; preds = %40, %38, %36, %34, %32, %30, %28, %26
  unreachable

26:                                               ; preds = %0
  %27 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10081758968786619921, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 126, ptr null)
  br label %25

28:                                               ; preds = %3
  %29 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16524708675501408318, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 127, ptr null)
  br label %25

30:                                               ; preds = %6
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8935485834063972330, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 128, ptr null)
  br label %25

32:                                               ; preds = %9
  %33 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5300301882725252942, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 129, ptr null)
  br label %25

34:                                               ; preds = %12
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8632710368751004317, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 130, ptr null)
  br label %25

36:                                               ; preds = %15
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_324297260880672661, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 131, ptr null)
  br label %25

38:                                               ; preds = %18
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5488436750520871312, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 132, ptr null)
  br label %25

40:                                               ; preds = %21
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5884349861029923589, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13523762721940311722, i64 8), i32 133, ptr null)
  br label %25
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
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

