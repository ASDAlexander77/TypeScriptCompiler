; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, double }
%H = type { ptr, ptr }
%Holder = type { ptr, ptr }

$V..size = comdat any

$H..size = comdat any

$Holder..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@s_11670276633975932207 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHolder\00", align 8
@s_416103670783521533 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnot moved\00", align 8
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
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_7204746640806189781 = internal constant [108 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_conditional_move.ts\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 16, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V.plus, ptr @V..rtti, ptr @V..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 16, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@Holder..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11670276633975932207, i64 8)
@Holder..size = global i64 16, comdat
@Holder..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Holder..instanceOf, ptr @Holder..new, ptr @Holder..rtti, ptr @Holder..size }
@finallies = private global double 0.000000e+00
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

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

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @V..instanceOf(ptr %0, ptr %1) #1 {
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
define ptr @V..new() #1 {
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
define void @V.constructor(ptr %0, double %1) #1 {
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
define ptr @V.plus(ptr %0, ptr %1) #1 {
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
define void @H.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 1
  store ptr null, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @H..instanceOf(ptr %0, ptr %1) #1 {
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
define ptr @H..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 0
  store ptr @H..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i1 @Holder..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Holder..rtti, align 8
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
define ptr @Holder..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Holder, ptr %3, i32 0, i32 0
  store ptr @Holder..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Holder.constructor(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Holder, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define double @keeps(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca <{ i64, %Holder }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Holder zeroinitializer, ptr %5, align 8
  store ptr %0, ptr %2, align 8
  store %Holder zeroinitializer, ptr %5, align 8
  %6 = getelementptr %Holder, ptr %5, i32 0, i32 0
  store ptr @Holder..vtbl, ptr %6, align 8
  %7 = load ptr, ptr %2, align 8
  call void @Holder.constructor(ptr %5, ptr %7)
  %8 = getelementptr %Holder, ptr %5, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr %V, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  store double %11, ptr %3, align 8
  %12 = load double, ptr %3, align 8
  ret double %12
}

; Function Attrs: mustprogress
define ptr @mkt(double %0) #1 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp ogt double %4, 5.000000e+00
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = load double, ptr %2, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 8)
  store double %7, ptr %8, align 8
  call void @__cxa_throw(ptr %8, ptr @_ZTId, ptr null)
  br label %15

9:                                                ; preds = %1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %V, ptr %11, i32 0, i32 0
  store ptr @V..vtbl, ptr %12, align 8
  %13 = load double, ptr %2, align 8
  call void @V.constructor(ptr %11, double %13)
  store ptr %11, ptr %3, align 8
  %14 = load ptr, ptr %3, align 8
  ret ptr %14

15:                                               ; preds = %6
  unreachable
}

; Function Attrs: mustprogress
define i1 @unwinds(i1 %0, double %1) #2 personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca i1, align 1
  %7 = alloca i1, align 1
  store i1 %0, ptr %6, align 1
  store double %1, ptr %5, align 8
  store ptr null, ptr %4, align 8
  %8 = call ptr @mkt(double 1.000000e+00)
  store ptr %8, ptr %3, align 8
  %9 = load i1, ptr %6, align 1
  br i1 %9, label %10, label %12

10:                                               ; preds = %2
  %11 = load ptr, ptr %3, align 8
  store ptr %11, ptr %4, align 8
  br label %12

12:                                               ; preds = %10, %2
  %13 = load double, ptr %5, align 8
  %14 = invoke ptr @mkt(double %13)
          to label %15 unwind label %22

15:                                               ; preds = %12
  %16 = load double, ptr @finallies, align 8
  %17 = fadd double %16, 1.000000e+00
  store double %17, ptr @finallies, align 8
  %18 = load ptr, ptr %4, align 8
  %19 = ptrtoint ptr %18 to i64
  %20 = icmp ne i64 %19, 0
  store i1 %20, ptr %7, align 1
  %21 = load i1, ptr %7, align 1
  ret i1 %21

22:                                               ; preds = %12
  %23 = landingpad { ptr, i32 }
          cleanup
  %24 = load double, ptr @finallies, align 8
  %25 = fadd double %24, 1.000000e+00
  store double %25, ptr @finallies, align 8
  resume { ptr, i32 } %23
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca double, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca double, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  %13 = alloca double, align 8
  %14 = alloca i32, align 4
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca ptr, align 8
  %19 = alloca i32, align 4
  %20 = alloca double, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca ptr, align 8
  %26 = alloca i32, align 4
  %27 = alloca i32, align 4
  %28 = alloca double, align 8
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %32 = alloca <{ i64, [3 x ptr] }>, align 16
  %33 = alloca <{ i64, %V }>, align 16
  %34 = alloca <{ i64, %V }>, align 16
  %35 = alloca <{ i64, %V }>, align 16
  %36 = alloca <{ i64, %V }>, align 16
  %37 = alloca <{ i64, %H }>, align 16
  %38 = alloca <{ i64, %V }>, align 16
  store i64 -1, ptr %38, align 8
  %39 = getelementptr i8, ptr %38, i64 8
  store %V zeroinitializer, ptr %39, align 8
  store i64 -1, ptr %37, align 8
  %40 = getelementptr i8, ptr %37, i64 8
  store %H zeroinitializer, ptr %40, align 8
  store i64 -1, ptr %36, align 8
  %41 = getelementptr i8, ptr %36, i64 8
  store %V zeroinitializer, ptr %41, align 8
  store i64 -1, ptr %35, align 8
  %42 = getelementptr i8, ptr %35, i64 8
  store %V zeroinitializer, ptr %42, align 8
  store i64 -1, ptr %34, align 8
  %43 = getelementptr i8, ptr %34, i64 8
  store %V zeroinitializer, ptr %43, align 8
  store i64 -1, ptr %33, align 8
  %44 = getelementptr i8, ptr %33, i64 8
  store %V zeroinitializer, ptr %44, align 8
  %45 = call ptr @GC_malloc(i64 24)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr %V, ptr %46, i32 0, i32 0
  store ptr @V..vtbl, ptr %47, align 8
  call void @V.constructor(ptr %46, double 1.000000e+00)
  %48 = call ptr @GC_malloc(i64 24)
  %49 = getelementptr i8, ptr %48, i64 8
  %50 = getelementptr %V, ptr %49, i32 0, i32 0
  store ptr @V..vtbl, ptr %50, align 8
  call void @V.constructor(ptr %49, double 3.000000e+00)
  %51 = call ptr @GC_malloc(i64 24)
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr %V, ptr %52, i32 0, i32 0
  store ptr @V..vtbl, ptr %53, align 8
  call void @V.constructor(ptr %52, double 2.000000e+00)
  store i64 -1, ptr %32, align 8
  %54 = getelementptr i8, ptr %32, i64 8
  store [3 x ptr] zeroinitializer, ptr %54, align 8
  %55 = getelementptr ptr, ptr %54, i64 0
  store ptr %46, ptr %55, align 8
  %56 = getelementptr ptr, ptr %54, i64 1
  store ptr %49, ptr %56, align 8
  %57 = getelementptr ptr, ptr %54, i64 2
  store ptr %52, ptr %57, align 8
  store i64 -1, ptr %31, align 8
  %58 = getelementptr i8, ptr %31, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %58, align 8
  %59 = getelementptr { ptr, i64, i64 }, ptr %58, i32 0, i32 0
  store ptr %54, ptr %59, align 8
  %60 = getelementptr { ptr, i64, i64 }, ptr %58, i32 0, i32 1
  store i64 3, ptr %60, align 8
  %61 = getelementptr { ptr, i64, i64 }, ptr %58, i32 0, i32 2
  store i64 3, ptr %61, align 8
  store ptr %58, ptr %30, align 8
  store ptr null, ptr %29, align 8
  store double 0.000000e+00, ptr %28, align 8
  store i32 0, ptr %27, align 4
  br label %62

62:                                               ; preds = %100, %0
  %63 = load i32, ptr %27, align 4
  %64 = load ptr, ptr %30, align 8
  %65 = icmp eq ptr %64, null
  %66 = select i1 %65, ptr @__ts_empty_array, ptr %64
  %67 = getelementptr { ptr, i64, i64 }, ptr %66, i32 0, i32 1
  %68 = load i64, ptr %67, align 8
  %69 = sext i32 %63 to i64
  %70 = icmp slt i64 %69, %68
  br i1 %70, label %71, label %103

71:                                               ; preds = %62
  %72 = load i32, ptr %27, align 4
  store i32 %72, ptr %26, align 4
  %73 = load ptr, ptr %30, align 8
  %74 = load i32, ptr %26, align 4
  %75 = icmp eq ptr %73, null
  %76 = select i1 %75, ptr @__ts_empty_array, ptr %73
  %77 = getelementptr { ptr, i64, i64 }, ptr %76, i32 0, i32 0
  %78 = load ptr, ptr %77, align 8
  %79 = getelementptr ptr, ptr %78, i32 %74
  %80 = load ptr, ptr %79, align 8
  %81 = load ptr, ptr %30, align 8
  %82 = load i32, ptr %26, align 4
  %83 = icmp eq ptr %81, null
  %84 = select i1 %83, ptr @__ts_empty_array, ptr %81
  %85 = getelementptr { ptr, i64, i64 }, ptr %84, i32 0, i32 0
  %86 = load ptr, ptr %85, align 8
  %87 = getelementptr ptr, ptr %86, i32 %82
  %88 = load ptr, ptr %87, align 8
  %89 = call ptr @V.plus(ptr %80, ptr %88)
  store ptr %89, ptr %25, align 8
  %90 = load ptr, ptr %25, align 8
  %91 = getelementptr %V, ptr %90, i32 0, i32 1
  %92 = load double, ptr %91, align 8
  %93 = load double, ptr %28, align 8
  %94 = fcmp ogt double %92, %93
  br i1 %94, label %95, label %100

95:                                               ; preds = %71
  %96 = load ptr, ptr %25, align 8
  store ptr %96, ptr %29, align 8
  %97 = load ptr, ptr %25, align 8
  %98 = getelementptr %V, ptr %97, i32 0, i32 1
  %99 = load double, ptr %98, align 8
  store double %99, ptr %28, align 8
  br label %100

100:                                              ; preds = %95, %71
  %101 = load i32, ptr %27, align 4
  %102 = add i32 %101, 1
  store i32 %102, ptr %27, align 4
  br label %62

103:                                              ; preds = %62
  %104 = load ptr, ptr %29, align 8
  %105 = getelementptr %V, ptr %104, i32 0, i32 1
  %106 = load double, ptr %105, align 8
  %107 = call ptr @GC_malloc_atomic(i64 40)
  %108 = getelementptr i8, ptr %107, i64 8
  call void @__ts_number_to_string(ptr %108, double %106)
  %109 = icmp eq ptr %108, null
  %110 = select i1 %109, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %108
  call void @puts(ptr %110)
  %111 = load ptr, ptr %29, align 8
  %112 = getelementptr %V, ptr %111, i32 0, i32 1
  %113 = load double, ptr %112, align 8
  %114 = fcmp oeq double %113, 6.000000e+00
  br i1 %114, label %115, label %462

115:                                              ; preds = %103
  store %V zeroinitializer, ptr %44, align 8
  %116 = getelementptr %V, ptr %44, i32 0, i32 0
  store ptr @V..vtbl, ptr %116, align 8
  call void @V.constructor(ptr %44, double 7.000000e+00)
  store ptr %44, ptr %24, align 8
  store ptr null, ptr %23, align 8
  %117 = load double, ptr %28, align 8
  %118 = fcmp ogt double %117, 5.000000e+00
  br i1 %118, label %119, label %121

119:                                              ; preds = %115
  %120 = load ptr, ptr %24, align 8
  store ptr %120, ptr %23, align 8
  br label %122

121:                                              ; preds = %115
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_416103670783521533, i64 8))
  br label %122

122:                                              ; preds = %119, %121
  %123 = load ptr, ptr %23, align 8
  %124 = getelementptr %V, ptr %123, i32 0, i32 1
  %125 = load double, ptr %124, align 8
  %126 = call ptr @GC_malloc_atomic(i64 40)
  %127 = getelementptr i8, ptr %126, i64 8
  call void @__ts_number_to_string(ptr %127, double %125)
  %128 = icmp eq ptr %127, null
  %129 = select i1 %128, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %127
  call void @puts(ptr %129)
  %130 = load ptr, ptr %23, align 8
  %131 = getelementptr %V, ptr %130, i32 0, i32 1
  %132 = load double, ptr %131, align 8
  %133 = fcmp oeq double %132, 7.000000e+00
  br i1 %133, label %134, label %464

134:                                              ; preds = %122
  store %V zeroinitializer, ptr %43, align 8
  %135 = getelementptr %V, ptr %43, i32 0, i32 0
  store ptr @V..vtbl, ptr %135, align 8
  call void @V.constructor(ptr %43, double 8.000000e+00)
  store ptr %43, ptr %22, align 8
  store ptr null, ptr %21, align 8
  %136 = load double, ptr %28, align 8
  %137 = fcmp ogt double %136, 1.000000e+02
  br i1 %137, label %138, label %140

138:                                              ; preds = %134
  %139 = load ptr, ptr %22, align 8
  store ptr %139, ptr %21, align 8
  br label %141

140:                                              ; preds = %134
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_416103670783521533, i64 8))
  br label %141

141:                                              ; preds = %138, %140
  %142 = load ptr, ptr %21, align 8
  %143 = ptrtoint ptr %142 to i64
  %144 = icmp eq i64 %143, 0
  br i1 %144, label %145, label %466

145:                                              ; preds = %141
  store double 0.000000e+00, ptr %20, align 8
  store i32 0, ptr %19, align 4
  br label %146

146:                                              ; preds = %179, %145
  %147 = load i32, ptr %19, align 4
  %148 = icmp slt i32 %147, 100000
  br i1 %148, label %149, label %182

149:                                              ; preds = %146
  store %V zeroinitializer, ptr %42, align 8
  %150 = getelementptr %V, ptr %42, i32 0, i32 0
  store ptr @V..vtbl, ptr %150, align 8
  %151 = load i32, ptr %19, align 4
  %152 = srem i32 %151, 3
  %153 = sitofp i32 %152 to double
  call void @V.constructor(ptr %42, double %153)
  store ptr %42, ptr %18, align 8
  store ptr null, ptr %17, align 8
  %154 = load double, ptr %20, align 8
  %155 = load ptr, ptr %18, align 8
  %156 = getelementptr %V, ptr %155, i32 0, i32 1
  %157 = load double, ptr %156, align 8
  %158 = fadd double %154, %157
  store double %158, ptr %20, align 8
  %159 = load i32, ptr %19, align 4
  %160 = srem i32 %159, 2
  %161 = icmp eq i32 %160, 0
  br i1 %161, label %162, label %169

162:                                              ; preds = %149
  %163 = load ptr, ptr %18, align 8
  store ptr %163, ptr %17, align 8
  %164 = load double, ptr %20, align 8
  %165 = load ptr, ptr %18, align 8
  %166 = getelementptr %V, ptr %165, i32 0, i32 1
  %167 = load double, ptr %166, align 8
  %168 = fadd double %164, %167
  store double %168, ptr %20, align 8
  br label %169

169:                                              ; preds = %162, %149
  %170 = load ptr, ptr %17, align 8
  %171 = ptrtoint ptr %170 to i64
  %172 = icmp ne i64 %171, 0
  br i1 %172, label %173, label %179

173:                                              ; preds = %169
  %174 = load double, ptr %20, align 8
  %175 = load ptr, ptr %17, align 8
  %176 = getelementptr %V, ptr %175, i32 0, i32 1
  %177 = load double, ptr %176, align 8
  %178 = fadd double %174, %177
  store double %178, ptr %20, align 8
  br label %179

179:                                              ; preds = %173, %169
  %180 = load i32, ptr %19, align 4
  %181 = add i32 %180, 1
  store i32 %181, ptr %19, align 4
  br label %146

182:                                              ; preds = %146
  %183 = load double, ptr %20, align 8
  %184 = call ptr @GC_malloc_atomic(i64 40)
  %185 = getelementptr i8, ptr %184, i64 8
  call void @__ts_number_to_string(ptr %185, double %183)
  %186 = icmp eq ptr %185, null
  %187 = select i1 %186, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %185
  call void @puts(ptr %187)
  %188 = load double, ptr %20, align 8
  %189 = fcmp oeq double %188, 1.999990e+05
  br i1 %189, label %190, label %468

190:                                              ; preds = %182
  store %V zeroinitializer, ptr %41, align 8
  %191 = getelementptr %V, ptr %41, i32 0, i32 0
  store ptr @V..vtbl, ptr %191, align 8
  call void @V.constructor(ptr %41, double 0.000000e+00)
  store ptr %41, ptr %16, align 8
  store ptr null, ptr %15, align 8
  store i32 0, ptr %14, align 4
  br label %192

192:                                              ; preds = %201, %190
  %193 = load i32, ptr %14, align 4
  %194 = icmp slt i32 %193, 1000
  br i1 %194, label %195, label %209

195:                                              ; preds = %192
  %196 = load i32, ptr %14, align 4
  %197 = srem i32 %196, 3
  %198 = icmp eq i32 %197, 0
  br i1 %198, label %199, label %201

199:                                              ; preds = %195
  %200 = load ptr, ptr %16, align 8
  store ptr %200, ptr %15, align 8
  br label %201

201:                                              ; preds = %199, %195
  %202 = call ptr @GC_malloc(i64 24)
  %203 = getelementptr i8, ptr %202, i64 8
  %204 = getelementptr %V, ptr %203, i32 0, i32 0
  store ptr @V..vtbl, ptr %204, align 8
  %205 = load i32, ptr %14, align 4
  %206 = sitofp i32 %205 to double
  call void @V.constructor(ptr %203, double %206)
  store ptr %203, ptr %16, align 8
  %207 = load i32, ptr %14, align 4
  %208 = add i32 %207, 1
  store i32 %208, ptr %14, align 4
  br label %192

209:                                              ; preds = %192
  %210 = load ptr, ptr %16, align 8
  %211 = getelementptr %V, ptr %210, i32 0, i32 1
  %212 = load double, ptr %211, align 8
  %213 = load ptr, ptr %15, align 8
  %214 = getelementptr %V, ptr %213, i32 0, i32 1
  %215 = load double, ptr %214, align 8
  %216 = call ptr @GC_malloc_atomic(i64 40)
  %217 = getelementptr i8, ptr %216, i64 8
  call void @__ts_number_to_string(ptr %217, double %212)
  %218 = call ptr @GC_malloc_atomic(i64 40)
  %219 = getelementptr i8, ptr %218, i64 8
  call void @__ts_number_to_string(ptr %219, double %215)
  %220 = call ptr @llvm.stacksave.p0()
  %221 = icmp eq ptr %217, null
  %222 = select i1 %221, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %217
  %223 = icmp eq ptr %219, null
  %224 = select i1 %223, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %219
  %225 = call i64 @strlen(ptr %222)
  %226 = add i64 1, %225
  %227 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %228 = add i64 %226, %227
  %229 = call i64 @strlen(ptr %224)
  %230 = add i64 %228, %229
  %231 = alloca i8, i64 %230, align 1
  %232 = getelementptr i8, ptr %231, i64 0
  %233 = sub i64 %230, 0
  %234 = call ptr @__memcpy_chk(ptr %232, ptr %222, i64 %225, i64 %233)
  %235 = add i64 0, %225
  %236 = getelementptr i8, ptr %231, i64 %235
  %237 = sub i64 %230, %235
  %238 = call ptr @__memcpy_chk(ptr %236, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %227, i64 %237)
  %239 = add i64 %235, %227
  %240 = getelementptr i8, ptr %231, i64 %239
  %241 = sub i64 %230, %239
  %242 = call ptr @__memcpy_chk(ptr %240, ptr %224, i64 %229, i64 %241)
  %243 = add i64 %239, %229
  %244 = getelementptr i8, ptr %231, i64 %243
  store i8 0, ptr %244, align 1
  call void @puts(ptr %231)
  call void @llvm.stackrestore.p0(ptr %220)
  %245 = load ptr, ptr %16, align 8
  %246 = getelementptr %V, ptr %245, i32 0, i32 1
  %247 = load double, ptr %246, align 8
  %248 = fcmp oeq double %247, 9.990000e+02
  br i1 %248, label %249, label %254

249:                                              ; preds = %209
  %250 = load ptr, ptr %15, align 8
  %251 = getelementptr %V, ptr %250, i32 0, i32 1
  %252 = load double, ptr %251, align 8
  %253 = fcmp oeq double %252, 9.980000e+02
  br label %254

254:                                              ; preds = %249, %209
  %255 = phi i1 [ %253, %249 ], [ %248, %209 ]
  br i1 %255, label %256, label %470

256:                                              ; preds = %254
  store double 0.000000e+00, ptr %13, align 8
  store i32 0, ptr %12, align 4
  br label %257

257:                                              ; preds = %287, %256
  %258 = load i32, ptr %12, align 4
  %259 = icmp slt i32 %258, 1000
  br i1 %259, label %260, label %290

260:                                              ; preds = %257
  store %H zeroinitializer, ptr %40, align 8
  %261 = getelementptr %H, ptr %40, i32 0, i32 0
  store ptr @H..vtbl, ptr %261, align 8
  call void @H.constructor(ptr %40)
  %262 = call ptr @GC_malloc(i64 24)
  %263 = getelementptr i8, ptr %262, i64 8
  %264 = getelementptr %V, ptr %263, i32 0, i32 0
  store ptr @V..vtbl, ptr %264, align 8
  %265 = load i32, ptr %12, align 4
  %266 = sitofp i32 %265 to double
  call void @V.constructor(ptr %263, double %266)
  store ptr %263, ptr %11, align 8
  %267 = getelementptr %H, ptr %40, i32 0, i32 1
  %268 = load ptr, ptr %11, align 8
  store ptr %268, ptr %267, align 8
  %269 = getelementptr %H, ptr %40, i32 0, i32 1
  %270 = load ptr, ptr %269, align 8
  %271 = getelementptr %V, ptr %270, i32 0, i32 1
  %272 = load double, ptr %271, align 8
  %273 = frem double %272, 2.000000e+00
  %274 = fcmp oeq double %273, 0.000000e+00
  br i1 %274, label %275, label %287

275:                                              ; preds = %260
  %276 = call ptr @GC_malloc(i64 24)
  %277 = getelementptr i8, ptr %276, i64 8
  %278 = getelementptr %V, ptr %277, i32 0, i32 0
  store ptr @V..vtbl, ptr %278, align 8
  %279 = load i32, ptr %12, align 4
  %280 = add i32 %279, 1
  %281 = sitofp i32 %280 to double
  call void @V.constructor(ptr %277, double %281)
  store ptr %277, ptr %11, align 8
  %282 = load double, ptr %13, align 8
  %283 = load ptr, ptr %11, align 8
  %284 = getelementptr %V, ptr %283, i32 0, i32 1
  %285 = load double, ptr %284, align 8
  %286 = fadd double %282, %285
  store double %286, ptr %13, align 8
  br label %287

287:                                              ; preds = %275, %260
  %288 = load i32, ptr %12, align 4
  %289 = add i32 %288, 1
  store i32 %289, ptr %12, align 4
  br label %257

290:                                              ; preds = %257
  %291 = load double, ptr %13, align 8
  %292 = call ptr @GC_malloc_atomic(i64 40)
  %293 = getelementptr i8, ptr %292, i64 8
  call void @__ts_number_to_string(ptr %293, double %291)
  %294 = icmp eq ptr %293, null
  %295 = select i1 %294, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %293
  call void @puts(ptr %295)
  %296 = load double, ptr %13, align 8
  %297 = fcmp oeq double %296, 2.500000e+05
  br i1 %297, label %298, label %472

298:                                              ; preds = %290
  store double 0.000000e+00, ptr %10, align 8
  store i32 0, ptr %9, align 4
  br label %299

299:                                              ; preds = %323, %298
  %300 = load i32, ptr %9, align 4
  %301 = icmp slt i32 %300, 1000
  br i1 %301, label %302, label %328

302:                                              ; preds = %299
  %303 = call ptr @GC_malloc(i64 24)
  %304 = getelementptr i8, ptr %303, i64 8
  %305 = getelementptr %V, ptr %304, i32 0, i32 0
  store ptr @V..vtbl, ptr %305, align 8
  %306 = load i32, ptr %9, align 4
  %307 = sitofp i32 %306 to double
  call void @V.constructor(ptr %304, double %307)
  store ptr %304, ptr %8, align 8
  store %V zeroinitializer, ptr %39, align 8
  %308 = getelementptr %V, ptr %39, i32 0, i32 0
  store ptr @V..vtbl, ptr %308, align 8
  %309 = load i32, ptr %9, align 4
  %310 = add i32 %309, 1
  %311 = sitofp i32 %310 to double
  call void @V.constructor(ptr %39, double %311)
  store ptr %39, ptr %7, align 8
  %312 = load double, ptr %10, align 8
  %313 = load i32, ptr %9, align 4
  %314 = srem i32 %313, 2
  %315 = icmp eq i32 %314, 0
  br i1 %315, label %316, label %319

316:                                              ; preds = %302
  %317 = load ptr, ptr %8, align 8
  %318 = call double @keeps(ptr %317)
  br label %323

319:                                              ; preds = %302
  %320 = load ptr, ptr %7, align 8
  %321 = getelementptr %V, ptr %320, i32 0, i32 1
  %322 = load double, ptr %321, align 8
  br label %323

323:                                              ; preds = %316, %319
  %324 = phi double [ %322, %319 ], [ %318, %316 ]
  %325 = fadd double %312, %324
  store double %325, ptr %10, align 8
  %326 = load i32, ptr %9, align 4
  %327 = add i32 %326, 1
  store i32 %327, ptr %9, align 4
  br label %299

328:                                              ; preds = %299
  %329 = load double, ptr %10, align 8
  %330 = call ptr @GC_malloc_atomic(i64 40)
  %331 = getelementptr i8, ptr %330, i64 8
  call void @__ts_number_to_string(ptr %331, double %329)
  %332 = icmp eq ptr %331, null
  %333 = select i1 %332, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %331
  call void @puts(ptr %333)
  %334 = load double, ptr %10, align 8
  %335 = fcmp oeq double %334, 5.000000e+05
  br i1 %335, label %336, label %474

336:                                              ; preds = %328
  store double 0.000000e+00, ptr %6, align 8
  store i32 0, ptr %5, align 4
  br label %337

337:                                              ; preds = %358, %336
  %338 = load i32, ptr %5, align 4
  %339 = icmp slt i32 %338, 1000
  br i1 %339, label %340, label %361

340:                                              ; preds = %337
  %341 = call ptr @GC_malloc(i64 24)
  %342 = getelementptr i8, ptr %341, i64 8
  %343 = getelementptr %V, ptr %342, i32 0, i32 0
  store ptr @V..vtbl, ptr %343, align 8
  %344 = load i32, ptr %5, align 4
  %345 = sitofp i32 %344 to double
  call void @V.constructor(ptr %342, double %345)
  store ptr %342, ptr %4, align 8
  %346 = load i32, ptr %5, align 4
  %347 = srem i32 %346, 2
  %348 = icmp eq i32 %347, 0
  br i1 %348, label %349, label %353

349:                                              ; preds = %340
  %350 = load ptr, ptr %4, align 8
  %351 = call double @keeps(ptr %350)
  %352 = fcmp oge double %351, 0.000000e+00
  br label %353

353:                                              ; preds = %349, %340
  %354 = phi i1 [ %352, %349 ], [ %348, %340 ]
  br i1 %354, label %355, label %358

355:                                              ; preds = %353
  %356 = load double, ptr %6, align 8
  %357 = fadd double %356, 1.000000e+00
  store double %357, ptr %6, align 8
  br label %358

358:                                              ; preds = %355, %353
  %359 = load i32, ptr %5, align 4
  %360 = add i32 %359, 1
  store i32 %360, ptr %5, align 4
  br label %337

361:                                              ; preds = %337
  %362 = load double, ptr %6, align 8
  %363 = call ptr @GC_malloc_atomic(i64 40)
  %364 = getelementptr i8, ptr %363, i64 8
  call void @__ts_number_to_string(ptr %364, double %362)
  %365 = icmp eq ptr %364, null
  %366 = select i1 %365, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %364
  call void @puts(ptr %366)
  %367 = load double, ptr %6, align 8
  %368 = fcmp oeq double %367, 5.000000e+02
  br i1 %368, label %369, label %476

369:                                              ; preds = %361
  store double 0.000000e+00, ptr %3, align 8
  store double 0.000000e+00, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %370

370:                                              ; preds = %396, %369
  %371 = load i32, ptr %1, align 4
  %372 = icmp slt i32 %371, 20000
  br i1 %372, label %373, label %399

373:                                              ; preds = %370
  %374 = load i32, ptr %1, align 4
  %375 = srem i32 %374, 2
  %376 = icmp eq i32 %375, 0
  %377 = load i32, ptr %1, align 4
  %378 = srem i32 %377, 4
  %379 = icmp slt i32 %378, 2
  br i1 %379, label %380, label %381

380:                                              ; preds = %373
  br label %382

381:                                              ; preds = %373
  br label %382

382:                                              ; preds = %380, %381
  %383 = phi i32 [ 9, %381 ], [ 1, %380 ]
  %384 = sitofp i32 %383 to double
  %385 = invoke i1 @unwinds(i1 %376, double %384)
          to label %386 unwind label %390

386:                                              ; preds = %382
  br i1 %385, label %387, label %396

387:                                              ; preds = %386
  %388 = load double, ptr %3, align 8
  %389 = fadd double %388, 1.000000e+00
  store double %389, ptr %3, align 8
  br label %396

390:                                              ; preds = %382
  %391 = landingpad { ptr, i32 }
          catch ptr null
  %392 = extractvalue { ptr, i32 } %391, 0
  %393 = call ptr @__cxa_begin_catch(ptr %392)
  %394 = load double, ptr %2, align 8
  %395 = fadd double %394, 1.000000e+00
  store double %395, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %396

396:                                              ; preds = %387, %386, %390
  %397 = load i32, ptr %1, align 4
  %398 = add i32 %397, 1
  store i32 %398, ptr %1, align 4
  br label %370

399:                                              ; preds = %370
  %400 = load double, ptr %3, align 8
  %401 = load double, ptr %2, align 8
  %402 = load double, ptr @finallies, align 8
  %403 = call ptr @GC_malloc_atomic(i64 40)
  %404 = getelementptr i8, ptr %403, i64 8
  call void @__ts_number_to_string(ptr %404, double %400)
  %405 = call ptr @GC_malloc_atomic(i64 40)
  %406 = getelementptr i8, ptr %405, i64 8
  call void @__ts_number_to_string(ptr %406, double %401)
  %407 = call ptr @GC_malloc_atomic(i64 40)
  %408 = getelementptr i8, ptr %407, i64 8
  call void @__ts_number_to_string(ptr %408, double %402)
  %409 = call ptr @llvm.stacksave.p0()
  %410 = icmp eq ptr %404, null
  %411 = select i1 %410, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %404
  %412 = icmp eq ptr %406, null
  %413 = select i1 %412, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %406
  %414 = icmp eq ptr %408, null
  %415 = select i1 %414, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %408
  %416 = call i64 @strlen(ptr %411)
  %417 = add i64 1, %416
  %418 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %419 = add i64 %417, %418
  %420 = call i64 @strlen(ptr %413)
  %421 = add i64 %419, %420
  %422 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %423 = add i64 %421, %422
  %424 = call i64 @strlen(ptr %415)
  %425 = add i64 %423, %424
  %426 = alloca i8, i64 %425, align 1
  %427 = getelementptr i8, ptr %426, i64 0
  %428 = sub i64 %425, 0
  %429 = call ptr @__memcpy_chk(ptr %427, ptr %411, i64 %416, i64 %428)
  %430 = add i64 0, %416
  %431 = getelementptr i8, ptr %426, i64 %430
  %432 = sub i64 %425, %430
  %433 = call ptr @__memcpy_chk(ptr %431, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %418, i64 %432)
  %434 = add i64 %430, %418
  %435 = getelementptr i8, ptr %426, i64 %434
  %436 = sub i64 %425, %434
  %437 = call ptr @__memcpy_chk(ptr %435, ptr %413, i64 %420, i64 %436)
  %438 = add i64 %434, %420
  %439 = getelementptr i8, ptr %426, i64 %438
  %440 = sub i64 %425, %438
  %441 = call ptr @__memcpy_chk(ptr %439, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %422, i64 %440)
  %442 = add i64 %438, %422
  %443 = getelementptr i8, ptr %426, i64 %442
  %444 = sub i64 %425, %442
  %445 = call ptr @__memcpy_chk(ptr %443, ptr %415, i64 %424, i64 %444)
  %446 = add i64 %442, %424
  %447 = getelementptr i8, ptr %426, i64 %446
  store i8 0, ptr %447, align 1
  call void @puts(ptr %426)
  call void @llvm.stackrestore.p0(ptr %409)
  %448 = load double, ptr %3, align 8
  %449 = fcmp oeq double %448, 5.000000e+03
  br i1 %449, label %450, label %453

450:                                              ; preds = %399
  %451 = load double, ptr %2, align 8
  %452 = fcmp oeq double %451, 1.000000e+04
  br label %453

453:                                              ; preds = %450, %399
  %454 = phi i1 [ %452, %450 ], [ %449, %399 ]
  br i1 %454, label %455, label %458

455:                                              ; preds = %453
  %456 = load double, ptr @finallies, align 8
  %457 = fcmp oeq double %456, 2.000000e+04
  br label %458

458:                                              ; preds = %455, %453
  %459 = phi i1 [ %457, %455 ], [ %454, %453 ]
  br i1 %459, label %460, label %478

460:                                              ; preds = %458
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

461:                                              ; preds = %478, %476, %474, %472, %470, %468, %466, %464, %462
  unreachable

462:                                              ; preds = %103
  %463 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 67, ptr null)
  br label %461

464:                                              ; preds = %122
  %465 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 79, ptr null)
  br label %461

466:                                              ; preds = %141
  %467 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 90, ptr null)
  br label %461

468:                                              ; preds = %182
  %469 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 109, ptr null)
  br label %461

470:                                              ; preds = %254
  %471 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 123, ptr null)
  br label %461

472:                                              ; preds = %290
  %473 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 139, ptr null)
  br label %461

474:                                              ; preds = %328
  %475 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 150, ptr null)
  br label %461

476:                                              ; preds = %361
  %477 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 163, ptr null)
  br label %461

478:                                              ; preds = %458
  %479 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7204746640806189781, i64 8), i32 178, ptr null)
  br label %461
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
declare double @llvm.fabs.f64(double) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #4

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

