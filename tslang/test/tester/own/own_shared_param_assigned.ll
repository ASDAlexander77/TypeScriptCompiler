; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, ptr, double }
%Junk = type { ptr, i32, i32, ptr }

$V..size = comdat any

$Junk..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_1560590634776335115 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFJunk\00", align 8
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
@s_16940945161124618027 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFjunk\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_8721998972162518564 = internal constant [113 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_shared_param_assigned.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@s_17226821747367941443 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassigned in a loop\00", align 8
@s_12217308626589154962 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassigned on one branch\00", align 8
@s_15927280465935436918 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassigned in a try\00", align 8
@s_464879406617847792 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassigned, then captured\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_15695622935707619001 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe caller's handle\00", align 8
@m_14838486305852456427 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFloop\00", align 8
@m_18256615127830573894 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbranch taken\00", align 8
@m_1186532226673667063 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbranch not taken\00", align 8
@m_6272037745333642222 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry\00", align 8
@m_16945420207626060738 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclosure\00", align 8
@m_15726944473330681470 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe caller's handle keeps its one count\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 24, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V..rtti, ptr @V..size }
@Junk..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1560590634776335115, i64 8)
@Junk..size = global i64 24, comdat
@Junk..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Junk..instanceOf, ptr @Junk..new, ptr @Junk..rtti, ptr @Junk..size }
@a_0_6706924739130738156 = internal constant [0 x ptr] undef
@td_2707451282177744033_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_2707451282177744033, ptr @tsretcb_2707451282177744033, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

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

define internal void @tsretcb_2707451282177744033(ptr %0) {
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

define internal void @tsrel_1375346448667278364(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %V, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrel_9169092472491273538(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  call void @tsrel_1375346448667278364(ptr %2)
  call void @__tslang_free_block(ptr %2)
  br label %7

7:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrelcb_2707451282177744033(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %14

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %14

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %13

10:                                               ; preds = %6
  %11 = call i1 @__tslang_dec_ref(ptr %8)
  br i1 %11, label %12, label %13

12:                                               ; preds = %10
  call void @tsrel_9169092472491273538(ptr %8)
  call void @__tslang_free_block(ptr %8)
  br label %13

13:                                               ; preds = %12, %10, %6
  call void @__tslang_free_block(ptr %2)
  br label %14

14:                                               ; preds = %13, %4, %1
  ret void
}

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
define void @Junk.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Junk, ptr %3, i32 0, i32 1
  store i32 -1, ptr %4, align 4
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Junk, ptr %5, i32 0, i32 2
  store i32 -2, ptr %6, align 4
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %Junk, ptr %7, i32 0, i32 3
  store ptr getelementptr inbounds nuw (i8, ptr @s_16940945161124618027, i64 8), ptr %8, align 8
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
  %2 = call ptr @GC_malloc(i64 32)
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
  %15 = call ptr @GC_malloc(i64 32)
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 21, ptr null)
  br label %73

76:                                               ; preds = %42
  %77 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 21, ptr null)
  br label %73

78:                                               ; preds = %44
  %79 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 21, ptr null)
  br label %73
}

; Function Attrs: mustprogress
define void @check(ptr %0, double %1, ptr %2) #2 {
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr %V, ptr %8, i32 0, i32 2
  %10 = load double, ptr %9, align 8
  %11 = load double, ptr %5, align 8
  %12 = fcmp oeq double %10, %11
  br i1 %12, label %13, label %55

13:                                               ; preds = %3
  %14 = load ptr, ptr %6, align 8
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr %V, ptr %15, i32 0, i32 1
  %17 = load ptr, ptr %16, align 8
  %18 = load double, ptr %5, align 8
  %19 = call ptr @GC_malloc_atomic(i64 40)
  %20 = getelementptr i8, ptr %19, i64 8
  call void @__ts_number_to_string(ptr %20, double %18)
  %21 = icmp eq ptr %20, null
  %22 = select i1 %21, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %20
  %23 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %24 = add i64 1, %23
  %25 = call i64 @strlen(ptr %22)
  %26 = add i64 %24, %25
  %27 = add i64 %26, 8
  %28 = call ptr @GC_malloc(i64 %27)
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr i8, ptr %29, i64 0
  %31 = sub i64 %26, 0
  %32 = call ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %23, i64 %31)
  %33 = add i64 0, %23
  %34 = getelementptr i8, ptr %29, i64 %33
  %35 = sub i64 %26, %33
  %36 = call ptr @__memcpy_chk(ptr %34, ptr %22, i64 %25, i64 %35)
  %37 = add i64 %33, %25
  %38 = getelementptr i8, ptr %29, i64 %37
  store i8 0, ptr %38, align 1
  %39 = ptrtoint ptr %17 to i64
  %40 = ptrtoint ptr %29 to i64
  %41 = icmp ne i64 %39, 0
  %42 = icmp ne i64 %40, 0
  %43 = zext i1 %41 to i32
  %44 = zext i1 %42 to i32
  %45 = and i32 %43, %44
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %50

47:                                               ; preds = %13
  %48 = call i32 @strcmp(ptr %17, ptr %29)
  %49 = icmp eq i32 %48, 0
  br label %52

50:                                               ; preds = %13
  %51 = icmp eq i64 %39, %40
  br label %52

52:                                               ; preds = %47, %50
  %53 = phi i1 [ %51, %50 ], [ %49, %47 ]
  br label %54

54:                                               ; preds = %52
  br label %55

55:                                               ; preds = %54, %3
  %56 = phi i1 [ %53, %54 ], [ %12, %3 ]
  %57 = load ptr, ptr %4, align 8
  br i1 %56, label %58, label %60

58:                                               ; preds = %55
  ret void

59:                                               ; preds = %60
  unreachable

60:                                               ; preds = %55
  %61 = call i32 @fflush(ptr null)
  %62 = icmp eq ptr %57, null
  %63 = select i1 %62, ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr %57
  call void @__assert_fail(ptr %63, ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 26, ptr null)
  br label %59
}

; Function Attrs: mustprogress
define double @inLoop(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store ptr %0, ptr %3, align 8
  store i32 0, ptr %2, align 4
  br label %5

5:                                                ; preds = %8, %1
  %6 = load i32, ptr %2, align 4
  %7 = icmp slt i32 %6, 3
  br i1 %7, label %8, label %19

8:                                                ; preds = %5
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %V, ptr %10, i32 0, i32 0
  store ptr @V..vtbl, ptr %11, align 8
  %12 = load i32, ptr %2, align 4
  %13 = add i32 %12, 10
  %14 = sitofp i32 %13 to double
  call void @V.constructor(ptr %10, double %14)
  %15 = call ptr @GC_malloc(i64 16)
  %16 = getelementptr i8, ptr %15, i64 8
  store ptr %10, ptr %16, align 8
  store ptr %16, ptr %3, align 8
  %17 = load i32, ptr %2, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %2, align 4
  br label %5

19:                                               ; preds = %5
  %20 = call i64 @churn()
  %21 = load ptr, ptr %3, align 8
  call void @check(ptr %21, double 1.200000e+01, ptr getelementptr inbounds nuw (i8, ptr @s_17226821747367941443, i64 8))
  %22 = load ptr, ptr %3, align 8
  %23 = load ptr, ptr %22, align 8
  %24 = getelementptr %V, ptr %23, i32 0, i32 2
  %25 = load double, ptr %24, align 8
  store double %25, ptr %4, align 8
  %26 = load double, ptr %4, align 8
  ret double %26
}

; Function Attrs: mustprogress
define double @onBranch(ptr %0, i1 %1) #2 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store i1 %1, ptr %3, align 1
  %6 = load i1, ptr %3, align 1
  br i1 %6, label %7, label %13

7:                                                ; preds = %2
  %8 = call ptr @GC_malloc(i64 32)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %V, ptr %9, i32 0, i32 0
  store ptr @V..vtbl, ptr %10, align 8
  call void @V.constructor(ptr %9, double 2.000000e+01)
  %11 = call ptr @GC_malloc(i64 16)
  %12 = getelementptr i8, ptr %11, i64 8
  store ptr %9, ptr %12, align 8
  store ptr %12, ptr %4, align 8
  br label %13

13:                                               ; preds = %7, %2
  %14 = call i64 @churn()
  %15 = load ptr, ptr %4, align 8
  %16 = load i1, ptr %3, align 1
  br i1 %16, label %17, label %18

17:                                               ; preds = %13
  br label %19

18:                                               ; preds = %13
  br label %19

19:                                               ; preds = %17, %18
  %20 = phi i32 [ 1, %18 ], [ 20, %17 ]
  %21 = sitofp i32 %20 to double
  call void @check(ptr %15, double %21, ptr getelementptr inbounds nuw (i8, ptr @s_12217308626589154962, i64 8))
  %22 = load ptr, ptr %4, align 8
  %23 = load ptr, ptr %22, align 8
  %24 = getelementptr %V, ptr %23, i32 0, i32 2
  %25 = load double, ptr %24, align 8
  store double %25, ptr %5, align 8
  %26 = load double, ptr %5, align 8
  ret double %26
}

; Function Attrs: mustprogress
define double @inTry(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %V, ptr %5, i32 0, i32 0
  store ptr @V..vtbl, ptr %6, align 8
  invoke void @V.constructor(ptr %5, double 3.000000e+01)
          to label %7 unwind label %17

7:                                                ; preds = %1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  store ptr %5, ptr %9, align 8
  store ptr %9, ptr %2, align 8
  %10 = call i64 @churn()
  %11 = load ptr, ptr %2, align 8
  call void @check(ptr %11, double 3.000000e+01, ptr getelementptr inbounds nuw (i8, ptr @s_15927280465935436918, i64 8))
  %12 = load ptr, ptr %2, align 8
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %V, ptr %13, i32 0, i32 2
  %15 = load double, ptr %14, align 8
  store double %15, ptr %3, align 8
  %16 = load double, ptr %3, align 8
  ret double %16

17:                                               ; preds = %1
  %18 = landingpad { ptr, i32 }
          cleanup
  %19 = call i64 @churn()
  resume { ptr, i32 } %18
}

; Function Attrs: mustprogress
define double @.f_captured..afL61C15FH1985836806(ptr %0) #2 {
  %2 = alloca double, align 8
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %5, align 8
  %7 = getelementptr %V, ptr %6, i32 0, i32 2
  %8 = load double, ptr %7, align 8
  store double %8, ptr %2, align 8
  %9 = load double, ptr %2, align 8
  ret double %9
}

; Function Attrs: mustprogress
define double @captured(ptr %0) #2 {
  %2 = alloca <{ i64, { ptr } }>, align 16
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  store ptr %0, ptr %6, align 8
  store i32 0, ptr %3, align 4
  br label %7

7:                                                ; preds = %10, %1
  %8 = load i32, ptr %3, align 4
  %9 = icmp slt i32 %8, 2
  br i1 %9, label %10, label %21

10:                                               ; preds = %7
  %11 = call ptr @GC_malloc(i64 32)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr %V, ptr %12, i32 0, i32 0
  store ptr @V..vtbl, ptr %13, align 8
  %14 = load i32, ptr %3, align 4
  %15 = add i32 %14, 40
  %16 = sitofp i32 %15 to double
  call void @V.constructor(ptr %12, double %16)
  %17 = call ptr @GC_malloc(i64 16)
  %18 = getelementptr i8, ptr %17, i64 8
  store ptr %12, ptr %18, align 8
  store ptr %18, ptr %6, align 8
  %19 = load i32, ptr %3, align 4
  %20 = add i32 %19, 1
  store i32 %20, ptr %3, align 4
  br label %7

21:                                               ; preds = %7
  store i64 -1, ptr %2, align 8
  %22 = getelementptr i8, ptr %2, i64 8
  store { ptr } zeroinitializer, ptr %22, align 8
  %23 = getelementptr { ptr }, ptr %22, i32 0, i32 0
  store ptr %6, ptr %23, align 8
  %24 = insertvalue { ptr, ptr, ptr } { ptr @.f_captured..afL61C15FH1985836806, ptr null, ptr null }, ptr %22, 1
  %25 = insertvalue { ptr, ptr, ptr } %24, ptr getelementptr inbounds nuw (i8, ptr @td_2707451282177744033_object, i64 40), 2
  %26 = call i64 @churn()
  %27 = load ptr, ptr %6, align 8
  call void @check(ptr %27, double 4.100000e+01, ptr getelementptr inbounds nuw (i8, ptr @s_464879406617847792, i64 8))
  %28 = call double @.f_captured..afL61C15FH1985836806(ptr %22)
  store double %28, ptr %4, align 8
  %29 = load double, ptr %4, align 8
  ret double %29
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %V, ptr %3, i32 0, i32 0
  store ptr @V..vtbl, ptr %4, align 8
  call void @V.constructor(ptr %3, double 1.000000e+00)
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  store ptr %3, ptr %6, align 8
  store ptr %6, ptr %1, align 8
  %7 = load ptr, ptr %1, align 8
  %8 = call double @inLoop(ptr %7)
  %9 = fcmp oeq double %8, 1.200000e+01
  br i1 %9, label %10, label %35

10:                                               ; preds = %0
  %11 = load ptr, ptr %1, align 8
  %12 = call double @onBranch(ptr %11, i1 true)
  %13 = fcmp oeq double %12, 2.000000e+01
  br i1 %13, label %14, label %37

14:                                               ; preds = %10
  %15 = load ptr, ptr %1, align 8
  %16 = call double @onBranch(ptr %15, i1 false)
  %17 = fcmp oeq double %16, 1.000000e+00
  br i1 %17, label %18, label %39

18:                                               ; preds = %14
  %19 = load ptr, ptr %1, align 8
  %20 = call double @inTry(ptr %19)
  %21 = fcmp oeq double %20, 3.000000e+01
  br i1 %21, label %22, label %41

22:                                               ; preds = %18
  %23 = load ptr, ptr %1, align 8
  %24 = call double @captured(ptr %23)
  %25 = fcmp oeq double %24, 4.100000e+01
  br i1 %25, label %26, label %43

26:                                               ; preds = %22
  %27 = call i64 @churn()
  %28 = load ptr, ptr %1, align 8
  call void @check(ptr %28, double 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @s_15695622935707619001, i64 8))
  %29 = load ptr, ptr %1, align 8
  br i1 true, label %31, label %30

30:                                               ; preds = %26
  br label %31

31:                                               ; preds = %30, %26
  %32 = phi i1 [ false, %30 ], [ true, %26 ]
  br i1 %32, label %33, label %45

33:                                               ; preds = %31
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

34:                                               ; preds = %45, %43, %41, %39, %37, %35
  unreachable

35:                                               ; preds = %0
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14838486305852456427, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 69, ptr null)
  br label %34

37:                                               ; preds = %10
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18256615127830573894, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 70, ptr null)
  br label %34

39:                                               ; preds = %14
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1186532226673667063, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 71, ptr null)
  br label %34

41:                                               ; preds = %18
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6272037745333642222, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 72, ptr null)
  br label %34

43:                                               ; preds = %22
  %44 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16945420207626060738, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 73, ptr null)
  br label %34

45:                                               ; preds = %31
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15726944473330681470, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8721998972162518564, i64 8), i32 77, ptr null)
  br label %34
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

