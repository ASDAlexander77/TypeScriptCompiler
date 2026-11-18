; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, ptr, double }
%N = type { ptr, ptr }
%Junk = type { ptr, i32, i32, ptr }

$V..size = comdat any

$N..size = comdat any

$Junk..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638230081509142225 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFN\00", align 8
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
@f_7662718949508697475 = internal constant [109 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00const_global_read.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_6716854176737176485 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa const read of a global, reassigned here\00", align 8
@m_4956652289863503429 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa const read of a global, reassigned in a call\00", align 8
@m_2100142264122129301 = internal constant [76 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field of a const read of a global, lent to a call that assigns it\00", align 8
@m_15037607468427923800 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFread in a finally\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@m_4173897229357701040 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan interface and a bound method over a global\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_12599839212798849691 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe globals' last values\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 24, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V.get, ptr @V..rtti, ptr @V..size }
@N..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638230081509142225, i64 8)
@N..size = global i64 16, comdat
@N..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @N..instanceOf, ptr @N..new, ptr @N..rtti, ptr @N..size }
@Junk..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1560590634776335115, i64 8)
@Junk..size = global i64 24, comdat
@Junk..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Junk..instanceOf, ptr @Junk..new, ptr @Junk..rtti, ptr @Junk..size }
@G = private global ptr undef
@GV = private global ptr undef
@V.IV..vtbl = private global { ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_1375346448667278364_class, i64 40), ptr @V.IV..vtbl..instanceOf.dispatch, ptr @V.IV..vtbl.get.dispatch, ptr inttoptr (i64 16 to ptr) }
@td_1375346448667278364_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_1375346448667278364, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@_ZTIPc = external constant ptr
@IV.null..vtbl = private global { ptr, ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @IV.null.get, ptr null }
@a_0_6706924739130738156 = internal constant [0 x ptr] undef
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @G__cctor, ptr null }, { i32, ptr, ptr } { i32 1001, ptr @GV__cctor, ptr null }]

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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

define internal void @G__cctor() {
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %N, ptr %2, i32 0, i32 0
  store ptr @N..vtbl, ptr %3, align 8
  call void @N.constructor(ptr %2, double 1.000000e+00)
  store ptr %2, ptr @G, align 8
  ret void
}

define internal void @GV__cctor() {
  %1 = call ptr @GC_malloc(i64 32)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %V, ptr %2, i32 0, i32 0
  store ptr @V..vtbl, ptr %3, align 8
  call void @V.constructor(ptr %2, double 1.000000e+00)
  store ptr %2, ptr @GV, align 8
  ret void
}

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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 37, ptr null)
  br label %73

76:                                               ; preds = %42
  %77 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 37, ptr null)
  br label %73

78:                                               ; preds = %44
  %79 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 37, ptr null)
  br label %73
}

; Function Attrs: mustprogress
define void @assignG(double %0) #2 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = call ptr @GC_malloc(i64 24)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %N, ptr %4, i32 0, i32 0
  store ptr @N..vtbl, ptr %5, align 8
  %6 = load double, ptr %2, align 8
  call void @N.constructor(ptr %4, double %6)
  store ptr %4, ptr @G, align 8
  %7 = call i64 @churn()
  ret void
}

; Function Attrs: mustprogress
define i1 @run(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  %7 = fadd double %6, 1.000000e+02
  call void @assignG(double %7)
  %8 = load ptr, ptr %4, align 8
  %9 = getelementptr %V, ptr %8, i32 0, i32 2
  %10 = load double, ptr %9, align 8
  %11 = load double, ptr %3, align 8
  %12 = fcmp oeq double %10, %11
  br i1 %12, label %13, label %54

13:                                               ; preds = %2
  %14 = load ptr, ptr %4, align 8
  %15 = getelementptr %V, ptr %14, i32 0, i32 1
  %16 = load ptr, ptr %15, align 8
  %17 = load double, ptr %3, align 8
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

46:                                               ; preds = %13
  %47 = call i32 @strcmp(ptr %16, ptr %28)
  %48 = icmp eq i32 %47, 0
  br label %51

49:                                               ; preds = %13
  %50 = icmp eq i64 %38, %39
  br label %51

51:                                               ; preds = %46, %49
  %52 = phi i1 [ %50, %49 ], [ %48, %46 ]
  br label %53

53:                                               ; preds = %51
  br label %54

54:                                               ; preds = %53, %2
  %55 = phi i1 [ %52, %53 ], [ %12, %2 ]
  store i1 %55, ptr %5, align 1
  %56 = load i1, ptr %5, align 1
  ret i1 %56
}

; Function Attrs: mustprogress
define void @reassignedHere(double %0) #2 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = call ptr @GC_malloc(i64 24)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %N, ptr %4, i32 0, i32 0
  store ptr @N..vtbl, ptr %5, align 8
  %6 = load double, ptr %2, align 8
  call void @N.constructor(ptr %4, double %6)
  store ptr %4, ptr @G, align 8
  %7 = load ptr, ptr @G, align 8
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %N, ptr %9, i32 0, i32 0
  store ptr @N..vtbl, ptr %10, align 8
  call void @N.constructor(ptr %9, double 9.000000e+00)
  store ptr %9, ptr @G, align 8
  %11 = call i64 @churn()
  %12 = getelementptr %N, ptr %7, i32 0, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %V, ptr %13, i32 0, i32 2
  %15 = load double, ptr %14, align 8
  %16 = load double, ptr %2, align 8
  %17 = fcmp oeq double %15, %16
  br i1 %17, label %18, label %60

18:                                               ; preds = %1
  %19 = getelementptr %N, ptr %7, i32 0, i32 1
  %20 = load ptr, ptr %19, align 8
  %21 = getelementptr %V, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %21, align 8
  %23 = load double, ptr %2, align 8
  %24 = call ptr @GC_malloc_atomic(i64 40)
  %25 = getelementptr i8, ptr %24, i64 8
  call void @__ts_number_to_string(ptr %25, double %23)
  %26 = icmp eq ptr %25, null
  %27 = select i1 %26, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %25
  %28 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %29 = add i64 1, %28
  %30 = call i64 @strlen(ptr %27)
  %31 = add i64 %29, %30
  %32 = add i64 %31, 8
  %33 = call ptr @GC_malloc(i64 %32)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr i8, ptr %34, i64 0
  %36 = sub i64 %31, 0
  %37 = call ptr @__memcpy_chk(ptr %35, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %28, i64 %36)
  %38 = add i64 0, %28
  %39 = getelementptr i8, ptr %34, i64 %38
  %40 = sub i64 %31, %38
  %41 = call ptr @__memcpy_chk(ptr %39, ptr %27, i64 %30, i64 %40)
  %42 = add i64 %38, %30
  %43 = getelementptr i8, ptr %34, i64 %42
  store i8 0, ptr %43, align 1
  %44 = ptrtoint ptr %22 to i64
  %45 = ptrtoint ptr %34 to i64
  %46 = icmp ne i64 %44, 0
  %47 = icmp ne i64 %45, 0
  %48 = zext i1 %46 to i32
  %49 = zext i1 %47 to i32
  %50 = and i32 %48, %49
  %51 = icmp ne i32 %50, 0
  br i1 %51, label %52, label %55

52:                                               ; preds = %18
  %53 = call i32 @strcmp(ptr %22, ptr %34)
  %54 = icmp eq i32 %53, 0
  br label %57

55:                                               ; preds = %18
  %56 = icmp eq i64 %44, %45
  br label %57

57:                                               ; preds = %52, %55
  %58 = phi i1 [ %56, %55 ], [ %54, %52 ]
  br label %59

59:                                               ; preds = %57
  br label %60

60:                                               ; preds = %59, %1
  %61 = phi i1 [ %58, %59 ], [ %17, %1 ]
  br i1 %61, label %62, label %64

62:                                               ; preds = %60
  ret void

63:                                               ; preds = %64
  unreachable

64:                                               ; preds = %60
  %65 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6716854176737176485, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 60, ptr null)
  br label %63
}

; Function Attrs: mustprogress
define void @reassignedInCall(double %0) #2 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = call ptr @GC_malloc(i64 24)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %N, ptr %4, i32 0, i32 0
  store ptr @N..vtbl, ptr %5, align 8
  %6 = load double, ptr %2, align 8
  call void @N.constructor(ptr %4, double %6)
  store ptr %4, ptr @G, align 8
  %7 = load ptr, ptr @G, align 8
  call void @assignG(double 9.000000e+00)
  %8 = getelementptr %N, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr %V, ptr %9, i32 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = load double, ptr %2, align 8
  %13 = fcmp oeq double %11, %12
  br i1 %13, label %14, label %21

14:                                               ; preds = %1
  %15 = getelementptr %N, ptr %7, i32 0, i32 1
  %16 = load ptr, ptr %15, align 8
  %17 = getelementptr %V, ptr %16, i32 0, i32 0
  %18 = call double @V.get(ptr %16)
  %19 = load double, ptr %2, align 8
  %20 = fcmp oeq double %18, %19
  br label %21

21:                                               ; preds = %14, %1
  %22 = phi i1 [ %20, %14 ], [ %13, %1 ]
  br i1 %22, label %23, label %25

23:                                               ; preds = %21
  ret void

24:                                               ; preds = %25
  unreachable

25:                                               ; preds = %21
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4956652289863503429, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 67, ptr null)
  br label %24
}

; Function Attrs: mustprogress
define void @lentFromConst(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca double, align 8
  store double %0, ptr %3, align 8
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 0
  store ptr @N..vtbl, ptr %6, align 8
  %7 = load double, ptr %3, align 8
  call void @N.constructor(ptr %5, double %7)
  store ptr %5, ptr @G, align 8
  %8 = load ptr, ptr @G, align 8
  store i32 0, ptr %2, align 4
  br label %9

9:                                                ; preds = %61, %1
  %10 = load i32, ptr %2, align 4
  %11 = icmp slt i32 %10, 3
  br i1 %11, label %12, label %108

12:                                               ; preds = %9
  %13 = getelementptr %N, ptr %8, i32 0, i32 1
  %14 = load ptr, ptr %13, align 8
  %15 = load double, ptr %3, align 8
  %16 = call i1 @run(ptr %14, double %15)
  br i1 %16, label %17, label %110

17:                                               ; preds = %12
  %18 = invoke i64 @churn()
          to label %19 unwind label %64

19:                                               ; preds = %17
  %20 = getelementptr %N, ptr %8, i32 0, i32 1
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr %V, ptr %21, i32 0, i32 1
  %23 = load ptr, ptr %22, align 8
  %24 = load double, ptr %3, align 8
  %25 = call ptr @GC_malloc_atomic(i64 40)
  %26 = getelementptr i8, ptr %25, i64 8
  call void @__ts_number_to_string(ptr %26, double %24)
  %27 = icmp eq ptr %26, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %26
  %29 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr %28)
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %28, i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  %45 = ptrtoint ptr %23 to i64
  %46 = ptrtoint ptr %35 to i64
  %47 = icmp ne i64 %45, 0
  %48 = icmp ne i64 %46, 0
  %49 = zext i1 %47 to i32
  %50 = zext i1 %48 to i32
  %51 = and i32 %49, %50
  %52 = icmp ne i32 %51, 0
  br i1 %52, label %53, label %56

53:                                               ; preds = %19
  %54 = call i32 @strcmp(ptr %23, ptr %35)
  %55 = icmp eq i32 %54, 0
  br label %58

56:                                               ; preds = %19
  %57 = icmp eq i64 %45, %46
  br label %58

58:                                               ; preds = %53, %56
  %59 = phi i1 [ %57, %56 ], [ %55, %53 ]
  br label %60

60:                                               ; preds = %58
  br i1 %59, label %61, label %112

61:                                               ; preds = %60
  %62 = load i32, ptr %2, align 4
  %63 = add i32 %62, 1
  store i32 %63, ptr %2, align 4
  br label %9

64:                                               ; preds = %17
  %65 = landingpad { ptr, i32 }
          cleanup
  %66 = getelementptr %N, ptr %8, i32 0, i32 1
  %67 = load ptr, ptr %66, align 8
  %68 = getelementptr %V, ptr %67, i32 0, i32 1
  %69 = load ptr, ptr %68, align 8
  %70 = load double, ptr %3, align 8
  %71 = call ptr @GC_malloc_atomic(i64 40)
  %72 = getelementptr i8, ptr %71, i64 8
  call void @__ts_number_to_string(ptr %72, double %70)
  %73 = icmp eq ptr %72, null
  %74 = select i1 %73, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %72
  %75 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %76 = add i64 1, %75
  %77 = call i64 @strlen(ptr %74)
  %78 = add i64 %76, %77
  %79 = add i64 %78, 8
  %80 = call ptr @GC_malloc(i64 %79)
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr i8, ptr %81, i64 0
  %83 = sub i64 %78, 0
  %84 = call ptr @__memcpy_chk(ptr %82, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %75, i64 %83)
  %85 = add i64 0, %75
  %86 = getelementptr i8, ptr %81, i64 %85
  %87 = sub i64 %78, %85
  %88 = call ptr @__memcpy_chk(ptr %86, ptr %74, i64 %77, i64 %87)
  %89 = add i64 %85, %77
  %90 = getelementptr i8, ptr %81, i64 %89
  store i8 0, ptr %90, align 1
  %91 = ptrtoint ptr %69 to i64
  %92 = ptrtoint ptr %81 to i64
  %93 = icmp ne i64 %91, 0
  %94 = icmp ne i64 %92, 0
  %95 = zext i1 %93 to i32
  %96 = zext i1 %94 to i32
  %97 = and i32 %95, %96
  %98 = icmp ne i32 %97, 0
  br i1 %98, label %99, label %102

99:                                               ; preds = %64
  %100 = call i32 @strcmp(ptr %69, ptr %81)
  %101 = icmp eq i32 %100, 0
  br label %104

102:                                              ; preds = %64
  %103 = icmp eq i64 %91, %92
  br label %104

104:                                              ; preds = %99, %102
  %105 = phi i1 [ %103, %102 ], [ %101, %99 ]
  br label %106

106:                                              ; preds = %104
  br i1 %105, label %107, label %114

107:                                              ; preds = %106
  resume { ptr, i32 } %65

108:                                              ; preds = %9
  ret void

109:                                              ; preds = %112, %114, %110
  unreachable

110:                                              ; preds = %12
  %111 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2100142264122129301, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 74, ptr null)
  br label %109

112:                                              ; preds = %60
  %113 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15037607468427923800, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 78, ptr null)
  br label %109

114:                                              ; preds = %106
  %115 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15037607468427923800, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 78, ptr null)
  br label %109
}

; Function Attrs: mustprogress
define internal i1 @V.IV..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  %4 = getelementptr %V, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = call i1 %7(ptr %0, ptr %1)
  store i1 %8, ptr %3, align 1
  %9 = load i1, ptr %3, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define internal double @V.IV..vtbl.get.dispatch(ptr %0) #2 {
  %2 = alloca double, align 8
  %3 = getelementptr %V, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr ptr, ptr %4, i32 2
  %6 = load ptr, ptr %5, align 8
  %7 = call double %6(ptr %0)
  store double %7, ptr %2, align 8
  %8 = load double, ptr %2, align 8
  ret double %8
}

; Function Attrs: mustprogress
define double @IV.null.get(ptr %0) #4 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define void @views(double %0) #2 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = call ptr @GC_malloc(i64 32)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %V, ptr %4, i32 0, i32 0
  store ptr @V..vtbl, ptr %5, align 8
  %6 = load double, ptr %2, align 8
  call void @V.constructor(ptr %4, double %6)
  store ptr %4, ptr @GV, align 8
  %7 = load ptr, ptr @GV, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  br i1 %9, label %10, label %11

10:                                               ; preds = %1
  br label %12

11:                                               ; preds = %1
  br label %12

12:                                               ; preds = %10, %11
  %13 = phi ptr [ @IV.null..vtbl, %11 ], [ @V.IV..vtbl, %10 ]
  %14 = insertvalue { ptr, ptr } undef, ptr %13, 0
  %15 = insertvalue { ptr, ptr } %14, ptr %7, 1
  %16 = load ptr, ptr @GV, align 8
  %17 = getelementptr %V, ptr %16, i32 0, i32 0
  %18 = call ptr @GC_malloc(i64 32)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr %V, ptr %19, i32 0, i32 0
  store ptr @V..vtbl, ptr %20, align 8
  call void @V.constructor(ptr %19, double 9.000000e+00)
  store ptr %19, ptr @GV, align 8
  %21 = call i64 @churn()
  %22 = extractvalue { ptr, ptr } %15, 0
  %23 = extractvalue { ptr, ptr } %15, 1
  %24 = getelementptr ptr, ptr %22, i32 3
  %25 = load ptr, ptr %24, align 8
  %26 = ptrtoint ptr %23 to i64
  %27 = ptrtoint ptr %25 to i64
  %28 = add i64 %26, %27
  %29 = inttoptr i64 %28 to ptr
  %30 = extractvalue { ptr, ptr } %15, 0
  %31 = extractvalue { ptr, ptr } %15, 1
  %32 = getelementptr ptr, ptr %30, i32 3
  %33 = load ptr, ptr %32, align 8
  %34 = ptrtoint ptr %33 to i64
  %35 = and i64 %34, -9223372036854775808
  %36 = icmp ne i64 %35, 0
  br i1 %36, label %37, label %43

37:                                               ; preds = %12
  %38 = sub i64 %34, -9223372036854775808
  %39 = inttoptr i64 %38 to ptr
  %40 = getelementptr { ptr, ptr }, ptr %39, i32 0, i32 0
  %41 = load ptr, ptr %40, align 8
  %42 = call double %41(ptr %31)
  br label %45

43:                                               ; preds = %12
  %44 = load double, ptr %29, align 8
  br label %45

45:                                               ; preds = %37, %43
  %46 = phi double [ %44, %43 ], [ %42, %37 ]
  br label %47

47:                                               ; preds = %45
  %48 = load double, ptr %2, align 8
  %49 = fcmp oeq double %46, %48
  br i1 %49, label %50, label %63

50:                                               ; preds = %47
  %51 = extractvalue { ptr, ptr } %15, 0
  %52 = extractvalue { ptr, ptr } %15, 1
  %53 = getelementptr ptr, ptr %51, i32 2
  %54 = load ptr, ptr %53, align 8
  %55 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %54, 0
  %56 = insertvalue { ptr, ptr, ptr } %55, ptr %52, 1
  %57 = insertvalue { ptr, ptr, ptr } %56, ptr null, 2
  %58 = extractvalue { ptr, ptr, ptr } %57, 1
  %59 = extractvalue { ptr, ptr, ptr } %57, 0
  %60 = call double %59(ptr %58)
  %61 = load double, ptr %2, align 8
  %62 = fcmp oeq double %60, %61
  br label %63

63:                                               ; preds = %50, %47
  %64 = phi i1 [ %62, %50 ], [ %49, %47 ]
  br i1 %64, label %65, label %69

65:                                               ; preds = %63
  %66 = call double @V.get(ptr %16)
  %67 = load double, ptr %2, align 8
  %68 = fcmp oeq double %66, %67
  br label %69

69:                                               ; preds = %65, %63
  %70 = phi i1 [ %68, %65 ], [ %64, %63 ]
  br i1 %70, label %71, label %73

71:                                               ; preds = %69
  ret void

72:                                               ; preds = %73
  unreachable

73:                                               ; preds = %69
  %74 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4173897229357701040, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 89, ptr null)
  br label %72
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  store i32 2, ptr %1, align 4
  br label %2

2:                                                ; preds = %5, %0
  %3 = load i32, ptr %1, align 4
  %4 = icmp slt i32 %3, 5
  br i1 %4, label %5, label %16

5:                                                ; preds = %2
  %6 = load i32, ptr %1, align 4
  %7 = sitofp i32 %6 to double
  call void @reassignedHere(double %7)
  %8 = load i32, ptr %1, align 4
  %9 = sitofp i32 %8 to double
  call void @reassignedInCall(double %9)
  %10 = load i32, ptr %1, align 4
  %11 = sitofp i32 %10 to double
  call void @lentFromConst(double %11)
  %12 = load i32, ptr %1, align 4
  %13 = sitofp i32 %12 to double
  call void @views(double %13)
  %14 = load i32, ptr %1, align 4
  %15 = add i32 %14, 1
  store i32 %15, ptr %1, align 4
  br label %2

16:                                               ; preds = %2
  %17 = call i64 @churn()
  %18 = load ptr, ptr @G, align 8
  %19 = getelementptr %N, ptr %18, i32 0, i32 1
  %20 = load ptr, ptr %19, align 8
  %21 = getelementptr %V, ptr %20, i32 0, i32 2
  %22 = load double, ptr %21, align 8
  %23 = fcmp ogt double %22, 1.000000e+02
  br i1 %23, label %24, label %29

24:                                               ; preds = %16
  %25 = load ptr, ptr @GV, align 8
  %26 = getelementptr %V, ptr %25, i32 0, i32 2
  %27 = load double, ptr %26, align 8
  %28 = fcmp oeq double %27, 9.000000e+00
  br label %29

29:                                               ; preds = %24, %16
  %30 = phi i1 [ %28, %24 ], [ %23, %16 ]
  br i1 %30, label %31, label %33

31:                                               ; preds = %29
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

32:                                               ; preds = %33
  unreachable

33:                                               ; preds = %29
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12599839212798849691, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7662718949508697475, i64 8), i32 101, ptr null)
  br label %32
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
declare double @llvm.fabs.f64(double) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { mustprogress "__null_interface_vtable" }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

