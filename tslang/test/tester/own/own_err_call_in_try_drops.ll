; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, double }
%H = type { ptr, ptr }

$C..size = comdat any

$H..size = comdat any

@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_10318347563643307040 = internal constant [113 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_err_call_in_try_drops.ts\00", align 8
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
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 16, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C..rtti, ptr @C..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 16, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_2926441800300339944 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

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
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 0
  store ptr @H..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10318347563643307040, i64 8), i32 14, ptr null)
  br label %66

69:                                               ; preds = %41
  %70 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10318347563643307040, i64 8), i32 14, ptr null)
  br label %66

71:                                               ; preds = %43
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10318347563643307040, i64 8), i32 14, ptr null)
  br label %66
}

; Function Attrs: mustprogress
define void @reset(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %C, ptr %6, i32 0, i32 0
  store ptr @C..vtbl, ptr %7, align 8
  call void @C.constructor(ptr %6, double 9.000000e+00)
  store ptr %6, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define void @read(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %H, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = load ptr, ptr %4, align 8
  invoke void @reset(ptr %8)
          to label %9 unwind label %10

9:                                                ; preds = %1
  br label %149

10:                                               ; preds = %1
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %10
  %15 = icmp eq ptr %14, @_ZTIi
  br i1 %15, label %16, label %23

16:                                               ; preds = %.noexc3
  %17 = load i32, ptr %13, align 4
  %18 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 0
  store i64 4, ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 2
  store i32 %17, ptr %22, align 4
  br label %146

23:                                               ; preds = %.noexc3
  %24 = icmp eq ptr %14, @_ZTIj
  br i1 %24, label %25, label %32

25:                                               ; preds = %23
  %26 = load i32, ptr %13, align 4
  %27 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %25
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 0
  store i64 4, ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 2
  store i32 %26, ptr %31, align 4
  br label %143

32:                                               ; preds = %23
  %33 = icmp eq ptr %14, @_ZTId
  br i1 %33, label %34, label %41

34:                                               ; preds = %32
  %35 = load double, ptr %13, align 8
  %36 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %34
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 0
  store i64 8, ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 2
  store double %35, ptr %40, align 8
  br label %140

41:                                               ; preds = %32
  %42 = icmp eq ptr %14, @_ZTIf
  br i1 %42, label %43, label %50

43:                                               ; preds = %41
  %44 = load float, ptr %13, align 4
  %45 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %43
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 0
  store i64 4, ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 2
  store float %44, ptr %49, align 4
  br label %137

50:                                               ; preds = %41
  %51 = icmp eq ptr %14, @_ZTIb
  br i1 %51, label %52, label %59

52:                                               ; preds = %50
  %53 = load i1, ptr %13, align 1
  %54 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 0
  store i64 1, ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 2
  store i1 %53, ptr %58, align 1
  br label %134

59:                                               ; preds = %50
  %60 = icmp eq ptr %14, @_ZTIx
  br i1 %60, label %61, label %68

61:                                               ; preds = %59
  %62 = load i64, ptr %13, align 8
  %63 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %61
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 0
  store i64 8, ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 2
  store i64 %62, ptr %67, align 8
  br label %131

68:                                               ; preds = %59
  %69 = icmp eq ptr %14, @_ZTIPc
  br i1 %69, label %70, label %77

70:                                               ; preds = %68
  %71 = load ptr, ptr %13, align 8
  %72 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %70
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 0
  store i64 8, ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 2
  store ptr %71, ptr %76, align 8
  br label %128

77:                                               ; preds = %68
  %78 = icmp eq ptr %14, @_ZTIPv
  br i1 %78, label %79, label %81

79:                                               ; preds = %77
  %80 = load ptr, ptr %13, align 8
  br label %125

81:                                               ; preds = %77
  %82 = load ptr, ptr %14, align 8
  %83 = icmp eq ptr %82, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %83, label %84, label %115

84:                                               ; preds = %81
  %85 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %14, i32 0, i32 2
  %86 = load i32, ptr %85, align 4
  %87 = and i32 %86, 1073741824
  %88 = icmp ne i32 %87, 0
  br i1 %88, label %89, label %105

89:                                               ; preds = %84
  %90 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %14, i32 0, i32 4
  %91 = load ptr, ptr %90, align 8
  %92 = icmp ne ptr %91, null
  br i1 %92, label %93, label %95

93:                                               ; preds = %89
  invoke void %91(ptr %2, ptr %13)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %93
  %94 = load ptr, ptr %2, align 8
  br label %102

95:                                               ; preds = %89
  %96 = load ptr, ptr %13, align 8
  %97 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %95
  %98 = getelementptr i8, ptr %97, i64 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 0
  store i64 8, ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 2
  store ptr %96, ptr %101, align 8
  br label %102

102:                                              ; preds = %.noexc4, %.noexc9
  %103 = phi ptr [ %98, %.noexc9 ], [ %94, %.noexc4 ]
  br label %104

104:                                              ; preds = %102
  br label %112

105:                                              ; preds = %84
  %106 = load ptr, ptr %13, align 8
  %107 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %105
  %108 = getelementptr i8, ptr %107, i64 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 0
  store i64 8, ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 2
  store ptr %106, ptr %111, align 8
  br label %112

112:                                              ; preds = %104, %.noexc5
  %113 = phi ptr [ %108, %.noexc5 ], [ %103, %104 ]
  br label %114

114:                                              ; preds = %112
  br label %122

115:                                              ; preds = %81
  %116 = load ptr, ptr %13, align 8
  %117 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %115
  %118 = getelementptr i8, ptr %117, i64 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 0
  store i64 8, ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 2
  store ptr %116, ptr %121, align 8
  br label %122

122:                                              ; preds = %114, %.noexc10
  %123 = phi ptr [ %118, %.noexc10 ], [ %113, %114 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %79, %124
  %126 = phi ptr [ %123, %124 ], [ %80, %79 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc7, %127
  %129 = phi ptr [ %126, %127 ], [ %73, %.noexc7 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc8, %130
  %132 = phi ptr [ %129, %130 ], [ %64, %.noexc8 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc11, %133
  %135 = phi ptr [ %132, %133 ], [ %55, %.noexc11 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc6, %136
  %138 = phi ptr [ %135, %136 ], [ %46, %.noexc6 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc2, %139
  %141 = phi ptr [ %138, %139 ], [ %37, %.noexc2 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc1, %142
  %144 = phi ptr [ %141, %142 ], [ %28, %.noexc1 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc, %145
  %147 = phi ptr [ %144, %145 ], [ %19, %.noexc ]
  br label %148

148:                                              ; preds = %146
  store ptr %147, ptr %3, align 8
  call void @__cxa_end_catch()
  br label %149

149:                                              ; preds = %9, %148
  call void @churn()
  %150 = getelementptr %C, ptr %7, i32 0, i32 1
  %151 = load double, ptr %150, align 8
  %152 = call ptr @GC_malloc_atomic(i64 40)
  %153 = getelementptr i8, ptr %152, i64 8
  call void @__ts_number_to_string(ptr %153, double %151)
  %154 = icmp eq ptr %153, null
  %155 = select i1 %154, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %153
  call void @puts(ptr %155)
  ret void

endcatch.resume:                                  ; preds = %52, %115, %95, %61, %70, %43, %105, %93, %10, %34, %25, %16
  %156 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %156
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %H, ptr %2, i32 0, i32 0
  store ptr @H..vtbl, ptr %3, align 8
  call void @H.constructor(ptr %2)
  call void @read(ptr %2)
  ret i32 0
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

