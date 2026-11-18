; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%H = type { ptr, ptr }
%C = type { ptr, i32, ptr, ptr }

$C..size = comdat any

$H..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_12638189399578898418 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc\00", align 8
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
@f_10689921613911257979 = internal constant [118 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00global_argument_reassigned.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_16940945161124618027 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFjunk\00", align 8
@s_2770870231077248403 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrown\00", align 8
@m_7368833367652526779 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa global, in a loop\00", align 8
@s_17761974936599606694 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc31\00", align 8
@m_8973867690343378340 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa later argument throws after the assignment\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_11652123548649057299 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa global given before an argument that assigns it\00", align 8
@m_18440882919609542205 = internal constant [66 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa global's field given before an argument that assigns it\00", align 8
@m_12599839212798849691 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe globals' last values\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 32, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C..rtti, ptr @C..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 16, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@G = private global ptr undef
@GH = private global ptr undef
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_17033125819989208466 = internal constant [0 x double] undef
@a_0_2926441800300339944 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @G__cctor, ptr null }, { i32, ptr, ptr } { i32 1001, ptr @GH__cctor, ptr null }]

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

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

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

define internal void @G__cctor() {
  %1 = call ptr @mk(double 1.000000e+00)
  store ptr %1, ptr @G, align 8
  ret void
}

define internal void @GH__cctor() {
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %H, ptr %2, i32 0, i32 0
  store ptr @H..vtbl, ptr %3, align 8
  call void @H.constructor(ptr %2)
  store ptr %2, ptr @GH, align 8
  ret void
}

; Function Attrs: mustprogress
define void @C.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %C, ptr %3, i32 0, i32 1
  store i32 5, ptr %4, align 4
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %C, ptr %5, i32 0, i32 2
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %C, ptr %7, i32 0, i32 3
  %9 = call ptr @GC_malloc(i64 8)
  %10 = getelementptr i8, ptr %9, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %10, ptr @a_0_17033125819989208466, i64 0, i1 false)
  %11 = call ptr @GC_malloc(i64 32)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 0
  store ptr %10, ptr %13, align 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 1
  store i64 0, ptr %14, align 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 2
  store i64 0, ptr %15, align 8
  store ptr %12, ptr %8, align 8
  ret void
}

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
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %C, ptr %3, i32 0, i32 0
  store ptr @C..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @H.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 1
  %5 = call ptr @mk(double 3.000000e+00)
  store ptr %5, ptr %4, align 8
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
define ptr @mk(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 40)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %C, ptr %5, i32 0, i32 0
  store ptr @C..vtbl, ptr %6, align 8
  call void @C.constructor(ptr %5)
  %7 = getelementptr %C, ptr %5, i32 0, i32 1
  %8 = load double, ptr %2, align 8
  %9 = fptosi double %8 to i64
  %10 = trunc i64 %9 to i32
  store i32 %10, ptr %7, align 4
  %11 = getelementptr %C, ptr %5, i32 0, i32 2
  %12 = load double, ptr %2, align 8
  %13 = call ptr @GC_malloc_atomic(i64 40)
  %14 = getelementptr i8, ptr %13, i64 8
  call void @__ts_number_to_string(ptr %14, double %12)
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr %16)
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %20, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr %16, i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  store ptr %23, ptr %11, align 8
  %33 = getelementptr %C, ptr %5, i32 0, i32 3
  %34 = load double, ptr %2, align 8
  %35 = load ptr, ptr %33, align 8
  %36 = icmp eq ptr %35, null
  br i1 %36, label %37, label %40

37:                                               ; preds = %1
  %38 = call ptr @GC_malloc(i64 32)
  %39 = getelementptr i8, ptr %38, i64 8
  store ptr %39, ptr %33, align 8
  br label %41

40:                                               ; preds = %1
  br label %41

41:                                               ; preds = %37, %40
  %42 = phi ptr [ %35, %40 ], [ %39, %37 ]
  br label %43

43:                                               ; preds = %41
  %44 = getelementptr { ptr, i64, i64 }, ptr %42, i32 0, i32 1
  %45 = load i64, ptr %44, align 8
  %46 = add i64 %45, 1
  %47 = getelementptr { ptr, i64, i64 }, ptr %42, i32 0, i32 0
  %48 = getelementptr { ptr, i64, i64 }, ptr %42, i32 0, i32 2
  %49 = load ptr, ptr %47, align 8
  %50 = load i64, ptr %48, align 8
  %51 = icmp ugt i64 %46, %50
  br i1 %51, label %52, label %76

52:                                               ; preds = %43
  %53 = mul i64 %50, 2
  %54 = icmp uge i64 %53, %46
  %55 = select i1 %54, i64 %53, i64 %46
  %56 = icmp ult i64 %55, 4
  %57 = select i1 %56, i64 4, i64 %55
  %58 = icmp ule i64 %46, 4294967295
  br i1 %58, label %59, label %83

59:                                               ; preds = %52
  %60 = icmp ule i64 %57, 2305843009213693951
  br i1 %60, label %61, label %85

61:                                               ; preds = %59
  %62 = mul i64 8, %57
  %63 = add i64 %62, 8
  %64 = icmp eq ptr %49, null
  %65 = getelementptr i8, ptr %49, i64 -8
  %66 = select i1 %64, ptr null, ptr %65
  %67 = call ptr @GC_realloc(ptr %66, i64 %63)
  %68 = getelementptr i8, ptr %67, i64 8
  %69 = icmp ne ptr %68, null
  %70 = icmp eq i64 %62, 0
  %71 = or i1 %69, %70
  br i1 %71, label %72, label %87

72:                                               ; preds = %61
  %73 = mul i64 8, %50
  %74 = getelementptr i8, ptr %68, i64 %73
  %75 = sub i64 %62, %73
  call void @llvm.memset.p0.i64(ptr %74, i8 0, i64 %75, i1 false)
  store ptr %68, ptr %47, align 8
  store i64 %57, ptr %48, align 8
  br label %77

76:                                               ; preds = %43
  br label %77

77:                                               ; preds = %72, %76
  %78 = phi ptr [ %49, %76 ], [ %68, %72 ]
  br label %79

79:                                               ; preds = %77
  %80 = getelementptr double, ptr %78, i64 %45
  store double %34, ptr %80, align 8
  store i64 %46, ptr %44, align 8
  store ptr %5, ptr %3, align 8
  %81 = load ptr, ptr %3, align 8
  ret ptr %81

82:                                               ; preds = %87, %85, %83
  unreachable

83:                                               ; preds = %52
  %84 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 22, ptr null)
  br label %82

85:                                               ; preds = %59
  %86 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 22, ptr null)
  br label %82

87:                                               ; preds = %61
  %88 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 22, ptr null)
  br label %82
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

10:                                               ; preds = %110, %0
  %11 = load i32, ptr %1, align 4
  %12 = icmp slt i32 %11, 64
  br i1 %12, label %13, label %114

13:                                               ; preds = %10
  %14 = call ptr @GC_malloc(i64 40)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %C, ptr %15, i32 0, i32 0
  store ptr @C..vtbl, ptr %16, align 8
  call void @C.constructor(ptr %15)
  %17 = getelementptr %C, ptr %15, i32 0, i32 1
  store i32 999, ptr %17, align 4
  %18 = getelementptr %C, ptr %15, i32 0, i32 2
  store ptr getelementptr inbounds nuw (i8, ptr @s_16940945161124618027, i64 8), ptr %18, align 8
  %19 = getelementptr %C, ptr %15, i32 0, i32 3
  %20 = load ptr, ptr %19, align 8
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %25

22:                                               ; preds = %13
  %23 = call ptr @GC_malloc(i64 32)
  %24 = getelementptr i8, ptr %23, i64 8
  store ptr %24, ptr %19, align 8
  br label %26

25:                                               ; preds = %13
  br label %26

26:                                               ; preds = %22, %25
  %27 = phi ptr [ %20, %25 ], [ %24, %22 ]
  br label %28

28:                                               ; preds = %26
  %29 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 1
  %30 = load i64, ptr %29, align 8
  %31 = add i64 %30, 1
  %32 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 0
  %33 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 2
  %34 = load ptr, ptr %32, align 8
  %35 = load i64, ptr %33, align 8
  %36 = icmp ugt i64 %31, %35
  br i1 %36, label %37, label %61

37:                                               ; preds = %28
  %38 = mul i64 %35, 2
  %39 = icmp uge i64 %38, %31
  %40 = select i1 %39, i64 %38, i64 %31
  %41 = icmp ult i64 %40, 4
  %42 = select i1 %41, i64 4, i64 %40
  %43 = icmp ule i64 %31, 4294967295
  br i1 %43, label %44, label %116

44:                                               ; preds = %37
  %45 = icmp ule i64 %42, 2305843009213693951
  br i1 %45, label %46, label %118

46:                                               ; preds = %44
  %47 = mul i64 8, %42
  %48 = add i64 %47, 8
  %49 = icmp eq ptr %34, null
  %50 = getelementptr i8, ptr %34, i64 -8
  %51 = select i1 %49, ptr null, ptr %50
  %52 = call ptr @GC_realloc(ptr %51, i64 %48)
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = icmp ne ptr %53, null
  %55 = icmp eq i64 %47, 0
  %56 = or i1 %54, %55
  br i1 %56, label %57, label %120

57:                                               ; preds = %46
  %58 = mul i64 8, %35
  %59 = getelementptr i8, ptr %53, i64 %58
  %60 = sub i64 %47, %58
  call void @llvm.memset.p0.i64(ptr %59, i8 0, i64 %60, i1 false)
  store ptr %53, ptr %32, align 8
  store i64 %42, ptr %33, align 8
  br label %62

61:                                               ; preds = %28
  br label %62

62:                                               ; preds = %57, %61
  %63 = phi ptr [ %34, %61 ], [ %53, %57 ]
  br label %64

64:                                               ; preds = %62
  %65 = getelementptr double, ptr %63, i64 %30
  store double 9.900000e+01, ptr %65, align 8
  store i64 %31, ptr %29, align 8
  %66 = load ptr, ptr %2, align 8
  %67 = icmp eq ptr %66, null
  br i1 %67, label %68, label %71

68:                                               ; preds = %64
  %69 = call ptr @GC_malloc(i64 32)
  %70 = getelementptr i8, ptr %69, i64 8
  store ptr %70, ptr %2, align 8
  br label %72

71:                                               ; preds = %64
  br label %72

72:                                               ; preds = %68, %71
  %73 = phi ptr [ %66, %71 ], [ %70, %68 ]
  br label %74

74:                                               ; preds = %72
  %75 = getelementptr { ptr, i64, i64 }, ptr %73, i32 0, i32 1
  %76 = load i64, ptr %75, align 8
  %77 = add i64 %76, 1
  %78 = getelementptr { ptr, i64, i64 }, ptr %73, i32 0, i32 0
  %79 = getelementptr { ptr, i64, i64 }, ptr %73, i32 0, i32 2
  %80 = load ptr, ptr %78, align 8
  %81 = load i64, ptr %79, align 8
  %82 = icmp ugt i64 %77, %81
  br i1 %82, label %83, label %107

83:                                               ; preds = %74
  %84 = mul i64 %81, 2
  %85 = icmp uge i64 %84, %77
  %86 = select i1 %85, i64 %84, i64 %77
  %87 = icmp ult i64 %86, 4
  %88 = select i1 %87, i64 4, i64 %86
  %89 = icmp ule i64 %77, 4294967295
  br i1 %89, label %90, label %122

90:                                               ; preds = %83
  %91 = icmp ule i64 %88, 2305843009213693951
  br i1 %91, label %92, label %124

92:                                               ; preds = %90
  %93 = mul i64 8, %88
  %94 = add i64 %93, 8
  %95 = icmp eq ptr %80, null
  %96 = getelementptr i8, ptr %80, i64 -8
  %97 = select i1 %95, ptr null, ptr %96
  %98 = call ptr @GC_realloc(ptr %97, i64 %94)
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = icmp ne ptr %99, null
  %101 = icmp eq i64 %93, 0
  %102 = or i1 %100, %101
  br i1 %102, label %103, label %126

103:                                              ; preds = %92
  %104 = mul i64 8, %81
  %105 = getelementptr i8, ptr %99, i64 %104
  %106 = sub i64 %93, %104
  call void @llvm.memset.p0.i64(ptr %105, i8 0, i64 %106, i1 false)
  store ptr %99, ptr %78, align 8
  store i64 %88, ptr %79, align 8
  br label %108

107:                                              ; preds = %74
  br label %108

108:                                              ; preds = %103, %107
  %109 = phi ptr [ %80, %107 ], [ %99, %103 ]
  br label %110

110:                                              ; preds = %108
  %111 = getelementptr ptr, ptr %109, i64 %76
  store ptr %15, ptr %111, align 8
  store i64 %77, ptr %75, align 8
  %112 = load i32, ptr %1, align 4
  %113 = add i32 %112, 1
  store i32 %113, ptr %1, align 4
  br label %10

114:                                              ; preds = %10
  ret void

115:                                              ; preds = %126, %124, %122, %120, %118, %116
  unreachable

116:                                              ; preds = %37
  %117 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 32, ptr null)
  br label %115

118:                                              ; preds = %44
  %119 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 32, ptr null)
  br label %115

120:                                              ; preds = %46
  %121 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 32, ptr null)
  br label %115

122:                                              ; preds = %83
  %123 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 33, ptr null)
  br label %115

124:                                              ; preds = %90
  %125 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 33, ptr null)
  br label %115

126:                                              ; preds = %92
  %127 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 33, ptr null)
  br label %115
}

; Function Attrs: mustprogress
define ptr @thrower() #2 {
  %1 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2770870231077248403, i64 8), ptr %1, align 8
  call void @__cxa_throw(ptr %1, ptr @_ZTIPc, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define i1 @check(ptr %0, ptr %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i1, align 1
  store ptr %0, ptr %6, align 8
  store ptr %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %8 = load double, ptr %4, align 8
  %9 = fadd double %8, 1.000000e+02
  %10 = call ptr @mk(double %9)
  store ptr %10, ptr @G, align 8
  %11 = load ptr, ptr @GH, align 8
  %12 = getelementptr %H, ptr %11, i32 0, i32 1
  %13 = load double, ptr %4, align 8
  %14 = fadd double %13, 2.000000e+02
  %15 = call ptr @mk(double %14)
  store ptr %15, ptr %12, align 8
  call void @churn()
  %16 = load ptr, ptr %6, align 8
  %17 = getelementptr %C, ptr %16, i32 0, i32 1
  %18 = load i32, ptr %17, align 4
  %19 = load double, ptr %4, align 8
  %20 = sitofp i32 %18 to double
  %21 = fcmp oeq double %20, %19
  br i1 %21, label %22, label %63

22:                                               ; preds = %3
  %23 = load ptr, ptr %6, align 8
  %24 = getelementptr %C, ptr %23, i32 0, i32 2
  %25 = load ptr, ptr %24, align 8
  %26 = load double, ptr %4, align 8
  %27 = call ptr @GC_malloc_atomic(i64 40)
  %28 = getelementptr i8, ptr %27, i64 8
  call void @__ts_number_to_string(ptr %28, double %26)
  %29 = icmp eq ptr %28, null
  %30 = select i1 %29, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %28
  %31 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %32 = add i64 1, %31
  %33 = call i64 @strlen(ptr %30)
  %34 = add i64 %32, %33
  %35 = add i64 %34, 8
  %36 = call ptr @GC_malloc(i64 %35)
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr i8, ptr %37, i64 0
  %39 = sub i64 %34, 0
  %40 = call ptr @__memcpy_chk(ptr %38, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %31, i64 %39)
  %41 = add i64 0, %31
  %42 = getelementptr i8, ptr %37, i64 %41
  %43 = sub i64 %34, %41
  %44 = call ptr @__memcpy_chk(ptr %42, ptr %30, i64 %33, i64 %43)
  %45 = add i64 %41, %33
  %46 = getelementptr i8, ptr %37, i64 %45
  store i8 0, ptr %46, align 1
  %47 = ptrtoint ptr %25 to i64
  %48 = ptrtoint ptr %37 to i64
  %49 = icmp ne i64 %47, 0
  %50 = icmp ne i64 %48, 0
  %51 = zext i1 %49 to i32
  %52 = zext i1 %50 to i32
  %53 = and i32 %51, %52
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %55, label %58

55:                                               ; preds = %22
  %56 = call i32 @strcmp(ptr %25, ptr %37)
  %57 = icmp eq i32 %56, 0
  br label %60

58:                                               ; preds = %22
  %59 = icmp eq i64 %47, %48
  br label %60

60:                                               ; preds = %55, %58
  %61 = phi i1 [ %59, %58 ], [ %57, %55 ]
  br label %62

62:                                               ; preds = %60
  br label %63

63:                                               ; preds = %62, %3
  %64 = phi i1 [ %61, %62 ], [ %21, %3 ]
  br i1 %64, label %65, label %77

65:                                               ; preds = %63
  %66 = load ptr, ptr %6, align 8
  %67 = getelementptr %C, ptr %66, i32 0, i32 3
  %68 = load ptr, ptr %67, align 8
  %69 = icmp eq ptr %68, null
  %70 = select i1 %69, ptr @__ts_empty_array, ptr %68
  %71 = getelementptr { ptr, i64, i64 }, ptr %70, i32 0, i32 0
  %72 = load ptr, ptr %71, align 8
  %73 = getelementptr double, ptr %72, i32 0
  %74 = load double, ptr %73, align 8
  %75 = load double, ptr %4, align 8
  %76 = fcmp oeq double %74, %75
  br label %77

77:                                               ; preds = %65, %63
  %78 = phi i1 [ %76, %65 ], [ %64, %63 ]
  br i1 %78, label %79, label %84

79:                                               ; preds = %77
  %80 = load ptr, ptr %5, align 8
  %81 = getelementptr %C, ptr %80, i32 0, i32 1
  %82 = load i32, ptr %81, align 4
  %83 = icmp sgt i32 %82, 0
  br label %84

84:                                               ; preds = %79, %77
  %85 = phi i1 [ %83, %79 ], [ %78, %77 ]
  store i1 %85, ptr %7, align 1
  %86 = load i1, ptr %7, align 1
  ret i1 %86
}

; Function Attrs: mustprogress
define void @inLoop() #2 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  br label %2

2:                                                ; preds = %17, %0
  %3 = load i32, ptr %1, align 4
  %4 = icmp slt i32 %3, 3
  br i1 %4, label %5, label %20

5:                                                ; preds = %2
  %6 = load ptr, ptr @G, align 8
  %7 = getelementptr %C, ptr %6, i32 0, i32 1
  %8 = load i32, ptr %7, align 4
  %9 = load ptr, ptr @G, align 8
  %10 = load i32, ptr %1, align 4
  %11 = add i32 %10, 10
  %12 = sitofp i32 %11 to double
  %13 = call ptr @mk(double %12)
  store ptr %13, ptr @G, align 8
  %14 = load ptr, ptr @G, align 8
  %15 = sitofp i32 %8 to double
  %16 = call i1 @check(ptr %9, ptr %14, double %15)
  br i1 %16, label %17, label %22

17:                                               ; preds = %5
  %18 = load i32, ptr %1, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %1, align 4
  br label %2

20:                                               ; preds = %2
  ret void

21:                                               ; preds = %22
  unreachable

22:                                               ; preds = %5
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7368833367652526779, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 55, ptr null)
  br label %21
}

; Function Attrs: mustprogress
define void @inTryCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = call ptr @mk(double 3.000000e+01)
  store ptr %4, ptr @G, align 8
  store i1 false, ptr %3, align 1
  %5 = load ptr, ptr @G, align 8
  %6 = invoke ptr @mk(double 3.100000e+01)
          to label %7 unwind label %12

7:                                                ; preds = %0
  store ptr %6, ptr @G, align 8
  %8 = invoke ptr @thrower()
          to label %9 unwind label %12

9:                                                ; preds = %7
  %10 = invoke i1 @check(ptr %5, ptr %8, double 3.000000e+01)
          to label %11 unwind label %12

11:                                               ; preds = %9
  br label %151

12:                                               ; preds = %9, %7, %0
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %12
  %17 = icmp eq ptr %16, @_ZTIi
  br i1 %17, label %18, label %25

18:                                               ; preds = %.noexc2
  %19 = load i32, ptr %15, align 4
  %20 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %18
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 0
  store i64 4, ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 2
  store i32 %19, ptr %24, align 4
  br label %148

25:                                               ; preds = %.noexc2
  %26 = icmp eq ptr %16, @_ZTIj
  br i1 %26, label %27, label %34

27:                                               ; preds = %25
  %28 = load i32, ptr %15, align 4
  %29 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %27
  %30 = getelementptr i8, ptr %29, i64 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 0
  store i64 4, ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %32, align 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 2
  store i32 %28, ptr %33, align 4
  br label %145

34:                                               ; preds = %25
  %35 = icmp eq ptr %16, @_ZTId
  br i1 %35, label %36, label %43

36:                                               ; preds = %34
  %37 = load double, ptr %15, align 8
  %38 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %36
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 0
  store i64 8, ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %41, align 8
  %42 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 2
  store double %37, ptr %42, align 8
  br label %142

43:                                               ; preds = %34
  %44 = icmp eq ptr %16, @_ZTIf
  br i1 %44, label %45, label %52

45:                                               ; preds = %43
  %46 = load float, ptr %15, align 4
  %47 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %45
  %48 = getelementptr i8, ptr %47, i64 8
  %49 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 0
  store i64 4, ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %50, align 8
  %51 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 2
  store float %46, ptr %51, align 4
  br label %139

52:                                               ; preds = %43
  %53 = icmp eq ptr %16, @_ZTIb
  br i1 %53, label %54, label %61

54:                                               ; preds = %52
  %55 = load i1, ptr %15, align 1
  %56 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %54
  %57 = getelementptr i8, ptr %56, i64 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 0
  store i64 1, ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %59, align 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 2
  store i1 %55, ptr %60, align 1
  br label %136

61:                                               ; preds = %52
  %62 = icmp eq ptr %16, @_ZTIx
  br i1 %62, label %63, label %70

63:                                               ; preds = %61
  %64 = load i64, ptr %15, align 8
  %65 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %63
  %66 = getelementptr i8, ptr %65, i64 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 0
  store i64 8, ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %68, align 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 2
  store i64 %64, ptr %69, align 8
  br label %133

70:                                               ; preds = %61
  %71 = icmp eq ptr %16, @_ZTIPc
  br i1 %71, label %72, label %79

72:                                               ; preds = %70
  %73 = load ptr, ptr %15, align 8
  %74 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %72
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 0
  store i64 8, ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %77, align 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 2
  store ptr %73, ptr %78, align 8
  br label %130

79:                                               ; preds = %70
  %80 = icmp eq ptr %16, @_ZTIPv
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = load ptr, ptr %15, align 8
  br label %127

83:                                               ; preds = %79
  %84 = load ptr, ptr %16, align 8
  %85 = icmp eq ptr %84, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %85, label %86, label %117

86:                                               ; preds = %83
  %87 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %16, i32 0, i32 2
  %88 = load i32, ptr %87, align 4
  %89 = and i32 %88, 1073741824
  %90 = icmp ne i32 %89, 0
  br i1 %90, label %91, label %107

91:                                               ; preds = %86
  %92 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %16, i32 0, i32 4
  %93 = load ptr, ptr %92, align 8
  %94 = icmp ne ptr %93, null
  br i1 %94, label %95, label %97

95:                                               ; preds = %91
  invoke void %93(ptr %1, ptr %15)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %95
  %96 = load ptr, ptr %1, align 8
  br label %104

97:                                               ; preds = %91
  %98 = load ptr, ptr %15, align 8
  %99 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %97
  %100 = getelementptr i8, ptr %99, i64 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 0
  store i64 8, ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 2
  store ptr %98, ptr %103, align 8
  br label %104

104:                                              ; preds = %.noexc4, %.noexc11
  %105 = phi ptr [ %100, %.noexc11 ], [ %96, %.noexc4 ]
  br label %106

106:                                              ; preds = %104
  br label %114

107:                                              ; preds = %86
  %108 = load ptr, ptr %15, align 8
  %109 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %107
  %110 = getelementptr i8, ptr %109, i64 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 0
  store i64 8, ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 2
  store ptr %108, ptr %113, align 8
  br label %114

114:                                              ; preds = %106, %.noexc9
  %115 = phi ptr [ %110, %.noexc9 ], [ %105, %106 ]
  br label %116

116:                                              ; preds = %114
  br label %124

117:                                              ; preds = %83
  %118 = load ptr, ptr %15, align 8
  %119 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %117
  %120 = getelementptr i8, ptr %119, i64 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 0
  store i64 8, ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 2
  store ptr %118, ptr %123, align 8
  br label %124

124:                                              ; preds = %116, %.noexc
  %125 = phi ptr [ %120, %.noexc ], [ %115, %116 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %81, %126
  %128 = phi ptr [ %125, %126 ], [ %82, %81 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc7, %129
  %131 = phi ptr [ %128, %129 ], [ %75, %.noexc7 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc6, %132
  %134 = phi ptr [ %131, %132 ], [ %66, %.noexc6 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc5, %135
  %137 = phi ptr [ %134, %135 ], [ %57, %.noexc5 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc10, %138
  %140 = phi ptr [ %137, %138 ], [ %48, %.noexc10 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc8, %141
  %143 = phi ptr [ %140, %141 ], [ %39, %.noexc8 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc3, %144
  %146 = phi ptr [ %143, %144 ], [ %30, %.noexc3 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc1, %147
  %149 = phi ptr [ %146, %147 ], [ %21, %.noexc1 ]
  br label %150

150:                                              ; preds = %148
  store ptr %149, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %151

151:                                              ; preds = %11, %150
  call void @churn()
  %152 = load i1, ptr %3, align 1
  br i1 %152, label %153, label %158

153:                                              ; preds = %151
  %154 = load ptr, ptr @G, align 8
  %155 = getelementptr %C, ptr %154, i32 0, i32 1
  %156 = load i32, ptr %155, align 4
  %157 = icmp eq i32 %156, 31
  br label %158

158:                                              ; preds = %153, %151
  %159 = phi i1 [ %157, %153 ], [ %152, %151 ]
  br i1 %159, label %160, label %177

160:                                              ; preds = %158
  %161 = load ptr, ptr @G, align 8
  %162 = getelementptr %C, ptr %161, i32 0, i32 2
  %163 = load ptr, ptr %162, align 8
  %164 = ptrtoint ptr %163 to i64
  %165 = icmp ne i64 %164, 0
  %166 = zext i1 %165 to i32
  %167 = and i32 %166, 1
  %168 = icmp ne i32 %167, 0
  br i1 %168, label %169, label %172

169:                                              ; preds = %160
  %170 = call i32 @strcmp(ptr %163, ptr getelementptr inbounds nuw (i8, ptr @s_17761974936599606694, i64 8))
  %171 = icmp eq i32 %170, 0
  br label %174

172:                                              ; preds = %160
  %173 = icmp eq i64 %164, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17761974936599606694, i64 8) to i64)
  br label %174

174:                                              ; preds = %169, %172
  %175 = phi i1 [ %173, %172 ], [ %171, %169 ]
  br label %176

176:                                              ; preds = %174
  br label %177

177:                                              ; preds = %176, %158
  %178 = phi i1 [ %175, %176 ], [ %159, %158 ]
  br i1 %178, label %179, label %181

179:                                              ; preds = %177
  ret void

180:                                              ; preds = %181
  unreachable

181:                                              ; preds = %177
  %182 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8973867690343378340, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 69, ptr null)
  br label %180

endcatch.resume:                                  ; preds = %97, %45, %107, %36, %72, %63, %54, %95, %27, %12, %18, %117
  %183 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %183
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = load ptr, ptr @G, align 8
  %2 = call ptr @mk(double 2.000000e+00)
  store ptr %2, ptr @G, align 8
  %3 = load ptr, ptr @G, align 8
  %4 = call i1 @check(ptr %1, ptr %3, double 1.000000e+00)
  br i1 %4, label %5, label %38

5:                                                ; preds = %0
  %6 = load ptr, ptr @GH, align 8
  %7 = getelementptr %H, ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr %C, ptr %8, i32 0, i32 1
  %10 = load i32, ptr %9, align 4
  %11 = load ptr, ptr @GH, align 8
  %12 = getelementptr %H, ptr %11, i32 0, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = load ptr, ptr @GH, align 8
  %15 = getelementptr %H, ptr %14, i32 0, i32 1
  %16 = call ptr @mk(double 4.000000e+00)
  store ptr %16, ptr %15, align 8
  %17 = load ptr, ptr @GH, align 8
  %18 = getelementptr %H, ptr %17, i32 0, i32 1
  %19 = load ptr, ptr %18, align 8
  %20 = sitofp i32 %10 to double
  %21 = call i1 @check(ptr %13, ptr %19, double %20)
  br i1 %21, label %22, label %40

22:                                               ; preds = %5
  call void @inLoop()
  call void @inTryCatch()
  call void @churn()
  %23 = load ptr, ptr @G, align 8
  %24 = getelementptr %C, ptr %23, i32 0, i32 1
  %25 = load i32, ptr %24, align 4
  %26 = icmp eq i32 %25, 31
  br i1 %26, label %27, label %34

27:                                               ; preds = %22
  %28 = load ptr, ptr @GH, align 8
  %29 = getelementptr %H, ptr %28, i32 0, i32 1
  %30 = load ptr, ptr %29, align 8
  %31 = getelementptr %C, ptr %30, i32 0, i32 1
  %32 = load i32, ptr %31, align 4
  %33 = icmp eq i32 %32, 701
  br label %34

34:                                               ; preds = %27, %22
  %35 = phi i1 [ %33, %27 ], [ %26, %22 ]
  br i1 %35, label %36, label %42

36:                                               ; preds = %34
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

37:                                               ; preds = %42, %40, %38
  unreachable

38:                                               ; preds = %0
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11652123548649057299, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 73, ptr null)
  br label %37

40:                                               ; preds = %5
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18440882919609542205, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 75, ptr null)
  br label %37

42:                                               ; preds = %34
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12599839212798849691, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10689921613911257979, i64 8), i32 79, ptr null)
  br label %37
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

