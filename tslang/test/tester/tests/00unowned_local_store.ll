; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, ptr, double }
%N = type { ptr, ptr }

$V..size = comdat any

$N..size = comdat any

@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638230081509142225 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFN\00", align 8
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
@f_3009451683414766259 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00unowned_local_store.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_634745896800656653 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv6\00", align 8
@s_634742598265772020 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv5\00", align 8
@s_7521248980907757358 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv12\00", align 8
@m_11184076928100809650 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local of a finally, assigned in a loop\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_10615361117518154669 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local with no initializer, assigned in an if\00", align 8
@m_8158167497407629090 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local with no initializer, assigned in a loop\00", align 8
@m_6963701317841836948 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local of a finally\00", align 8
@m_2589278777131849494 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local of a catch clause, assigned in an if\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 24, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V..rtti, ptr @V..size }
@N..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638230081509142225, i64 8)
@N..size = global i64 16, comdat
@N..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @N..instanceOf, ptr @N..new, ptr @N..rtti, ptr @N..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@IX.12252034448159921135..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12252034448159921135_object, i64 40), ptr @.instanceOf.none, ptr null }
@td_12252034448159921135_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrel_12252034448159921135, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@a_0_14389619779565911031 = internal constant [0 x ptr] undef
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

declare void @puts(ptr)

declare void @__cxa_end_catch()

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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

define internal void @tsrel_12252034448159921135(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_1375346448667278364(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
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
define ptr @mk(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %N, ptr %5, i32 0, i32 0
  store ptr @N..vtbl, ptr %6, align 8
  %7 = load double, ptr %2, align 8
  call void @N.constructor(ptr %5, double %7)
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
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_14389619779565911031, i64 0, i1 false)
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
  %16 = getelementptr %N, ptr %15, i32 0, i32 0
  store ptr @N..vtbl, ptr %16, align 8
  call void @N.constructor(ptr %15, double 9.990000e+02)
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 32, ptr null)
  br label %66

69:                                               ; preds = %41
  %70 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 32, ptr null)
  br label %66

71:                                               ; preds = %43
  %72 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 32, ptr null)
  br label %66
}

; Function Attrs: mustprogress
define i1 @noInitializer(i1 %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca i1, align 1
  store i1 %0, ptr %3, align 1
  %5 = load i1, ptr %3, align 1
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = call ptr @mk(double 5.000000e+00)
  store ptr %7, ptr %2, align 8
  br label %10

8:                                                ; preds = %1
  %9 = call ptr @mk(double 6.000000e+00)
  store ptr %9, ptr %2, align 8
  br label %10

10:                                               ; preds = %6, %8
  call void @churn()
  %11 = load ptr, ptr %2, align 8
  %12 = getelementptr %N, ptr %11, i32 0, i32 1
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr %V, ptr %13, i32 0, i32 2
  %15 = load double, ptr %14, align 8
  %16 = load i1, ptr %3, align 1
  br i1 %16, label %17, label %18

17:                                               ; preds = %10
  br label %19

18:                                               ; preds = %10
  br label %19

19:                                               ; preds = %17, %18
  %20 = phi i32 [ 6, %18 ], [ 5, %17 ]
  %21 = sitofp i32 %20 to double
  %22 = fcmp oeq double %15, %21
  br i1 %22, label %23, label %50

23:                                               ; preds = %19
  %24 = load ptr, ptr %2, align 8
  %25 = getelementptr %N, ptr %24, i32 0, i32 1
  %26 = load ptr, ptr %25, align 8
  %27 = getelementptr %V, ptr %26, i32 0, i32 1
  %28 = load ptr, ptr %27, align 8
  %29 = load i1, ptr %3, align 1
  br i1 %29, label %30, label %31

30:                                               ; preds = %23
  br label %32

31:                                               ; preds = %23
  br label %32

32:                                               ; preds = %30, %31
  %33 = phi ptr [ getelementptr inbounds nuw (i8, ptr @s_634745896800656653, i64 8), %31 ], [ getelementptr inbounds nuw (i8, ptr @s_634742598265772020, i64 8), %30 ]
  %34 = ptrtoint ptr %28 to i64
  %35 = ptrtoint ptr %33 to i64
  %36 = icmp ne i64 %34, 0
  %37 = icmp ne i64 %35, 0
  %38 = zext i1 %36 to i32
  %39 = zext i1 %37 to i32
  %40 = and i32 %38, %39
  %41 = icmp ne i32 %40, 0
  br i1 %41, label %42, label %45

42:                                               ; preds = %32
  %43 = call i32 @strcmp(ptr %28, ptr %33)
  %44 = icmp eq i32 %43, 0
  br label %47

45:                                               ; preds = %32
  %46 = icmp eq i64 %34, %35
  br label %47

47:                                               ; preds = %42, %45
  %48 = phi i1 [ %46, %45 ], [ %44, %42 ]
  br label %49

49:                                               ; preds = %47
  br label %50

50:                                               ; preds = %49, %19
  %51 = phi i1 [ %48, %49 ], [ %22, %19 ]
  store i1 %51, ptr %4, align 1
  %52 = load i1, ptr %4, align 1
  ret i1 %52
}

; Function Attrs: mustprogress
define i1 @noInitializerInLoop() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = call ptr @mk(double 0.000000e+00)
  store ptr %4, ptr %2, align 8
  store i32 1, ptr %1, align 4
  br label %5

5:                                                ; preds = %8, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 4
  br i1 %7, label %8, label %14

8:                                                ; preds = %5
  %9 = load i32, ptr %1, align 4
  %10 = sitofp i32 %9 to double
  %11 = call ptr @mk(double %10)
  store ptr %11, ptr %2, align 8
  %12 = load i32, ptr %1, align 4
  %13 = add i32 %12, 1
  store i32 %13, ptr %1, align 4
  br label %5

14:                                               ; preds = %5
  call void @churn()
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr %N, ptr %15, i32 0, i32 1
  %17 = load ptr, ptr %16, align 8
  %18 = getelementptr %V, ptr %17, i32 0, i32 2
  %19 = load double, ptr %18, align 8
  %20 = fcmp oeq double %19, 3.000000e+00
  store i1 %20, ptr %3, align 1
  %21 = load i1, ptr %3, align 1
  ret i1 %21
}

; Function Attrs: mustprogress
define i1 @loopInFinally({ ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca { ptr, ptr }, align 8
  %5 = alloca i1, align 1
  store { ptr, ptr } %0, ptr %4, align 8
  %6 = call ptr @mk(double 5.000000e+00)
  store ptr %6, ptr %3, align 8
  store i32 0, ptr %2, align 4
  br label %7

7:                                                ; preds = %42, %1
  %8 = load i32, ptr %2, align 4
  %9 = icmp slt i32 %8, 3
  br i1 %9, label %10, label %49

10:                                               ; preds = %7
  %11 = load { ptr, ptr }, ptr %4, align 8
  %12 = extractvalue { ptr, ptr } %11, 0
  %13 = extractvalue { ptr, ptr } %11, 1
  %14 = getelementptr ptr, ptr %12, i32 2
  %15 = load ptr, ptr %14, align 8
  %16 = ptrtoint ptr %13 to i64
  %17 = ptrtoint ptr %15 to i64
  %18 = add i64 %16, %17
  %19 = inttoptr i64 %18 to ptr
  %20 = load ptr, ptr %3, align 8
  %21 = getelementptr %N, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %21, align 8
  %23 = extractvalue { ptr, ptr } %11, 0
  %24 = extractvalue { ptr, ptr } %11, 1
  %25 = getelementptr ptr, ptr %23, i32 2
  %26 = load ptr, ptr %25, align 8
  %27 = ptrtoint ptr %26 to i64
  %28 = and i64 %27, -9223372036854775808
  %29 = icmp ne i64 %28, 0
  br i1 %29, label %30, label %40

30:                                               ; preds = %10
  %31 = sub i64 %27, -9223372036854775808
  %32 = inttoptr i64 %31 to ptr
  %33 = getelementptr { ptr, ptr }, ptr %32, i32 0, i32 1
  %34 = load ptr, ptr %33, align 8
  %35 = icmp ne ptr %34, null
  br i1 %35, label %36, label %37

36:                                               ; preds = %30
  call void %34(ptr %24, ptr %22)
  br label %38

37:                                               ; preds = %30
  br label %38

38:                                               ; preds = %36, %37
  br label %39

39:                                               ; preds = %38
  br label %41

40:                                               ; preds = %10
  store ptr %22, ptr %19, align 8
  br label %41

41:                                               ; preds = %39, %40
  br label %42

42:                                               ; preds = %41
  %43 = load i32, ptr %2, align 4
  %44 = add i32 %43, 10
  %45 = sitofp i32 %44 to double
  %46 = call ptr @mk(double %45)
  store ptr %46, ptr %3, align 8
  call void @churn()
  %47 = load i32, ptr %2, align 4
  %48 = add i32 %47, 1
  store i32 %48, ptr %2, align 4
  br label %7

49:                                               ; preds = %7
  %50 = load ptr, ptr %3, align 8
  %51 = getelementptr %N, ptr %50, i32 0, i32 1
  %52 = load ptr, ptr %51, align 8
  %53 = getelementptr %V, ptr %52, i32 0, i32 2
  %54 = load double, ptr %53, align 8
  %55 = fcmp oeq double %54, 1.200000e+01
  br i1 %55, label %56, label %75

56:                                               ; preds = %49
  %57 = load ptr, ptr %3, align 8
  %58 = getelementptr %N, ptr %57, i32 0, i32 1
  %59 = load ptr, ptr %58, align 8
  %60 = getelementptr %V, ptr %59, i32 0, i32 1
  %61 = load ptr, ptr %60, align 8
  %62 = ptrtoint ptr %61 to i64
  %63 = icmp ne i64 %62, 0
  %64 = zext i1 %63 to i32
  %65 = and i32 %64, 1
  %66 = icmp ne i32 %65, 0
  br i1 %66, label %67, label %70

67:                                               ; preds = %56
  %68 = call i32 @strcmp(ptr %61, ptr getelementptr inbounds nuw (i8, ptr @s_7521248980907757358, i64 8))
  %69 = icmp eq i32 %68, 0
  br label %72

70:                                               ; preds = %56
  %71 = icmp eq i64 %62, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_7521248980907757358, i64 8) to i64)
  br label %72

72:                                               ; preds = %67, %70
  %73 = phi i1 [ %71, %70 ], [ %69, %67 ]
  br label %74

74:                                               ; preds = %72
  br label %75

75:                                               ; preds = %74, %49
  %76 = phi i1 [ %73, %74 ], [ %55, %49 ]
  br i1 %76, label %77, label %110

77:                                               ; preds = %75
  call void @churn()
  %78 = load { ptr, ptr }, ptr %4, align 8
  %79 = extractvalue { ptr, ptr } %78, 0
  %80 = extractvalue { ptr, ptr } %78, 1
  %81 = getelementptr ptr, ptr %79, i32 2
  %82 = load ptr, ptr %81, align 8
  %83 = ptrtoint ptr %80 to i64
  %84 = ptrtoint ptr %82 to i64
  %85 = add i64 %83, %84
  %86 = inttoptr i64 %85 to ptr
  %87 = extractvalue { ptr, ptr } %78, 0
  %88 = extractvalue { ptr, ptr } %78, 1
  %89 = getelementptr ptr, ptr %87, i32 2
  %90 = load ptr, ptr %89, align 8
  %91 = ptrtoint ptr %90 to i64
  %92 = and i64 %91, -9223372036854775808
  %93 = icmp ne i64 %92, 0
  br i1 %93, label %94, label %100

94:                                               ; preds = %77
  %95 = sub i64 %91, -9223372036854775808
  %96 = inttoptr i64 %95 to ptr
  %97 = getelementptr { ptr, ptr }, ptr %96, i32 0, i32 0
  %98 = load ptr, ptr %97, align 8
  %99 = call ptr %98(ptr %88)
  br label %102

100:                                              ; preds = %77
  %101 = load ptr, ptr %86, align 8
  br label %102

102:                                              ; preds = %94, %100
  %103 = phi ptr [ %101, %100 ], [ %99, %94 ]
  br label %104

104:                                              ; preds = %102
  %105 = getelementptr %V, ptr %103, i32 0, i32 2
  %106 = load double, ptr %105, align 8
  %107 = fcmp oeq double %106, 1.100000e+01
  store i1 %107, ptr %5, align 1
  %108 = load i1, ptr %5, align 1
  ret i1 %108

109:                                              ; preds = %110
  unreachable

110:                                              ; preds = %75
  %111 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11184076928100809650, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 69, ptr null)
  br label %109
}

; Function Attrs: mustprogress
define i1 @ifInCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca i1, align 1
  store double 0.000000e+00, ptr %4, align 8
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %160 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  %11 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %7
  %12 = icmp eq ptr %11, @_ZTIi
  br i1 %12, label %13, label %20

13:                                               ; preds = %.noexc5
  %14 = load i32, ptr %10, align 4
  %15 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %13
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 0
  store i64 4, ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 2
  store i32 %14, ptr %19, align 4
  br label %143

20:                                               ; preds = %.noexc5
  %21 = icmp eq ptr %11, @_ZTIj
  br i1 %21, label %22, label %29

22:                                               ; preds = %20
  %23 = load i32, ptr %10, align 4
  %24 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %22
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 0
  store i64 4, ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %27, align 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 2
  store i32 %23, ptr %28, align 4
  br label %140

29:                                               ; preds = %20
  %30 = icmp eq ptr %11, @_ZTId
  br i1 %30, label %31, label %38

31:                                               ; preds = %29
  %32 = load double, ptr %10, align 8
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 0
  store i64 8, ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 2
  store double %32, ptr %37, align 8
  br label %137

38:                                               ; preds = %29
  %39 = icmp eq ptr %11, @_ZTIf
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load float, ptr %10, align 4
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 0
  store i64 4, ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 2
  store float %41, ptr %46, align 4
  br label %134

47:                                               ; preds = %38
  %48 = icmp eq ptr %11, @_ZTIb
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load i1, ptr %10, align 1
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 0
  store i64 1, ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 2
  store i1 %50, ptr %55, align 1
  br label %131

56:                                               ; preds = %47
  %57 = icmp eq ptr %11, @_ZTIx
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load i64, ptr %10, align 8
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 0
  store i64 8, ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 2
  store i64 %59, ptr %64, align 8
  br label %128

65:                                               ; preds = %56
  %66 = icmp eq ptr %11, @_ZTIPc
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load ptr, ptr %10, align 8
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 0
  store i64 8, ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 2
  store ptr %68, ptr %73, align 8
  br label %125

74:                                               ; preds = %65
  %75 = icmp eq ptr %11, @_ZTIPv
  br i1 %75, label %76, label %78

76:                                               ; preds = %74
  %77 = load ptr, ptr %10, align 8
  br label %122

78:                                               ; preds = %74
  %79 = load ptr, ptr %11, align 8
  %80 = icmp eq ptr %79, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %80, label %81, label %112

81:                                               ; preds = %78
  %82 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %11, i32 0, i32 2
  %83 = load i32, ptr %82, align 4
  %84 = and i32 %83, 1073741824
  %85 = icmp ne i32 %84, 0
  br i1 %85, label %86, label %102

86:                                               ; preds = %81
  %87 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %11, i32 0, i32 4
  %88 = load ptr, ptr %87, align 8
  %89 = icmp ne ptr %88, null
  br i1 %89, label %90, label %92

90:                                               ; preds = %86
  invoke void %88(ptr %1, ptr %10)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %90
  %91 = load ptr, ptr %1, align 8
  br label %99

92:                                               ; preds = %86
  %93 = load ptr, ptr %10, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %99

99:                                               ; preds = %.noexc13, %.noexc4
  %100 = phi ptr [ %95, %.noexc4 ], [ %91, %.noexc13 ]
  br label %101

101:                                              ; preds = %99
  br label %109

102:                                              ; preds = %81
  %103 = load ptr, ptr %10, align 8
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 0
  store i64 8, ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 2
  store ptr %103, ptr %108, align 8
  br label %109

109:                                              ; preds = %101, %.noexc14
  %110 = phi ptr [ %105, %.noexc14 ], [ %100, %101 ]
  br label %111

111:                                              ; preds = %109
  br label %119

112:                                              ; preds = %78
  %113 = load ptr, ptr %10, align 8
  %114 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %112
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 0
  store i64 8, ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 2
  store ptr %113, ptr %118, align 8
  br label %119

119:                                              ; preds = %111, %.noexc
  %120 = phi ptr [ %115, %.noexc ], [ %110, %111 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %76, %121
  %123 = phi ptr [ %120, %121 ], [ %77, %76 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc7, %124
  %126 = phi ptr [ %123, %124 ], [ %70, %.noexc7 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc6, %127
  %129 = phi ptr [ %126, %127 ], [ %61, %.noexc6 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc2, %130
  %132 = phi ptr [ %129, %130 ], [ %52, %.noexc2 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc9, %133
  %135 = phi ptr [ %132, %133 ], [ %43, %.noexc9 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc1, %136
  %138 = phi ptr [ %135, %136 ], [ %34, %.noexc1 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc12, %139
  %141 = phi ptr [ %138, %139 ], [ %25, %.noexc12 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc3, %142
  %144 = phi ptr [ %141, %142 ], [ %16, %.noexc3 ]
  br label %145

145:                                              ; preds = %143
  store ptr %144, ptr %3, align 8
  %146 = invoke ptr @mk(double 1.000000e+00)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %145
  store ptr %146, ptr %2, align 8
  %147 = load double, ptr %4, align 8
  %148 = fcmp oeq double %147, 0.000000e+00
  br i1 %148, label %149, label %151

149:                                              ; preds = %.noexc10
  %150 = invoke ptr @mk(double 7.000000e+00)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %149
  store ptr %150, ptr %2, align 8
  br label %151

151:                                              ; preds = %.noexc8, %.noexc10
  invoke void @churn()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %151
  %152 = load ptr, ptr %2, align 8
  %153 = getelementptr %N, ptr %152, i32 0, i32 1
  %154 = load ptr, ptr %153, align 8
  %155 = getelementptr %V, ptr %154, i32 0, i32 2
  %156 = load double, ptr %155, align 8
  store double %156, ptr %4, align 8
  call void @__cxa_end_catch()
  %157 = load double, ptr %4, align 8
  %158 = fcmp oeq double %157, 7.000000e+00
  store i1 %158, ptr %5, align 1
  %159 = load i1, ptr %5, align 1
  ret i1 %159

160:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %102, %90, %22, %151, %145, %40, %149, %67, %58, %7, %92, %13, %49, %31, %112
  %161 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %161
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca { ptr }, align 8
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %25, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 3
  br i1 %5, label %6, label %28

6:                                                ; preds = %3
  %7 = load i32, ptr %2, align 4
  %8 = srem i32 %7, 2
  %9 = icmp eq i32 %8, 0
  %10 = call i1 @noInitializer(i1 %9)
  br i1 %10, label %11, label %30

11:                                               ; preds = %6
  %12 = call i1 @noInitializerInLoop()
  br i1 %12, label %13, label %32

13:                                               ; preds = %11
  %14 = call ptr @GC_malloc(i64 32)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %V, ptr %15, i32 0, i32 0
  store ptr @V..vtbl, ptr %16, align 8
  call void @V.constructor(ptr %15, double 0.000000e+00)
  store { ptr } zeroinitializer, ptr %1, align 8
  %17 = getelementptr { ptr }, ptr %1, i32 0, i32 0
  store ptr %15, ptr %17, align 8
  %18 = load { ptr }, ptr %1, align 8
  %19 = call ptr @GC_malloc(i64 16)
  %20 = getelementptr i8, ptr %19, i64 8
  store { ptr } %18, ptr %20, align 8
  %21 = insertvalue { ptr, ptr } { ptr @IX.12252034448159921135..vtbl, ptr undef }, ptr %20, 1
  %22 = call i1 @loopInFinally({ ptr, ptr } %21)
  br i1 %22, label %23, label %34

23:                                               ; preds = %13
  %24 = call i1 @ifInCatch()
  br i1 %24, label %25, label %36

25:                                               ; preds = %23
  %26 = load i32, ptr %2, align 4
  %27 = add i32 %26, 1
  store i32 %27, ptr %2, align 4
  br label %3

28:                                               ; preds = %3
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

29:                                               ; preds = %36, %34, %32, %30
  unreachable

30:                                               ; preds = %6
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10615361117518154669, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 95, ptr null)
  br label %29

32:                                               ; preds = %11
  %33 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8158167497407629090, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 96, ptr null)
  br label %29

34:                                               ; preds = %13
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6963701317841836948, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 97, ptr null)
  br label %29

36:                                               ; preds = %23
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2589278777131849494, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3009451683414766259, i64 8), i32 98, ptr null)
  br label %29
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

