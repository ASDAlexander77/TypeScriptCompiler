; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%R = type { ptr, double, ptr }

$R..size = comdat any

@s_12638243275648680757 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFR\00", align 8
@s_12638208091276578005 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFr\00", align 8
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
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_4502274137427804027 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrowOutOfCatch\00", align 8
@f_642979808102785306 = internal constant [121 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00catch_finally_owned_locals_eh.ts\00", align 8
@m_11328015321383437826 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreturnFromFinally\00", align 8
@m_15098827765242065975 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreakFromFinally\00", align 8
@m_1165181803863095751 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnestedTryInCatch\00", align 8
@m_15111080100605800680 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFrethrow\00", align 8
@m_18370820746968551330 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinallyAfterThrow\00", align 8
@R..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638243275648680757, i64 8)
@R..size = global i64 24, comdat
@R..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @R..instanceOf, ptr @R..new, ptr @R..rtti, ptr @R..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
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

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

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
define i1 @R..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @R..rtti, align 8
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
define ptr @R..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %R, ptr %3, i32 0, i32 0
  store ptr @R..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @R.constructor(ptr %0, double %1, ptr %2) #1 {
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %R, ptr %7, i32 0, i32 1
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %R, ptr %10, i32 0, i32 2
  %12 = load ptr, ptr %4, align 8
  store ptr %12, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress noinline
define ptr @mk(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %R, ptr %5, i32 0, i32 0
  store ptr @R..vtbl, ptr %6, align 8
  %7 = load double, ptr %2, align 8
  %8 = load double, ptr %2, align 8
  %9 = call ptr @GC_malloc_atomic(i64 40)
  %10 = getelementptr i8, ptr %9, i64 8
  call void @__ts_number_to_string(ptr %10, double %8)
  %11 = icmp eq ptr %10, null
  %12 = select i1 %11, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %10
  %13 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638208091276578005, i64 8))
  %14 = add i64 1, %13
  %15 = call i64 @strlen(ptr %12)
  %16 = add i64 %14, %15
  %17 = add i64 %16, 8
  %18 = call ptr @GC_malloc(i64 %17)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr i8, ptr %19, i64 0
  %21 = sub i64 %16, 0
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638208091276578005, i64 8), i64 %13, i64 %21)
  %23 = add i64 0, %13
  %24 = getelementptr i8, ptr %19, i64 %23
  %25 = sub i64 %16, %23
  %26 = call ptr @__memcpy_chk(ptr %24, ptr %12, i64 %15, i64 %25)
  %27 = add i64 %23, %15
  %28 = getelementptr i8, ptr %19, i64 %27
  store i8 0, ptr %28, align 1
  call void @R.constructor(ptr %5, double %7, ptr %19)
  store ptr %5, ptr %3, align 8
  %29 = load ptr, ptr %3, align 8
  ret ptr %29
}

; Function Attrs: mustprogress
define double @throwOutOfCatch(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  store double %0, ptr %6, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %302 unwind label %9

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc3
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc3
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %142

31:                                               ; preds = %22
  %32 = icmp eq ptr %13, @_ZTId
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load double, ptr %12, align 8
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 0
  store i64 8, ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 2
  store double %34, ptr %39, align 8
  br label %139

40:                                               ; preds = %31
  %41 = icmp eq ptr %13, @_ZTIf
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load float, ptr %12, align 4
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 0
  store i64 4, ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 2
  store float %43, ptr %48, align 4
  br label %136

49:                                               ; preds = %40
  %50 = icmp eq ptr %13, @_ZTIb
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load i1, ptr %12, align 1
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 0
  store i64 1, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 2
  store i1 %52, ptr %57, align 1
  br label %133

58:                                               ; preds = %49
  %59 = icmp eq ptr %13, @_ZTIx
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i64, ptr %12, align 8
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 0
  store i64 8, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 2
  store i64 %61, ptr %66, align 8
  br label %130

67:                                               ; preds = %58
  %68 = icmp eq ptr %13, @_ZTIPc
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load ptr, ptr %12, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 2
  store ptr %70, ptr %75, align 8
  br label %127

76:                                               ; preds = %67
  %77 = icmp eq ptr %13, @_ZTIPv
  br i1 %77, label %78, label %80

78:                                               ; preds = %76
  %79 = load ptr, ptr %12, align 8
  br label %124

80:                                               ; preds = %76
  %81 = load ptr, ptr %13, align 8
  %82 = icmp eq ptr %81, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %82, label %83, label %114

83:                                               ; preds = %80
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 2
  %85 = load i32, ptr %84, align 4
  %86 = and i32 %85, 1073741824
  %87 = icmp ne i32 %86, 0
  br i1 %87, label %88, label %104

88:                                               ; preds = %83
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 4
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %94

92:                                               ; preds = %88
  invoke void %90(ptr %3, ptr %12)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %92
  %93 = load ptr, ptr %3, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc17, %.noexc20
  %102 = phi ptr [ %97, %.noexc20 ], [ %93, %.noexc17 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc19
  %112 = phi ptr [ %107, %.noexc19 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc16
  %122 = phi ptr [ %117, %.noexc16 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc10, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc10 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc7, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc7 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc15, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc15 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc9, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc9 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc11, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc11 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc13, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc13 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc2, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc2 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %4, align 8
  %148 = load double, ptr %6, align 8
  %149 = invoke ptr @mk(double %148)
          to label %150 unwind label %endcatch.pad

150:                                              ; preds = %147
  %151 = getelementptr %R, ptr %149, i32 0, i32 1
  %152 = load double, ptr %151, align 8
  %153 = fcmp oge double %152, 0.000000e+00
  br i1 %153, label %154, label %156

154:                                              ; preds = %150
  call void @__cxa_end_catch()
  %155 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %155, align 4
  invoke void @__cxa_throw(ptr %155, ptr @_ZTIi, ptr null)
          to label %302 unwind label %160

156:                                              ; preds = %150
  %157 = getelementptr %R, ptr %149, i32 0, i32 1
  %158 = load double, ptr %157, align 8
  store double %158, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %300

endcatch.pad:                                     ; preds = %147
  %159 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

160:                                              ; preds = %154
  %161 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %160
  %162 = phi { ptr, i32 } [ %161, %160 ], [ %159, %endcatch.pad ]
  %163 = extractvalue { ptr, i32 } %162, 0
  %164 = call ptr @__cxa_begin_catch(ptr %163)
  %165 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.caught
  %166 = icmp eq ptr %165, @_ZTIi
  br i1 %166, label %167, label %174

167:                                              ; preds = %.noexc6
  %168 = load i32, ptr %164, align 4
  %169 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %167
  %170 = getelementptr i8, ptr %169, i64 8
  %171 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 0
  store i64 4, ptr %171, align 8
  %172 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %172, align 8
  %173 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 2
  store i32 %168, ptr %173, align 4
  br label %297

174:                                              ; preds = %.noexc6
  %175 = icmp eq ptr %165, @_ZTIj
  br i1 %175, label %176, label %183

176:                                              ; preds = %174
  %177 = load i32, ptr %164, align 4
  %178 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %176
  %179 = getelementptr i8, ptr %178, i64 8
  %180 = getelementptr { i64, ptr, i32 }, ptr %179, i32 0, i32 0
  store i64 4, ptr %180, align 8
  %181 = getelementptr { i64, ptr, i32 }, ptr %179, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %181, align 8
  %182 = getelementptr { i64, ptr, i32 }, ptr %179, i32 0, i32 2
  store i32 %177, ptr %182, align 4
  br label %294

183:                                              ; preds = %174
  %184 = icmp eq ptr %165, @_ZTId
  br i1 %184, label %185, label %192

185:                                              ; preds = %183
  %186 = load double, ptr %164, align 8
  %187 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %185
  %188 = getelementptr i8, ptr %187, i64 8
  %189 = getelementptr { i64, ptr, double }, ptr %188, i32 0, i32 0
  store i64 8, ptr %189, align 8
  %190 = getelementptr { i64, ptr, double }, ptr %188, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %190, align 8
  %191 = getelementptr { i64, ptr, double }, ptr %188, i32 0, i32 2
  store double %186, ptr %191, align 8
  br label %291

192:                                              ; preds = %183
  %193 = icmp eq ptr %165, @_ZTIf
  br i1 %193, label %194, label %201

194:                                              ; preds = %192
  %195 = load float, ptr %164, align 4
  %196 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %194
  %197 = getelementptr i8, ptr %196, i64 8
  %198 = getelementptr { i64, ptr, float }, ptr %197, i32 0, i32 0
  store i64 4, ptr %198, align 8
  %199 = getelementptr { i64, ptr, float }, ptr %197, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %199, align 8
  %200 = getelementptr { i64, ptr, float }, ptr %197, i32 0, i32 2
  store float %195, ptr %200, align 4
  br label %288

201:                                              ; preds = %192
  %202 = icmp eq ptr %165, @_ZTIb
  br i1 %202, label %203, label %210

203:                                              ; preds = %201
  %204 = load i1, ptr %164, align 1
  %205 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %203
  %206 = getelementptr i8, ptr %205, i64 8
  %207 = getelementptr { i64, ptr, i1 }, ptr %206, i32 0, i32 0
  store i64 1, ptr %207, align 8
  %208 = getelementptr { i64, ptr, i1 }, ptr %206, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %208, align 8
  %209 = getelementptr { i64, ptr, i1 }, ptr %206, i32 0, i32 2
  store i1 %204, ptr %209, align 1
  br label %285

210:                                              ; preds = %201
  %211 = icmp eq ptr %165, @_ZTIx
  br i1 %211, label %212, label %219

212:                                              ; preds = %210
  %213 = load i64, ptr %164, align 8
  %214 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %212
  %215 = getelementptr i8, ptr %214, i64 8
  %216 = getelementptr { i64, ptr, i64 }, ptr %215, i32 0, i32 0
  store i64 8, ptr %216, align 8
  %217 = getelementptr { i64, ptr, i64 }, ptr %215, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %217, align 8
  %218 = getelementptr { i64, ptr, i64 }, ptr %215, i32 0, i32 2
  store i64 %213, ptr %218, align 8
  br label %282

219:                                              ; preds = %210
  %220 = icmp eq ptr %165, @_ZTIPc
  br i1 %220, label %221, label %228

221:                                              ; preds = %219
  %222 = load ptr, ptr %164, align 8
  %223 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %221
  %224 = getelementptr i8, ptr %223, i64 8
  %225 = getelementptr { i64, ptr, ptr }, ptr %224, i32 0, i32 0
  store i64 8, ptr %225, align 8
  %226 = getelementptr { i64, ptr, ptr }, ptr %224, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %226, align 8
  %227 = getelementptr { i64, ptr, ptr }, ptr %224, i32 0, i32 2
  store ptr %222, ptr %227, align 8
  br label %279

228:                                              ; preds = %219
  %229 = icmp eq ptr %165, @_ZTIPv
  br i1 %229, label %230, label %232

230:                                              ; preds = %228
  %231 = load ptr, ptr %164, align 8
  br label %276

232:                                              ; preds = %228
  %233 = load ptr, ptr %165, align 8
  %234 = icmp eq ptr %233, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %234, label %235, label %266

235:                                              ; preds = %232
  %236 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %165, i32 0, i32 2
  %237 = load i32, ptr %236, align 4
  %238 = and i32 %237, 1073741824
  %239 = icmp ne i32 %238, 0
  br i1 %239, label %240, label %256

240:                                              ; preds = %235
  %241 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %165, i32 0, i32 4
  %242 = load ptr, ptr %241, align 8
  %243 = icmp ne ptr %242, null
  br i1 %243, label %244, label %246

244:                                              ; preds = %240
  invoke void %242(ptr %2, ptr %164)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %244
  %245 = load ptr, ptr %2, align 8
  br label %253

246:                                              ; preds = %240
  %247 = load ptr, ptr %164, align 8
  %248 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %246
  %249 = getelementptr i8, ptr %248, i64 8
  %250 = getelementptr { i64, ptr, ptr }, ptr %249, i32 0, i32 0
  store i64 8, ptr %250, align 8
  %251 = getelementptr { i64, ptr, ptr }, ptr %249, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %251, align 8
  %252 = getelementptr { i64, ptr, ptr }, ptr %249, i32 0, i32 2
  store ptr %247, ptr %252, align 8
  br label %253

253:                                              ; preds = %.noexc, %.noexc5
  %254 = phi ptr [ %249, %.noexc5 ], [ %245, %.noexc ]
  br label %255

255:                                              ; preds = %253
  br label %263

256:                                              ; preds = %235
  %257 = load ptr, ptr %164, align 8
  %258 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %256
  %259 = getelementptr i8, ptr %258, i64 8
  %260 = getelementptr { i64, ptr, ptr }, ptr %259, i32 0, i32 0
  store i64 8, ptr %260, align 8
  %261 = getelementptr { i64, ptr, ptr }, ptr %259, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %261, align 8
  %262 = getelementptr { i64, ptr, ptr }, ptr %259, i32 0, i32 2
  store ptr %257, ptr %262, align 8
  br label %263

263:                                              ; preds = %255, %.noexc21
  %264 = phi ptr [ %259, %.noexc21 ], [ %254, %255 ]
  br label %265

265:                                              ; preds = %263
  br label %273

266:                                              ; preds = %232
  %267 = load ptr, ptr %164, align 8
  %268 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %266
  %269 = getelementptr i8, ptr %268, i64 8
  %270 = getelementptr { i64, ptr, ptr }, ptr %269, i32 0, i32 0
  store i64 8, ptr %270, align 8
  %271 = getelementptr { i64, ptr, ptr }, ptr %269, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %271, align 8
  %272 = getelementptr { i64, ptr, ptr }, ptr %269, i32 0, i32 2
  store ptr %267, ptr %272, align 8
  br label %273

273:                                              ; preds = %265, %.noexc18
  %274 = phi ptr [ %269, %.noexc18 ], [ %264, %265 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %230, %275
  %277 = phi ptr [ %274, %275 ], [ %231, %230 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc8, %278
  %280 = phi ptr [ %277, %278 ], [ %224, %.noexc8 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc23, %281
  %283 = phi ptr [ %280, %281 ], [ %215, %.noexc23 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc22, %284
  %286 = phi ptr [ %283, %284 ], [ %206, %.noexc22 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc12, %287
  %289 = phi ptr [ %286, %287 ], [ %197, %.noexc12 ]
  br label %290

290:                                              ; preds = %288
  br label %291

291:                                              ; preds = %.noexc4, %290
  %292 = phi ptr [ %289, %290 ], [ %188, %.noexc4 ]
  br label %293

293:                                              ; preds = %291
  br label %294

294:                                              ; preds = %.noexc14, %293
  %295 = phi ptr [ %292, %293 ], [ %179, %.noexc14 ]
  br label %296

296:                                              ; preds = %294
  br label %297

297:                                              ; preds = %.noexc1, %296
  %298 = phi ptr [ %295, %296 ], [ %170, %.noexc1 ]
  br label %299

299:                                              ; preds = %297
  store ptr %298, ptr %5, align 8
  store double 7.000000e+00, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %300

300:                                              ; preds = %299, %156
  %301 = load double, ptr %7, align 8
  ret double %301

302:                                              ; preds = %154, %1
  unreachable

endcatch.resume:                                  ; preds = %212, %203, %256, %94, %104, %266, %92, %114, %51, %176, %24, %194, %33, %69, %42, %221, %60, %.caught, %246, %185, %9, %15, %167, %244
  %303 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %303
}

; Function Attrs: mustprogress
define double @returnFromFinally(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = invoke ptr @mk(double %4)
          to label %6 unwind label %12

6:                                                ; preds = %1
  %7 = load double, ptr %2, align 8
  %8 = fadd double %7, 1.000000e+00
  %9 = call ptr @mk(double %8)
  %10 = getelementptr %R, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  store double %11, ptr %3, align 8
  br label %21

12:                                               ; preds = %1
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load double, ptr %2, align 8
  %17 = fadd double %16, 1.000000e+00
  %18 = invoke ptr @mk(double %17)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %12
  %19 = getelementptr %R, ptr %18, i32 0, i32 1
  %20 = load double, ptr %19, align 8
  store double %20, ptr %3, align 8
  call void @__cxa_end_catch()
  br label %21

21:                                               ; preds = %6, %.noexc
  %22 = load double, ptr %3, align 8
  ret double %22

endcatch.resume:                                  ; preds = %12
  %23 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %23
}

; Function Attrs: mustprogress
define double @breakFromFinally(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  store double %0, ptr %4, align 8
  store double 0.000000e+00, ptr %3, align 8
  store i32 0, ptr %2, align 4
  br label %6

6:                                                ; preds = %26, %1
  %7 = load i32, ptr %2, align 4
  %8 = load double, ptr %4, align 8
  %9 = sitofp i32 %7 to double
  %10 = fcmp olt double %9, %8
  br i1 %10, label %11, label %29

11:                                               ; preds = %6
  %12 = load double, ptr %3, align 8
  %13 = load i32, ptr %2, align 4
  %14 = sitofp i32 %13 to double
  %15 = fadd double %12, %14
  store double %15, ptr %3, align 8
  %16 = load i32, ptr %2, align 4
  %17 = sitofp i32 %16 to double
  %18 = call ptr @mk(double %17)
  %19 = load double, ptr %3, align 8
  %20 = getelementptr %R, ptr %18, i32 0, i32 1
  %21 = load double, ptr %20, align 8
  %22 = fadd double %19, %21
  store double %22, ptr %3, align 8
  %23 = load i32, ptr %2, align 4
  %24 = icmp eq i32 %23, 3
  br i1 %24, label %25, label %26

25:                                               ; preds = %11
  br label %29

26:                                               ; preds = %11
  %27 = load i32, ptr %2, align 4
  %28 = add i32 %27, 1
  store i32 %28, ptr %2, align 4
  br label %6

29:                                               ; preds = %25, %6
  %30 = load double, ptr %3, align 8
  store double %30, ptr %5, align 8
  %31 = load double, ptr %5, align 8
  ret double %31
}

; Function Attrs: mustprogress
define double @nestedTryInCatch(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  store double %0, ptr %6, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %303 unwind label %9

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc8
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc8
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %142

31:                                               ; preds = %22
  %32 = icmp eq ptr %13, @_ZTId
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load double, ptr %12, align 8
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 0
  store i64 8, ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 2
  store double %34, ptr %39, align 8
  br label %139

40:                                               ; preds = %31
  %41 = icmp eq ptr %13, @_ZTIf
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load float, ptr %12, align 4
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 0
  store i64 4, ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 2
  store float %43, ptr %48, align 4
  br label %136

49:                                               ; preds = %40
  %50 = icmp eq ptr %13, @_ZTIb
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load i1, ptr %12, align 1
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 0
  store i64 1, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 2
  store i1 %52, ptr %57, align 1
  br label %133

58:                                               ; preds = %49
  %59 = icmp eq ptr %13, @_ZTIx
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i64, ptr %12, align 8
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 0
  store i64 8, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 2
  store i64 %61, ptr %66, align 8
  br label %130

67:                                               ; preds = %58
  %68 = icmp eq ptr %13, @_ZTIPc
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load ptr, ptr %12, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 2
  store ptr %70, ptr %75, align 8
  br label %127

76:                                               ; preds = %67
  %77 = icmp eq ptr %13, @_ZTIPv
  br i1 %77, label %78, label %80

78:                                               ; preds = %76
  %79 = load ptr, ptr %12, align 8
  br label %124

80:                                               ; preds = %76
  %81 = load ptr, ptr %13, align 8
  %82 = icmp eq ptr %81, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %82, label %83, label %114

83:                                               ; preds = %80
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 2
  %85 = load i32, ptr %84, align 4
  %86 = and i32 %85, 1073741824
  %87 = icmp ne i32 %86, 0
  br i1 %87, label %88, label %104

88:                                               ; preds = %83
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 4
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %94

92:                                               ; preds = %88
  invoke void %90(ptr %4, ptr %12)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %92
  %93 = load ptr, ptr %4, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc27, %.noexc24
  %102 = phi ptr [ %97, %.noexc24 ], [ %93, %.noexc27 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc15
  %112 = phi ptr [ %107, %.noexc15 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc21
  %122 = phi ptr [ %117, %.noexc21 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc14, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc14 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc9, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc9 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc11, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc11 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc2, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc2 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc16, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc16 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc1, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc1 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %5, align 8
  %148 = load double, ptr %6, align 8
  %149 = invoke ptr @mk(double %148)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %147
  %150 = load double, ptr %6, align 8
  %151 = fadd double %150, 1.000000e+00
  %152 = invoke ptr @mk(double %151)
          to label %153 unwind label %155

153:                                              ; preds = %.noexc13
  %154 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %153
  store i32 3, ptr %154, align 4
  invoke void @__cxa_throw(ptr %154, ptr @_ZTIi, ptr null)
          to label %303 unwind label %155

155:                                              ; preds = %.noexc7, %.noexc13
  %156 = landingpad { ptr, i32 }
          catch ptr null
  %157 = extractvalue { ptr, i32 } %156, 0
  %158 = call ptr @__cxa_begin_catch(ptr %157)
  %159 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc18 unwind label %endcatch.resume3

.noexc18:                                         ; preds = %155
  %160 = icmp eq ptr %159, @_ZTIi
  br i1 %160, label %161, label %168

161:                                              ; preds = %.noexc18
  %162 = load i32, ptr %158, align 4
  %163 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume3

.noexc10:                                         ; preds = %161
  %164 = getelementptr i8, ptr %163, i64 8
  %165 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 0
  store i64 4, ptr %165, align 8
  %166 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %166, align 8
  %167 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 2
  store i32 %162, ptr %167, align 4
  br label %291

168:                                              ; preds = %.noexc18
  %169 = icmp eq ptr %159, @_ZTIj
  br i1 %169, label %170, label %177

170:                                              ; preds = %168
  %171 = load i32, ptr %158, align 4
  %172 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume3

.noexc12:                                         ; preds = %170
  %173 = getelementptr i8, ptr %172, i64 8
  %174 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 0
  store i64 4, ptr %174, align 8
  %175 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %175, align 8
  %176 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 2
  store i32 %171, ptr %176, align 4
  br label %288

177:                                              ; preds = %168
  %178 = icmp eq ptr %159, @_ZTId
  br i1 %178, label %179, label %186

179:                                              ; preds = %177
  %180 = load double, ptr %158, align 8
  %181 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume3

.noexc6:                                          ; preds = %179
  %182 = getelementptr i8, ptr %181, i64 8
  %183 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 0
  store i64 8, ptr %183, align 8
  %184 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %184, align 8
  %185 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 2
  store double %180, ptr %185, align 8
  br label %285

186:                                              ; preds = %177
  %187 = icmp eq ptr %159, @_ZTIf
  br i1 %187, label %188, label %195

188:                                              ; preds = %186
  %189 = load float, ptr %158, align 4
  %190 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume3

.noexc17:                                         ; preds = %188
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 0
  store i64 4, ptr %192, align 8
  %193 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %193, align 8
  %194 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 2
  store float %189, ptr %194, align 4
  br label %282

195:                                              ; preds = %186
  %196 = icmp eq ptr %159, @_ZTIb
  br i1 %196, label %197, label %204

197:                                              ; preds = %195
  %198 = load i1, ptr %158, align 1
  %199 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume3

.noexc26:                                         ; preds = %197
  %200 = getelementptr i8, ptr %199, i64 8
  %201 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 0
  store i64 1, ptr %201, align 8
  %202 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %202, align 8
  %203 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 2
  store i1 %198, ptr %203, align 1
  br label %279

204:                                              ; preds = %195
  %205 = icmp eq ptr %159, @_ZTIx
  br i1 %205, label %206, label %213

206:                                              ; preds = %204
  %207 = load i64, ptr %158, align 8
  %208 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume3

.noexc19:                                         ; preds = %206
  %209 = getelementptr i8, ptr %208, i64 8
  %210 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 0
  store i64 8, ptr %210, align 8
  %211 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %211, align 8
  %212 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 2
  store i64 %207, ptr %212, align 8
  br label %276

213:                                              ; preds = %204
  %214 = icmp eq ptr %159, @_ZTIPc
  br i1 %214, label %215, label %222

215:                                              ; preds = %213
  %216 = load ptr, ptr %158, align 8
  %217 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume3

.noexc20:                                         ; preds = %215
  %218 = getelementptr i8, ptr %217, i64 8
  %219 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 0
  store i64 8, ptr %219, align 8
  %220 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %220, align 8
  %221 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 2
  store ptr %216, ptr %221, align 8
  br label %273

222:                                              ; preds = %213
  %223 = icmp eq ptr %159, @_ZTIPv
  br i1 %223, label %224, label %226

224:                                              ; preds = %222
  %225 = load ptr, ptr %158, align 8
  br label %270

226:                                              ; preds = %222
  %227 = load ptr, ptr %159, align 8
  %228 = icmp eq ptr %227, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %228, label %229, label %260

229:                                              ; preds = %226
  %230 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %159, i32 0, i32 2
  %231 = load i32, ptr %230, align 4
  %232 = and i32 %231, 1073741824
  %233 = icmp ne i32 %232, 0
  br i1 %233, label %234, label %250

234:                                              ; preds = %229
  %235 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %159, i32 0, i32 4
  %236 = load ptr, ptr %235, align 8
  %237 = icmp ne ptr %236, null
  br i1 %237, label %238, label %240

238:                                              ; preds = %234
  invoke void %236(ptr %2, ptr %158)
          to label %.noexc25 unwind label %endcatch.resume3

.noexc25:                                         ; preds = %238
  %239 = load ptr, ptr %2, align 8
  br label %247

240:                                              ; preds = %234
  %241 = load ptr, ptr %158, align 8
  %242 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume3

.noexc23:                                         ; preds = %240
  %243 = getelementptr i8, ptr %242, i64 8
  %244 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 0
  store i64 8, ptr %244, align 8
  %245 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %245, align 8
  %246 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 2
  store ptr %241, ptr %246, align 8
  br label %247

247:                                              ; preds = %.noexc25, %.noexc23
  %248 = phi ptr [ %243, %.noexc23 ], [ %239, %.noexc25 ]
  br label %249

249:                                              ; preds = %247
  br label %257

250:                                              ; preds = %229
  %251 = load ptr, ptr %158, align 8
  %252 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume3

.noexc22:                                         ; preds = %250
  %253 = getelementptr i8, ptr %252, i64 8
  %254 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 0
  store i64 8, ptr %254, align 8
  %255 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %255, align 8
  %256 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 2
  store ptr %251, ptr %256, align 8
  br label %257

257:                                              ; preds = %249, %.noexc22
  %258 = phi ptr [ %253, %.noexc22 ], [ %248, %249 ]
  br label %259

259:                                              ; preds = %257
  br label %267

260:                                              ; preds = %226
  %261 = load ptr, ptr %158, align 8
  %262 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume3

.noexc5:                                          ; preds = %260
  %263 = getelementptr i8, ptr %262, i64 8
  %264 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 0
  store i64 8, ptr %264, align 8
  %265 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %265, align 8
  %266 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 2
  store ptr %261, ptr %266, align 8
  br label %267

267:                                              ; preds = %259, %.noexc5
  %268 = phi ptr [ %263, %.noexc5 ], [ %258, %259 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %224, %269
  %271 = phi ptr [ %268, %269 ], [ %225, %224 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc20, %272
  %274 = phi ptr [ %271, %272 ], [ %218, %.noexc20 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc19, %275
  %277 = phi ptr [ %274, %275 ], [ %209, %.noexc19 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc26, %278
  %280 = phi ptr [ %277, %278 ], [ %200, %.noexc26 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc17, %281
  %283 = phi ptr [ %280, %281 ], [ %191, %.noexc17 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc6, %284
  %286 = phi ptr [ %283, %284 ], [ %182, %.noexc6 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc12, %287
  %289 = phi ptr [ %286, %287 ], [ %173, %.noexc12 ]
  br label %290

290:                                              ; preds = %288
  br label %291

291:                                              ; preds = %.noexc10, %290
  %292 = phi ptr [ %289, %290 ], [ %164, %.noexc10 ]
  br label %293

293:                                              ; preds = %291
  store ptr %292, ptr %3, align 8
  %294 = load double, ptr %6, align 8
  %295 = fadd double %294, 2.000000e+00
  %296 = invoke ptr @mk(double %295)
          to label %.noexc4 unwind label %endcatch.resume3

.noexc4:                                          ; preds = %293
  %297 = getelementptr %R, ptr %149, i32 0, i32 1
  %298 = load double, ptr %297, align 8
  %299 = getelementptr %R, ptr %296, i32 0, i32 1
  %300 = load double, ptr %299, align 8
  %301 = fadd double %298, %300
  store double %301, ptr %7, align 8
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  %302 = load double, ptr %7, align 8
  ret double %302

303:                                              ; preds = %.noexc7, %1
  unreachable

endcatch.resume:                                  ; preds = %92, %94, %114, %24, %104, %60, %147, %42, %51, %9, %153, %33, %15, %69
  %304 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %304

endcatch.resume3:                                 ; preds = %197, %238, %240, %250, %215, %206, %155, %188, %170, %161, %179, %260, %293
  %305 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %305
}

; Function Attrs: mustprogress
define double @rethrow(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  store double %0, ptr %6, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %295 unwind label %9

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc16
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc16
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %142

31:                                               ; preds = %22
  %32 = icmp eq ptr %13, @_ZTId
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load double, ptr %12, align 8
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 0
  store i64 8, ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 2
  store double %34, ptr %39, align 8
  br label %139

40:                                               ; preds = %31
  %41 = icmp eq ptr %13, @_ZTIf
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load float, ptr %12, align 4
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 0
  store i64 4, ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 2
  store float %43, ptr %48, align 4
  br label %136

49:                                               ; preds = %40
  %50 = icmp eq ptr %13, @_ZTIb
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load i1, ptr %12, align 1
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 0
  store i64 1, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 2
  store i1 %52, ptr %57, align 1
  br label %133

58:                                               ; preds = %49
  %59 = icmp eq ptr %13, @_ZTIx
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i64, ptr %12, align 8
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 0
  store i64 8, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 2
  store i64 %61, ptr %66, align 8
  br label %130

67:                                               ; preds = %58
  %68 = icmp eq ptr %13, @_ZTIPc
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load ptr, ptr %12, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 2
  store ptr %70, ptr %75, align 8
  br label %127

76:                                               ; preds = %67
  %77 = icmp eq ptr %13, @_ZTIPv
  br i1 %77, label %78, label %80

78:                                               ; preds = %76
  %79 = load ptr, ptr %12, align 8
  br label %124

80:                                               ; preds = %76
  %81 = load ptr, ptr %13, align 8
  %82 = icmp eq ptr %81, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %82, label %83, label %114

83:                                               ; preds = %80
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 2
  %85 = load i32, ptr %84, align 4
  %86 = and i32 %85, 1073741824
  %87 = icmp ne i32 %86, 0
  br i1 %87, label %88, label %104

88:                                               ; preds = %83
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 4
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %94

92:                                               ; preds = %88
  invoke void %90(ptr %3, ptr %12)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %92
  %93 = load ptr, ptr %3, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc22, %.noexc9
  %102 = phi ptr [ %97, %.noexc9 ], [ %93, %.noexc22 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc2
  %112 = phi ptr [ %107, %.noexc2 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc7
  %122 = phi ptr [ %117, %.noexc7 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc11, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc11 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc13, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc13 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc20, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc20 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc15, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc15 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc10, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc10 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc18, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc18 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc12, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc12 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %4, align 8
  %148 = load double, ptr %6, align 8
  %149 = invoke ptr @mk(double %148)
          to label %150 unwind label %endcatch.pad

150:                                              ; preds = %147
  %151 = load ptr, ptr %4, align 8
  call void @__cxa_end_catch()
  %152 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %151, ptr %152, align 8
  invoke void @__cxa_throw(ptr %152, ptr @_ZTIPv, ptr null)
          to label %295 unwind label %154

endcatch.pad:                                     ; preds = %147
  %153 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

154:                                              ; preds = %150
  %155 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %154
  %156 = phi { ptr, i32 } [ %155, %154 ], [ %153, %endcatch.pad ]
  %157 = extractvalue { ptr, i32 } %156, 0
  %158 = call ptr @__cxa_begin_catch(ptr %157)
  %159 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %.caught
  %160 = icmp eq ptr %159, @_ZTIi
  br i1 %160, label %161, label %168

161:                                              ; preds = %.noexc
  %162 = load i32, ptr %158, align 4
  %163 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %161
  %164 = getelementptr i8, ptr %163, i64 8
  %165 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 0
  store i64 4, ptr %165, align 8
  %166 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %166, align 8
  %167 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 2
  store i32 %162, ptr %167, align 4
  br label %291

168:                                              ; preds = %.noexc
  %169 = icmp eq ptr %159, @_ZTIj
  br i1 %169, label %170, label %177

170:                                              ; preds = %168
  %171 = load i32, ptr %158, align 4
  %172 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %170
  %173 = getelementptr i8, ptr %172, i64 8
  %174 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 0
  store i64 4, ptr %174, align 8
  %175 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %175, align 8
  %176 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 2
  store i32 %171, ptr %176, align 4
  br label %288

177:                                              ; preds = %168
  %178 = icmp eq ptr %159, @_ZTId
  br i1 %178, label %179, label %186

179:                                              ; preds = %177
  %180 = load double, ptr %158, align 8
  %181 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %179
  %182 = getelementptr i8, ptr %181, i64 8
  %183 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 0
  store i64 8, ptr %183, align 8
  %184 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %184, align 8
  %185 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 2
  store double %180, ptr %185, align 8
  br label %285

186:                                              ; preds = %177
  %187 = icmp eq ptr %159, @_ZTIf
  br i1 %187, label %188, label %195

188:                                              ; preds = %186
  %189 = load float, ptr %158, align 4
  %190 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %188
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 0
  store i64 4, ptr %192, align 8
  %193 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %193, align 8
  %194 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 2
  store float %189, ptr %194, align 4
  br label %282

195:                                              ; preds = %186
  %196 = icmp eq ptr %159, @_ZTIb
  br i1 %196, label %197, label %204

197:                                              ; preds = %195
  %198 = load i1, ptr %158, align 1
  %199 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %197
  %200 = getelementptr i8, ptr %199, i64 8
  %201 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 0
  store i64 1, ptr %201, align 8
  %202 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %202, align 8
  %203 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 2
  store i1 %198, ptr %203, align 1
  br label %279

204:                                              ; preds = %195
  %205 = icmp eq ptr %159, @_ZTIx
  br i1 %205, label %206, label %213

206:                                              ; preds = %204
  %207 = load i64, ptr %158, align 8
  %208 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %206
  %209 = getelementptr i8, ptr %208, i64 8
  %210 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 0
  store i64 8, ptr %210, align 8
  %211 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %211, align 8
  %212 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 2
  store i64 %207, ptr %212, align 8
  br label %276

213:                                              ; preds = %204
  %214 = icmp eq ptr %159, @_ZTIPc
  br i1 %214, label %215, label %222

215:                                              ; preds = %213
  %216 = load ptr, ptr %158, align 8
  %217 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %215
  %218 = getelementptr i8, ptr %217, i64 8
  %219 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 0
  store i64 8, ptr %219, align 8
  %220 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %220, align 8
  %221 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 2
  store ptr %216, ptr %221, align 8
  br label %273

222:                                              ; preds = %213
  %223 = icmp eq ptr %159, @_ZTIPv
  br i1 %223, label %224, label %226

224:                                              ; preds = %222
  %225 = load ptr, ptr %158, align 8
  br label %270

226:                                              ; preds = %222
  %227 = load ptr, ptr %159, align 8
  %228 = icmp eq ptr %227, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %228, label %229, label %260

229:                                              ; preds = %226
  %230 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %159, i32 0, i32 2
  %231 = load i32, ptr %230, align 4
  %232 = and i32 %231, 1073741824
  %233 = icmp ne i32 %232, 0
  br i1 %233, label %234, label %250

234:                                              ; preds = %229
  %235 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %159, i32 0, i32 4
  %236 = load ptr, ptr %235, align 8
  %237 = icmp ne ptr %236, null
  br i1 %237, label %238, label %240

238:                                              ; preds = %234
  invoke void %236(ptr %2, ptr %158)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %238
  %239 = load ptr, ptr %2, align 8
  br label %247

240:                                              ; preds = %234
  %241 = load ptr, ptr %158, align 8
  %242 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %240
  %243 = getelementptr i8, ptr %242, i64 8
  %244 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 0
  store i64 8, ptr %244, align 8
  %245 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %245, align 8
  %246 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 2
  store ptr %241, ptr %246, align 8
  br label %247

247:                                              ; preds = %.noexc17, %.noexc1
  %248 = phi ptr [ %243, %.noexc1 ], [ %239, %.noexc17 ]
  br label %249

249:                                              ; preds = %247
  br label %257

250:                                              ; preds = %229
  %251 = load ptr, ptr %158, align 8
  %252 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %250
  %253 = getelementptr i8, ptr %252, i64 8
  %254 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 0
  store i64 8, ptr %254, align 8
  %255 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %255, align 8
  %256 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 2
  store ptr %251, ptr %256, align 8
  br label %257

257:                                              ; preds = %249, %.noexc5
  %258 = phi ptr [ %253, %.noexc5 ], [ %248, %249 ]
  br label %259

259:                                              ; preds = %257
  br label %267

260:                                              ; preds = %226
  %261 = load ptr, ptr %158, align 8
  %262 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %260
  %263 = getelementptr i8, ptr %262, i64 8
  %264 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 0
  store i64 8, ptr %264, align 8
  %265 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %265, align 8
  %266 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 2
  store ptr %261, ptr %266, align 8
  br label %267

267:                                              ; preds = %259, %.noexc3
  %268 = phi ptr [ %263, %.noexc3 ], [ %258, %259 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %224, %269
  %271 = phi ptr [ %268, %269 ], [ %225, %224 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc8, %272
  %274 = phi ptr [ %271, %272 ], [ %218, %.noexc8 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc19, %275
  %277 = phi ptr [ %274, %275 ], [ %209, %.noexc19 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc21, %278
  %280 = phi ptr [ %277, %278 ], [ %200, %.noexc21 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc23, %281
  %283 = phi ptr [ %280, %281 ], [ %191, %.noexc23 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc6, %284
  %286 = phi ptr [ %283, %284 ], [ %182, %.noexc6 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc14, %287
  %289 = phi ptr [ %286, %287 ], [ %173, %.noexc14 ]
  br label %290

290:                                              ; preds = %288
  br label %291

291:                                              ; preds = %.noexc4, %290
  %292 = phi ptr [ %289, %290 ], [ %164, %.noexc4 ]
  br label %293

293:                                              ; preds = %291
  store ptr %292, ptr %5, align 8
  store double 9.000000e+00, ptr %7, align 8
  call void @__cxa_end_catch()
  %294 = load double, ptr %7, align 8
  ret double %294

295:                                              ; preds = %150, %1
  unreachable

endcatch.resume:                                  ; preds = %188, %92, %197, %51, %206, %24, %238, %9, %42, %170, %60, %15, %69, %33, %94, %215, %114, %179, %250, %161, %260, %104, %240, %.caught
  %296 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %296
}

; Function Attrs: mustprogress
define double @finallyAfterThrow(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  store double %0, ptr %5, align 8
  store double 0.000000e+00, ptr %4, align 8
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 5, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIi, ptr null)
          to label %162 unwind label %8

8:                                                ; preds = %1
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = load double, ptr %5, align 8
  %13 = invoke ptr @mk(double %12)
          to label %14 unwind label %endcatch.pad

14:                                               ; preds = %8
  %15 = getelementptr %R, ptr %13, i32 0, i32 1
  %16 = load double, ptr %15, align 8
  store double %16, ptr %4, align 8
  invoke void @__cxa_rethrow()
          to label %162 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %14, %8
  %17 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

18:                                               ; No predecessors!
  %19 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %18
  %20 = phi { ptr, i32 } [ %19, %18 ], [ %17, %endcatch.pad ]
  %21 = extractvalue { ptr, i32 } %20, 0
  %22 = call ptr @__cxa_begin_catch(ptr %21)
  %23 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.caught
  %24 = icmp eq ptr %23, @_ZTIi
  br i1 %24, label %25, label %32

25:                                               ; preds = %.noexc1
  %26 = load i32, ptr %22, align 4
  %27 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %25
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 0
  store i64 4, ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 2
  store i32 %26, ptr %31, align 4
  br label %155

32:                                               ; preds = %.noexc1
  %33 = icmp eq ptr %23, @_ZTIj
  br i1 %33, label %34, label %41

34:                                               ; preds = %32
  %35 = load i32, ptr %22, align 4
  %36 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %34
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 0
  store i64 4, ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %39, align 8
  %40 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 2
  store i32 %35, ptr %40, align 4
  br label %152

41:                                               ; preds = %32
  %42 = icmp eq ptr %23, @_ZTId
  br i1 %42, label %43, label %50

43:                                               ; preds = %41
  %44 = load double, ptr %22, align 8
  %45 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %43
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr { i64, ptr, double }, ptr %46, i32 0, i32 0
  store i64 8, ptr %47, align 8
  %48 = getelementptr { i64, ptr, double }, ptr %46, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %48, align 8
  %49 = getelementptr { i64, ptr, double }, ptr %46, i32 0, i32 2
  store double %44, ptr %49, align 8
  br label %149

50:                                               ; preds = %41
  %51 = icmp eq ptr %23, @_ZTIf
  br i1 %51, label %52, label %59

52:                                               ; preds = %50
  %53 = load float, ptr %22, align 4
  %54 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = getelementptr { i64, ptr, float }, ptr %55, i32 0, i32 0
  store i64 4, ptr %56, align 8
  %57 = getelementptr { i64, ptr, float }, ptr %55, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %57, align 8
  %58 = getelementptr { i64, ptr, float }, ptr %55, i32 0, i32 2
  store float %53, ptr %58, align 4
  br label %146

59:                                               ; preds = %50
  %60 = icmp eq ptr %23, @_ZTIb
  br i1 %60, label %61, label %68

61:                                               ; preds = %59
  %62 = load i1, ptr %22, align 1
  %63 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %61
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr { i64, ptr, i1 }, ptr %64, i32 0, i32 0
  store i64 1, ptr %65, align 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %64, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %66, align 8
  %67 = getelementptr { i64, ptr, i1 }, ptr %64, i32 0, i32 2
  store i1 %62, ptr %67, align 1
  br label %143

68:                                               ; preds = %59
  %69 = icmp eq ptr %23, @_ZTIx
  br i1 %69, label %70, label %77

70:                                               ; preds = %68
  %71 = load i64, ptr %22, align 8
  %72 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %70
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr { i64, ptr, i64 }, ptr %73, i32 0, i32 0
  store i64 8, ptr %74, align 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %73, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %75, align 8
  %76 = getelementptr { i64, ptr, i64 }, ptr %73, i32 0, i32 2
  store i64 %71, ptr %76, align 8
  br label %140

77:                                               ; preds = %68
  %78 = icmp eq ptr %23, @_ZTIPc
  br i1 %78, label %79, label %86

79:                                               ; preds = %77
  %80 = load ptr, ptr %22, align 8
  %81 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %79
  %82 = getelementptr i8, ptr %81, i64 8
  %83 = getelementptr { i64, ptr, ptr }, ptr %82, i32 0, i32 0
  store i64 8, ptr %83, align 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %82, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %84, align 8
  %85 = getelementptr { i64, ptr, ptr }, ptr %82, i32 0, i32 2
  store ptr %80, ptr %85, align 8
  br label %137

86:                                               ; preds = %77
  %87 = icmp eq ptr %23, @_ZTIPv
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = load ptr, ptr %22, align 8
  br label %134

90:                                               ; preds = %86
  %91 = load ptr, ptr %23, align 8
  %92 = icmp eq ptr %91, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %92, label %93, label %124

93:                                               ; preds = %90
  %94 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %23, i32 0, i32 2
  %95 = load i32, ptr %94, align 4
  %96 = and i32 %95, 1073741824
  %97 = icmp ne i32 %96, 0
  br i1 %97, label %98, label %114

98:                                               ; preds = %93
  %99 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %23, i32 0, i32 4
  %100 = load ptr, ptr %99, align 8
  %101 = icmp ne ptr %100, null
  br i1 %101, label %102, label %104

102:                                              ; preds = %98
  invoke void %100(ptr %2, ptr %22)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %102
  %103 = load ptr, ptr %2, align 8
  br label %111

104:                                              ; preds = %98
  %105 = load ptr, ptr %22, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %.noexc7, %.noexc10
  %112 = phi ptr [ %107, %.noexc10 ], [ %103, %.noexc7 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %93
  %115 = load ptr, ptr %22, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc11
  %122 = phi ptr [ %117, %.noexc11 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %131

124:                                              ; preds = %90
  %125 = load ptr, ptr %22, align 8
  %126 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %124
  %127 = getelementptr i8, ptr %126, i64 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %127, i32 0, i32 0
  store i64 8, ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %127, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %129, align 8
  %130 = getelementptr { i64, ptr, ptr }, ptr %127, i32 0, i32 2
  store ptr %125, ptr %130, align 8
  br label %131

131:                                              ; preds = %123, %.noexc6
  %132 = phi ptr [ %127, %.noexc6 ], [ %122, %123 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %88, %133
  %135 = phi ptr [ %132, %133 ], [ %89, %88 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc9, %136
  %138 = phi ptr [ %135, %136 ], [ %82, %.noexc9 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc8, %139
  %141 = phi ptr [ %138, %139 ], [ %73, %.noexc8 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc4, %142
  %144 = phi ptr [ %141, %142 ], [ %64, %.noexc4 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc2, %145
  %147 = phi ptr [ %144, %145 ], [ %55, %.noexc2 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc, %148
  %150 = phi ptr [ %147, %148 ], [ %46, %.noexc ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc5, %151
  %153 = phi ptr [ %150, %151 ], [ %37, %.noexc5 ]
  br label %154

154:                                              ; preds = %152
  br label %155

155:                                              ; preds = %.noexc3, %154
  %156 = phi ptr [ %153, %154 ], [ %28, %.noexc3 ]
  br label %157

157:                                              ; preds = %155
  store ptr %156, ptr %3, align 8
  %158 = load double, ptr %4, align 8
  %159 = fadd double %158, 1.000000e+02
  store double %159, ptr %4, align 8
  call void @__cxa_end_catch()
  %160 = load double, ptr %4, align 8
  store double %160, ptr %6, align 8
  %161 = load double, ptr %6, align 8
  ret double %161

162:                                              ; preds = %14, %1
  unreachable

endcatch.resume:                                  ; preds = %114, %104, %79, %70, %102, %124, %34, %61, %25, %52, %.caught, %43
  %163 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %163
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  br label %2

2:                                                ; preds = %43, %0
  %3 = load i32, ptr %1, align 4
  %4 = icmp slt i32 %3, 300
  br i1 %4, label %5, label %46

5:                                                ; preds = %2
  %6 = load i32, ptr %1, align 4
  %7 = sitofp i32 %6 to double
  %8 = call double @throwOutOfCatch(double %7)
  %9 = fcmp oeq double %8, 7.000000e+00
  br i1 %9, label %10, label %48

10:                                               ; preds = %5
  %11 = load i32, ptr %1, align 4
  %12 = sitofp i32 %11 to double
  %13 = call double @returnFromFinally(double %12)
  %14 = load i32, ptr %1, align 4
  %15 = add i32 %14, 1
  %16 = sitofp i32 %15 to double
  %17 = fcmp oeq double %13, %16
  br i1 %17, label %18, label %50

18:                                               ; preds = %10
  %19 = call double @breakFromFinally(double 1.000000e+01)
  %20 = fcmp oeq double %19, 1.200000e+01
  br i1 %20, label %21, label %52

21:                                               ; preds = %18
  %22 = load i32, ptr %1, align 4
  %23 = sitofp i32 %22 to double
  %24 = call double @nestedTryInCatch(double %23)
  %25 = load i32, ptr %1, align 4
  %26 = mul i32 %25, 2
  %27 = add i32 %26, 2
  %28 = sitofp i32 %27 to double
  %29 = fcmp oeq double %24, %28
  br i1 %29, label %30, label %54

30:                                               ; preds = %21
  %31 = load i32, ptr %1, align 4
  %32 = sitofp i32 %31 to double
  %33 = call double @rethrow(double %32)
  %34 = fcmp oeq double %33, 9.000000e+00
  br i1 %34, label %35, label %56

35:                                               ; preds = %30
  %36 = load i32, ptr %1, align 4
  %37 = sitofp i32 %36 to double
  %38 = call double @finallyAfterThrow(double %37)
  %39 = load i32, ptr %1, align 4
  %40 = add i32 %39, 100
  %41 = sitofp i32 %40 to double
  %42 = fcmp oeq double %38, %41
  br i1 %42, label %43, label %58

43:                                               ; preds = %35
  %44 = load i32, ptr %1, align 4
  %45 = add i32 %44, 1
  store i32 %45, ptr %1, align 4
  br label %2

46:                                               ; preds = %2
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

47:                                               ; preds = %58, %56, %54, %52, %50, %48
  unreachable

48:                                               ; preds = %5
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4502274137427804027, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_642979808102785306, i64 8), i32 41, ptr null)
  br label %47

50:                                               ; preds = %10
  %51 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11328015321383437826, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_642979808102785306, i64 8), i32 42, ptr null)
  br label %47

52:                                               ; preds = %18
  %53 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15098827765242065975, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_642979808102785306, i64 8), i32 43, ptr null)
  br label %47

54:                                               ; preds = %21
  %55 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1165181803863095751, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_642979808102785306, i64 8), i32 44, ptr null)
  br label %47

56:                                               ; preds = %30
  %57 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15111080100605800680, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_642979808102785306, i64 8), i32 45, ptr null)
  br label %47

58:                                               ; preds = %35
  %59 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18370820746968551330, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_642979808102785306, i64 8), i32 46, ptr null)
  br label %47
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

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress noinline }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

