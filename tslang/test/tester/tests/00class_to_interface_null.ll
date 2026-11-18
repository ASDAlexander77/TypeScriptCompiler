; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, double }
%D = type { ptr, double }

$C..size = comdat any

$D..size = comdat any

@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_12638219086392860115 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFD\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_13245662913593092714 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field read first\00", align 8
@f_1661742625200120008 = internal constant [115 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00class_to_interface_null.ts\00", align 8
@m_7312824761881269932 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null class object\00", align 8
@m_8070126613336195185 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null class object, loosely\00", align 8
@m_10031112863601155095 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null class object is false\00", align 8
@m_13851024118215216466 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null object of a class naming the interface\00", align 8
@m_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_13831382456710198045 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined is loosely null\00", align 8
@m_2495249626774661283 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined\00", align 8
@m_7775772421422883010 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa method of a null class object throws\00", align 8
@m_14477900776849559952 = internal constant [73 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa method of a null object of a class naming the interface throws\00", align 8
@m_3102676674652066716 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa method of null throws\00", align 8
@m_8420587522146389367 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa method of undefined throws\00", align 8
@m_9037292245450521232 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa method of a class object\00", align 8
@m_10978865040198224589 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan argument\00", align 8
@m_16748551001661421664 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null argument\00", align 8
@m_11878420861875365987 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan argument of a class naming the interface\00", align 8
@m_585771283767515882 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa null argument of a class naming the interface\00", align 8
@m_13239218876778101673 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo objects\00", align 8
@m_8769865063380825445 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe same object\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 16, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C.get, ptr @C..rtti, ptr @C..size }
@D..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638219086392860115, i64 8)
@D..size = global i64 16, comdat
@D.I..vtbl = private global { ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_8297331101459474202_class, i64 40), ptr @D..instanceOf, ptr @D.get, ptr inttoptr (i64 8 to ptr) }
@td_8297331101459474202_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_8297331101459474202, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@D..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @D..instanceOf, ptr @D.I..vtbl, ptr @D..new, ptr @D.get, ptr @D..rtti, ptr @D..size }
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@C.I..vtbl = private global { ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12330864546263951727_class, i64 40), ptr @C.I..vtbl..instanceOf.dispatch, ptr @C.I..vtbl.get.dispatch, ptr inttoptr (i64 8 to ptr) }
@td_12330864546263951727_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_12330864546263951727, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@I.null..vtbl = private global { ptr, ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @I.null.get, ptr null }
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

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @__cxa_end_catch()

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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_12330864546263951727(ptr %0) {
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

define internal void @tsrel_8297331101459474202(ptr %0) {
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

; Function Attrs: mustprogress
define i1 @C..instanceOf(ptr %0, ptr %1) #1 {
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
define ptr @C..new() #1 {
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
define void @C.constructor(ptr %0, double %1) #1 {
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
define double @C.get(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %C, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, 1.000000e+01
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress
define i1 @D..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @D..rtti, align 8
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
define ptr @D..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %D, ptr %3, i32 0, i32 0
  store ptr @D..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @D.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %D, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define double @D.get(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %D, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, 1.000000e+02
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress noinline
define ptr @mkC(i1 %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  store i1 %0, ptr %4, align 1
  store double %1, ptr %3, align 8
  %6 = load i1, ptr %4, align 1
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %C, ptr %9, i32 0, i32 0
  store ptr @C..vtbl, ptr %10, align 8
  %11 = load double, ptr %3, align 8
  call void @C.constructor(ptr %9, double %11)
  br label %13

12:                                               ; preds = %2
  br label %13

13:                                               ; preds = %7, %12
  %14 = phi ptr [ null, %12 ], [ %9, %7 ]
  store ptr %14, ptr %5, align 8
  %15 = load ptr, ptr %5, align 8
  ret ptr %15
}

; Function Attrs: mustprogress noinline
define ptr @mkD(i1 %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  store i1 %0, ptr %4, align 1
  store double %1, ptr %3, align 8
  %6 = load i1, ptr %4, align 1
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %D, ptr %9, i32 0, i32 0
  store ptr @D..vtbl, ptr %10, align 8
  %11 = load double, ptr %3, align 8
  call void @D.constructor(ptr %9, double %11)
  br label %13

12:                                               ; preds = %2
  br label %13

13:                                               ; preds = %7, %12
  %14 = phi ptr [ null, %12 ], [ %9, %7 ]
  store ptr %14, ptr %5, align 8
  %15 = load ptr, ptr %5, align 8
  ret ptr %15
}

; Function Attrs: mustprogress
define double @readOf({ ptr, ptr } %0) #1 {
  %2 = alloca { ptr, ptr }, align 8
  %3 = alloca double, align 8
  store { ptr, ptr } %0, ptr %2, align 8
  %4 = load { ptr, ptr }, ptr %2, align 8
  %5 = extractvalue { ptr, ptr } %4, 1
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %1
  store double -1.000000e+00, ptr %3, align 8
  br label %49

9:                                                ; preds = %1
  %10 = load { ptr, ptr }, ptr %2, align 8
  %11 = extractvalue { ptr, ptr } %10, 0
  %12 = extractvalue { ptr, ptr } %10, 1
  %13 = getelementptr ptr, ptr %11, i32 3
  %14 = load ptr, ptr %13, align 8
  %15 = ptrtoint ptr %12 to i64
  %16 = ptrtoint ptr %14 to i64
  %17 = add i64 %15, %16
  %18 = inttoptr i64 %17 to ptr
  %19 = extractvalue { ptr, ptr } %10, 0
  %20 = extractvalue { ptr, ptr } %10, 1
  %21 = getelementptr ptr, ptr %19, i32 3
  %22 = load ptr, ptr %21, align 8
  %23 = ptrtoint ptr %22 to i64
  %24 = and i64 %23, -9223372036854775808
  %25 = icmp ne i64 %24, 0
  br i1 %25, label %26, label %32

26:                                               ; preds = %9
  %27 = sub i64 %23, -9223372036854775808
  %28 = inttoptr i64 %27 to ptr
  %29 = getelementptr { ptr, ptr }, ptr %28, i32 0, i32 0
  %30 = load ptr, ptr %29, align 8
  %31 = call double %30(ptr %20)
  br label %34

32:                                               ; preds = %9
  %33 = load double, ptr %18, align 8
  br label %34

34:                                               ; preds = %26, %32
  %35 = phi double [ %33, %32 ], [ %31, %26 ]
  br label %36

36:                                               ; preds = %34
  %37 = load { ptr, ptr }, ptr %2, align 8
  %38 = extractvalue { ptr, ptr } %37, 0
  %39 = extractvalue { ptr, ptr } %37, 1
  %40 = getelementptr ptr, ptr %38, i32 2
  %41 = load ptr, ptr %40, align 8
  %42 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %41, 0
  %43 = insertvalue { ptr, ptr, ptr } %42, ptr %39, 1
  %44 = insertvalue { ptr, ptr, ptr } %43, ptr null, 2
  %45 = extractvalue { ptr, ptr, ptr } %44, 1
  %46 = extractvalue { ptr, ptr, ptr } %44, 0
  %47 = call double %46(ptr %45)
  %48 = fadd double %35, %47
  store double %48, ptr %3, align 8
  br label %49

49:                                               ; preds = %8, %36
  %50 = load double, ptr %3, align 8
  ret double %50
}

; Function Attrs: mustprogress
define double @call({ ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca { ptr, ptr }, align 8
  %5 = alloca double, align 8
  store { ptr, ptr } %0, ptr %4, align 8
  %6 = load { ptr, ptr }, ptr %4, align 8
  %7 = extractvalue { ptr, ptr } %6, 0
  %8 = extractvalue { ptr, ptr } %6, 1
  %9 = getelementptr ptr, ptr %7, i32 2
  %10 = load ptr, ptr %9, align 8
  %11 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %10, 0
  %12 = insertvalue { ptr, ptr, ptr } %11, ptr %8, 1
  %13 = insertvalue { ptr, ptr, ptr } %12, ptr null, 2
  %14 = extractvalue { ptr, ptr, ptr } %13, 1
  %15 = extractvalue { ptr, ptr, ptr } %13, 0
  %16 = invoke double %15(ptr %14)
          to label %17 unwind label %18

17:                                               ; preds = %1
  store double %16, ptr %5, align 8
  br label %157

18:                                               ; preds = %1
  %19 = landingpad { ptr, i32 }
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  %22 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %18
  %23 = icmp eq ptr %22, @_ZTIi
  br i1 %23, label %24, label %31

24:                                               ; preds = %.noexc8
  %25 = load i32, ptr %21, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %154

31:                                               ; preds = %.noexc8
  %32 = icmp eq ptr %22, @_ZTIj
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load i32, ptr %21, align 4
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 0
  store i64 4, ptr %37, align 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 2
  store i32 %34, ptr %39, align 4
  br label %151

40:                                               ; preds = %31
  %41 = icmp eq ptr %22, @_ZTId
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load double, ptr %21, align 8
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 0
  store i64 8, ptr %46, align 8
  %47 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 2
  store double %43, ptr %48, align 8
  br label %148

49:                                               ; preds = %40
  %50 = icmp eq ptr %22, @_ZTIf
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load float, ptr %21, align 4
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 0
  store i64 4, ptr %55, align 8
  %56 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 2
  store float %52, ptr %57, align 4
  br label %145

58:                                               ; preds = %49
  %59 = icmp eq ptr %22, @_ZTIb
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i1, ptr %21, align 1
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 0
  store i64 1, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 2
  store i1 %61, ptr %66, align 1
  br label %142

67:                                               ; preds = %58
  %68 = icmp eq ptr %22, @_ZTIx
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load i64, ptr %21, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 2
  store i64 %70, ptr %75, align 8
  br label %139

76:                                               ; preds = %67
  %77 = icmp eq ptr %22, @_ZTIPc
  br i1 %77, label %78, label %85

78:                                               ; preds = %76
  %79 = load ptr, ptr %21, align 8
  %80 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %78
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 0
  store i64 8, ptr %82, align 8
  %83 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %83, align 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 2
  store ptr %79, ptr %84, align 8
  br label %136

85:                                               ; preds = %76
  %86 = icmp eq ptr %22, @_ZTIPv
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = load ptr, ptr %21, align 8
  br label %133

89:                                               ; preds = %85
  %90 = load ptr, ptr %22, align 8
  %91 = icmp eq ptr %90, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %91, label %92, label %123

92:                                               ; preds = %89
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 2
  %94 = load i32, ptr %93, align 4
  %95 = and i32 %94, 1073741824
  %96 = icmp ne i32 %95, 0
  br i1 %96, label %97, label %113

97:                                               ; preds = %92
  %98 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 4
  %99 = load ptr, ptr %98, align 8
  %100 = icmp ne ptr %99, null
  br i1 %100, label %101, label %103

101:                                              ; preds = %97
  invoke void %99(ptr %2, ptr %21)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %101
  %102 = load ptr, ptr %2, align 8
  br label %110

103:                                              ; preds = %97
  %104 = load ptr, ptr %21, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %.noexc3, %.noexc5
  %111 = phi ptr [ %106, %.noexc5 ], [ %102, %.noexc3 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %92
  %114 = load ptr, ptr %21, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc1
  %121 = phi ptr [ %116, %.noexc1 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %89
  %124 = load ptr, ptr %21, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc6
  %131 = phi ptr [ %126, %.noexc6 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %87, %132
  %134 = phi ptr [ %131, %132 ], [ %88, %87 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc10, %135
  %137 = phi ptr [ %134, %135 ], [ %81, %.noexc10 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc2, %138
  %140 = phi ptr [ %137, %138 ], [ %72, %.noexc2 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc4, %141
  %143 = phi ptr [ %140, %141 ], [ %63, %.noexc4 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc, %144
  %146 = phi ptr [ %143, %144 ], [ %54, %.noexc ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc11, %147
  %149 = phi ptr [ %146, %147 ], [ %45, %.noexc11 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc9, %150
  %152 = phi ptr [ %149, %150 ], [ %36, %.noexc9 ]
  br label %153

153:                                              ; preds = %151
  br label %154

154:                                              ; preds = %.noexc7, %153
  %155 = phi ptr [ %152, %153 ], [ %27, %.noexc7 ]
  br label %156

156:                                              ; preds = %154
  store ptr %155, ptr %3, align 8
  store double -2.000000e+00, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %157

157:                                              ; preds = %17, %156
  %158 = load double, ptr %5, align 8
  ret double %158

endcatch.resume:                                  ; preds = %42, %78, %33, %18, %24, %123, %103, %60, %101, %69, %113, %51
  %159 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %159
}

; Function Attrs: mustprogress
define internal i1 @C.I..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  %4 = getelementptr %C, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = call i1 %7(ptr %0, ptr %1)
  store i1 %8, ptr %3, align 1
  %9 = load i1, ptr %3, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define internal double @C.I..vtbl.get.dispatch(ptr %0) #1 {
  %2 = alloca double, align 8
  %3 = getelementptr %C, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr ptr, ptr %4, i32 2
  %6 = load ptr, ptr %5, align 8
  %7 = call double %6(ptr %0)
  store double %7, ptr %2, align 8
  %8 = load double, ptr %2, align 8
  ret double %8
}

; Function Attrs: mustprogress
define double @I.null.get(ptr %0) #4 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call ptr @mkC(i1 true, double 1.000000e+00)
  %2 = ptrtoint ptr %1 to i64
  %3 = icmp ne i64 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  br label %6

5:                                                ; preds = %0
  br label %6

6:                                                ; preds = %4, %5
  %7 = phi ptr [ @I.null..vtbl, %5 ], [ @C.I..vtbl, %4 ]
  %8 = insertvalue { ptr, ptr } undef, ptr %7, 0
  %9 = insertvalue { ptr, ptr } %8, ptr %1, 1
  %10 = extractvalue { ptr, ptr } %9, 1
  %11 = ptrtoint ptr %10 to i64
  %12 = icmp ne i64 %11, 0
  br i1 %12, label %13, label %41

13:                                               ; preds = %6
  %14 = extractvalue { ptr, ptr } %9, 0
  %15 = extractvalue { ptr, ptr } %9, 1
  %16 = getelementptr ptr, ptr %14, i32 3
  %17 = load ptr, ptr %16, align 8
  %18 = ptrtoint ptr %15 to i64
  %19 = ptrtoint ptr %17 to i64
  %20 = add i64 %18, %19
  %21 = inttoptr i64 %20 to ptr
  %22 = extractvalue { ptr, ptr } %9, 0
  %23 = extractvalue { ptr, ptr } %9, 1
  %24 = getelementptr ptr, ptr %22, i32 3
  %25 = load ptr, ptr %24, align 8
  %26 = ptrtoint ptr %25 to i64
  %27 = and i64 %26, -9223372036854775808
  %28 = icmp ne i64 %27, 0
  br i1 %28, label %29, label %35

29:                                               ; preds = %13
  %30 = sub i64 %26, -9223372036854775808
  %31 = inttoptr i64 %30 to ptr
  %32 = getelementptr { ptr, ptr }, ptr %31, i32 0, i32 0
  %33 = load ptr, ptr %32, align 8
  %34 = call double %33(ptr %23)
  br label %37

35:                                               ; preds = %13
  %36 = load double, ptr %21, align 8
  br label %37

37:                                               ; preds = %29, %35
  %38 = phi double [ %36, %35 ], [ %34, %29 ]
  br label %39

39:                                               ; preds = %37
  %40 = fcmp oeq double %38, 1.000000e+00
  br label %41

41:                                               ; preds = %39, %6
  %42 = phi i1 [ %40, %39 ], [ %12, %6 ]
  br i1 %42, label %43, label %199

43:                                               ; preds = %41
  %44 = call ptr @mkC(i1 false, double 2.000000e+00)
  %45 = ptrtoint ptr %44 to i64
  %46 = icmp ne i64 %45, 0
  br i1 %46, label %47, label %48

47:                                               ; preds = %43
  br label %49

48:                                               ; preds = %43
  br label %49

49:                                               ; preds = %47, %48
  %50 = phi ptr [ @I.null..vtbl, %48 ], [ @C.I..vtbl, %47 ]
  %51 = insertvalue { ptr, ptr } undef, ptr %50, 0
  %52 = insertvalue { ptr, ptr } %51, ptr %44, 1
  %53 = extractvalue { ptr, ptr } %52, 1
  %54 = ptrtoint ptr %53 to i64
  %55 = icmp eq i64 %54, 0
  br i1 %55, label %56, label %201

56:                                               ; preds = %49
  %57 = extractvalue { ptr, ptr } %52, 1
  %58 = ptrtoint ptr %57 to i64
  %59 = icmp eq i64 %58, 0
  br i1 %59, label %60, label %203

60:                                               ; preds = %56
  %61 = extractvalue { ptr, ptr } %52, 1
  %62 = ptrtoint ptr %61 to i64
  %63 = icmp ne i64 %62, 0
  %64 = xor i1 %63, true
  br i1 %64, label %65, label %205

65:                                               ; preds = %60
  %66 = call ptr @mkD(i1 false, double 3.000000e+00)
  %67 = ptrtoint ptr %66 to i64
  %68 = icmp ne i64 %67, 0
  br i1 %68, label %69, label %74

69:                                               ; preds = %65
  %70 = getelementptr %D, ptr %66, i32 0, i32 0
  %71 = load ptr, ptr %70, align 8
  %72 = getelementptr ptr, ptr %71, i32 1
  %73 = load ptr, ptr %72, align 8
  br label %75

74:                                               ; preds = %65
  br label %75

75:                                               ; preds = %69, %74
  %76 = phi ptr [ @I.null..vtbl, %74 ], [ %73, %69 ]
  %77 = insertvalue { ptr, ptr } undef, ptr %76, 0
  %78 = insertvalue { ptr, ptr } %77, ptr %66, 1
  %79 = extractvalue { ptr, ptr } %78, 1
  %80 = ptrtoint ptr %79 to i64
  %81 = icmp eq i64 %80, 0
  br i1 %81, label %82, label %207

82:                                               ; preds = %75
  br i1 true, label %83, label %209

83:                                               ; preds = %82
  br i1 true, label %84, label %211

84:                                               ; preds = %83
  br i1 true, label %85, label %213

85:                                               ; preds = %84
  %86 = call double @call({ ptr, ptr } %52)
  %87 = fcmp oeq double %86, -2.000000e+00
  br i1 %87, label %88, label %215

88:                                               ; preds = %85
  %89 = call double @call({ ptr, ptr } %78)
  %90 = fcmp oeq double %89, -2.000000e+00
  br i1 %90, label %91, label %217

91:                                               ; preds = %88
  %92 = call double @call({ ptr, ptr } { ptr @I.null..vtbl, ptr null })
  %93 = fcmp oeq double %92, -2.000000e+00
  br i1 %93, label %94, label %219

94:                                               ; preds = %91
  %95 = call double @call({ ptr, ptr } { ptr @I.null..vtbl, ptr null })
  %96 = fcmp oeq double %95, -2.000000e+00
  br i1 %96, label %97, label %221

97:                                               ; preds = %94
  %98 = call ptr @mkC(i1 true, double 5.000000e+00)
  %99 = ptrtoint ptr %98 to i64
  %100 = icmp ne i64 %99, 0
  br i1 %100, label %101, label %102

101:                                              ; preds = %97
  br label %103

102:                                              ; preds = %97
  br label %103

103:                                              ; preds = %101, %102
  %104 = phi ptr [ @I.null..vtbl, %102 ], [ @C.I..vtbl, %101 ]
  %105 = insertvalue { ptr, ptr } undef, ptr %104, 0
  %106 = insertvalue { ptr, ptr } %105, ptr %98, 1
  %107 = call double @call({ ptr, ptr } %106)
  %108 = fcmp oeq double %107, 5.000000e+01
  br i1 %108, label %109, label %223

109:                                              ; preds = %103
  %110 = call ptr @mkC(i1 true, double 4.000000e+00)
  %111 = ptrtoint ptr %110 to i64
  %112 = icmp ne i64 %111, 0
  br i1 %112, label %113, label %114

113:                                              ; preds = %109
  br label %115

114:                                              ; preds = %109
  br label %115

115:                                              ; preds = %113, %114
  %116 = phi ptr [ @I.null..vtbl, %114 ], [ @C.I..vtbl, %113 ]
  %117 = insertvalue { ptr, ptr } undef, ptr %116, 0
  %118 = insertvalue { ptr, ptr } %117, ptr %110, 1
  %119 = call double @readOf({ ptr, ptr } %118)
  %120 = fcmp oeq double %119, 4.400000e+01
  br i1 %120, label %121, label %225

121:                                              ; preds = %115
  %122 = call ptr @mkC(i1 false, double 4.000000e+00)
  %123 = ptrtoint ptr %122 to i64
  %124 = icmp ne i64 %123, 0
  br i1 %124, label %125, label %126

125:                                              ; preds = %121
  br label %127

126:                                              ; preds = %121
  br label %127

127:                                              ; preds = %125, %126
  %128 = phi ptr [ @I.null..vtbl, %126 ], [ @C.I..vtbl, %125 ]
  %129 = insertvalue { ptr, ptr } undef, ptr %128, 0
  %130 = insertvalue { ptr, ptr } %129, ptr %122, 1
  %131 = call double @readOf({ ptr, ptr } %130)
  %132 = fcmp oeq double %131, -1.000000e+00
  br i1 %132, label %133, label %227

133:                                              ; preds = %127
  %134 = call ptr @mkD(i1 true, double 5.000000e+00)
  %135 = ptrtoint ptr %134 to i64
  %136 = icmp ne i64 %135, 0
  br i1 %136, label %137, label %142

137:                                              ; preds = %133
  %138 = getelementptr %D, ptr %134, i32 0, i32 0
  %139 = load ptr, ptr %138, align 8
  %140 = getelementptr ptr, ptr %139, i32 1
  %141 = load ptr, ptr %140, align 8
  br label %143

142:                                              ; preds = %133
  br label %143

143:                                              ; preds = %137, %142
  %144 = phi ptr [ @I.null..vtbl, %142 ], [ %141, %137 ]
  %145 = insertvalue { ptr, ptr } undef, ptr %144, 0
  %146 = insertvalue { ptr, ptr } %145, ptr %134, 1
  %147 = call double @readOf({ ptr, ptr } %146)
  %148 = fcmp oeq double %147, 5.050000e+02
  br i1 %148, label %149, label %229

149:                                              ; preds = %143
  %150 = call ptr @mkD(i1 false, double 5.000000e+00)
  %151 = ptrtoint ptr %150 to i64
  %152 = icmp ne i64 %151, 0
  br i1 %152, label %153, label %158

153:                                              ; preds = %149
  %154 = getelementptr %D, ptr %150, i32 0, i32 0
  %155 = load ptr, ptr %154, align 8
  %156 = getelementptr ptr, ptr %155, i32 1
  %157 = load ptr, ptr %156, align 8
  br label %159

158:                                              ; preds = %149
  br label %159

159:                                              ; preds = %153, %158
  %160 = phi ptr [ @I.null..vtbl, %158 ], [ %157, %153 ]
  %161 = insertvalue { ptr, ptr } undef, ptr %160, 0
  %162 = insertvalue { ptr, ptr } %161, ptr %150, 1
  %163 = call double @readOf({ ptr, ptr } %162)
  %164 = fcmp oeq double %163, -1.000000e+00
  br i1 %164, label %165, label %231

165:                                              ; preds = %159
  %166 = call ptr @GC_malloc(i64 24)
  %167 = getelementptr i8, ptr %166, i64 8
  %168 = getelementptr %C, ptr %167, i32 0, i32 0
  store ptr @C..vtbl, ptr %168, align 8
  call void @C.constructor(ptr %167, double 6.000000e+00)
  %169 = ptrtoint ptr %167 to i64
  %170 = icmp ne i64 %169, 0
  br i1 %170, label %171, label %172

171:                                              ; preds = %165
  br label %173

172:                                              ; preds = %165
  br label %173

173:                                              ; preds = %171, %172
  %174 = phi ptr [ @I.null..vtbl, %172 ], [ @C.I..vtbl, %171 ]
  %175 = insertvalue { ptr, ptr } undef, ptr %174, 0
  %176 = insertvalue { ptr, ptr } %175, ptr %167, 1
  %177 = call ptr @mkC(i1 true, double 6.000000e+00)
  %178 = ptrtoint ptr %177 to i64
  %179 = icmp ne i64 %178, 0
  br i1 %179, label %180, label %181

180:                                              ; preds = %173
  br label %182

181:                                              ; preds = %173
  br label %182

182:                                              ; preds = %180, %181
  %183 = phi ptr [ @I.null..vtbl, %181 ], [ @C.I..vtbl, %180 ]
  %184 = insertvalue { ptr, ptr } undef, ptr %183, 0
  %185 = insertvalue { ptr, ptr } %184, ptr %177, 1
  %186 = extractvalue { ptr, ptr } %176, 1
  %187 = extractvalue { ptr, ptr } %185, 1
  %188 = ptrtoint ptr %186 to i64
  %189 = ptrtoint ptr %187 to i64
  %190 = icmp ne i64 %188, %189
  br i1 %190, label %191, label %233

191:                                              ; preds = %182
  %192 = extractvalue { ptr, ptr } %176, 1
  %193 = extractvalue { ptr, ptr } %176, 1
  %194 = ptrtoint ptr %192 to i64
  %195 = ptrtoint ptr %193 to i64
  %196 = icmp eq i64 %194, %195
  br i1 %196, label %197, label %235

197:                                              ; preds = %191
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

198:                                              ; preds = %235, %233, %231, %229, %227, %225, %223, %221, %219, %217, %215, %213, %211, %209, %207, %205, %203, %201, %199
  unreachable

199:                                              ; preds = %41
  %200 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13245662913593092714, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 62, ptr null)
  br label %198

201:                                              ; preds = %49
  %202 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7312824761881269932, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 65, ptr null)
  br label %198

203:                                              ; preds = %56
  %204 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8070126613336195185, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 66, ptr null)
  br label %198

205:                                              ; preds = %60
  %206 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10031112863601155095, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 67, ptr null)
  br label %198

207:                                              ; preds = %75
  %208 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13851024118215216466, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 69, ptr null)
  br label %198

209:                                              ; preds = %82
  %210 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6601085983368743140, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 71, ptr null)
  br label %198

211:                                              ; preds = %83
  %212 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13831382456710198045, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 73, ptr null)
  br label %198

213:                                              ; preds = %84
  %214 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2495249626774661283, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 74, ptr null)
  br label %198

215:                                              ; preds = %85
  %216 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7775772421422883010, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 76, ptr null)
  br label %198

217:                                              ; preds = %88
  %218 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14477900776849559952, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 77, ptr null)
  br label %198

219:                                              ; preds = %91
  %220 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3102676674652066716, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 78, ptr null)
  br label %198

221:                                              ; preds = %94
  %222 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8420587522146389367, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 79, ptr null)
  br label %198

223:                                              ; preds = %103
  %224 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9037292245450521232, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 80, ptr null)
  br label %198

225:                                              ; preds = %115
  %226 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10978865040198224589, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 82, ptr null)
  br label %198

227:                                              ; preds = %127
  %228 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16748551001661421664, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 83, ptr null)
  br label %198

229:                                              ; preds = %143
  %230 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11878420861875365987, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 84, ptr null)
  br label %198

231:                                              ; preds = %159
  %232 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_585771283767515882, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 85, ptr null)
  br label %198

233:                                              ; preds = %182
  %234 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13239218876778101673, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 91, ptr null)
  br label %198

235:                                              ; preds = %191
  %236 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8769865063380825445, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1661742625200120008, i64 8), i32 92, ptr null)
  br label %198
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

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress noinline }
attributes #3 = { mustprogress "personality" }
attributes #4 = { mustprogress "__null_interface_vtable" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

