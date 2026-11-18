; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Node = type { ptr, ptr, ptr }
%Holder = type { ptr, ptr }

$"___unbox<!ts.class<Node, !ts.class_storage<Node>>>" = comdat any

$Node..size = comdat any

$Holder..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_7403105002735449741 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNode\00", align 8
@s_11670276633975932207 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHolder\00", align 8
@s_12638116831811436492 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF!\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_10626865055412510189 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_try_let_borrows.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_12638197096160295895 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFh\00", align 8
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_12638193797625411262 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFo\00", align 8
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
@s_12638186101044013785 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_12638194897137039473 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFn\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@m_3019566071980655109 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught\00", align 8
@m_1102633763587963156 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFn alive\00", align 8
@m_3300820206774538237 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtotal\00", align 8
@Node..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7403105002735449741, i64 8)
@Node..size = global i64 24, comdat
@Node..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Node..instanceOf, ptr @Node..new, ptr @Node..rtti, ptr @Node..size }
@Holder..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11670276633975932207, i64 8)
@Holder..size = global i64 16, comdat
@Holder..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Holder..instanceOf, ptr @Holder..new, ptr @Holder..rtti, ptr @Holder..size }
@_ZTIPc = external constant ptr
@a_0_17051528429890938572 = internal constant [0 x ptr] undef
@td_4612575052580656710_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_4612575052580656710, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8

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

define internal void @tsrel_15139385226834798362(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %18

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %18

6:                                                ; preds = %4
  %7 = getelementptr { ptr, i64, i64 }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %2, i32 0, i32 1
  %10 = load i64, ptr %9, align 8
  br label %11

11:                                               ; preds = %14, %6
  %12 = phi i64 [ %16, %14 ], [ 0, %6 ]
  %13 = icmp slt i64 %12, %10
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  %15 = getelementptr ptr, ptr %8, i64 %12
  call void @tsrel_1752011451674293319(ptr %15)
  %16 = add i64 %12, 1
  br label %11

17:                                               ; preds = %11
  call void @__tslang_free_block(ptr %8)
  call void @__tslang_free_block(ptr %2)
  br label %18

18:                                               ; preds = %17, %4, %1
  ret void
}

define internal void @tsrel_4612575052580656710(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %9

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = getelementptr %Node, ptr %2, i32 0, i32 1
  call void @tsrel_15139385226834798362(ptr %7)
  %8 = getelementptr %Node, ptr %2, i32 0, i32 2
  call void @tsrel_1752011451674293319(ptr %8)
  call void @__tslang_free_block(ptr %2)
  br label %9

9:                                                ; preds = %6, %4, %1
  ret void
}

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

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Node..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Node..rtti, align 8
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
define ptr @Node..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 0
  store ptr @Node..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Node.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Node, ptr %5, i32 0, i32 1
  %7 = call ptr @GC_malloc(i64 8)
  %8 = getelementptr i8, ptr %7, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %8, ptr @a_0_17051528429890938572, i64 0, i1 false)
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  store ptr %8, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  store i64 0, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 2
  store i64 0, ptr %13, align 8
  store ptr %10, ptr %6, align 8
  %14 = load ptr, ptr %4, align 8
  %15 = getelementptr %Node, ptr %14, i32 0, i32 2
  %16 = load ptr, ptr %3, align 8
  store ptr %16, ptr %15, align 8
  %17 = load ptr, ptr %4, align 8
  %18 = getelementptr %Node, ptr %17, i32 0, i32 1
  %19 = load ptr, ptr %3, align 8
  %20 = icmp eq ptr %19, null
  %21 = select i1 %20, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %19
  %22 = call i64 @strlen(ptr %21)
  %23 = add i64 1, %22
  %24 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8))
  %25 = add i64 %23, %24
  %26 = add i64 %25, 8
  %27 = call ptr @GC_malloc(i64 %26)
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr i8, ptr %28, i64 0
  %30 = sub i64 %25, 0
  %31 = call ptr @__memcpy_chk(ptr %29, ptr %21, i64 %22, i64 %30)
  %32 = add i64 0, %22
  %33 = getelementptr i8, ptr %28, i64 %32
  %34 = sub i64 %25, %32
  %35 = call ptr @__memcpy_chk(ptr %33, ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8), i64 %24, i64 %34)
  %36 = add i64 %32, %24
  %37 = getelementptr i8, ptr %28, i64 %36
  store i8 0, ptr %37, align 1
  %38 = load ptr, ptr %18, align 8
  %39 = icmp eq ptr %38, null
  br i1 %39, label %40, label %43

40:                                               ; preds = %2
  %41 = call ptr @GC_malloc(i64 32)
  %42 = getelementptr i8, ptr %41, i64 8
  store ptr %42, ptr %18, align 8
  br label %44

43:                                               ; preds = %2
  br label %44

44:                                               ; preds = %40, %43
  %45 = phi ptr [ %38, %43 ], [ %42, %40 ]
  br label %46

46:                                               ; preds = %44
  %47 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 1
  %48 = load i64, ptr %47, align 8
  %49 = add i64 %48, 1
  %50 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 0
  %51 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 2
  %52 = load ptr, ptr %50, align 8
  %53 = load i64, ptr %51, align 8
  %54 = icmp ugt i64 %49, %53
  br i1 %54, label %55, label %79

55:                                               ; preds = %46
  %56 = mul i64 %53, 2
  %57 = icmp uge i64 %56, %49
  %58 = select i1 %57, i64 %56, i64 %49
  %59 = icmp ult i64 %58, 4
  %60 = select i1 %59, i64 4, i64 %58
  %61 = icmp ule i64 %49, 4294967295
  br i1 %61, label %62, label %85

62:                                               ; preds = %55
  %63 = icmp ule i64 %60, 2305843009213693951
  br i1 %63, label %64, label %87

64:                                               ; preds = %62
  %65 = mul i64 8, %60
  %66 = add i64 %65, 8
  %67 = icmp eq ptr %52, null
  %68 = getelementptr i8, ptr %52, i64 -8
  %69 = select i1 %67, ptr null, ptr %68
  %70 = call ptr @GC_realloc(ptr %69, i64 %66)
  %71 = getelementptr i8, ptr %70, i64 8
  %72 = icmp ne ptr %71, null
  %73 = icmp eq i64 %65, 0
  %74 = or i1 %72, %73
  br i1 %74, label %75, label %89

75:                                               ; preds = %64
  %76 = mul i64 8, %53
  %77 = getelementptr i8, ptr %71, i64 %76
  %78 = sub i64 %65, %76
  call void @llvm.memset.p0.i64(ptr %77, i8 0, i64 %78, i1 false)
  store ptr %71, ptr %50, align 8
  store i64 %60, ptr %51, align 8
  br label %80

79:                                               ; preds = %46
  br label %80

80:                                               ; preds = %75, %79
  %81 = phi ptr [ %52, %79 ], [ %71, %75 ]
  br label %82

82:                                               ; preds = %80
  %83 = getelementptr ptr, ptr %81, i64 %48
  store ptr %28, ptr %83, align 8
  store i64 %49, ptr %47, align 8
  ret void

84:                                               ; preds = %89, %87, %85
  unreachable

85:                                               ; preds = %55
  %86 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10626865055412510189, i64 8), i32 8, ptr null)
  br label %84

87:                                               ; preds = %62
  %88 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10626865055412510189, i64 8), i32 8, ptr null)
  br label %84

89:                                               ; preds = %64
  %90 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10626865055412510189, i64 8), i32 8, ptr null)
  br label %84
}

; Function Attrs: mustprogress
define void @Holder.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Holder, ptr %3, i32 0, i32 1
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Node, ptr %6, i32 0, i32 0
  store ptr @Node..vtbl, ptr %7, align 8
  call void @Node.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_12638197096160295895, i64 8))
  store ptr %6, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Holder..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @Holder..new() #2 {
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
define i64 @fromParam(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  store ptr %5, ptr %2, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = getelementptr %Node, ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %7, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr @__ts_empty_array, ptr %8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  %12 = load i64, ptr %11, align 8
  store i64 %12, ptr %4, align 8
  %13 = load i64, ptr %4, align 8
  ret i64 %13
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"___unbox<!ts.class<Node, !ts.class_storage<Node>>>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr { i64, ptr, i8 }, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %1
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %1
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %42

21:                                               ; preds = %20
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 2
  %24 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 0
  %25 = load i64, ptr %24, align 8
  %26 = load i64, ptr %23, align 8
  %27 = icmp uge i64 %25, 8
  %28 = select i1 %27, i64 0, i64 %25
  %29 = mul i64 %28, 8
  %30 = shl i64 1, %29
  %31 = sub i64 %30, 1
  %32 = and i64 %26, %31
  %33 = select i1 %27, i64 %26, i64 %32
  %34 = inttoptr i64 %33 to ptr
  %35 = getelementptr ptr, ptr %34, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr ptr, ptr %36, i32 0
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr @Node..rtti, align 8
  %40 = call i1 %38(ptr %34, ptr %39)
  br i1 %40, label %41, label %91

41:                                               ; preds = %21
  store ptr %34, ptr %3, align 8
  br label %93

42:                                               ; preds = %20
  %43 = load ptr, ptr %2, align 8
  %44 = getelementptr { i64, ptr, i8 }, ptr %43, i32 0, i32 1
  %45 = load ptr, ptr %44, align 8
  %46 = ptrtoint ptr %45 to i64
  %47 = icmp ne i64 %46, 0
  %48 = zext i1 %47 to i32
  %49 = and i32 %48, 1
  %50 = icmp ne i32 %49, 0
  br i1 %50, label %51, label %54

51:                                               ; preds = %42
  %52 = call i32 @strcmp(ptr %45, ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8))
  %53 = icmp eq i32 %52, 0
  br label %56

54:                                               ; preds = %42
  %55 = icmp eq i64 %46, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8) to i64)
  br label %56

56:                                               ; preds = %51, %54
  %57 = phi i1 [ %55, %54 ], [ %53, %51 ]
  br label %58

58:                                               ; preds = %56
  br i1 %57, label %59, label %91

59:                                               ; preds = %58
  %60 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %4, i32 0, i32 2
  %61 = load { ptr, ptr }, ptr %60, align 8
  %62 = extractvalue { ptr, ptr } %61, 1
  %63 = ptrtoint ptr %62 to i64
  %64 = icmp ne i64 %63, 0
  br i1 %64, label %65, label %72

65:                                               ; preds = %59
  %66 = extractvalue { ptr, ptr } %61, 0
  %67 = extractvalue { ptr, ptr } %61, 1
  %68 = getelementptr ptr, ptr %66, i32 1
  %69 = load ptr, ptr %68, align 8
  %70 = load ptr, ptr @Node..rtti, align 8
  %71 = call i1 %69(ptr %67, ptr %70)
  br label %72

72:                                               ; preds = %65, %59
  %73 = phi i1 [ %71, %65 ], [ false, %59 ]
  br i1 %73, label %74, label %91

74:                                               ; preds = %72
  %75 = extractvalue { ptr, ptr } %61, 1
  %76 = ptrtoint ptr %75 to i64
  %77 = icmp ne i64 %76, 0
  br i1 %77, label %78, label %85

78:                                               ; preds = %74
  %79 = extractvalue { ptr, ptr } %61, 0
  %80 = extractvalue { ptr, ptr } %61, 1
  %81 = getelementptr ptr, ptr %79, i32 1
  %82 = load ptr, ptr %81, align 8
  %83 = load ptr, ptr @Node..rtti, align 8
  %84 = call i1 %82(ptr %80, ptr %83)
  br label %85

85:                                               ; preds = %78, %74
  %86 = phi i1 [ %84, %78 ], [ false, %74 ]
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = extractvalue { ptr, ptr } %61, 1
  store ptr %88, ptr %3, align 8
  br label %93

89:                                               ; preds = %85
  %90 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %90, align 8
  call void @__cxa_throw(ptr %90, ptr @_ZTIPc, ptr null)
  br label %95

91:                                               ; preds = %21, %72, %58
  %92 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %92, align 8
  call void @__cxa_throw(ptr %92, ptr @_ZTIPc, ptr null)
  br label %95

93:                                               ; preds = %41, %87
  %94 = load ptr, ptr %3, align 8
  ret ptr %94

95:                                               ; preds = %91, %89
  unreachable
}

; Function Attrs: mustprogress
define i64 @fromAny(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = invoke ptr @"___unbox<!ts.class<Node, !ts.class_storage<Node>>>"(ptr %5)
          to label %7 unwind label %12

7:                                                ; preds = %1
  store ptr %6, ptr %2, align 8
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr %Node, ptr %8, i32 0, i32 2
  %10 = load ptr, ptr %9, align 8
  %11 = call i64 @strlen(ptr %10)
  store i64 %11, ptr %4, align 8
  br label %16

12:                                               ; preds = %1
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  store i64 0, ptr %4, align 8
  call void @__cxa_end_catch()
  br label %16

16:                                               ; preds = %7, %12
  %17 = load i64, ptr %4, align 8
  ret i64 %17
}

; Function Attrs: mustprogress
define i64 @fromField(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr %Holder, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %2, align 8
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr %Node, ptr %8, i32 0, i32 1
  %10 = load ptr, ptr %9, align 8
  %11 = icmp eq ptr %10, null
  %12 = select i1 %11, ptr @__ts_empty_array, ptr %10
  %13 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 0
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr ptr, ptr %14, i32 0
  %16 = load ptr, ptr %15, align 8
  %17 = call i64 @strlen(ptr %16)
  store i64 %17, ptr %4, align 8
  %18 = load i64, ptr %4, align 8
  ret i64 %18
}

; Function Attrs: mustprogress
define i32 @fromLocal(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca i32, align 4
  store double %0, ptr %4, align 8
  %6 = call ptr @GC_malloc(i64 32)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Node, ptr %7, i32 0, i32 0
  store ptr @Node..vtbl, ptr %8, align 8
  %9 = load double, ptr %4, align 8
  %10 = call ptr @GC_malloc_atomic(i64 40)
  %11 = getelementptr i8, ptr %10, i64 8
  call void @__ts_number_to_string(ptr %11, double %9)
  %12 = icmp eq ptr %11, null
  %13 = select i1 %12, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %11
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638193797625411262, i64 8))
  %15 = add i64 1, %14
  %16 = call i64 @strlen(ptr %13)
  %17 = add i64 %15, %16
  %18 = add i64 %17, 8
  %19 = call ptr @GC_malloc(i64 %18)
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr i8, ptr %20, i64 0
  %22 = sub i64 %17, 0
  %23 = call ptr @__memcpy_chk(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_12638193797625411262, i64 8), i64 %14, i64 %22)
  %24 = add i64 0, %14
  %25 = getelementptr i8, ptr %20, i64 %24
  %26 = sub i64 %17, %24
  %27 = call ptr @__memcpy_chk(ptr %25, ptr %13, i64 %16, i64 %26)
  %28 = add i64 %24, %16
  %29 = getelementptr i8, ptr %20, i64 %28
  store i8 0, ptr %29, align 1
  call void @Node.constructor(ptr %7, ptr %20)
  store i32 0, ptr %3, align 4
  store ptr %7, ptr %2, align 8
  %30 = load i32, ptr %3, align 4
  %31 = load ptr, ptr %2, align 8
  %32 = getelementptr %Node, ptr %31, i32 0, i32 2
  %33 = load ptr, ptr %32, align 8
  %34 = call i64 @strlen(ptr %33)
  %35 = sext i32 %30 to i64
  %36 = add i64 %35, %34
  %37 = trunc i64 %36 to i32
  store i32 %37, ptr %3, align 4
  %38 = load i32, ptr %3, align 4
  %39 = getelementptr %Node, ptr %7, i32 0, i32 1
  %40 = load ptr, ptr %39, align 8
  %41 = icmp eq ptr %40, null
  %42 = select i1 %41, ptr @__ts_empty_array, ptr %40
  %43 = getelementptr { ptr, i64, i64 }, ptr %42, i32 0, i32 1
  %44 = load i64, ptr %43, align 8
  %45 = sext i32 %38 to i64
  %46 = add i64 %45, %44
  %47 = trunc i64 %46 to i32
  store i32 %47, ptr %3, align 4
  %48 = load i32, ptr %3, align 4
  store i32 %48, ptr %5, align 4
  %49 = load i32, ptr %5, align 4
  ret i32 %49
}

; Function Attrs: mustprogress
define i64 @fresh(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca i64, align 8
  store double %0, ptr %3, align 8
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Node, ptr %6, i32 0, i32 0
  store ptr @Node..vtbl, ptr %7, align 8
  %8 = load double, ptr %3, align 8
  %9 = call ptr @GC_malloc_atomic(i64 40)
  %10 = getelementptr i8, ptr %9, i64 8
  call void @__ts_number_to_string(ptr %10, double %8)
  %11 = icmp eq ptr %10, null
  %12 = select i1 %11, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %10
  %13 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638186101044013785, i64 8))
  %14 = add i64 1, %13
  %15 = call i64 @strlen(ptr %12)
  %16 = add i64 %14, %15
  %17 = add i64 %16, 8
  %18 = call ptr @GC_malloc(i64 %17)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr i8, ptr %19, i64 0
  %21 = sub i64 %16, 0
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638186101044013785, i64 8), i64 %13, i64 %21)
  %23 = add i64 0, %13
  %24 = getelementptr i8, ptr %19, i64 %23
  %25 = sub i64 %16, %23
  %26 = call ptr @__memcpy_chk(ptr %24, ptr %12, i64 %15, i64 %25)
  %27 = add i64 %23, %15
  %28 = getelementptr i8, ptr %19, i64 %27
  store i8 0, ptr %28, align 1
  invoke void @Node.constructor(ptr %6, ptr %19)
          to label %29 unwind label %40

29:                                               ; preds = %1
  store ptr %6, ptr %2, align 8
  %30 = load ptr, ptr %2, align 8
  %31 = getelementptr %Node, ptr %30, i32 0, i32 1
  %32 = load ptr, ptr %31, align 8
  %33 = icmp eq ptr %32, null
  %34 = select i1 %33, ptr @__ts_empty_array, ptr %32
  %35 = getelementptr { ptr, i64, i64 }, ptr %34, i32 0, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr ptr, ptr %36, i32 0
  %38 = load ptr, ptr %37, align 8
  %39 = call i64 @strlen(ptr %38)
  store i64 %39, ptr %4, align 8
  br label %44

40:                                               ; preds = %1
  %41 = landingpad { ptr, i32 }
          catch ptr null
  %42 = extractvalue { ptr, i32 } %41, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  store i64 0, ptr %4, align 8
  call void @__cxa_end_catch()
  br label %44

44:                                               ; preds = %29, %40
  %45 = load i64, ptr %4, align 8
  ret i64 %45
}

; Function Attrs: mustprogress
define i1 @throwsInside(ptr %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store i1 false, ptr %3, align 1
  %6 = load ptr, ptr %4, align 8
  store ptr %6, ptr %2, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %Node, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = icmp eq ptr %9, null
  %11 = select i1 %10, ptr @__ts_empty_array, ptr %9
  %12 = getelementptr { ptr, i64, i64 }, ptr %11, i32 0, i32 1
  %13 = load i64, ptr %12, align 8
  %14 = icmp sgt i64 %13, 0
  br i1 %14, label %15, label %17

15:                                               ; preds = %1
  %16 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %16, align 8
  invoke void @__cxa_throw(ptr %16, ptr @_ZTIPc, ptr null)
          to label %25 unwind label %18

17:                                               ; preds = %1
  br label %22

18:                                               ; preds = %15
  %19 = landingpad { ptr, i32 }
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %22

22:                                               ; preds = %18, %17
  %23 = load i1, ptr %3, align 1
  store i1 %23, ptr %5, align 1
  %24 = load i1, ptr %5, align 1
  ret i1 %24

25:                                               ; preds = %15
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca <{ i64, %Holder }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Holder zeroinitializer, ptr %4, align 8
  store %Holder zeroinitializer, ptr %4, align 8
  %5 = getelementptr %Holder, ptr %4, i32 0, i32 0
  store ptr @Holder..vtbl, ptr %5, align 8
  call void @Holder.constructor(ptr %4)
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %6

6:                                                ; preds = %137, %0
  %7 = load i32, ptr %1, align 4
  %8 = icmp slt i32 %7, 2000
  br i1 %8, label %9, label %140

9:                                                ; preds = %6
  %10 = call ptr @GC_malloc(i64 32)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Node, ptr %11, i32 0, i32 0
  store ptr @Node..vtbl, ptr %12, align 8
  %13 = load i32, ptr %1, align 4
  %14 = call ptr @GC_malloc_atomic(i64 58)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %15, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %13)
  %17 = icmp eq ptr %15, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638194897137039473, i64 8))
  %20 = add i64 1, %19
  %21 = call i64 @strlen(ptr %18)
  %22 = add i64 %20, %21
  %23 = add i64 %22, 8
  %24 = call ptr @GC_malloc(i64 %23)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %22, 0
  %28 = call ptr @__memcpy_chk(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_12638194897137039473, i64 8), i64 %19, i64 %27)
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %22, %29
  %32 = call ptr @__memcpy_chk(ptr %30, ptr %18, i64 %21, i64 %31)
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  store i8 0, ptr %34, align 1
  call void @Node.constructor(ptr %11, ptr %25)
  %35 = call ptr @GC_malloc(i64 32)
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr %Node, ptr %36, i32 0, i32 0
  store ptr @Node..vtbl, ptr %37, align 8
  %38 = load i32, ptr %1, align 4
  %39 = call ptr @GC_malloc_atomic(i64 58)
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %40, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %38)
  %42 = icmp eq ptr %40, null
  %43 = select i1 %42, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %40
  %44 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %45 = add i64 1, %44
  %46 = call i64 @strlen(ptr %43)
  %47 = add i64 %45, %46
  %48 = add i64 %47, 8
  %49 = call ptr @GC_malloc(i64 %48)
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr i8, ptr %50, i64 0
  %52 = sub i64 %47, 0
  %53 = call ptr @__memcpy_chk(ptr %51, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i64 %44, i64 %52)
  %54 = add i64 0, %44
  %55 = getelementptr i8, ptr %50, i64 %54
  %56 = sub i64 %47, %54
  %57 = call ptr @__memcpy_chk(ptr %55, ptr %43, i64 %46, i64 %56)
  %58 = add i64 %54, %46
  %59 = getelementptr i8, ptr %50, i64 %58
  store i8 0, ptr %59, align 1
  call void @Node.constructor(ptr %36, ptr %50)
  %60 = call ptr @GC_malloc(i64 32)
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, ptr }, ptr %61, i32 0, i32 0
  store i64 8, ptr %62, align 8
  %63 = getelementptr { i64, ptr, ptr }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4612575052580656710_class, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, ptr }, ptr %61, i32 0, i32 2
  store ptr %36, ptr %64, align 8
  %65 = load i32, ptr %2, align 4
  %66 = call i64 @fromParam(ptr %11)
  %67 = call i64 @fromAny(ptr %61)
  %68 = add i64 %66, %67
  %69 = call i64 @fromField(ptr %4)
  %70 = add i64 %68, %69
  %71 = load i32, ptr %1, align 4
  %72 = sitofp i32 %71 to double
  %73 = call i32 @fromLocal(double %72)
  %74 = sext i32 %73 to i64
  %75 = add i64 %70, %74
  %76 = load i32, ptr %1, align 4
  %77 = sitofp i32 %76 to double
  %78 = call i64 @fresh(double %77)
  %79 = add i64 %75, %78
  %80 = sext i32 %65 to i64
  %81 = add i64 %80, %79
  %82 = trunc i64 %81 to i32
  store i32 %82, ptr %2, align 4
  %83 = call i1 @throwsInside(ptr %11)
  br i1 %83, label %84, label %145

84:                                               ; preds = %9
  %85 = getelementptr %Node, ptr %11, i32 0, i32 1
  %86 = load ptr, ptr %85, align 8
  %87 = icmp eq ptr %86, null
  %88 = select i1 %87, ptr @__ts_empty_array, ptr %86
  %89 = getelementptr { ptr, i64, i64 }, ptr %88, i32 0, i32 0
  %90 = load ptr, ptr %89, align 8
  %91 = getelementptr ptr, ptr %90, i32 0
  %92 = load ptr, ptr %91, align 8
  %93 = load i32, ptr %1, align 4
  %94 = call ptr @GC_malloc_atomic(i64 58)
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %95, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %93)
  %97 = icmp eq ptr %95, null
  %98 = select i1 %97, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %95
  %99 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638194897137039473, i64 8))
  %100 = add i64 1, %99
  %101 = call i64 @strlen(ptr %98)
  %102 = add i64 %100, %101
  %103 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8))
  %104 = add i64 %102, %103
  %105 = add i64 %104, 8
  %106 = call ptr @GC_malloc(i64 %105)
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr i8, ptr %107, i64 0
  %109 = sub i64 %104, 0
  %110 = call ptr @__memcpy_chk(ptr %108, ptr getelementptr inbounds nuw (i8, ptr @s_12638194897137039473, i64 8), i64 %99, i64 %109)
  %111 = add i64 0, %99
  %112 = getelementptr i8, ptr %107, i64 %111
  %113 = sub i64 %104, %111
  %114 = call ptr @__memcpy_chk(ptr %112, ptr %98, i64 %101, i64 %113)
  %115 = add i64 %111, %101
  %116 = getelementptr i8, ptr %107, i64 %115
  %117 = sub i64 %104, %115
  %118 = call ptr @__memcpy_chk(ptr %116, ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8), i64 %103, i64 %117)
  %119 = add i64 %115, %103
  %120 = getelementptr i8, ptr %107, i64 %119
  store i8 0, ptr %120, align 1
  %121 = ptrtoint ptr %92 to i64
  %122 = ptrtoint ptr %107 to i64
  %123 = icmp ne i64 %121, 0
  %124 = icmp ne i64 %122, 0
  %125 = zext i1 %123 to i32
  %126 = zext i1 %124 to i32
  %127 = and i32 %125, %126
  %128 = icmp ne i32 %127, 0
  br i1 %128, label %129, label %132

129:                                              ; preds = %84
  %130 = call i32 @strcmp(ptr %92, ptr %107)
  %131 = icmp eq i32 %130, 0
  br label %134

132:                                              ; preds = %84
  %133 = icmp eq i64 %121, %122
  br label %134

134:                                              ; preds = %129, %132
  %135 = phi i1 [ %133, %132 ], [ %131, %129 ]
  br label %136

136:                                              ; preds = %134
  br i1 %135, label %137, label %147

137:                                              ; preds = %136
  %138 = load i32, ptr %1, align 4
  %139 = add i32 %138, 1
  store i32 %139, ptr %1, align 4
  br label %6

140:                                              ; preds = %6
  %141 = load i32, ptr %2, align 4
  %142 = icmp sgt i32 %141, 0
  br i1 %142, label %143, label %149

143:                                              ; preds = %140
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

144:                                              ; preds = %147, %145, %149
  unreachable

145:                                              ; preds = %9
  %146 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3019566071980655109, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10626865055412510189, i64 8), i32 86, ptr null)
  br label %144

147:                                              ; preds = %136
  %148 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1102633763587963156, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10626865055412510189, i64 8), i32 87, ptr null)
  br label %144

149:                                              ; preds = %140
  %150 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3300820206774538237, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10626865055412510189, i64 8), i32 90, ptr null)
  br label %144
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

