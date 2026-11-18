; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr, ptr }
%Throws = type { ptr, ptr }

$Res..size = comdat any

$Throws..size = comdat any

@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_7060529149083524436 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFThrows\00", align 8
@s_12638145419113769978 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF;\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_15839897912459002582 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFafter;\00", align 8
@s_12638189399578898418 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_11858887882840284586 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody;\00", align 8
@s_8008412449099960154 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught;\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_584909432750204567 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF: \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_4044071365934239186 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00using_dispose_order.ts\00", align 8
@m_2432362340776716864 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmust throw\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_9120454858007850942 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak out of a switch case\00", align 8
@s_8676791534656714797 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb;a;after;\00", align 8
@s_6620153503825695794 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcontinue and break out of a loop body\00", align 8
@s_1538657466990327383 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb0;a0;b1;a1;\00", align 8
@s_7575827081894217403 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa dispose that throws\00", align 8
@s_11545804706003810958 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody;b;a;caught;\00", align 8
@s_5966407086411829586 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrow out of a try body\00", align 8
@s_4159161771213527651 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb;a;caught;\00", align 8
@s_1254165271164641062 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrow out of a block\00", align 8
@s_8473372317540028708 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfalling off a nested block\00", align 8
@s_739271353490714855 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc;b;after;a;\00", align 8
@s_14848773516908418703 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreturn out of a nested block\00", align 8
@s_1309934839817893640 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc;b;a;\00", align 8
@s_9316178885392575187 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfunction body\00", align 8
@s_17181858102171564713 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody;b;a;\00", align 8
@s_10625962022409473026 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFplain block\00", align 8
@s_4970319309024353835 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc;b;a;after;\00", align 8
@trail = private global ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 16, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIi = external constant ptr
@Throws..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7060529149083524436, i64 8)
@Throws..size = global i64 16, comdat
@Throws..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Throws..instanceOf, ptr @Throws..new, ptr @Throws.dispose, ptr @Throws..rtti, ptr @Throws..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @__cxa_end_catch()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Res..rtti, align 8
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
define ptr @Res..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @trail, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %9 = call i64 @strlen(ptr %8)
  %10 = add i64 1, %9
  %11 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %12 = add i64 %10, %11
  %13 = add i64 %12, 8
  %14 = call ptr @GC_malloc(i64 %13)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i8, ptr %15, i64 0
  %17 = sub i64 %12, 0
  %18 = call ptr @__memcpy_chk(ptr %16, ptr %8, i64 %9, i64 %17)
  %19 = add i64 0, %9
  %20 = getelementptr i8, ptr %15, i64 %19
  %21 = sub i64 %12, %19
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %11, i64 %21)
  %23 = add i64 %19, %11
  %24 = getelementptr i8, ptr %15, i64 %23
  store i8 0, ptr %24, align 1
  %25 = icmp eq ptr %3, null
  %26 = select i1 %25, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  %27 = icmp eq ptr %15, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %29 = call i64 @strlen(ptr %26)
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr %28)
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %26, i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %28, i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  store ptr %35, ptr @trail, align 8
  ret void
}

; Function Attrs: mustprogress
define void @plainBlock() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  %6 = alloca ptr, align 8
  store i1 false, ptr %5, align 1
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Res, ptr %8, i32 0, i32 0
  store ptr @Res..vtbl, ptr %9, align 8
  invoke void @Res.constructor(ptr %8, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %10 unwind label %56

10:                                               ; preds = %0
  store ptr %8, ptr %6, align 8
  store i1 true, ptr %5, align 1
  %11 = call ptr @GC_malloc(i64 24)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr %Res, ptr %12, i32 0, i32 0
  store ptr @Res..vtbl, ptr %13, align 8
  invoke void @Res.constructor(ptr %12, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %14 unwind label %56

14:                                               ; preds = %10
  store ptr %12, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %Res, ptr %16, i32 0, i32 0
  store ptr @Res..vtbl, ptr %17, align 8
  invoke void @Res.constructor(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
          to label %18 unwind label %56

18:                                               ; preds = %14
  store ptr %16, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %24

20:                                               ; preds = %18
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %Res, ptr %21, i32 0, i32 0
  invoke void @Res.dispose(ptr %21)
          to label %23 unwind label %56

23:                                               ; preds = %20
  br label %24

24:                                               ; preds = %23, %18
  %25 = load i1, ptr %3, align 1
  br i1 %25, label %26, label %30

26:                                               ; preds = %24
  store i1 false, ptr %3, align 1
  %27 = load ptr, ptr %4, align 8
  %28 = getelementptr %Res, ptr %27, i32 0, i32 0
  invoke void @Res.dispose(ptr %27)
          to label %29 unwind label %56

29:                                               ; preds = %26
  br label %30

30:                                               ; preds = %29, %24
  %31 = load i1, ptr %5, align 1
  br i1 %31, label %32, label %36

32:                                               ; preds = %30
  store i1 false, ptr %5, align 1
  %33 = load ptr, ptr %6, align 8
  %34 = getelementptr %Res, ptr %33, i32 0, i32 0
  invoke void @Res.dispose(ptr %33)
          to label %35 unwind label %56

35:                                               ; preds = %32
  br label %36

36:                                               ; preds = %35, %30
  %37 = load ptr, ptr @trail, align 8
  %38 = icmp eq ptr %37, null
  %39 = select i1 %38, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %37
  %40 = call i64 @strlen(ptr %39)
  %41 = add i64 1, %40
  %42 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %43 = add i64 %41, %42
  %44 = add i64 %43, 8
  %45 = call ptr @GC_malloc(i64 %44)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr i8, ptr %46, i64 0
  %48 = sub i64 %43, 0
  %49 = call ptr @__memcpy_chk(ptr %47, ptr %39, i64 %40, i64 %48)
  %50 = add i64 0, %40
  %51 = getelementptr i8, ptr %46, i64 %50
  %52 = sub i64 %43, %50
  %53 = call ptr @__memcpy_chk(ptr %51, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %42, i64 %52)
  %54 = add i64 %50, %42
  %55 = getelementptr i8, ptr %46, i64 %54
  store i8 0, ptr %55, align 1
  store ptr %46, ptr @trail, align 8
  ret void

56:                                               ; preds = %32, %26, %20, %14, %10, %0
  %57 = landingpad { ptr, i32 }
          cleanup
  %58 = load i1, ptr %1, align 1
  br i1 %58, label %59, label %62

59:                                               ; preds = %56
  store i1 false, ptr %1, align 1
  %60 = load ptr, ptr %2, align 8
  %61 = getelementptr %Res, ptr %60, i32 0, i32 0
  call void @Res.dispose(ptr %60)
  br label %62

62:                                               ; preds = %59, %56
  %63 = load i1, ptr %3, align 1
  br i1 %63, label %64, label %67

64:                                               ; preds = %62
  store i1 false, ptr %3, align 1
  %65 = load ptr, ptr %4, align 8
  %66 = getelementptr %Res, ptr %65, i32 0, i32 0
  call void @Res.dispose(ptr %65)
  br label %67

67:                                               ; preds = %64, %62
  %68 = load i1, ptr %5, align 1
  br i1 %68, label %69, label %72

69:                                               ; preds = %67
  store i1 false, ptr %5, align 1
  %70 = load ptr, ptr %6, align 8
  %71 = getelementptr %Res, ptr %70, i32 0, i32 0
  call void @Res.dispose(ptr %70)
  br label %72

72:                                               ; preds = %69, %67
  resume { ptr, i32 } %57
}

; Function Attrs: mustprogress
define void @functionBody() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  invoke void @Res.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %8 unwind label %44

8:                                                ; preds = %0
  store ptr %6, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %9 = call ptr @GC_malloc(i64 24)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %Res, ptr %10, i32 0, i32 0
  store ptr @Res..vtbl, ptr %11, align 8
  invoke void @Res.constructor(ptr %10, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %12 unwind label %44

12:                                               ; preds = %8
  store ptr %10, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %13 = load ptr, ptr @trail, align 8
  %14 = icmp eq ptr %13, null
  %15 = select i1 %14, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %13
  %16 = call i64 @strlen(ptr %15)
  %17 = add i64 1, %16
  %18 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_11858887882840284586, i64 8))
  %19 = add i64 %17, %18
  %20 = add i64 %19, 8
  %21 = call ptr @GC_malloc(i64 %20)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr i8, ptr %22, i64 0
  %24 = sub i64 %19, 0
  %25 = call ptr @__memcpy_chk(ptr %23, ptr %15, i64 %16, i64 %24)
  %26 = add i64 0, %16
  %27 = getelementptr i8, ptr %22, i64 %26
  %28 = sub i64 %19, %26
  %29 = call ptr @__memcpy_chk(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_11858887882840284586, i64 8), i64 %18, i64 %28)
  %30 = add i64 %26, %18
  %31 = getelementptr i8, ptr %22, i64 %30
  store i8 0, ptr %31, align 1
  store ptr %22, ptr @trail, align 8
  %32 = load i1, ptr %1, align 1
  br i1 %32, label %33, label %37

33:                                               ; preds = %12
  store i1 false, ptr %1, align 1
  %34 = load ptr, ptr %2, align 8
  %35 = getelementptr %Res, ptr %34, i32 0, i32 0
  invoke void @Res.dispose(ptr %34)
          to label %36 unwind label %44

36:                                               ; preds = %33
  br label %37

37:                                               ; preds = %36, %12
  %38 = load i1, ptr %3, align 1
  br i1 %38, label %39, label %43

39:                                               ; preds = %37
  store i1 false, ptr %3, align 1
  %40 = load ptr, ptr %4, align 8
  %41 = getelementptr %Res, ptr %40, i32 0, i32 0
  invoke void @Res.dispose(ptr %40)
          to label %42 unwind label %44

42:                                               ; preds = %39
  br label %43

43:                                               ; preds = %42, %37
  ret void

44:                                               ; preds = %39, %33, %8, %0
  %45 = landingpad { ptr, i32 }
          cleanup
  %46 = load i1, ptr %1, align 1
  br i1 %46, label %47, label %50

47:                                               ; preds = %44
  store i1 false, ptr %1, align 1
  %48 = load ptr, ptr %2, align 8
  %49 = getelementptr %Res, ptr %48, i32 0, i32 0
  call void @Res.dispose(ptr %48)
  br label %50

50:                                               ; preds = %47, %44
  %51 = load i1, ptr %3, align 1
  br i1 %51, label %52, label %55

52:                                               ; preds = %50
  store i1 false, ptr %3, align 1
  %53 = load ptr, ptr %4, align 8
  %54 = getelementptr %Res, ptr %53, i32 0, i32 0
  call void @Res.dispose(ptr %53)
  br label %55

55:                                               ; preds = %52, %50
  resume { ptr, i32 } %45
}

; Function Attrs: mustprogress
define void @earlyReturn(i1 %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca i1, align 1
  %7 = alloca ptr, align 8
  %8 = alloca i1, align 1
  store i1 %0, ptr %8, align 1
  store i1 false, ptr %6, align 1
  %9 = call ptr @GC_malloc(i64 24)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %Res, ptr %10, i32 0, i32 0
  store ptr @Res..vtbl, ptr %11, align 8
  invoke void @Res.constructor(ptr %10, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %12 unwind label %112

12:                                               ; preds = %1
  store ptr %10, ptr %7, align 8
  store i1 true, ptr %6, align 1
  store i1 false, ptr %4, align 1
  store i1 false, ptr %2, align 1
  %13 = call ptr @GC_malloc(i64 24)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr %Res, ptr %14, i32 0, i32 0
  store ptr @Res..vtbl, ptr %15, align 8
  invoke void @Res.constructor(ptr %14, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %16 unwind label %74

16:                                               ; preds = %12
  store ptr %14, ptr %5, align 8
  store i1 true, ptr %4, align 1
  %17 = call ptr @GC_malloc(i64 24)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr %Res, ptr %18, i32 0, i32 0
  store ptr @Res..vtbl, ptr %19, align 8
  invoke void @Res.constructor(ptr %18, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
          to label %20 unwind label %74

20:                                               ; preds = %16
  store ptr %18, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %21 = load i1, ptr %8, align 1
  br i1 %21, label %22, label %41

22:                                               ; preds = %20
  %23 = load i1, ptr %2, align 1
  br i1 %23, label %24, label %28

24:                                               ; preds = %22
  store i1 false, ptr %2, align 1
  %25 = load ptr, ptr %3, align 8
  %26 = getelementptr %Res, ptr %25, i32 0, i32 0
  invoke void @Res.dispose(ptr %25)
          to label %27 unwind label %74

27:                                               ; preds = %24
  br label %28

28:                                               ; preds = %27, %22
  %29 = load i1, ptr %4, align 1
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  store i1 false, ptr %4, align 1
  %31 = load ptr, ptr %5, align 8
  %32 = getelementptr %Res, ptr %31, i32 0, i32 0
  invoke void @Res.dispose(ptr %31)
          to label %33 unwind label %74

33:                                               ; preds = %30
  br label %34

34:                                               ; preds = %33, %28
  %35 = load i1, ptr %6, align 1
  br i1 %35, label %36, label %40

36:                                               ; preds = %34
  store i1 false, ptr %6, align 1
  %37 = load ptr, ptr %7, align 8
  %38 = getelementptr %Res, ptr %37, i32 0, i32 0
  invoke void @Res.dispose(ptr %37)
          to label %39 unwind label %74

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %39, %34
  br label %120

41:                                               ; preds = %20
  %42 = load i1, ptr %2, align 1
  br i1 %42, label %43, label %47

43:                                               ; preds = %41
  store i1 false, ptr %2, align 1
  %44 = load ptr, ptr %3, align 8
  %45 = getelementptr %Res, ptr %44, i32 0, i32 0
  invoke void @Res.dispose(ptr %44)
          to label %46 unwind label %74

46:                                               ; preds = %43
  br label %47

47:                                               ; preds = %46, %41
  %48 = load i1, ptr %4, align 1
  br i1 %48, label %49, label %53

49:                                               ; preds = %47
  store i1 false, ptr %4, align 1
  %50 = load ptr, ptr %5, align 8
  %51 = getelementptr %Res, ptr %50, i32 0, i32 0
  invoke void @Res.dispose(ptr %50)
          to label %52 unwind label %74

52:                                               ; preds = %49
  br label %53

53:                                               ; preds = %52, %47
  %54 = load ptr, ptr @trail, align 8
  %55 = icmp eq ptr %54, null
  %56 = select i1 %55, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %54
  %57 = call i64 @strlen(ptr %56)
  %58 = add i64 1, %57
  %59 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %60 = add i64 %58, %59
  %61 = add i64 %60, 8
  %62 = call ptr @GC_malloc(i64 %61)
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr i8, ptr %63, i64 0
  %65 = sub i64 %60, 0
  %66 = call ptr @__memcpy_chk(ptr %64, ptr %56, i64 %57, i64 %65)
  %67 = add i64 0, %57
  %68 = getelementptr i8, ptr %63, i64 %67
  %69 = sub i64 %60, %67
  %70 = call ptr @__memcpy_chk(ptr %68, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %59, i64 %69)
  %71 = add i64 %67, %59
  %72 = getelementptr i8, ptr %63, i64 %71
  store i8 0, ptr %72, align 1
  store ptr %63, ptr @trail, align 8
  %73 = load i1, ptr %6, align 1
  br i1 %73, label %106, label %110

74:                                               ; preds = %36, %30, %24, %49, %43, %16, %12
  %75 = landingpad { ptr, i32 }
          catch ptr null
  %76 = extractvalue { ptr, i32 } %75, 0
  %77 = call ptr @__cxa_begin_catch(ptr %76)
  %78 = load i1, ptr %2, align 1
  br i1 %78, label %79, label %83

79:                                               ; preds = %74
  store i1 false, ptr %2, align 1
  %80 = load ptr, ptr %3, align 8
  %81 = getelementptr %Res, ptr %80, i32 0, i32 0
  invoke void @Res.dispose(ptr %80)
          to label %82 unwind label %endcatch.pad

82:                                               ; preds = %79
  br label %83

83:                                               ; preds = %82, %74
  %84 = load i1, ptr %4, align 1
  br i1 %84, label %85, label %89

85:                                               ; preds = %83
  store i1 false, ptr %4, align 1
  %86 = load ptr, ptr %5, align 8
  %87 = getelementptr %Res, ptr %86, i32 0, i32 0
  invoke void @Res.dispose(ptr %86)
          to label %88 unwind label %endcatch.pad

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %88, %83
  invoke void @__cxa_rethrow()
          to label %121 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %79, %85
  %90 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

91:                                               ; No predecessors!
  %92 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %91
  %93 = phi { ptr, i32 } [ %92, %91 ], [ %90, %endcatch.pad ]
  %94 = extractvalue { ptr, i32 } %93, 0
  %95 = call ptr @__cxa_begin_catch(ptr %94)
  %96 = load i1, ptr %2, align 1
  br i1 %96, label %97, label %100

97:                                               ; preds = %.caught
  store i1 false, ptr %2, align 1
  %98 = load ptr, ptr %3, align 8
  %99 = getelementptr %Res, ptr %98, i32 0, i32 0
  invoke void @Res.dispose(ptr %98)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %97
  br label %100

100:                                              ; preds = %.noexc3, %.caught
  %101 = load i1, ptr %4, align 1
  br i1 %101, label %102, label %105

102:                                              ; preds = %100
  store i1 false, ptr %4, align 1
  %103 = load ptr, ptr %5, align 8
  %104 = getelementptr %Res, ptr %103, i32 0, i32 0
  invoke void @Res.dispose(ptr %103)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %102
  br label %105

105:                                              ; preds = %.noexc, %100
  invoke void @__cxa_rethrow()
          to label %121 unwind label %endcatch.pad2

106:                                              ; preds = %53
  store i1 false, ptr %6, align 1
  %107 = load ptr, ptr %7, align 8
  %108 = getelementptr %Res, ptr %107, i32 0, i32 0
  invoke void @Res.dispose(ptr %107)
          to label %109 unwind label %112

109:                                              ; preds = %106
  br label %110

110:                                              ; preds = %109, %53
  br label %120

endcatch.pad2:                                    ; preds = %89, %105
  %111 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught1

112:                                              ; preds = %106, %1
  %113 = landingpad { ptr, i32 }
          cleanup
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %112
  %114 = phi { ptr, i32 } [ %113, %112 ], [ %111, %endcatch.pad2 ]
  %115 = load i1, ptr %6, align 1
  br i1 %115, label %116, label %119

116:                                              ; preds = %.caught1
  store i1 false, ptr %6, align 1
  %117 = load ptr, ptr %7, align 8
  %118 = getelementptr %Res, ptr %117, i32 0, i32 0
  call void @Res.dispose(ptr %117)
  br label %119

119:                                              ; preds = %116, %.caught1
  resume { ptr, i32 } %114

120:                                              ; preds = %40, %110
  ret void

121:                                              ; preds = %89, %105
  unreachable

endcatch.resume:                                  ; preds = %97, %102
  %122 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %122
}

; Function Attrs: mustprogress
define void @throwsInBlock() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  %6 = alloca ptr, align 8
  store i1 false, ptr %5, align 1
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Res, ptr %8, i32 0, i32 0
  store ptr @Res..vtbl, ptr %9, align 8
  invoke void @Res.constructor(ptr %8, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %10 unwind label %20

10:                                               ; preds = %0
  store ptr %8, ptr %6, align 8
  store i1 true, ptr %5, align 1
  %11 = call ptr @GC_malloc(i64 24)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr %Res, ptr %12, i32 0, i32 0
  store ptr @Res..vtbl, ptr %13, align 8
  invoke void @Res.constructor(ptr %12, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %14 unwind label %20

14:                                               ; preds = %10
  store ptr %12, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %Res, ptr %16, i32 0, i32 0
  store ptr @Res..vtbl, ptr %17, align 8
  invoke void @Res.constructor(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
          to label %18 unwind label %20

18:                                               ; preds = %14
  store ptr %16, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %19 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %19, align 4
  invoke void @__cxa_throw(ptr %19, ptr @_ZTIi, ptr null)
          to label %37 unwind label %20

20:                                               ; preds = %18, %14, %10, %0
  %21 = landingpad { ptr, i32 }
          cleanup
  %22 = load i1, ptr %1, align 1
  br i1 %22, label %23, label %26

23:                                               ; preds = %20
  store i1 false, ptr %1, align 1
  %24 = load ptr, ptr %2, align 8
  %25 = getelementptr %Res, ptr %24, i32 0, i32 0
  call void @Res.dispose(ptr %24)
  br label %26

26:                                               ; preds = %23, %20
  %27 = load i1, ptr %3, align 1
  br i1 %27, label %28, label %31

28:                                               ; preds = %26
  store i1 false, ptr %3, align 1
  %29 = load ptr, ptr %4, align 8
  %30 = getelementptr %Res, ptr %29, i32 0, i32 0
  call void @Res.dispose(ptr %29)
  br label %31

31:                                               ; preds = %28, %26
  %32 = load i1, ptr %5, align 1
  br i1 %32, label %33, label %36

33:                                               ; preds = %31
  store i1 false, ptr %5, align 1
  %34 = load ptr, ptr %6, align 8
  %35 = getelementptr %Res, ptr %34, i32 0, i32 0
  call void @Res.dispose(ptr %34)
  br label %36

36:                                               ; preds = %33, %31
  resume { ptr, i32 } %21

37:                                               ; preds = %18
  unreachable
}

; Function Attrs: mustprogress
define void @throwsInTryBody() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Res, ptr %7, i32 0, i32 0
  store ptr @Res..vtbl, ptr %8, align 8
  invoke void @Res.constructor(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %9 unwind label %15

9:                                                ; preds = %0
  store ptr %7, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Res, ptr %11, i32 0, i32 0
  store ptr @Res..vtbl, ptr %12, align 8
  invoke void @Res.constructor(ptr %11, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %13 unwind label %15

13:                                               ; preds = %9
  store ptr %11, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %14 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %14, align 4
  invoke void @__cxa_throw(ptr %14, ptr @_ZTIi, ptr null)
          to label %80 unwind label %15

15:                                               ; preds = %13, %9, %0
  %16 = landingpad { ptr, i32 }
          catch ptr null
  %17 = extractvalue { ptr, i32 } %16, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %24

20:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %Res, ptr %21, i32 0, i32 0
  invoke void @Res.dispose(ptr %21)
          to label %23 unwind label %endcatch.pad

23:                                               ; preds = %20
  br label %24

24:                                               ; preds = %23, %15
  %25 = load i1, ptr %3, align 1
  br i1 %25, label %26, label %30

26:                                               ; preds = %24
  store i1 false, ptr %3, align 1
  %27 = load ptr, ptr %4, align 8
  %28 = getelementptr %Res, ptr %27, i32 0, i32 0
  invoke void @Res.dispose(ptr %27)
          to label %29 unwind label %endcatch.pad

29:                                               ; preds = %26
  br label %30

30:                                               ; preds = %29, %24
  invoke void @__cxa_rethrow()
          to label %80 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %20, %26
  %31 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

32:                                               ; No predecessors!
  %33 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %32
  %34 = phi { ptr, i32 } [ %33, %32 ], [ %31, %endcatch.pad ]
  %35 = extractvalue { ptr, i32 } %34, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  %37 = load i1, ptr %1, align 1
  br i1 %37, label %38, label %41

38:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %39 = load ptr, ptr %2, align 8
  %40 = getelementptr %Res, ptr %39, i32 0, i32 0
  invoke void @Res.dispose(ptr %39)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %38
  br label %41

41:                                               ; preds = %.noexc4, %.caught
  %42 = load i1, ptr %3, align 1
  br i1 %42, label %43, label %46

43:                                               ; preds = %41
  store i1 false, ptr %3, align 1
  %44 = load ptr, ptr %4, align 8
  %45 = getelementptr %Res, ptr %44, i32 0, i32 0
  invoke void @Res.dispose(ptr %44)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %43
  br label %46

46:                                               ; preds = %.noexc3, %41
  invoke void @__cxa_rethrow()
          to label %80 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %46, %30
  %47 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught1

48:                                               ; No predecessors!
  %49 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %48
  %50 = phi { ptr, i32 } [ %49, %48 ], [ %47, %endcatch.pad2 ]
  %51 = extractvalue { ptr, i32 } %50, 1
  %52 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %53 = icmp eq i32 %51, %52
  br i1 %53, label %57, label %54

54:                                               ; preds = %.caught1
  %55 = extractvalue { ptr, i32 } %50, 0
  %56 = call ptr @__cxa_begin_catch(ptr %55)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %54
  br label %80

57:                                               ; preds = %.caught1
  %58 = extractvalue { ptr, i32 } %50, 0
  %59 = call ptr @__cxa_begin_catch(ptr %58)
  %60 = load i32, ptr %59, align 4
  store i32 %60, ptr %5, align 4
  %61 = load ptr, ptr @trail, align 8
  %62 = icmp eq ptr %61, null
  %63 = select i1 %62, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %61
  %64 = invoke i64 @strlen(ptr %63)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %57
  %65 = add i64 1, %64
  %66 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8))
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %67 = add i64 %65, %66
  %68 = add i64 %67, 8
  %69 = invoke ptr @GC_malloc(i64 %68)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr i8, ptr %70, i64 0
  %72 = sub i64 %67, 0
  %73 = invoke ptr @__memcpy_chk(ptr %71, ptr %63, i64 %64, i64 %72)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %74 = add i64 0, %64
  %75 = getelementptr i8, ptr %70, i64 %74
  %76 = sub i64 %67, %74
  %77 = invoke ptr @__memcpy_chk(ptr %75, ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8), i64 %66, i64 %76)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %78 = add i64 %74, %66
  %79 = getelementptr i8, ptr %70, i64 %78
  store i8 0, ptr %79, align 1
  store ptr %70, ptr @trail, align 8
  call void @__cxa_end_catch()
  ret void

80:                                               ; preds = %.noexc, %30, %46, %13
  unreachable

endcatch.resume:                                  ; preds = %.noexc8, %.noexc7, %.noexc6, %.noexc5, %57, %38, %43, %54
  %81 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %81
}

; Function Attrs: mustprogress
define i1 @Throws..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Throws..rtti, align 8
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
define ptr @Throws..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Throws, ptr %3, i32 0, i32 0
  store ptr @Throws..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Throws.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Throws, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Throws.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @trail, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Throws, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %9 = call i64 @strlen(ptr %8)
  %10 = add i64 1, %9
  %11 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %12 = add i64 %10, %11
  %13 = add i64 %12, 8
  %14 = call ptr @GC_malloc(i64 %13)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i8, ptr %15, i64 0
  %17 = sub i64 %12, 0
  %18 = call ptr @__memcpy_chk(ptr %16, ptr %8, i64 %9, i64 %17)
  %19 = add i64 0, %9
  %20 = getelementptr i8, ptr %15, i64 %19
  %21 = sub i64 %12, %19
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %11, i64 %21)
  %23 = add i64 %19, %11
  %24 = getelementptr i8, ptr %15, i64 %23
  store i8 0, ptr %24, align 1
  %25 = icmp eq ptr %3, null
  %26 = select i1 %25, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  %27 = icmp eq ptr %15, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %29 = call i64 @strlen(ptr %26)
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr %28)
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %26, i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %28, i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  store ptr %35, ptr @trail, align 8
  %45 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %45, align 4
  call void @__cxa_throw(ptr %45, ptr @_ZTIi, ptr null)
  br label %46

46:                                               ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define void @throwingDispose() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Res, ptr %7, i32 0, i32 0
  store ptr @Res..vtbl, ptr %8, align 8
  invoke void @Res.constructor(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %9 unwind label %45

9:                                                ; preds = %0
  store ptr %7, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Throws, ptr %11, i32 0, i32 0
  store ptr @Throws..vtbl, ptr %12, align 8
  invoke void @Throws.constructor(ptr %11, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %13 unwind label %45

13:                                               ; preds = %9
  store ptr %11, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %14 = load ptr, ptr @trail, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr %16)
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_11858887882840284586, i64 8))
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr %16, i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %20, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr getelementptr inbounds nuw (i8, ptr @s_11858887882840284586, i64 8), i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  store ptr %23, ptr @trail, align 8
  %33 = load i1, ptr %1, align 1
  br i1 %33, label %34, label %38

34:                                               ; preds = %13
  store i1 false, ptr %1, align 1
  %35 = load ptr, ptr %2, align 8
  %36 = getelementptr %Throws, ptr %35, i32 0, i32 0
  invoke void @Throws.dispose(ptr %35)
          to label %37 unwind label %45

37:                                               ; preds = %34
  br label %38

38:                                               ; preds = %37, %13
  %39 = load i1, ptr %3, align 1
  br i1 %39, label %40, label %44

40:                                               ; preds = %38
  store i1 false, ptr %3, align 1
  %41 = load ptr, ptr %4, align 8
  %42 = getelementptr %Res, ptr %41, i32 0, i32 0
  invoke void @Res.dispose(ptr %41)
          to label %43 unwind label %45

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %43, %38
  br label %110

45:                                               ; preds = %40, %34, %9, %0
  %46 = landingpad { ptr, i32 }
          catch ptr null
  %47 = extractvalue { ptr, i32 } %46, 0
  %48 = call ptr @__cxa_begin_catch(ptr %47)
  %49 = load i1, ptr %1, align 1
  br i1 %49, label %50, label %54

50:                                               ; preds = %45
  store i1 false, ptr %1, align 1
  %51 = load ptr, ptr %2, align 8
  %52 = getelementptr %Throws, ptr %51, i32 0, i32 0
  invoke void @Throws.dispose(ptr %51)
          to label %53 unwind label %endcatch.pad

53:                                               ; preds = %50
  br label %54

54:                                               ; preds = %53, %45
  %55 = load i1, ptr %3, align 1
  br i1 %55, label %56, label %60

56:                                               ; preds = %54
  store i1 false, ptr %3, align 1
  %57 = load ptr, ptr %4, align 8
  %58 = getelementptr %Res, ptr %57, i32 0, i32 0
  invoke void @Res.dispose(ptr %57)
          to label %59 unwind label %endcatch.pad

59:                                               ; preds = %56
  br label %60

60:                                               ; preds = %59, %54
  invoke void @__cxa_rethrow()
          to label %111 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %56, %50
  %61 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

62:                                               ; No predecessors!
  %63 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %62
  %64 = phi { ptr, i32 } [ %63, %62 ], [ %61, %endcatch.pad ]
  %65 = extractvalue { ptr, i32 } %64, 0
  %66 = call ptr @__cxa_begin_catch(ptr %65)
  %67 = load i1, ptr %1, align 1
  br i1 %67, label %68, label %71

68:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %69 = load ptr, ptr %2, align 8
  %70 = getelementptr %Throws, ptr %69, i32 0, i32 0
  invoke void @Throws.dispose(ptr %69)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %68
  br label %71

71:                                               ; preds = %.noexc, %.caught
  %72 = load i1, ptr %3, align 1
  br i1 %72, label %73, label %76

73:                                               ; preds = %71
  store i1 false, ptr %3, align 1
  %74 = load ptr, ptr %4, align 8
  %75 = getelementptr %Res, ptr %74, i32 0, i32 0
  invoke void @Res.dispose(ptr %74)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %73
  br label %76

76:                                               ; preds = %.noexc3, %71
  invoke void @__cxa_rethrow()
          to label %111 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %76, %60
  %77 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught1

78:                                               ; No predecessors!
  %79 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %78
  %80 = phi { ptr, i32 } [ %79, %78 ], [ %77, %endcatch.pad2 ]
  %81 = extractvalue { ptr, i32 } %80, 1
  %82 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %83 = icmp eq i32 %81, %82
  br i1 %83, label %87, label %84

84:                                               ; preds = %.caught1
  %85 = extractvalue { ptr, i32 } %80, 0
  %86 = call ptr @__cxa_begin_catch(ptr %85)
  invoke void @__cxa_rethrow()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %84
  br label %111

87:                                               ; preds = %.caught1
  %88 = extractvalue { ptr, i32 } %80, 0
  %89 = call ptr @__cxa_begin_catch(ptr %88)
  %90 = load i32, ptr %89, align 4
  store i32 %90, ptr %5, align 4
  %91 = load ptr, ptr @trail, align 8
  %92 = icmp eq ptr %91, null
  %93 = select i1 %92, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %91
  %94 = invoke i64 @strlen(ptr %93)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %87
  %95 = add i64 1, %94
  %96 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8))
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %97 = add i64 %95, %96
  %98 = add i64 %97, 8
  %99 = invoke ptr @GC_malloc(i64 %98)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %100 = getelementptr i8, ptr %99, i64 8
  %101 = getelementptr i8, ptr %100, i64 0
  %102 = sub i64 %97, 0
  %103 = invoke ptr @__memcpy_chk(ptr %101, ptr %93, i64 %94, i64 %102)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %104 = add i64 0, %94
  %105 = getelementptr i8, ptr %100, i64 %104
  %106 = sub i64 %97, %104
  %107 = invoke ptr @__memcpy_chk(ptr %105, ptr getelementptr inbounds nuw (i8, ptr @s_8008412449099960154, i64 8), i64 %96, i64 %106)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %108 = add i64 %104, %96
  %109 = getelementptr i8, ptr %100, i64 %108
  store i8 0, ptr %109, align 1
  store ptr %100, ptr @trail, align 8
  call void @__cxa_end_catch()
  br label %110

110:                                              ; preds = %44, %.noexc8
  ret void

111:                                              ; preds = %.noexc9, %60, %76
  unreachable

endcatch.resume:                                  ; preds = %84, %.noexc7, %.noexc6, %.noexc5, %.noexc4, %87, %73, %68
  %112 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %112
}

; Function Attrs: mustprogress
define void @loopBody() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  br label %6

6:                                                ; preds = %76, %0
  %7 = load i32, ptr %5, align 4
  %8 = icmp slt i32 %7, 2
  br i1 %8, label %9, label %104

9:                                                ; preds = %6
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Res, ptr %11, i32 0, i32 0
  store ptr @Res..vtbl, ptr %12, align 8
  %13 = load i32, ptr %5, align 4
  %14 = call ptr @GC_malloc_atomic(i64 58)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %15, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %13)
  %17 = icmp eq ptr %15, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %20 = add i64 1, %19
  %21 = call i64 @strlen(ptr %18)
  %22 = add i64 %20, %21
  %23 = add i64 %22, 8
  %24 = call ptr @GC_malloc(i64 %23)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %22, 0
  %28 = call ptr @__memcpy_chk(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i64 %19, i64 %27)
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %22, %29
  %32 = call ptr @__memcpy_chk(ptr %30, ptr %18, i64 %21, i64 %31)
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  store i8 0, ptr %34, align 1
  invoke void @Res.constructor(ptr %11, ptr %25)
          to label %35 unwind label %92

35:                                               ; preds = %9
  store ptr %11, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %36 = call ptr @GC_malloc(i64 24)
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr %Res, ptr %37, i32 0, i32 0
  store ptr @Res..vtbl, ptr %38, align 8
  %39 = load i32, ptr %5, align 4
  %40 = call ptr @GC_malloc_atomic(i64 58)
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %41, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %39)
  %43 = icmp eq ptr %41, null
  %44 = select i1 %43, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %41
  %45 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
  %46 = add i64 1, %45
  %47 = call i64 @strlen(ptr %44)
  %48 = add i64 %46, %47
  %49 = add i64 %48, 8
  %50 = call ptr @GC_malloc(i64 %49)
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr i8, ptr %51, i64 0
  %53 = sub i64 %48, 0
  %54 = call ptr @__memcpy_chk(ptr %52, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8), i64 %45, i64 %53)
  %55 = add i64 0, %45
  %56 = getelementptr i8, ptr %51, i64 %55
  %57 = sub i64 %48, %55
  %58 = call ptr @__memcpy_chk(ptr %56, ptr %44, i64 %47, i64 %57)
  %59 = add i64 %55, %47
  %60 = getelementptr i8, ptr %51, i64 %59
  store i8 0, ptr %60, align 1
  invoke void @Res.constructor(ptr %37, ptr %51)
          to label %61 unwind label %92

61:                                               ; preds = %35
  store ptr %37, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %62 = load i32, ptr %5, align 4
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %79

64:                                               ; preds = %61
  %65 = load i1, ptr %1, align 1
  br i1 %65, label %66, label %70

66:                                               ; preds = %64
  store i1 false, ptr %1, align 1
  %67 = load ptr, ptr %2, align 8
  %68 = getelementptr %Res, ptr %67, i32 0, i32 0
  invoke void @Res.dispose(ptr %67)
          to label %69 unwind label %92

69:                                               ; preds = %66
  br label %70

70:                                               ; preds = %69, %64
  %71 = load i1, ptr %3, align 1
  br i1 %71, label %72, label %76

72:                                               ; preds = %70
  store i1 false, ptr %3, align 1
  %73 = load ptr, ptr %4, align 8
  %74 = getelementptr %Res, ptr %73, i32 0, i32 0
  invoke void @Res.dispose(ptr %73)
          to label %75 unwind label %92

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %70
  %77 = load i32, ptr %5, align 4
  %78 = add i32 %77, 1
  store i32 %78, ptr %5, align 4
  br label %6

79:                                               ; preds = %61
  %80 = load i1, ptr %1, align 1
  br i1 %80, label %81, label %85

81:                                               ; preds = %79
  store i1 false, ptr %1, align 1
  %82 = load ptr, ptr %2, align 8
  %83 = getelementptr %Res, ptr %82, i32 0, i32 0
  invoke void @Res.dispose(ptr %82)
          to label %84 unwind label %92

84:                                               ; preds = %81
  br label %85

85:                                               ; preds = %84, %79
  %86 = load i1, ptr %3, align 1
  br i1 %86, label %87, label %91

87:                                               ; preds = %85
  store i1 false, ptr %3, align 1
  %88 = load ptr, ptr %4, align 8
  %89 = getelementptr %Res, ptr %88, i32 0, i32 0
  invoke void @Res.dispose(ptr %88)
          to label %90 unwind label %92

90:                                               ; preds = %87
  br label %91

91:                                               ; preds = %90, %85
  br label %104

92:                                               ; preds = %72, %66, %87, %81, %35, %9
  %93 = landingpad { ptr, i32 }
          cleanup
  %94 = load i1, ptr %1, align 1
  br i1 %94, label %95, label %98

95:                                               ; preds = %92
  store i1 false, ptr %1, align 1
  %96 = load ptr, ptr %2, align 8
  %97 = getelementptr %Res, ptr %96, i32 0, i32 0
  call void @Res.dispose(ptr %96)
  br label %98

98:                                               ; preds = %95, %92
  %99 = load i1, ptr %3, align 1
  br i1 %99, label %100, label %103

100:                                              ; preds = %98
  store i1 false, ptr %3, align 1
  %101 = load ptr, ptr %4, align 8
  %102 = getelementptr %Res, ptr %101, i32 0, i32 0
  call void @Res.dispose(ptr %101)
  br label %103

103:                                              ; preds = %100, %98
  resume { ptr, i32 } %93

104:                                              ; preds = %91, %6
  ret void
}

; Function Attrs: mustprogress
define void @switchCase(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  store double %0, ptr %6, align 8
  %7 = load double, ptr %6, align 8
  store i1 false, ptr %4, align 1
  store i1 false, ptr %2, align 1
  %8 = fptosi double %7 to i64
  %9 = trunc i64 %8 to i32
  %10 = icmp eq i32 %9, 1
  br i1 %10, label %11, label %20

11:                                               ; preds = %1
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %Res, ptr %13, i32 0, i32 0
  store ptr @Res..vtbl, ptr %14, align 8
  invoke void @Res.constructor(ptr %13, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %15 unwind label %52

15:                                               ; preds = %11
  store ptr %13, ptr %5, align 8
  store i1 true, ptr %4, align 1
  %16 = call ptr @GC_malloc(i64 24)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr %Res, ptr %17, i32 0, i32 0
  store ptr @Res..vtbl, ptr %18, align 8
  invoke void @Res.constructor(ptr %17, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %19 unwind label %52

19:                                               ; preds = %15
  store ptr %17, ptr %3, align 8
  store i1 true, ptr %2, align 1
  br label %20

20:                                               ; preds = %19, %1
  %21 = load i1, ptr %2, align 1
  br i1 %21, label %22, label %26

22:                                               ; preds = %20
  store i1 false, ptr %2, align 1
  %23 = load ptr, ptr %3, align 8
  %24 = getelementptr %Res, ptr %23, i32 0, i32 0
  invoke void @Res.dispose(ptr %23)
          to label %25 unwind label %52

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %25, %20
  %27 = load i1, ptr %4, align 1
  br i1 %27, label %28, label %32

28:                                               ; preds = %26
  store i1 false, ptr %4, align 1
  %29 = load ptr, ptr %5, align 8
  %30 = getelementptr %Res, ptr %29, i32 0, i32 0
  invoke void @Res.dispose(ptr %29)
          to label %31 unwind label %52

31:                                               ; preds = %28
  br label %32

32:                                               ; preds = %31, %26
  %33 = load ptr, ptr @trail, align 8
  %34 = icmp eq ptr %33, null
  %35 = select i1 %34, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %33
  %36 = call i64 @strlen(ptr %35)
  %37 = add i64 1, %36
  %38 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %39 = add i64 %37, %38
  %40 = add i64 %39, 8
  %41 = call ptr @GC_malloc(i64 %40)
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr i8, ptr %42, i64 0
  %44 = sub i64 %39, 0
  %45 = call ptr @__memcpy_chk(ptr %43, ptr %35, i64 %36, i64 %44)
  %46 = add i64 0, %36
  %47 = getelementptr i8, ptr %42, i64 %46
  %48 = sub i64 %39, %46
  %49 = call ptr @__memcpy_chk(ptr %47, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %38, i64 %48)
  %50 = add i64 %46, %38
  %51 = getelementptr i8, ptr %42, i64 %50
  store i8 0, ptr %51, align 1
  store ptr %42, ptr @trail, align 8
  ret void

52:                                               ; preds = %28, %22, %15, %11
  %53 = landingpad { ptr, i32 }
          cleanup
  %54 = load i1, ptr %2, align 1
  br i1 %54, label %55, label %58

55:                                               ; preds = %52
  store i1 false, ptr %2, align 1
  %56 = load ptr, ptr %3, align 8
  %57 = getelementptr %Res, ptr %56, i32 0, i32 0
  call void @Res.dispose(ptr %56)
  br label %58

58:                                               ; preds = %55, %52
  %59 = load i1, ptr %4, align 1
  br i1 %59, label %60, label %63

60:                                               ; preds = %58
  store i1 false, ptr %4, align 1
  %61 = load ptr, ptr %5, align 8
  %62 = getelementptr %Res, ptr %61, i32 0, i32 0
  call void @Res.dispose(ptr %61)
  br label %63

63:                                               ; preds = %60, %58
  resume { ptr, i32 } %53
}

; Function Attrs: mustprogress
define i1 @expectThrow({ ptr, ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca i1, align 1
  store { ptr, ptr, ptr } %0, ptr %3, align 8
  %5 = load { ptr, ptr, ptr }, ptr %3, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %10(ptr %6)
          to label %11 unwind label %18

11:                                               ; preds = %9
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %13()
          to label %14 unwind label %18

14:                                               ; preds = %12
  br label %15

15:                                               ; preds = %11, %14
  br label %16

16:                                               ; preds = %15
  br label %17

17:                                               ; preds = %16
  store i1 false, ptr %4, align 1
  br label %30

18:                                               ; preds = %9, %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %23
  br label %32

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %2, align 4
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %30

30:                                               ; preds = %17, %26
  %31 = load i1, ptr %4, align 1
  ret i1 %31

32:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %23
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define void @check({ ptr, ptr, ptr } %0, ptr %1, ptr %2) #2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca { ptr, ptr, ptr }, align 8
  store { ptr, ptr, ptr } %0, ptr %6, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %3
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  call void %12(ptr %8)
  br label %15

13:                                               ; preds = %3
  %14 = extractvalue { ptr, ptr, ptr } %7, 0
  call void %14()
  br label %15

15:                                               ; preds = %11, %13
  br label %16

16:                                               ; preds = %15
  %17 = load ptr, ptr %4, align 8
  %18 = load ptr, ptr @trail, align 8
  %19 = icmp eq ptr %17, null
  %20 = select i1 %19, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %17
  %21 = icmp eq ptr %18, null
  %22 = select i1 %21, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %18
  %23 = call i64 @strlen(ptr %20)
  %24 = add i64 1, %23
  %25 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_584909432750204567, i64 8))
  %26 = add i64 %24, %25
  %27 = call i64 @strlen(ptr %22)
  %28 = add i64 %26, %27
  %29 = add i64 %28, 8
  %30 = call ptr @GC_malloc(i64 %29)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr i8, ptr %31, i64 0
  %33 = sub i64 %28, 0
  %34 = call ptr @__memcpy_chk(ptr %32, ptr %20, i64 %23, i64 %33)
  %35 = add i64 0, %23
  %36 = getelementptr i8, ptr %31, i64 %35
  %37 = sub i64 %28, %35
  %38 = call ptr @__memcpy_chk(ptr %36, ptr getelementptr inbounds nuw (i8, ptr @s_584909432750204567, i64 8), i64 %25, i64 %37)
  %39 = add i64 %35, %25
  %40 = getelementptr i8, ptr %31, i64 %39
  %41 = sub i64 %28, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %22, i64 %27, i64 %41)
  %43 = add i64 %39, %27
  %44 = getelementptr i8, ptr %31, i64 %43
  store i8 0, ptr %44, align 1
  %45 = icmp eq ptr %31, null
  %46 = select i1 %45, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %31
  call void @puts(ptr %46)
  %47 = load ptr, ptr @trail, align 8
  %48 = load ptr, ptr %5, align 8
  %49 = ptrtoint ptr %47 to i64
  %50 = ptrtoint ptr %48 to i64
  %51 = icmp ne i64 %49, 0
  %52 = icmp ne i64 %50, 0
  %53 = zext i1 %51 to i32
  %54 = zext i1 %52 to i32
  %55 = and i32 %53, %54
  %56 = icmp ne i32 %55, 0
  br i1 %56, label %57, label %60

57:                                               ; preds = %16
  %58 = call i32 @strcmp(ptr %47, ptr %48)
  %59 = icmp eq i32 %58, 0
  br label %62

60:                                               ; preds = %16
  %61 = icmp eq i64 %49, %50
  br label %62

62:                                               ; preds = %57, %60
  %63 = phi i1 [ %61, %60 ], [ %59, %57 ]
  br label %64

64:                                               ; preds = %62
  %65 = load ptr, ptr %4, align 8
  br i1 %63, label %66, label %68

66:                                               ; preds = %64
  ret void

67:                                               ; preds = %68
  unreachable

68:                                               ; preds = %64
  %69 = call i32 @fflush(ptr null)
  %70 = icmp eq ptr %65, null
  %71 = select i1 %70, ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr %65
  call void @__assert_fail(ptr %71, ptr getelementptr inbounds nuw (i8, ptr @f_4044071365934239186, i64 8), i32 126, ptr null)
  br label %67
}

; Function Attrs: mustprogress
define void @.f_main..afL132C11FH3933701474() #2 {
  call void @earlyReturn(i1 true)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL133C11FH3933701474() #2 {
  call void @earlyReturn(i1 false)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL134C11FH3933701474() #2 {
  %1 = call i1 @expectThrow({ ptr, ptr, ptr } { ptr @throwsInBlock, ptr null, ptr null })
  br i1 %1, label %2, label %4

2:                                                ; preds = %0
  ret void

3:                                                ; preds = %4
  unreachable

4:                                                ; preds = %0
  %5 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2432362340776716864, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4044071365934239186, i64 8), i32 134, ptr null)
  br label %3
}

; Function Attrs: mustprogress
define void @.f_main..afL138C11FH3933701474() #2 {
  call void @switchCase(double 1.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @check({ ptr, ptr, ptr } { ptr @plainBlock, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_4970319309024353835, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_10625962022409473026, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @functionBody, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_17181858102171564713, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_9316178885392575187, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL132C11FH3933701474, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_1309934839817893640, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14848773516908418703, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL133C11FH3933701474, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_739271353490714855, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_8473372317540028708, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL134C11FH3933701474, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_1309934839817893640, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_1254165271164641062, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @throwsInTryBody, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_4159161771213527651, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_5966407086411829586, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @throwingDispose, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_11545804706003810958, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_7575827081894217403, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @loopBody, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_1538657466990327383, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_6620153503825695794, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL138C11FH3933701474, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_8676791534656714797, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_9120454858007850942, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
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

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

