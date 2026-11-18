; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr, ptr }
%Shape = type { ptr, ptr }

$Res..size = comdat any

$Shape..size = comdat any

@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_3308291604520544166 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFShape\00", align 8
@s_12638145419113769978 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF;\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_15839897912459002582 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFafter;\00", align 8
@s_9739607905776723093 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody1;\00", align 8
@s_9740593068195410924 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody0;\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_11012790127938244031 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdefault;\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_3446774449893561193 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunreached;\00", align 8
@s_619343937915704565 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfn\00", align 8
@s_14046730643414667274 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFend\00", align 8
@s_16982411286042166782 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdefault\00", align 8
@s_13068990443679199185 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcase\00", align 8
@s_14838486305852456427 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFloop\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_11858887882840284586 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody;\00", align 8
@s_8684338226774033990 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFsquare\00", align 8
@s_3817786554982179638 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcircle;\00", align 8
@s_3196563492271544521 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcircle\00", align 8
@s_14599009836556477644 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdef\00", align 8
@s_1875936269717626031 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFone\00", align 8
@s_2461754175573238659 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFzero\00", align 8
@s_12638193797625411262 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFo\00", align 8
@s_12638116831811436492 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF!\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_12638202593718436950 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFw\00", align 8
@s_12638203693230065161 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv\00", align 8
@s_584909432750204567 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF: \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_16907858675607739117 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00switch_case_scope_exit.ts\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_4313850459576425155 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv0wv1;v2x;!;\00", align 8
@s_2394902519263854347 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFobdefb\00", align 8
@s_5873411151622665522 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFoaonea\00", align 8
@s_4459931025994418872 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFolong\00", align 8
@s_14837330719131395891 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFlong\00", align 8
@s_16843719094328038368 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFozeroone\00", align 8
@s_8803624280936245040 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch on a property, other case\00", align 8
@s_8791758811777272261 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcircle;after;\00", align 8
@s_2624917850085765850 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch on a property\00", align 8
@s_17980368551178335669 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody;square;after;\00", align 8
@s_1035073461916539064 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch in a loop body with a using\00", align 8
@s_10920641262281999976 = internal constant [92 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcase0;loop0;case1;end1;loop1;default2;end2;loop2;default;default3;end3;loop3;after;\00", align 8
@s_15651424497846631000 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak from an if, case not taken\00", align 8
@s_2719078386004316387 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody1;after;fn;\00", align 8
@s_9359698059718277970 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak from an if in a case\00", align 8
@s_5169259296458898883 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa;after;fn;\00", align 8
@s_8763251578505882867 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbraced cases falling through\00", align 8
@s_17030157935117728868 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody0;a;body1;b;after;\00", align 8
@s_8352128398236822016 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using in each of two cases, first skipped\00", align 8
@s_12123086994723438001 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody1;b;after;\00", align 8
@s_3602924200895940314 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using in each of two cases\00", align 8
@s_17267362973927909224 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody0;body1;b;a;after;\00", align 8
@s_1190972510352184004 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdefault;after;\00", align 8
@s_12151427799237431201 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfalling through\00", align 8
@s_11703912382116847995 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody0;body1;a;after;\00", align 8
@s_17800509762665512158 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFno case matched\00", align 8
@s_8813813054881352361 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFusing in a case not taken\00", align 8
@s_6154944742766402726 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody1;after;\00", align 8
@s_2946054492843531381 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFusing + break\00", align 8
@s_10830661723522802543 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody0;a;after;\00", align 8
@m_4371274227580367113 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstrings, fall through\00", align 8
@m_11410955366148025694 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstrings, break from an if\00", align 8
@m_7857526575750099638 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstrings, case 1\00", align 8
@m_14779713799370698046 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstrings, default\00", align 8
@m_1996736543775029239 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstrings in a loop\00", align 8
@trail = private global ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 16, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@Shape..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_3308291604520544166, i64 8)
@Shape..size = global i64 16, comdat
@Shape..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Shape..instanceOf, ptr @Shape..new, ptr @Shape..rtti, ptr @Shape..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

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
define i1 @Res..instanceOf(ptr %0, ptr %1) #1 {
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
define ptr @Res..new() #1 {
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
define void @Res.constructor(ptr %0, ptr %1) #1 {
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
define void @Res.dispose(ptr %0) #1 {
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
define void @caseWithBreak(double %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store double %0, ptr %4, align 8
  %5 = load double, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %6 = fptosi double %5 to i64
  %7 = trunc i64 %6 to i32
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %33

9:                                                ; preds = %1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Res, ptr %11, i32 0, i32 0
  store ptr @Res..vtbl, ptr %12, align 8
  invoke void @Res.constructor(ptr %11, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %13 unwind label %83

13:                                               ; preds = %9
  store ptr %11, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %14 = load ptr, ptr @trail, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr %16)
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8))
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
  %30 = call ptr @__memcpy_chk(ptr %28, ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8), i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  store ptr %23, ptr @trail, align 8
  br label %57

33:                                               ; preds = %1
  %34 = fptosi double %5 to i64
  %35 = trunc i64 %34 to i32
  %36 = icmp eq i32 %35, 1
  br i1 %36, label %37, label %57

37:                                               ; preds = %33
  %38 = load ptr, ptr @trail, align 8
  %39 = icmp eq ptr %38, null
  %40 = select i1 %39, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %38
  %41 = call i64 @strlen(ptr %40)
  %42 = add i64 1, %41
  %43 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8))
  %44 = add i64 %42, %43
  %45 = add i64 %44, 8
  %46 = call ptr @GC_malloc(i64 %45)
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr i8, ptr %47, i64 0
  %49 = sub i64 %44, 0
  %50 = call ptr @__memcpy_chk(ptr %48, ptr %40, i64 %41, i64 %49)
  %51 = add i64 0, %41
  %52 = getelementptr i8, ptr %47, i64 %51
  %53 = sub i64 %44, %51
  %54 = call ptr @__memcpy_chk(ptr %52, ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8), i64 %43, i64 %53)
  %55 = add i64 %51, %43
  %56 = getelementptr i8, ptr %47, i64 %55
  store i8 0, ptr %56, align 1
  store ptr %47, ptr @trail, align 8
  br label %57

57:                                               ; preds = %13, %37, %33
  %58 = load i1, ptr %2, align 1
  br i1 %58, label %59, label %63

59:                                               ; preds = %57
  store i1 false, ptr %2, align 1
  %60 = load ptr, ptr %3, align 8
  %61 = getelementptr %Res, ptr %60, i32 0, i32 0
  invoke void @Res.dispose(ptr %60)
          to label %62 unwind label %83

62:                                               ; preds = %59
  br label %63

63:                                               ; preds = %62, %57
  %64 = load ptr, ptr @trail, align 8
  %65 = icmp eq ptr %64, null
  %66 = select i1 %65, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %64
  %67 = call i64 @strlen(ptr %66)
  %68 = add i64 1, %67
  %69 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %70 = add i64 %68, %69
  %71 = add i64 %70, 8
  %72 = call ptr @GC_malloc(i64 %71)
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr i8, ptr %73, i64 0
  %75 = sub i64 %70, 0
  %76 = call ptr @__memcpy_chk(ptr %74, ptr %66, i64 %67, i64 %75)
  %77 = add i64 0, %67
  %78 = getelementptr i8, ptr %73, i64 %77
  %79 = sub i64 %70, %77
  %80 = call ptr @__memcpy_chk(ptr %78, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %69, i64 %79)
  %81 = add i64 %77, %69
  %82 = getelementptr i8, ptr %73, i64 %81
  store i8 0, ptr %82, align 1
  store ptr %73, ptr @trail, align 8
  ret void

83:                                               ; preds = %59, %9
  %84 = landingpad { ptr, i32 }
          cleanup
  %85 = load i1, ptr %2, align 1
  br i1 %85, label %86, label %89

86:                                               ; preds = %83
  store i1 false, ptr %2, align 1
  %87 = load ptr, ptr %3, align 8
  %88 = getelementptr %Res, ptr %87, i32 0, i32 0
  call void @Res.dispose(ptr %87)
  br label %89

89:                                               ; preds = %86, %83
  resume { ptr, i32 } %84
}

; Function Attrs: mustprogress
define void @fallThrough(double %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store double %0, ptr %4, align 8
  %5 = load double, ptr %4, align 8
  store i1 false, ptr %2, align 1
  %6 = fptosi double %5 to i64
  %7 = trunc i64 %6 to i32
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %33

9:                                                ; preds = %1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Res, ptr %11, i32 0, i32 0
  store ptr @Res..vtbl, ptr %12, align 8
  invoke void @Res.constructor(ptr %11, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %13 unwind label %103

13:                                               ; preds = %9
  store ptr %11, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %14 = load ptr, ptr @trail, align 8
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr %16)
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8))
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
  %30 = call ptr @__memcpy_chk(ptr %28, ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8), i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  store ptr %23, ptr @trail, align 8
  br label %37

33:                                               ; preds = %1
  %34 = fptosi double %5 to i64
  %35 = trunc i64 %34 to i32
  %36 = icmp eq i32 %35, 1
  br i1 %36, label %37, label %57

37:                                               ; preds = %13, %33
  %38 = load ptr, ptr @trail, align 8
  %39 = icmp eq ptr %38, null
  %40 = select i1 %39, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %38
  %41 = call i64 @strlen(ptr %40)
  %42 = add i64 1, %41
  %43 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8))
  %44 = add i64 %42, %43
  %45 = add i64 %44, 8
  %46 = call ptr @GC_malloc(i64 %45)
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr i8, ptr %47, i64 0
  %49 = sub i64 %44, 0
  %50 = call ptr @__memcpy_chk(ptr %48, ptr %40, i64 %41, i64 %49)
  %51 = add i64 0, %41
  %52 = getelementptr i8, ptr %47, i64 %51
  %53 = sub i64 %44, %51
  %54 = call ptr @__memcpy_chk(ptr %52, ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8), i64 %43, i64 %53)
  %55 = add i64 %51, %43
  %56 = getelementptr i8, ptr %47, i64 %55
  store i8 0, ptr %56, align 1
  store ptr %47, ptr @trail, align 8
  br label %77

57:                                               ; preds = %33
  %58 = load ptr, ptr @trail, align 8
  %59 = icmp eq ptr %58, null
  %60 = select i1 %59, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %58
  %61 = call i64 @strlen(ptr %60)
  %62 = add i64 1, %61
  %63 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_11012790127938244031, i64 8))
  %64 = add i64 %62, %63
  %65 = add i64 %64, 8
  %66 = call ptr @GC_malloc(i64 %65)
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr i8, ptr %67, i64 0
  %69 = sub i64 %64, 0
  %70 = call ptr @__memcpy_chk(ptr %68, ptr %60, i64 %61, i64 %69)
  %71 = add i64 0, %61
  %72 = getelementptr i8, ptr %67, i64 %71
  %73 = sub i64 %64, %71
  %74 = call ptr @__memcpy_chk(ptr %72, ptr getelementptr inbounds nuw (i8, ptr @s_11012790127938244031, i64 8), i64 %63, i64 %73)
  %75 = add i64 %71, %63
  %76 = getelementptr i8, ptr %67, i64 %75
  store i8 0, ptr %76, align 1
  store ptr %67, ptr @trail, align 8
  br label %77

77:                                               ; preds = %37, %57
  %78 = load i1, ptr %2, align 1
  br i1 %78, label %79, label %83

79:                                               ; preds = %77
  store i1 false, ptr %2, align 1
  %80 = load ptr, ptr %3, align 8
  %81 = getelementptr %Res, ptr %80, i32 0, i32 0
  invoke void @Res.dispose(ptr %80)
          to label %82 unwind label %103

82:                                               ; preds = %79
  br label %83

83:                                               ; preds = %82, %77
  %84 = load ptr, ptr @trail, align 8
  %85 = icmp eq ptr %84, null
  %86 = select i1 %85, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %84
  %87 = call i64 @strlen(ptr %86)
  %88 = add i64 1, %87
  %89 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %90 = add i64 %88, %89
  %91 = add i64 %90, 8
  %92 = call ptr @GC_malloc(i64 %91)
  %93 = getelementptr i8, ptr %92, i64 8
  %94 = getelementptr i8, ptr %93, i64 0
  %95 = sub i64 %90, 0
  %96 = call ptr @__memcpy_chk(ptr %94, ptr %86, i64 %87, i64 %95)
  %97 = add i64 0, %87
  %98 = getelementptr i8, ptr %93, i64 %97
  %99 = sub i64 %90, %97
  %100 = call ptr @__memcpy_chk(ptr %98, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %89, i64 %99)
  %101 = add i64 %97, %89
  %102 = getelementptr i8, ptr %93, i64 %101
  store i8 0, ptr %102, align 1
  store ptr %93, ptr @trail, align 8
  ret void

103:                                              ; preds = %79, %9
  %104 = landingpad { ptr, i32 }
          cleanup
  %105 = load i1, ptr %2, align 1
  br i1 %105, label %106, label %109

106:                                              ; preds = %103
  store i1 false, ptr %2, align 1
  %107 = load ptr, ptr %3, align 8
  %108 = getelementptr %Res, ptr %107, i32 0, i32 0
  call void @Res.dispose(ptr %107)
  br label %109

109:                                              ; preds = %106, %103
  resume { ptr, i32 } %104
}

; Function Attrs: mustprogress
define void @usingInTwoCases(double %0) #2 personality ptr @__gxx_personality_v0 {
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
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %35

11:                                               ; preds = %1
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %Res, ptr %13, i32 0, i32 0
  store ptr @Res..vtbl, ptr %14, align 8
  invoke void @Res.constructor(ptr %13, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %15 unwind label %95

15:                                               ; preds = %11
  store ptr %13, ptr %5, align 8
  store i1 true, ptr %4, align 1
  %16 = load ptr, ptr @trail, align 8
  %17 = icmp eq ptr %16, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %16
  %19 = call i64 @strlen(ptr %18)
  %20 = add i64 1, %19
  %21 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8))
  %22 = add i64 %20, %21
  %23 = add i64 %22, 8
  %24 = call ptr @GC_malloc(i64 %23)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %22, 0
  %28 = call ptr @__memcpy_chk(ptr %26, ptr %18, i64 %19, i64 %27)
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %22, %29
  %32 = call ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8), i64 %21, i64 %31)
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  store i8 0, ptr %34, align 1
  store ptr %25, ptr @trail, align 8
  br label %39

35:                                               ; preds = %1
  %36 = fptosi double %7 to i64
  %37 = trunc i64 %36 to i32
  %38 = icmp eq i32 %37, 1
  br i1 %38, label %39, label %63

39:                                               ; preds = %15, %35
  %40 = call ptr @GC_malloc(i64 24)
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr %Res, ptr %41, i32 0, i32 0
  store ptr @Res..vtbl, ptr %42, align 8
  invoke void @Res.constructor(ptr %41, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %43 unwind label %95

43:                                               ; preds = %39
  store ptr %41, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %44 = load ptr, ptr @trail, align 8
  %45 = icmp eq ptr %44, null
  %46 = select i1 %45, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %44
  %47 = call i64 @strlen(ptr %46)
  %48 = add i64 1, %47
  %49 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8))
  %50 = add i64 %48, %49
  %51 = add i64 %50, 8
  %52 = call ptr @GC_malloc(i64 %51)
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = getelementptr i8, ptr %53, i64 0
  %55 = sub i64 %50, 0
  %56 = call ptr @__memcpy_chk(ptr %54, ptr %46, i64 %47, i64 %55)
  %57 = add i64 0, %47
  %58 = getelementptr i8, ptr %53, i64 %57
  %59 = sub i64 %50, %57
  %60 = call ptr @__memcpy_chk(ptr %58, ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8), i64 %49, i64 %59)
  %61 = add i64 %57, %49
  %62 = getelementptr i8, ptr %53, i64 %61
  store i8 0, ptr %62, align 1
  store ptr %53, ptr @trail, align 8
  br label %63

63:                                               ; preds = %43, %35
  %64 = load i1, ptr %2, align 1
  br i1 %64, label %65, label %69

65:                                               ; preds = %63
  store i1 false, ptr %2, align 1
  %66 = load ptr, ptr %3, align 8
  %67 = getelementptr %Res, ptr %66, i32 0, i32 0
  invoke void @Res.dispose(ptr %66)
          to label %68 unwind label %95

68:                                               ; preds = %65
  br label %69

69:                                               ; preds = %68, %63
  %70 = load i1, ptr %4, align 1
  br i1 %70, label %71, label %75

71:                                               ; preds = %69
  store i1 false, ptr %4, align 1
  %72 = load ptr, ptr %5, align 8
  %73 = getelementptr %Res, ptr %72, i32 0, i32 0
  invoke void @Res.dispose(ptr %72)
          to label %74 unwind label %95

74:                                               ; preds = %71
  br label %75

75:                                               ; preds = %74, %69
  %76 = load ptr, ptr @trail, align 8
  %77 = icmp eq ptr %76, null
  %78 = select i1 %77, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %76
  %79 = call i64 @strlen(ptr %78)
  %80 = add i64 1, %79
  %81 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %82 = add i64 %80, %81
  %83 = add i64 %82, 8
  %84 = call ptr @GC_malloc(i64 %83)
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = getelementptr i8, ptr %85, i64 0
  %87 = sub i64 %82, 0
  %88 = call ptr @__memcpy_chk(ptr %86, ptr %78, i64 %79, i64 %87)
  %89 = add i64 0, %79
  %90 = getelementptr i8, ptr %85, i64 %89
  %91 = sub i64 %82, %89
  %92 = call ptr @__memcpy_chk(ptr %90, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %81, i64 %91)
  %93 = add i64 %89, %81
  %94 = getelementptr i8, ptr %85, i64 %93
  store i8 0, ptr %94, align 1
  store ptr %85, ptr @trail, align 8
  ret void

95:                                               ; preds = %71, %65, %39, %11
  %96 = landingpad { ptr, i32 }
          cleanup
  %97 = load i1, ptr %2, align 1
  br i1 %97, label %98, label %101

98:                                               ; preds = %95
  store i1 false, ptr %2, align 1
  %99 = load ptr, ptr %3, align 8
  %100 = getelementptr %Res, ptr %99, i32 0, i32 0
  call void @Res.dispose(ptr %99)
  br label %101

101:                                              ; preds = %98, %95
  %102 = load i1, ptr %4, align 1
  br i1 %102, label %103, label %106

103:                                              ; preds = %101
  store i1 false, ptr %4, align 1
  %104 = load ptr, ptr %5, align 8
  %105 = getelementptr %Res, ptr %104, i32 0, i32 0
  call void @Res.dispose(ptr %104)
  br label %106

106:                                              ; preds = %103, %101
  resume { ptr, i32 } %96
}

; Function Attrs: mustprogress
define void @bracedCases(double %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  store double %0, ptr %6, align 8
  %7 = load double, ptr %6, align 8
  %8 = fptosi double %7 to i64
  %9 = trunc i64 %8 to i32
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %48

11:                                               ; preds = %1
  store i1 false, ptr %4, align 1
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %Res, ptr %13, i32 0, i32 0
  store ptr @Res..vtbl, ptr %14, align 8
  invoke void @Res.constructor(ptr %13, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %15 unwind label %41

15:                                               ; preds = %11
  store ptr %13, ptr %5, align 8
  store i1 true, ptr %4, align 1
  %16 = load ptr, ptr @trail, align 8
  %17 = icmp eq ptr %16, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %16
  %19 = call i64 @strlen(ptr %18)
  %20 = add i64 1, %19
  %21 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8))
  %22 = add i64 %20, %21
  %23 = add i64 %22, 8
  %24 = call ptr @GC_malloc(i64 %23)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %22, 0
  %28 = call ptr @__memcpy_chk(ptr %26, ptr %18, i64 %19, i64 %27)
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %22, %29
  %32 = call ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_9740593068195410924, i64 8), i64 %21, i64 %31)
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  store i8 0, ptr %34, align 1
  store ptr %25, ptr @trail, align 8
  %35 = load i1, ptr %4, align 1
  br i1 %35, label %36, label %40

36:                                               ; preds = %15
  store i1 false, ptr %4, align 1
  %37 = load ptr, ptr %5, align 8
  %38 = getelementptr %Res, ptr %37, i32 0, i32 0
  invoke void @Res.dispose(ptr %37)
          to label %39 unwind label %41

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %39, %15
  br label %52

41:                                               ; preds = %36, %11
  %42 = landingpad { ptr, i32 }
          cleanup
  %43 = load i1, ptr %4, align 1
  br i1 %43, label %44, label %47

44:                                               ; preds = %41
  store i1 false, ptr %4, align 1
  %45 = load ptr, ptr %5, align 8
  %46 = getelementptr %Res, ptr %45, i32 0, i32 0
  call void @Res.dispose(ptr %45)
  br label %47

47:                                               ; preds = %44, %41
  resume { ptr, i32 } %42

48:                                               ; preds = %1
  %49 = fptosi double %7 to i64
  %50 = trunc i64 %49 to i32
  %51 = icmp eq i32 %50, 1
  br i1 %51, label %52, label %89

52:                                               ; preds = %40, %48
  store i1 false, ptr %2, align 1
  %53 = call ptr @GC_malloc(i64 24)
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr %Res, ptr %54, i32 0, i32 0
  store ptr @Res..vtbl, ptr %55, align 8
  invoke void @Res.constructor(ptr %54, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %56 unwind label %82

56:                                               ; preds = %52
  store ptr %54, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %57 = load ptr, ptr @trail, align 8
  %58 = icmp eq ptr %57, null
  %59 = select i1 %58, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %57
  %60 = call i64 @strlen(ptr %59)
  %61 = add i64 1, %60
  %62 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8))
  %63 = add i64 %61, %62
  %64 = add i64 %63, 8
  %65 = call ptr @GC_malloc(i64 %64)
  %66 = getelementptr i8, ptr %65, i64 8
  %67 = getelementptr i8, ptr %66, i64 0
  %68 = sub i64 %63, 0
  %69 = call ptr @__memcpy_chk(ptr %67, ptr %59, i64 %60, i64 %68)
  %70 = add i64 0, %60
  %71 = getelementptr i8, ptr %66, i64 %70
  %72 = sub i64 %63, %70
  %73 = call ptr @__memcpy_chk(ptr %71, ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8), i64 %62, i64 %72)
  %74 = add i64 %70, %62
  %75 = getelementptr i8, ptr %66, i64 %74
  store i8 0, ptr %75, align 1
  store ptr %66, ptr @trail, align 8
  %76 = load i1, ptr %2, align 1
  br i1 %76, label %77, label %81

77:                                               ; preds = %56
  store i1 false, ptr %2, align 1
  %78 = load ptr, ptr %3, align 8
  %79 = getelementptr %Res, ptr %78, i32 0, i32 0
  invoke void @Res.dispose(ptr %78)
          to label %80 unwind label %82

80:                                               ; preds = %77
  br label %81

81:                                               ; preds = %80, %56
  br label %89

82:                                               ; preds = %77, %52
  %83 = landingpad { ptr, i32 }
          cleanup
  %84 = load i1, ptr %2, align 1
  br i1 %84, label %85, label %88

85:                                               ; preds = %82
  store i1 false, ptr %2, align 1
  %86 = load ptr, ptr %3, align 8
  %87 = getelementptr %Res, ptr %86, i32 0, i32 0
  call void @Res.dispose(ptr %86)
  br label %88

88:                                               ; preds = %85, %82
  resume { ptr, i32 } %83

89:                                               ; preds = %81, %48
  %90 = load ptr, ptr @trail, align 8
  %91 = icmp eq ptr %90, null
  %92 = select i1 %91, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %90
  %93 = call i64 @strlen(ptr %92)
  %94 = add i64 1, %93
  %95 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %96 = add i64 %94, %95
  %97 = add i64 %96, 8
  %98 = call ptr @GC_malloc(i64 %97)
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr i8, ptr %99, i64 0
  %101 = sub i64 %96, 0
  %102 = call ptr @__memcpy_chk(ptr %100, ptr %92, i64 %93, i64 %101)
  %103 = add i64 0, %93
  %104 = getelementptr i8, ptr %99, i64 %103
  %105 = sub i64 %96, %103
  %106 = call ptr @__memcpy_chk(ptr %104, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %95, i64 %105)
  %107 = add i64 %103, %95
  %108 = getelementptr i8, ptr %99, i64 %107
  store i8 0, ptr %108, align 1
  store ptr %99, ptr @trail, align 8
  ret void
}

; Function Attrs: mustprogress
define void @breakFromIf(double %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  store double %0, ptr %6, align 8
  store i1 false, ptr %4, align 1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Res, ptr %8, i32 0, i32 0
  store ptr @Res..vtbl, ptr %9, align 8
  invoke void @Res.constructor(ptr %8, ptr getelementptr inbounds nuw (i8, ptr @s_619343937915704565, i64 8))
          to label %10 unwind label %142

10:                                               ; preds = %1
  store ptr %8, ptr %5, align 8
  store i1 true, ptr %4, align 1
  %11 = load double, ptr %6, align 8
  store i1 false, ptr %2, align 1
  %12 = fptosi double %11 to i64
  %13 = trunc i64 %12 to i32
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %15, label %63

15:                                               ; preds = %10
  %16 = call ptr @GC_malloc(i64 24)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr %Res, ptr %17, i32 0, i32 0
  store ptr @Res..vtbl, ptr %18, align 8
  invoke void @Res.constructor(ptr %17, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %19 unwind label %94

19:                                               ; preds = %15
  store ptr %17, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %20 = load ptr, ptr %3, align 8
  %21 = getelementptr %Res, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %21, align 8
  %23 = ptrtoint ptr %22 to i64
  %24 = icmp ne i64 %23, 0
  %25 = zext i1 %24 to i32
  %26 = and i32 %25, 1
  %27 = icmp ne i32 %26, 0
  br i1 %27, label %28, label %31

28:                                               ; preds = %19
  %29 = call i32 @strcmp(ptr %22, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %30 = icmp eq i32 %29, 0
  br label %33

31:                                               ; preds = %19
  %32 = icmp eq i64 %23, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8) to i64)
  br label %33

33:                                               ; preds = %28, %31
  %34 = phi i1 [ %32, %31 ], [ %30, %28 ]
  br label %35

35:                                               ; preds = %33
  br i1 %34, label %36, label %43

36:                                               ; preds = %35
  %37 = load i1, ptr %2, align 1
  br i1 %37, label %38, label %42

38:                                               ; preds = %36
  store i1 false, ptr %2, align 1
  %39 = load ptr, ptr %3, align 8
  %40 = getelementptr %Res, ptr %39, i32 0, i32 0
  invoke void @Res.dispose(ptr %39)
          to label %41 unwind label %94

41:                                               ; preds = %38
  br label %42

42:                                               ; preds = %41, %36
  br label %115

43:                                               ; preds = %35
  %44 = load ptr, ptr @trail, align 8
  %45 = icmp eq ptr %44, null
  %46 = select i1 %45, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %44
  %47 = call i64 @strlen(ptr %46)
  %48 = add i64 1, %47
  %49 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8))
  %50 = add i64 %48, %49
  %51 = add i64 %50, 8
  %52 = call ptr @GC_malloc(i64 %51)
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = getelementptr i8, ptr %53, i64 0
  %55 = sub i64 %50, 0
  %56 = call ptr @__memcpy_chk(ptr %54, ptr %46, i64 %47, i64 %55)
  %57 = add i64 0, %47
  %58 = getelementptr i8, ptr %53, i64 %57
  %59 = sub i64 %50, %57
  %60 = call ptr @__memcpy_chk(ptr %58, ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8), i64 %49, i64 %59)
  %61 = add i64 %57, %49
  %62 = getelementptr i8, ptr %53, i64 %61
  store i8 0, ptr %62, align 1
  store ptr %53, ptr @trail, align 8
  br label %67

63:                                               ; preds = %10
  %64 = fptosi double %11 to i64
  %65 = trunc i64 %64 to i32
  %66 = icmp eq i32 %65, 1
  br i1 %66, label %67, label %87

67:                                               ; preds = %43, %63
  %68 = load ptr, ptr @trail, align 8
  %69 = icmp eq ptr %68, null
  %70 = select i1 %69, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %68
  %71 = call i64 @strlen(ptr %70)
  %72 = add i64 1, %71
  %73 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8))
  %74 = add i64 %72, %73
  %75 = add i64 %74, 8
  %76 = call ptr @GC_malloc(i64 %75)
  %77 = getelementptr i8, ptr %76, i64 8
  %78 = getelementptr i8, ptr %77, i64 0
  %79 = sub i64 %74, 0
  %80 = call ptr @__memcpy_chk(ptr %78, ptr %70, i64 %71, i64 %79)
  %81 = add i64 0, %71
  %82 = getelementptr i8, ptr %77, i64 %81
  %83 = sub i64 %74, %81
  %84 = call ptr @__memcpy_chk(ptr %82, ptr getelementptr inbounds nuw (i8, ptr @s_9739607905776723093, i64 8), i64 %73, i64 %83)
  %85 = add i64 %81, %73
  %86 = getelementptr i8, ptr %77, i64 %85
  store i8 0, ptr %86, align 1
  store ptr %77, ptr @trail, align 8
  br label %87

87:                                               ; preds = %67, %63
  %88 = load i1, ptr %2, align 1
  br i1 %88, label %89, label %93

89:                                               ; preds = %87
  store i1 false, ptr %2, align 1
  %90 = load ptr, ptr %3, align 8
  %91 = getelementptr %Res, ptr %90, i32 0, i32 0
  invoke void @Res.dispose(ptr %90)
          to label %92 unwind label %94

92:                                               ; preds = %89
  br label %93

93:                                               ; preds = %92, %87
  br label %115

94:                                               ; preds = %38, %89, %15
  %95 = landingpad { ptr, i32 }
          catch ptr null
  %96 = extractvalue { ptr, i32 } %95, 0
  %97 = call ptr @__cxa_begin_catch(ptr %96)
  %98 = load i1, ptr %2, align 1
  br i1 %98, label %99, label %103

99:                                               ; preds = %94
  store i1 false, ptr %2, align 1
  %100 = load ptr, ptr %3, align 8
  %101 = getelementptr %Res, ptr %100, i32 0, i32 0
  invoke void @Res.dispose(ptr %100)
          to label %102 unwind label %endcatch.pad2

102:                                              ; preds = %99
  br label %103

103:                                              ; preds = %102, %94
  invoke void @__cxa_rethrow()
          to label %150 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %99
  %104 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

105:                                              ; No predecessors!
  %106 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %105
  %107 = phi { ptr, i32 } [ %106, %105 ], [ %104, %endcatch.pad2 ]
  %108 = extractvalue { ptr, i32 } %107, 0
  %109 = call ptr @__cxa_begin_catch(ptr %108)
  %110 = load i1, ptr %2, align 1
  br i1 %110, label %111, label %114

111:                                              ; preds = %.caught1
  store i1 false, ptr %2, align 1
  %112 = load ptr, ptr %3, align 8
  %113 = getelementptr %Res, ptr %112, i32 0, i32 0
  invoke void @Res.dispose(ptr %112)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %111
  br label %114

114:                                              ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %150 unwind label %endcatch.pad

115:                                              ; preds = %42, %93
  %116 = load ptr, ptr @trail, align 8
  %117 = icmp eq ptr %116, null
  %118 = select i1 %117, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %116
  %119 = call i64 @strlen(ptr %118)
  %120 = add i64 1, %119
  %121 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %122 = add i64 %120, %121
  %123 = add i64 %122, 8
  %124 = call ptr @GC_malloc(i64 %123)
  %125 = getelementptr i8, ptr %124, i64 8
  %126 = getelementptr i8, ptr %125, i64 0
  %127 = sub i64 %122, 0
  %128 = call ptr @__memcpy_chk(ptr %126, ptr %118, i64 %119, i64 %127)
  %129 = add i64 0, %119
  %130 = getelementptr i8, ptr %125, i64 %129
  %131 = sub i64 %122, %129
  %132 = call ptr @__memcpy_chk(ptr %130, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %121, i64 %131)
  %133 = add i64 %129, %121
  %134 = getelementptr i8, ptr %125, i64 %133
  store i8 0, ptr %134, align 1
  store ptr %125, ptr @trail, align 8
  %135 = load i1, ptr %4, align 1
  br i1 %135, label %136, label %140

136:                                              ; preds = %115
  store i1 false, ptr %4, align 1
  %137 = load ptr, ptr %5, align 8
  %138 = getelementptr %Res, ptr %137, i32 0, i32 0
  invoke void @Res.dispose(ptr %137)
          to label %139 unwind label %142

139:                                              ; preds = %136
  br label %140

140:                                              ; preds = %139, %115
  ret void

endcatch.pad:                                     ; preds = %114, %103
  %141 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

142:                                              ; preds = %136, %1
  %143 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %142
  %144 = phi { ptr, i32 } [ %143, %142 ], [ %141, %endcatch.pad ]
  %145 = load i1, ptr %4, align 1
  br i1 %145, label %146, label %149

146:                                              ; preds = %.caught
  store i1 false, ptr %4, align 1
  %147 = load ptr, ptr %5, align 8
  %148 = getelementptr %Res, ptr %147, i32 0, i32 0
  call void @Res.dispose(ptr %147)
  br label %149

149:                                              ; preds = %146, %.caught
  resume { ptr, i32 } %144

150:                                              ; preds = %103, %114
  unreachable

endcatch.resume:                                  ; preds = %111
  %151 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %151
}

; Function Attrs: mustprogress
define void @inLoop() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  %6 = alloca ptr, align 8
  %7 = alloca i1, align 1
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  store i32 0, ptr %9, align 4
  br label %10

10:                                               ; preds = %305, %0
  %11 = load i32, ptr %9, align 4
  %12 = icmp slt i32 %11, 4
  br i1 %12, label %13, label %308

13:                                               ; preds = %10
  store i1 false, ptr %7, align 1
  %14 = call ptr @GC_malloc(i64 24)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %Res, ptr %15, i32 0, i32 0
  store ptr @Res..vtbl, ptr %16, align 8
  %17 = load i32, ptr %9, align 4
  %18 = call ptr @GC_malloc_atomic(i64 58)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %19, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %17)
  %21 = icmp eq ptr %19, null
  %22 = select i1 %21, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %19
  %23 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14838486305852456427, i64 8))
  %24 = add i64 1, %23
  %25 = call i64 @strlen(ptr %22)
  %26 = add i64 %24, %25
  %27 = add i64 %26, 8
  %28 = call ptr @GC_malloc(i64 %27)
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr i8, ptr %29, i64 0
  %31 = sub i64 %26, 0
  %32 = call ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_14838486305852456427, i64 8), i64 %23, i64 %31)
  %33 = add i64 0, %23
  %34 = getelementptr i8, ptr %29, i64 %33
  %35 = sub i64 %26, %33
  %36 = call ptr @__memcpy_chk(ptr %34, ptr %22, i64 %25, i64 %35)
  %37 = add i64 %33, %25
  %38 = getelementptr i8, ptr %29, i64 %37
  store i8 0, ptr %38, align 1
  invoke void @Res.constructor(ptr %15, ptr %29)
          to label %39 unwind label %297

39:                                               ; preds = %13
  store ptr %15, ptr %8, align 8
  store i1 true, ptr %7, align 1
  %40 = load i32, ptr %9, align 4
  store i1 false, ptr %5, align 1
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %42, label %81

42:                                               ; preds = %39
  %43 = call ptr @GC_malloc(i64 24)
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = getelementptr %Res, ptr %44, i32 0, i32 0
  store ptr @Res..vtbl, ptr %45, align 8
  %46 = load i32, ptr %9, align 4
  %47 = call ptr @GC_malloc_atomic(i64 58)
  %48 = getelementptr i8, ptr %47, i64 8
  %49 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %48, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %46)
  %50 = icmp eq ptr %48, null
  %51 = select i1 %50, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %48
  %52 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_13068990443679199185, i64 8))
  %53 = add i64 1, %52
  %54 = call i64 @strlen(ptr %51)
  %55 = add i64 %53, %54
  %56 = add i64 %55, 8
  %57 = call ptr @GC_malloc(i64 %56)
  %58 = getelementptr i8, ptr %57, i64 8
  %59 = getelementptr i8, ptr %58, i64 0
  %60 = sub i64 %55, 0
  %61 = call ptr @__memcpy_chk(ptr %59, ptr getelementptr inbounds nuw (i8, ptr @s_13068990443679199185, i64 8), i64 %52, i64 %60)
  %62 = add i64 0, %52
  %63 = getelementptr i8, ptr %58, i64 %62
  %64 = sub i64 %55, %62
  %65 = call ptr @__memcpy_chk(ptr %63, ptr %51, i64 %54, i64 %64)
  %66 = add i64 %62, %54
  %67 = getelementptr i8, ptr %58, i64 %66
  store i8 0, ptr %67, align 1
  invoke void @Res.constructor(ptr %44, ptr %58)
          to label %68 unwind label %197

68:                                               ; preds = %42
  store ptr %44, ptr %6, align 8
  store i1 true, ptr %5, align 1
  %69 = load i1, ptr %5, align 1
  br i1 %69, label %70, label %74

70:                                               ; preds = %68
  store i1 false, ptr %5, align 1
  %71 = load ptr, ptr %6, align 8
  %72 = getelementptr %Res, ptr %71, i32 0, i32 0
  invoke void @Res.dispose(ptr %71)
          to label %73 unwind label %197

73:                                               ; preds = %70
  br label %74

74:                                               ; preds = %73, %68
  %75 = load i1, ptr %7, align 1
  br i1 %75, label %76, label %80

76:                                               ; preds = %74
  store i1 false, ptr %7, align 1
  %77 = load ptr, ptr %8, align 8
  %78 = getelementptr %Res, ptr %77, i32 0, i32 0
  invoke void @Res.dispose(ptr %77)
          to label %79 unwind label %197

79:                                               ; preds = %76
  br label %80

80:                                               ; preds = %79, %74
  br label %305

81:                                               ; preds = %39
  %82 = icmp eq i32 %40, 1
  br i1 %82, label %83, label %110

83:                                               ; preds = %81
  %84 = call ptr @GC_malloc(i64 24)
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = getelementptr %Res, ptr %85, i32 0, i32 0
  store ptr @Res..vtbl, ptr %86, align 8
  %87 = load i32, ptr %9, align 4
  %88 = call ptr @GC_malloc_atomic(i64 58)
  %89 = getelementptr i8, ptr %88, i64 8
  %90 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %89, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %87)
  %91 = icmp eq ptr %89, null
  %92 = select i1 %91, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %89
  %93 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_13068990443679199185, i64 8))
  %94 = add i64 1, %93
  %95 = call i64 @strlen(ptr %92)
  %96 = add i64 %94, %95
  %97 = add i64 %96, 8
  %98 = call ptr @GC_malloc(i64 %97)
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr i8, ptr %99, i64 0
  %101 = sub i64 %96, 0
  %102 = call ptr @__memcpy_chk(ptr %100, ptr getelementptr inbounds nuw (i8, ptr @s_13068990443679199185, i64 8), i64 %93, i64 %101)
  %103 = add i64 0, %93
  %104 = getelementptr i8, ptr %99, i64 %103
  %105 = sub i64 %96, %103
  %106 = call ptr @__memcpy_chk(ptr %104, ptr %92, i64 %95, i64 %105)
  %107 = add i64 %103, %95
  %108 = getelementptr i8, ptr %99, i64 %107
  store i8 0, ptr %108, align 1
  invoke void @Res.constructor(ptr %85, ptr %99)
          to label %109 unwind label %197

109:                                              ; preds = %83
  store ptr %85, ptr %4, align 8
  store i1 true, ptr %3, align 1
  br label %178

110:                                              ; preds = %81
  %111 = call ptr @GC_malloc(i64 24)
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr %Res, ptr %112, i32 0, i32 0
  store ptr @Res..vtbl, ptr %113, align 8
  %114 = load i32, ptr %9, align 4
  %115 = call ptr @GC_malloc_atomic(i64 58)
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %116, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %114)
  %118 = icmp eq ptr %116, null
  %119 = select i1 %118, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %116
  %120 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_16982411286042166782, i64 8))
  %121 = add i64 1, %120
  %122 = call i64 @strlen(ptr %119)
  %123 = add i64 %121, %122
  %124 = add i64 %123, 8
  %125 = call ptr @GC_malloc(i64 %124)
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr i8, ptr %126, i64 0
  %128 = sub i64 %123, 0
  %129 = call ptr @__memcpy_chk(ptr %127, ptr getelementptr inbounds nuw (i8, ptr @s_16982411286042166782, i64 8), i64 %120, i64 %128)
  %130 = add i64 0, %120
  %131 = getelementptr i8, ptr %126, i64 %130
  %132 = sub i64 %123, %130
  %133 = call ptr @__memcpy_chk(ptr %131, ptr %119, i64 %122, i64 %132)
  %134 = add i64 %130, %122
  %135 = getelementptr i8, ptr %126, i64 %134
  store i8 0, ptr %135, align 1
  invoke void @Res.constructor(ptr %112, ptr %126)
          to label %136 unwind label %197

136:                                              ; preds = %110
  store ptr %112, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %137 = load i32, ptr %9, align 4
  %138 = icmp eq i32 %137, 2
  br i1 %138, label %139, label %158

139:                                              ; preds = %136
  %140 = load i1, ptr %1, align 1
  br i1 %140, label %141, label %145

141:                                              ; preds = %139
  store i1 false, ptr %1, align 1
  %142 = load ptr, ptr %2, align 8
  %143 = getelementptr %Res, ptr %142, i32 0, i32 0
  invoke void @Res.dispose(ptr %142)
          to label %144 unwind label %197

144:                                              ; preds = %141
  br label %145

145:                                              ; preds = %144, %139
  %146 = load i1, ptr %3, align 1
  br i1 %146, label %147, label %151

147:                                              ; preds = %145
  store i1 false, ptr %3, align 1
  %148 = load ptr, ptr %4, align 8
  %149 = getelementptr %Res, ptr %148, i32 0, i32 0
  invoke void @Res.dispose(ptr %148)
          to label %150 unwind label %197

150:                                              ; preds = %147
  br label %151

151:                                              ; preds = %150, %145
  %152 = load i1, ptr %5, align 1
  br i1 %152, label %153, label %157

153:                                              ; preds = %151
  store i1 false, ptr %5, align 1
  %154 = load ptr, ptr %6, align 8
  %155 = getelementptr %Res, ptr %154, i32 0, i32 0
  invoke void @Res.dispose(ptr %154)
          to label %156 unwind label %197

156:                                              ; preds = %153
  br label %157

157:                                              ; preds = %156, %151
  br label %240

158:                                              ; preds = %136
  %159 = load ptr, ptr @trail, align 8
  %160 = icmp eq ptr %159, null
  %161 = select i1 %160, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %159
  %162 = call i64 @strlen(ptr %161)
  %163 = add i64 1, %162
  %164 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_11012790127938244031, i64 8))
  %165 = add i64 %163, %164
  %166 = add i64 %165, 8
  %167 = call ptr @GC_malloc(i64 %166)
  %168 = getelementptr i8, ptr %167, i64 8
  %169 = getelementptr i8, ptr %168, i64 0
  %170 = sub i64 %165, 0
  %171 = call ptr @__memcpy_chk(ptr %169, ptr %161, i64 %162, i64 %170)
  %172 = add i64 0, %162
  %173 = getelementptr i8, ptr %168, i64 %172
  %174 = sub i64 %165, %172
  %175 = call ptr @__memcpy_chk(ptr %173, ptr getelementptr inbounds nuw (i8, ptr @s_11012790127938244031, i64 8), i64 %164, i64 %174)
  %176 = add i64 %172, %164
  %177 = getelementptr i8, ptr %168, i64 %176
  store i8 0, ptr %177, align 1
  store ptr %168, ptr @trail, align 8
  br label %178

178:                                              ; preds = %109, %158
  %179 = load i1, ptr %1, align 1
  br i1 %179, label %180, label %184

180:                                              ; preds = %178
  store i1 false, ptr %1, align 1
  %181 = load ptr, ptr %2, align 8
  %182 = getelementptr %Res, ptr %181, i32 0, i32 0
  invoke void @Res.dispose(ptr %181)
          to label %183 unwind label %197

183:                                              ; preds = %180
  br label %184

184:                                              ; preds = %183, %178
  %185 = load i1, ptr %3, align 1
  br i1 %185, label %186, label %190

186:                                              ; preds = %184
  store i1 false, ptr %3, align 1
  %187 = load ptr, ptr %4, align 8
  %188 = getelementptr %Res, ptr %187, i32 0, i32 0
  invoke void @Res.dispose(ptr %187)
          to label %189 unwind label %197

189:                                              ; preds = %186
  br label %190

190:                                              ; preds = %189, %184
  %191 = load i1, ptr %5, align 1
  br i1 %191, label %192, label %196

192:                                              ; preds = %190
  store i1 false, ptr %5, align 1
  %193 = load ptr, ptr %6, align 8
  %194 = getelementptr %Res, ptr %193, i32 0, i32 0
  invoke void @Res.dispose(ptr %193)
          to label %195 unwind label %197

195:                                              ; preds = %192
  br label %196

196:                                              ; preds = %195, %190
  br label %240

197:                                              ; preds = %76, %70, %42, %192, %186, %180, %83, %153, %147, %141, %110
  %198 = landingpad { ptr, i32 }
          catch ptr null
  %199 = extractvalue { ptr, i32 } %198, 0
  %200 = call ptr @__cxa_begin_catch(ptr %199)
  %201 = load i1, ptr %1, align 1
  br i1 %201, label %202, label %206

202:                                              ; preds = %197
  store i1 false, ptr %1, align 1
  %203 = load ptr, ptr %2, align 8
  %204 = getelementptr %Res, ptr %203, i32 0, i32 0
  invoke void @Res.dispose(ptr %203)
          to label %205 unwind label %endcatch.pad2

205:                                              ; preds = %202
  br label %206

206:                                              ; preds = %205, %197
  %207 = load i1, ptr %3, align 1
  br i1 %207, label %208, label %212

208:                                              ; preds = %206
  store i1 false, ptr %3, align 1
  %209 = load ptr, ptr %4, align 8
  %210 = getelementptr %Res, ptr %209, i32 0, i32 0
  invoke void @Res.dispose(ptr %209)
          to label %211 unwind label %endcatch.pad2

211:                                              ; preds = %208
  br label %212

212:                                              ; preds = %211, %206
  %213 = load i1, ptr %5, align 1
  br i1 %213, label %214, label %218

214:                                              ; preds = %212
  store i1 false, ptr %5, align 1
  %215 = load ptr, ptr %6, align 8
  %216 = getelementptr %Res, ptr %215, i32 0, i32 0
  invoke void @Res.dispose(ptr %215)
          to label %217 unwind label %endcatch.pad2

217:                                              ; preds = %214
  br label %218

218:                                              ; preds = %217, %212
  invoke void @__cxa_rethrow()
          to label %328 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %208, %214, %202
  %219 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

220:                                              ; No predecessors!
  %221 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %220
  %222 = phi { ptr, i32 } [ %221, %220 ], [ %219, %endcatch.pad2 ]
  %223 = extractvalue { ptr, i32 } %222, 0
  %224 = call ptr @__cxa_begin_catch(ptr %223)
  %225 = load i1, ptr %1, align 1
  br i1 %225, label %226, label %229

226:                                              ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %227 = load ptr, ptr %2, align 8
  %228 = getelementptr %Res, ptr %227, i32 0, i32 0
  invoke void @Res.dispose(ptr %227)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %226
  br label %229

229:                                              ; preds = %.noexc3, %.caught1
  %230 = load i1, ptr %3, align 1
  br i1 %230, label %231, label %234

231:                                              ; preds = %229
  store i1 false, ptr %3, align 1
  %232 = load ptr, ptr %4, align 8
  %233 = getelementptr %Res, ptr %232, i32 0, i32 0
  invoke void @Res.dispose(ptr %232)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %231
  br label %234

234:                                              ; preds = %.noexc, %229
  %235 = load i1, ptr %5, align 1
  br i1 %235, label %236, label %239

236:                                              ; preds = %234
  store i1 false, ptr %5, align 1
  %237 = load ptr, ptr %6, align 8
  %238 = getelementptr %Res, ptr %237, i32 0, i32 0
  invoke void @Res.dispose(ptr %237)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %236
  br label %239

239:                                              ; preds = %.noexc4, %234
  invoke void @__cxa_rethrow()
          to label %328 unwind label %endcatch.pad

240:                                              ; preds = %196, %157
  %241 = load ptr, ptr @trail, align 8
  %242 = load i32, ptr %9, align 4
  %243 = call ptr @GC_malloc_atomic(i64 58)
  %244 = getelementptr i8, ptr %243, i64 8
  %245 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %244, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %242)
  %246 = icmp eq ptr %244, null
  %247 = select i1 %246, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %244
  %248 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14046730643414667274, i64 8))
  %249 = add i64 1, %248
  %250 = call i64 @strlen(ptr %247)
  %251 = add i64 %249, %250
  %252 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %253 = add i64 %251, %252
  %254 = add i64 %253, 8
  %255 = call ptr @GC_malloc(i64 %254)
  %256 = getelementptr i8, ptr %255, i64 8
  %257 = getelementptr i8, ptr %256, i64 0
  %258 = sub i64 %253, 0
  %259 = call ptr @__memcpy_chk(ptr %257, ptr getelementptr inbounds nuw (i8, ptr @s_14046730643414667274, i64 8), i64 %248, i64 %258)
  %260 = add i64 0, %248
  %261 = getelementptr i8, ptr %256, i64 %260
  %262 = sub i64 %253, %260
  %263 = call ptr @__memcpy_chk(ptr %261, ptr %247, i64 %250, i64 %262)
  %264 = add i64 %260, %250
  %265 = getelementptr i8, ptr %256, i64 %264
  %266 = sub i64 %253, %264
  %267 = call ptr @__memcpy_chk(ptr %265, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %252, i64 %266)
  %268 = add i64 %264, %252
  %269 = getelementptr i8, ptr %256, i64 %268
  store i8 0, ptr %269, align 1
  %270 = icmp eq ptr %241, null
  %271 = select i1 %270, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %241
  %272 = icmp eq ptr %256, null
  %273 = select i1 %272, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %256
  %274 = call i64 @strlen(ptr %271)
  %275 = add i64 1, %274
  %276 = call i64 @strlen(ptr %273)
  %277 = add i64 %275, %276
  %278 = add i64 %277, 8
  %279 = call ptr @GC_malloc(i64 %278)
  %280 = getelementptr i8, ptr %279, i64 8
  %281 = getelementptr i8, ptr %280, i64 0
  %282 = sub i64 %277, 0
  %283 = call ptr @__memcpy_chk(ptr %281, ptr %271, i64 %274, i64 %282)
  %284 = add i64 0, %274
  %285 = getelementptr i8, ptr %280, i64 %284
  %286 = sub i64 %277, %284
  %287 = call ptr @__memcpy_chk(ptr %285, ptr %273, i64 %276, i64 %286)
  %288 = add i64 %284, %276
  %289 = getelementptr i8, ptr %280, i64 %288
  store i8 0, ptr %289, align 1
  store ptr %280, ptr @trail, align 8
  %290 = load i1, ptr %7, align 1
  br i1 %290, label %291, label %295

291:                                              ; preds = %240
  store i1 false, ptr %7, align 1
  %292 = load ptr, ptr %8, align 8
  %293 = getelementptr %Res, ptr %292, i32 0, i32 0
  invoke void @Res.dispose(ptr %292)
          to label %294 unwind label %297

294:                                              ; preds = %291
  br label %295

295:                                              ; preds = %294, %240
  br label %305

endcatch.pad:                                     ; preds = %239, %218
  %296 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

297:                                              ; preds = %291, %13
  %298 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %297
  %299 = phi { ptr, i32 } [ %298, %297 ], [ %296, %endcatch.pad ]
  %300 = load i1, ptr %7, align 1
  br i1 %300, label %301, label %304

301:                                              ; preds = %.caught
  store i1 false, ptr %7, align 1
  %302 = load ptr, ptr %8, align 8
  %303 = getelementptr %Res, ptr %302, i32 0, i32 0
  call void @Res.dispose(ptr %302)
  br label %304

304:                                              ; preds = %301, %.caught
  resume { ptr, i32 } %299

305:                                              ; preds = %80, %295
  %306 = load i32, ptr %9, align 4
  %307 = add i32 %306, 1
  store i32 %307, ptr %9, align 4
  br label %10

308:                                              ; preds = %10
  %309 = load ptr, ptr @trail, align 8
  %310 = icmp eq ptr %309, null
  %311 = select i1 %310, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %309
  %312 = call i64 @strlen(ptr %311)
  %313 = add i64 1, %312
  %314 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %315 = add i64 %313, %314
  %316 = add i64 %315, 8
  %317 = call ptr @GC_malloc(i64 %316)
  %318 = getelementptr i8, ptr %317, i64 8
  %319 = getelementptr i8, ptr %318, i64 0
  %320 = sub i64 %315, 0
  %321 = call ptr @__memcpy_chk(ptr %319, ptr %311, i64 %312, i64 %320)
  %322 = add i64 0, %312
  %323 = getelementptr i8, ptr %318, i64 %322
  %324 = sub i64 %315, %322
  %325 = call ptr @__memcpy_chk(ptr %323, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %314, i64 %324)
  %326 = add i64 %322, %314
  %327 = getelementptr i8, ptr %318, i64 %326
  store i8 0, ptr %327, align 1
  store ptr %318, ptr @trail, align 8
  ret void

328:                                              ; preds = %218, %239
  unreachable

endcatch.resume:                                  ; preds = %236, %226, %231
  %329 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %329
}

; Function Attrs: mustprogress
define i1 @Shape..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Shape..rtti, align 8
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
define ptr @Shape..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Shape, ptr %3, i32 0, i32 0
  store ptr @Shape..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Shape.constructor(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Shape, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @onProperty(ptr %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Shape, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  store i1 false, ptr %2, align 1
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %1
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_3196563492271544521, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %1
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3196563492271544521, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %41

21:                                               ; preds = %20
  %22 = load ptr, ptr @trail, align 8
  %23 = icmp eq ptr %22, null
  %24 = select i1 %23, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %22
  %25 = call i64 @strlen(ptr %24)
  %26 = add i64 1, %25
  %27 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3817786554982179638, i64 8))
  %28 = add i64 %26, %27
  %29 = add i64 %28, 8
  %30 = call ptr @GC_malloc(i64 %29)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr i8, ptr %31, i64 0
  %33 = sub i64 %28, 0
  %34 = call ptr @__memcpy_chk(ptr %32, ptr %24, i64 %25, i64 %33)
  %35 = add i64 0, %25
  %36 = getelementptr i8, ptr %31, i64 %35
  %37 = sub i64 %28, %35
  %38 = call ptr @__memcpy_chk(ptr %36, ptr getelementptr inbounds nuw (i8, ptr @s_3817786554982179638, i64 8), i64 %27, i64 %37)
  %39 = add i64 %35, %27
  %40 = getelementptr i8, ptr %31, i64 %39
  store i8 0, ptr %40, align 1
  store ptr %31, ptr @trail, align 8
  br label %79

41:                                               ; preds = %20
  %42 = ptrtoint ptr %7 to i64
  %43 = icmp ne i64 %42, 0
  %44 = zext i1 %43 to i32
  %45 = and i32 %44, 1
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %50

47:                                               ; preds = %41
  %48 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_8684338226774033990, i64 8))
  %49 = icmp eq i32 %48, 0
  br label %52

50:                                               ; preds = %41
  %51 = icmp eq i64 %42, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_8684338226774033990, i64 8) to i64)
  br label %52

52:                                               ; preds = %47, %50
  %53 = phi i1 [ %51, %50 ], [ %49, %47 ]
  br label %54

54:                                               ; preds = %52
  br i1 %53, label %55, label %79

55:                                               ; preds = %54
  %56 = call ptr @GC_malloc(i64 24)
  %57 = getelementptr i8, ptr %56, i64 8
  %58 = getelementptr %Res, ptr %57, i32 0, i32 0
  store ptr @Res..vtbl, ptr %58, align 8
  invoke void @Res.constructor(ptr %57, ptr getelementptr inbounds nuw (i8, ptr @s_8684338226774033990, i64 8))
          to label %59 unwind label %105

59:                                               ; preds = %55
  store ptr %57, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %60 = load ptr, ptr @trail, align 8
  %61 = icmp eq ptr %60, null
  %62 = select i1 %61, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %60
  %63 = call i64 @strlen(ptr %62)
  %64 = add i64 1, %63
  %65 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_11858887882840284586, i64 8))
  %66 = add i64 %64, %65
  %67 = add i64 %66, 8
  %68 = call ptr @GC_malloc(i64 %67)
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr i8, ptr %69, i64 0
  %71 = sub i64 %66, 0
  %72 = call ptr @__memcpy_chk(ptr %70, ptr %62, i64 %63, i64 %71)
  %73 = add i64 0, %63
  %74 = getelementptr i8, ptr %69, i64 %73
  %75 = sub i64 %66, %73
  %76 = call ptr @__memcpy_chk(ptr %74, ptr getelementptr inbounds nuw (i8, ptr @s_11858887882840284586, i64 8), i64 %65, i64 %75)
  %77 = add i64 %73, %65
  %78 = getelementptr i8, ptr %69, i64 %77
  store i8 0, ptr %78, align 1
  store ptr %69, ptr @trail, align 8
  br label %79

79:                                               ; preds = %21, %59, %54
  %80 = load i1, ptr %2, align 1
  br i1 %80, label %81, label %85

81:                                               ; preds = %79
  store i1 false, ptr %2, align 1
  %82 = load ptr, ptr %3, align 8
  %83 = getelementptr %Res, ptr %82, i32 0, i32 0
  invoke void @Res.dispose(ptr %82)
          to label %84 unwind label %105

84:                                               ; preds = %81
  br label %85

85:                                               ; preds = %84, %79
  %86 = load ptr, ptr @trail, align 8
  %87 = icmp eq ptr %86, null
  %88 = select i1 %87, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %86
  %89 = call i64 @strlen(ptr %88)
  %90 = add i64 1, %89
  %91 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %92 = add i64 %90, %91
  %93 = add i64 %92, 8
  %94 = call ptr @GC_malloc(i64 %93)
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr i8, ptr %95, i64 0
  %97 = sub i64 %92, 0
  %98 = call ptr @__memcpy_chk(ptr %96, ptr %88, i64 %89, i64 %97)
  %99 = add i64 0, %89
  %100 = getelementptr i8, ptr %95, i64 %99
  %101 = sub i64 %92, %99
  %102 = call ptr @__memcpy_chk(ptr %100, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %91, i64 %101)
  %103 = add i64 %99, %91
  %104 = getelementptr i8, ptr %95, i64 %103
  store i8 0, ptr %104, align 1
  store ptr %95, ptr @trail, align 8
  ret void

105:                                              ; preds = %81, %55
  %106 = landingpad { ptr, i32 }
          cleanup
  %107 = load i1, ptr %2, align 1
  br i1 %107, label %108, label %111

108:                                              ; preds = %105
  store i1 false, ptr %2, align 1
  %109 = load ptr, ptr %3, align 8
  %110 = getelementptr %Res, ptr %109, i32 0, i32 0
  call void @Res.dispose(ptr %109)
  br label %111

111:                                              ; preds = %108, %105
  resume { ptr, i32 } %106
}

; Function Attrs: mustprogress
define ptr @stringsInCases(double %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca double, align 8
  %9 = alloca ptr, align 8
  store double %0, ptr %8, align 8
  store ptr %1, ptr %7, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = icmp eq ptr %10, null
  %12 = select i1 %11, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %10
  %13 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638193797625411262, i64 8))
  %14 = add i64 1, %13
  %15 = call i64 @strlen(ptr %12)
  %16 = add i64 %14, %15
  %17 = add i64 %16, 8
  %18 = call ptr @GC_malloc(i64 %17)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr i8, ptr %19, i64 0
  %21 = sub i64 %16, 0
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638193797625411262, i64 8), i64 %13, i64 %21)
  %23 = add i64 0, %13
  %24 = getelementptr i8, ptr %19, i64 %23
  %25 = sub i64 %16, %23
  %26 = call ptr @__memcpy_chk(ptr %24, ptr %12, i64 %15, i64 %25)
  %27 = add i64 %23, %15
  %28 = getelementptr i8, ptr %19, i64 %27
  store i8 0, ptr %28, align 1
  store ptr %19, ptr %6, align 8
  %29 = load double, ptr %8, align 8
  %30 = fptosi double %29 to i64
  %31 = trunc i64 %30 to i32
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %33, label %80

33:                                               ; preds = %2
  %34 = load ptr, ptr %7, align 8
  %35 = icmp eq ptr %34, null
  %36 = select i1 %35, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %34
  %37 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_2461754175573238659, i64 8))
  %38 = add i64 1, %37
  %39 = call i64 @strlen(ptr %36)
  %40 = add i64 %38, %39
  %41 = add i64 %40, 8
  %42 = call ptr @GC_malloc(i64 %41)
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr i8, ptr %43, i64 0
  %45 = sub i64 %40, 0
  %46 = call ptr @__memcpy_chk(ptr %44, ptr getelementptr inbounds nuw (i8, ptr @s_2461754175573238659, i64 8), i64 %37, i64 %45)
  %47 = add i64 0, %37
  %48 = getelementptr i8, ptr %43, i64 %47
  %49 = sub i64 %40, %47
  %50 = call ptr @__memcpy_chk(ptr %48, ptr %36, i64 %39, i64 %49)
  %51 = add i64 %47, %39
  %52 = getelementptr i8, ptr %43, i64 %51
  store i8 0, ptr %52, align 1
  store ptr %43, ptr %5, align 8
  %53 = load ptr, ptr %5, align 8
  %54 = call i64 @strlen(ptr %53)
  %55 = icmp sgt i64 %54, 4
  br i1 %55, label %56, label %57

56:                                               ; preds = %33
  br label %169

57:                                               ; preds = %33
  %58 = load ptr, ptr %6, align 8
  %59 = load ptr, ptr %5, align 8
  %60 = icmp eq ptr %58, null
  %61 = select i1 %60, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %58
  %62 = icmp eq ptr %59, null
  %63 = select i1 %62, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %59
  %64 = call i64 @strlen(ptr %61)
  %65 = add i64 1, %64
  %66 = call i64 @strlen(ptr %63)
  %67 = add i64 %65, %66
  %68 = add i64 %67, 8
  %69 = call ptr @GC_malloc(i64 %68)
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr i8, ptr %70, i64 0
  %72 = sub i64 %67, 0
  %73 = call ptr @__memcpy_chk(ptr %71, ptr %61, i64 %64, i64 %72)
  %74 = add i64 0, %64
  %75 = getelementptr i8, ptr %70, i64 %74
  %76 = sub i64 %67, %74
  %77 = call ptr @__memcpy_chk(ptr %75, ptr %63, i64 %66, i64 %76)
  %78 = add i64 %74, %66
  %79 = getelementptr i8, ptr %70, i64 %78
  store i8 0, ptr %79, align 1
  store ptr %70, ptr %6, align 8
  br label %84

80:                                               ; preds = %2
  %81 = fptosi double %29 to i64
  %82 = trunc i64 %81 to i32
  %83 = icmp eq i32 %82, 1
  br i1 %83, label %84, label %126

84:                                               ; preds = %57, %80
  %85 = load ptr, ptr %7, align 8
  %86 = icmp eq ptr %85, null
  %87 = select i1 %86, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %85
  %88 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_1875936269717626031, i64 8))
  %89 = add i64 1, %88
  %90 = call i64 @strlen(ptr %87)
  %91 = add i64 %89, %90
  %92 = add i64 %91, 8
  %93 = call ptr @GC_malloc(i64 %92)
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr i8, ptr %94, i64 0
  %96 = sub i64 %91, 0
  %97 = call ptr @__memcpy_chk(ptr %95, ptr getelementptr inbounds nuw (i8, ptr @s_1875936269717626031, i64 8), i64 %88, i64 %96)
  %98 = add i64 0, %88
  %99 = getelementptr i8, ptr %94, i64 %98
  %100 = sub i64 %91, %98
  %101 = call ptr @__memcpy_chk(ptr %99, ptr %87, i64 %90, i64 %100)
  %102 = add i64 %98, %90
  %103 = getelementptr i8, ptr %94, i64 %102
  store i8 0, ptr %103, align 1
  store ptr %94, ptr %4, align 8
  %104 = load ptr, ptr %6, align 8
  %105 = load ptr, ptr %4, align 8
  %106 = icmp eq ptr %104, null
  %107 = select i1 %106, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %104
  %108 = icmp eq ptr %105, null
  %109 = select i1 %108, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %105
  %110 = call i64 @strlen(ptr %107)
  %111 = add i64 1, %110
  %112 = call i64 @strlen(ptr %109)
  %113 = add i64 %111, %112
  %114 = add i64 %113, 8
  %115 = call ptr @GC_malloc(i64 %114)
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr i8, ptr %116, i64 0
  %118 = sub i64 %113, 0
  %119 = call ptr @__memcpy_chk(ptr %117, ptr %107, i64 %110, i64 %118)
  %120 = add i64 0, %110
  %121 = getelementptr i8, ptr %116, i64 %120
  %122 = sub i64 %113, %120
  %123 = call ptr @__memcpy_chk(ptr %121, ptr %109, i64 %112, i64 %122)
  %124 = add i64 %120, %112
  %125 = getelementptr i8, ptr %116, i64 %124
  store i8 0, ptr %125, align 1
  store ptr %116, ptr %6, align 8
  br label %168

126:                                              ; preds = %80
  %127 = load ptr, ptr %7, align 8
  %128 = icmp eq ptr %127, null
  %129 = select i1 %128, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %127
  %130 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14599009836556477644, i64 8))
  %131 = add i64 1, %130
  %132 = call i64 @strlen(ptr %129)
  %133 = add i64 %131, %132
  %134 = add i64 %133, 8
  %135 = call ptr @GC_malloc(i64 %134)
  %136 = getelementptr i8, ptr %135, i64 8
  %137 = getelementptr i8, ptr %136, i64 0
  %138 = sub i64 %133, 0
  %139 = call ptr @__memcpy_chk(ptr %137, ptr getelementptr inbounds nuw (i8, ptr @s_14599009836556477644, i64 8), i64 %130, i64 %138)
  %140 = add i64 0, %130
  %141 = getelementptr i8, ptr %136, i64 %140
  %142 = sub i64 %133, %140
  %143 = call ptr @__memcpy_chk(ptr %141, ptr %129, i64 %132, i64 %142)
  %144 = add i64 %140, %132
  %145 = getelementptr i8, ptr %136, i64 %144
  store i8 0, ptr %145, align 1
  store ptr %136, ptr %3, align 8
  %146 = load ptr, ptr %6, align 8
  %147 = load ptr, ptr %3, align 8
  %148 = icmp eq ptr %146, null
  %149 = select i1 %148, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %146
  %150 = icmp eq ptr %147, null
  %151 = select i1 %150, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %147
  %152 = call i64 @strlen(ptr %149)
  %153 = add i64 1, %152
  %154 = call i64 @strlen(ptr %151)
  %155 = add i64 %153, %154
  %156 = add i64 %155, 8
  %157 = call ptr @GC_malloc(i64 %156)
  %158 = getelementptr i8, ptr %157, i64 8
  %159 = getelementptr i8, ptr %158, i64 0
  %160 = sub i64 %155, 0
  %161 = call ptr @__memcpy_chk(ptr %159, ptr %149, i64 %152, i64 %160)
  %162 = add i64 0, %152
  %163 = getelementptr i8, ptr %158, i64 %162
  %164 = sub i64 %155, %162
  %165 = call ptr @__memcpy_chk(ptr %163, ptr %151, i64 %154, i64 %164)
  %166 = add i64 %162, %154
  %167 = getelementptr i8, ptr %158, i64 %166
  store i8 0, ptr %167, align 1
  store ptr %158, ptr %6, align 8
  br label %168

168:                                              ; preds = %84, %126
  br label %169

169:                                              ; preds = %56, %168
  %170 = load ptr, ptr %6, align 8
  store ptr %170, ptr %9, align 8
  %171 = load ptr, ptr %9, align 8
  ret ptr %171
}

; Function Attrs: mustprogress
define ptr @stringsInLoop() #1 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %5, align 8
  store i32 0, ptr %4, align 4
  br label %7

7:                                                ; preds = %188, %0
  %8 = load i32, ptr %4, align 4
  %9 = icmp slt i32 %8, 4
  br i1 %9, label %10, label %191

10:                                               ; preds = %7
  %11 = load i32, ptr %4, align 4
  %12 = call ptr @GC_malloc_atomic(i64 58)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %13, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %11)
  %15 = icmp eq ptr %13, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %13
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
  store ptr %23, ptr %3, align 8
  %33 = load i32, ptr %4, align 4
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %35, label %77

35:                                               ; preds = %10
  %36 = load ptr, ptr %3, align 8
  %37 = icmp eq ptr %36, null
  %38 = select i1 %37, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %36
  %39 = call i64 @strlen(ptr %38)
  %40 = add i64 1, %39
  %41 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638202593718436950, i64 8))
  %42 = add i64 %40, %41
  %43 = add i64 %42, 8
  %44 = call ptr @GC_malloc(i64 %43)
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr i8, ptr %45, i64 0
  %47 = sub i64 %42, 0
  %48 = call ptr @__memcpy_chk(ptr %46, ptr %38, i64 %39, i64 %47)
  %49 = add i64 0, %39
  %50 = getelementptr i8, ptr %45, i64 %49
  %51 = sub i64 %42, %49
  %52 = call ptr @__memcpy_chk(ptr %50, ptr getelementptr inbounds nuw (i8, ptr @s_12638202593718436950, i64 8), i64 %41, i64 %51)
  %53 = add i64 %49, %41
  %54 = getelementptr i8, ptr %45, i64 %53
  store i8 0, ptr %54, align 1
  store ptr %45, ptr %2, align 8
  %55 = load ptr, ptr %5, align 8
  %56 = load ptr, ptr %2, align 8
  %57 = icmp eq ptr %55, null
  %58 = select i1 %57, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %55
  %59 = icmp eq ptr %56, null
  %60 = select i1 %59, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %56
  %61 = call i64 @strlen(ptr %58)
  %62 = add i64 1, %61
  %63 = call i64 @strlen(ptr %60)
  %64 = add i64 %62, %63
  %65 = add i64 %64, 8
  %66 = call ptr @GC_malloc(i64 %65)
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr i8, ptr %67, i64 0
  %69 = sub i64 %64, 0
  %70 = call ptr @__memcpy_chk(ptr %68, ptr %58, i64 %61, i64 %69)
  %71 = add i64 0, %61
  %72 = getelementptr i8, ptr %67, i64 %71
  %73 = sub i64 %64, %71
  %74 = call ptr @__memcpy_chk(ptr %72, ptr %60, i64 %63, i64 %73)
  %75 = add i64 %71, %63
  %76 = getelementptr i8, ptr %67, i64 %75
  store i8 0, ptr %76, align 1
  store ptr %67, ptr %5, align 8
  br label %188

77:                                               ; preds = %10
  %78 = icmp eq i32 %33, 1
  br i1 %78, label %79, label %102

79:                                               ; preds = %77
  %80 = load ptr, ptr %5, align 8
  %81 = load ptr, ptr %3, align 8
  %82 = icmp eq ptr %80, null
  %83 = select i1 %82, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %80
  %84 = icmp eq ptr %81, null
  %85 = select i1 %84, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %81
  %86 = call i64 @strlen(ptr %83)
  %87 = add i64 1, %86
  %88 = call i64 @strlen(ptr %85)
  %89 = add i64 %87, %88
  %90 = add i64 %89, 8
  %91 = call ptr @GC_malloc(i64 %90)
  %92 = getelementptr i8, ptr %91, i64 8
  %93 = getelementptr i8, ptr %92, i64 0
  %94 = sub i64 %89, 0
  %95 = call ptr @__memcpy_chk(ptr %93, ptr %83, i64 %86, i64 %94)
  %96 = add i64 0, %86
  %97 = getelementptr i8, ptr %92, i64 %96
  %98 = sub i64 %89, %96
  %99 = call ptr @__memcpy_chk(ptr %97, ptr %85, i64 %88, i64 %98)
  %100 = add i64 %96, %88
  %101 = getelementptr i8, ptr %92, i64 %100
  store i8 0, ptr %101, align 1
  store ptr %92, ptr %5, align 8
  br label %167

102:                                              ; preds = %77
  %103 = load ptr, ptr %3, align 8
  %104 = icmp eq ptr %103, null
  %105 = select i1 %104, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %103
  %106 = call i64 @strlen(ptr %105)
  %107 = add i64 1, %106
  %108 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %109 = add i64 %107, %108
  %110 = add i64 %109, 8
  %111 = call ptr @GC_malloc(i64 %110)
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr i8, ptr %112, i64 0
  %114 = sub i64 %109, 0
  %115 = call ptr @__memcpy_chk(ptr %113, ptr %105, i64 %106, i64 %114)
  %116 = add i64 0, %106
  %117 = getelementptr i8, ptr %112, i64 %116
  %118 = sub i64 %109, %116
  %119 = call ptr @__memcpy_chk(ptr %117, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), i64 %108, i64 %118)
  %120 = add i64 %116, %108
  %121 = getelementptr i8, ptr %112, i64 %120
  store i8 0, ptr %121, align 1
  store ptr %112, ptr %1, align 8
  %122 = load i32, ptr %4, align 4
  %123 = icmp eq i32 %122, 2
  br i1 %123, label %124, label %147

124:                                              ; preds = %102
  %125 = load ptr, ptr %5, align 8
  %126 = load ptr, ptr %1, align 8
  %127 = icmp eq ptr %125, null
  %128 = select i1 %127, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %125
  %129 = icmp eq ptr %126, null
  %130 = select i1 %129, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %126
  %131 = call i64 @strlen(ptr %128)
  %132 = add i64 1, %131
  %133 = call i64 @strlen(ptr %130)
  %134 = add i64 %132, %133
  %135 = add i64 %134, 8
  %136 = call ptr @GC_malloc(i64 %135)
  %137 = getelementptr i8, ptr %136, i64 8
  %138 = getelementptr i8, ptr %137, i64 0
  %139 = sub i64 %134, 0
  %140 = call ptr @__memcpy_chk(ptr %138, ptr %128, i64 %131, i64 %139)
  %141 = add i64 0, %131
  %142 = getelementptr i8, ptr %137, i64 %141
  %143 = sub i64 %134, %141
  %144 = call ptr @__memcpy_chk(ptr %142, ptr %130, i64 %133, i64 %143)
  %145 = add i64 %141, %133
  %146 = getelementptr i8, ptr %137, i64 %145
  store i8 0, ptr %146, align 1
  store ptr %137, ptr %5, align 8
  br label %168

147:                                              ; preds = %102
  %148 = load ptr, ptr %5, align 8
  %149 = icmp eq ptr %148, null
  %150 = select i1 %149, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %148
  %151 = call i64 @strlen(ptr %150)
  %152 = add i64 1, %151
  %153 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8))
  %154 = add i64 %152, %153
  %155 = add i64 %154, 8
  %156 = call ptr @GC_malloc(i64 %155)
  %157 = getelementptr i8, ptr %156, i64 8
  %158 = getelementptr i8, ptr %157, i64 0
  %159 = sub i64 %154, 0
  %160 = call ptr @__memcpy_chk(ptr %158, ptr %150, i64 %151, i64 %159)
  %161 = add i64 0, %151
  %162 = getelementptr i8, ptr %157, i64 %161
  %163 = sub i64 %154, %161
  %164 = call ptr @__memcpy_chk(ptr %162, ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8), i64 %153, i64 %163)
  %165 = add i64 %161, %153
  %166 = getelementptr i8, ptr %157, i64 %165
  store i8 0, ptr %166, align 1
  store ptr %157, ptr %5, align 8
  br label %167

167:                                              ; preds = %79, %147
  br label %168

168:                                              ; preds = %167, %124
  %169 = load ptr, ptr %5, align 8
  %170 = icmp eq ptr %169, null
  %171 = select i1 %170, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %169
  %172 = call i64 @strlen(ptr %171)
  %173 = add i64 1, %172
  %174 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %175 = add i64 %173, %174
  %176 = add i64 %175, 8
  %177 = call ptr @GC_malloc(i64 %176)
  %178 = getelementptr i8, ptr %177, i64 8
  %179 = getelementptr i8, ptr %178, i64 0
  %180 = sub i64 %175, 0
  %181 = call ptr @__memcpy_chk(ptr %179, ptr %171, i64 %172, i64 %180)
  %182 = add i64 0, %172
  %183 = getelementptr i8, ptr %178, i64 %182
  %184 = sub i64 %175, %182
  %185 = call ptr @__memcpy_chk(ptr %183, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %174, i64 %184)
  %186 = add i64 %182, %174
  %187 = getelementptr i8, ptr %178, i64 %186
  store i8 0, ptr %187, align 1
  store ptr %178, ptr %5, align 8
  br label %188

188:                                              ; preds = %35, %168
  %189 = load i32, ptr %4, align 4
  %190 = add i32 %189, 1
  store i32 %190, ptr %4, align 4
  br label %7

191:                                              ; preds = %7
  %192 = load ptr, ptr %5, align 8
  store ptr %192, ptr %6, align 8
  %193 = load ptr, ptr %6, align 8
  ret ptr %193
}

; Function Attrs: mustprogress
define void @check({ ptr, ptr, ptr } %0, ptr %1, ptr %2) #1 {
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
  call void @__assert_fail(ptr %71, ptr getelementptr inbounds nuw (i8, ptr @f_16907858675607739117, i64 8), i32 188, ptr null)
  br label %67
}

; Function Attrs: mustprogress
define void @.f_main..afL192C11FH3922103366() #1 {
  call void @caseWithBreak(double 0.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL193C11FH3922103366() #1 {
  call void @caseWithBreak(double 1.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL194C11FH3922103366() #1 {
  call void @caseWithBreak(double 5.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL195C11FH3922103366() #1 {
  call void @fallThrough(double 0.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL196C11FH3922103366() #1 {
  call void @fallThrough(double 7.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL197C11FH3922103366() #1 {
  call void @usingInTwoCases(double 0.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL198C11FH3922103366() #1 {
  call void @usingInTwoCases(double 1.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL199C11FH3922103366() #1 {
  call void @bracedCases(double 0.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL200C11FH3922103366() #1 {
  call void @breakFromIf(double 0.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL201C11FH3922103366() #1 {
  call void @breakFromIf(double 1.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL204C11FH3922103366() #1 {
  %1 = alloca <{ i64, %Shape }>, align 16
  store i64 -1, ptr %1, align 8
  %2 = getelementptr i8, ptr %1, i64 8
  store %Shape zeroinitializer, ptr %2, align 8
  store %Shape zeroinitializer, ptr %2, align 8
  %3 = getelementptr %Shape, ptr %2, i32 0, i32 0
  store ptr @Shape..vtbl, ptr %3, align 8
  call void @Shape.constructor(ptr %2, ptr getelementptr inbounds nuw (i8, ptr @s_8684338226774033990, i64 8))
  call void @onProperty(ptr %2)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL205C11FH3922103366() #1 {
  %1 = alloca <{ i64, %Shape }>, align 16
  store i64 -1, ptr %1, align 8
  %2 = getelementptr i8, ptr %1, i64 8
  store %Shape zeroinitializer, ptr %2, align 8
  store %Shape zeroinitializer, ptr %2, align 8
  %3 = getelementptr %Shape, ptr %2, i32 0, i32 0
  store ptr @Shape..vtbl, ptr %3, align 8
  call void @Shape.constructor(ptr %2, ptr getelementptr inbounds nuw (i8, ptr @s_3196563492271544521, i64 8))
  call void @onProperty(ptr %2)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL192C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_10830661723522802543, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_2946054492843531381, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL193C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_6154944742766402726, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_8813813054881352361, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL194C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_17800509762665512158, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL195C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_11703912382116847995, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_12151427799237431201, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL196C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_1190972510352184004, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_16982411286042166782, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL197C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_17267362973927909224, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_3602924200895940314, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL198C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_12123086994723438001, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_8352128398236822016, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL199C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_17030157935117728868, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_8763251578505882867, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL200C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_5169259296458898883, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_9359698059718277970, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL201C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_2719078386004316387, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_15651424497846631000, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @inLoop, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_10920641262281999976, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_1035073461916539064, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL204C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_17980368551178335669, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_2624917850085765850, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @.f_main..afL205C11FH3922103366, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_8791758811777272261, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_8803624280936245040, i64 8))
  %1 = call ptr @stringsInCases(double 0.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %2 = ptrtoint ptr %1 to i64
  %3 = icmp ne i64 %2, 0
  %4 = zext i1 %3 to i32
  %5 = and i32 %4, 1
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %0
  %8 = call i32 @strcmp(ptr %1, ptr getelementptr inbounds nuw (i8, ptr @s_16843719094328038368, i64 8))
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %0
  %11 = icmp eq i64 %2, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16843719094328038368, i64 8) to i64)
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i1 [ %11, %10 ], [ %9, %7 ]
  br label %14

14:                                               ; preds = %12
  br i1 %13, label %15, label %77

15:                                               ; preds = %14
  %16 = call ptr @stringsInCases(double 0.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @s_14837330719131395891, i64 8))
  %17 = ptrtoint ptr %16 to i64
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i32
  %20 = and i32 %19, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %15
  %23 = call i32 @strcmp(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_4459931025994418872, i64 8))
  %24 = icmp eq i32 %23, 0
  br label %27

25:                                               ; preds = %15
  %26 = icmp eq i64 %17, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_4459931025994418872, i64 8) to i64)
  br label %27

27:                                               ; preds = %22, %25
  %28 = phi i1 [ %26, %25 ], [ %24, %22 ]
  br label %29

29:                                               ; preds = %27
  br i1 %28, label %30, label %79

30:                                               ; preds = %29
  %31 = call ptr @stringsInCases(double 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %32 = ptrtoint ptr %31 to i64
  %33 = icmp ne i64 %32, 0
  %34 = zext i1 %33 to i32
  %35 = and i32 %34, 1
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %40

37:                                               ; preds = %30
  %38 = call i32 @strcmp(ptr %31, ptr getelementptr inbounds nuw (i8, ptr @s_5873411151622665522, i64 8))
  %39 = icmp eq i32 %38, 0
  br label %42

40:                                               ; preds = %30
  %41 = icmp eq i64 %32, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5873411151622665522, i64 8) to i64)
  br label %42

42:                                               ; preds = %37, %40
  %43 = phi i1 [ %41, %40 ], [ %39, %37 ]
  br label %44

44:                                               ; preds = %42
  br i1 %43, label %45, label %81

45:                                               ; preds = %44
  %46 = call ptr @stringsInCases(double 9.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
  %47 = ptrtoint ptr %46 to i64
  %48 = icmp ne i64 %47, 0
  %49 = zext i1 %48 to i32
  %50 = and i32 %49, 1
  %51 = icmp ne i32 %50, 0
  br i1 %51, label %52, label %55

52:                                               ; preds = %45
  %53 = call i32 @strcmp(ptr %46, ptr getelementptr inbounds nuw (i8, ptr @s_2394902519263854347, i64 8))
  %54 = icmp eq i32 %53, 0
  br label %57

55:                                               ; preds = %45
  %56 = icmp eq i64 %47, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2394902519263854347, i64 8) to i64)
  br label %57

57:                                               ; preds = %52, %55
  %58 = phi i1 [ %56, %55 ], [ %54, %52 ]
  br label %59

59:                                               ; preds = %57
  br i1 %58, label %60, label %83

60:                                               ; preds = %59
  %61 = call ptr @stringsInLoop()
  %62 = ptrtoint ptr %61 to i64
  %63 = icmp ne i64 %62, 0
  %64 = zext i1 %63 to i32
  %65 = and i32 %64, 1
  %66 = icmp ne i32 %65, 0
  br i1 %66, label %67, label %70

67:                                               ; preds = %60
  %68 = call i32 @strcmp(ptr %61, ptr getelementptr inbounds nuw (i8, ptr @s_4313850459576425155, i64 8))
  %69 = icmp eq i32 %68, 0
  br label %72

70:                                               ; preds = %60
  %71 = icmp eq i64 %62, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_4313850459576425155, i64 8) to i64)
  br label %72

72:                                               ; preds = %67, %70
  %73 = phi i1 [ %71, %70 ], [ %69, %67 ]
  br label %74

74:                                               ; preds = %72
  br i1 %73, label %75, label %85

75:                                               ; preds = %74
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

76:                                               ; preds = %85, %83, %81, %79, %77
  unreachable

77:                                               ; preds = %14
  %78 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4371274227580367113, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16907858675607739117, i64 8), i32 207, ptr null)
  br label %76

79:                                               ; preds = %29
  %80 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11410955366148025694, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16907858675607739117, i64 8), i32 208, ptr null)
  br label %76

81:                                               ; preds = %44
  %82 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7857526575750099638, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16907858675607739117, i64 8), i32 209, ptr null)
  br label %76

83:                                               ; preds = %59
  %84 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14779713799370698046, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16907858675607739117, i64 8), i32 210, ptr null)
  br label %76

85:                                               ; preds = %74
  %86 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1996736543775029239, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_16907858675607739117, i64 8), i32 211, ptr null)
  br label %76
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

declare void @__cxa_end_catch()

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

