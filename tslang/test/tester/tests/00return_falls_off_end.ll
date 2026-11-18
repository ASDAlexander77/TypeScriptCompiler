; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@a_2_3485755902904131247 = internal constant <2 x i32> <i32 1, i32 2>
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_2262031468633127938 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmany\00", align 8
@s_1875936269717626031 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFone\00", align 8
@s_6274872286310602505 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo\00", align 8
@s_12638182802509129152 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFe\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_14950064912268949816 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFaonex\00", align 8
@m_13728289955581695356 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFif without else: falls off\00", align 8
@f_12241132126650093672 = internal constant [112 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00return_falls_off_end.ts\00", align 8
@m_6882272181918989582 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFif without else: value\00", align 8
@m_14019708241179978786 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFif/else\00", align 8
@m_7769326792922808834 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch with default\00", align 8
@m_1683807704439455073 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch without default: falls off\00", align 8
@m_12445626418152020835 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch without default: value\00", align 8
@m_11986880034789336809 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwhile (true)\00", align 8
@m_1678380722958654368 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfor (;;) with break\00", align 8
@m_16525322597876987486 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/catch\00", align 8
@m_13683019591201459580 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreturn or throw\00", align 8
@m_16733090097380495956 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFloop: falls off\00", align 8
@m_12158829899270978022 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFloop: value\00", align 8
@m_14320216098002003288 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdeclared T | undefined: falls off\00", align 8
@m_12518217282484098206 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdeclared: value\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_8284080662672882251 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFalways-returning functions keep T\00", align 8
@_ZTIi = external constant ptr
@_ZTIPc = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare i32 @strcmp(ptr, ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define { ptr, i1 } @ifWithoutElse(i1 %0) #2 {
  %2 = alloca i1, align 1
  %3 = alloca { ptr, i1 }, align 8
  store i1 %0, ptr %2, align 1
  %4 = load i1, ptr %2, align 1
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i1 true }, ptr %3, align 8
  br label %7

6:                                                ; preds = %1
  store { ptr, i1 } zeroinitializer, ptr %3, align 8
  br label %7

7:                                                ; preds = %5, %6
  %8 = load { ptr, i1 }, ptr %3, align 8
  ret { ptr, i1 } %8
}

; Function Attrs: mustprogress
define ptr @ifElse(i1 %0) #2 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  store i1 %0, ptr %2, align 1
  %4 = load i1, ptr %2, align 1
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), ptr %3, align 8
  br label %7

6:                                                ; preds = %1
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8), ptr %3, align 8
  br label %7

7:                                                ; preds = %5, %6
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define ptr @switchWithDefault(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fptosi double %4 to i64
  %6 = trunc i64 %5 to i32
  %7 = icmp eq i32 %6, 1
  br i1 %7, label %8, label %9

8:                                                ; preds = %1
  store ptr getelementptr inbounds nuw (i8, ptr @s_1875936269717626031, i64 8), ptr %3, align 8
  br label %10

9:                                                ; preds = %1
  store ptr getelementptr inbounds nuw (i8, ptr @s_2262031468633127938, i64 8), ptr %3, align 8
  br label %10

10:                                               ; preds = %8, %9
  %11 = load ptr, ptr %3, align 8
  ret ptr %11
}

; Function Attrs: mustprogress
define { ptr, i1 } @switchWithoutDefault(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca { ptr, i1 }, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fptosi double %4 to i64
  %6 = trunc i64 %5 to i32
  %7 = icmp eq i32 %6, 1
  br i1 %7, label %8, label %9

8:                                                ; preds = %1
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_1875936269717626031, i64 8), i1 true }, ptr %3, align 8
  br label %15

9:                                                ; preds = %1
  %10 = fptosi double %4 to i64
  %11 = trunc i64 %10 to i32
  %12 = icmp eq i32 %11, 2
  br i1 %12, label %13, label %14

13:                                               ; preds = %9
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_6274872286310602505, i64 8), i1 true }, ptr %3, align 8
  br label %15

14:                                               ; preds = %9
  store { ptr, i1 } zeroinitializer, ptr %3, align 8
  br label %15

15:                                               ; preds = %8, %13, %14
  %16 = load { ptr, i1 }, ptr %3, align 8
  ret { ptr, i1 } %16
}

; Function Attrs: mustprogress
define double @infiniteLoop(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  br label %4

4:                                                ; preds = %10, %1
  br i1 true, label %5, label %13

5:                                                ; preds = %4
  %6 = load double, ptr %2, align 8
  %7 = fcmp ogt double %6, 0.000000e+00
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  %9 = load double, ptr %2, align 8
  store double %9, ptr %3, align 8
  br label %13

10:                                               ; preds = %5
  %11 = load double, ptr %2, align 8
  %12 = fadd double %11, 1.000000e+00
  store double %12, ptr %2, align 8
  br label %4

13:                                               ; preds = %8, %4
  %14 = load double, ptr %3, align 8
  ret double %14
}

; Function Attrs: mustprogress
define double @loopWithBreak(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  br label %4

4:                                                ; preds = %7, %1
  %5 = load double, ptr %2, align 8
  %6 = fcmp ogt double %5, 2.500000e+00
  br i1 %6, label %10, label %7

7:                                                ; preds = %4
  %8 = load double, ptr %2, align 8
  %9 = fadd double %8, 1.000000e+00
  store double %9, ptr %2, align 8
  br label %4

10:                                               ; preds = %4
  %11 = load double, ptr %2, align 8
  store double %11, ptr %3, align 8
  %12 = load double, ptr %3, align 8
  ret double %12
}

; Function Attrs: mustprogress
define i32 @tryCatch(i1 %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca i32, align 4
  store i1 %0, ptr %2, align 1
  %4 = load i1, ptr %2, align 1
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  store i32 1, ptr %3, align 4
  br label %12

6:                                                ; preds = %1
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIi, ptr null)
          to label %14 unwind label %8

8:                                                ; preds = %6
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  store i32 2, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %12

12:                                               ; preds = %5, %8
  %13 = load i32, ptr %3, align 4
  ret i32 %13

14:                                               ; preds = %6
  unreachable
}

; Function Attrs: mustprogress
define ptr @returnOrThrow(i1 %0) #2 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  store i1 %0, ptr %2, align 1
  %4 = load i1, ptr %2, align 1
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  ret ptr %6

7:                                                ; preds = %1
  %8 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638182802509129152, i64 8), ptr %8, align 8
  call void @__cxa_throw(ptr %8, ptr @_ZTIPc, ptr null)
  br label %9

9:                                                ; preds = %7
  unreachable
}

; Function Attrs: mustprogress
define { i32, i1 } @loopReturn(double %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca { i32, i1 }, align 8
  store double %0, ptr %4, align 8
  store i32 0, ptr %3, align 4
  %6 = call ptr @GC_malloc(i64 16)
  %7 = getelementptr i8, ptr %6, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %7, ptr @a_2_3485755902904131247, i64 8, i1 false)
  %8 = call ptr @GC_malloc(i64 32)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 0
  store ptr %7, ptr %10, align 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 1
  store i64 2, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 2
  store i64 2, ptr %12, align 8
  store ptr %9, ptr %2, align 8
  br label %13

13:                                               ; preds = %37, %1
  %14 = load i32, ptr %3, align 4
  %15 = load ptr, ptr %2, align 8
  %16 = icmp eq ptr %15, null
  %17 = select i1 %16, ptr @__ts_empty_array, ptr %15
  %18 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 1
  %19 = load i64, ptr %18, align 8
  %20 = sext i32 %14 to i64
  %21 = icmp slt i64 %20, %19
  br i1 %21, label %22, label %40

22:                                               ; preds = %13
  %23 = load i32, ptr %3, align 4
  %24 = load ptr, ptr %2, align 8
  %25 = icmp eq ptr %24, null
  %26 = select i1 %25, ptr @__ts_empty_array, ptr %24
  %27 = getelementptr { ptr, i64, i64 }, ptr %26, i32 0, i32 0
  %28 = load ptr, ptr %27, align 8
  %29 = getelementptr i32, ptr %28, i32 %23
  %30 = load i32, ptr %29, align 4
  %31 = load double, ptr %4, align 8
  %32 = sitofp i32 %30 to double
  %33 = fcmp oeq double %32, %31
  br i1 %33, label %34, label %37

34:                                               ; preds = %22
  %35 = insertvalue { i32, i1 } undef, i32 %30, 0
  %36 = insertvalue { i32, i1 } %35, i1 true, 1
  store { i32, i1 } %36, ptr %5, align 4
  br label %41

37:                                               ; preds = %22
  %38 = load i32, ptr %3, align 4
  %39 = add i32 %38, 1
  store i32 %39, ptr %3, align 4
  br label %13

40:                                               ; preds = %13
  store { i32, i1 } zeroinitializer, ptr %5, align 4
  br label %41

41:                                               ; preds = %34, %40
  %42 = load { i32, i1 }, ptr %5, align 4
  ret { i32, i1 } %42
}

; Function Attrs: mustprogress
define { double, i1 } @declared(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca { double, i1 }, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp ogt double %4, 0.000000e+00
  br i1 %5, label %6, label %10

6:                                                ; preds = %1
  %7 = load double, ptr %2, align 8
  %8 = insertvalue { double, i1 } undef, double %7, 0
  %9 = insertvalue { double, i1 } %8, i1 true, 1
  store { double, i1 } %9, ptr %3, align 8
  br label %11

10:                                               ; preds = %1
  store { double, i1 } zeroinitializer, ptr %3, align 8
  br label %11

11:                                               ; preds = %6, %10
  %12 = load { double, i1 }, ptr %3, align 8
  ret { double, i1 } %12
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call { ptr, i1 } @ifWithoutElse(i1 false)
  %2 = extractvalue { ptr, i1 } %1, 1
  %3 = zext i1 %2 to i32
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %185

5:                                                ; preds = %0
  %6 = call { ptr, i1 } @ifWithoutElse(i1 true)
  %7 = extractvalue { ptr, i1 } %6, 1
  br i1 %7, label %8, label %23

8:                                                ; preds = %5
  %9 = extractvalue { ptr, i1 } %6, 0
  %10 = ptrtoint ptr %9 to i64
  %11 = icmp ne i64 %10, 0
  %12 = zext i1 %11 to i32
  %13 = and i32 %12, 1
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %18

15:                                               ; preds = %8
  %16 = call i32 @strcmp(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %17 = icmp eq i32 %16, 0
  br label %20

18:                                               ; preds = %8
  %19 = icmp eq i64 %10, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8) to i64)
  br label %20

20:                                               ; preds = %15, %18
  %21 = phi i1 [ %19, %18 ], [ %17, %15 ]
  br label %22

22:                                               ; preds = %20
  br label %24

23:                                               ; preds = %5
  br label %24

24:                                               ; preds = %22, %23
  %25 = phi i1 [ false, %23 ], [ %21, %22 ]
  br label %26

26:                                               ; preds = %24
  br i1 %25, label %27, label %187

27:                                               ; preds = %26
  %28 = call ptr @ifElse(i1 false)
  %29 = ptrtoint ptr %28 to i64
  %30 = icmp ne i64 %29, 0
  %31 = zext i1 %30 to i32
  %32 = and i32 %31, 1
  %33 = icmp ne i32 %32, 0
  br i1 %33, label %34, label %37

34:                                               ; preds = %27
  %35 = call i32 @strcmp(ptr %28, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
  %36 = icmp eq i32 %35, 0
  br label %39

37:                                               ; preds = %27
  %38 = icmp eq i64 %29, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8) to i64)
  br label %39

39:                                               ; preds = %34, %37
  %40 = phi i1 [ %38, %37 ], [ %36, %34 ]
  br label %41

41:                                               ; preds = %39
  br i1 %40, label %42, label %189

42:                                               ; preds = %41
  %43 = call ptr @switchWithDefault(double 5.000000e+00)
  %44 = ptrtoint ptr %43 to i64
  %45 = icmp ne i64 %44, 0
  %46 = zext i1 %45 to i32
  %47 = and i32 %46, 1
  %48 = icmp ne i32 %47, 0
  br i1 %48, label %49, label %52

49:                                               ; preds = %42
  %50 = call i32 @strcmp(ptr %43, ptr getelementptr inbounds nuw (i8, ptr @s_2262031468633127938, i64 8))
  %51 = icmp eq i32 %50, 0
  br label %54

52:                                               ; preds = %42
  %53 = icmp eq i64 %44, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2262031468633127938, i64 8) to i64)
  br label %54

54:                                               ; preds = %49, %52
  %55 = phi i1 [ %53, %52 ], [ %51, %49 ]
  br label %56

56:                                               ; preds = %54
  br i1 %55, label %57, label %191

57:                                               ; preds = %56
  %58 = call { ptr, i1 } @switchWithoutDefault(double 3.000000e+00)
  %59 = extractvalue { ptr, i1 } %58, 1
  %60 = zext i1 %59 to i32
  %61 = icmp eq i32 %60, 0
  br i1 %61, label %62, label %193

62:                                               ; preds = %57
  %63 = call { ptr, i1 } @switchWithoutDefault(double 2.000000e+00)
  %64 = extractvalue { ptr, i1 } %63, 1
  br i1 %64, label %65, label %80

65:                                               ; preds = %62
  %66 = extractvalue { ptr, i1 } %63, 0
  %67 = ptrtoint ptr %66 to i64
  %68 = icmp ne i64 %67, 0
  %69 = zext i1 %68 to i32
  %70 = and i32 %69, 1
  %71 = icmp ne i32 %70, 0
  br i1 %71, label %72, label %75

72:                                               ; preds = %65
  %73 = call i32 @strcmp(ptr %66, ptr getelementptr inbounds nuw (i8, ptr @s_6274872286310602505, i64 8))
  %74 = icmp eq i32 %73, 0
  br label %77

75:                                               ; preds = %65
  %76 = icmp eq i64 %67, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6274872286310602505, i64 8) to i64)
  br label %77

77:                                               ; preds = %72, %75
  %78 = phi i1 [ %76, %75 ], [ %74, %72 ]
  br label %79

79:                                               ; preds = %77
  br label %81

80:                                               ; preds = %62
  br label %81

81:                                               ; preds = %79, %80
  %82 = phi i1 [ false, %80 ], [ %78, %79 ]
  br label %83

83:                                               ; preds = %81
  br i1 %82, label %84, label %195

84:                                               ; preds = %83
  %85 = call double @infiniteLoop(double 1.000000e+00)
  %86 = fcmp oeq double %85, 1.000000e+00
  br i1 %86, label %87, label %197

87:                                               ; preds = %84
  %88 = call double @loopWithBreak(double 0.000000e+00)
  %89 = fcmp oeq double %88, 3.000000e+00
  br i1 %89, label %90, label %199

90:                                               ; preds = %87
  %91 = call i32 @tryCatch(i1 false)
  %92 = icmp eq i32 %91, 2
  br i1 %92, label %93, label %201

93:                                               ; preds = %90
  %94 = call ptr @returnOrThrow(i1 true)
  %95 = ptrtoint ptr %94 to i64
  %96 = icmp ne i64 %95, 0
  %97 = zext i1 %96 to i32
  %98 = and i32 %97, 1
  %99 = icmp ne i32 %98, 0
  br i1 %99, label %100, label %103

100:                                              ; preds = %93
  %101 = call i32 @strcmp(ptr %94, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %102 = icmp eq i32 %101, 0
  br label %105

103:                                              ; preds = %93
  %104 = icmp eq i64 %95, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8) to i64)
  br label %105

105:                                              ; preds = %100, %103
  %106 = phi i1 [ %104, %103 ], [ %102, %100 ]
  br label %107

107:                                              ; preds = %105
  br i1 %106, label %108, label %203

108:                                              ; preds = %107
  %109 = call { i32, i1 } @loopReturn(double 5.000000e+00)
  %110 = extractvalue { i32, i1 } %109, 1
  %111 = zext i1 %110 to i32
  %112 = icmp eq i32 %111, 0
  br i1 %112, label %113, label %205

113:                                              ; preds = %108
  %114 = call { i32, i1 } @loopReturn(double 2.000000e+00)
  %115 = extractvalue { i32, i1 } %114, 1
  br i1 %115, label %116, label %119

116:                                              ; preds = %113
  %117 = extractvalue { i32, i1 } %114, 0
  %118 = icmp eq i32 %117, 2
  br label %120

119:                                              ; preds = %113
  br label %120

120:                                              ; preds = %116, %119
  %121 = phi i1 [ false, %119 ], [ %118, %116 ]
  br label %122

122:                                              ; preds = %120
  br i1 %121, label %123, label %207

123:                                              ; preds = %122
  %124 = call { double, i1 } @declared(double -1.000000e+00)
  %125 = extractvalue { double, i1 } %124, 1
  %126 = zext i1 %125 to i32
  %127 = icmp eq i32 %126, 0
  br i1 %127, label %128, label %209

128:                                              ; preds = %123
  %129 = call { double, i1 } @declared(double 2.000000e+00)
  %130 = extractvalue { double, i1 } %129, 1
  br i1 %130, label %131, label %134

131:                                              ; preds = %128
  %132 = extractvalue { double, i1 } %129, 0
  %133 = fcmp oeq double %132, 2.000000e+00
  br label %135

134:                                              ; preds = %128
  br label %135

135:                                              ; preds = %131, %134
  %136 = phi i1 [ false, %134 ], [ %133, %131 ]
  br label %137

137:                                              ; preds = %135
  br i1 %136, label %138, label %211

138:                                              ; preds = %137
  %139 = call ptr @ifElse(i1 true)
  %140 = call ptr @switchWithDefault(double 1.000000e+00)
  %141 = call ptr @returnOrThrow(i1 true)
  %142 = icmp eq ptr %139, null
  %143 = select i1 %142, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %139
  %144 = icmp eq ptr %140, null
  %145 = select i1 %144, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %140
  %146 = icmp eq ptr %141, null
  %147 = select i1 %146, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %141
  %148 = call i64 @strlen(ptr %143)
  %149 = add i64 1, %148
  %150 = call i64 @strlen(ptr %145)
  %151 = add i64 %149, %150
  %152 = call i64 @strlen(ptr %147)
  %153 = add i64 %151, %152
  %154 = add i64 %153, 8
  %155 = call ptr @GC_malloc(i64 %154)
  %156 = getelementptr i8, ptr %155, i64 8
  %157 = getelementptr i8, ptr %156, i64 0
  %158 = sub i64 %153, 0
  %159 = call ptr @__memcpy_chk(ptr %157, ptr %143, i64 %148, i64 %158)
  %160 = add i64 0, %148
  %161 = getelementptr i8, ptr %156, i64 %160
  %162 = sub i64 %153, %160
  %163 = call ptr @__memcpy_chk(ptr %161, ptr %145, i64 %150, i64 %162)
  %164 = add i64 %160, %150
  %165 = getelementptr i8, ptr %156, i64 %164
  %166 = sub i64 %153, %164
  %167 = call ptr @__memcpy_chk(ptr %165, ptr %147, i64 %152, i64 %166)
  %168 = add i64 %164, %152
  %169 = getelementptr i8, ptr %156, i64 %168
  store i8 0, ptr %169, align 1
  %170 = ptrtoint ptr %156 to i64
  %171 = icmp ne i64 %170, 0
  %172 = zext i1 %171 to i32
  %173 = and i32 %172, 1
  %174 = icmp ne i32 %173, 0
  br i1 %174, label %175, label %178

175:                                              ; preds = %138
  %176 = call i32 @strcmp(ptr %156, ptr getelementptr inbounds nuw (i8, ptr @s_14950064912268949816, i64 8))
  %177 = icmp eq i32 %176, 0
  br label %180

178:                                              ; preds = %138
  %179 = icmp eq i64 %170, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14950064912268949816, i64 8) to i64)
  br label %180

180:                                              ; preds = %175, %178
  %181 = phi i1 [ %179, %178 ], [ %177, %175 ]
  br label %182

182:                                              ; preds = %180
  br i1 %181, label %183, label %213

183:                                              ; preds = %182
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

184:                                              ; preds = %213, %211, %209, %207, %205, %203, %201, %199, %197, %195, %193, %191, %189, %187, %185
  unreachable

185:                                              ; preds = %0
  %186 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13728289955581695356, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 70, ptr null)
  br label %184

187:                                              ; preds = %26
  %188 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6882272181918989582, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 71, ptr null)
  br label %184

189:                                              ; preds = %41
  %190 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14019708241179978786, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 72, ptr null)
  br label %184

191:                                              ; preds = %56
  %192 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7769326792922808834, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 73, ptr null)
  br label %184

193:                                              ; preds = %57
  %194 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1683807704439455073, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 74, ptr null)
  br label %184

195:                                              ; preds = %83
  %196 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12445626418152020835, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 75, ptr null)
  br label %184

197:                                              ; preds = %84
  %198 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11986880034789336809, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 76, ptr null)
  br label %184

199:                                              ; preds = %87
  %200 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1678380722958654368, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 77, ptr null)
  br label %184

201:                                              ; preds = %90
  %202 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16525322597876987486, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 78, ptr null)
  br label %184

203:                                              ; preds = %107
  %204 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13683019591201459580, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 79, ptr null)
  br label %184

205:                                              ; preds = %108
  %206 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16733090097380495956, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 80, ptr null)
  br label %184

207:                                              ; preds = %122
  %208 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12158829899270978022, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 81, ptr null)
  br label %184

209:                                              ; preds = %123
  %210 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14320216098002003288, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 82, ptr null)
  br label %184

211:                                              ; preds = %137
  %212 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12518217282484098206, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 83, ptr null)
  br label %184

213:                                              ; preds = %182
  %214 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8284080662672882251, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12241132126650093672, i64 8), i32 87, ptr null)
  br label %184
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

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

