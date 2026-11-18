; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_6587055343100548874 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry \00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_12686345154644660721 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-finally: wrong iteration count\00", align 8
@f_270882832475389302 = internal constant [118 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_finally_break_continue.ts\00", align 8
@m_13628654338892272833 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-finally: wrong finally-run count\00", align 8
@m_13372472186300162813 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcontinue-in-finally: wrong finally-run count\00", align 8
@s_3263991042789642887 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner\00", align 8
@s_16003262670419624063 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFouter try \00", align 8
@m_12555396343315472575 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested-try-in-finally: wrong finally-run count\00", align 8
@m_5472161639373763668 = internal constant [66 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested-try-in-finally: inner catch did not run every time\00", align 8
@m_5207221870498065998 = internal constant [45 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-body: wrong iteration count\00", align 8
@m_7371798567406796300 = internal constant [67 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-body: finally must run on the break iteration too\00", align 8
@m_2424048783479429998 = internal constant [73 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcontinue-in-body: finally must run on the continue iteration too\00", align 8
@m_12931911836797756509 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested-chain: wrong iteration count\00", align 8
@m_16144272942233416155 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested-chain: inner finally skipped\00", align 8
@m_11943526830848849044 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested-chain: outer finally skipped\00", align 8
@s_14785268767687447306 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboom\00", align 8
@m_16592658429638420717 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-catch: wrong iteration count\00", align 8
@m_2613860599191423582 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-catch: catch did not run\00", align 8
@m_13602277166351569620 = internal constant [63 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak-in-catch: finally must run when catch breaks out\00", align 8
@m_5613525259335373314 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo-targets: wrong sum\00", align 8
@m_17418128109042334737 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo-targets: finally runs for i = 0..7\00", align 8
@m_9566113852364671064 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner-loop: wrong inner count\00", align 8
@m_14410997880283190912 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner-loop: finally must run exactly once\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_522455371173767105 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 1\00", align 8
@m_522452072638882472 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 2\00", align 8
@m_522453172150510683 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 3\00", align 8
@m_522458669708651738 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 4\00", align 8
@m_522459769220279949 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 5\00", align 8
@m_522456470685395316 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 6\00", align 8
@m_522457570197023527 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 7\00", align 8
@m_522445475569113206 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 8\00", align 8
@m_522446575080741417 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfailed. 9\00", align 8
@_ZTIPc = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare void @puts(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define i32 @test_break_in_finally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %5

5:                                                ; preds = %42, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 5
  br i1 %7, label %8, label %45

8:                                                ; preds = %5
  %9 = load i32, ptr %3, align 4
  %10 = add i32 %9, 1
  store i32 %10, ptr %3, align 4
  %11 = load i32, ptr %1, align 4
  %12 = call ptr @GC_malloc_atomic(i64 58)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %13, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %11)
  %15 = call ptr @llvm.stacksave.p0()
  %16 = icmp eq ptr %13, null
  %17 = select i1 %16, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %13
  %18 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_6587055343100548874, i64 8))
  %19 = add i64 1, %18
  %20 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %21 = add i64 %19, %20
  %22 = call i64 @strlen(ptr %17)
  %23 = add i64 %21, %22
  %24 = alloca i8, i64 %23, align 1
  %25 = getelementptr i8, ptr %24, i64 0
  %26 = sub i64 %23, 0
  %27 = call ptr @__memcpy_chk(ptr %25, ptr getelementptr inbounds nuw (i8, ptr @s_6587055343100548874, i64 8), i64 %18, i64 %26)
  %28 = add i64 0, %18
  %29 = getelementptr i8, ptr %24, i64 %28
  %30 = sub i64 %23, %28
  %31 = call ptr @__memcpy_chk(ptr %29, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %20, i64 %30)
  %32 = add i64 %28, %20
  %33 = getelementptr i8, ptr %24, i64 %32
  %34 = sub i64 %23, %32
  %35 = call ptr @__memcpy_chk(ptr %33, ptr %17, i64 %22, i64 %34)
  %36 = add i64 %32, %22
  %37 = getelementptr i8, ptr %24, i64 %36
  store i8 0, ptr %37, align 1
  call void @puts(ptr %24)
  call void @llvm.stackrestore.p0(ptr %15)
  %38 = load i32, ptr %2, align 4
  %39 = add i32 %38, 1
  store i32 %39, ptr %2, align 4
  %40 = load i32, ptr %1, align 4
  %41 = icmp eq i32 %40, 2
  br i1 %41, label %45, label %42

42:                                               ; preds = %8
  %43 = load i32, ptr %1, align 4
  %44 = add i32 %43, 1
  store i32 %44, ptr %1, align 4
  br label %5

45:                                               ; preds = %8, %5
  %46 = load i32, ptr %3, align 4
  %47 = icmp eq i32 %46, 3
  br i1 %47, label %48, label %58

48:                                               ; preds = %45
  %49 = load i32, ptr %2, align 4
  %50 = icmp eq i32 %49, 3
  br i1 %50, label %51, label %60

51:                                               ; preds = %48
  %52 = load i32, ptr %3, align 4
  %53 = mul i32 %52, 10
  %54 = load i32, ptr %2, align 4
  %55 = add i32 %53, %54
  store i32 %55, ptr %4, align 4
  %56 = load i32, ptr %4, align 4
  ret i32 %56

57:                                               ; preds = %60, %58
  unreachable

58:                                               ; preds = %45
  %59 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12686345154644660721, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 25, ptr null)
  br label %57

60:                                               ; preds = %48
  %61 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13628654338892272833, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 26, ptr null)
  br label %57
}

; Function Attrs: mustprogress
define i32 @test_continue_in_finally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %5

5:                                                ; preds = %44, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 5
  br i1 %7, label %8, label %47

8:                                                ; preds = %5
  %9 = load i32, ptr %1, align 4
  %10 = call ptr @GC_malloc_atomic(i64 58)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %11, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %9)
  %13 = call ptr @llvm.stacksave.p0()
  %14 = icmp eq ptr %11, null
  %15 = select i1 %14, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %11
  %16 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_6587055343100548874, i64 8))
  %17 = add i64 1, %16
  %18 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %19 = add i64 %17, %18
  %20 = call i64 @strlen(ptr %15)
  %21 = add i64 %19, %20
  %22 = alloca i8, i64 %21, align 1
  %23 = getelementptr i8, ptr %22, i64 0
  %24 = sub i64 %21, 0
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_6587055343100548874, i64 8), i64 %16, i64 %24)
  %26 = add i64 0, %16
  %27 = getelementptr i8, ptr %22, i64 %26
  %28 = sub i64 %21, %26
  %29 = call ptr @__memcpy_chk(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %18, i64 %28)
  %30 = add i64 %26, %18
  %31 = getelementptr i8, ptr %22, i64 %30
  %32 = sub i64 %21, %30
  %33 = call ptr @__memcpy_chk(ptr %31, ptr %15, i64 %20, i64 %32)
  %34 = add i64 %30, %20
  %35 = getelementptr i8, ptr %22, i64 %34
  store i8 0, ptr %35, align 1
  call void @puts(ptr %22)
  call void @llvm.stackrestore.p0(ptr %13)
  %36 = load i32, ptr %2, align 4
  %37 = add i32 %36, 1
  store i32 %37, ptr %2, align 4
  %38 = load i32, ptr %1, align 4
  %39 = icmp eq i32 %38, 2
  br i1 %39, label %44, label %40

40:                                               ; preds = %8
  %41 = load i32, ptr %3, align 4
  %42 = load i32, ptr %1, align 4
  %43 = add i32 %41, %42
  store i32 %43, ptr %3, align 4
  br label %44

44:                                               ; preds = %40, %8
  %45 = load i32, ptr %1, align 4
  %46 = add i32 %45, 1
  store i32 %46, ptr %1, align 4
  br label %5

47:                                               ; preds = %5
  %48 = load i32, ptr %2, align 4
  %49 = icmp eq i32 %48, 5
  br i1 %49, label %50, label %54

50:                                               ; preds = %47
  %51 = load i32, ptr %3, align 4
  store i32 %51, ptr %4, align 4
  %52 = load i32, ptr %4, align 4
  ret i32 %52

53:                                               ; preds = %54
  unreachable

54:                                               ; preds = %47
  %55 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13372472186300162813, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 47, ptr null)
  br label %53
}

; Function Attrs: mustprogress
define i32 @test_nested_try_in_finally() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  br label %6

6:                                                ; preds = %48, %0
  %7 = load i32, ptr %2, align 4
  %8 = icmp slt i32 %7, 3
  br i1 %8, label %9, label %55

9:                                                ; preds = %6
  %10 = load i32, ptr %2, align 4
  %11 = call ptr @GC_malloc_atomic(i64 58)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %12, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %10)
  %14 = call ptr @llvm.stacksave.p0()
  %15 = icmp eq ptr %12, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %12
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_16003262670419624063, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %20 = add i64 %18, %19
  %21 = call i64 @strlen(ptr %16)
  %22 = add i64 %20, %21
  %23 = alloca i8, i64 %22, align 1
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %22, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_16003262670419624063, i64 8), i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %22, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  %33 = sub i64 %22, %31
  %34 = call ptr @__memcpy_chk(ptr %32, ptr %16, i64 %21, i64 %33)
  %35 = add i64 %31, %21
  %36 = getelementptr i8, ptr %23, i64 %35
  store i8 0, ptr %36, align 1
  call void @puts(ptr %23)
  call void @llvm.stackrestore.p0(ptr %14)
  %37 = load i32, ptr %3, align 4
  %38 = add i32 %37, 1
  store i32 %38, ptr %3, align 4
  %39 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8), ptr %39, align 8
  invoke void @__cxa_throw(ptr %39, ptr @_ZTIPc, ptr null)
          to label %64 unwind label %40

40:                                               ; preds = %9
  %41 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %42 = extractvalue { ptr, i32 } %41, 1
  %43 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %44 = icmp eq i32 %42, %43
  br i1 %44, label %48, label %45

45:                                               ; preds = %40
  %46 = extractvalue { ptr, i32 } %41, 0
  %47 = call ptr @__cxa_begin_catch(ptr %46)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %45
  br label %64

48:                                               ; preds = %40
  %49 = extractvalue { ptr, i32 } %41, 0
  %50 = call ptr @__cxa_begin_catch(ptr %49)
  store ptr %50, ptr %1, align 8
  %51 = load i32, ptr %4, align 4
  %52 = add i32 %51, 1
  store i32 %52, ptr %4, align 4
  call void @__cxa_end_catch()
  %53 = load i32, ptr %2, align 4
  %54 = add i32 %53, 1
  store i32 %54, ptr %2, align 4
  br label %6

55:                                               ; preds = %6
  %56 = load i32, ptr %3, align 4
  %57 = icmp eq i32 %56, 3
  br i1 %57, label %58, label %65

58:                                               ; preds = %55
  %59 = load i32, ptr %4, align 4
  %60 = icmp eq i32 %59, 3
  br i1 %60, label %61, label %67

61:                                               ; preds = %58
  %62 = load i32, ptr %4, align 4
  store i32 %62, ptr %5, align 4
  %63 = load i32, ptr %5, align 4
  ret i32 %63

64:                                               ; preds = %.noexc, %9, %67, %65
  unreachable

65:                                               ; preds = %55
  %66 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12555396343315472575, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 71, ptr null)
  br label %64

67:                                               ; preds = %58
  %68 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5472161639373763668, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 72, ptr null)
  br label %64

endcatch.resume:                                  ; preds = %45
  %69 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %69
}

; Function Attrs: mustprogress
define i32 @test_break_in_try_body() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %5

5:                                                ; preds = %16, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 5
  br i1 %7, label %8, label %21

8:                                                ; preds = %5
  %9 = load i32, ptr %3, align 4
  %10 = add i32 %9, 1
  store i32 %10, ptr %3, align 4
  %11 = load i32, ptr %1, align 4
  %12 = icmp eq i32 %11, 2
  br i1 %12, label %13, label %16

13:                                               ; preds = %8
  %14 = load i32, ptr %2, align 4
  %15 = add i32 %14, 1
  store i32 %15, ptr %2, align 4
  br label %21

16:                                               ; preds = %8
  %17 = load i32, ptr %2, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %2, align 4
  %19 = load i32, ptr %1, align 4
  %20 = add i32 %19, 1
  store i32 %20, ptr %1, align 4
  br label %5

21:                                               ; preds = %13, %5
  %22 = load i32, ptr %3, align 4
  %23 = icmp eq i32 %22, 3
  br i1 %23, label %24, label %30

24:                                               ; preds = %21
  %25 = load i32, ptr %2, align 4
  %26 = icmp eq i32 %25, 3
  br i1 %26, label %27, label %32

27:                                               ; preds = %24
  store i32 1, ptr %4, align 4
  %28 = load i32, ptr %4, align 4
  ret i32 %28

29:                                               ; preds = %32, %30
  unreachable

30:                                               ; preds = %21
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5207221870498065998, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 92, ptr null)
  br label %29

32:                                               ; preds = %24
  %33 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7371798567406796300, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 93, ptr null)
  br label %29
}

; Function Attrs: mustprogress
define i32 @test_continue_in_try_body() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %5

5:                                                ; preds = %20, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 5
  br i1 %7, label %8, label %23

8:                                                ; preds = %5
  %9 = load i32, ptr %1, align 4
  %10 = icmp eq i32 %9, 2
  br i1 %10, label %17, label %11

11:                                               ; preds = %8
  %12 = load i32, ptr %3, align 4
  %13 = load i32, ptr %1, align 4
  %14 = add i32 %12, %13
  store i32 %14, ptr %3, align 4
  %15 = load i32, ptr %2, align 4
  %16 = add i32 %15, 1
  store i32 %16, ptr %2, align 4
  br label %20

17:                                               ; preds = %8
  %18 = load i32, ptr %2, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %2, align 4
  br label %20

20:                                               ; preds = %17, %11
  %21 = load i32, ptr %1, align 4
  %22 = add i32 %21, 1
  store i32 %22, ptr %1, align 4
  br label %5

23:                                               ; preds = %5
  %24 = load i32, ptr %2, align 4
  %25 = icmp eq i32 %24, 5
  br i1 %25, label %26, label %30

26:                                               ; preds = %23
  %27 = load i32, ptr %3, align 4
  store i32 %27, ptr %4, align 4
  %28 = load i32, ptr %4, align 4
  ret i32 %28

29:                                               ; preds = %30
  unreachable

30:                                               ; preds = %23
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2424048783479429998, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 112, ptr null)
  br label %29
}

; Function Attrs: mustprogress
define i32 @test_nested_finally_chain() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %6

6:                                                ; preds = %19, %0
  %7 = load i32, ptr %1, align 4
  %8 = icmp slt i32 %7, 5
  br i1 %8, label %9, label %26

9:                                                ; preds = %6
  %10 = load i32, ptr %2, align 4
  %11 = add i32 %10, 1
  store i32 %11, ptr %2, align 4
  %12 = load i32, ptr %1, align 4
  %13 = icmp eq i32 %12, 1
  br i1 %13, label %14, label %19

14:                                               ; preds = %9
  %15 = load i32, ptr %4, align 4
  %16 = add i32 %15, 1
  store i32 %16, ptr %4, align 4
  %17 = load i32, ptr %3, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %3, align 4
  br label %26

19:                                               ; preds = %9
  %20 = load i32, ptr %4, align 4
  %21 = add i32 %20, 1
  store i32 %21, ptr %4, align 4
  %22 = load i32, ptr %3, align 4
  %23 = add i32 %22, 1
  store i32 %23, ptr %3, align 4
  %24 = load i32, ptr %1, align 4
  %25 = add i32 %24, 1
  store i32 %25, ptr %1, align 4
  br label %6

26:                                               ; preds = %14, %6
  %27 = load i32, ptr %2, align 4
  %28 = icmp eq i32 %27, 2
  br i1 %28, label %29, label %38

29:                                               ; preds = %26
  %30 = load i32, ptr %4, align 4
  %31 = icmp eq i32 %30, 2
  br i1 %31, label %32, label %40

32:                                               ; preds = %29
  %33 = load i32, ptr %3, align 4
  %34 = icmp eq i32 %33, 2
  br i1 %34, label %35, label %42

35:                                               ; preds = %32
  store i32 1, ptr %5, align 4
  %36 = load i32, ptr %5, align 4
  ret i32 %36

37:                                               ; preds = %42, %40, %38
  unreachable

38:                                               ; preds = %26
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12931911836797756509, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 138, ptr null)
  br label %37

40:                                               ; preds = %29
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16144272942233416155, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 139, ptr null)
  br label %37

42:                                               ; preds = %32
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11943526830848849044, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 140, ptr null)
  br label %37
}

; Function Attrs: mustprogress
define i32 @test_break_in_catch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  br label %7

7:                                                ; preds = %38, %0
  %8 = load i32, ptr %2, align 4
  %9 = icmp slt i32 %8, 5
  br i1 %9, label %10, label %43

10:                                               ; preds = %7
  %11 = load i32, ptr %3, align 4
  %12 = add i32 %11, 1
  store i32 %12, ptr %3, align 4
  %13 = load i32, ptr %2, align 4
  %14 = icmp eq i32 %13, 2
  br i1 %14, label %15, label %38

15:                                               ; preds = %10
  %16 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8), ptr %16, align 8
  invoke void @__cxa_throw(ptr %16, ptr @_ZTIPc, ptr null)
          to label %54 unwind label %17

17:                                               ; preds = %15
  %18 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %19 = extractvalue { ptr, i32 } %18, 1
  %20 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %21 = icmp eq i32 %19, %20
  br i1 %21, label %25, label %22

22:                                               ; preds = %17
  %23 = extractvalue { ptr, i32 } %18, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  invoke void @__cxa_rethrow()
          to label %54 unwind label %endcatch.pad

25:                                               ; preds = %17
  %26 = extractvalue { ptr, i32 } %18, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  store ptr %27, ptr %1, align 8
  %28 = load i32, ptr %5, align 4
  %29 = add i32 %28, 1
  store i32 %29, ptr %5, align 4
  call void @__cxa_end_catch()
  %30 = load i32, ptr %4, align 4
  %31 = add i32 %30, 1
  store i32 %31, ptr %4, align 4
  br label %43

endcatch.pad:                                     ; preds = %22
  %32 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

33:                                               ; No predecessors!
  %34 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %33
  %35 = phi { ptr, i32 } [ %34, %33 ], [ %32, %endcatch.pad ]
  %36 = load i32, ptr %4, align 4
  %37 = add i32 %36, 1
  store i32 %37, ptr %4, align 4
  resume { ptr, i32 } %35

38:                                               ; preds = %10
  %39 = load i32, ptr %4, align 4
  %40 = add i32 %39, 1
  store i32 %40, ptr %4, align 4
  %41 = load i32, ptr %2, align 4
  %42 = add i32 %41, 1
  store i32 %42, ptr %2, align 4
  br label %7

43:                                               ; preds = %25, %7
  %44 = load i32, ptr %3, align 4
  %45 = icmp eq i32 %44, 3
  br i1 %45, label %46, label %55

46:                                               ; preds = %43
  %47 = load i32, ptr %5, align 4
  %48 = icmp eq i32 %47, 1
  br i1 %48, label %49, label %57

49:                                               ; preds = %46
  %50 = load i32, ptr %4, align 4
  %51 = icmp eq i32 %50, 3
  br i1 %51, label %52, label %59

52:                                               ; preds = %49
  store i32 1, ptr %6, align 4
  %53 = load i32, ptr %6, align 4
  ret i32 %53

54:                                               ; preds = %59, %57, %55, %22, %15
  unreachable

55:                                               ; preds = %43
  %56 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16592658429638420717, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 163, ptr null)
  br label %54

57:                                               ; preds = %46
  %58 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2613860599191423582, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 164, ptr null)
  br label %54

59:                                               ; preds = %49
  %60 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13602277166351569620, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 165, ptr null)
  br label %54
}

; Function Attrs: mustprogress
define i32 @test_continue_and_break_two_targets() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %5

5:                                                ; preds = %27, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 10
  br i1 %7, label %8, label %30

8:                                                ; preds = %5
  %9 = load i32, ptr %1, align 4
  %10 = srem i32 %9, 2
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %21, label %12

12:                                               ; preds = %8
  %13 = load i32, ptr %1, align 4
  %14 = icmp eq i32 %13, 7
  br i1 %14, label %24, label %15

15:                                               ; preds = %12
  %16 = load i32, ptr %3, align 4
  %17 = load i32, ptr %1, align 4
  %18 = add i32 %16, %17
  store i32 %18, ptr %3, align 4
  %19 = load i32, ptr %2, align 4
  %20 = add i32 %19, 1
  store i32 %20, ptr %2, align 4
  br label %27

21:                                               ; preds = %8
  %22 = load i32, ptr %2, align 4
  %23 = add i32 %22, 1
  store i32 %23, ptr %2, align 4
  br label %27

24:                                               ; preds = %12
  %25 = load i32, ptr %2, align 4
  %26 = add i32 %25, 1
  store i32 %26, ptr %2, align 4
  br label %30

27:                                               ; preds = %21, %15
  %28 = load i32, ptr %1, align 4
  %29 = add i32 %28, 1
  store i32 %29, ptr %1, align 4
  br label %5

30:                                               ; preds = %24, %5
  %31 = load i32, ptr %3, align 4
  %32 = icmp eq i32 %31, 9
  br i1 %32, label %33, label %39

33:                                               ; preds = %30
  %34 = load i32, ptr %2, align 4
  %35 = icmp eq i32 %34, 8
  br i1 %35, label %36, label %41

36:                                               ; preds = %33
  store i32 1, ptr %4, align 4
  %37 = load i32, ptr %4, align 4
  ret i32 %37

38:                                               ; preds = %41, %39
  unreachable

39:                                               ; preds = %30
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5613525259335373314, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 190, ptr null)
  br label %38

41:                                               ; preds = %33
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17418128109042334737, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 191, ptr null)
  br label %38
}

; Function Attrs: mustprogress
define i32 @test_inner_loop_break_untouched() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %5

5:                                                ; preds = %11, %0
  %6 = load i32, ptr %1, align 4
  %7 = icmp slt i32 %6, 10
  br i1 %7, label %8, label %16

8:                                                ; preds = %5
  %9 = load i32, ptr %1, align 4
  %10 = icmp eq i32 %9, 3
  br i1 %10, label %16, label %11

11:                                               ; preds = %8
  %12 = load i32, ptr %2, align 4
  %13 = add i32 %12, 1
  store i32 %13, ptr %2, align 4
  %14 = load i32, ptr %1, align 4
  %15 = add i32 %14, 1
  store i32 %15, ptr %1, align 4
  br label %5

16:                                               ; preds = %8, %5
  %17 = load i32, ptr %3, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %3, align 4
  %19 = load i32, ptr %2, align 4
  %20 = icmp eq i32 %19, 3
  br i1 %20, label %21, label %27

21:                                               ; preds = %16
  %22 = load i32, ptr %3, align 4
  %23 = icmp eq i32 %22, 1
  br i1 %23, label %24, label %29

24:                                               ; preds = %21
  store i32 1, ptr %4, align 4
  %25 = load i32, ptr %4, align 4
  ret i32 %25

26:                                               ; preds = %29, %27
  unreachable

27:                                               ; preds = %16
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9566113852364671064, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 212, ptr null)
  br label %26

29:                                               ; preds = %21
  %30 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14410997880283190912, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 213, ptr null)
  br label %26
}

; Function Attrs: mustprogress
define i32 @main() #3 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i32 @test_break_in_finally()
  %2 = icmp eq i32 %1, 33
  br i1 %2, label %3, label %29

3:                                                ; preds = %0
  %4 = call i32 @test_continue_in_finally()
  %5 = icmp eq i32 %4, 8
  br i1 %5, label %6, label %31

6:                                                ; preds = %3
  %7 = call i32 @test_nested_try_in_finally()
  %8 = icmp eq i32 %7, 3
  br i1 %8, label %9, label %33

9:                                                ; preds = %6
  %10 = call i32 @test_break_in_try_body()
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %35

12:                                               ; preds = %9
  %13 = call i32 @test_continue_in_try_body()
  %14 = icmp eq i32 %13, 8
  br i1 %14, label %15, label %37

15:                                               ; preds = %12
  %16 = call i32 @test_nested_finally_chain()
  %17 = icmp eq i32 %16, 1
  br i1 %17, label %18, label %39

18:                                               ; preds = %15
  %19 = call i32 @test_break_in_catch()
  %20 = icmp eq i32 %19, 1
  br i1 %20, label %21, label %41

21:                                               ; preds = %18
  %22 = call i32 @test_continue_and_break_two_targets()
  %23 = icmp eq i32 %22, 1
  br i1 %23, label %24, label %43

24:                                               ; preds = %21
  %25 = call i32 @test_inner_loop_break_untouched()
  %26 = icmp eq i32 %25, 1
  br i1 %26, label %27, label %45

27:                                               ; preds = %24
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

28:                                               ; preds = %45, %43, %41, %39, %37, %35, %33, %31, %29
  unreachable

29:                                               ; preds = %0
  %30 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522455371173767105, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 218, ptr null)
  br label %28

31:                                               ; preds = %3
  %32 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522452072638882472, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 219, ptr null)
  br label %28

33:                                               ; preds = %6
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522453172150510683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 220, ptr null)
  br label %28

35:                                               ; preds = %9
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522458669708651738, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 221, ptr null)
  br label %28

37:                                               ; preds = %12
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522459769220279949, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 222, ptr null)
  br label %28

39:                                               ; preds = %15
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522456470685395316, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 223, ptr null)
  br label %28

41:                                               ; preds = %18
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522457570197023527, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 224, ptr null)
  br label %28

43:                                               ; preds = %21
  %44 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522445475569113206, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 225, ptr null)
  br label %28

45:                                               ; preds = %24
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_522446575080741417, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_270882832475389302, i64 8), i32 226, ptr null)
  br label %28
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

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #4

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress "personality" }
attributes #3 = { mustprogress }
attributes #4 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

