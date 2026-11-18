; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_13009877982135032956 = internal constant [67 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery iteration disposes, including the one that continues\00", align 8
@f_6502071642676186315 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00break_continue_scope_exit.ts\00", align 8
@m_5391744636001514824 = internal constant [71 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboth iterations reached dispose, including the one that breaks\00", align 8
@m_15753006951660162703 = internal constant [64 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo levels of nesting between the continue and the loop\00", align 8
@m_15007125011798170106 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe control: continue directly in the loop body\00", align 8
@m_3160601473079357917 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe intermediate scope's using disposes too\00", align 8
@m_13074288573532299537 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe labelled continue reaches its own loop\00", align 8
@m_7710794585267746139 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFand disposes both scopes it left, each round\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@disposed = private global i32 0
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

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
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #1 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  ret void
}

; Function Attrs: mustprogress
define void @continueFromIf() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %34, %0
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %37

7:                                                ; preds = %4
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  store ptr %9, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %11 = load i32, ptr %3, align 4
  %12 = icmp eq i32 %11, 1
  br i1 %12, label %13, label %20

13:                                               ; preds = %7
  %14 = load i1, ptr %1, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %13
  store i1 false, ptr %1, align 1
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Res, ptr %16, i32 0, i32 0
  invoke void @Res.dispose(ptr %16)
          to label %18 unwind label %27

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %13
  br label %34

20:                                               ; preds = %7
  %21 = load i1, ptr %1, align 1
  br i1 %21, label %22, label %26

22:                                               ; preds = %20
  store i1 false, ptr %1, align 1
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %Res, ptr %23, i32 0, i32 0
  invoke void @Res.dispose(ptr %23)
          to label %25 unwind label %27

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %25, %20
  br label %34

27:                                               ; preds = %15, %22
  %28 = landingpad { ptr, i32 }
          cleanup
  %29 = load i1, ptr %1, align 1
  br i1 %29, label %30, label %33

30:                                               ; preds = %27
  store i1 false, ptr %1, align 1
  %31 = load ptr, ptr %2, align 8
  %32 = getelementptr %Res, ptr %31, i32 0, i32 0
  call void @Res.dispose(ptr %31)
  br label %33

33:                                               ; preds = %30, %27
  resume { ptr, i32 } %28

34:                                               ; preds = %19, %26
  %35 = load i32, ptr %3, align 4
  %36 = add i32 %35, 1
  store i32 %36, ptr %3, align 4
  br label %4

37:                                               ; preds = %4
  ret void
}

; Function Attrs: mustprogress
define void @breakFromIf() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %26, %0
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %36

7:                                                ; preds = %4
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  store ptr %9, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %11 = load i32, ptr %3, align 4
  %12 = icmp eq i32 %11, 1
  br i1 %12, label %13, label %20

13:                                               ; preds = %7
  %14 = load i1, ptr %1, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %13
  store i1 false, ptr %1, align 1
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Res, ptr %16, i32 0, i32 0
  invoke void @Res.dispose(ptr %16)
          to label %18 unwind label %29

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %13
  br label %36

20:                                               ; preds = %7
  %21 = load i1, ptr %1, align 1
  br i1 %21, label %22, label %26

22:                                               ; preds = %20
  store i1 false, ptr %1, align 1
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %Res, ptr %23, i32 0, i32 0
  invoke void @Res.dispose(ptr %23)
          to label %25 unwind label %29

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %25, %20
  %27 = load i32, ptr %3, align 4
  %28 = add i32 %27, 1
  store i32 %28, ptr %3, align 4
  br label %4

29:                                               ; preds = %15, %22
  %30 = landingpad { ptr, i32 }
          cleanup
  %31 = load i1, ptr %1, align 1
  br i1 %31, label %32, label %35

32:                                               ; preds = %29
  store i1 false, ptr %1, align 1
  %33 = load ptr, ptr %2, align 8
  %34 = getelementptr %Res, ptr %33, i32 0, i32 0
  call void @Res.dispose(ptr %33)
  br label %35

35:                                               ; preds = %32, %29
  resume { ptr, i32 } %30

36:                                               ; preds = %19, %4
  ret void
}

; Function Attrs: mustprogress
define void @continueFromNestedBlock() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %34, %0
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %37

7:                                                ; preds = %4
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  store ptr %9, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %11 = load i32, ptr %3, align 4
  %12 = icmp eq i32 %11, 1
  br i1 %12, label %13, label %20

13:                                               ; preds = %7
  %14 = load i1, ptr %1, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %13
  store i1 false, ptr %1, align 1
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Res, ptr %16, i32 0, i32 0
  invoke void @Res.dispose(ptr %16)
          to label %18 unwind label %27

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %13
  br label %34

20:                                               ; preds = %7
  %21 = load i1, ptr %1, align 1
  br i1 %21, label %22, label %26

22:                                               ; preds = %20
  store i1 false, ptr %1, align 1
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %Res, ptr %23, i32 0, i32 0
  invoke void @Res.dispose(ptr %23)
          to label %25 unwind label %27

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %25, %20
  br label %34

27:                                               ; preds = %15, %22
  %28 = landingpad { ptr, i32 }
          cleanup
  %29 = load i1, ptr %1, align 1
  br i1 %29, label %30, label %33

30:                                               ; preds = %27
  store i1 false, ptr %1, align 1
  %31 = load ptr, ptr %2, align 8
  %32 = getelementptr %Res, ptr %31, i32 0, i32 0
  call void @Res.dispose(ptr %31)
  br label %33

33:                                               ; preds = %30, %27
  resume { ptr, i32 } %28

34:                                               ; preds = %19, %26
  %35 = load i32, ptr %3, align 4
  %36 = add i32 %35, 1
  store i32 %36, ptr %3, align 4
  br label %4

37:                                               ; preds = %4
  ret void
}

; Function Attrs: mustprogress
define void @continueDirect() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %16, %0
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %26

7:                                                ; preds = %4
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  store ptr %9, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %11 = load i1, ptr %1, align 1
  br i1 %11, label %12, label %16

12:                                               ; preds = %7
  store i1 false, ptr %1, align 1
  %13 = load ptr, ptr %2, align 8
  %14 = getelementptr %Res, ptr %13, i32 0, i32 0
  invoke void @Res.dispose(ptr %13)
          to label %15 unwind label %19

15:                                               ; preds = %12
  br label %16

16:                                               ; preds = %15, %7
  %17 = load i32, ptr %3, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %3, align 4
  br label %4

19:                                               ; preds = %12
  %20 = landingpad { ptr, i32 }
          cleanup
  %21 = load i1, ptr %1, align 1
  br i1 %21, label %22, label %25

22:                                               ; preds = %19
  store i1 false, ptr %1, align 1
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %Res, ptr %23, i32 0, i32 0
  call void @Res.dispose(ptr %23)
  br label %25

25:                                               ; preds = %22, %19
  resume { ptr, i32 } %20

26:                                               ; preds = %4
  ret void
}

; Function Attrs: mustprogress
define void @bothScopes() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  br label %6

6:                                                ; preds = %68, %0
  %7 = load i32, ptr %5, align 4
  %8 = icmp slt i32 %7, 2
  br i1 %8, label %9, label %71

9:                                                ; preds = %6
  store i1 false, ptr %3, align 1
  %10 = call ptr @GC_malloc(i64 16)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Res, ptr %11, i32 0, i32 0
  store ptr @Res..vtbl, ptr %12, align 8
  store ptr %11, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %13 = load i32, ptr %5, align 4
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %15, label %52

15:                                               ; preds = %9
  store i1 false, ptr %1, align 1
  %16 = call ptr @GC_malloc(i64 16)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr %Res, ptr %17, i32 0, i32 0
  store ptr @Res..vtbl, ptr %18, align 8
  store ptr %17, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %24

20:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %Res, ptr %21, i32 0, i32 0
  invoke void @Res.dispose(ptr %21)
          to label %23 unwind label %31

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
          to label %29 unwind label %31

29:                                               ; preds = %26
  br label %30

30:                                               ; preds = %29, %24
  br label %68

31:                                               ; preds = %26, %20
  %32 = landingpad { ptr, i32 }
          catch ptr null
  %33 = extractvalue { ptr, i32 } %32, 0
  %34 = call ptr @__cxa_begin_catch(ptr %33)
  %35 = load i1, ptr %1, align 1
  br i1 %35, label %36, label %40

36:                                               ; preds = %31
  store i1 false, ptr %1, align 1
  %37 = load ptr, ptr %2, align 8
  %38 = getelementptr %Res, ptr %37, i32 0, i32 0
  invoke void @Res.dispose(ptr %37)
          to label %39 unwind label %endcatch.pad2

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %39, %31
  invoke void @__cxa_rethrow()
          to label %72 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %36
  %41 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

42:                                               ; No predecessors!
  %43 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %42
  %44 = phi { ptr, i32 } [ %43, %42 ], [ %41, %endcatch.pad2 ]
  %45 = extractvalue { ptr, i32 } %44, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  %47 = load i1, ptr %1, align 1
  br i1 %47, label %48, label %51

48:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %49 = load ptr, ptr %2, align 8
  %50 = getelementptr %Res, ptr %49, i32 0, i32 0
  invoke void @Res.dispose(ptr %49)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %48
  br label %51

51:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %72 unwind label %endcatch.pad

52:                                               ; preds = %9
  %53 = load i1, ptr %3, align 1
  br i1 %53, label %54, label %58

54:                                               ; preds = %52
  store i1 false, ptr %3, align 1
  %55 = load ptr, ptr %4, align 8
  %56 = getelementptr %Res, ptr %55, i32 0, i32 0
  invoke void @Res.dispose(ptr %55)
          to label %57 unwind label %60

57:                                               ; preds = %54
  br label %58

58:                                               ; preds = %57, %52
  br label %68

endcatch.pad:                                     ; preds = %40, %51
  %59 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

60:                                               ; preds = %54
  %61 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %60
  %62 = phi { ptr, i32 } [ %61, %60 ], [ %59, %endcatch.pad ]
  %63 = load i1, ptr %3, align 1
  br i1 %63, label %64, label %67

64:                                               ; preds = %.caught
  store i1 false, ptr %3, align 1
  %65 = load ptr, ptr %4, align 8
  %66 = getelementptr %Res, ptr %65, i32 0, i32 0
  call void @Res.dispose(ptr %65)
  br label %67

67:                                               ; preds = %64, %.caught
  resume { ptr, i32 } %62

68:                                               ; preds = %30, %58
  %69 = load i32, ptr %5, align 4
  %70 = add i32 %69, 1
  store i32 %70, ptr %5, align 4
  br label %6

71:                                               ; preds = %6
  ret void

72:                                               ; preds = %40, %51
  unreachable

endcatch.resume:                                  ; preds = %48
  %73 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %73
}

; Function Attrs: mustprogress
define i32 @labelledContinue() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 0, ptr %6, align 4
  br label %8

8:                                                ; preds = %35, %63, %0
  %9 = load i32, ptr %6, align 4
  %10 = icmp slt i32 %9, 2
  br i1 %10, label %11, label %73

11:                                               ; preds = %8
  store i1 false, ptr %4, align 1
  %12 = load i32, ptr %6, align 4
  %13 = add i32 %12, 1
  store i32 %13, ptr %6, align 4
  %14 = call ptr @GC_malloc(i64 16)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %Res, ptr %15, i32 0, i32 0
  store ptr @Res..vtbl, ptr %16, align 8
  store ptr %15, ptr %5, align 8
  store i1 true, ptr %4, align 1
  store i32 2, ptr %3, align 4
  %17 = load i32, ptr %3, align 4
  %18 = sub i32 %17, 1
  store i32 %18, ptr %3, align 4
  %19 = icmp sgt i32 %17, 0
  br i1 %19, label %20, label %57

20:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %21 = call ptr @GC_malloc(i64 16)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr %Res, ptr %22, i32 0, i32 0
  store ptr @Res..vtbl, ptr %23, align 8
  store ptr %22, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %24 = load i1, ptr %1, align 1
  br i1 %24, label %25, label %29

25:                                               ; preds = %20
  store i1 false, ptr %1, align 1
  %26 = load ptr, ptr %2, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %28 unwind label %36

28:                                               ; preds = %25
  br label %29

29:                                               ; preds = %28, %20
  %30 = load i1, ptr %4, align 1
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  store i1 false, ptr %4, align 1
  %32 = load ptr, ptr %5, align 8
  %33 = getelementptr %Res, ptr %32, i32 0, i32 0
  invoke void @Res.dispose(ptr %32)
          to label %34 unwind label %36

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %34, %29
  br label %8

36:                                               ; preds = %31, %25
  %37 = landingpad { ptr, i32 }
          catch ptr null
  %38 = extractvalue { ptr, i32 } %37, 0
  %39 = call ptr @__cxa_begin_catch(ptr %38)
  %40 = load i1, ptr %1, align 1
  br i1 %40, label %41, label %45

41:                                               ; preds = %36
  store i1 false, ptr %1, align 1
  %42 = load ptr, ptr %2, align 8
  %43 = getelementptr %Res, ptr %42, i32 0, i32 0
  invoke void @Res.dispose(ptr %42)
          to label %44 unwind label %endcatch.pad2

44:                                               ; preds = %41
  br label %45

45:                                               ; preds = %44, %36
  invoke void @__cxa_rethrow()
          to label %76 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %41
  %46 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

47:                                               ; No predecessors!
  %48 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %47
  %49 = phi { ptr, i32 } [ %48, %47 ], [ %46, %endcatch.pad2 ]
  %50 = extractvalue { ptr, i32 } %49, 0
  %51 = call ptr @__cxa_begin_catch(ptr %50)
  %52 = load i1, ptr %1, align 1
  br i1 %52, label %53, label %56

53:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %54 = load ptr, ptr %2, align 8
  %55 = getelementptr %Res, ptr %54, i32 0, i32 0
  invoke void @Res.dispose(ptr %54)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %53
  br label %56

56:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %76 unwind label %endcatch.pad

57:                                               ; preds = %11
  %58 = load i1, ptr %4, align 1
  br i1 %58, label %59, label %63

59:                                               ; preds = %57
  store i1 false, ptr %4, align 1
  %60 = load ptr, ptr %5, align 8
  %61 = getelementptr %Res, ptr %60, i32 0, i32 0
  invoke void @Res.dispose(ptr %60)
          to label %62 unwind label %65

62:                                               ; preds = %59
  br label %63

63:                                               ; preds = %62, %57
  br label %8

endcatch.pad:                                     ; preds = %45, %56
  %64 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

65:                                               ; preds = %59
  %66 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %65
  %67 = phi { ptr, i32 } [ %66, %65 ], [ %64, %endcatch.pad ]
  %68 = load i1, ptr %4, align 1
  br i1 %68, label %69, label %72

69:                                               ; preds = %.caught
  store i1 false, ptr %4, align 1
  %70 = load ptr, ptr %5, align 8
  %71 = getelementptr %Res, ptr %70, i32 0, i32 0
  call void @Res.dispose(ptr %70)
  br label %72

72:                                               ; preds = %69, %.caught
  resume { ptr, i32 } %67

73:                                               ; preds = %8
  %74 = load i32, ptr %6, align 4
  store i32 %74, ptr %7, align 4
  %75 = load i32, ptr %7, align 4
  ret i32 %75

76:                                               ; preds = %45, %56
  unreachable

endcatch.resume:                                  ; preds = %53
  %77 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %77
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  store i32 0, ptr @disposed, align 4
  call void @continueFromIf()
  %1 = load i32, ptr @disposed, align 4
  %2 = icmp eq i32 %1, 3
  br i1 %2, label %3, label %23

3:                                                ; preds = %0
  store i32 0, ptr @disposed, align 4
  call void @breakFromIf()
  %4 = load i32, ptr @disposed, align 4
  %5 = icmp eq i32 %4, 2
  br i1 %5, label %6, label %25

6:                                                ; preds = %3
  store i32 0, ptr @disposed, align 4
  call void @continueFromNestedBlock()
  %7 = load i32, ptr @disposed, align 4
  %8 = icmp eq i32 %7, 3
  br i1 %8, label %9, label %27

9:                                                ; preds = %6
  store i32 0, ptr @disposed, align 4
  call void @continueDirect()
  %10 = load i32, ptr @disposed, align 4
  %11 = icmp eq i32 %10, 3
  br i1 %11, label %12, label %29

12:                                               ; preds = %9
  store i32 0, ptr @disposed, align 4
  call void @bothScopes()
  %13 = load i32, ptr @disposed, align 4
  %14 = icmp eq i32 %13, 3
  br i1 %14, label %15, label %31

15:                                               ; preds = %12
  store i32 0, ptr @disposed, align 4
  %16 = call i32 @labelledContinue()
  %17 = icmp eq i32 %16, 2
  br i1 %17, label %18, label %33

18:                                               ; preds = %15
  %19 = load i32, ptr @disposed, align 4
  %20 = icmp eq i32 %19, 4
  br i1 %20, label %21, label %35

21:                                               ; preds = %18
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

22:                                               ; preds = %35, %33, %31, %29, %27, %25, %23
  unreachable

23:                                               ; preds = %0
  %24 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13009877982135032956, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 101, ptr null)
  br label %22

25:                                               ; preds = %3
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5391744636001514824, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 105, ptr null)
  br label %22

27:                                               ; preds = %6
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15753006951660162703, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 109, ptr null)
  br label %22

29:                                               ; preds = %9
  %30 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15007125011798170106, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 113, ptr null)
  br label %22

31:                                               ; preds = %12
  %32 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3160601473079357917, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 117, ptr null)
  br label %22

33:                                               ; preds = %15
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13074288573532299537, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 120, ptr null)
  br label %22

35:                                               ; preds = %18
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7710794585267746139, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6502071642676186315, i64 8), i32 121, ptr null)
  br label %22
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

