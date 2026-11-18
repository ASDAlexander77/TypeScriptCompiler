; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_10598823285094962729 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught inner\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_2884202050812095309 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin catch\00", align 8
@s_666023999292716942 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry body\00", align 8
@s_7849594394962741973 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin finally\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_6393293660524519084 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnested block must throw\00", align 8
@f_11237846694651913704 = internal constant [102 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\04disposable.ts\00", align 8
@m_18063489156658645917 = internal constant [61 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFusing in a nested block must dispose while unwinding\00", align 8
@m_17214008398373060254 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFloop body must throw\00", align 8
@m_4677460654917946411 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFusing in a loop body must dispose while unwinding\00", align 8
@m_5126667335064581011 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdeep block must throw\00", align 8
@m_2962093585528290310 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFusing two scopes deep must dispose while unwinding\00", align 8
@m_7270621579697366617 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrowing path must throw\00", align 8
@m_17305234372915070146 = internal constant [79 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using sharing its function with a return must still dispose on throw\00", align 8
@m_11832716718392578126 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe ordinary return path must dispose exactly once\00", align 8
@m_2099669539852787566 = internal constant [68 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using inside a hand-written try must dispose exactly once\00", align 8
@m_8177213613051484080 = internal constant [64 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using inside a catch clause must dispose exactly once\00", align 8
@m_8933989265959140716 = internal constant [66 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using inside a finally clause must dispose exactly once\00", align 8
@disposed = private global i32 0
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

declare void @puts(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

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
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  ret void
}

; Function Attrs: mustprogress
define void @inNestedBlock(i1 %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca <{ i64, %Res }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Res zeroinitializer, ptr %6, align 8
  store i1 %0, ptr %4, align 1
  %7 = load i1, ptr %4, align 1
  br i1 %7, label %8, label %18

8:                                                ; preds = %1
  store i1 false, ptr %2, align 1
  store %Res zeroinitializer, ptr %6, align 8
  %9 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %9, align 8
  store ptr %6, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %10 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %10, align 4
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIi, ptr null)
          to label %19 unwind label %11

11:                                               ; preds = %8
  %12 = landingpad { ptr, i32 }
          cleanup
  %13 = load i1, ptr %2, align 1
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  store i1 false, ptr %2, align 1
  %15 = load ptr, ptr %3, align 8
  %16 = getelementptr %Res, ptr %15, i32 0, i32 0
  call void @Res.dispose(ptr %15)
  br label %17

17:                                               ; preds = %14, %11
  resume { ptr, i32 } %12

18:                                               ; preds = %1
  ret void

19:                                               ; preds = %8
  unreachable
}

; Function Attrs: mustprogress
define void @inLoopBody() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca <{ i64, %Res }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Res zeroinitializer, ptr %5, align 8
  store i32 0, ptr %3, align 4
  %6 = load i32, ptr %3, align 4
  %7 = icmp slt i32 %6, 1
  br i1 %7, label %8, label %18

8:                                                ; preds = %0
  store i1 false, ptr %1, align 1
  store %Res zeroinitializer, ptr %5, align 8
  %9 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %9, align 8
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %10 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %10, align 4
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIi, ptr null)
          to label %19 unwind label %11

11:                                               ; preds = %8
  %12 = landingpad { ptr, i32 }
          cleanup
  %13 = load i1, ptr %1, align 1
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr %Res, ptr %15, i32 0, i32 0
  call void @Res.dispose(ptr %15)
  br label %17

17:                                               ; preds = %14, %11
  resume { ptr, i32 } %12

18:                                               ; preds = %0
  ret void

19:                                               ; preds = %8
  unreachable
}

; Function Attrs: mustprogress
define void @inDeepBlock() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca <{ i64, %Res }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Res zeroinitializer, ptr %5, align 8
  store i32 0, ptr %3, align 4
  br label %6

6:                                                ; preds = %22, %0
  %7 = load i32, ptr %3, align 4
  %8 = icmp slt i32 %7, 2
  br i1 %8, label %9, label %25

9:                                                ; preds = %6
  %10 = load i32, ptr %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %22

12:                                               ; preds = %9
  store i1 false, ptr %1, align 1
  store %Res zeroinitializer, ptr %5, align 8
  %13 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %13, align 8
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %14 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %14, align 4
  invoke void @__cxa_throw(ptr %14, ptr @_ZTIi, ptr null)
          to label %26 unwind label %15

15:                                               ; preds = %12
  %16 = landingpad { ptr, i32 }
          cleanup
  %17 = load i1, ptr %1, align 1
  br i1 %17, label %18, label %21

18:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %19 = load ptr, ptr %2, align 8
  %20 = getelementptr %Res, ptr %19, i32 0, i32 0
  call void @Res.dispose(ptr %19)
  br label %21

21:                                               ; preds = %18, %15
  resume { ptr, i32 } %16

22:                                               ; preds = %9
  %23 = load i32, ptr %3, align 4
  %24 = add i32 %23, 1
  store i32 %24, ptr %3, align 4
  br label %6

25:                                               ; preds = %6
  ret void

26:                                               ; preds = %12
  unreachable
}

; Function Attrs: mustprogress
define void @throwsOrReturns(i1 %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store i1 %0, ptr %4, align 1
  store i1 false, ptr %2, align 1
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  store ptr %6, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %8 = load i1, ptr %4, align 1
  br i1 %8, label %9, label %11

9:                                                ; preds = %1
  %10 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %10, align 4
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIi, ptr null)
          to label %25 unwind label %18

11:                                               ; preds = %1
  %12 = load i1, ptr %2, align 1
  br i1 %12, label %13, label %17

13:                                               ; preds = %11
  store i1 false, ptr %2, align 1
  %14 = load ptr, ptr %3, align 8
  %15 = getelementptr %Res, ptr %14, i32 0, i32 0
  invoke void @Res.dispose(ptr %14)
          to label %16 unwind label %18

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %16, %11
  ret void

18:                                               ; preds = %9, %13
  %19 = landingpad { ptr, i32 }
          cleanup
  %20 = load i1, ptr %2, align 1
  br i1 %20, label %21, label %24

21:                                               ; preds = %18
  store i1 false, ptr %2, align 1
  %22 = load ptr, ptr %3, align 8
  %23 = getelementptr %Res, ptr %22, i32 0, i32 0
  call void @Res.dispose(ptr %22)
  br label %24

24:                                               ; preds = %21, %18
  resume { ptr, i32 } %19

25:                                               ; preds = %9
  unreachable
}

; Function Attrs: mustprogress
define void @insideHandWrittenTry() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 16)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %6, align 8
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIi, ptr null)
          to label %43 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = load i1, ptr %1, align 1
  br i1 %12, label %13, label %17

13:                                               ; preds = %8
  store i1 false, ptr %1, align 1
  %14 = load ptr, ptr %2, align 8
  %15 = getelementptr %Res, ptr %14, i32 0, i32 0
  invoke void @Res.dispose(ptr %14)
          to label %16 unwind label %endcatch.pad2

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %16, %8
  invoke void @__cxa_rethrow()
          to label %43 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %13
  %18 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

19:                                               ; No predecessors!
  %20 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %19
  %21 = phi { ptr, i32 } [ %20, %19 ], [ %18, %endcatch.pad2 ]
  %22 = extractvalue { ptr, i32 } %21, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = load i1, ptr %1, align 1
  br i1 %24, label %25, label %28

25:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %26 = load ptr, ptr %2, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %25
  br label %28

28:                                               ; preds = %.noexc3, %.caught1
  invoke void @__cxa_rethrow()
          to label %43 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %28, %17
  %29 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

30:                                               ; No predecessors!
  %31 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %30
  %32 = phi { ptr, i32 } [ %31, %30 ], [ %29, %endcatch.pad ]
  %33 = extractvalue { ptr, i32 } %32, 1
  %34 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %35 = icmp eq i32 %33, %34
  br i1 %35, label %39, label %36

36:                                               ; preds = %.caught
  %37 = extractvalue { ptr, i32 } %32, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  invoke void @__cxa_rethrow()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %36
  br label %43

39:                                               ; preds = %.caught
  %40 = extractvalue { ptr, i32 } %32, 0
  %41 = call ptr @__cxa_begin_catch(ptr %40)
  %42 = load i32, ptr %41, align 4
  store i32 %42, ptr %3, align 4
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10598823285094962729, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %39
  call void @__cxa_end_catch()
  ret void

43:                                               ; preds = %.noexc4, %17, %28, %0
  unreachable

endcatch.resume:                                  ; preds = %36, %25, %39
  %44 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %44
}

; Function Attrs: mustprogress
define void @insideCatchClause() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca <{ i64, %Res }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Res zeroinitializer, ptr %5, align 8
  store i1 false, ptr %1, align 1
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %25 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = extractvalue { ptr, i32 } %8, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %12
  br label %25

15:                                               ; preds = %7
  %16 = extractvalue { ptr, i32 } %8, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = load i32, ptr %17, align 4
  store i32 %18, ptr %3, align 4
  store %Res zeroinitializer, ptr %5, align 8
  %19 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %19, align 8
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_2884202050812095309, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %15
  %20 = load i1, ptr %1, align 1
  br i1 %20, label %21, label %24

21:                                               ; preds = %.noexc
  store i1 false, ptr %1, align 1
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr %Res, ptr %22, i32 0, i32 0
  invoke void @Res.dispose(ptr %22)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %21
  br label %24

24:                                               ; preds = %.noexc1, %.noexc
  call void @__cxa_end_catch()
  ret void

25:                                               ; preds = %.noexc2, %0
  unreachable

endcatch.resume:                                  ; preds = %12, %21, %15
  %26 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %26
}

; Function Attrs: mustprogress
define void @insideFinallyClause() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, %Res }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Res zeroinitializer, ptr %4, align 8
  store i1 false, ptr %1, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_666023999292716942, i64 8))
  store %Res zeroinitializer, ptr %4, align 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 0
  store ptr @Res..vtbl, ptr %5, align 8
  store ptr %4, ptr %2, align 8
  store i1 true, ptr %1, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_7849594394962741973, i64 8))
  %6 = load i1, ptr %1, align 1
  br i1 %6, label %7, label %10

7:                                                ; preds = %0
  store i1 false, ptr %1, align 1
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr %Res, ptr %8, i32 0, i32 0
  call void @Res.dispose(ptr %8)
  br label %10

10:                                               ; preds = %7, %0
  ret void
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
define void @.f_main..afL100C24FH3461771208() #2 {
  call void @inNestedBlock(i1 true)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL104C24FH3461771208() #2 {
  call void @inLoopBody()
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL108C24FH3461771208() #2 {
  call void @inDeepBlock()
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL112C24FH3461771208() #2 {
  call void @throwsOrReturns(i1 true)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  store i32 0, ptr @disposed, align 4
  %1 = call i1 @expectThrow({ ptr, ptr, ptr } { ptr @.f_main..afL100C24FH3461771208, ptr null, ptr null })
  br i1 %1, label %2, label %34

2:                                                ; preds = %0
  %3 = load i32, ptr @disposed, align 4
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %5, label %36

5:                                                ; preds = %2
  store i32 0, ptr @disposed, align 4
  %6 = call i1 @expectThrow({ ptr, ptr, ptr } { ptr @.f_main..afL104C24FH3461771208, ptr null, ptr null })
  br i1 %6, label %7, label %38

7:                                                ; preds = %5
  %8 = load i32, ptr @disposed, align 4
  %9 = icmp eq i32 %8, 1
  br i1 %9, label %10, label %40

10:                                               ; preds = %7
  store i32 0, ptr @disposed, align 4
  %11 = call i1 @expectThrow({ ptr, ptr, ptr } { ptr @.f_main..afL108C24FH3461771208, ptr null, ptr null })
  br i1 %11, label %12, label %42

12:                                               ; preds = %10
  %13 = load i32, ptr @disposed, align 4
  %14 = icmp eq i32 %13, 1
  br i1 %14, label %15, label %44

15:                                               ; preds = %12
  store i32 0, ptr @disposed, align 4
  %16 = call i1 @expectThrow({ ptr, ptr, ptr } { ptr @.f_main..afL112C24FH3461771208, ptr null, ptr null })
  br i1 %16, label %17, label %46

17:                                               ; preds = %15
  %18 = load i32, ptr @disposed, align 4
  %19 = icmp eq i32 %18, 1
  br i1 %19, label %20, label %48

20:                                               ; preds = %17
  store i32 0, ptr @disposed, align 4
  call void @throwsOrReturns(i1 false)
  %21 = load i32, ptr @disposed, align 4
  %22 = icmp eq i32 %21, 1
  br i1 %22, label %23, label %50

23:                                               ; preds = %20
  store i32 0, ptr @disposed, align 4
  call void @insideHandWrittenTry()
  %24 = load i32, ptr @disposed, align 4
  %25 = icmp eq i32 %24, 1
  br i1 %25, label %26, label %52

26:                                               ; preds = %23
  store i32 0, ptr @disposed, align 4
  call void @insideCatchClause()
  %27 = load i32, ptr @disposed, align 4
  %28 = icmp eq i32 %27, 1
  br i1 %28, label %29, label %54

29:                                               ; preds = %26
  store i32 0, ptr @disposed, align 4
  call void @insideFinallyClause()
  %30 = load i32, ptr @disposed, align 4
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %56

32:                                               ; preds = %29
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

33:                                               ; preds = %56, %54, %52, %50, %48, %46, %44, %42, %40, %38, %36, %34
  unreachable

34:                                               ; preds = %0
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6393293660524519084, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 100, ptr null)
  br label %33

36:                                               ; preds = %2
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18063489156658645917, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 101, ptr null)
  br label %33

38:                                               ; preds = %5
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17214008398373060254, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 104, ptr null)
  br label %33

40:                                               ; preds = %7
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4677460654917946411, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 105, ptr null)
  br label %33

42:                                               ; preds = %10
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5126667335064581011, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 108, ptr null)
  br label %33

44:                                               ; preds = %12
  %45 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2962093585528290310, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 109, ptr null)
  br label %33

46:                                               ; preds = %15
  %47 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7270621579697366617, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 112, ptr null)
  br label %33

48:                                               ; preds = %17
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17305234372915070146, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 113, ptr null)
  br label %33

50:                                               ; preds = %20
  %51 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11832716718392578126, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 117, ptr null)
  br label %33

52:                                               ; preds = %23
  %53 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2099669539852787566, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 121, ptr null)
  br label %33

54:                                               ; preds = %26
  %55 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8177213613051484080, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 125, ptr null)
  br label %33

56:                                               ; preds = %29
  %57 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8933989265959140716, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11237846694651913704, i64 8), i32 129, ptr null)
  br label %33
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

