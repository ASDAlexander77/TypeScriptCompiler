; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Leaf = type { ptr, double }

$Leaf..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@a_3_11910754096253460095 = internal constant <3 x i32> <i32 3, i32 5, i32 9>
@s_11943693223072092597 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFLeaf\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_17800423962769158986 = internal constant [60 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa value allocated in a catch clause is usable there\00", align 8
@f_9568412707720620255 = internal constant [106 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00alloc_in_catch.ts\00", align 8
@m_12441992499582873086 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFand in a finally clause\00", align 8
@m_1893519899433542135 = internal constant [69 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan array literal allocated in a catch clause is usable there\00", align 8
@m_11991097666452197790 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtwo allocations in one handler both survive\00", align 8
@m_14581256387588719820 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa handler nested inside a handler allocates too\00", align 8
@m_15079221229597488554 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa handler that allocates can still throw on\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Leaf..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11943693223072092597, i64 8)
@Leaf..size = global i64 16, comdat
@Leaf..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Leaf..instanceOf, ptr @Leaf..new, ptr @Leaf.get, ptr @Leaf..rtti, ptr @Leaf..size }
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #2

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Leaf..instanceOf(ptr %0, ptr %1) #3 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Leaf..rtti, align 8
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
define ptr @Leaf..new() #3 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Leaf, ptr %3, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Leaf.constructor(ptr %0, double %1) #3 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Leaf, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Leaf.get(ptr %0) #3 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Leaf, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define i32 @allocInCatch() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca <{ i64, %Leaf }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Leaf zeroinitializer, ptr %5, align 8
  store i32 0, ptr %2, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %26 unwind label %7

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
  br label %26

15:                                               ; preds = %7
  %16 = extractvalue { ptr, i32 } %8, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = load i32, ptr %17, align 4
  store i32 %18, ptr %1, align 4
  store %Leaf zeroinitializer, ptr %5, align 8
  %19 = getelementptr %Leaf, ptr %5, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %19, align 8
  invoke void @Leaf.constructor(ptr %5, double 7.000000e+00)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %15
  %20 = getelementptr %Leaf, ptr %5, i32 0, i32 0
  %21 = invoke double @Leaf.get(ptr %5)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.noexc
  %22 = fptosi double %21 to i64
  %23 = trunc i64 %22 to i32
  store i32 %23, ptr %2, align 4
  call void @__cxa_end_catch()
  %24 = load i32, ptr %2, align 4
  store i32 %24, ptr %3, align 4
  %25 = load i32, ptr %3, align 4
  ret i32 %25

26:                                               ; preds = %.noexc2, %0
  unreachable

endcatch.resume:                                  ; preds = %12, %.noexc, %15
  %27 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %27
}

; Function Attrs: mustprogress
define i32 @allocInFinally() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca <{ i64, %Leaf }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Leaf zeroinitializer, ptr %4, align 8
  store i32 0, ptr %1, align 4
  store i32 1, ptr %1, align 4
  store %Leaf zeroinitializer, ptr %4, align 8
  %5 = getelementptr %Leaf, ptr %4, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %5, align 8
  call void @Leaf.constructor(ptr %4, double 4.000000e+00)
  %6 = load i32, ptr %1, align 4
  %7 = getelementptr %Leaf, ptr %4, i32 0, i32 0
  %8 = call double @Leaf.get(ptr %4)
  %9 = sitofp i32 %6 to double
  %10 = fadd double %9, %8
  %11 = fptosi double %10 to i64
  %12 = trunc i64 %11 to i32
  store i32 %12, ptr %1, align 4
  %13 = load i32, ptr %1, align 4
  store i32 %13, ptr %2, align 4
  %14 = load i32, ptr %2, align 4
  ret i32 %14
}

; Function Attrs: mustprogress
define i32 @arrayInCatch() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store ptr null, ptr %1, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %42 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %11
  br label %42

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %2, align 4
  %18 = invoke ptr @GC_malloc(i64 20)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %14
  %19 = getelementptr i8, ptr %18, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %19, ptr @a_3_11910754096253460095, i64 12, i1 false)
  %20 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr { ptr, i64, i64 }, ptr %21, i32 0, i32 0
  store ptr %19, ptr %22, align 8
  %23 = getelementptr { ptr, i64, i64 }, ptr %21, i32 0, i32 1
  store i64 3, ptr %23, align 8
  %24 = getelementptr { ptr, i64, i64 }, ptr %21, i32 0, i32 2
  store i64 3, ptr %24, align 8
  store ptr %21, ptr %1, align 8
  %25 = load ptr, ptr %1, align 8
  %26 = icmp eq ptr %25, null
  %27 = select i1 %26, ptr @__ts_empty_array, ptr %25
  %28 = getelementptr { ptr, i64, i64 }, ptr %27, i32 0, i32 0
  %29 = load ptr, ptr %28, align 8
  %30 = getelementptr i32, ptr %29, i32 0
  %31 = load i32, ptr %30, align 4
  %32 = load ptr, ptr %1, align 8
  %33 = icmp eq ptr %32, null
  %34 = select i1 %33, ptr @__ts_empty_array, ptr %32
  %35 = getelementptr { ptr, i64, i64 }, ptr %34, i32 0, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr i32, ptr %36, i32 2
  %38 = load i32, ptr %37, align 4
  %39 = add i32 %31, %38
  store i32 %39, ptr %3, align 4
  call void @__cxa_end_catch()
  %40 = load i32, ptr %3, align 4
  store i32 %40, ptr %4, align 4
  %41 = load i32, ptr %4, align 4
  ret i32 %41

42:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %.noexc1, %14, %11
  %43 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %43
}

; Function Attrs: mustprogress
define i32 @twoAllocsInCatch() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca <{ i64, %Leaf }>, align 16
  %5 = alloca <{ i64, %Leaf }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Leaf zeroinitializer, ptr %6, align 8
  store i64 -1, ptr %4, align 8
  %7 = getelementptr i8, ptr %4, i64 8
  store %Leaf zeroinitializer, ptr %7, align 8
  store i32 0, ptr %2, align 4
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %32 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %11 = extractvalue { ptr, i32 } %10, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %17, label %14

14:                                               ; preds = %9
  %15 = extractvalue { ptr, i32 } %10, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  invoke void @__cxa_rethrow()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %14
  br label %32

17:                                               ; preds = %9
  %18 = extractvalue { ptr, i32 } %10, 0
  %19 = call ptr @__cxa_begin_catch(ptr %18)
  %20 = load i32, ptr %19, align 4
  store i32 %20, ptr %1, align 4
  store %Leaf zeroinitializer, ptr %7, align 8
  %21 = getelementptr %Leaf, ptr %7, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %21, align 8
  invoke void @Leaf.constructor(ptr %7, double 2.000000e+00)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %17
  store %Leaf zeroinitializer, ptr %6, align 8
  %22 = getelementptr %Leaf, ptr %6, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %22, align 8
  invoke void @Leaf.constructor(ptr %6, double 3.000000e+00)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.noexc
  %23 = getelementptr %Leaf, ptr %7, i32 0, i32 0
  %24 = invoke double @Leaf.get(ptr %7)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %25 = getelementptr %Leaf, ptr %6, i32 0, i32 0
  %26 = invoke double @Leaf.get(ptr %6)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  %27 = fmul double %24, %26
  %28 = fptosi double %27 to i64
  %29 = trunc i64 %28 to i32
  store i32 %29, ptr %2, align 4
  call void @__cxa_end_catch()
  %30 = load i32, ptr %2, align 4
  store i32 %30, ptr %3, align 4
  %31 = load i32, ptr %3, align 4
  ret i32 %31

32:                                               ; preds = %.noexc4, %0
  unreachable

endcatch.resume:                                  ; preds = %14, %.noexc2, %.noexc1, %.noexc, %17
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define i32 @allocInNestedCatch() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca <{ i64, %Leaf }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Leaf zeroinitializer, ptr %6, align 8
  store i32 0, ptr %3, align 4
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIi, ptr null)
          to label %40 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %10 = extractvalue { ptr, i32 } %9, 1
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %16, label %13

13:                                               ; preds = %8
  %14 = extractvalue { ptr, i32 } %9, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  invoke void @__cxa_rethrow()
          to label %.noexc5 unwind label %endcatch.resume3

.noexc5:                                          ; preds = %13
  br label %40

16:                                               ; preds = %8
  %17 = extractvalue { ptr, i32 } %9, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = load i32, ptr %18, align 4
  store i32 %19, ptr %2, align 4
  %20 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc4 unwind label %endcatch.resume3

.noexc4:                                          ; preds = %16
  store i32 2, ptr %20, align 4
  invoke void @__cxa_throw(ptr %20, ptr @_ZTIi, ptr null)
          to label %40 unwind label %21

21:                                               ; preds = %.noexc4
  %22 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %23 = extractvalue { ptr, i32 } %22, 1
  %24 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %25 = icmp eq i32 %23, %24
  br i1 %25, label %29, label %26

26:                                               ; preds = %21
  %27 = extractvalue { ptr, i32 } %22, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %26
  br label %40

29:                                               ; preds = %21
  %30 = extractvalue { ptr, i32 } %22, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  %32 = load i32, ptr %31, align 4
  store i32 %32, ptr %1, align 4
  store %Leaf zeroinitializer, ptr %6, align 8
  %33 = getelementptr %Leaf, ptr %6, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %33, align 8
  invoke void @Leaf.constructor(ptr %6, double 1.100000e+01)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %29
  %34 = getelementptr %Leaf, ptr %6, i32 0, i32 0
  %35 = invoke double @Leaf.get(ptr %6)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %36 = fptosi double %35 to i64
  %37 = trunc i64 %36 to i32
  store i32 %37, ptr %3, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  %38 = load i32, ptr %3, align 4
  store i32 %38, ptr %4, align 4
  %39 = load i32, ptr %4, align 4
  ret i32 %39

40:                                               ; preds = %.noexc, %.noexc4, %.noexc5, %0
  unreachable

endcatch.resume:                                  ; preds = %.noexc1, %29, %26
  %41 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %41

endcatch.resume3:                                 ; preds = %13, %16
  %42 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %42
}

; Function Attrs: mustprogress
define i32 @allocInCatchThenThrow() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca <{ i64, %Leaf }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Leaf zeroinitializer, ptr %4, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %26 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %11
  br label %26

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %1, align 4
  store %Leaf zeroinitializer, ptr %4, align 8
  %18 = getelementptr %Leaf, ptr %4, i32 0, i32 0
  store ptr @Leaf..vtbl, ptr %18, align 8
  invoke void @Leaf.constructor(ptr %4, double 6.000000e+00)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %14
  %19 = getelementptr %Leaf, ptr %4, i32 0, i32 0
  %20 = invoke double @Leaf.get(ptr %4)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.noexc
  %21 = fcmp ogt double %20, 0.000000e+00
  br i1 %21, label %22, label %24

22:                                               ; preds = %.noexc1
  call void @__cxa_end_catch()
  %23 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %23, align 4
  call void @__cxa_throw(ptr %23, ptr @_ZTIi, ptr null)
  br label %26

24:                                               ; preds = %.noexc1
  call void @__cxa_end_catch()
  store i32 0, ptr %2, align 4
  %25 = load i32, ptr %2, align 4
  ret i32 %25

26:                                               ; preds = %22, %.noexc2, %0
  unreachable

endcatch.resume:                                  ; preds = %11, %.noexc, %14
  %27 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %27
}

; Function Attrs: mustprogress
define i32 @main() #4 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca i1, align 1
  %3 = call i32 @allocInCatch()
  %4 = icmp eq i32 %3, 7
  br i1 %4, label %5, label %36

5:                                                ; preds = %0
  %6 = call i32 @allocInFinally()
  %7 = icmp eq i32 %6, 5
  br i1 %7, label %8, label %38

8:                                                ; preds = %5
  %9 = call i32 @arrayInCatch()
  %10 = icmp eq i32 %9, 12
  br i1 %10, label %11, label %40

11:                                               ; preds = %8
  %12 = call i32 @twoAllocsInCatch()
  %13 = icmp eq i32 %12, 6
  br i1 %13, label %14, label %42

14:                                               ; preds = %11
  %15 = call i32 @allocInNestedCatch()
  %16 = icmp eq i32 %15, 11
  br i1 %16, label %17, label %44

17:                                               ; preds = %14
  store i1 false, ptr %2, align 1
  %18 = invoke i32 @allocInCatchThenThrow()
          to label %19 unwind label %20

19:                                               ; preds = %17
  br label %32

20:                                               ; preds = %17
  %21 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %22 = extractvalue { ptr, i32 } %21, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %24 = icmp eq i32 %22, %23
  br i1 %24, label %28, label %25

25:                                               ; preds = %20
  %26 = extractvalue { ptr, i32 } %21, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %25
  br label %35

28:                                               ; preds = %20
  %29 = extractvalue { ptr, i32 } %21, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  %31 = load i32, ptr %30, align 4
  store i32 %31, ptr %1, align 4
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  br label %32

32:                                               ; preds = %19, %28
  %33 = load i1, ptr %2, align 1
  br i1 %33, label %34, label %46

34:                                               ; preds = %32
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

35:                                               ; preds = %46, %.noexc, %44, %42, %40, %38, %36
  unreachable

36:                                               ; preds = %0
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17800423962769158986, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9568412707720620255, i64 8), i32 123, ptr null)
  br label %35

38:                                               ; preds = %5
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12441992499582873086, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9568412707720620255, i64 8), i32 124, ptr null)
  br label %35

40:                                               ; preds = %8
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1893519899433542135, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9568412707720620255, i64 8), i32 125, ptr null)
  br label %35

42:                                               ; preds = %11
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11991097666452197790, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9568412707720620255, i64 8), i32 126, ptr null)
  br label %35

44:                                               ; preds = %14
  %45 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14581256387588719820, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9568412707720620255, i64 8), i32 127, ptr null)
  br label %35

46:                                               ; preds = %32
  %47 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15079221229597488554, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9568412707720620255, i64 8), i32 137, ptr null)
  br label %35

endcatch.resume:                                  ; preds = %25
  %48 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %48
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
attributes #2 = { nounwind memory(none) }
attributes #3 = { mustprogress }
attributes #4 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

