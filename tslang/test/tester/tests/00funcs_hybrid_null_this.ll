; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%C = type { ptr, double }

$C..size = comdat any

@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_6224306806586216532 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdirect call\00", align 8
@f_1079202370173667373 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00funcs_hybrid_null_this.ts\00", align 8
@m_2205649361275638209 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdirect call inside try\00", align 8
@m_6511625932774262644 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbound method\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 16, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @C.m, ptr @C..rtti, ptr @C..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define double @add1(double %0) #1 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fadd double %4, 1.000000e+00
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  ret double %6
}

; Function Attrs: mustprogress
define void @C.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %C, ptr %3, i32 0, i32 1
  store double 1.000000e+01, ptr %4, align 8
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
define double @C.m(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %C, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = call double @add1(double 4.100000e+01)
  %3 = fcmp oeq double %2, 4.200000e+01
  br i1 %3, label %4, label %38

4:                                                ; preds = %0
  store i32 0, ptr %1, align 4
  %5 = invoke double @add1(double 1.000000e+00)
          to label %6 unwind label %32

6:                                                ; preds = %4
  %7 = fptosi double %5 to i64
  %8 = trunc i64 %7 to i32
  store i32 %8, ptr %1, align 4
  %9 = load i32, ptr %1, align 4
  %10 = icmp eq i32 %9, 2
  br i1 %10, label %11, label %40

11:                                               ; preds = %6
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %C, ptr %13, i32 0, i32 0
  store ptr @C..vtbl, ptr %14, align 8
  call void @C.constructor(ptr %13)
  %15 = getelementptr %C, ptr %13, i32 0, i32 0
  %16 = insertvalue { ptr, ptr, ptr } { ptr @C.m, ptr null, ptr null }, ptr %13, 1
  %17 = insertvalue { ptr, ptr, ptr } %16, ptr null, 2
  %18 = extractvalue { ptr, ptr, ptr } %17, 1
  %19 = ptrtoint ptr %18 to i64
  %20 = icmp ne i64 %19, 0
  br i1 %20, label %21, label %24

21:                                               ; preds = %11
  %22 = extractvalue { ptr, ptr, ptr } %17, 0
  %23 = call double %22(ptr %18)
  br label %27

24:                                               ; preds = %11
  %25 = extractvalue { ptr, ptr, ptr } %17, 0
  %26 = call double %25()
  br label %27

27:                                               ; preds = %21, %24
  %28 = phi double [ %26, %24 ], [ %23, %21 ]
  br label %29

29:                                               ; preds = %27
  %30 = fcmp oeq double %28, 1.000000e+01
  br i1 %30, label %31, label %42

31:                                               ; preds = %29
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

32:                                               ; preds = %4
  %33 = landingpad { ptr, i32 }
          cleanup
  %34 = load i32, ptr %1, align 4
  %35 = icmp eq i32 %34, 2
  br i1 %35, label %36, label %44

36:                                               ; preds = %32
  resume { ptr, i32 } %33

37:                                               ; preds = %42, %40, %44, %38
  unreachable

38:                                               ; preds = %0
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6224306806586216532, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1079202370173667373, i64 8), i32 17, ptr null)
  br label %37

40:                                               ; preds = %6
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2205649361275638209, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1079202370173667373, i64 8), i32 23, ptr null)
  br label %37

42:                                               ; preds = %29
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6511625932774262644, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1079202370173667373, i64 8), i32 29, ptr null)
  br label %37

44:                                               ; preds = %32
  %45 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2205649361275638209, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1079202370173667373, i64 8), i32 23, ptr null)
  br label %37
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
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

