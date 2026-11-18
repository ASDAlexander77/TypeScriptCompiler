; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Limits = type { ptr }

$Limits..size = comdat any

@s_12780458308147366181 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFLimits\00", align 8
@s_1680353445305148769 = internal constant [297 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnamespace Forward {\0A@dllimport\0Afunction first(p0 : number) : number;\0A\0A}\0Anamespace Forward {\0A@dllimport\0Afunction second(p0 : number) : 0 | 1;\0A\0A}\0Anamespace Forward {\0A@dllimport\0Afunction second(p0 : number) : 0 | 1;\0A\0A}\0Anamespace Forward {\0A@dllimport\0Afunction second(p0 : number) : 0 | 1;\0A\0A}\0A\00", align 8
@s_618458831055183935 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFgc\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_2282180227854419514 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe function generated on the first try\00", align 8
@f_1180329139714669523 = internal constant [115 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00namespace_forward_class.ts\00", align 8
@m_18250523046359488025 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe function naming the later class\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Limits..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12780458308147366181, i64 8)
@Limits..size = global i64 8, comdat
@Limits.BIG = global double 5.000000e+00
@Limits..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Limits..instanceOf, ptr @Limits..new, ptr @Limits..rtti, ptr @Limits..size, ptr @Limits.BIG }
@__decls_00namespace_forward_class_15604214425213433341 = global ptr getelementptr inbounds nuw (i8, ptr @s_1680353445305148769, i64 8)
@__tsmm_gc_00namespace_forward_class_15604214425213433341 = global ptr getelementptr inbounds nuw (i8, ptr @s_618458831055183935, i64 8)
@__tsborrow_00namespace_forward_class_15604214425213433341 = constant [18 x i8] c"Forward.second 0\0A\00"

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define double @Forward.first(double %0) #1 {
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
define i1 @Limits..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Limits..rtti, align 8
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
define ptr @Limits..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Limits, ptr %3, i32 0, i32 0
  store ptr @Limits..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call double @Forward.first(double 1.000000e+00)
  %2 = fcmp oeq double %1, 2.000000e+00
  br i1 %2, label %3, label %13

3:                                                ; preds = %0
  %4 = call i32 @Forward.second(double 5.000000e+00)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = call i32 @Forward.second(double 2.000000e+00)
  %8 = icmp eq i32 %7, 1
  br label %9

9:                                                ; preds = %6, %3
  %10 = phi i1 [ %8, %6 ], [ %5, %3 ]
  br i1 %10, label %11, label %15

11:                                               ; preds = %9
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

12:                                               ; preds = %15, %13
  unreachable

13:                                               ; preds = %0
  %14 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2282180227854419514, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1180329139714669523, i64 8), i32 18, ptr null)
  br label %12

15:                                               ; preds = %9
  %16 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18250523046359488025, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1180329139714669523, i64 8), i32 19, ptr null)
  br label %12
}

; Function Attrs: mustprogress
define i32 @Forward.second(double %0) #1 {
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = load double, ptr @Limits.BIG, align 8
  %6 = fcmp oeq double %4, %5
  br i1 %6, label %7, label %8

7:                                                ; preds = %1
  br label %9

8:                                                ; preds = %1
  br label %9

9:                                                ; preds = %7, %8
  %10 = phi i32 [ 1, %8 ], [ 0, %7 ]
  store i32 %10, ptr %3, align 4
  %11 = load i32, ptr %3, align 4
  ret i32 %11
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

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

