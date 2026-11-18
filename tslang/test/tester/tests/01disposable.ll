; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%.f_loggy..objL6C12FH1066116305 = type { ptr, ptr }

@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_2976386471716474339 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFDisposing \00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_9592168636274987234 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCreating \00", align 8
@s_12638182802509129152 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFe\00", align 8
@s_12638183902020757363 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFd\00", align 8
@s_12638189399578898418 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_9002913425553465966 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnot equal create-dispose\00", align 8
@f_11613059637441161894 = internal constant [102 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\01disposable.ts\00", align 8
@count_created = private global i32 0
@count_disp = private global i32 0

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

; Function Attrs: mustprogress
define void @.f_loggy..f_loggy..objL6C12FH1066116305.dispose(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %.f_loggy..objL6C12FH1066116305, ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = icmp eq ptr %7, null
  %9 = select i1 %8, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %7
  %10 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_2976386471716474339, i64 8))
  %11 = add i64 1, %10
  %12 = call i64 @strlen(ptr %9)
  %13 = add i64 %11, %12
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %15 = add i64 %13, %14
  %16 = add i64 %15, 8
  %17 = call ptr @GC_malloc(i64 %16)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr i8, ptr %18, i64 0
  %20 = sub i64 %15, 0
  %21 = call ptr @__memcpy_chk(ptr %19, ptr getelementptr inbounds nuw (i8, ptr @s_2976386471716474339, i64 8), i64 %10, i64 %20)
  %22 = add i64 0, %10
  %23 = getelementptr i8, ptr %18, i64 %22
  %24 = sub i64 %15, %22
  %25 = call ptr @__memcpy_chk(ptr %23, ptr %9, i64 %12, i64 %24)
  %26 = add i64 %22, %12
  %27 = getelementptr i8, ptr %18, i64 %26
  %28 = sub i64 %15, %26
  %29 = call ptr @__memcpy_chk(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %14, i64 %28)
  %30 = add i64 %26, %14
  %31 = getelementptr i8, ptr %18, i64 %30
  store i8 0, ptr %31, align 1
  %32 = icmp eq ptr %18, null
  %33 = select i1 %32, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %18
  call void @puts(ptr %33)
  %34 = load i32, ptr @count_disp, align 4
  %35 = add i32 %34, 1
  store i32 %35, ptr @count_disp, align 4
  ret void
}

; Function Attrs: mustprogress
define ptr @loggy(ptr %0) #1 {
  %2 = alloca { ptr, ptr }, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = icmp eq ptr %5, null
  %7 = select i1 %6, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %5
  %8 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9592168636274987234, i64 8))
  %9 = add i64 1, %8
  %10 = call i64 @strlen(ptr %7)
  %11 = add i64 %9, %10
  %12 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %13 = add i64 %11, %12
  %14 = add i64 %13, 8
  %15 = call ptr @GC_malloc(i64 %14)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr i8, ptr %16, i64 0
  %18 = sub i64 %13, 0
  %19 = call ptr @__memcpy_chk(ptr %17, ptr getelementptr inbounds nuw (i8, ptr @s_9592168636274987234, i64 8), i64 %8, i64 %18)
  %20 = add i64 0, %8
  %21 = getelementptr i8, ptr %16, i64 %20
  %22 = sub i64 %13, %20
  %23 = call ptr @__memcpy_chk(ptr %21, ptr %7, i64 %10, i64 %22)
  %24 = add i64 %20, %10
  %25 = getelementptr i8, ptr %16, i64 %24
  %26 = sub i64 %13, %24
  %27 = call ptr @__memcpy_chk(ptr %25, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %12, i64 %26)
  %28 = add i64 %24, %12
  %29 = getelementptr i8, ptr %16, i64 %28
  store i8 0, ptr %29, align 1
  %30 = icmp eq ptr %16, null
  %31 = select i1 %30, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %16
  call void @puts(ptr %31)
  %32 = load i32, ptr @count_created, align 4
  %33 = add i32 %32, 1
  store i32 %33, ptr @count_created, align 4
  %34 = load ptr, ptr %3, align 8
  %35 = call ptr @GC_malloc(i64 16)
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { ptr }, ptr %36, i32 0, i32 0
  %38 = load ptr, ptr %3, align 8
  store ptr %38, ptr %37, align 8
  store { ptr, ptr } { ptr @.f_loggy..f_loggy..objL6C12FH1066116305.dispose, ptr null }, ptr %2, align 8
  %39 = getelementptr { ptr, ptr }, ptr %2, i32 0, i32 1
  store ptr %36, ptr %39, align 8
  %40 = load { ptr, ptr }, ptr %2, align 8
  %41 = call ptr @GC_malloc(i64 24)
  %42 = getelementptr i8, ptr %41, i64 8
  store { ptr, ptr } %40, ptr %42, align 8
  store ptr %42, ptr %4, align 8
  %43 = load ptr, ptr %4, align 8
  ret ptr %43
}

; Function Attrs: mustprogress
define void @func() #1 {
  %1 = call ptr @loggy(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %2 = call ptr @loggy(ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
  %3 = call ptr @loggy(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %4 = call ptr @loggy(ptr getelementptr inbounds nuw (i8, ptr @s_12638183902020757363, i64 8))
  %5 = getelementptr { ptr, ptr }, ptr %4, i32 0, i32 0
  %6 = insertvalue { ptr, ptr } zeroinitializer, ptr %5, 0
  %7 = insertvalue { ptr, ptr } %6, ptr %4, 1
  %8 = extractvalue { ptr, ptr } %7, 1
  %9 = extractvalue { ptr, ptr } %7, 0
  %10 = load ptr, ptr %9, align 8
  %11 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %10, 0
  %12 = insertvalue { ptr, ptr, ptr } %11, ptr %8, 1
  %13 = insertvalue { ptr, ptr, ptr } %12, ptr null, 2
  %14 = extractvalue { ptr, ptr, ptr } %13, 1
  %15 = extractvalue { ptr, ptr, ptr } %13, 0
  call void %15(ptr %14)
  %16 = getelementptr { ptr, ptr }, ptr %3, i32 0, i32 0
  %17 = insertvalue { ptr, ptr } zeroinitializer, ptr %16, 0
  %18 = insertvalue { ptr, ptr } %17, ptr %3, 1
  %19 = extractvalue { ptr, ptr } %18, 1
  %20 = extractvalue { ptr, ptr } %18, 0
  %21 = load ptr, ptr %20, align 8
  %22 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %21, 0
  %23 = insertvalue { ptr, ptr, ptr } %22, ptr %19, 1
  %24 = insertvalue { ptr, ptr, ptr } %23, ptr null, 2
  %25 = extractvalue { ptr, ptr, ptr } %24, 1
  %26 = extractvalue { ptr, ptr, ptr } %24, 0
  call void %26(ptr %25)
  %27 = call ptr @loggy(ptr getelementptr inbounds nuw (i8, ptr @s_12638182802509129152, i64 8))
  %28 = getelementptr { ptr, ptr }, ptr %27, i32 0, i32 0
  %29 = insertvalue { ptr, ptr } zeroinitializer, ptr %28, 0
  %30 = insertvalue { ptr, ptr } %29, ptr %27, 1
  %31 = extractvalue { ptr, ptr } %30, 1
  %32 = extractvalue { ptr, ptr } %30, 0
  %33 = load ptr, ptr %32, align 8
  %34 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %33, 0
  %35 = insertvalue { ptr, ptr, ptr } %34, ptr %31, 1
  %36 = insertvalue { ptr, ptr, ptr } %35, ptr null, 2
  %37 = extractvalue { ptr, ptr, ptr } %36, 1
  %38 = extractvalue { ptr, ptr, ptr } %36, 0
  call void %38(ptr %37)
  %39 = getelementptr { ptr, ptr }, ptr %2, i32 0, i32 0
  %40 = insertvalue { ptr, ptr } zeroinitializer, ptr %39, 0
  %41 = insertvalue { ptr, ptr } %40, ptr %2, 1
  %42 = extractvalue { ptr, ptr } %41, 1
  %43 = extractvalue { ptr, ptr } %41, 0
  %44 = load ptr, ptr %43, align 8
  %45 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %44, 0
  %46 = insertvalue { ptr, ptr, ptr } %45, ptr %42, 1
  %47 = insertvalue { ptr, ptr, ptr } %46, ptr null, 2
  %48 = extractvalue { ptr, ptr, ptr } %47, 1
  %49 = extractvalue { ptr, ptr, ptr } %47, 0
  call void %49(ptr %48)
  %50 = getelementptr { ptr, ptr }, ptr %1, i32 0, i32 0
  %51 = insertvalue { ptr, ptr } zeroinitializer, ptr %50, 0
  %52 = insertvalue { ptr, ptr } %51, ptr %1, 1
  %53 = extractvalue { ptr, ptr } %52, 1
  %54 = extractvalue { ptr, ptr } %52, 0
  %55 = load ptr, ptr %54, align 8
  %56 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %55, 0
  %57 = insertvalue { ptr, ptr, ptr } %56, ptr %53, 1
  %58 = insertvalue { ptr, ptr, ptr } %57, ptr null, 2
  %59 = extractvalue { ptr, ptr, ptr } %58, 1
  %60 = extractvalue { ptr, ptr, ptr } %58, 0
  call void %60(ptr %59)
  ret void
}

; Function Attrs: mustprogress
define void @func2({ i32, i1 } %0) #1 {
  %2 = alloca i32, align 4
  %3 = extractvalue { i32, i1 } %0, 1
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  %5 = extractvalue { i32, i1 } %0, 0
  br label %7

6:                                                ; preds = %1
  br label %7

7:                                                ; preds = %4, %6
  %8 = phi i32 [ 0, %6 ], [ %5, %4 ]
  store i32 %8, ptr %2, align 4
  %9 = call ptr @loggy(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %10 = load i32, ptr %2, align 4
  %11 = icmp sgt i32 %10, 0
  br i1 %11, label %12, label %24

12:                                               ; preds = %7
  %13 = getelementptr { ptr, ptr }, ptr %9, i32 0, i32 0
  %14 = insertvalue { ptr, ptr } zeroinitializer, ptr %13, 0
  %15 = insertvalue { ptr, ptr } %14, ptr %9, 1
  %16 = extractvalue { ptr, ptr } %15, 1
  %17 = extractvalue { ptr, ptr } %15, 0
  %18 = load ptr, ptr %17, align 8
  %19 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %18, 0
  %20 = insertvalue { ptr, ptr, ptr } %19, ptr %16, 1
  %21 = insertvalue { ptr, ptr, ptr } %20, ptr null, 2
  %22 = extractvalue { ptr, ptr, ptr } %21, 1
  %23 = extractvalue { ptr, ptr, ptr } %21, 0
  call void %23(ptr %22)
  br label %36

24:                                               ; preds = %7
  %25 = getelementptr { ptr, ptr }, ptr %9, i32 0, i32 0
  %26 = insertvalue { ptr, ptr } zeroinitializer, ptr %25, 0
  %27 = insertvalue { ptr, ptr } %26, ptr %9, 1
  %28 = extractvalue { ptr, ptr } %27, 1
  %29 = extractvalue { ptr, ptr } %27, 0
  %30 = load ptr, ptr %29, align 8
  %31 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %30, 0
  %32 = insertvalue { ptr, ptr, ptr } %31, ptr %28, 1
  %33 = insertvalue { ptr, ptr, ptr } %32, ptr null, 2
  %34 = extractvalue { ptr, ptr, ptr } %33, 1
  %35 = extractvalue { ptr, ptr, ptr } %33, 0
  call void %35(ptr %34)
  br label %36

36:                                               ; preds = %12, %24
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @func()
  call void @func2({ i32, i1 } zeroinitializer)
  %1 = load i32, ptr @count_created, align 4
  %2 = load i32, ptr @count_disp, align 4
  %3 = icmp eq i32 %1, %2
  br i1 %3, label %4, label %6

4:                                                ; preds = %0
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

5:                                                ; preds = %6
  unreachable

6:                                                ; preds = %0
  %7 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9002913425553465966, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11613059637441161894, i64 8), i32 37, ptr null)
  br label %5
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

