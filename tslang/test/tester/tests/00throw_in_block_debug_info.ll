; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Err = type { ptr, ptr }

$Err..size = comdat any

$_ZTSP3Err = comdat any

$_ZTIP3Err = comdat any

$_ZTS3Err = comdat any

$_ZTI3Err = comdat any

@s_15638227547778902572 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFErr\00", align 8
@Err..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_15638227547778902572, i64 8)
@Err..size = global i64 16, comdat
@Err..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Err..instanceOf, ptr @Err..new, ptr @Err..rtti, ptr @Err..size }
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP3Err = linkonce_odr constant [6 x i8] c"P3Err\00", comdat
@_ZTIP3Err = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP3Err, i32 1073741824, ptr @_ZTI3Err, ptr @.eh.copy.box.P3Err }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS3Err = linkonce_odr constant [5 x i8] c"3Err\00", comdat
@s_12682538616232704602 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnegative\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_15927202708308728429 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf(1)\00", align 8
@f_7584780597882467706 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_in_block_debug_info.ts\00", align 8
@m_5827285053737075023 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFg(-1)\00", align 8
@m_11332068698168565619 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFg(2)\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTI3Err = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS3Err }, comdat
@td_2217350317242203440_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_2217350317242203440, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

define internal void @__tslang_inc_ref(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %13

3:                                                ; preds = %1
  %4 = getelementptr i8, ptr %0, i64 -8
  %5 = load i64, ptr %4, align 8
  %6 = icmp ne i64 %5, -1
  %7 = icmp sge i64 %5, 0
  br i1 %7, label %11, label %8

8:                                                ; preds = %3
  br i1 %6, label %9, label %13

9:                                                ; preds = %8
  %10 = atomicrmw add ptr %4, i64 1 monotonic, align 8
  br label %13

11:                                               ; preds = %3
  %12 = add i64 %5, 1
  store i64 %12, ptr %4, align 8
  br label %13

13:                                               ; preds = %11, %9, %8, %1
  ret void
}

define internal void @__tslang_retain_block_slot(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal i1 @__tslang_dec_ref(ptr %0) {
  %2 = getelementptr i8, ptr %0, i64 -8
  %3 = load i64, ptr %2, align 8
  %4 = icmp ne i64 %3, -1
  %5 = icmp sge i64 %3, 0
  br i1 %5, label %11, label %6

6:                                                ; preds = %1
  br i1 %4, label %7, label %15

7:                                                ; preds = %6
  %8 = atomicrmw sub ptr %2, i64 1 release, align 8
  %9 = and i64 %8, 9223372036854775807
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %14, label %15

11:                                               ; preds = %1
  %12 = sub i64 %3, 1
  store i64 %12, ptr %2, align 8
  %13 = icmp eq i64 %12, 0
  ret i1 %13

14:                                               ; preds = %7
  fence acquire
  ret i1 true

15:                                               ; preds = %7, %6
  ret i1 false
}

declare void @GC_free(ptr)

define internal void @__tslang_free_block(ptr %0) {
  %2 = icmp eq ptr %0, null
  %3 = getelementptr i8, ptr %0, i64 -8
  %4 = select i1 %2, ptr null, ptr %3
  call void @GC_free(ptr %4)
  ret void
}

define internal void @tsrel_1752011451674293319(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  call void @__tslang_free_block(ptr %2)
  br label %7

7:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrel_2217350317242203440(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Err, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Err..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Err..rtti, align 8
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
define ptr @Err..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Err, ptr %3, i32 0, i32 0
  store ptr @Err..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Err.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Err, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P3Err(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_2217350317242203440_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define double @f(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp olt double %4, 0.000000e+00
  br i1 %5, label %6, label %11

6:                                                ; preds = %1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Err, ptr %8, i32 0, i32 0
  store ptr @Err..vtbl, ptr %9, align 8
  call void @Err.constructor(ptr %8, ptr getelementptr inbounds nuw (i8, ptr @s_12682538616232704602, i64 8))
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %8, ptr %10, align 8
  call void @__cxa_throw(ptr %10, ptr @_ZTIP3Err, ptr null)
  br label %14

11:                                               ; preds = %1
  %12 = load double, ptr %2, align 8
  store double %12, ptr %3, align 8
  %13 = load double, ptr %3, align 8
  ret double %13

14:                                               ; preds = %6
  unreachable
}

; Function Attrs: mustprogress
define ptr @g(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  store double %0, ptr %4, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %3, align 8
  %6 = load double, ptr %4, align 8
  %7 = invoke double @f(double %6)
          to label %8 unwind label %9

8:                                                ; preds = %1
  br label %23

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr @_ZTIP3Err
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP3Err)
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %17, label %14

14:                                               ; preds = %9
  %15 = extractvalue { ptr, i32 } %10, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  invoke void @__cxa_rethrow()
          to label %30 unwind label %endcatch.pad

17:                                               ; preds = %9
  %18 = extractvalue { ptr, i32 } %10, 0
  %19 = call ptr @__cxa_begin_catch(ptr %18)
  store ptr %19, ptr %2, align 8
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr %Err, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %21, align 8
  store ptr %22, ptr %3, align 8
  call void @__cxa_end_catch()
  br label %23

23:                                               ; preds = %8, %17
  %24 = load ptr, ptr %3, align 8
  store ptr %24, ptr %5, align 8
  %25 = load ptr, ptr %5, align 8
  ret ptr %25

endcatch.pad:                                     ; preds = %14
  %26 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

27:                                               ; No predecessors!
  %28 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %27
  %29 = phi { ptr, i32 } [ %28, %27 ], [ %26, %endcatch.pad ]
  resume { ptr, i32 } %29

30:                                               ; preds = %14
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call double @f(double 1.000000e+00)
  %2 = fcmp oeq double %1, 1.000000e+00
  br i1 %2, label %3, label %35

3:                                                ; preds = %0
  %4 = call ptr @g(double -1.000000e+00)
  %5 = ptrtoint ptr %4 to i64
  %6 = icmp ne i64 %5, 0
  %7 = zext i1 %6 to i32
  %8 = and i32 %7, 1
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %10, label %13

10:                                               ; preds = %3
  %11 = call i32 @strcmp(ptr %4, ptr getelementptr inbounds nuw (i8, ptr @s_12682538616232704602, i64 8))
  %12 = icmp eq i32 %11, 0
  br label %15

13:                                               ; preds = %3
  %14 = icmp eq i64 %5, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12682538616232704602, i64 8) to i64)
  br label %15

15:                                               ; preds = %10, %13
  %16 = phi i1 [ %14, %13 ], [ %12, %10 ]
  br label %17

17:                                               ; preds = %15
  br i1 %16, label %18, label %37

18:                                               ; preds = %17
  %19 = call ptr @g(double 2.000000e+00)
  %20 = ptrtoint ptr %19 to i64
  %21 = icmp ne i64 %20, 0
  %22 = zext i1 %21 to i32
  %23 = and i32 %22, 1
  %24 = icmp ne i32 %23, 0
  br i1 %24, label %25, label %28

25:                                               ; preds = %18
  %26 = call i32 @strcmp(ptr %19, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %27 = icmp eq i32 %26, 0
  br label %30

28:                                               ; preds = %18
  %29 = icmp eq i64 %20, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8) to i64)
  br label %30

30:                                               ; preds = %25, %28
  %31 = phi i1 [ %29, %28 ], [ %27, %25 ]
  br label %32

32:                                               ; preds = %30
  br i1 %31, label %33, label %39

33:                                               ; preds = %32
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

34:                                               ; preds = %39, %37, %35
  unreachable

35:                                               ; preds = %0
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15927202708308728429, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7584780597882467706, i64 8), i32 29, ptr null)
  br label %34

37:                                               ; preds = %17
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5827285053737075023, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7584780597882467706, i64 8), i32 30, ptr null)
  br label %34

39:                                               ; preds = %32
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11332068698168565619, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7584780597882467706, i64 8), i32 31, ptr null)
  br label %34
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

attributes #0 = { nounwind memory(none) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

