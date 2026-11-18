; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%E = type { ptr, ptr }
%async_execute_fn.Frame = type { ptr, ptr, ptr, ptr, i1 }
%async_execute_fn_0.Frame = type { ptr, ptr, ptr, ptr, i1 }

$E..size = comdat any

$_ZTSP1E = comdat any

$_ZTIP1E = comdat any

$_ZTS1E = comdat any

$_ZTI1E = comdat any

@assert_msg_0 = private constant [40 x i8] c"Awaited async operand is in error state\00"
@assert_msg = private constant [40 x i8] c"Awaited async operand is in error state\00"
@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638217986881231904 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFE\00", align 8
@E..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638217986881231904, i64 8)
@E..size = global i64 16, comdat
@E..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @E..instanceOf, ptr @E..new, ptr @E..rtti, ptr @E..size }
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP1E = linkonce_odr constant [4 x i8] c"P1E\00", comdat
@_ZTIP1E = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP1E, i32 1073741824, ptr @_ZTI1E, ptr @.eh.copy.box.P1E }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS1E = linkonce_odr constant [3 x i8] c"1E\00", comdat
@s_9443188008722762343 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcheck\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_3734525892634768032 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00own_move_in_unwound_scope.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@m_18025199179193165964 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin try, no throw\00", align 8
@m_10559759729922936440 = internal constant [39 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin try, thrown before the move\00", align 8
@m_7625973300497561953 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin try, thrown after the move\00", align 8
@m_17904541597739498615 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbeside a catch\00", align 8
@m_15907365327259447507 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbeside a catch, caught\00", align 8
@m_11381653347708366518 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan await beside a catch\00", align 8
@m_17996328347166339198 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan await beside a catch, caught\00", align 8
@m_14459584306309247786 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan awaited array of async calls beside a catch\00", align 8
@m_7781056549463363810 = internal constant [63 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan awaited array of async calls beside a catch, caught\00", align 8
@m_3181862927481854341 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmoved, then thrown out of the function\00", align 8
@_ZTI1E = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1E }, comdat
@td_7632308648875476676_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_7632308648875476676, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@a_0_5295668518244173888 = internal constant [0 x i32] undef

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_uncollectable(i64) #0

declare void @abort()

declare void @puts(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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

define internal void @tsrel_7632308648875476676(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %E, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

declare void @GC_free(ptr)

; Function Attrs: mustprogress
define i1 @E..instanceOf(ptr %0, ptr %1) #3 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @E..rtti, align 8
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
define ptr @E..new() #3 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %E, ptr %3, i32 0, i32 0
  store ptr @E..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @E.constructor(ptr %0, ptr %1) #3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %E, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @make(i32 %0) #3 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  store ptr null, ptr %3, align 8
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = add i32 %5, 1
  %7 = call ptr @GC_malloc(i64 16)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr i32, ptr %8, i64 0
  store i32 %4, ptr %9, align 4
  %10 = getelementptr i32, ptr %8, i64 1
  store i32 %6, ptr %10, align 4
  %11 = call ptr @GC_malloc(i64 32)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 0
  store ptr %8, ptr %13, align 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 1
  store i64 2, ptr %14, align 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 2
  store i64 2, ptr %15, align 8
  store ptr %12, ptr %3, align 8
  %16 = load ptr, ptr %3, align 8
  ret ptr %16
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P1E(ptr %0, ptr %1) #3 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_7632308648875476676_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define i32 @check(ptr %0, i1 %1) #3 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i1 %1, ptr %3, align 1
  %6 = load i1, ptr %3, align 1
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %E, ptr %9, i32 0, i32 0
  store ptr @E..vtbl, ptr %10, align 8
  call void @E.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_9443188008722762343, i64 8))
  %11 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %9, ptr %11, align 8
  call void @__cxa_throw(ptr %11, ptr @_ZTIP1E, ptr null)
  br label %20

12:                                               ; preds = %2
  %13 = load ptr, ptr %4, align 8
  %14 = icmp eq ptr %13, null
  %15 = select i1 %14, ptr @__ts_empty_array, ptr %13
  %16 = getelementptr { ptr, i64, i64 }, ptr %15, i32 0, i32 1
  %17 = load i64, ptr %16, align 8
  %18 = trunc i64 %17 to i32
  store i32 %18, ptr %5, align 4
  %19 = load i32, ptr %5, align 4
  ret i32 %19

20:                                               ; preds = %7
  unreachable
}

; Function Attrs: mustprogress
define i32 @twice(i32 %0) #3 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = mul i32 %4, 2
  store i32 %5, ptr %3, align 4
  %6 = load i32, ptr %3, align 4
  ret i32 %6
}

; Function Attrs: mustprogress
define ptr @pair(i32 %0) #3 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  store ptr null, ptr %3, align 8
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = mul i32 %5, 2
  %7 = call ptr @GC_malloc(i64 16)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr i32, ptr %8, i64 0
  store i32 %4, ptr %9, align 4
  %10 = getelementptr i32, ptr %8, i64 1
  store i32 %6, ptr %10, align 4
  %11 = call ptr @GC_malloc(i64 32)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 0
  store ptr %8, ptr %13, align 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 1
  store i64 2, ptr %14, align 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %12, i32 0, i32 2
  store i64 2, ptr %15, align 8
  store ptr %12, ptr %3, align 8
  %16 = load ptr, ptr %3, align 8
  ret ptr %16
}

; Function Attrs: mustprogress
define ptr @all(ptr %0) #3 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr null, ptr %5, align 8
  store ptr %0, ptr %4, align 8
  %6 = call ptr @GC_malloc(i64 8)
  %7 = getelementptr i8, ptr %6, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %7, ptr @a_0_5295668518244173888, i64 0, i1 false)
  %8 = call ptr @GC_malloc(i64 32)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 0
  store ptr %7, ptr %10, align 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 1
  store i64 0, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %9, i32 0, i32 2
  store i64 0, ptr %12, align 8
  store ptr %9, ptr %3, align 8
  store i32 0, ptr %2, align 4
  br label %13

13:                                               ; preds = %75, %1
  %14 = load i32, ptr %2, align 4
  %15 = load ptr, ptr %4, align 8
  %16 = icmp eq ptr %15, null
  %17 = select i1 %16, ptr @__ts_empty_array, ptr %15
  %18 = getelementptr { ptr, i64, i64 }, ptr %17, i32 0, i32 1
  %19 = load i64, ptr %18, align 8
  %20 = sext i32 %14 to i64
  %21 = icmp slt i64 %20, %19
  br i1 %21, label %22, label %79

22:                                               ; preds = %13
  %23 = load i32, ptr %2, align 4
  %24 = load ptr, ptr %4, align 8
  %25 = icmp eq ptr %24, null
  %26 = select i1 %25, ptr @__ts_empty_array, ptr %24
  %27 = getelementptr { ptr, i64, i64 }, ptr %26, i32 0, i32 0
  %28 = load ptr, ptr %27, align 8
  %29 = getelementptr i32, ptr %28, i32 %23
  %30 = load i32, ptr %29, align 4
  %31 = load ptr, ptr %3, align 8
  %32 = icmp eq ptr %31, null
  br i1 %32, label %33, label %36

33:                                               ; preds = %22
  %34 = call ptr @GC_malloc(i64 32)
  %35 = getelementptr i8, ptr %34, i64 8
  store ptr %35, ptr %3, align 8
  br label %37

36:                                               ; preds = %22
  br label %37

37:                                               ; preds = %33, %36
  %38 = phi ptr [ %31, %36 ], [ %35, %33 ]
  br label %39

39:                                               ; preds = %37
  %40 = getelementptr { ptr, i64, i64 }, ptr %38, i32 0, i32 1
  %41 = load i64, ptr %40, align 8
  %42 = add i64 %41, 1
  %43 = getelementptr { ptr, i64, i64 }, ptr %38, i32 0, i32 0
  %44 = getelementptr { ptr, i64, i64 }, ptr %38, i32 0, i32 2
  %45 = load ptr, ptr %43, align 8
  %46 = load i64, ptr %44, align 8
  %47 = icmp ugt i64 %42, %46
  br i1 %47, label %48, label %72

48:                                               ; preds = %39
  %49 = mul i64 %46, 2
  %50 = icmp uge i64 %49, %42
  %51 = select i1 %50, i64 %49, i64 %42
  %52 = icmp ult i64 %51, 4
  %53 = select i1 %52, i64 4, i64 %51
  %54 = icmp ule i64 %42, 4294967295
  br i1 %54, label %55, label %83

55:                                               ; preds = %48
  %56 = icmp ule i64 %53, 4611686018427387903
  br i1 %56, label %57, label %85

57:                                               ; preds = %55
  %58 = mul i64 4, %53
  %59 = add i64 %58, 8
  %60 = icmp eq ptr %45, null
  %61 = getelementptr i8, ptr %45, i64 -8
  %62 = select i1 %60, ptr null, ptr %61
  %63 = call ptr @GC_realloc(ptr %62, i64 %59)
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = icmp ne ptr %64, null
  %66 = icmp eq i64 %58, 0
  %67 = or i1 %65, %66
  br i1 %67, label %68, label %87

68:                                               ; preds = %57
  %69 = mul i64 4, %46
  %70 = getelementptr i8, ptr %64, i64 %69
  %71 = sub i64 %58, %69
  call void @llvm.memset.p0.i64(ptr %70, i8 0, i64 %71, i1 false)
  store ptr %64, ptr %43, align 8
  store i64 %53, ptr %44, align 8
  br label %73

72:                                               ; preds = %39
  br label %73

73:                                               ; preds = %68, %72
  %74 = phi ptr [ %45, %72 ], [ %64, %68 ]
  br label %75

75:                                               ; preds = %73
  %76 = getelementptr i32, ptr %74, i64 %41
  store i32 %30, ptr %76, align 4
  store i64 %42, ptr %40, align 8
  %77 = load i32, ptr %2, align 4
  %78 = add i32 %77, 1
  store i32 %78, ptr %2, align 4
  br label %13

79:                                               ; preds = %13
  %80 = load ptr, ptr %3, align 8
  store ptr %80, ptr %5, align 8
  %81 = load ptr, ptr %5, align 8
  ret ptr %81

82:                                               ; preds = %87, %85, %83
  unreachable

83:                                               ; preds = %48
  %84 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 35, ptr null)
  br label %82

85:                                               ; preds = %55
  %86 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 35, ptr null)
  br label %82

87:                                               ; preds = %57
  %88 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 35, ptr null)
  br label %82
}

; Function Attrs: mustprogress
define i32 @inTry(i1 %0, i1 %1) #4 personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i1, align 1
  %8 = alloca i1, align 1
  %9 = alloca i32, align 4
  store i1 %0, ptr %8, align 1
  store i1 %1, ptr %7, align 1
  store i32 0, ptr %6, align 4
  store ptr null, ptr %4, align 8
  store ptr null, ptr %3, align 8
  %10 = invoke ptr @make(i32 1)
          to label %11 unwind label %21

11:                                               ; preds = %2
  store ptr %10, ptr %4, align 8
  %12 = load ptr, ptr %4, align 8
  %13 = load i1, ptr %8, align 1
  %14 = invoke i32 @check(ptr %12, i1 %13)
          to label %15 unwind label %21

15:                                               ; preds = %11
  %16 = load ptr, ptr %4, align 8
  store ptr %16, ptr %3, align 8
  %17 = load ptr, ptr %3, align 8
  %18 = load i1, ptr %7, align 1
  %19 = invoke i32 @check(ptr %17, i1 %18)
          to label %20 unwind label %21

20:                                               ; preds = %15
  store i32 %19, ptr %6, align 4
  br label %32

21:                                               ; preds = %15, %11, %2
  %22 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %23 = extractvalue { ptr, i32 } %22, 1
  %24 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %25 = icmp eq i32 %23, %24
  br i1 %25, label %29, label %26

26:                                               ; preds = %21
  %27 = extractvalue { ptr, i32 } %22, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %26
  br label %35

29:                                               ; preds = %21
  %30 = extractvalue { ptr, i32 } %22, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  store ptr %31, ptr %5, align 8
  store i32 -1, ptr %6, align 4
  call void @__cxa_end_catch()
  br label %32

32:                                               ; preds = %20, %29
  %33 = load i32, ptr %6, align 4
  store i32 %33, ptr %9, align 4
  %34 = load i32, ptr %9, align 4
  ret i32 %34

35:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %26
  %36 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %36
}

; Function Attrs: mustprogress
define i32 @besideCatch(i1 %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i1, align 1
  %9 = alloca i32, align 4
  store i1 %0, ptr %8, align 1
  store ptr null, ptr %7, align 8
  store ptr null, ptr %6, align 8
  store ptr null, ptr %5, align 8
  %10 = invoke ptr @make(i32 1)
          to label %11 unwind label %59

11:                                               ; preds = %1
  store ptr %10, ptr %7, align 8
  %12 = invoke ptr @make(i32 2)
          to label %13 unwind label %59

13:                                               ; preds = %11
  store ptr %12, ptr %6, align 8
  %14 = load ptr, ptr %6, align 8
  store ptr %14, ptr %5, align 8
  store i1 false, ptr %4, align 1
  store ptr null, ptr %2, align 8
  %15 = call ptr @GC_malloc(i64 8)
  %16 = getelementptr i8, ptr %15, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %16, ptr @a_0_5295668518244173888, i64 0, i1 false)
  %17 = call ptr @GC_malloc(i64 32)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 0
  store ptr %16, ptr %19, align 8
  %20 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 1
  store i64 0, ptr %20, align 8
  %21 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 2
  store i64 0, ptr %21, align 8
  store ptr %18, ptr %2, align 8
  %22 = load ptr, ptr %2, align 8
  %23 = load i1, ptr %8, align 1
  %24 = invoke i32 @check(ptr %22, i1 %23)
          to label %25 unwind label %26

25:                                               ; preds = %13
  br label %37

26:                                               ; preds = %13
  %27 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %28 = extractvalue { ptr, i32 } %27, 1
  %29 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %30 = icmp eq i32 %28, %29
  br i1 %30, label %34, label %31

31:                                               ; preds = %26
  %32 = extractvalue { ptr, i32 } %27, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  invoke void @__cxa_rethrow()
          to label %62 unwind label %endcatch.pad

34:                                               ; preds = %26
  %35 = extractvalue { ptr, i32 } %27, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  store ptr %36, ptr %3, align 8
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %37

37:                                               ; preds = %25, %34
  %38 = load ptr, ptr %7, align 8
  %39 = icmp eq ptr %38, null
  %40 = select i1 %39, ptr @__ts_empty_array, ptr %38
  %41 = getelementptr { ptr, i64, i64 }, ptr %40, i32 0, i32 1
  %42 = load i64, ptr %41, align 8
  %43 = load ptr, ptr %5, align 8
  %44 = icmp eq ptr %43, null
  %45 = select i1 %44, ptr @__ts_empty_array, ptr %43
  %46 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 1
  %47 = load i64, ptr %46, align 8
  %48 = add i64 %42, %47
  %49 = load i1, ptr %4, align 1
  br i1 %49, label %50, label %51

50:                                               ; preds = %37
  br label %52

51:                                               ; preds = %37
  br label %52

52:                                               ; preds = %50, %51
  %53 = phi i32 [ 0, %51 ], [ 10, %50 ]
  %54 = sext i32 %53 to i64
  %55 = add i64 %48, %54
  %56 = trunc i64 %55 to i32
  store i32 %56, ptr %9, align 4
  %57 = load i32, ptr %9, align 4
  ret i32 %57

endcatch.pad:                                     ; preds = %31
  %58 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

59:                                               ; preds = %11, %1
  %60 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %59
  %61 = phi { ptr, i32 } [ %60, %59 ], [ %58, %endcatch.pad ]
  resume { ptr, i32 } %61

62:                                               ; preds = %31
  unreachable
}

; Function Attrs: mustprogress
define i32 @awaitBesideCatch(i1 %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i1, align 1
  %8 = alloca i32, align 4
  store i1 %0, ptr %7, align 1
  store ptr null, ptr %6, align 8
  store ptr null, ptr %5, align 8
  %9 = invoke ptr @make(i32 1)
          to label %10 unwind label %55

10:                                               ; preds = %1
  store ptr %9, ptr %6, align 8
  store ptr null, ptr %4, align 8
  %11 = call ptr @async_execute_fn(ptr %4)
  call void @mlirAsyncRuntimeAwaitToken(ptr %11)
  %12 = call i1 @mlirAsyncRuntimeIsTokenError(ptr %11)
  call void @mlirAsyncRuntimeDropRef(ptr %11, i64 1)
  %13 = xor i1 %12, true
  br i1 %13, label %14, label %58

14:                                               ; preds = %10
  %15 = load ptr, ptr %4, align 8
  store ptr %15, ptr %5, align 8
  store i1 false, ptr %3, align 1
  %16 = load ptr, ptr %6, align 8
  %17 = load i1, ptr %7, align 1
  %18 = invoke i32 @check(ptr %16, i1 %17)
          to label %19 unwind label %20

19:                                               ; preds = %14
  br label %31

20:                                               ; preds = %14
  %21 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %22 = extractvalue { ptr, i32 } %21, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %24 = icmp eq i32 %22, %23
  br i1 %24, label %28, label %25

25:                                               ; preds = %20
  %26 = extractvalue { ptr, i32 } %21, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  invoke void @__cxa_rethrow()
          to label %58 unwind label %endcatch.pad

28:                                               ; preds = %20
  %29 = extractvalue { ptr, i32 } %21, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  store ptr %30, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %31

31:                                               ; preds = %19, %28
  %32 = load ptr, ptr %5, align 8
  %33 = icmp eq ptr %32, null
  %34 = select i1 %33, ptr @__ts_empty_array, ptr %32
  %35 = getelementptr { ptr, i64, i64 }, ptr %34, i32 0, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr i32, ptr %36, i32 0
  %38 = load i32, ptr %37, align 4
  %39 = load ptr, ptr %5, align 8
  %40 = icmp eq ptr %39, null
  %41 = select i1 %40, ptr @__ts_empty_array, ptr %39
  %42 = getelementptr { ptr, i64, i64 }, ptr %41, i32 0, i32 0
  %43 = load ptr, ptr %42, align 8
  %44 = getelementptr i32, ptr %43, i32 1
  %45 = load i32, ptr %44, align 4
  %46 = add i32 %38, %45
  %47 = load i1, ptr %3, align 1
  br i1 %47, label %48, label %49

48:                                               ; preds = %31
  br label %50

49:                                               ; preds = %31
  br label %50

50:                                               ; preds = %48, %49
  %51 = phi i32 [ 0, %49 ], [ 100, %48 ]
  %52 = add i32 %46, %51
  store i32 %52, ptr %8, align 4
  %53 = load i32, ptr %8, align 4
  ret i32 %53

endcatch.pad:                                     ; preds = %25
  %54 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

55:                                               ; preds = %1
  %56 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %55
  %57 = phi { ptr, i32 } [ %56, %55 ], [ %54, %endcatch.pad ]
  resume { ptr, i32 } %57

58:                                               ; preds = %25, %10
  call void @puts(ptr @assert_msg)
  call void @abort()
  unreachable
}

; Function Attrs: mustprogress
define i32 @awaitAllBesideCatch(i1 %0) #4 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i1, align 1
  %9 = alloca i32, align 4
  store i1 %0, ptr %8, align 1
  store ptr null, ptr %7, align 8
  store ptr null, ptr %6, align 8
  %10 = invoke i32 @twice(i32 2)
          to label %11 unwind label %82

11:                                               ; preds = %1
  %12 = invoke i32 @twice(i32 5)
          to label %13 unwind label %82

13:                                               ; preds = %11
  %14 = call ptr @GC_malloc(i64 16)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i32, ptr %15, i64 0
  store i32 %10, ptr %16, align 4
  %17 = getelementptr i32, ptr %15, i64 1
  store i32 %12, ptr %17, align 4
  %18 = call ptr @GC_malloc(i64 32)
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 0
  store ptr %15, ptr %20, align 8
  %21 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 1
  store i64 2, ptr %21, align 8
  %22 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 2
  store i64 2, ptr %22, align 8
  store ptr %19, ptr %7, align 8
  store ptr null, ptr %5, align 8
  %23 = call ptr @async_execute_fn_0(ptr %5)
  call void @mlirAsyncRuntimeAwaitToken(ptr %23)
  %24 = call i1 @mlirAsyncRuntimeIsTokenError(ptr %23)
  call void @mlirAsyncRuntimeDropRef(ptr %23, i64 1)
  %25 = xor i1 %24, true
  br i1 %25, label %26, label %85

26:                                               ; preds = %13
  %27 = load ptr, ptr %5, align 8
  store ptr %27, ptr %6, align 8
  store i1 false, ptr %4, align 1
  store ptr null, ptr %2, align 8
  %28 = call ptr @GC_malloc(i64 8)
  %29 = getelementptr i8, ptr %28, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %29, ptr @a_0_5295668518244173888, i64 0, i1 false)
  %30 = call ptr @GC_malloc(i64 32)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr { ptr, i64, i64 }, ptr %31, i32 0, i32 0
  store ptr %29, ptr %32, align 8
  %33 = getelementptr { ptr, i64, i64 }, ptr %31, i32 0, i32 1
  store i64 0, ptr %33, align 8
  %34 = getelementptr { ptr, i64, i64 }, ptr %31, i32 0, i32 2
  store i64 0, ptr %34, align 8
  store ptr %31, ptr %2, align 8
  %35 = load ptr, ptr %2, align 8
  %36 = load i1, ptr %8, align 1
  %37 = invoke i32 @check(ptr %35, i1 %36)
          to label %38 unwind label %39

38:                                               ; preds = %26
  br label %50

39:                                               ; preds = %26
  %40 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %41 = extractvalue { ptr, i32 } %40, 1
  %42 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %43 = icmp eq i32 %41, %42
  br i1 %43, label %47, label %44

44:                                               ; preds = %39
  %45 = extractvalue { ptr, i32 } %40, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  invoke void @__cxa_rethrow()
          to label %85 unwind label %endcatch.pad

47:                                               ; preds = %39
  %48 = extractvalue { ptr, i32 } %40, 0
  %49 = call ptr @__cxa_begin_catch(ptr %48)
  store ptr %49, ptr %3, align 8
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %50

50:                                               ; preds = %38, %47
  %51 = load ptr, ptr %7, align 8
  %52 = icmp eq ptr %51, null
  %53 = select i1 %52, ptr @__ts_empty_array, ptr %51
  %54 = getelementptr { ptr, i64, i64 }, ptr %53, i32 0, i32 0
  %55 = load ptr, ptr %54, align 8
  %56 = getelementptr i32, ptr %55, i32 0
  %57 = load i32, ptr %56, align 4
  %58 = load ptr, ptr %6, align 8
  %59 = icmp eq ptr %58, null
  %60 = select i1 %59, ptr @__ts_empty_array, ptr %58
  %61 = getelementptr { ptr, i64, i64 }, ptr %60, i32 0, i32 0
  %62 = load ptr, ptr %61, align 8
  %63 = getelementptr i32, ptr %62, i32 0
  %64 = load i32, ptr %63, align 4
  %65 = add i32 %57, %64
  %66 = load ptr, ptr %6, align 8
  %67 = icmp eq ptr %66, null
  %68 = select i1 %67, ptr @__ts_empty_array, ptr %66
  %69 = getelementptr { ptr, i64, i64 }, ptr %68, i32 0, i32 0
  %70 = load ptr, ptr %69, align 8
  %71 = getelementptr i32, ptr %70, i32 1
  %72 = load i32, ptr %71, align 4
  %73 = add i32 %65, %72
  %74 = load i1, ptr %4, align 1
  br i1 %74, label %75, label %76

75:                                               ; preds = %50
  br label %77

76:                                               ; preds = %50
  br label %77

77:                                               ; preds = %75, %76
  %78 = phi i32 [ 0, %76 ], [ 100, %75 ]
  %79 = add i32 %73, %78
  store i32 %79, ptr %9, align 4
  %80 = load i32, ptr %9, align 4
  ret i32 %80

endcatch.pad:                                     ; preds = %44
  %81 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

82:                                               ; preds = %11, %1
  %83 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %82
  %84 = phi { ptr, i32 } [ %83, %82 ], [ %81, %endcatch.pad ]
  resume { ptr, i32 } %84

85:                                               ; preds = %44, %13
  call void @puts(ptr @assert_msg_0)
  call void @abort()
  unreachable
}

; Function Attrs: mustprogress
define i32 @moveThenThrow() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store ptr null, ptr %4, align 8
  store ptr null, ptr %3, align 8
  %6 = invoke ptr @make(i32 4)
          to label %7 unwind label %34

7:                                                ; preds = %0
  store ptr %6, ptr %4, align 8
  %8 = load ptr, ptr %4, align 8
  store ptr %8, ptr %3, align 8
  store i1 false, ptr %2, align 1
  %9 = load ptr, ptr %3, align 8
  %10 = invoke i32 @check(ptr %9, i1 false)
          to label %11 unwind label %12

11:                                               ; preds = %7
  br label %23

12:                                               ; preds = %7
  %13 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 1
  %15 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %16 = icmp eq i32 %14, %15
  br i1 %16, label %20, label %17

17:                                               ; preds = %12
  %18 = extractvalue { ptr, i32 } %13, 0
  %19 = call ptr @__cxa_begin_catch(ptr %18)
  invoke void @__cxa_rethrow()
          to label %37 unwind label %endcatch.pad

20:                                               ; preds = %12
  %21 = extractvalue { ptr, i32 } %13, 0
  %22 = call ptr @__cxa_begin_catch(ptr %21)
  store ptr %22, ptr %1, align 8
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  br label %23

23:                                               ; preds = %11, %20
  %24 = load ptr, ptr %3, align 8
  %25 = invoke i32 @check(ptr %24, i1 true)
          to label %26 unwind label %34

26:                                               ; preds = %23
  %27 = load i1, ptr %2, align 1
  br i1 %27, label %28, label %29

28:                                               ; preds = %26
  br label %30

29:                                               ; preds = %26
  br label %30

30:                                               ; preds = %28, %29
  %31 = phi i32 [ 0, %29 ], [ 1, %28 ]
  store i32 %31, ptr %5, align 4
  %32 = load i32, ptr %5, align 4
  ret i32 %32

endcatch.pad:                                     ; preds = %17
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

34:                                               ; preds = %23, %0
  %35 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %33, %endcatch.pad ]
  resume { ptr, i32 } %36

37:                                               ; preds = %17
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #4 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %75, %0
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 1000
  br i1 %6, label %7, label %82

7:                                                ; preds = %4
  %8 = invoke i32 @inTry(i1 false, i1 false)
          to label %9 unwind label %79

9:                                                ; preds = %7
  %10 = icmp eq i32 %8, 2
  br i1 %10, label %11, label %84

11:                                               ; preds = %9
  %12 = invoke i32 @inTry(i1 true, i1 false)
          to label %13 unwind label %79

13:                                               ; preds = %11
  %14 = icmp eq i32 %12, -1
  br i1 %14, label %15, label %86

15:                                               ; preds = %13
  %16 = invoke i32 @inTry(i1 false, i1 true)
          to label %17 unwind label %79

17:                                               ; preds = %15
  %18 = icmp eq i32 %16, -1
  br i1 %18, label %19, label %88

19:                                               ; preds = %17
  %20 = invoke i32 @besideCatch(i1 false)
          to label %21 unwind label %79

21:                                               ; preds = %19
  %22 = icmp eq i32 %20, 4
  br i1 %22, label %23, label %90

23:                                               ; preds = %21
  %24 = invoke i32 @besideCatch(i1 true)
          to label %25 unwind label %79

25:                                               ; preds = %23
  %26 = icmp eq i32 %24, 14
  br i1 %26, label %27, label %92

27:                                               ; preds = %25
  %28 = invoke i32 @awaitBesideCatch(i1 false)
          to label %29 unwind label %79

29:                                               ; preds = %27
  %30 = icmp eq i32 %28, 9
  br i1 %30, label %31, label %94

31:                                               ; preds = %29
  %32 = invoke i32 @awaitBesideCatch(i1 true)
          to label %33 unwind label %79

33:                                               ; preds = %31
  %34 = icmp eq i32 %32, 109
  br i1 %34, label %35, label %96

35:                                               ; preds = %33
  %36 = invoke i32 @awaitAllBesideCatch(i1 false)
          to label %37 unwind label %79

37:                                               ; preds = %35
  %38 = icmp eq i32 %36, 18
  br i1 %38, label %39, label %98

39:                                               ; preds = %37
  %40 = invoke i32 @awaitAllBesideCatch(i1 true)
          to label %41 unwind label %79

41:                                               ; preds = %39
  %42 = icmp eq i32 %40, 118
  br i1 %42, label %43, label %100

43:                                               ; preds = %41
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %2, align 8
  %44 = invoke i32 @moveThenThrow()
          to label %45 unwind label %46

45:                                               ; preds = %43
  br label %60

46:                                               ; preds = %43
  %47 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %48 = extractvalue { ptr, i32 } %47, 1
  %49 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %50 = icmp eq i32 %48, %49
  br i1 %50, label %54, label %51

51:                                               ; preds = %46
  %52 = extractvalue { ptr, i32 } %47, 0
  %53 = call ptr @__cxa_begin_catch(ptr %52)
  invoke void @__cxa_rethrow()
          to label %83 unwind label %endcatch.pad

54:                                               ; preds = %46
  %55 = extractvalue { ptr, i32 } %47, 0
  %56 = call ptr @__cxa_begin_catch(ptr %55)
  store ptr %56, ptr %1, align 8
  %57 = load ptr, ptr %1, align 8
  %58 = getelementptr %E, ptr %57, i32 0, i32 1
  %59 = load ptr, ptr %58, align 8
  store ptr %59, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %60

60:                                               ; preds = %45, %54
  %61 = load ptr, ptr %2, align 8
  %62 = ptrtoint ptr %61 to i64
  %63 = icmp ne i64 %62, 0
  %64 = zext i1 %63 to i32
  %65 = and i32 %64, 1
  %66 = icmp ne i32 %65, 0
  br i1 %66, label %67, label %70

67:                                               ; preds = %60
  %68 = call i32 @strcmp(ptr %61, ptr getelementptr inbounds nuw (i8, ptr @s_9443188008722762343, i64 8))
  %69 = icmp eq i32 %68, 0
  br label %72

70:                                               ; preds = %60
  %71 = icmp eq i64 %62, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9443188008722762343, i64 8) to i64)
  br label %72

72:                                               ; preds = %67, %70
  %73 = phi i1 [ %71, %70 ], [ %69, %67 ]
  br label %74

74:                                               ; preds = %72
  br i1 %73, label %75, label %102

75:                                               ; preds = %74
  %76 = load i32, ptr %3, align 4
  %77 = add i32 %76, 1
  store i32 %77, ptr %3, align 4
  br label %4

endcatch.pad:                                     ; preds = %51
  %78 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

79:                                               ; preds = %39, %35, %31, %27, %23, %19, %15, %11, %7
  %80 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %79
  %81 = phi { ptr, i32 } [ %80, %79 ], [ %78, %endcatch.pad ]
  resume { ptr, i32 } %81

82:                                               ; preds = %4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

83:                                               ; preds = %102, %51, %100, %98, %96, %94, %92, %90, %88, %86, %84
  unreachable

84:                                               ; preds = %9
  %85 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18025199179193165964, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 118, ptr null)
  br label %83

86:                                               ; preds = %13
  %87 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10559759729922936440, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 119, ptr null)
  br label %83

88:                                               ; preds = %17
  %89 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7625973300497561953, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 120, ptr null)
  br label %83

90:                                               ; preds = %21
  %91 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17904541597739498615, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 122, ptr null)
  br label %83

92:                                               ; preds = %25
  %93 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15907365327259447507, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 123, ptr null)
  br label %83

94:                                               ; preds = %29
  %95 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11381653347708366518, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 125, ptr null)
  br label %83

96:                                               ; preds = %33
  %97 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17996328347166339198, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 126, ptr null)
  br label %83

98:                                               ; preds = %37
  %99 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14459584306309247786, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 128, ptr null)
  br label %83

100:                                              ; preds = %41
  %101 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7781056549463363810, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 129, ptr null)
  br label %83

102:                                              ; preds = %74
  %103 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3181862927481854341, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3734525892634768032, i64 8), i32 138, ptr null)
  br label %83
}

define ptr @async_execute_fn(ptr %0) {
AllocaSpillBB:
  %1 = call ptr @mlirAsyncRuntimeCreateToken()
  %2 = add i64 40, 8
  %3 = sub i64 %2, 1
  %4 = sub i64 0, 8
  %5 = and i64 %3, %4
  %6 = call ptr @GC_malloc_uncollectable(i64 %5)
  %resume.addr = getelementptr inbounds nuw %async_execute_fn.Frame, ptr %6, i32 0, i32 0
  store ptr @async_execute_fn.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %async_execute_fn.Frame, ptr %6, i32 0, i32 1
  store ptr @async_execute_fn.destroy, ptr %destroy.addr, align 8
  %.spill.addr1 = getelementptr inbounds %async_execute_fn.Frame, ptr %6, i32 0, i32 3
  store ptr %1, ptr %.spill.addr1, align 8
  %.spill.addr = getelementptr inbounds %async_execute_fn.Frame, ptr %6, i32 0, i32 2
  store ptr %0, ptr %.spill.addr, align 8
  %index.addr4 = getelementptr inbounds nuw %async_execute_fn.Frame, ptr %6, i32 0, i32 4
  store i1 false, ptr %index.addr4, align 1
  call void @mlirAsyncRuntimeExecute(ptr %6, ptr @__resume)
  ret ptr %1
}

define ptr @async_execute_fn_0(ptr %0) {
AllocaSpillBB:
  %1 = call ptr @mlirAsyncRuntimeCreateToken()
  %2 = add i64 40, 8
  %3 = sub i64 %2, 1
  %4 = sub i64 0, 8
  %5 = and i64 %3, %4
  %6 = call ptr @GC_malloc_uncollectable(i64 %5)
  %resume.addr = getelementptr inbounds nuw %async_execute_fn_0.Frame, ptr %6, i32 0, i32 0
  store ptr @async_execute_fn_0.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %async_execute_fn_0.Frame, ptr %6, i32 0, i32 1
  store ptr @async_execute_fn_0.destroy, ptr %destroy.addr, align 8
  %.spill.addr1 = getelementptr inbounds %async_execute_fn_0.Frame, ptr %6, i32 0, i32 3
  store ptr %1, ptr %.spill.addr1, align 8
  %.spill.addr = getelementptr inbounds %async_execute_fn_0.Frame, ptr %6, i32 0, i32 2
  store ptr %0, ptr %.spill.addr, align 8
  %index.addr4 = getelementptr inbounds nuw %async_execute_fn_0.Frame, ptr %6, i32 0, i32 4
  store i1 false, ptr %index.addr4, align 1
  call void @mlirAsyncRuntimeExecute(ptr %6, ptr @__resume)
  ret ptr %1
}

declare void @mlirAsyncRuntimeDropRef(ptr, i64)

declare ptr @mlirAsyncRuntimeCreateToken()

declare void @mlirAsyncRuntimeEmplaceToken(ptr)

declare i1 @mlirAsyncRuntimeIsTokenError(ptr)

declare void @mlirAsyncRuntimeAwaitToken(ptr)

declare void @mlirAsyncRuntimeExecute(ptr, ptr)

define void @__resume(ptr %0) {
  %2 = getelementptr inbounds { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  call fastcc void %3(ptr %0)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

define internal fastcc void @async_execute_fn.resume(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.resume:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.resume
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 0 to i32
  switch i32 %1, label %5 [
    i32 0, label %2
    i32 1, label %4
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @pair(i32 3)
  store ptr %3, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %5

4:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %5

5:                                                ; preds = %4, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %5
  ret void
}

define internal fastcc void @async_execute_fn.destroy(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.destroy:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.destroy
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 1 to i32
  switch i32 %1, label %5 [
    i32 0, label %2
    i32 1, label %4
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @pair(i32 3)
  store ptr %3, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %5

4:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %5

5:                                                ; preds = %4, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %5
  ret void
}

define internal fastcc void @async_execute_fn_0.resume(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.resume:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.resume
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 0 to i32
  switch i32 %1, label %16 [
    i32 0, label %2
    i32 1, label %15
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call i32 @twice(i32 3)
  %4 = call i32 @twice(i32 4)
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr i32, ptr %6, i64 0
  store i32 %3, ptr %7, align 4
  %8 = getelementptr i32, ptr %6, i64 1
  store i32 %4, ptr %8, align 4
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  store ptr %6, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  store i64 2, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 2
  store i64 2, ptr %13, align 8
  %14 = call ptr @all(ptr %10)
  store ptr %14, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %16

15:                                               ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %16

16:                                               ; preds = %15, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %16
  ret void
}

define internal fastcc void @async_execute_fn_0.destroy(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.destroy:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.destroy
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 1 to i32
  switch i32 %1, label %16 [
    i32 0, label %2
    i32 1, label %15
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call i32 @twice(i32 3)
  %4 = call i32 @twice(i32 4)
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr i32, ptr %6, i64 0
  store i32 %3, ptr %7, align 4
  %8 = getelementptr i32, ptr %6, i64 1
  store i32 %4, ptr %8, align 4
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  store ptr %6, ptr %11, align 8
  %12 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  store i64 2, ptr %12, align 8
  %13 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 2
  store i64 2, ptr %13, align 8
  %14 = call ptr @all(ptr %10)
  store ptr %14, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %16

15:                                               ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %16

16:                                               ; preds = %15, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %16
  ret void
}

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress }
attributes #4 = { mustprogress "personality" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

