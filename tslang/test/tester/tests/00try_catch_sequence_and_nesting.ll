; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%A = type { ptr, i32 }
%B = type { ptr, i32 }
%Base = type { ptr, i32 }
%Derived = type { %Base, i32 }

$A..size = comdat any

$B..size = comdat any

$Base..size = comdat any

$Derived..size = comdat any

$_ZTSP1A = comdat any

$_ZTIP1A = comdat any

$_ZTS1A = comdat any

$_ZTI1A = comdat any

$_ZTSP1B = comdat any

$_ZTIP1B = comdat any

$_ZTS1B = comdat any

$_ZTI1B = comdat any

$_ZTSP7Derived = comdat any

$_ZTIP7Derived = comdat any

$_ZTS7Derived = comdat any

$_ZTI7Derived = comdat any

$_ZTS4Base = comdat any

$_ZTI4Base = comdat any

$_ZTSP4Base = comdat any

$_ZTIP4Base = comdat any

@s_12638222384927744748 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFA\00", align 8
@s_12638225683462629381 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFB\00", align 8
@s_13953526755221441880 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBase\00", align 8
@s_4179427500551276790 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFDerived\00", align 8
@A..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8)
@A..size = global i64 16, comdat
@A..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @A..instanceOf, ptr @A..new, ptr @A..rtti, ptr @A..size }
@B..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8)
@B..size = global i64 16, comdat
@B..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @B..instanceOf, ptr @B..new, ptr @B..rtti, ptr @B..size }
@Base..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_13953526755221441880, i64 8)
@Base..size = global i64 16, comdat
@Base..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Base..instanceOf, ptr @Base..new, ptr @Base..rtti, ptr @Base..size }
@Derived..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_4179427500551276790, i64 8)
@Derived..size = global i64 24, comdat
@Derived..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Derived..instanceOf, ptr @Derived..new, ptr @Base..rtti, ptr @Base..size, ptr @Derived..rtti, ptr @Derived..size }
@_ZTIi = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP1A = linkonce_odr constant [4 x i8] c"P1A\00", comdat
@_ZTIP1A = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP1A, i32 1073741824, ptr @_ZTI1A, ptr @.eh.copy.box.P1A }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00", comdat
@_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1A }, comdat
@_ZTSP1B = linkonce_odr constant [4 x i8] c"P1B\00", comdat
@_ZTIP1B = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP1B, i32 1073741824, ptr @_ZTI1B, ptr @.eh.copy.box.P1B }, comdat
@_ZTS1B = linkonce_odr constant [3 x i8] c"1B\00", comdat
@_ZTI1B = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1B }, comdat
@_ZTSP7Derived = linkonce_odr constant [10 x i8] c"P7Derived\00", comdat
@_ZTIP7Derived = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP7Derived, i32 1073741824, ptr @_ZTI7Derived, ptr @.eh.copy.box.P7Derived }, comdat
@_ZTVN10__cxxabiv120__si_class_type_infoE = external constant ptr
@_ZTS7Derived = linkonce_odr constant [9 x i8] c"7Derived\00", comdat
@_ZTI7Derived = linkonce_odr constant { ptr, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTS7Derived, ptr @_ZTI4Base }, comdat
@_ZTS4Base = linkonce_odr constant [6 x i8] c"4Base\00", comdat
@_ZTI4Base = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS4Base }, comdat
@_ZTSP4Base = linkonce_odr constant [7 x i8] c"P4Base\00", comdat
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_3543526056949381152 = internal constant [75 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery catch in a sequence of try/catch statements runs and returns\00", align 8
@f_13634195528786589160 = internal constant [122 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch_sequence_and_nesting.ts\00", align 8
@m_5991587839298454140 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFeach thrown class reaches its own typed catch\00", align 8
@m_16776048796568430220 = internal constant [81 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested catch that does not match leaves the exception to the outer try\00", align 8
@m_321027830215491700 = internal constant [87 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested class catch that does not match leaves the exception to the outer try\00", align 8
@m_8809383550796196472 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested catch that matches keeps the exception\00", align 8
@m_9543189600576715580 = internal constant [70 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan exception passes two non-matching catches to reach a third\00", align 8
@m_12619662736874877513 = internal constant [65 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa catch that does not match still runs its try's finally\00", align 8
@m_3021526374194178609 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe finally runs before the caller's catch\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIP4Base = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP4Base, i32 1073741824, ptr @_ZTI4Base, ptr @.eh.copy.box.P4Base }, comdat
@_ZTIPc = external constant ptr
@trail = private global i32 0
@td_4117155613450509001_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_4117155613450509001, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_781263649003409679_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_781263649003409679, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_856374508671476337_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_856374508671476337, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_3176987943180271046_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_3176987943180271046, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

define internal void @tsrel_3176987943180271046(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %12

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %12

6:                                                ; preds = %4
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, @Derived..vtbl
  br i1 %8, label %9, label %10

9:                                                ; preds = %6
  br label %11

10:                                               ; preds = %6
  br label %11

11:                                               ; preds = %9, %10
  call void @__tslang_free_block(ptr %2)
  br label %12

12:                                               ; preds = %11, %4, %1
  ret void
}

define internal void @tsrel_856374508671476337(ptr %0) {
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

define internal void @tsrel_781263649003409679(ptr %0) {
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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

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

define internal void @tsrel_4117155613450509001(ptr %0) {
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

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define void @A.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %A, ptr %3, i32 0, i32 1
  store i32 1, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @A..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @A..rtti, align 8
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
define ptr @A..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %A, ptr %3, i32 0, i32 0
  store ptr @A..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @B.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %B, ptr %3, i32 0, i32 1
  store i32 2, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @B..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @B..rtti, align 8
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
define ptr @B..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %B, ptr %3, i32 0, i32 0
  store ptr @B..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Base.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Base, ptr %3, i32 0, i32 1
  store i32 3, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Base..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Base..rtti, align 8
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
define ptr @Base..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Base, ptr %3, i32 0, i32 0
  store ptr @Base..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Derived.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Derived, ptr %3, i32 0, i32 0
  call void @Base.constructor(ptr %4)
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Derived, ptr %5, i32 0, i32 1
  store i32 4, ptr %6, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Derived..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @Derived..rtti, align 8
  %8 = ptrtoint ptr %6 to i64
  %9 = ptrtoint ptr %7 to i64
  %10 = icmp ne i64 %8, 0
  %11 = icmp ne i64 %9, 0
  %12 = zext i1 %10 to i32
  %13 = zext i1 %11 to i32
  %14 = and i32 %12, %13
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %2
  %17 = call i32 @strcmp(ptr %6, ptr %7)
  %18 = icmp eq i32 %17, 0
  br label %21

19:                                               ; preds = %2
  %20 = icmp eq i64 %8, %9
  br label %21

21:                                               ; preds = %16, %19
  %22 = phi i1 [ %20, %19 ], [ %18, %16 ]
  br label %23

23:                                               ; preds = %21
  br i1 %22, label %29, label %24

24:                                               ; preds = %23
  %25 = load ptr, ptr %4, align 8
  %26 = getelementptr %Derived, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %3, align 8
  %28 = call i1 @Base..instanceOf(ptr %26, ptr %27)
  br label %29

29:                                               ; preds = %24, %23
  %30 = phi i1 [ %28, %24 ], [ %22, %23 ]
  store i1 %30, ptr %5, align 1
  %31 = load i1, ptr %5, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define ptr @Derived..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Derived, ptr %3, i32 0, i32 0
  %5 = getelementptr %Base, ptr %4, i32 0, i32 0
  store ptr @Derived..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define i32 @one() #2 {
  %1 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

; Function Attrs: mustprogress
define i32 @sequentialTryWithCall() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIi, ptr null)
          to label %60 unwind label %8

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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %13
  br label %60

16:                                               ; preds = %8
  %17 = extractvalue { ptr, i32 } %9, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = load i32, ptr %18, align 4
  store i32 %19, ptr %4, align 4
  %20 = load i32, ptr %5, align 4
  %21 = add i32 %20, 1
  store i32 %21, ptr %5, align 4
  call void @__cxa_end_catch()
  %22 = invoke i32 @one()
          to label %23 unwind label %25

23:                                               ; preds = %16
  %24 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %24, align 4
  invoke void @__cxa_throw(ptr %24, ptr @_ZTIi, ptr null)
          to label %60 unwind label %25

25:                                               ; preds = %23, %16
  %26 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %27 = extractvalue { ptr, i32 } %26, 1
  %28 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %29 = icmp eq i32 %27, %28
  br i1 %29, label %33, label %30

30:                                               ; preds = %25
  %31 = extractvalue { ptr, i32 } %26, 0
  %32 = call ptr @__cxa_begin_catch(ptr %31)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %30
  br label %60

33:                                               ; preds = %25
  %34 = extractvalue { ptr, i32 } %26, 0
  %35 = call ptr @__cxa_begin_catch(ptr %34)
  %36 = load i32, ptr %35, align 4
  store i32 %36, ptr %3, align 4
  %37 = load i32, ptr %5, align 4
  %38 = add i32 %37, 10
  store i32 %38, ptr %5, align 4
  call void @__cxa_end_catch()
  %39 = call ptr @GC_malloc(i64 24)
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = getelementptr %A, ptr %40, i32 0, i32 0
  store ptr @A..vtbl, ptr %41, align 8
  invoke void @A.constructor(ptr %40)
          to label %42 unwind label %44

42:                                               ; preds = %33
  store ptr %40, ptr %1, align 8
  %43 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 3, ptr %43, align 4
  invoke void @__cxa_throw(ptr %43, ptr @_ZTIi, ptr null)
          to label %60 unwind label %44

44:                                               ; preds = %42, %33
  %45 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %46 = extractvalue { ptr, i32 } %45, 1
  %47 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %48 = icmp eq i32 %46, %47
  br i1 %48, label %52, label %49

49:                                               ; preds = %44
  %50 = extractvalue { ptr, i32 } %45, 0
  %51 = call ptr @__cxa_begin_catch(ptr %50)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %49
  br label %60

52:                                               ; preds = %44
  %53 = extractvalue { ptr, i32 } %45, 0
  %54 = call ptr @__cxa_begin_catch(ptr %53)
  %55 = load i32, ptr %54, align 4
  store i32 %55, ptr %2, align 4
  %56 = load i32, ptr %5, align 4
  %57 = add i32 %56, 100
  store i32 %57, ptr %5, align 4
  call void @__cxa_end_catch()
  %58 = load i32, ptr %5, align 4
  store i32 %58, ptr %6, align 4
  %59 = load i32, ptr %6, align 4
  ret i32 %59

60:                                               ; preds = %.noexc, %42, %.noexc1, %23, %.noexc2, %0
  unreachable

endcatch.resume:                                  ; preds = %13, %30, %49
  %61 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %61
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P1A(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4117155613450509001_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P1B(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_781263649003409679_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P7Derived(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_856374508671476337_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P4Base(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_3176987943180271046_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define i32 @twoClassesThrown() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %A, ptr %7, i32 0, i32 0
  store ptr @A..vtbl, ptr %8, align 8
  invoke void @A.constructor(ptr %7)
          to label %9 unwind label %11

9:                                                ; preds = %0
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %7, ptr %10, align 8
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIP1A, ptr null)
          to label %74 unwind label %11

11:                                               ; preds = %9, %0
  %12 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1A
  %13 = extractvalue { ptr, i32 } %12, 1
  %14 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1A)
  %15 = icmp eq i32 %13, %14
  br i1 %15, label %19, label %16

16:                                               ; preds = %11
  %17 = extractvalue { ptr, i32 } %12, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  br label %74

19:                                               ; preds = %11
  %20 = extractvalue { ptr, i32 } %12, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  store ptr %21, ptr %3, align 8
  %22 = load i32, ptr %4, align 4
  %23 = load ptr, ptr %3, align 8
  %24 = getelementptr %A, ptr %23, i32 0, i32 1
  %25 = load i32, ptr %24, align 4
  %26 = add i32 %22, %25
  store i32 %26, ptr %4, align 4
  call void @__cxa_end_catch()
  %27 = call ptr @GC_malloc(i64 24)
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr %B, ptr %28, i32 0, i32 0
  store ptr @B..vtbl, ptr %29, align 8
  invoke void @B.constructor(ptr %28)
          to label %30 unwind label %32

30:                                               ; preds = %19
  %31 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %28, ptr %31, align 8
  invoke void @__cxa_throw(ptr %31, ptr @_ZTIP1B, ptr null)
          to label %74 unwind label %32

32:                                               ; preds = %30, %19
  %33 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1B
  %34 = extractvalue { ptr, i32 } %33, 1
  %35 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1B)
  %36 = icmp eq i32 %34, %35
  br i1 %36, label %40, label %37

37:                                               ; preds = %32
  %38 = extractvalue { ptr, i32 } %33, 0
  %39 = call ptr @__cxa_begin_catch(ptr %38)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %37
  br label %74

40:                                               ; preds = %32
  %41 = extractvalue { ptr, i32 } %33, 0
  %42 = call ptr @__cxa_begin_catch(ptr %41)
  store ptr %42, ptr %2, align 8
  %43 = load i32, ptr %4, align 4
  %44 = load ptr, ptr %2, align 8
  %45 = getelementptr %B, ptr %44, i32 0, i32 1
  %46 = load i32, ptr %45, align 4
  %47 = mul i32 %46, 10
  %48 = add i32 %43, %47
  store i32 %48, ptr %4, align 4
  call void @__cxa_end_catch()
  %49 = call ptr @GC_malloc(i64 32)
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr %Derived, ptr %50, i32 0, i32 0
  %52 = getelementptr %Base, ptr %51, i32 0, i32 0
  store ptr @Derived..vtbl, ptr %52, align 8
  invoke void @Derived.constructor(ptr %50)
          to label %53 unwind label %55

53:                                               ; preds = %40
  %54 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %50, ptr %54, align 8
  invoke void @__cxa_throw(ptr %54, ptr @_ZTIP7Derived, ptr null)
          to label %74 unwind label %55

55:                                               ; preds = %53, %40
  %56 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Base
  %57 = extractvalue { ptr, i32 } %56, 1
  %58 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Base)
  %59 = icmp eq i32 %57, %58
  br i1 %59, label %63, label %60

60:                                               ; preds = %55
  %61 = extractvalue { ptr, i32 } %56, 0
  %62 = call ptr @__cxa_begin_catch(ptr %61)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %60
  br label %74

63:                                               ; preds = %55
  %64 = extractvalue { ptr, i32 } %56, 0
  %65 = call ptr @__cxa_begin_catch(ptr %64)
  store ptr %65, ptr %1, align 8
  %66 = load i32, ptr %4, align 4
  %67 = load ptr, ptr %1, align 8
  %68 = getelementptr %Base, ptr %67, i32 0, i32 1
  %69 = load i32, ptr %68, align 4
  %70 = mul i32 %69, 100
  %71 = add i32 %66, %70
  store i32 %71, ptr %4, align 4
  call void @__cxa_end_catch()
  %72 = load i32, ptr %4, align 4
  store i32 %72, ptr %5, align 4
  %73 = load i32, ptr %5, align 4
  ret i32 %73

74:                                               ; preds = %.noexc1, %53, %.noexc2, %30, %.noexc, %9
  unreachable

endcatch.resume:                                  ; preds = %37, %60, %16
  %75 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %75
}

; Function Attrs: mustprogress
define i32 @nestedMismatchPrimitive() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %36 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %36 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  store ptr %16, ptr %1, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %33

endcatch.pad:                                     ; preds = %11
  %17 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

18:                                               ; No predecessors!
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %18
  %20 = phi { ptr, i32 } [ %19, %18 ], [ %17, %endcatch.pad ]
  %21 = extractvalue { ptr, i32 } %20, 1
  %22 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %27, label %24

24:                                               ; preds = %.caught
  %25 = extractvalue { ptr, i32 } %20, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %24
  br label %36

27:                                               ; preds = %.caught
  %28 = extractvalue { ptr, i32 } %20, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  %30 = load i32, ptr %29, align 4
  store i32 %30, ptr %2, align 4
  %31 = load i32, ptr %3, align 4
  %32 = add i32 %31, 10
  store i32 %32, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %33

33:                                               ; preds = %14, %27
  %34 = load i32, ptr %3, align 4
  store i32 %34, ptr %4, align 4
  %35 = load i32, ptr %4, align 4
  ret i32 %35

36:                                               ; preds = %.noexc, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %24
  %37 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %37
}

; Function Attrs: mustprogress
define i32 @nestedMismatchClass() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %B, ptr %6, i32 0, i32 0
  store ptr @B..vtbl, ptr %7, align 8
  invoke void @B.constructor(ptr %6)
          to label %8 unwind label %10

8:                                                ; preds = %0
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %6, ptr %9, align 8
  invoke void @__cxa_throw(ptr %9, ptr @_ZTIP1B, ptr null)
          to label %43 unwind label %10

10:                                               ; preds = %8, %0
  %11 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1A
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1A)
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %18, label %15

15:                                               ; preds = %10
  %16 = extractvalue { ptr, i32 } %11, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  invoke void @__cxa_rethrow()
          to label %43 unwind label %endcatch.pad

18:                                               ; preds = %10
  %19 = extractvalue { ptr, i32 } %11, 0
  %20 = call ptr @__cxa_begin_catch(ptr %19)
  store ptr %20, ptr %1, align 8
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %40

endcatch.pad:                                     ; preds = %15
  %21 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1B
  call void @__cxa_end_catch()
  br label %.caught

22:                                               ; No predecessors!
  %23 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1B
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad ]
  %25 = extractvalue { ptr, i32 } %24, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1B)
  %27 = icmp eq i32 %25, %26
  br i1 %27, label %31, label %28

28:                                               ; preds = %.caught
  %29 = extractvalue { ptr, i32 } %24, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  br label %43

31:                                               ; preds = %.caught
  %32 = extractvalue { ptr, i32 } %24, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  store ptr %33, ptr %2, align 8
  %34 = load i32, ptr %3, align 4
  %35 = load ptr, ptr %2, align 8
  %36 = getelementptr %B, ptr %35, i32 0, i32 1
  %37 = load i32, ptr %36, align 4
  %38 = mul i32 %37, 10
  %39 = add i32 %34, %38
  store i32 %39, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %40

40:                                               ; preds = %18, %31
  %41 = load i32, ptr %3, align 4
  store i32 %41, ptr %4, align 4
  %42 = load i32, ptr %4, align 4
  ret i32 %42

43:                                               ; preds = %.noexc, %15, %8
  unreachable

endcatch.resume:                                  ; preds = %28
  %44 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %44
}

; Function Attrs: mustprogress
define i32 @nestedMatchStaysInner() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %39 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %39 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %1, align 4
  %18 = load i32, ptr %3, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %36

endcatch.pad:                                     ; preds = %11
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

21:                                               ; No predecessors!
  %22 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %21
  %23 = phi { ptr, i32 } [ %22, %21 ], [ %20, %endcatch.pad ]
  %24 = extractvalue { ptr, i32 } %23, 1
  %25 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %26 = icmp eq i32 %24, %25
  br i1 %26, label %30, label %27

27:                                               ; preds = %.caught
  %28 = extractvalue { ptr, i32 } %23, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %27
  br label %39

30:                                               ; preds = %.caught
  %31 = extractvalue { ptr, i32 } %23, 0
  %32 = call ptr @__cxa_begin_catch(ptr %31)
  %33 = load i32, ptr %32, align 4
  store i32 %33, ptr %2, align 4
  %34 = load i32, ptr %3, align 4
  %35 = add i32 %34, 10
  store i32 %35, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %36

36:                                               ; preds = %14, %30
  %37 = load i32, ptr %3, align 4
  store i32 %37, ptr %4, align 4
  %38 = load i32, ptr %4, align 4
  ret i32 %38

39:                                               ; preds = %.noexc, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %27
  %40 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %40
}

; Function Attrs: mustprogress
define i32 @threeDeepMismatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %54 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = extractvalue { ptr, i32 } %8, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  invoke void @__cxa_rethrow()
          to label %54 unwind label %endcatch.pad2

15:                                               ; preds = %7
  %16 = extractvalue { ptr, i32 } %8, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  store ptr %17, ptr %1, align 8
  %18 = load i32, ptr %4, align 4
  %19 = add i32 %18, 1
  store i32 %19, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %51

endcatch.pad2:                                    ; preds = %12
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1B
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

21:                                               ; No predecessors!
  %22 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1B
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %21
  %23 = phi { ptr, i32 } [ %22, %21 ], [ %20, %endcatch.pad2 ]
  %24 = extractvalue { ptr, i32 } %23, 1
  %25 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1B)
  %26 = icmp eq i32 %24, %25
  br i1 %26, label %30, label %27

27:                                               ; preds = %.caught1
  %28 = extractvalue { ptr, i32 } %23, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  invoke void @__cxa_rethrow()
          to label %54 unwind label %endcatch.pad

30:                                               ; preds = %.caught1
  %31 = extractvalue { ptr, i32 } %23, 0
  %32 = call ptr @__cxa_begin_catch(ptr %31)
  store ptr %32, ptr %2, align 8
  %33 = load i32, ptr %4, align 4
  %34 = add i32 %33, 10
  store i32 %34, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %51

endcatch.pad:                                     ; preds = %27
  %35 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

36:                                               ; No predecessors!
  %37 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %36
  %38 = phi { ptr, i32 } [ %37, %36 ], [ %35, %endcatch.pad ]
  %39 = extractvalue { ptr, i32 } %38, 1
  %40 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %41 = icmp eq i32 %39, %40
  br i1 %41, label %45, label %42

42:                                               ; preds = %.caught
  %43 = extractvalue { ptr, i32 } %38, 0
  %44 = call ptr @__cxa_begin_catch(ptr %43)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %42
  br label %54

45:                                               ; preds = %.caught
  %46 = extractvalue { ptr, i32 } %38, 0
  %47 = call ptr @__cxa_begin_catch(ptr %46)
  %48 = load i32, ptr %47, align 4
  store i32 %48, ptr %3, align 4
  %49 = load i32, ptr %4, align 4
  %50 = add i32 %49, 100
  store i32 %50, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %51

51:                                               ; preds = %15, %30, %45
  %52 = load i32, ptr %4, align 4
  store i32 %52, ptr %5, align 4
  %53 = load i32, ptr %5, align 4
  ret i32 %53

54:                                               ; preds = %.noexc, %27, %12, %0
  unreachable

endcatch.resume:                                  ; preds = %42
  %55 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %55
}

; Function Attrs: mustprogress
define i32 @mismatchRunsOwnFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %48 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %48 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  store ptr %16, ptr %1, align 8
  %17 = load i32, ptr %3, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %3, align 4
  call void @__cxa_end_catch()
  %19 = load i32, ptr %3, align 4
  %20 = add i32 %19, 10
  store i32 %20, ptr %3, align 4
  br label %45

endcatch.pad:                                     ; preds = %11
  %21 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

22:                                               ; No predecessors!
  %23 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad ]
  %25 = extractvalue { ptr, i32 } %24, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = load i32, ptr %3, align 4
  %28 = add i32 %27, 10
  store i32 %28, ptr %3, align 4
  invoke void @__cxa_rethrow()
          to label %48 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %.caught
  %29 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught1

30:                                               ; No predecessors!
  %31 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %30
  %32 = phi { ptr, i32 } [ %31, %30 ], [ %29, %endcatch.pad2 ]
  %33 = extractvalue { ptr, i32 } %32, 1
  %34 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %35 = icmp eq i32 %33, %34
  br i1 %35, label %39, label %36

36:                                               ; preds = %.caught1
  %37 = extractvalue { ptr, i32 } %32, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %36
  br label %48

39:                                               ; preds = %.caught1
  %40 = extractvalue { ptr, i32 } %32, 0
  %41 = call ptr @__cxa_begin_catch(ptr %40)
  %42 = load i32, ptr %41, align 4
  store i32 %42, ptr %2, align 4
  %43 = load i32, ptr %3, align 4
  %44 = add i32 %43, 100
  store i32 %44, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %45

45:                                               ; preds = %14, %39
  %46 = load i32, ptr %3, align 4
  store i32 %46, ptr %4, align 4
  %47 = load i32, ptr %4, align 4
  ret i32 %47

48:                                               ; preds = %.noexc, %.caught, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %36
  %49 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %49
}

; Function Attrs: mustprogress
define void @mismatchInCalleeRunsFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %2, align 4
  invoke void @__cxa_throw(ptr %2, ptr @_ZTIi, ptr null)
          to label %27 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %5 = extractvalue { ptr, i32 } %4, 1
  %6 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %11, label %8

8:                                                ; preds = %3
  %9 = extractvalue { ptr, i32 } %4, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  invoke void @__cxa_rethrow()
          to label %27 unwind label %endcatch.pad

11:                                               ; preds = %3
  %12 = extractvalue { ptr, i32 } %4, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  store ptr %13, ptr %1, align 8
  %14 = load i32, ptr @trail, align 4
  %15 = mul i32 %14, 10
  %16 = add i32 %15, 1
  store i32 %16, ptr @trail, align 4
  call void @__cxa_end_catch()
  %17 = load i32, ptr @trail, align 4
  %18 = mul i32 %17, 10
  %19 = add i32 %18, 2
  store i32 %19, ptr @trail, align 4
  ret void

endcatch.pad:                                     ; preds = %8
  %20 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

21:                                               ; No predecessors!
  %22 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %21
  %23 = phi { ptr, i32 } [ %22, %21 ], [ %20, %endcatch.pad ]
  %24 = load i32, ptr @trail, align 4
  %25 = mul i32 %24, 10
  %26 = add i32 %25, 2
  store i32 %26, ptr @trail, align 4
  resume { ptr, i32 } %23

27:                                               ; preds = %8, %0
  unreachable
}

; Function Attrs: mustprogress
define i32 @callerCatchesAfterFinally() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr @trail, align 4
  invoke void @mismatchInCalleeRunsFinally()
          to label %3 unwind label %4

3:                                                ; preds = %0
  br label %19

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %9
  br label %22

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %1, align 4
  %16 = load i32, ptr @trail, align 4
  %17 = mul i32 %16, 10
  %18 = add i32 %17, 3
  store i32 %18, ptr @trail, align 4
  call void @__cxa_end_catch()
  br label %19

19:                                               ; preds = %3, %12
  %20 = load i32, ptr @trail, align 4
  store i32 %20, ptr %2, align 4
  %21 = load i32, ptr %2, align 4
  ret i32 %21

22:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %9
  %23 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %23
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i32 @sequentialTryWithCall()
  %2 = icmp eq i32 %1, 111
  br i1 %2, label %3, label %26

3:                                                ; preds = %0
  %4 = call i32 @twoClassesThrown()
  %5 = icmp eq i32 %4, 321
  br i1 %5, label %6, label %28

6:                                                ; preds = %3
  %7 = call i32 @nestedMismatchPrimitive()
  %8 = icmp eq i32 %7, 10
  br i1 %8, label %9, label %30

9:                                                ; preds = %6
  %10 = call i32 @nestedMismatchClass()
  %11 = icmp eq i32 %10, 20
  br i1 %11, label %12, label %32

12:                                               ; preds = %9
  %13 = call i32 @nestedMatchStaysInner()
  %14 = icmp eq i32 %13, 1
  br i1 %14, label %15, label %34

15:                                               ; preds = %12
  %16 = call i32 @threeDeepMismatch()
  %17 = icmp eq i32 %16, 100
  br i1 %17, label %18, label %36

18:                                               ; preds = %15
  %19 = call i32 @mismatchRunsOwnFinally()
  %20 = icmp eq i32 %19, 110
  br i1 %20, label %21, label %38

21:                                               ; preds = %18
  %22 = call i32 @callerCatchesAfterFinally()
  %23 = icmp eq i32 %22, 23
  br i1 %23, label %24, label %40

24:                                               ; preds = %21
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

25:                                               ; preds = %40, %38, %36, %34, %32, %30, %28, %26
  unreachable

26:                                               ; preds = %0
  %27 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3543526056949381152, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 200, ptr null)
  br label %25

28:                                               ; preds = %3
  %29 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5991587839298454140, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 201, ptr null)
  br label %25

30:                                               ; preds = %6
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16776048796568430220, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 202, ptr null)
  br label %25

32:                                               ; preds = %9
  %33 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_321027830215491700, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 203, ptr null)
  br label %25

34:                                               ; preds = %12
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8809383550796196472, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 204, ptr null)
  br label %25

36:                                               ; preds = %15
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9543189600576715580, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 205, ptr null)
  br label %25

38:                                               ; preds = %18
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12619662736874877513, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 206, ptr null)
  br label %25

40:                                               ; preds = %21
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3021526374194178609, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13634195528786589160, i64 8), i32 207, ptr null)
  br label %25
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

