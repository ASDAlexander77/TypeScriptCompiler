; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%A = type { ptr, i32 }
%B = type { ptr, i32 }

$"___cast<!ts.union<!ts.class<A, !ts.class_storage<A>>, !ts.class<B, !ts.class_storage<B>>>,!ts.boolean>" = comdat any

$A..size = comdat any

$B..size = comdat any

@s_12638222384927744748 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFA\00", align 8
@s_12638225683462629381 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFB\00", align 8
@s_7341527364769810245 = internal constant [70 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class value is null or undefined: there is no class to make\00", align 8
@s_5730213751506031948 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from union type\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@f_3100737149123232877 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00class_value_union_null.ts\00", align 8
@m_924000406299335852 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class\00", align 8
@m_2271584025657132856 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmade\00", align 8
@m_9864376297169422948 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmake(B)\00", align 8
@m_9861561547401770463 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmake(A)\00", align 8
@m_13491316165430368537 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmakeOne(A)\00", align 8
@m_14207965758675701437 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFno class: throws\00", align 8
@m_164902768108348953 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunion null\00", align 8
@m_4967424912731520900 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunion class\00", align 8
@m_3447504299588790889 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class value is true\00", align 8
@m_6609465532978911506 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispatch with null\00", align 8
@m_3938361494966664765 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdispatch with undefined\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@A..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8)
@A..size = global i64 16, comdat
@A..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @A..instanceOf, ptr @A..new, ptr @A.name, ptr @A..rtti, ptr @A..size }
@B..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8)
@B..size = global i64 16, comdat
@B..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @B..instanceOf, ptr @B..new, ptr @B.name, ptr @B..rtti, ptr @B..size }
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@td_676157629225208357_undefined = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [10 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 12, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [10 x i8] c"undefined\00" }, align 8
@td_4117155613450509001_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_4117155613450509001, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_14527407094101050897_null = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [5 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 13, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [5 x i8] c"null\00" }, align 8
@td_18357227943351561971_function = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [9 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 10, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [9 x i8] c"function\00" }, align 8
@td_17485477222143314784_function = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [9 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 10, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [9 x i8] c"function\00" }, align 8
@td_781263649003409679_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_781263649003409679, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__cxa_end_catch()

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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

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

declare void @ts.internal.eh_try(i32)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @__gxx_personality_v0(...)

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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

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
define void @A.constructor(ptr %0, { i32, i1 } %1) #2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = extractvalue { i32, i1 } %1, 1
  br i1 %5, label %6, label %8

6:                                                ; preds = %2
  %7 = extractvalue { i32, i1 } %1, 0
  br label %9

8:                                                ; preds = %2
  br label %9

9:                                                ; preds = %6, %8
  %10 = phi i32 [ 0, %8 ], [ %7, %6 ]
  store i32 %10, ptr %3, align 4
  %11 = load ptr, ptr %4, align 8
  %12 = getelementptr %A, ptr %11, i32 0, i32 1
  %13 = load i32, ptr %3, align 4
  store i32 %13, ptr %12, align 4
  ret void
}

; Function Attrs: mustprogress
define ptr @A.name(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8), ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
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
define void @B.constructor(ptr %0, { i32, i1 } %1) #2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = extractvalue { i32, i1 } %1, 1
  br i1 %5, label %6, label %8

6:                                                ; preds = %2
  %7 = extractvalue { i32, i1 } %1, 0
  br label %9

8:                                                ; preds = %2
  br label %9

9:                                                ; preds = %6, %8
  %10 = phi i32 [ 0, %8 ], [ %7, %6 ]
  store i32 %10, ptr %3, align 4
  %11 = load ptr, ptr %4, align 8
  %12 = getelementptr %B, ptr %11, i32 0, i32 1
  %13 = load i32, ptr %3, align 4
  store i32 %13, ptr %12, align 4
  ret void
}

; Function Attrs: mustprogress
define ptr @B.name(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8), ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: mustprogress
define void @.class_value.missing(ptr %0) #3 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_7341527364769810245, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define <{ ptr, <{ [1 x i64] }> }> @make(<{ ptr, <{ [4 x i64] }> }> %0, double %1) #2 {
  %3 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %4 = alloca <{ ptr, ptr }>, align 8
  %5 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %6 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %7 = alloca double, align 8
  %8 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %9 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  store <{ ptr, <{ [4 x i64] }> }> %0, ptr %8, align 1
  store double %1, ptr %7, align 8
  %10 = load <{ ptr, <{ [4 x i64] }> }>, ptr %8, align 1
  %11 = load double, ptr %7, align 8
  %12 = extractvalue <{ ptr, <{ [4 x i64] }> }> %10, 0
  %13 = icmp ne ptr %12, null
  %14 = select i1 %13, ptr %12, ptr getelementptr inbounds nuw (i8, ptr @td_676157629225208357_undefined, i64 40)
  %15 = getelementptr i8, ptr %14, i64 -40
  %16 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %15, i32 0, i32 0
  %17 = load i32, ptr %16, align 4
  %18 = icmp eq i32 %17, 13
  br i1 %18, label %19, label %20

19:                                               ; preds = %2
  call void @.class_value.missing(ptr null)
  br label %20

20:                                               ; preds = %19, %2
  store <{ ptr, <{ [4 x i64] }> }> %10, ptr %6, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %5, ptr %6, i64 40, i1 false)
  %21 = load <{ ptr, { ptr, ptr, ptr, ptr } }>, ptr %5, align 1
  %22 = extractvalue <{ ptr, { ptr, ptr, ptr, ptr } }> %21, 1
  %23 = extractvalue { ptr, ptr, ptr, ptr } %22, 1
  %24 = call ptr %23()
  %25 = extractvalue { ptr, ptr, ptr, ptr } %22, 2
  %26 = ptrtoint ptr %25 to i64
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %33

28:                                               ; preds = %20
  %29 = fptosi double %11 to i64
  %30 = trunc i64 %29 to i32
  %31 = insertvalue { i32, i1 } undef, i32 %30, 0
  %32 = insertvalue { i32, i1 } %31, i1 true, 1
  call void %25(ptr %24, { i32, i1 } %32)
  br label %33

33:                                               ; preds = %28, %20
  %34 = insertvalue <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_4117155613450509001_class, i64 40), ptr undef }>, ptr %24, 1
  store <{ ptr, ptr }> %34, ptr %4, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %3, ptr %4, i64 16, i1 false)
  %35 = load <{ ptr, <{ [1 x i64] }> }>, ptr %3, align 1
  store <{ ptr, <{ [1 x i64] }> }> %35, ptr %9, align 1
  %36 = load <{ ptr, <{ [1 x i64] }> }>, ptr %9, align 1
  ret <{ ptr, <{ [1 x i64] }> }> %36
}

; Function Attrs: mustprogress
define ptr @makeOne({ { ptr, ptr, ptr, ptr }, i1 } %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca { { ptr, ptr, ptr, ptr }, i1 }, align 8
  %5 = alloca ptr, align 8
  store { { ptr, ptr, ptr, ptr }, i1 } %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load { { ptr, ptr, ptr, ptr }, i1 }, ptr %4, align 8
  %7 = extractvalue { { ptr, ptr, ptr, ptr }, i1 } %6, 1
  br i1 %7, label %9, label %8

8:                                                ; preds = %2
  call void @.class_value.missing(ptr null)
  br label %9

9:                                                ; preds = %8, %2
  %10 = extractvalue { { ptr, ptr, ptr, ptr }, i1 } %6, 0
  %11 = load double, ptr %3, align 8
  %12 = extractvalue { ptr, ptr, ptr, ptr } %10, 1
  %13 = call ptr %12()
  %14 = extractvalue { ptr, ptr, ptr, ptr } %10, 2
  %15 = ptrtoint ptr %14 to i64
  %16 = icmp ne i64 %15, 0
  br i1 %16, label %17, label %22

17:                                               ; preds = %9
  %18 = fptosi double %11 to i64
  %19 = trunc i64 %18 to i32
  %20 = insertvalue { i32, i1 } undef, i32 %19, 0
  %21 = insertvalue { i32, i1 } %20, i1 true, 1
  call void %14(ptr %13, { i32, i1 } %21)
  br label %22

22:                                               ; preds = %17, %9
  store ptr %13, ptr %5, align 8
  %23 = load ptr, ptr %5, align 8
  ret ptr %23
}

; Function Attrs: mustprogress
define i1 @isSet(<{ ptr, <{ [4 x i64] }> }> %0) #2 {
  %2 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %3 = alloca i1, align 1
  store <{ ptr, <{ [4 x i64] }> }> %0, ptr %2, align 1
  %4 = load <{ ptr, <{ [4 x i64] }> }>, ptr %2, align 1
  %5 = extractvalue <{ ptr, <{ [4 x i64] }> }> %4, 0
  %6 = icmp ne ptr %5, null
  %7 = select i1 %6, ptr %5, ptr getelementptr inbounds nuw (i8, ptr @td_676157629225208357_undefined, i64 40)
  %8 = getelementptr i8, ptr %7, i64 -40
  %9 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %8, i32 0, i32 0
  %10 = load i32, ptr %9, align 4
  %11 = icmp eq i32 %10, 13
  %12 = icmp ne i1 %11, true
  %13 = xor i1 %12, true
  %14 = xor i1 %13, true
  store i1 %14, ptr %3, align 1
  %15 = load i1, ptr %3, align 1
  ret i1 %15
}

; Function Attrs: mustprogress
define i1 @isAnySet(<{ ptr, <{ [4 x i64] }> }> %0) #2 {
  %2 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %3 = alloca i1, align 1
  store <{ ptr, <{ [4 x i64] }> }> %0, ptr %2, align 1
  %4 = load <{ ptr, <{ [4 x i64] }> }>, ptr %2, align 1
  %5 = extractvalue <{ ptr, <{ [4 x i64] }> }> %4, 0
  %6 = icmp ne ptr %5, null
  %7 = select i1 %6, ptr %5, ptr getelementptr inbounds nuw (i8, ptr @td_676157629225208357_undefined, i64 40)
  %8 = getelementptr i8, ptr %7, i64 -40
  %9 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %8, i32 0, i32 0
  %10 = load i32, ptr %9, align 4
  %11 = icmp eq i32 %10, 13
  %12 = icmp ne i1 %11, true
  br i1 %12, label %13, label %14

13:                                               ; preds = %1
  store i1 true, ptr %3, align 1
  br label %15

14:                                               ; preds = %1
  store i1 false, ptr %3, align 1
  br label %15

15:                                               ; preds = %13, %14
  %16 = load i1, ptr %3, align 1
  ret i1 %16
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"___cast<!ts.union<!ts.class<A, !ts.class_storage<A>>, !ts.class<B, !ts.class_storage<B>>>,!ts.boolean>"(<{ ptr, <{ [1 x i64] }> }> %0) #2 comdat {
  %2 = alloca <{ ptr, ptr }>, align 8
  %3 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %4 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %5 = alloca i1, align 1
  store <{ ptr, <{ [1 x i64] }> }> %0, ptr %4, align 1
  %6 = load <{ ptr, <{ [1 x i64] }> }>, ptr %4, align 1
  %7 = extractvalue <{ ptr, <{ [1 x i64] }> }> %6, 1, 0, 0
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %9, label %10

9:                                                ; preds = %1
  store i1 false, ptr %5, align 1
  br label %51

10:                                               ; preds = %1
  %11 = load <{ ptr, <{ [1 x i64] }> }>, ptr %4, align 1
  %12 = extractvalue <{ ptr, <{ [1 x i64] }> }> %11, 0
  %13 = ptrtoint ptr %12 to i64
  %14 = icmp ne i64 %13, 0
  %15 = zext i1 %14 to i32
  %16 = and i32 %15, 1
  %17 = icmp ne i32 %16, 0
  br i1 %17, label %18, label %21

18:                                               ; preds = %10
  %19 = call i32 @strcmp(ptr %12, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %20 = icmp eq i32 %19, 0
  br label %23

21:                                               ; preds = %10
  %22 = icmp eq i64 %13, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %23

23:                                               ; preds = %18, %21
  %24 = phi i1 [ %22, %21 ], [ %20, %18 ]
  br label %25

25:                                               ; preds = %23
  br i1 %24, label %26, label %49

26:                                               ; preds = %25
  %27 = load <{ ptr, <{ [1 x i64] }> }>, ptr %4, align 1
  store <{ ptr, <{ [1 x i64] }> }> %27, ptr %3, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %2, ptr %3, i64 16, i1 false)
  %28 = load <{ ptr, ptr }>, ptr %2, align 1
  %29 = extractvalue <{ ptr, ptr }> %28, 1
  %30 = getelementptr ptr, ptr %29, i32 0
  %31 = load ptr, ptr %30, align 8
  %32 = getelementptr ptr, ptr %31, i32 0
  %33 = load ptr, ptr %32, align 8
  %34 = load ptr, ptr @A..rtti, align 8
  %35 = call i1 %33(ptr %29, ptr %34)
  br i1 %35, label %36, label %39

36:                                               ; preds = %26
  %37 = ptrtoint ptr %29 to i64
  %38 = icmp ne i64 %37, 0
  store i1 %38, ptr %5, align 1
  br label %51

39:                                               ; preds = %26
  %40 = getelementptr ptr, ptr %29, i32 0
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr ptr, ptr %41, i32 0
  %43 = load ptr, ptr %42, align 8
  %44 = load ptr, ptr @B..rtti, align 8
  %45 = call i1 %43(ptr %29, ptr %44)
  br i1 %45, label %46, label %49

46:                                               ; preds = %39
  %47 = ptrtoint ptr %29 to i64
  %48 = icmp ne i64 %47, 0
  store i1 %48, ptr %5, align 1
  br label %51

49:                                               ; preds = %39, %25
  %50 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_5730213751506031948, i64 8), ptr %50, align 8
  call void @__cxa_throw(ptr %50, ptr @_ZTIPc, ptr null)
  br label %53

51:                                               ; preds = %9, %36, %46
  %52 = load i1, ptr %5, align 1
  ret i1 %52

53:                                               ; preds = %49
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #4 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca <{ ptr, ptr }>, align 8
  %4 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %5 = alloca <{ ptr, ptr }>, align 8
  %6 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %7 = alloca <{ ptr, ptr }>, align 8
  %8 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %9 = alloca <{ ptr, ptr }>, align 8
  %10 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %11 = alloca { <{ ptr, <{ [1 x i64] }> }>, i1 }, align 8
  %12 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %13 = alloca <{ ptr, ptr }>, align 8
  %14 = alloca <{ ptr, ptr }>, align 8
  %15 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %16 = alloca <{ ptr, ptr }>, align 8
  %17 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %18 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %19 = alloca <{ ptr, ptr }>, align 8
  %20 = alloca i1, align 1
  %21 = alloca { ptr, ptr, ptr, ptr }, align 8
  %22 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %23 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %24 = alloca { ptr, ptr, ptr, ptr }, align 8
  %25 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %26 = alloca <{ ptr, ptr }>, align 8
  %27 = alloca ptr, align 8
  %28 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %29 = alloca <{ ptr, ptr }>, align 8
  %30 = alloca ptr, align 8
  %31 = alloca i32, align 4
  %32 = alloca { ptr, ptr, ptr, ptr }, align 8
  %33 = alloca <{ ptr, ptr }>, align 8
  %34 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %35 = alloca <{ ptr, ptr }>, align 8
  %36 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %37 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %38 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %39 = alloca { ptr, ptr, ptr, ptr }, align 8
  %40 = alloca <{ ptr, ptr }>, align 8
  %41 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %42 = alloca <{ ptr, ptr }>, align 8
  %43 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %44 = alloca <{ ptr, ptr }>, align 8
  %45 = alloca <{ ptr, <{ [1 x i64] }> }>, align 8
  %46 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %47 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %48 = alloca { ptr, ptr, ptr, ptr }, align 8
  %49 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %50 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %51 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %52 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %53 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %54 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %55 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %56 = alloca <{ ptr, { ptr, ptr, ptr, ptr } }>, align 8
  %57 = alloca { ptr, ptr, ptr, ptr }, align 8
  %58 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %59 = alloca <{ ptr, ptr }>, align 8
  %60 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %61 = alloca <{ ptr, <{ [4 x i64] }> }>, align 8
  %62 = alloca <{ ptr, ptr }>, align 8
  store <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_14527407094101050897_null, i64 40), ptr null }>, ptr %62, align 1
  store <{ ptr, <{ [4 x i64] }> }> zeroinitializer, ptr %61, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %61, ptr %62, i64 16, i1 false)
  %63 = load <{ ptr, <{ [4 x i64] }> }>, ptr %61, align 1
  store <{ ptr, <{ [4 x i64] }> }> %63, ptr %60, align 1
  %64 = load <{ ptr, <{ [4 x i64] }> }>, ptr %60, align 1
  %65 = extractvalue <{ ptr, <{ [4 x i64] }> }> %64, 0
  %66 = icmp ne ptr %65, null
  %67 = select i1 %66, ptr %65, ptr getelementptr inbounds nuw (i8, ptr @td_676157629225208357_undefined, i64 40)
  %68 = getelementptr i8, ptr %67, i64 -40
  %69 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %68, i32 0, i32 0
  %70 = load i32, ptr %69, align 4
  %71 = icmp eq i32 %70, 13
  br i1 %71, label %72, label %76

72:                                               ; preds = %0
  store <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_14527407094101050897_null, i64 40), ptr null }>, ptr %59, align 1
  store <{ ptr, <{ [4 x i64] }> }> zeroinitializer, ptr %58, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %58, ptr %59, i64 16, i1 false)
  %73 = load <{ ptr, <{ [4 x i64] }> }>, ptr %58, align 1
  %74 = call i1 @isSet(<{ ptr, <{ [4 x i64] }> }> %73)
  %75 = xor i1 %74, true
  br label %76

76:                                               ; preds = %72, %0
  %77 = phi i1 [ %75, %72 ], [ %71, %0 ]
  br i1 %77, label %78, label %640

78:                                               ; preds = %76
  %79 = getelementptr { ptr, ptr, ptr, ptr }, ptr %57, i32 0, i32 0
  store ptr @A..vtbl, ptr %79, align 8
  %80 = getelementptr { ptr, ptr, ptr, ptr }, ptr %57, i32 0, i32 1
  store ptr @A..new, ptr %80, align 8
  %81 = getelementptr { ptr, ptr, ptr, ptr }, ptr %57, i32 0, i32 2
  store ptr @A.constructor, ptr %81, align 8
  %82 = getelementptr { ptr, ptr, ptr, ptr }, ptr %57, i32 0, i32 3
  store ptr @A..rtti, ptr %82, align 8
  %83 = load { ptr, ptr, ptr, ptr }, ptr %57, align 8
  %84 = insertvalue <{ ptr, { ptr, ptr, ptr, ptr } }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_18357227943351561971_function, i64 40), { ptr, ptr, ptr, ptr } undef }>, { ptr, ptr, ptr, ptr } %83, 1
  store <{ ptr, { ptr, ptr, ptr, ptr } }> %84, ptr %56, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %55, ptr %56, i64 40, i1 false)
  %85 = load <{ ptr, <{ [4 x i64] }> }>, ptr %55, align 1
  store <{ ptr, <{ [4 x i64] }> }> %85, ptr %60, align 1
  %86 = load <{ ptr, <{ [4 x i64] }> }>, ptr %60, align 1
  %87 = extractvalue <{ ptr, <{ [4 x i64] }> }> %86, 0
  %88 = icmp ne ptr %87, null
  %89 = select i1 %88, ptr %87, ptr getelementptr inbounds nuw (i8, ptr @td_676157629225208357_undefined, i64 40)
  %90 = getelementptr i8, ptr %89, i64 -40
  %91 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %90, i32 0, i32 0
  %92 = load i32, ptr %91, align 4
  %93 = icmp eq i32 %92, 13
  %94 = icmp ne i1 %93, true
  br i1 %94, label %95, label %102

95:                                               ; preds = %78
  %96 = load <{ ptr, <{ [4 x i64] }> }>, ptr %60, align 1
  store <{ ptr, <{ [4 x i64] }> }> %96, ptr %54, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %53, ptr %54, i64 40, i1 false)
  %97 = load <{ ptr, { ptr, ptr, ptr, ptr } }>, ptr %53, align 1
  %98 = extractvalue <{ ptr, { ptr, ptr, ptr, ptr } }> %97, 1
  %99 = insertvalue <{ ptr, { ptr, ptr, ptr, ptr } }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_18357227943351561971_function, i64 40), { ptr, ptr, ptr, ptr } undef }>, { ptr, ptr, ptr, ptr } %98, 1
  store <{ ptr, { ptr, ptr, ptr, ptr } }> %99, ptr %52, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %51, ptr %52, i64 40, i1 false)
  %100 = load <{ ptr, <{ [4 x i64] }> }>, ptr %51, align 1
  %101 = call i1 @isSet(<{ ptr, <{ [4 x i64] }> }> %100)
  br label %102

102:                                              ; preds = %95, %78
  %103 = phi i1 [ %101, %95 ], [ %94, %78 ]
  br i1 %103, label %104, label %642

104:                                              ; preds = %102
  %105 = load <{ ptr, <{ [4 x i64] }> }>, ptr %60, align 1
  %106 = extractvalue <{ ptr, <{ [4 x i64] }> }> %105, 0
  %107 = icmp ne ptr %106, null
  %108 = select i1 %107, ptr %106, ptr getelementptr inbounds nuw (i8, ptr @td_676157629225208357_undefined, i64 40)
  %109 = getelementptr i8, ptr %108, i64 -40
  %110 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %109, i32 0, i32 0
  %111 = load i32, ptr %110, align 4
  %112 = icmp eq i32 %111, 13
  %113 = icmp ne i1 %112, true
  br i1 %113, label %114, label %129

114:                                              ; preds = %104
  %115 = load <{ ptr, <{ [4 x i64] }> }>, ptr %60, align 1
  store <{ ptr, <{ [4 x i64] }> }> %115, ptr %50, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %49, ptr %50, i64 40, i1 false)
  %116 = load <{ ptr, { ptr, ptr, ptr, ptr } }>, ptr %49, align 1
  %117 = extractvalue <{ ptr, { ptr, ptr, ptr, ptr } }> %116, 1
  %118 = extractvalue { ptr, ptr, ptr, ptr } %117, 1
  %119 = call ptr %118()
  %120 = extractvalue { ptr, ptr, ptr, ptr } %117, 2
  %121 = ptrtoint ptr %120 to i64
  %122 = icmp ne i64 %121, 0
  br i1 %122, label %123, label %124

123:                                              ; preds = %114
  call void %120(ptr %119, { i32, i1 } { i32 4, i1 true })
  br label %124

124:                                              ; preds = %123, %114
  %125 = getelementptr %A, ptr %119, i32 0, i32 1
  %126 = load i32, ptr %125, align 4
  %127 = icmp eq i32 %126, 4
  br i1 %127, label %128, label %644

128:                                              ; preds = %124
  br label %129

129:                                              ; preds = %128, %104
  %130 = getelementptr { ptr, ptr, ptr, ptr }, ptr %48, i32 0, i32 0
  store ptr @B..vtbl, ptr %130, align 8
  %131 = getelementptr { ptr, ptr, ptr, ptr }, ptr %48, i32 0, i32 1
  store ptr @B..new, ptr %131, align 8
  %132 = getelementptr { ptr, ptr, ptr, ptr }, ptr %48, i32 0, i32 2
  store ptr @B.constructor, ptr %132, align 8
  %133 = getelementptr { ptr, ptr, ptr, ptr }, ptr %48, i32 0, i32 3
  store ptr @B..rtti, ptr %133, align 8
  %134 = load { ptr, ptr, ptr, ptr }, ptr %48, align 8
  %135 = insertvalue <{ ptr, { ptr, ptr, ptr, ptr } }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_17485477222143314784_function, i64 40), { ptr, ptr, ptr, ptr } undef }>, { ptr, ptr, ptr, ptr } %134, 1
  store <{ ptr, { ptr, ptr, ptr, ptr } }> %135, ptr %47, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %46, ptr %47, i64 40, i1 false)
  %136 = load <{ ptr, <{ [4 x i64] }> }>, ptr %46, align 1
  %137 = call <{ ptr, <{ [1 x i64] }> }> @make(<{ ptr, <{ [4 x i64] }> }> %136, double 2.000000e+00)
  %138 = extractvalue <{ ptr, <{ [1 x i64] }> }> %137, 0
  %139 = ptrtoint ptr %138 to i64
  %140 = icmp ne i64 %139, 0
  %141 = zext i1 %140 to i32
  %142 = and i32 %141, 1
  %143 = icmp ne i32 %142, 0
  br i1 %143, label %144, label %147

144:                                              ; preds = %129
  %145 = call i32 @strcmp(ptr %138, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %146 = icmp eq i32 %145, 0
  br label %149

147:                                              ; preds = %129
  %148 = icmp eq i64 %139, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %149

149:                                              ; preds = %144, %147
  %150 = phi i1 [ %148, %147 ], [ %146, %144 ]
  br label %151

151:                                              ; preds = %149
  br i1 %150, label %152, label %161

152:                                              ; preds = %151
  store <{ ptr, <{ [1 x i64] }> }> %137, ptr %45, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %44, ptr %45, i64 16, i1 false)
  %153 = load <{ ptr, ptr }>, ptr %44, align 1
  %154 = extractvalue <{ ptr, ptr }> %153, 1
  %155 = getelementptr ptr, ptr %154, i32 0
  %156 = load ptr, ptr %155, align 8
  %157 = getelementptr ptr, ptr %156, i32 0
  %158 = load ptr, ptr %157, align 8
  %159 = load ptr, ptr @B..rtti, align 8
  %160 = call i1 %158(ptr %154, ptr %159)
  br label %161

161:                                              ; preds = %152, %151
  %162 = phi i1 [ %160, %152 ], [ false, %151 ]
  br i1 %162, label %163, label %170

163:                                              ; preds = %161
  store <{ ptr, <{ [1 x i64] }> }> %137, ptr %43, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %42, ptr %43, i64 16, i1 false)
  %164 = load <{ ptr, ptr }>, ptr %42, align 1
  %165 = extractvalue <{ ptr, ptr }> %164, 1
  %166 = getelementptr %B, ptr %165, i32 0, i32 0
  %167 = load ptr, ptr @A..rtti, align 8
  %168 = call i1 @B..instanceOf(ptr %165, ptr %167)
  %169 = xor i1 %168, true
  br label %170

170:                                              ; preds = %163, %161
  %171 = phi i1 [ %169, %163 ], [ %162, %161 ]
  br i1 %171, label %172, label %178

172:                                              ; preds = %170
  store <{ ptr, <{ [1 x i64] }> }> %137, ptr %41, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %40, ptr %41, i64 16, i1 false)
  %173 = load <{ ptr, ptr }>, ptr %40, align 1
  %174 = extractvalue <{ ptr, ptr }> %173, 1
  %175 = getelementptr %B, ptr %174, i32 0, i32 1
  %176 = load i32, ptr %175, align 4
  %177 = icmp eq i32 %176, 2
  br label %178

178:                                              ; preds = %172, %170
  %179 = phi i1 [ %177, %172 ], [ %171, %170 ]
  br i1 %179, label %180, label %646

180:                                              ; preds = %178
  %181 = getelementptr { ptr, ptr, ptr, ptr }, ptr %39, i32 0, i32 0
  store ptr @A..vtbl, ptr %181, align 8
  %182 = getelementptr { ptr, ptr, ptr, ptr }, ptr %39, i32 0, i32 1
  store ptr @A..new, ptr %182, align 8
  %183 = getelementptr { ptr, ptr, ptr, ptr }, ptr %39, i32 0, i32 2
  store ptr @A.constructor, ptr %183, align 8
  %184 = getelementptr { ptr, ptr, ptr, ptr }, ptr %39, i32 0, i32 3
  store ptr @A..rtti, ptr %184, align 8
  %185 = load { ptr, ptr, ptr, ptr }, ptr %39, align 8
  %186 = insertvalue <{ ptr, { ptr, ptr, ptr, ptr } }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_18357227943351561971_function, i64 40), { ptr, ptr, ptr, ptr } undef }>, { ptr, ptr, ptr, ptr } %185, 1
  store <{ ptr, { ptr, ptr, ptr, ptr } }> %186, ptr %38, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %37, ptr %38, i64 40, i1 false)
  %187 = load <{ ptr, <{ [4 x i64] }> }>, ptr %37, align 1
  %188 = call <{ ptr, <{ [1 x i64] }> }> @make(<{ ptr, <{ [4 x i64] }> }> %187, double 3.000000e+00)
  %189 = extractvalue <{ ptr, <{ [1 x i64] }> }> %188, 0
  %190 = ptrtoint ptr %189 to i64
  %191 = icmp ne i64 %190, 0
  %192 = zext i1 %191 to i32
  %193 = and i32 %192, 1
  %194 = icmp ne i32 %193, 0
  br i1 %194, label %195, label %198

195:                                              ; preds = %180
  %196 = call i32 @strcmp(ptr %189, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %197 = icmp eq i32 %196, 0
  br label %200

198:                                              ; preds = %180
  %199 = icmp eq i64 %190, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %200

200:                                              ; preds = %195, %198
  %201 = phi i1 [ %199, %198 ], [ %197, %195 ]
  br label %202

202:                                              ; preds = %200
  br i1 %201, label %203, label %212

203:                                              ; preds = %202
  store <{ ptr, <{ [1 x i64] }> }> %188, ptr %36, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %35, ptr %36, i64 16, i1 false)
  %204 = load <{ ptr, ptr }>, ptr %35, align 1
  %205 = extractvalue <{ ptr, ptr }> %204, 1
  %206 = getelementptr ptr, ptr %205, i32 0
  %207 = load ptr, ptr %206, align 8
  %208 = getelementptr ptr, ptr %207, i32 0
  %209 = load ptr, ptr %208, align 8
  %210 = load ptr, ptr @A..rtti, align 8
  %211 = call i1 %209(ptr %205, ptr %210)
  br label %212

212:                                              ; preds = %203, %202
  %213 = phi i1 [ %211, %203 ], [ false, %202 ]
  br i1 %213, label %214, label %220

214:                                              ; preds = %212
  store <{ ptr, <{ [1 x i64] }> }> %188, ptr %34, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %33, ptr %34, i64 16, i1 false)
  %215 = load <{ ptr, ptr }>, ptr %33, align 1
  %216 = extractvalue <{ ptr, ptr }> %215, 1
  %217 = getelementptr %A, ptr %216, i32 0, i32 1
  %218 = load i32, ptr %217, align 4
  %219 = icmp eq i32 %218, 3
  br label %220

220:                                              ; preds = %214, %212
  %221 = phi i1 [ %219, %214 ], [ %213, %212 ]
  br i1 %221, label %222, label %648

222:                                              ; preds = %220
  %223 = getelementptr { ptr, ptr, ptr, ptr }, ptr %32, i32 0, i32 0
  store ptr @A..vtbl, ptr %223, align 8
  %224 = getelementptr { ptr, ptr, ptr, ptr }, ptr %32, i32 0, i32 1
  store ptr @A..new, ptr %224, align 8
  %225 = getelementptr { ptr, ptr, ptr, ptr }, ptr %32, i32 0, i32 2
  store ptr @A.constructor, ptr %225, align 8
  %226 = getelementptr { ptr, ptr, ptr, ptr }, ptr %32, i32 0, i32 3
  store ptr @A..rtti, ptr %226, align 8
  %227 = load { ptr, ptr, ptr, ptr }, ptr %32, align 8
  %228 = insertvalue { { ptr, ptr, ptr, ptr }, i1 } undef, { ptr, ptr, ptr, ptr } %227, 0
  %229 = insertvalue { { ptr, ptr, ptr, ptr }, i1 } %228, i1 true, 1
  %230 = call ptr @makeOne({ { ptr, ptr, ptr, ptr }, i1 } %229, double 5.000000e+00)
  %231 = getelementptr %A, ptr %230, i32 0, i32 1
  %232 = load i32, ptr %231, align 4
  %233 = icmp eq i32 %232, 5
  br i1 %233, label %234, label %650

234:                                              ; preds = %222
  store i32 0, ptr %31, align 4
  store <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_14527407094101050897_null, i64 40), ptr null }>, ptr %29, align 1
  store <{ ptr, <{ [4 x i64] }> }> zeroinitializer, ptr %28, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %28, ptr %29, i64 16, i1 false)
  %235 = load <{ ptr, <{ [4 x i64] }> }>, ptr %28, align 1
  %236 = invoke <{ ptr, <{ [1 x i64] }> }> @make(<{ ptr, <{ [4 x i64] }> }> %235, double 1.000000e+00)
          to label %237 unwind label %238

237:                                              ; preds = %234
  br label %379

238:                                              ; preds = %234
  %239 = landingpad { ptr, i32 }
          catch ptr null
  %240 = extractvalue { ptr, i32 } %239, 0
  %241 = call ptr @__cxa_begin_catch(ptr %240)
  %242 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %238
  %243 = icmp eq ptr %242, @_ZTIi
  br i1 %243, label %244, label %251

244:                                              ; preds = %.noexc3
  %245 = load i32, ptr %241, align 4
  %246 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %244
  %247 = getelementptr i8, ptr %246, i64 8
  %248 = getelementptr { i64, ptr, i32 }, ptr %247, i32 0, i32 0
  store i64 4, ptr %248, align 8
  %249 = getelementptr { i64, ptr, i32 }, ptr %247, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %249, align 8
  %250 = getelementptr { i64, ptr, i32 }, ptr %247, i32 0, i32 2
  store i32 %245, ptr %250, align 4
  br label %374

251:                                              ; preds = %.noexc3
  %252 = icmp eq ptr %242, @_ZTIj
  br i1 %252, label %253, label %260

253:                                              ; preds = %251
  %254 = load i32, ptr %241, align 4
  %255 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %253
  %256 = getelementptr i8, ptr %255, i64 8
  %257 = getelementptr { i64, ptr, i32 }, ptr %256, i32 0, i32 0
  store i64 4, ptr %257, align 8
  %258 = getelementptr { i64, ptr, i32 }, ptr %256, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %258, align 8
  %259 = getelementptr { i64, ptr, i32 }, ptr %256, i32 0, i32 2
  store i32 %254, ptr %259, align 4
  br label %371

260:                                              ; preds = %251
  %261 = icmp eq ptr %242, @_ZTId
  br i1 %261, label %262, label %269

262:                                              ; preds = %260
  %263 = load double, ptr %241, align 8
  %264 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %262
  %265 = getelementptr i8, ptr %264, i64 8
  %266 = getelementptr { i64, ptr, double }, ptr %265, i32 0, i32 0
  store i64 8, ptr %266, align 8
  %267 = getelementptr { i64, ptr, double }, ptr %265, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %267, align 8
  %268 = getelementptr { i64, ptr, double }, ptr %265, i32 0, i32 2
  store double %263, ptr %268, align 8
  br label %368

269:                                              ; preds = %260
  %270 = icmp eq ptr %242, @_ZTIf
  br i1 %270, label %271, label %278

271:                                              ; preds = %269
  %272 = load float, ptr %241, align 4
  %273 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %271
  %274 = getelementptr i8, ptr %273, i64 8
  %275 = getelementptr { i64, ptr, float }, ptr %274, i32 0, i32 0
  store i64 4, ptr %275, align 8
  %276 = getelementptr { i64, ptr, float }, ptr %274, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %276, align 8
  %277 = getelementptr { i64, ptr, float }, ptr %274, i32 0, i32 2
  store float %272, ptr %277, align 4
  br label %365

278:                                              ; preds = %269
  %279 = icmp eq ptr %242, @_ZTIb
  br i1 %279, label %280, label %287

280:                                              ; preds = %278
  %281 = load i1, ptr %241, align 1
  %282 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %280
  %283 = getelementptr i8, ptr %282, i64 8
  %284 = getelementptr { i64, ptr, i1 }, ptr %283, i32 0, i32 0
  store i64 1, ptr %284, align 8
  %285 = getelementptr { i64, ptr, i1 }, ptr %283, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %285, align 8
  %286 = getelementptr { i64, ptr, i1 }, ptr %283, i32 0, i32 2
  store i1 %281, ptr %286, align 1
  br label %362

287:                                              ; preds = %278
  %288 = icmp eq ptr %242, @_ZTIx
  br i1 %288, label %289, label %296

289:                                              ; preds = %287
  %290 = load i64, ptr %241, align 8
  %291 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %289
  %292 = getelementptr i8, ptr %291, i64 8
  %293 = getelementptr { i64, ptr, i64 }, ptr %292, i32 0, i32 0
  store i64 8, ptr %293, align 8
  %294 = getelementptr { i64, ptr, i64 }, ptr %292, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %294, align 8
  %295 = getelementptr { i64, ptr, i64 }, ptr %292, i32 0, i32 2
  store i64 %290, ptr %295, align 8
  br label %359

296:                                              ; preds = %287
  %297 = icmp eq ptr %242, @_ZTIPc
  br i1 %297, label %298, label %305

298:                                              ; preds = %296
  %299 = load ptr, ptr %241, align 8
  %300 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %298
  %301 = getelementptr i8, ptr %300, i64 8
  %302 = getelementptr { i64, ptr, ptr }, ptr %301, i32 0, i32 0
  store i64 8, ptr %302, align 8
  %303 = getelementptr { i64, ptr, ptr }, ptr %301, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %303, align 8
  %304 = getelementptr { i64, ptr, ptr }, ptr %301, i32 0, i32 2
  store ptr %299, ptr %304, align 8
  br label %356

305:                                              ; preds = %296
  %306 = icmp eq ptr %242, @_ZTIPv
  br i1 %306, label %307, label %309

307:                                              ; preds = %305
  %308 = load ptr, ptr %241, align 8
  br label %353

309:                                              ; preds = %305
  %310 = load ptr, ptr %242, align 8
  %311 = icmp eq ptr %310, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %311, label %312, label %343

312:                                              ; preds = %309
  %313 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %242, i32 0, i32 2
  %314 = load i32, ptr %313, align 4
  %315 = and i32 %314, 1073741824
  %316 = icmp ne i32 %315, 0
  br i1 %316, label %317, label %333

317:                                              ; preds = %312
  %318 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %242, i32 0, i32 4
  %319 = load ptr, ptr %318, align 8
  %320 = icmp ne ptr %319, null
  br i1 %320, label %321, label %323

321:                                              ; preds = %317
  invoke void %319(ptr %1, ptr %241)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %321
  %322 = load ptr, ptr %1, align 8
  br label %330

323:                                              ; preds = %317
  %324 = load ptr, ptr %241, align 8
  %325 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %323
  %326 = getelementptr i8, ptr %325, i64 8
  %327 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 0
  store i64 8, ptr %327, align 8
  %328 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %328, align 8
  %329 = getelementptr { i64, ptr, ptr }, ptr %326, i32 0, i32 2
  store ptr %324, ptr %329, align 8
  br label %330

330:                                              ; preds = %.noexc5, %.noexc18
  %331 = phi ptr [ %326, %.noexc18 ], [ %322, %.noexc5 ]
  br label %332

332:                                              ; preds = %330
  br label %340

333:                                              ; preds = %312
  %334 = load ptr, ptr %241, align 8
  %335 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %333
  %336 = getelementptr i8, ptr %335, i64 8
  %337 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 0
  store i64 8, ptr %337, align 8
  %338 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %338, align 8
  %339 = getelementptr { i64, ptr, ptr }, ptr %336, i32 0, i32 2
  store ptr %334, ptr %339, align 8
  br label %340

340:                                              ; preds = %332, %.noexc2
  %341 = phi ptr [ %336, %.noexc2 ], [ %331, %332 ]
  br label %342

342:                                              ; preds = %340
  br label %350

343:                                              ; preds = %309
  %344 = load ptr, ptr %241, align 8
  %345 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %343
  %346 = getelementptr i8, ptr %345, i64 8
  %347 = getelementptr { i64, ptr, ptr }, ptr %346, i32 0, i32 0
  store i64 8, ptr %347, align 8
  %348 = getelementptr { i64, ptr, ptr }, ptr %346, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %348, align 8
  %349 = getelementptr { i64, ptr, ptr }, ptr %346, i32 0, i32 2
  store ptr %344, ptr %349, align 8
  br label %350

350:                                              ; preds = %342, %.noexc19
  %351 = phi ptr [ %346, %.noexc19 ], [ %341, %342 ]
  br label %352

352:                                              ; preds = %350
  br label %353

353:                                              ; preds = %307, %352
  %354 = phi ptr [ %351, %352 ], [ %308, %307 ]
  br label %355

355:                                              ; preds = %353
  br label %356

356:                                              ; preds = %.noexc11, %355
  %357 = phi ptr [ %354, %355 ], [ %301, %.noexc11 ]
  br label %358

358:                                              ; preds = %356
  br label %359

359:                                              ; preds = %.noexc9, %358
  %360 = phi ptr [ %357, %358 ], [ %292, %.noexc9 ]
  br label %361

361:                                              ; preds = %359
  br label %362

362:                                              ; preds = %.noexc22, %361
  %363 = phi ptr [ %360, %361 ], [ %283, %.noexc22 ]
  br label %364

364:                                              ; preds = %362
  br label %365

365:                                              ; preds = %.noexc16, %364
  %366 = phi ptr [ %363, %364 ], [ %274, %.noexc16 ]
  br label %367

367:                                              ; preds = %365
  br label %368

368:                                              ; preds = %.noexc14, %367
  %369 = phi ptr [ %366, %367 ], [ %265, %.noexc14 ]
  br label %370

370:                                              ; preds = %368
  br label %371

371:                                              ; preds = %.noexc12, %370
  %372 = phi ptr [ %369, %370 ], [ %256, %.noexc12 ]
  br label %373

373:                                              ; preds = %371
  br label %374

374:                                              ; preds = %.noexc, %373
  %375 = phi ptr [ %372, %373 ], [ %247, %.noexc ]
  br label %376

376:                                              ; preds = %374
  store ptr %375, ptr %30, align 8
  %377 = load i32, ptr %31, align 4
  %378 = add i32 %377, 1
  store i32 %378, ptr %31, align 4
  call void @__cxa_end_catch()
  br label %379

379:                                              ; preds = %237, %376
  %380 = invoke ptr @makeOne({ { ptr, ptr, ptr, ptr }, i1 } zeroinitializer, double 1.000000e+00)
          to label %381 unwind label %382

381:                                              ; preds = %379
  br label %523

382:                                              ; preds = %379
  %383 = landingpad { ptr, i32 }
          catch ptr null
  %384 = extractvalue { ptr, i32 } %383, 0
  %385 = call ptr @__cxa_begin_catch(ptr %384)
  %386 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %382
  %387 = icmp eq ptr %386, @_ZTIi
  br i1 %387, label %388, label %395

388:                                              ; preds = %.noexc8
  %389 = load i32, ptr %385, align 4
  %390 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %388
  %391 = getelementptr i8, ptr %390, i64 8
  %392 = getelementptr { i64, ptr, i32 }, ptr %391, i32 0, i32 0
  store i64 4, ptr %392, align 8
  %393 = getelementptr { i64, ptr, i32 }, ptr %391, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %393, align 8
  %394 = getelementptr { i64, ptr, i32 }, ptr %391, i32 0, i32 2
  store i32 %389, ptr %394, align 4
  br label %518

395:                                              ; preds = %.noexc8
  %396 = icmp eq ptr %386, @_ZTIj
  br i1 %396, label %397, label %404

397:                                              ; preds = %395
  %398 = load i32, ptr %385, align 4
  %399 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %397
  %400 = getelementptr i8, ptr %399, i64 8
  %401 = getelementptr { i64, ptr, i32 }, ptr %400, i32 0, i32 0
  store i64 4, ptr %401, align 8
  %402 = getelementptr { i64, ptr, i32 }, ptr %400, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %402, align 8
  %403 = getelementptr { i64, ptr, i32 }, ptr %400, i32 0, i32 2
  store i32 %398, ptr %403, align 4
  br label %515

404:                                              ; preds = %395
  %405 = icmp eq ptr %386, @_ZTId
  br i1 %405, label %406, label %413

406:                                              ; preds = %404
  %407 = load double, ptr %385, align 8
  %408 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %406
  %409 = getelementptr i8, ptr %408, i64 8
  %410 = getelementptr { i64, ptr, double }, ptr %409, i32 0, i32 0
  store i64 8, ptr %410, align 8
  %411 = getelementptr { i64, ptr, double }, ptr %409, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %411, align 8
  %412 = getelementptr { i64, ptr, double }, ptr %409, i32 0, i32 2
  store double %407, ptr %412, align 8
  br label %512

413:                                              ; preds = %404
  %414 = icmp eq ptr %386, @_ZTIf
  br i1 %414, label %415, label %422

415:                                              ; preds = %413
  %416 = load float, ptr %385, align 4
  %417 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %415
  %418 = getelementptr i8, ptr %417, i64 8
  %419 = getelementptr { i64, ptr, float }, ptr %418, i32 0, i32 0
  store i64 4, ptr %419, align 8
  %420 = getelementptr { i64, ptr, float }, ptr %418, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %420, align 8
  %421 = getelementptr { i64, ptr, float }, ptr %418, i32 0, i32 2
  store float %416, ptr %421, align 4
  br label %509

422:                                              ; preds = %413
  %423 = icmp eq ptr %386, @_ZTIb
  br i1 %423, label %424, label %431

424:                                              ; preds = %422
  %425 = load i1, ptr %385, align 1
  %426 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %424
  %427 = getelementptr i8, ptr %426, i64 8
  %428 = getelementptr { i64, ptr, i1 }, ptr %427, i32 0, i32 0
  store i64 1, ptr %428, align 8
  %429 = getelementptr { i64, ptr, i1 }, ptr %427, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %429, align 8
  %430 = getelementptr { i64, ptr, i1 }, ptr %427, i32 0, i32 2
  store i1 %425, ptr %430, align 1
  br label %506

431:                                              ; preds = %422
  %432 = icmp eq ptr %386, @_ZTIx
  br i1 %432, label %433, label %440

433:                                              ; preds = %431
  %434 = load i64, ptr %385, align 8
  %435 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %433
  %436 = getelementptr i8, ptr %435, i64 8
  %437 = getelementptr { i64, ptr, i64 }, ptr %436, i32 0, i32 0
  store i64 8, ptr %437, align 8
  %438 = getelementptr { i64, ptr, i64 }, ptr %436, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %438, align 8
  %439 = getelementptr { i64, ptr, i64 }, ptr %436, i32 0, i32 2
  store i64 %434, ptr %439, align 8
  br label %503

440:                                              ; preds = %431
  %441 = icmp eq ptr %386, @_ZTIPc
  br i1 %441, label %442, label %449

442:                                              ; preds = %440
  %443 = load ptr, ptr %385, align 8
  %444 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %442
  %445 = getelementptr i8, ptr %444, i64 8
  %446 = getelementptr { i64, ptr, ptr }, ptr %445, i32 0, i32 0
  store i64 8, ptr %446, align 8
  %447 = getelementptr { i64, ptr, ptr }, ptr %445, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %447, align 8
  %448 = getelementptr { i64, ptr, ptr }, ptr %445, i32 0, i32 2
  store ptr %443, ptr %448, align 8
  br label %500

449:                                              ; preds = %440
  %450 = icmp eq ptr %386, @_ZTIPv
  br i1 %450, label %451, label %453

451:                                              ; preds = %449
  %452 = load ptr, ptr %385, align 8
  br label %497

453:                                              ; preds = %449
  %454 = load ptr, ptr %386, align 8
  %455 = icmp eq ptr %454, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %455, label %456, label %487

456:                                              ; preds = %453
  %457 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %386, i32 0, i32 2
  %458 = load i32, ptr %457, align 4
  %459 = and i32 %458, 1073741824
  %460 = icmp ne i32 %459, 0
  br i1 %460, label %461, label %477

461:                                              ; preds = %456
  %462 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %386, i32 0, i32 4
  %463 = load ptr, ptr %462, align 8
  %464 = icmp ne ptr %463, null
  br i1 %464, label %465, label %467

465:                                              ; preds = %461
  invoke void %463(ptr %2, ptr %385)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %465
  %466 = load ptr, ptr %2, align 8
  br label %474

467:                                              ; preds = %461
  %468 = load ptr, ptr %385, align 8
  %469 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %467
  %470 = getelementptr i8, ptr %469, i64 8
  %471 = getelementptr { i64, ptr, ptr }, ptr %470, i32 0, i32 0
  store i64 8, ptr %471, align 8
  %472 = getelementptr { i64, ptr, ptr }, ptr %470, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %472, align 8
  %473 = getelementptr { i64, ptr, ptr }, ptr %470, i32 0, i32 2
  store ptr %468, ptr %473, align 8
  br label %474

474:                                              ; preds = %.noexc23, %.noexc6
  %475 = phi ptr [ %470, %.noexc6 ], [ %466, %.noexc23 ]
  br label %476

476:                                              ; preds = %474
  br label %484

477:                                              ; preds = %456
  %478 = load ptr, ptr %385, align 8
  %479 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %477
  %480 = getelementptr i8, ptr %479, i64 8
  %481 = getelementptr { i64, ptr, ptr }, ptr %480, i32 0, i32 0
  store i64 8, ptr %481, align 8
  %482 = getelementptr { i64, ptr, ptr }, ptr %480, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %482, align 8
  %483 = getelementptr { i64, ptr, ptr }, ptr %480, i32 0, i32 2
  store ptr %478, ptr %483, align 8
  br label %484

484:                                              ; preds = %476, %.noexc21
  %485 = phi ptr [ %480, %.noexc21 ], [ %475, %476 ]
  br label %486

486:                                              ; preds = %484
  br label %494

487:                                              ; preds = %453
  %488 = load ptr, ptr %385, align 8
  %489 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %487
  %490 = getelementptr i8, ptr %489, i64 8
  %491 = getelementptr { i64, ptr, ptr }, ptr %490, i32 0, i32 0
  store i64 8, ptr %491, align 8
  %492 = getelementptr { i64, ptr, ptr }, ptr %490, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %492, align 8
  %493 = getelementptr { i64, ptr, ptr }, ptr %490, i32 0, i32 2
  store ptr %488, ptr %493, align 8
  br label %494

494:                                              ; preds = %486, %.noexc20
  %495 = phi ptr [ %490, %.noexc20 ], [ %485, %486 ]
  br label %496

496:                                              ; preds = %494
  br label %497

497:                                              ; preds = %451, %496
  %498 = phi ptr [ %495, %496 ], [ %452, %451 ]
  br label %499

499:                                              ; preds = %497
  br label %500

500:                                              ; preds = %.noexc1, %499
  %501 = phi ptr [ %498, %499 ], [ %445, %.noexc1 ]
  br label %502

502:                                              ; preds = %500
  br label %503

503:                                              ; preds = %.noexc7, %502
  %504 = phi ptr [ %501, %502 ], [ %436, %.noexc7 ]
  br label %505

505:                                              ; preds = %503
  br label %506

506:                                              ; preds = %.noexc13, %505
  %507 = phi ptr [ %504, %505 ], [ %427, %.noexc13 ]
  br label %508

508:                                              ; preds = %506
  br label %509

509:                                              ; preds = %.noexc10, %508
  %510 = phi ptr [ %507, %508 ], [ %418, %.noexc10 ]
  br label %511

511:                                              ; preds = %509
  br label %512

512:                                              ; preds = %.noexc4, %511
  %513 = phi ptr [ %510, %511 ], [ %409, %.noexc4 ]
  br label %514

514:                                              ; preds = %512
  br label %515

515:                                              ; preds = %.noexc15, %514
  %516 = phi ptr [ %513, %514 ], [ %400, %.noexc15 ]
  br label %517

517:                                              ; preds = %515
  br label %518

518:                                              ; preds = %.noexc17, %517
  %519 = phi ptr [ %516, %517 ], [ %391, %.noexc17 ]
  br label %520

520:                                              ; preds = %518
  store ptr %519, ptr %27, align 8
  %521 = load i32, ptr %31, align 4
  %522 = add i32 %521, 1
  store i32 %522, ptr %31, align 4
  call void @__cxa_end_catch()
  br label %523

523:                                              ; preds = %381, %520
  %524 = load i32, ptr %31, align 4
  %525 = icmp eq i32 %524, 2
  br i1 %525, label %526, label %652

526:                                              ; preds = %523
  store <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_14527407094101050897_null, i64 40), ptr null }>, ptr %26, align 1
  store <{ ptr, <{ [4 x i64] }> }> zeroinitializer, ptr %25, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %25, ptr %26, i64 16, i1 false)
  %527 = load <{ ptr, <{ [4 x i64] }> }>, ptr %25, align 1
  %528 = call i1 @isAnySet(<{ ptr, <{ [4 x i64] }> }> %527)
  %529 = xor i1 %528, true
  br i1 %529, label %530, label %654

530:                                              ; preds = %526
  %531 = getelementptr { ptr, ptr, ptr, ptr }, ptr %24, i32 0, i32 0
  store ptr @B..vtbl, ptr %531, align 8
  %532 = getelementptr { ptr, ptr, ptr, ptr }, ptr %24, i32 0, i32 1
  store ptr @B..new, ptr %532, align 8
  %533 = getelementptr { ptr, ptr, ptr, ptr }, ptr %24, i32 0, i32 2
  store ptr @B.constructor, ptr %533, align 8
  %534 = getelementptr { ptr, ptr, ptr, ptr }, ptr %24, i32 0, i32 3
  store ptr @B..rtti, ptr %534, align 8
  %535 = load { ptr, ptr, ptr, ptr }, ptr %24, align 8
  %536 = insertvalue <{ ptr, { ptr, ptr, ptr, ptr } }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_17485477222143314784_function, i64 40), { ptr, ptr, ptr, ptr } undef }>, { ptr, ptr, ptr, ptr } %535, 1
  store <{ ptr, { ptr, ptr, ptr, ptr } }> %536, ptr %23, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %22, ptr %23, i64 40, i1 false)
  %537 = load <{ ptr, <{ [4 x i64] }> }>, ptr %22, align 1
  %538 = call i1 @isAnySet(<{ ptr, <{ [4 x i64] }> }> %537)
  br i1 %538, label %539, label %656

539:                                              ; preds = %530
  %540 = getelementptr { ptr, ptr, ptr, ptr }, ptr %21, i32 0, i32 0
  store ptr @A..vtbl, ptr %540, align 8
  %541 = getelementptr { ptr, ptr, ptr, ptr }, ptr %21, i32 0, i32 1
  store ptr @A..new, ptr %541, align 8
  %542 = getelementptr { ptr, ptr, ptr, ptr }, ptr %21, i32 0, i32 2
  store ptr @A.constructor, ptr %542, align 8
  %543 = getelementptr { ptr, ptr, ptr, ptr }, ptr %21, i32 0, i32 3
  store ptr @A..rtti, ptr %543, align 8
  %544 = load { ptr, ptr, ptr, ptr }, ptr %21, align 8
  store i1 false, ptr %20, align 1
  store i1 true, ptr %20, align 1
  %545 = load i1, ptr %20, align 1
  br i1 %545, label %546, label %658

546:                                              ; preds = %539
  %547 = call ptr @GC_malloc(i64 24)
  %548 = getelementptr i8, ptr %547, i64 8
  %549 = getelementptr %B, ptr %548, i32 0, i32 0
  store ptr @B..vtbl, ptr %549, align 8
  call void @B.constructor(ptr %548, { i32, i1 } zeroinitializer)
  %550 = insertvalue <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_781263649003409679_class, i64 40), ptr undef }>, ptr %548, 1
  store <{ ptr, ptr }> %550, ptr %19, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %18, ptr %19, i64 16, i1 false)
  %551 = load <{ ptr, <{ [1 x i64] }> }>, ptr %18, align 1
  store <{ ptr, <{ [1 x i64] }> }> %551, ptr %17, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %16, ptr %17, i64 16, i1 false)
  %552 = load <{ ptr, ptr }>, ptr %16, align 1
  %553 = extractvalue <{ ptr, ptr }> %552, 1
  %554 = getelementptr ptr, ptr %553, i32 0
  %555 = load ptr, ptr %554, align 8
  %556 = getelementptr ptr, ptr %555, i32 0
  %557 = load ptr, ptr %556, align 8
  %558 = load ptr, ptr @A..rtti, align 8
  %559 = call i1 %557(ptr %553, ptr %558)
  br i1 %559, label %560, label %565

560:                                              ; preds = %546
  store <{ ptr, <{ [1 x i64] }> }> %551, ptr %15, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %14, ptr %15, i64 16, i1 false)
  %561 = load <{ ptr, ptr }>, ptr %14, align 1
  %562 = extractvalue <{ ptr, ptr }> %561, 1
  %563 = getelementptr %A, ptr %562, i32 0, i32 0
  %564 = call ptr @A.name(ptr %562)
  br label %570

565:                                              ; preds = %546
  store <{ ptr, <{ [1 x i64] }> }> %551, ptr %4, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %3, ptr %4, i64 16, i1 false)
  %566 = load <{ ptr, ptr }>, ptr %3, align 1
  %567 = extractvalue <{ ptr, ptr }> %566, 1
  %568 = getelementptr %B, ptr %567, i32 0, i32 0
  %569 = call ptr @B.name(ptr %567)
  br label %570

570:                                              ; preds = %560, %565
  %571 = phi ptr [ %569, %565 ], [ %564, %560 ]
  %572 = ptrtoint ptr %571 to i64
  %573 = icmp ne i64 %572, 0
  %574 = zext i1 %573 to i32
  %575 = and i32 %574, 1
  %576 = icmp ne i32 %575, 0
  br i1 %576, label %577, label %580

577:                                              ; preds = %570
  %578 = call i32 @strcmp(ptr %571, ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8))
  %579 = icmp eq i32 %578, 0
  br label %582

580:                                              ; preds = %570
  %581 = icmp eq i64 %572, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8) to i64)
  br label %582

582:                                              ; preds = %577, %580
  %583 = phi i1 [ %581, %580 ], [ %579, %577 ]
  br label %584

584:                                              ; preds = %582
  br i1 %583, label %585, label %660

585:                                              ; preds = %584
  %586 = call ptr @GC_malloc(i64 24)
  %587 = getelementptr i8, ptr %586, i64 8
  %588 = getelementptr %A, ptr %587, i32 0, i32 0
  store ptr @A..vtbl, ptr %588, align 8
  call void @A.constructor(ptr %587, { i32, i1 } zeroinitializer)
  %589 = insertvalue <{ ptr, ptr }> <{ ptr getelementptr inbounds nuw (i8, ptr @td_4117155613450509001_class, i64 40), ptr undef }>, ptr %587, 1
  store <{ ptr, ptr }> %589, ptr %13, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %12, ptr %13, i64 16, i1 false)
  %590 = load <{ ptr, <{ [1 x i64] }> }>, ptr %12, align 1
  %591 = insertvalue { <{ ptr, <{ [1 x i64] }> }>, i1 } undef, <{ ptr, <{ [1 x i64] }> }> %590, 0
  %592 = insertvalue { <{ ptr, <{ [1 x i64] }> }>, i1 } %591, i1 true, 1
  store { <{ ptr, <{ [1 x i64] }> }>, i1 } %592, ptr %11, align 1
  %593 = load { <{ ptr, <{ [1 x i64] }> }>, i1 }, ptr %11, align 1
  %594 = extractvalue { <{ ptr, <{ [1 x i64] }> }>, i1 } %593, 1
  br i1 %594, label %595, label %598

595:                                              ; preds = %585
  %596 = extractvalue { <{ ptr, <{ [1 x i64] }> }>, i1 } %593, 0
  %597 = call i1 @"___cast<!ts.union<!ts.class<A, !ts.class_storage<A>>, !ts.class<B, !ts.class_storage<B>>>,!ts.boolean>"(<{ ptr, <{ [1 x i64] }> }> %596)
  br label %599

598:                                              ; preds = %585
  br label %599

599:                                              ; preds = %595, %598
  %600 = phi i1 [ false, %598 ], [ %597, %595 ]
  br i1 %600, label %601, label %638

601:                                              ; preds = %599
  %602 = load { <{ ptr, <{ [1 x i64] }> }>, i1 }, ptr %11, align 1
  %603 = extractvalue { <{ ptr, <{ [1 x i64] }> }>, i1 } %602, 0
  store <{ ptr, <{ [1 x i64] }> }> %603, ptr %10, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %9, ptr %10, i64 16, i1 false)
  %604 = load <{ ptr, ptr }>, ptr %9, align 1
  %605 = extractvalue <{ ptr, ptr }> %604, 1
  %606 = getelementptr ptr, ptr %605, i32 0
  %607 = load ptr, ptr %606, align 8
  %608 = getelementptr ptr, ptr %607, i32 0
  %609 = load ptr, ptr %608, align 8
  %610 = load ptr, ptr @A..rtti, align 8
  %611 = call i1 %609(ptr %605, ptr %610)
  br i1 %611, label %612, label %617

612:                                              ; preds = %601
  store <{ ptr, <{ [1 x i64] }> }> %603, ptr %8, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %7, ptr %8, i64 16, i1 false)
  %613 = load <{ ptr, ptr }>, ptr %7, align 1
  %614 = extractvalue <{ ptr, ptr }> %613, 1
  %615 = getelementptr %A, ptr %614, i32 0, i32 0
  %616 = call ptr @A.name(ptr %614)
  br label %622

617:                                              ; preds = %601
  store <{ ptr, <{ [1 x i64] }> }> %603, ptr %6, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %5, ptr %6, i64 16, i1 false)
  %618 = load <{ ptr, ptr }>, ptr %5, align 1
  %619 = extractvalue <{ ptr, ptr }> %618, 1
  %620 = getelementptr %B, ptr %619, i32 0, i32 0
  %621 = call ptr @B.name(ptr %619)
  br label %622

622:                                              ; preds = %612, %617
  %623 = phi ptr [ %621, %617 ], [ %616, %612 ]
  %624 = ptrtoint ptr %623 to i64
  %625 = icmp ne i64 %624, 0
  %626 = zext i1 %625 to i32
  %627 = and i32 %626, 1
  %628 = icmp ne i32 %627, 0
  br i1 %628, label %629, label %632

629:                                              ; preds = %622
  %630 = call i32 @strcmp(ptr %623, ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8))
  %631 = icmp eq i32 %630, 0
  br label %634

632:                                              ; preds = %622
  %633 = icmp eq i64 %624, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8) to i64)
  br label %634

634:                                              ; preds = %629, %632
  %635 = phi i1 [ %633, %632 ], [ %631, %629 ]
  br label %636

636:                                              ; preds = %634
  br i1 %635, label %637, label %662

637:                                              ; preds = %636
  br label %638

638:                                              ; preds = %637, %599
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

639:                                              ; preds = %662, %660, %658, %656, %654, %652, %650, %648, %646, %644, %642, %640
  unreachable

640:                                              ; preds = %76
  %641 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6601085983368743140, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 45, ptr null)
  br label %639

642:                                              ; preds = %102
  %643 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_924000406299335852, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 47, ptr null)
  br label %639

644:                                              ; preds = %124
  %645 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2271584025657132856, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 49, ptr null)
  br label %639

646:                                              ; preds = %178
  %647 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9864376297169422948, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 54, ptr null)
  br label %639

648:                                              ; preds = %220
  %649 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9861561547401770463, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 56, ptr null)
  br label %639

650:                                              ; preds = %222
  %651 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13491316165430368537, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 57, ptr null)
  br label %639

652:                                              ; preds = %523
  %653 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14207965758675701437, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 73, ptr null)
  br label %639

654:                                              ; preds = %526
  %655 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_164902768108348953, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 76, ptr null)
  br label %639

656:                                              ; preds = %530
  %657 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4967424912731520900, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 77, ptr null)
  br label %639

658:                                              ; preds = %539
  %659 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3447504299588790889, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 84, ptr null)
  br label %639

660:                                              ; preds = %584
  %661 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6609465532978911506, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 88, ptr null)
  br label %639

662:                                              ; preds = %636
  %663 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3938361494966664765, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_3100737149123232877, i64 8), i32 91, ptr null)
  br label %639

endcatch.resume:                                  ; preds = %465, %280, %477, %487, %343, %323, %388, %271, %397, %262, %424, %253, %298, %415, %289, %382, %433, %467, %321, %406, %238, %333, %442, %244
  %664 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %664
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
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "__null_interface_vtable" }
attributes #4 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

