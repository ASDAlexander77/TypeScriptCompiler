; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Node = type { ptr, i32, ptr }

$Node..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_7403105002735449741 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNode\00", align 8
@s_11131215595135224269 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbare\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_16940945161124618027 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFjunk\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_4585100004468534399 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_shared_global_argument.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_16713252134513893189 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFreplaced\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@m_3229447641280194799 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe caller's hold is the only count left\00", align 8
@s_12644979747856758223 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFagain\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_3019566071980655109 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught\00", align 8
@m_7998919652572515854 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa call that does not assign the global takes no count\00", align 8
@m_5341766768668344537 = internal constant [57 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa handle given to a call that assigns its global\00", align 8
@m_36228825792445631 = internal constant [74 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa handle given to a call that assigns its global through a callee\00", align 8
@m_6211200806403035524 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa handle given in a try body\00", align 8
@m_3281035815395069118 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe global's own count after the calls\00", align 8
@Node..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7403105002735449741, i64 8)
@Node..size = global i64 24, comdat
@Node..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Node..instanceOf, ptr @Node..new, ptr @Node..rtti, ptr @Node..size }
@bare = private global ptr undef
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_773448429319617018 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @bare__cctor, ptr null }]

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__cxa_end_catch()

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

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

define internal void @bare__cctor() {
  %1 = call ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_11131215595135224269, i64 8), double 2.000000e+00)
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  store ptr %1, ptr %3, align 8
  store ptr %3, ptr @bare, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Node.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 1
  store i32 0, ptr %4, align 4
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Node, ptr %5, i32 0, i32 2
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Node..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Node..rtti, align 8
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
define ptr @Node..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 0
  store ptr @Node..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i64 @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = call ptr @GC_malloc(i64 8)
  %5 = getelementptr i8, ptr %4, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %5, ptr @a_0_773448429319617018, i64 0, i1 false)
  %6 = call ptr @GC_malloc(i64 32)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 0
  store ptr %5, ptr %8, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 1
  store i64 0, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 2
  store i64 0, ptr %10, align 8
  store ptr %7, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %11

11:                                               ; preds = %88, %0
  %12 = load i32, ptr %1, align 4
  %13 = icmp slt i32 %12, 1000
  br i1 %13, label %14, label %92

14:                                               ; preds = %11
  %15 = call ptr @GC_malloc(i64 32)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %Node, ptr %16, i32 0, i32 0
  store ptr @Node..vtbl, ptr %17, align 8
  call void @Node.constructor(ptr %16)
  %18 = getelementptr %Node, ptr %16, i32 0, i32 1
  %19 = load i32, ptr %1, align 4
  %20 = sub i32 0, %19
  store i32 %20, ptr %18, align 4
  %21 = getelementptr %Node, ptr %16, i32 0, i32 2
  %22 = load i32, ptr %1, align 4
  %23 = call ptr @GC_malloc_atomic(i64 58)
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %24, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %22)
  %26 = icmp eq ptr %24, null
  %27 = select i1 %26, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %24
  %28 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_16940945161124618027, i64 8))
  %29 = add i64 1, %28
  %30 = call i64 @strlen(ptr %27)
  %31 = add i64 %29, %30
  %32 = add i64 %31, 8
  %33 = call ptr @GC_malloc(i64 %32)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr i8, ptr %34, i64 0
  %36 = sub i64 %31, 0
  %37 = call ptr @__memcpy_chk(ptr %35, ptr getelementptr inbounds nuw (i8, ptr @s_16940945161124618027, i64 8), i64 %28, i64 %36)
  %38 = add i64 0, %28
  %39 = getelementptr i8, ptr %34, i64 %38
  %40 = sub i64 %31, %38
  %41 = call ptr @__memcpy_chk(ptr %39, ptr %27, i64 %30, i64 %40)
  %42 = add i64 %38, %30
  %43 = getelementptr i8, ptr %34, i64 %42
  store i8 0, ptr %43, align 1
  store ptr %34, ptr %21, align 8
  %44 = load ptr, ptr %2, align 8
  %45 = icmp eq ptr %44, null
  br i1 %45, label %46, label %49

46:                                               ; preds = %14
  %47 = call ptr @GC_malloc(i64 32)
  %48 = getelementptr i8, ptr %47, i64 8
  store ptr %48, ptr %2, align 8
  br label %50

49:                                               ; preds = %14
  br label %50

50:                                               ; preds = %46, %49
  %51 = phi ptr [ %44, %49 ], [ %48, %46 ]
  br label %52

52:                                               ; preds = %50
  %53 = getelementptr { ptr, i64, i64 }, ptr %51, i32 0, i32 1
  %54 = load i64, ptr %53, align 8
  %55 = add i64 %54, 1
  %56 = getelementptr { ptr, i64, i64 }, ptr %51, i32 0, i32 0
  %57 = getelementptr { ptr, i64, i64 }, ptr %51, i32 0, i32 2
  %58 = load ptr, ptr %56, align 8
  %59 = load i64, ptr %57, align 8
  %60 = icmp ugt i64 %55, %59
  br i1 %60, label %61, label %85

61:                                               ; preds = %52
  %62 = mul i64 %59, 2
  %63 = icmp uge i64 %62, %55
  %64 = select i1 %63, i64 %62, i64 %55
  %65 = icmp ult i64 %64, 4
  %66 = select i1 %65, i64 4, i64 %64
  %67 = icmp ule i64 %55, 4294967295
  br i1 %67, label %68, label %100

68:                                               ; preds = %61
  %69 = icmp ule i64 %66, 2305843009213693951
  br i1 %69, label %70, label %102

70:                                               ; preds = %68
  %71 = mul i64 8, %66
  %72 = add i64 %71, 8
  %73 = icmp eq ptr %58, null
  %74 = getelementptr i8, ptr %58, i64 -8
  %75 = select i1 %73, ptr null, ptr %74
  %76 = call ptr @GC_realloc(ptr %75, i64 %72)
  %77 = getelementptr i8, ptr %76, i64 8
  %78 = icmp ne ptr %77, null
  %79 = icmp eq i64 %71, 0
  %80 = or i1 %78, %79
  br i1 %80, label %81, label %104

81:                                               ; preds = %70
  %82 = mul i64 8, %59
  %83 = getelementptr i8, ptr %77, i64 %82
  %84 = sub i64 %71, %82
  call void @llvm.memset.p0.i64(ptr %83, i8 0, i64 %84, i1 false)
  store ptr %77, ptr %56, align 8
  store i64 %66, ptr %57, align 8
  br label %86

85:                                               ; preds = %52
  br label %86

86:                                               ; preds = %81, %85
  %87 = phi ptr [ %58, %85 ], [ %77, %81 ]
  br label %88

88:                                               ; preds = %86
  %89 = getelementptr ptr, ptr %87, i64 %54
  store ptr %16, ptr %89, align 8
  store i64 %55, ptr %53, align 8
  %90 = load i32, ptr %1, align 4
  %91 = add i32 %90, 1
  store i32 %91, ptr %1, align 4
  br label %11

92:                                               ; preds = %11
  %93 = load ptr, ptr %2, align 8
  %94 = icmp eq ptr %93, null
  %95 = select i1 %94, ptr @__ts_empty_array, ptr %93
  %96 = getelementptr { ptr, i64, i64 }, ptr %95, i32 0, i32 1
  %97 = load i64, ptr %96, align 8
  store i64 %97, ptr %3, align 8
  %98 = load i64, ptr %3, align 8
  ret i64 %98

99:                                               ; preds = %104, %102, %100
  unreachable

100:                                              ; preds = %61
  %101 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 18, ptr null)
  br label %99

102:                                              ; preds = %68
  %103 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 18, ptr null)
  br label %99

104:                                              ; preds = %70
  %105 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 18, ptr null)
  br label %99
}

; Function Attrs: mustprogress
define i1 @countIs(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  br i1 true, label %10, label %7

7:                                                ; preds = %2
  %8 = load double, ptr %3, align 8
  %9 = fcmp oeq double -1.000000e+00, %8
  br label %10

10:                                               ; preds = %7, %2
  %11 = phi i1 [ %9, %7 ], [ true, %2 ]
  store i1 %11, ptr %5, align 1
  %12 = load i1, ptr %5, align 1
  ret i1 %12
}

; Function Attrs: mustprogress
define ptr @mk(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = call ptr @GC_malloc(i64 32)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Node, ptr %7, i32 0, i32 0
  store ptr @Node..vtbl, ptr %8, align 8
  call void @Node.constructor(ptr %7)
  %9 = getelementptr %Node, ptr %7, i32 0, i32 2
  %10 = load ptr, ptr %4, align 8
  store ptr %10, ptr %9, align 8
  %11 = getelementptr %Node, ptr %7, i32 0, i32 1
  %12 = load double, ptr %3, align 8
  %13 = fptosi double %12 to i64
  %14 = trunc i64 %13 to i32
  store i32 %14, ptr %11, align 4
  store ptr %7, ptr %5, align 8
  %15 = load ptr, ptr %5, align 8
  ret ptr %15
}

; Function Attrs: mustprogress
define i1 @run(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = call ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8), double 5.000000e+00)
  %7 = call ptr @GC_malloc(i64 16)
  %8 = getelementptr i8, ptr %7, i64 8
  store ptr %6, ptr %8, align 8
  store ptr %8, ptr @bare, align 8
  %9 = call i64 @churn()
  %10 = icmp eq i64 %9, 1000
  br i1 %10, label %11, label %24

11:                                               ; preds = %2
  %12 = load ptr, ptr %4, align 8
  %13 = call i1 @countIs(ptr %12, double 1.000000e+00)
  br i1 %13, label %14, label %26

14:                                               ; preds = %11
  %15 = load ptr, ptr %4, align 8
  %16 = load ptr, ptr %15, align 8
  %17 = getelementptr %Node, ptr %16, i32 0, i32 1
  %18 = load i32, ptr %17, align 4
  %19 = load double, ptr %3, align 8
  %20 = sitofp i32 %18 to double
  %21 = fcmp oeq double %20, %19
  store i1 %21, ptr %5, align 1
  %22 = load i1, ptr %5, align 1
  ret i1 %22

23:                                               ; preds = %26, %24
  unreachable

24:                                               ; preds = %2
  %25 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 41, ptr null)
  br label %23

26:                                               ; preds = %11
  %27 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3229447641280194799, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 42, ptr null)
  br label %23
}

; Function Attrs: mustprogress
define void @replaceBare() #2 {
  %1 = call ptr @mk(ptr getelementptr inbounds nuw (i8, ptr @s_12644979747856758223, i64 8), double 6.000000e+00)
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  store ptr %1, ptr %3, align 8
  store ptr %3, ptr @bare, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @runThrough(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  store ptr %0, ptr %2, align 8
  call void @replaceBare()
  %4 = call i64 @churn()
  %5 = icmp eq i64 %4, 1000
  br i1 %5, label %6, label %34

6:                                                ; preds = %1
  %7 = load ptr, ptr %2, align 8
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr %Node, ptr %8, i32 0, i32 1
  %10 = load i32, ptr %9, align 4
  %11 = icmp eq i32 %10, 5
  br i1 %11, label %12, label %30

12:                                               ; preds = %6
  %13 = load ptr, ptr %2, align 8
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr %Node, ptr %14, i32 0, i32 2
  %16 = load ptr, ptr %15, align 8
  %17 = ptrtoint ptr %16 to i64
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i32
  %20 = and i32 %19, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %12
  %23 = call i32 @strcmp(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8))
  %24 = icmp eq i32 %23, 0
  br label %27

25:                                               ; preds = %12
  %26 = icmp eq i64 %17, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8) to i64)
  br label %27

27:                                               ; preds = %22, %25
  %28 = phi i1 [ %26, %25 ], [ %24, %22 ]
  br label %29

29:                                               ; preds = %27
  br label %30

30:                                               ; preds = %29, %6
  %31 = phi i1 [ %28, %29 ], [ %11, %6 ]
  store i1 %31, ptr %3, align 1
  %32 = load i1, ptr %3, align 1
  ret i1 %32

33:                                               ; preds = %34
  unreachable

34:                                               ; preds = %1
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 53, ptr null)
  br label %33
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = load ptr, ptr @bare, align 8
  %5 = call i1 @countIs(ptr %4, double 1.000000e+00)
  br i1 %5, label %6, label %182

6:                                                ; preds = %0
  %7 = load ptr, ptr @bare, align 8
  %8 = call i1 @run(ptr %7, double 2.000000e+00)
  br i1 %8, label %9, label %184

9:                                                ; preds = %6
  %10 = load ptr, ptr @bare, align 8
  %11 = call i1 @runThrough(ptr %10)
  br i1 %11, label %12, label %186

12:                                               ; preds = %9
  store i1 false, ptr %3, align 1
  %13 = load ptr, ptr @bare, align 8
  %14 = invoke i1 @run(ptr %13, double 6.000000e+00)
          to label %15 unwind label %16

15:                                               ; preds = %12
  store i1 %14, ptr %3, align 1
  br label %155

16:                                               ; preds = %12
  %17 = landingpad { ptr, i32 }
          catch ptr null
  %18 = extractvalue { ptr, i32 } %17, 0
  %19 = call ptr @__cxa_begin_catch(ptr %18)
  %20 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %16
  %21 = icmp eq ptr %20, @_ZTIi
  br i1 %21, label %22, label %29

22:                                               ; preds = %.noexc9
  %23 = load i32, ptr %19, align 4
  %24 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %22
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 0
  store i64 4, ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %27, align 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 2
  store i32 %23, ptr %28, align 4
  br label %152

29:                                               ; preds = %.noexc9
  %30 = icmp eq ptr %20, @_ZTIj
  br i1 %30, label %31, label %38

31:                                               ; preds = %29
  %32 = load i32, ptr %19, align 4
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 0
  store i64 4, ptr %35, align 8
  %36 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 2
  store i32 %32, ptr %37, align 4
  br label %149

38:                                               ; preds = %29
  %39 = icmp eq ptr %20, @_ZTId
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load double, ptr %19, align 8
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, double }, ptr %43, i32 0, i32 0
  store i64 8, ptr %44, align 8
  %45 = getelementptr { i64, ptr, double }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, double }, ptr %43, i32 0, i32 2
  store double %41, ptr %46, align 8
  br label %146

47:                                               ; preds = %38
  %48 = icmp eq ptr %20, @_ZTIf
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load float, ptr %19, align 4
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, float }, ptr %52, i32 0, i32 0
  store i64 4, ptr %53, align 8
  %54 = getelementptr { i64, ptr, float }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, float }, ptr %52, i32 0, i32 2
  store float %50, ptr %55, align 4
  br label %143

56:                                               ; preds = %47
  %57 = icmp eq ptr %20, @_ZTIb
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load i1, ptr %19, align 1
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, i1 }, ptr %61, i32 0, i32 0
  store i64 1, ptr %62, align 8
  %63 = getelementptr { i64, ptr, i1 }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, i1 }, ptr %61, i32 0, i32 2
  store i1 %59, ptr %64, align 1
  br label %140

65:                                               ; preds = %56
  %66 = icmp eq ptr %20, @_ZTIx
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load i64, ptr %19, align 8
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, i64 }, ptr %70, i32 0, i32 0
  store i64 8, ptr %71, align 8
  %72 = getelementptr { i64, ptr, i64 }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, i64 }, ptr %70, i32 0, i32 2
  store i64 %68, ptr %73, align 8
  br label %137

74:                                               ; preds = %65
  %75 = icmp eq ptr %20, @_ZTIPc
  br i1 %75, label %76, label %83

76:                                               ; preds = %74
  %77 = load ptr, ptr %19, align 8
  %78 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %76
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr { i64, ptr, ptr }, ptr %79, i32 0, i32 0
  store i64 8, ptr %80, align 8
  %81 = getelementptr { i64, ptr, ptr }, ptr %79, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %81, align 8
  %82 = getelementptr { i64, ptr, ptr }, ptr %79, i32 0, i32 2
  store ptr %77, ptr %82, align 8
  br label %134

83:                                               ; preds = %74
  %84 = icmp eq ptr %20, @_ZTIPv
  br i1 %84, label %85, label %87

85:                                               ; preds = %83
  %86 = load ptr, ptr %19, align 8
  br label %131

87:                                               ; preds = %83
  %88 = load ptr, ptr %20, align 8
  %89 = icmp eq ptr %88, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %89, label %90, label %121

90:                                               ; preds = %87
  %91 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %20, i32 0, i32 2
  %92 = load i32, ptr %91, align 4
  %93 = and i32 %92, 1073741824
  %94 = icmp ne i32 %93, 0
  br i1 %94, label %95, label %111

95:                                               ; preds = %90
  %96 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %20, i32 0, i32 4
  %97 = load ptr, ptr %96, align 8
  %98 = icmp ne ptr %97, null
  br i1 %98, label %99, label %101

99:                                               ; preds = %95
  invoke void %97(ptr %1, ptr %19)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %99
  %100 = load ptr, ptr %1, align 8
  br label %108

101:                                              ; preds = %95
  %102 = load ptr, ptr %19, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %.noexc10, %.noexc2
  %109 = phi ptr [ %104, %.noexc2 ], [ %100, %.noexc10 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %90
  %112 = load ptr, ptr %19, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc4
  %119 = phi ptr [ %114, %.noexc4 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %128

121:                                              ; preds = %87
  %122 = load ptr, ptr %19, align 8
  %123 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %121
  %124 = getelementptr i8, ptr %123, i64 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %124, i32 0, i32 0
  store i64 8, ptr %125, align 8
  %126 = getelementptr { i64, ptr, ptr }, ptr %124, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %126, align 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %124, i32 0, i32 2
  store ptr %122, ptr %127, align 8
  br label %128

128:                                              ; preds = %120, %.noexc8
  %129 = phi ptr [ %124, %.noexc8 ], [ %119, %120 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %85, %130
  %132 = phi ptr [ %129, %130 ], [ %86, %85 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc12, %133
  %135 = phi ptr [ %132, %133 ], [ %79, %.noexc12 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc1, %136
  %138 = phi ptr [ %135, %136 ], [ %70, %.noexc1 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc, %139
  %141 = phi ptr [ %138, %139 ], [ %61, %.noexc ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc3, %142
  %144 = phi ptr [ %141, %142 ], [ %52, %.noexc3 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc6, %145
  %147 = phi ptr [ %144, %145 ], [ %43, %.noexc6 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc7, %148
  %150 = phi ptr [ %147, %148 ], [ %34, %.noexc7 ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc11, %151
  %153 = phi ptr [ %150, %151 ], [ %25, %.noexc11 ]
  br label %154

154:                                              ; preds = %152
  store ptr %153, ptr %2, align 8
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8))
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %154
  call void @__cxa_end_catch()
  br label %155

155:                                              ; preds = %15, %.noexc5
  %156 = load i1, ptr %3, align 1
  br i1 %156, label %157, label %188

157:                                              ; preds = %155
  %158 = load ptr, ptr @bare, align 8
  %159 = call i1 @countIs(ptr %158, double 1.000000e+00)
  br i1 %159, label %160, label %178

160:                                              ; preds = %157
  %161 = load ptr, ptr @bare, align 8
  %162 = load ptr, ptr %161, align 8
  %163 = getelementptr %Node, ptr %162, i32 0, i32 2
  %164 = load ptr, ptr %163, align 8
  %165 = ptrtoint ptr %164 to i64
  %166 = icmp ne i64 %165, 0
  %167 = zext i1 %166 to i32
  %168 = and i32 %167, 1
  %169 = icmp ne i32 %168, 0
  br i1 %169, label %170, label %173

170:                                              ; preds = %160
  %171 = call i32 @strcmp(ptr %164, ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8))
  %172 = icmp eq i32 %171, 0
  br label %175

173:                                              ; preds = %160
  %174 = icmp eq i64 %165, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16713252134513893189, i64 8) to i64)
  br label %175

175:                                              ; preds = %170, %173
  %176 = phi i1 [ %174, %173 ], [ %172, %170 ]
  br label %177

177:                                              ; preds = %175
  br label %178

178:                                              ; preds = %177, %157
  %179 = phi i1 [ %176, %177 ], [ %159, %157 ]
  br i1 %179, label %180, label %190

180:                                              ; preds = %178
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

181:                                              ; preds = %190, %188, %186, %184, %182
  unreachable

182:                                              ; preds = %0
  %183 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7998919652572515854, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 58, ptr null)
  br label %181

184:                                              ; preds = %6
  %185 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5341766768668344537, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 59, ptr null)
  br label %181

186:                                              ; preds = %9
  %187 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_36228825792445631, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 60, ptr null)
  br label %181

188:                                              ; preds = %155
  %189 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6211200806403035524, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 69, ptr null)
  br label %181

190:                                              ; preds = %178
  %191 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3281035815395069118, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4585100004468534399, i64 8), i32 70, ptr null)
  br label %181

endcatch.resume:                                  ; preds = %76, %22, %99, %16, %121, %31, %40, %154, %111, %49, %101, %67, %58
  %192 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %192
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

