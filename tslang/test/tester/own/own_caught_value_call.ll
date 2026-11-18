; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Failure = type { ptr, i32 }

$"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>" = comdat any

$Failure..size = comdat any

$_ZTSP7Failure = comdat any

$_ZTIP7Failure = comdat any

$_ZTS7Failure = comdat any

$_ZTI7Failure = comdat any

@s_9133338390725077489 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFailure\00", align 8
@Failure..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_9133338390725077489, i64 8)
@Failure..size = global i64 16, comdat
@Failure..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Failure..instanceOf, ptr @Failure..new, ptr @Failure..rtti, ptr @Failure..size }
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP7Failure = linkonce_odr constant [10 x i8] c"P7Failure\00", comdat
@_ZTIP7Failure = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP7Failure, i32 1073741824, ptr @_ZTI7Failure, ptr @.eh.copy.box.P7Failure }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS7Failure = linkonce_odr constant [9 x i8] c"7Failure\00", comdat
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@__false__ = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfalse\00", align 8
@__true__ = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtrue\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTI7Failure = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS7Failure }, comdat
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@td_2515608230574705588_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_2515608230574705588, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
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

declare void @puts(ptr)

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

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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

define internal void @tsrel_2515608230574705588(ptr %0) {
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

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Failure..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Failure..rtti, align 8
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
define ptr @Failure..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Failure, ptr %3, i32 0, i32 0
  store ptr @Failure..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Failure.constructor(ptr %0, i32 %1) #1 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %3, align 4
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Failure, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %3, align 4
  store i32 %7, ptr %6, align 4
  ret void
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P7Failure(ptr %0, ptr %1) #1 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_2515608230574705588_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define void @fail(i32 %0) #1 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = icmp sgt i32 %3, 0
  br i1 %4, label %5, label %11

5:                                                ; preds = %1
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Failure, ptr %7, i32 0, i32 0
  store ptr @Failure..vtbl, ptr %8, align 8
  %9 = load i32, ptr %2, align 4
  call void @Failure.constructor(ptr %7, i32 %9)
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %7, ptr %10, align 8
  call void @__cxa_throw(ptr %10, ptr @_ZTIP7Failure, ptr null)
  br label %12

11:                                               ; preds = %1
  ret void

12:                                               ; preds = %5
  unreachable
}

; Function Attrs: mustprogress
declare zeroext i1 @checkCode(ptr, i32) #1

; Function Attrs: mustprogress
define linkonce_odr ptr @"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr { i64, ptr, i8 }, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %1
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %1
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %42

21:                                               ; preds = %20
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 2
  %24 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 0
  %25 = load i64, ptr %24, align 8
  %26 = load i64, ptr %23, align 8
  %27 = icmp uge i64 %25, 8
  %28 = select i1 %27, i64 0, i64 %25
  %29 = mul i64 %28, 8
  %30 = shl i64 1, %29
  %31 = sub i64 %30, 1
  %32 = and i64 %26, %31
  %33 = select i1 %27, i64 %26, i64 %32
  %34 = inttoptr i64 %33 to ptr
  %35 = getelementptr ptr, ptr %34, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr ptr, ptr %36, i32 0
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr @Failure..rtti, align 8
  %40 = call i1 %38(ptr %34, ptr %39)
  br i1 %40, label %41, label %91

41:                                               ; preds = %21
  store ptr %34, ptr %3, align 8
  br label %93

42:                                               ; preds = %20
  %43 = load ptr, ptr %2, align 8
  %44 = getelementptr { i64, ptr, i8 }, ptr %43, i32 0, i32 1
  %45 = load ptr, ptr %44, align 8
  %46 = ptrtoint ptr %45 to i64
  %47 = icmp ne i64 %46, 0
  %48 = zext i1 %47 to i32
  %49 = and i32 %48, 1
  %50 = icmp ne i32 %49, 0
  br i1 %50, label %51, label %54

51:                                               ; preds = %42
  %52 = call i32 @strcmp(ptr %45, ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8))
  %53 = icmp eq i32 %52, 0
  br label %56

54:                                               ; preds = %42
  %55 = icmp eq i64 %46, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8) to i64)
  br label %56

56:                                               ; preds = %51, %54
  %57 = phi i1 [ %55, %54 ], [ %53, %51 ]
  br label %58

58:                                               ; preds = %56
  br i1 %57, label %59, label %91

59:                                               ; preds = %58
  %60 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %4, i32 0, i32 2
  %61 = load { ptr, ptr }, ptr %60, align 8
  %62 = extractvalue { ptr, ptr } %61, 1
  %63 = ptrtoint ptr %62 to i64
  %64 = icmp ne i64 %63, 0
  br i1 %64, label %65, label %72

65:                                               ; preds = %59
  %66 = extractvalue { ptr, ptr } %61, 0
  %67 = extractvalue { ptr, ptr } %61, 1
  %68 = getelementptr ptr, ptr %66, i32 1
  %69 = load ptr, ptr %68, align 8
  %70 = load ptr, ptr @Failure..rtti, align 8
  %71 = call i1 %69(ptr %67, ptr %70)
  br label %72

72:                                               ; preds = %65, %59
  %73 = phi i1 [ %71, %65 ], [ false, %59 ]
  br i1 %73, label %74, label %91

74:                                               ; preds = %72
  %75 = extractvalue { ptr, ptr } %61, 1
  %76 = ptrtoint ptr %75 to i64
  %77 = icmp ne i64 %76, 0
  br i1 %77, label %78, label %85

78:                                               ; preds = %74
  %79 = extractvalue { ptr, ptr } %61, 0
  %80 = extractvalue { ptr, ptr } %61, 1
  %81 = getelementptr ptr, ptr %79, i32 1
  %82 = load ptr, ptr %81, align 8
  %83 = load ptr, ptr @Failure..rtti, align 8
  %84 = call i1 %82(ptr %80, ptr %83)
  br label %85

85:                                               ; preds = %78, %74
  %86 = phi i1 [ %84, %78 ], [ false, %74 ]
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = extractvalue { ptr, ptr } %61, 1
  store ptr %88, ptr %3, align 8
  br label %93

89:                                               ; preds = %85
  %90 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %90, align 8
  call void @__cxa_throw(ptr %90, ptr @_ZTIPc, ptr null)
  br label %95

91:                                               ; preds = %21, %72, %58
  %92 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %92, align 8
  call void @__cxa_throw(ptr %92, ptr @_ZTIPc, ptr null)
  br label %95

93:                                               ; preds = %41, %87
  %94 = load ptr, ptr %3, align 8
  ret ptr %94

95:                                               ; preds = %91, %89
  unreachable
}

; Function Attrs: mustprogress
define i1 @caught(i32 %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca i32, align 4
  %6 = alloca i1, align 1
  store i32 %0, ptr %5, align 4
  store i1 false, ptr %4, align 1
  %7 = load i32, ptr %5, align 4
  invoke void @fail(i32 %7)
          to label %8 unwind label %9

8:                                                ; preds = %1
  br label %160

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc13
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc13
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %142

31:                                               ; preds = %22
  %32 = icmp eq ptr %13, @_ZTId
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load double, ptr %12, align 8
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 0
  store i64 8, ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %36, i32 0, i32 2
  store double %34, ptr %39, align 8
  br label %139

40:                                               ; preds = %31
  %41 = icmp eq ptr %13, @_ZTIf
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load float, ptr %12, align 4
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 0
  store i64 4, ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %45, i32 0, i32 2
  store float %43, ptr %48, align 4
  br label %136

49:                                               ; preds = %40
  %50 = icmp eq ptr %13, @_ZTIb
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load i1, ptr %12, align 1
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 0
  store i64 1, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %54, i32 0, i32 2
  store i1 %52, ptr %57, align 1
  br label %133

58:                                               ; preds = %49
  %59 = icmp eq ptr %13, @_ZTIx
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i64, ptr %12, align 8
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 0
  store i64 8, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %63, i32 0, i32 2
  store i64 %61, ptr %66, align 8
  br label %130

67:                                               ; preds = %58
  %68 = icmp eq ptr %13, @_ZTIPc
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load ptr, ptr %12, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %72, i32 0, i32 2
  store ptr %70, ptr %75, align 8
  br label %127

76:                                               ; preds = %67
  %77 = icmp eq ptr %13, @_ZTIPv
  br i1 %77, label %78, label %80

78:                                               ; preds = %76
  %79 = load ptr, ptr %12, align 8
  br label %124

80:                                               ; preds = %76
  %81 = load ptr, ptr %13, align 8
  %82 = icmp eq ptr %81, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %82, label %83, label %114

83:                                               ; preds = %80
  %84 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 2
  %85 = load i32, ptr %84, align 4
  %86 = and i32 %85, 1073741824
  %87 = icmp ne i32 %86, 0
  br i1 %87, label %88, label %104

88:                                               ; preds = %83
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %13, i32 0, i32 4
  %90 = load ptr, ptr %89, align 8
  %91 = icmp ne ptr %90, null
  br i1 %91, label %92, label %94

92:                                               ; preds = %88
  invoke void %90(ptr %2, ptr %12)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %92
  %93 = load ptr, ptr %2, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc8, %.noexc3
  %102 = phi ptr [ %97, %.noexc3 ], [ %93, %.noexc8 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc6
  %112 = phi ptr [ %107, %.noexc6 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc2
  %122 = phi ptr [ %117, %.noexc2 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc1, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc1 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc5, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc5 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc4, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc4 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc9, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc9 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc7, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc7 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc12, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc12 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %3, align 8
  %148 = load ptr, ptr %3, align 8
  %149 = invoke ptr @"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>"(ptr %148)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %147
  %150 = load i32, ptr %5, align 4
  %151 = invoke i1 @checkCode(ptr %149, i32 %150)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  store i1 %151, ptr %4, align 1
  %152 = load i1, ptr %4, align 1
  br i1 %152, label %153, label %158

153:                                              ; preds = %.noexc11
  %154 = getelementptr %Failure, ptr %149, i32 0, i32 1
  %155 = load i32, ptr %154, align 4
  %156 = load i32, ptr %5, align 4
  %157 = icmp eq i32 %155, %156
  br label %158

158:                                              ; preds = %153, %.noexc11
  %159 = phi i1 [ %157, %153 ], [ %152, %.noexc11 ]
  store i1 %159, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %160

160:                                              ; preds = %8, %158
  %161 = load i1, ptr %4, align 1
  store i1 %161, ptr %6, align 1
  %162 = load i1, ptr %6, align 1
  ret i1 %162

endcatch.resume:                                  ; preds = %9, %15, %.noexc10, %147, %42, %92, %33, %104, %60, %51, %94, %114, %69, %24
  %163 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %163
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i1 @caught(i32 0)
  %2 = select i1 %1, ptr getelementptr inbounds nuw (i8, ptr @__true__, i64 8), ptr getelementptr inbounds nuw (i8, ptr @__false__, i64 8)
  %3 = icmp eq ptr %2, null
  %4 = select i1 %3, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %2
  call void @puts(ptr %4)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0
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

