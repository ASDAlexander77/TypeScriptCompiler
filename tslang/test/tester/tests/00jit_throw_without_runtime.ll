; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Failure = type { ptr, double }

$Failure..size = comdat any

$_ZTSP7Failure = comdat any

$_ZTIP7Failure = comdat any

$_ZTS7Failure = comdat any

$_ZTI7Failure = comdat any

@s_9133338390725077489 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFailure\00", align 8
@Failure..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_9133338390725077489, i64 8)
@Failure..size = global i64 16, comdat
@Failure..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Failure..instanceOf, ptr @Failure..new, ptr @Failure..rtti, ptr @Failure..size }
@_ZTIi = external constant ptr
@_ZTIPc = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP7Failure = linkonce_odr constant [10 x i8] c"P7Failure\00", comdat
@_ZTIP7Failure = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP7Failure, i32 1073741824, ptr @_ZTI7Failure, ptr @.eh.copy.box.P7Failure }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS7Failure = linkonce_odr constant [9 x i8] c"7Failure\00", comdat
@s_18015793717152399486 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtext\00", align 8
@s_14478803536488843468 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFALL DONE\00", align 8
@m_2082538791733186936 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery throw is caught\00", align 8
@f_5699888325272079283 = internal constant [117 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00jit_throw_without_runtime.ts\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTI7Failure = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS7Failure }, comdat
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
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

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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
define void @Failure.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Failure, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @throwNumber() #1 {
  %1 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %1, align 4
  call void @__cxa_throw(ptr %1, ptr @_ZTIi, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @throwString() #1 {
  %1 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_18015793717152399486, i64 8), ptr %1, align 8
  call void @__cxa_throw(ptr %1, ptr @_ZTIPc, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
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
define void @throwClass() #1 {
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %Failure, ptr %2, i32 0, i32 0
  store ptr @Failure..vtbl, ptr %3, align 8
  call void @Failure.constructor(ptr %2, double 3.000000e+00)
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %2, ptr %4, align 8
  call void @__cxa_throw(ptr %4, ptr @_ZTIP7Failure, ptr null)
  br label %5

5:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store i32 0, ptr %7, align 4
  invoke void @throwNumber()
          to label %8 unwind label %9

8:                                                ; preds = %0
  br label %150

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc33
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc33
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %24
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
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %33
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
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %42
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %51
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
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %60
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
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %69
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
  invoke void %90(ptr %1, ptr %12)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %92
  %93 = load ptr, ptr %1, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc12, %.noexc8
  %102 = phi ptr [ %97, %.noexc8 ], [ %93, %.noexc12 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc10
  %112 = phi ptr [ %107, %.noexc10 ], [ %102, %103 ]
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

127:                                              ; preds = %.noexc16, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc16 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc19, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc19 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc26, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc26 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc24, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc24 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc20, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc20 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc23, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc23 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %6, align 8
  %148 = load i32, ptr %7, align 4
  %149 = add i32 %148, 1
  store i32 %149, ptr %7, align 4
  call void @__cxa_end_catch()
  br label %150

150:                                              ; preds = %8, %147
  invoke void @throwString()
          to label %151 unwind label %152

151:                                              ; preds = %150
  br label %293

152:                                              ; preds = %150
  %153 = landingpad { ptr, i32 }
          catch ptr null
  %154 = extractvalue { ptr, i32 } %153, 0
  %155 = call ptr @__cxa_begin_catch(ptr %154)
  %156 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc34 unwind label %endcatch.resume

.noexc34:                                         ; preds = %152
  %157 = icmp eq ptr %156, @_ZTIi
  br i1 %157, label %158, label %165

158:                                              ; preds = %.noexc34
  %159 = load i32, ptr %155, align 4
  %160 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %158
  %161 = getelementptr i8, ptr %160, i64 8
  %162 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 0
  store i64 4, ptr %162, align 8
  %163 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %163, align 8
  %164 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 2
  store i32 %159, ptr %164, align 4
  br label %288

165:                                              ; preds = %.noexc34
  %166 = icmp eq ptr %156, @_ZTIj
  br i1 %166, label %167, label %174

167:                                              ; preds = %165
  %168 = load i32, ptr %155, align 4
  %169 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %167
  %170 = getelementptr i8, ptr %169, i64 8
  %171 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 0
  store i64 4, ptr %171, align 8
  %172 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %172, align 8
  %173 = getelementptr { i64, ptr, i32 }, ptr %170, i32 0, i32 2
  store i32 %168, ptr %173, align 4
  br label %285

174:                                              ; preds = %165
  %175 = icmp eq ptr %156, @_ZTId
  br i1 %175, label %176, label %183

176:                                              ; preds = %174
  %177 = load double, ptr %155, align 8
  %178 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc31 unwind label %endcatch.resume

.noexc31:                                         ; preds = %176
  %179 = getelementptr i8, ptr %178, i64 8
  %180 = getelementptr { i64, ptr, double }, ptr %179, i32 0, i32 0
  store i64 8, ptr %180, align 8
  %181 = getelementptr { i64, ptr, double }, ptr %179, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %181, align 8
  %182 = getelementptr { i64, ptr, double }, ptr %179, i32 0, i32 2
  store double %177, ptr %182, align 8
  br label %282

183:                                              ; preds = %174
  %184 = icmp eq ptr %156, @_ZTIf
  br i1 %184, label %185, label %192

185:                                              ; preds = %183
  %186 = load float, ptr %155, align 4
  %187 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %185
  %188 = getelementptr i8, ptr %187, i64 8
  %189 = getelementptr { i64, ptr, float }, ptr %188, i32 0, i32 0
  store i64 4, ptr %189, align 8
  %190 = getelementptr { i64, ptr, float }, ptr %188, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %190, align 8
  %191 = getelementptr { i64, ptr, float }, ptr %188, i32 0, i32 2
  store float %186, ptr %191, align 4
  br label %279

192:                                              ; preds = %183
  %193 = icmp eq ptr %156, @_ZTIb
  br i1 %193, label %194, label %201

194:                                              ; preds = %192
  %195 = load i1, ptr %155, align 1
  %196 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %194
  %197 = getelementptr i8, ptr %196, i64 8
  %198 = getelementptr { i64, ptr, i1 }, ptr %197, i32 0, i32 0
  store i64 1, ptr %198, align 8
  %199 = getelementptr { i64, ptr, i1 }, ptr %197, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %199, align 8
  %200 = getelementptr { i64, ptr, i1 }, ptr %197, i32 0, i32 2
  store i1 %195, ptr %200, align 1
  br label %276

201:                                              ; preds = %192
  %202 = icmp eq ptr %156, @_ZTIx
  br i1 %202, label %203, label %210

203:                                              ; preds = %201
  %204 = load i64, ptr %155, align 8
  %205 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %203
  %206 = getelementptr i8, ptr %205, i64 8
  %207 = getelementptr { i64, ptr, i64 }, ptr %206, i32 0, i32 0
  store i64 8, ptr %207, align 8
  %208 = getelementptr { i64, ptr, i64 }, ptr %206, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %208, align 8
  %209 = getelementptr { i64, ptr, i64 }, ptr %206, i32 0, i32 2
  store i64 %204, ptr %209, align 8
  br label %273

210:                                              ; preds = %201
  %211 = icmp eq ptr %156, @_ZTIPc
  br i1 %211, label %212, label %219

212:                                              ; preds = %210
  %213 = load ptr, ptr %155, align 8
  %214 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %212
  %215 = getelementptr i8, ptr %214, i64 8
  %216 = getelementptr { i64, ptr, ptr }, ptr %215, i32 0, i32 0
  store i64 8, ptr %216, align 8
  %217 = getelementptr { i64, ptr, ptr }, ptr %215, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %217, align 8
  %218 = getelementptr { i64, ptr, ptr }, ptr %215, i32 0, i32 2
  store ptr %213, ptr %218, align 8
  br label %270

219:                                              ; preds = %210
  %220 = icmp eq ptr %156, @_ZTIPv
  br i1 %220, label %221, label %223

221:                                              ; preds = %219
  %222 = load ptr, ptr %155, align 8
  br label %267

223:                                              ; preds = %219
  %224 = load ptr, ptr %156, align 8
  %225 = icmp eq ptr %224, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %225, label %226, label %257

226:                                              ; preds = %223
  %227 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %156, i32 0, i32 2
  %228 = load i32, ptr %227, align 4
  %229 = and i32 %228, 1073741824
  %230 = icmp ne i32 %229, 0
  br i1 %230, label %231, label %247

231:                                              ; preds = %226
  %232 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %156, i32 0, i32 4
  %233 = load ptr, ptr %232, align 8
  %234 = icmp ne ptr %233, null
  br i1 %234, label %235, label %237

235:                                              ; preds = %231
  invoke void %233(ptr %2, ptr %155)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %235
  %236 = load ptr, ptr %2, align 8
  br label %244

237:                                              ; preds = %231
  %238 = load ptr, ptr %155, align 8
  %239 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %237
  %240 = getelementptr i8, ptr %239, i64 8
  %241 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 0
  store i64 8, ptr %241, align 8
  %242 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %242, align 8
  %243 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 2
  store ptr %238, ptr %243, align 8
  br label %244

244:                                              ; preds = %.noexc11, %.noexc9
  %245 = phi ptr [ %240, %.noexc9 ], [ %236, %.noexc11 ]
  br label %246

246:                                              ; preds = %244
  br label %254

247:                                              ; preds = %226
  %248 = load ptr, ptr %155, align 8
  %249 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %247
  %250 = getelementptr i8, ptr %249, i64 8
  %251 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 0
  store i64 8, ptr %251, align 8
  %252 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %252, align 8
  %253 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 2
  store ptr %248, ptr %253, align 8
  br label %254

254:                                              ; preds = %246, %.noexc17
  %255 = phi ptr [ %250, %.noexc17 ], [ %245, %246 ]
  br label %256

256:                                              ; preds = %254
  br label %264

257:                                              ; preds = %223
  %258 = load ptr, ptr %155, align 8
  %259 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %257
  %260 = getelementptr i8, ptr %259, i64 8
  %261 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 0
  store i64 8, ptr %261, align 8
  %262 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %262, align 8
  %263 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 2
  store ptr %258, ptr %263, align 8
  br label %264

264:                                              ; preds = %256, %.noexc6
  %265 = phi ptr [ %260, %.noexc6 ], [ %255, %256 ]
  br label %266

266:                                              ; preds = %264
  br label %267

267:                                              ; preds = %221, %266
  %268 = phi ptr [ %265, %266 ], [ %222, %221 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %.noexc27, %269
  %271 = phi ptr [ %268, %269 ], [ %215, %.noexc27 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc30, %272
  %274 = phi ptr [ %271, %272 ], [ %206, %.noexc30 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc28, %275
  %277 = phi ptr [ %274, %275 ], [ %197, %.noexc28 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc32, %278
  %280 = phi ptr [ %277, %278 ], [ %188, %.noexc32 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc31, %281
  %283 = phi ptr [ %280, %281 ], [ %179, %.noexc31 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc22, %284
  %286 = phi ptr [ %283, %284 ], [ %170, %.noexc22 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc25, %287
  %289 = phi ptr [ %286, %287 ], [ %161, %.noexc25 ]
  br label %290

290:                                              ; preds = %288
  store ptr %289, ptr %5, align 8
  %291 = load i32, ptr %7, align 4
  %292 = add i32 %291, 1
  store i32 %292, ptr %7, align 4
  call void @__cxa_end_catch()
  br label %293

293:                                              ; preds = %151, %290
  invoke void @throwClass()
          to label %294 unwind label %295

294:                                              ; preds = %293
  br label %436

295:                                              ; preds = %293
  %296 = landingpad { ptr, i32 }
          catch ptr null
  %297 = extractvalue { ptr, i32 } %296, 0
  %298 = call ptr @__cxa_begin_catch(ptr %297)
  %299 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %295
  %300 = icmp eq ptr %299, @_ZTIi
  br i1 %300, label %301, label %308

301:                                              ; preds = %.noexc13
  %302 = load i32, ptr %298, align 4
  %303 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %301
  %304 = getelementptr i8, ptr %303, i64 8
  %305 = getelementptr { i64, ptr, i32 }, ptr %304, i32 0, i32 0
  store i64 4, ptr %305, align 8
  %306 = getelementptr { i64, ptr, i32 }, ptr %304, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %306, align 8
  %307 = getelementptr { i64, ptr, i32 }, ptr %304, i32 0, i32 2
  store i32 %302, ptr %307, align 4
  br label %431

308:                                              ; preds = %.noexc13
  %309 = icmp eq ptr %299, @_ZTIj
  br i1 %309, label %310, label %317

310:                                              ; preds = %308
  %311 = load i32, ptr %298, align 4
  %312 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %310
  %313 = getelementptr i8, ptr %312, i64 8
  %314 = getelementptr { i64, ptr, i32 }, ptr %313, i32 0, i32 0
  store i64 4, ptr %314, align 8
  %315 = getelementptr { i64, ptr, i32 }, ptr %313, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %315, align 8
  %316 = getelementptr { i64, ptr, i32 }, ptr %313, i32 0, i32 2
  store i32 %311, ptr %316, align 4
  br label %428

317:                                              ; preds = %308
  %318 = icmp eq ptr %299, @_ZTId
  br i1 %318, label %319, label %326

319:                                              ; preds = %317
  %320 = load double, ptr %298, align 8
  %321 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %319
  %322 = getelementptr i8, ptr %321, i64 8
  %323 = getelementptr { i64, ptr, double }, ptr %322, i32 0, i32 0
  store i64 8, ptr %323, align 8
  %324 = getelementptr { i64, ptr, double }, ptr %322, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %324, align 8
  %325 = getelementptr { i64, ptr, double }, ptr %322, i32 0, i32 2
  store double %320, ptr %325, align 8
  br label %425

326:                                              ; preds = %317
  %327 = icmp eq ptr %299, @_ZTIf
  br i1 %327, label %328, label %335

328:                                              ; preds = %326
  %329 = load float, ptr %298, align 4
  %330 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %328
  %331 = getelementptr i8, ptr %330, i64 8
  %332 = getelementptr { i64, ptr, float }, ptr %331, i32 0, i32 0
  store i64 4, ptr %332, align 8
  %333 = getelementptr { i64, ptr, float }, ptr %331, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %333, align 8
  %334 = getelementptr { i64, ptr, float }, ptr %331, i32 0, i32 2
  store float %329, ptr %334, align 4
  br label %422

335:                                              ; preds = %326
  %336 = icmp eq ptr %299, @_ZTIb
  br i1 %336, label %337, label %344

337:                                              ; preds = %335
  %338 = load i1, ptr %298, align 1
  %339 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %337
  %340 = getelementptr i8, ptr %339, i64 8
  %341 = getelementptr { i64, ptr, i1 }, ptr %340, i32 0, i32 0
  store i64 1, ptr %341, align 8
  %342 = getelementptr { i64, ptr, i1 }, ptr %340, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %342, align 8
  %343 = getelementptr { i64, ptr, i1 }, ptr %340, i32 0, i32 2
  store i1 %338, ptr %343, align 1
  br label %419

344:                                              ; preds = %335
  %345 = icmp eq ptr %299, @_ZTIx
  br i1 %345, label %346, label %353

346:                                              ; preds = %344
  %347 = load i64, ptr %298, align 8
  %348 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %346
  %349 = getelementptr i8, ptr %348, i64 8
  %350 = getelementptr { i64, ptr, i64 }, ptr %349, i32 0, i32 0
  store i64 8, ptr %350, align 8
  %351 = getelementptr { i64, ptr, i64 }, ptr %349, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %351, align 8
  %352 = getelementptr { i64, ptr, i64 }, ptr %349, i32 0, i32 2
  store i64 %347, ptr %352, align 8
  br label %416

353:                                              ; preds = %344
  %354 = icmp eq ptr %299, @_ZTIPc
  br i1 %354, label %355, label %362

355:                                              ; preds = %353
  %356 = load ptr, ptr %298, align 8
  %357 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %355
  %358 = getelementptr i8, ptr %357, i64 8
  %359 = getelementptr { i64, ptr, ptr }, ptr %358, i32 0, i32 0
  store i64 8, ptr %359, align 8
  %360 = getelementptr { i64, ptr, ptr }, ptr %358, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %360, align 8
  %361 = getelementptr { i64, ptr, ptr }, ptr %358, i32 0, i32 2
  store ptr %356, ptr %361, align 8
  br label %413

362:                                              ; preds = %353
  %363 = icmp eq ptr %299, @_ZTIPv
  br i1 %363, label %364, label %366

364:                                              ; preds = %362
  %365 = load ptr, ptr %298, align 8
  br label %410

366:                                              ; preds = %362
  %367 = load ptr, ptr %299, align 8
  %368 = icmp eq ptr %367, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %368, label %369, label %400

369:                                              ; preds = %366
  %370 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %299, i32 0, i32 2
  %371 = load i32, ptr %370, align 4
  %372 = and i32 %371, 1073741824
  %373 = icmp ne i32 %372, 0
  br i1 %373, label %374, label %390

374:                                              ; preds = %369
  %375 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %299, i32 0, i32 4
  %376 = load ptr, ptr %375, align 8
  %377 = icmp ne ptr %376, null
  br i1 %377, label %378, label %380

378:                                              ; preds = %374
  invoke void %376(ptr %3, ptr %298)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %378
  %379 = load ptr, ptr %3, align 8
  br label %387

380:                                              ; preds = %374
  %381 = load ptr, ptr %298, align 8
  %382 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc29 unwind label %endcatch.resume

.noexc29:                                         ; preds = %380
  %383 = getelementptr i8, ptr %382, i64 8
  %384 = getelementptr { i64, ptr, ptr }, ptr %383, i32 0, i32 0
  store i64 8, ptr %384, align 8
  %385 = getelementptr { i64, ptr, ptr }, ptr %383, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %385, align 8
  %386 = getelementptr { i64, ptr, ptr }, ptr %383, i32 0, i32 2
  store ptr %381, ptr %386, align 8
  br label %387

387:                                              ; preds = %.noexc18, %.noexc29
  %388 = phi ptr [ %383, %.noexc29 ], [ %379, %.noexc18 ]
  br label %389

389:                                              ; preds = %387
  br label %397

390:                                              ; preds = %369
  %391 = load ptr, ptr %298, align 8
  %392 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc35 unwind label %endcatch.resume

.noexc35:                                         ; preds = %390
  %393 = getelementptr i8, ptr %392, i64 8
  %394 = getelementptr { i64, ptr, ptr }, ptr %393, i32 0, i32 0
  store i64 8, ptr %394, align 8
  %395 = getelementptr { i64, ptr, ptr }, ptr %393, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %395, align 8
  %396 = getelementptr { i64, ptr, ptr }, ptr %393, i32 0, i32 2
  store ptr %391, ptr %396, align 8
  br label %397

397:                                              ; preds = %389, %.noexc35
  %398 = phi ptr [ %393, %.noexc35 ], [ %388, %389 ]
  br label %399

399:                                              ; preds = %397
  br label %407

400:                                              ; preds = %366
  %401 = load ptr, ptr %298, align 8
  %402 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %400
  %403 = getelementptr i8, ptr %402, i64 8
  %404 = getelementptr { i64, ptr, ptr }, ptr %403, i32 0, i32 0
  store i64 8, ptr %404, align 8
  %405 = getelementptr { i64, ptr, ptr }, ptr %403, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %405, align 8
  %406 = getelementptr { i64, ptr, ptr }, ptr %403, i32 0, i32 2
  store ptr %401, ptr %406, align 8
  br label %407

407:                                              ; preds = %399, %.noexc21
  %408 = phi ptr [ %403, %.noexc21 ], [ %398, %399 ]
  br label %409

409:                                              ; preds = %407
  br label %410

410:                                              ; preds = %364, %409
  %411 = phi ptr [ %408, %409 ], [ %365, %364 ]
  br label %412

412:                                              ; preds = %410
  br label %413

413:                                              ; preds = %.noexc4, %412
  %414 = phi ptr [ %411, %412 ], [ %358, %.noexc4 ]
  br label %415

415:                                              ; preds = %413
  br label %416

416:                                              ; preds = %.noexc3, %415
  %417 = phi ptr [ %414, %415 ], [ %349, %.noexc3 ]
  br label %418

418:                                              ; preds = %416
  br label %419

419:                                              ; preds = %.noexc7, %418
  %420 = phi ptr [ %417, %418 ], [ %340, %.noexc7 ]
  br label %421

421:                                              ; preds = %419
  br label %422

422:                                              ; preds = %.noexc15, %421
  %423 = phi ptr [ %420, %421 ], [ %331, %.noexc15 ]
  br label %424

424:                                              ; preds = %422
  br label %425

425:                                              ; preds = %.noexc14, %424
  %426 = phi ptr [ %423, %424 ], [ %322, %.noexc14 ]
  br label %427

427:                                              ; preds = %425
  br label %428

428:                                              ; preds = %.noexc1, %427
  %429 = phi ptr [ %426, %427 ], [ %313, %.noexc1 ]
  br label %430

430:                                              ; preds = %428
  br label %431

431:                                              ; preds = %.noexc5, %430
  %432 = phi ptr [ %429, %430 ], [ %304, %.noexc5 ]
  br label %433

433:                                              ; preds = %431
  store ptr %432, ptr %4, align 8
  %434 = load i32, ptr %7, align 4
  %435 = add i32 %434, 1
  store i32 %435, ptr %7, align 4
  call void @__cxa_end_catch()
  br label %436

436:                                              ; preds = %294, %433
  %437 = load i32, ptr %7, align 4
  %438 = icmp eq i32 %437, 3
  br i1 %438, label %439, label %441

439:                                              ; preds = %436
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14478803536488843468, i64 8))
  ret i32 0

440:                                              ; preds = %441
  unreachable

441:                                              ; preds = %436
  %442 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2082538791733186936, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5699888325272079283, i64 8), i32 26, ptr null)
  br label %440

endcatch.resume:                                  ; preds = %390, %152, %9, %185, %176, %203, %380, %194, %212, %42, %158, %33, %15, %167, %400, %24, %60, %378, %247, %69, %328, %319, %295, %92, %235, %104, %237, %94, %337, %257, %301, %355, %346, %114, %310, %51
  %443 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %443
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

