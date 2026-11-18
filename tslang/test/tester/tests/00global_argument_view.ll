; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Base = type { ptr, ptr, double }
%C = type { %Base, ptr }
%H = type { ptr, ptr }

$Base..size = comdat any

$C..size = comdat any

$H..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_13953526755221441880 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBase\00", align 8
@s_12638224583951001170 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFC\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@s_12638189399578898418 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_3978046374080941754 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNaN\00", align 8
@s_865306855574198937 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInfinity\00", align 8
@s_5514277459305441004 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-Infinity\00", align 8
@s_555417232352566889 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s\00", align 8
@s_12638135523509116079 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF0\00", align 8
@s_12638121229857949336 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-\00", align 8
@s_13533418361890864287 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%.*e\00", align 8
@s_565227158805101677 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%c%.*s\00", align 8
@s_3707881812392243519 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF000000000000000000000\00", align 8
@s_14597275362824401209 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%s%.*s\00", align 8
@s_2324455066069071285 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%.*s.%s\00", align 8
@s_3663401783476796495 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s0.%.*s%s\00", align 8
@s_12638124528392833969 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.\00", align 8
@s_17791929768168046296 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%c%s%se%+d\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_4050726357520103011 = internal constant [112 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00global_argument_view.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_16940945161124618027 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFjunk\00", align 8
@s_2770870231077248403 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthrown\00", align 8
@m_9664403948889819029 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan interface over a global, in a loop\00", align 8
@m_6521797870148510405 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan object literal holding a global, in a loop\00", align 8
@s_17761974936599606694 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc31\00", align 8
@m_8973867690343378340 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa later argument throws after the assignment\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_7995604983552920533 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan interface over a global\00", align 8
@m_7621023459670631976 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa global as its base class\00", align 8
@m_14966932188815007743 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan interface over a global's field\00", align 8
@m_1747183150290012101 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan object literal holding a global\00", align 8
@m_5566661821194917391 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan object literal holding a global's field\00", align 8
@m_7978875794277107671 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan object literal given as a record\00", align 8
@m_12599839212798849691 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe globals' last values\00", align 8
@Base..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_13953526755221441880, i64 8)
@Base..size = global i64 24, comdat
@Base..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Base..instanceOf, ptr @Base..new, ptr @Base..rtti, ptr @Base..size }
@C..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638224583951001170, i64 8)
@C..size = global i64 32, comdat
@C..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @C..instanceOf, ptr @C..new, ptr @Base..rtti, ptr @Base..size, ptr @C..rtti, ptr @C..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 16, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@G = private global ptr undef
@GH = private global ptr undef
@_ZTIPc = external constant ptr
@C.IC..vtbl = private global { ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12330864546263951727_class, i64 40), ptr @C.IC..vtbl..instanceOf.dispatch, ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 8 to ptr) }
@td_12330864546263951727_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_12330864546263951727, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@IC.null..vtbl = private global { ptr, ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr null, ptr null }
@IHolder.1938816268306746371..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_1938816268306746371_object, i64 40), ptr @.instanceOf.none, ptr null }
@td_1938816268306746371_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrel_1938816268306746371, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_17033125819989208466 = internal constant [0 x double] undef
@a_0_2926441800300339944 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @G__cctor, ptr null }, { i32, ptr, ptr } { i32 1001, ptr @GH__cctor, ptr null }]

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @__cxa_end_catch()

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

define internal void @__ts_number_to_string(ptr %0, double %1) {
  %3 = alloca i8, i32 32, align 1
  %4 = alloca i8, i32 32, align 1
  %5 = fcmp olt double %1, 0.000000e+00
  %6 = fcmp uno double %1, %1
  br i1 %6, label %7, label %9

7:                                                ; preds = %2
  %8 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_3978046374080941754, i64 8))
  ret void

9:                                                ; preds = %2
  %10 = call double @llvm.fabs.f64(double %1)
  %11 = fcmp oeq double %10, 0x7FF0000000000000
  br i1 %11, label %12, label %15

12:                                               ; preds = %9
  %13 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_5514277459305441004, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_865306855574198937, i64 8)
  %14 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_555417232352566889, i64 8), ptr %13)
  ret void

15:                                               ; preds = %9
  %16 = fcmp oeq double %1, 0.000000e+00
  br i1 %16, label %17, label %19

17:                                               ; preds = %15
  %18 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_12638135523509116079, i64 8))
  ret void

19:                                               ; preds = %15
  %20 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_12638121229857949336, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
  br label %21

21:                                               ; preds = %21, %19
  %22 = phi i32 [ %28, %21 ], [ 0, %19 ]
  %23 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %3, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_13533418361890864287, i64 8), i32 %22, double %10)
  %24 = call double @strtod(ptr %3, ptr null)
  %25 = fcmp oeq double %24, %10
  %26 = icmp sge i32 %22, 16
  %27 = or i1 %25, %26
  %28 = add i32 %22, 1
  br i1 %27, label %29, label %21

29:                                               ; preds = %21
  %30 = add i32 %22, 1
  %31 = icmp eq i32 %22, 0
  %32 = add i32 %22, 3
  %33 = select i1 %31, i32 2, i32 %32
  %34 = getelementptr i8, ptr %3, i32 %33
  %35 = call double @strtod(ptr %34, ptr null)
  %36 = fptosi double %35 to i32
  %37 = add i32 %36, 1
  %38 = load i8, ptr %3, align 1
  %39 = sext i8 %38 to i32
  %40 = getelementptr i8, ptr %3, i32 2
  %41 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %4, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_565227158805101677, i64 8), i32 %39, i32 %22, ptr %40)
  %42 = icmp sle i32 %37, 21
  %43 = icmp sle i32 %30, %37
  %44 = and i1 %43, %42
  br i1 %44, label %45, label %48

45:                                               ; preds = %29
  %46 = sub i32 %37, %30
  %47 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_14597275362824401209, i64 8), ptr %20, ptr %4, i32 %46, ptr getelementptr inbounds nuw (i8, ptr @s_3707881812392243519, i64 8))
  ret void

48:                                               ; preds = %29
  %49 = icmp sgt i32 %37, 0
  %50 = and i1 %49, %42
  br i1 %50, label %51, label %54

51:                                               ; preds = %48
  %52 = getelementptr i8, ptr %4, i32 %37
  %53 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_2324455066069071285, i64 8), ptr %20, i32 %37, ptr %4, ptr %52)
  ret void

54:                                               ; preds = %48
  %55 = icmp sle i32 %37, 0
  %56 = icmp sgt i32 %37, -6
  %57 = and i1 %56, %55
  br i1 %57, label %58, label %61

58:                                               ; preds = %54
  %59 = sub i32 0, %37
  %60 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_3663401783476796495, i64 8), ptr %20, i32 %59, ptr getelementptr inbounds nuw (i8, ptr @s_3707881812392243519, i64 8), ptr %4)
  ret void

61:                                               ; preds = %54
  %62 = icmp sgt i32 %30, 1
  %63 = select i1 %62, ptr getelementptr inbounds nuw (i8, ptr @s_12638124528392833969, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
  %64 = getelementptr i8, ptr %4, i32 1
  %65 = sub i32 %37, 1
  %66 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_17791929768168046296, i64 8), ptr %20, i32 %39, ptr %63, ptr %64, i32 %65)
  ret void
}

declare double @strtod(ptr, ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_1938816268306746371(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_12330864546263951727(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

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

define internal void @tsrel_8309307313031868504(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %9

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = getelementptr { ptr, i64, i64 }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  call void @__tslang_free_block(ptr %8)
  call void @__tslang_free_block(ptr %2)
  br label %9

9:                                                ; preds = %6, %4, %1
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

define internal void @tsrel_7788728893611325085(ptr %0) {
  %2 = getelementptr %Base, ptr %0, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %2)
  ret void
}

define internal void @tsrel_12330864546263951727(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %9

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %9

6:                                                ; preds = %4
  %7 = getelementptr %C, ptr %2, i32 0, i32 0
  call void @tsrel_7788728893611325085(ptr %7)
  %8 = getelementptr %C, ptr %2, i32 0, i32 1
  call void @tsrel_8309307313031868504(ptr %8)
  call void @__tslang_free_block(ptr %2)
  br label %9

9:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @G__cctor() {
  %1 = call ptr @mk(double 1.000000e+00)
  store ptr %1, ptr @G, align 8
  ret void
}

define internal void @GH__cctor() {
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %H, ptr %2, i32 0, i32 0
  store ptr @H..vtbl, ptr %3, align 8
  call void @H.constructor(ptr %2)
  store ptr %2, ptr @GH, align 8
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
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Base, ptr %3, i32 0, i32 0
  store ptr @Base..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Base.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Base, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Base, ptr %7, i32 0, i32 2
  %9 = load double, ptr %3, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr %Base, ptr %10, i32 0, i32 1
  %12 = load double, ptr %3, align 8
  %13 = call ptr @GC_malloc_atomic(i64 40)
  %14 = getelementptr i8, ptr %13, i64 8
  call void @__ts_number_to_string(ptr %14, double %12)
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr %16)
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %20, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr %16, i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  store ptr %23, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress
define void @C.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %C, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  call void @Base.constructor(ptr %6, double %7)
  %8 = load ptr, ptr %4, align 8
  %9 = getelementptr %C, ptr %8, i32 0, i32 1
  %10 = call ptr @GC_malloc(i64 8)
  %11 = getelementptr i8, ptr %10, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %11, ptr @a_0_17033125819989208466, i64 0, i1 false)
  %12 = call ptr @GC_malloc(i64 32)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 0
  store ptr %11, ptr %14, align 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 1
  store i64 0, ptr %15, align 8
  %16 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 2
  store i64 0, ptr %16, align 8
  store ptr %13, ptr %9, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @C..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @C..rtti, align 8
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
  %26 = getelementptr %C, ptr %25, i32 0, i32 0
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
define ptr @C..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %C, ptr %3, i32 0, i32 0
  %5 = getelementptr %Base, ptr %4, i32 0, i32 0
  store ptr @C..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define void @H.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 1
  %5 = call ptr @mk(double 3.000000e+00)
  store ptr %5, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @H..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @H..rtti, align 8
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
define ptr @H..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 0
  store ptr @H..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define ptr @mk(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 40)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %C, ptr %5, i32 0, i32 0
  %7 = getelementptr %Base, ptr %6, i32 0, i32 0
  store ptr @C..vtbl, ptr %7, align 8
  %8 = load double, ptr %2, align 8
  call void @C.constructor(ptr %5, double %8)
  %9 = getelementptr %C, ptr %5, i32 0, i32 1
  %10 = load double, ptr %2, align 8
  %11 = load ptr, ptr %9, align 8
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %16

13:                                               ; preds = %1
  %14 = call ptr @GC_malloc(i64 32)
  %15 = getelementptr i8, ptr %14, i64 8
  store ptr %15, ptr %9, align 8
  br label %17

16:                                               ; preds = %1
  br label %17

17:                                               ; preds = %13, %16
  %18 = phi ptr [ %11, %16 ], [ %15, %13 ]
  br label %19

19:                                               ; preds = %17
  %20 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 1
  %21 = load i64, ptr %20, align 8
  %22 = add i64 %21, 1
  %23 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 0
  %24 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 2
  %25 = load ptr, ptr %23, align 8
  %26 = load i64, ptr %24, align 8
  %27 = icmp ugt i64 %22, %26
  br i1 %27, label %28, label %52

28:                                               ; preds = %19
  %29 = mul i64 %26, 2
  %30 = icmp uge i64 %29, %22
  %31 = select i1 %30, i64 %29, i64 %22
  %32 = icmp ult i64 %31, 4
  %33 = select i1 %32, i64 4, i64 %31
  %34 = icmp ule i64 %22, 4294967295
  br i1 %34, label %35, label %59

35:                                               ; preds = %28
  %36 = icmp ule i64 %33, 2305843009213693951
  br i1 %36, label %37, label %61

37:                                               ; preds = %35
  %38 = mul i64 8, %33
  %39 = add i64 %38, 8
  %40 = icmp eq ptr %25, null
  %41 = getelementptr i8, ptr %25, i64 -8
  %42 = select i1 %40, ptr null, ptr %41
  %43 = call ptr @GC_realloc(ptr %42, i64 %39)
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = icmp ne ptr %44, null
  %46 = icmp eq i64 %38, 0
  %47 = or i1 %45, %46
  br i1 %47, label %48, label %63

48:                                               ; preds = %37
  %49 = mul i64 8, %26
  %50 = getelementptr i8, ptr %44, i64 %49
  %51 = sub i64 %38, %49
  call void @llvm.memset.p0.i64(ptr %50, i8 0, i64 %51, i1 false)
  store ptr %44, ptr %23, align 8
  store i64 %33, ptr %24, align 8
  br label %53

52:                                               ; preds = %19
  br label %53

53:                                               ; preds = %48, %52
  %54 = phi ptr [ %25, %52 ], [ %44, %48 ]
  br label %55

55:                                               ; preds = %53
  %56 = getelementptr double, ptr %54, i64 %21
  store double %10, ptr %56, align 8
  store i64 %22, ptr %20, align 8
  store ptr %5, ptr %3, align 8
  %57 = load ptr, ptr %3, align 8
  ret ptr %57

58:                                               ; preds = %63, %61, %59
  unreachable

59:                                               ; preds = %28
  %60 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 34, ptr null)
  br label %58

61:                                               ; preds = %35
  %62 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 34, ptr null)
  br label %58

63:                                               ; preds = %37
  %64 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 34, ptr null)
  br label %58
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 8)
  %4 = getelementptr i8, ptr %3, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_2926441800300339944, i64 0, i1 false)
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 0
  store ptr %4, ptr %7, align 8
  %8 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 1
  store i64 0, ptr %8, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 2
  store i64 0, ptr %9, align 8
  store ptr %6, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %10

10:                                               ; preds = %111, %0
  %11 = load i32, ptr %1, align 4
  %12 = icmp slt i32 %11, 64
  br i1 %12, label %13, label %115

13:                                               ; preds = %10
  %14 = call ptr @GC_malloc(i64 40)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %C, ptr %15, i32 0, i32 0
  %17 = getelementptr %Base, ptr %16, i32 0, i32 0
  store ptr @C..vtbl, ptr %17, align 8
  call void @C.constructor(ptr %15, double 9.990000e+02)
  %18 = getelementptr %C, ptr %15, i32 0, i32 0
  %19 = getelementptr %Base, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_16940945161124618027, i64 8), ptr %19, align 8
  %20 = getelementptr %C, ptr %15, i32 0, i32 1
  %21 = load ptr, ptr %20, align 8
  %22 = icmp eq ptr %21, null
  br i1 %22, label %23, label %26

23:                                               ; preds = %13
  %24 = call ptr @GC_malloc(i64 32)
  %25 = getelementptr i8, ptr %24, i64 8
  store ptr %25, ptr %20, align 8
  br label %27

26:                                               ; preds = %13
  br label %27

27:                                               ; preds = %23, %26
  %28 = phi ptr [ %21, %26 ], [ %25, %23 ]
  br label %29

29:                                               ; preds = %27
  %30 = getelementptr { ptr, i64, i64 }, ptr %28, i32 0, i32 1
  %31 = load i64, ptr %30, align 8
  %32 = add i64 %31, 1
  %33 = getelementptr { ptr, i64, i64 }, ptr %28, i32 0, i32 0
  %34 = getelementptr { ptr, i64, i64 }, ptr %28, i32 0, i32 2
  %35 = load ptr, ptr %33, align 8
  %36 = load i64, ptr %34, align 8
  %37 = icmp ugt i64 %32, %36
  br i1 %37, label %38, label %62

38:                                               ; preds = %29
  %39 = mul i64 %36, 2
  %40 = icmp uge i64 %39, %32
  %41 = select i1 %40, i64 %39, i64 %32
  %42 = icmp ult i64 %41, 4
  %43 = select i1 %42, i64 4, i64 %41
  %44 = icmp ule i64 %32, 4294967295
  br i1 %44, label %45, label %117

45:                                               ; preds = %38
  %46 = icmp ule i64 %43, 2305843009213693951
  br i1 %46, label %47, label %119

47:                                               ; preds = %45
  %48 = mul i64 8, %43
  %49 = add i64 %48, 8
  %50 = icmp eq ptr %35, null
  %51 = getelementptr i8, ptr %35, i64 -8
  %52 = select i1 %50, ptr null, ptr %51
  %53 = call ptr @GC_realloc(ptr %52, i64 %49)
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = icmp ne ptr %54, null
  %56 = icmp eq i64 %48, 0
  %57 = or i1 %55, %56
  br i1 %57, label %58, label %121

58:                                               ; preds = %47
  %59 = mul i64 8, %36
  %60 = getelementptr i8, ptr %54, i64 %59
  %61 = sub i64 %48, %59
  call void @llvm.memset.p0.i64(ptr %60, i8 0, i64 %61, i1 false)
  store ptr %54, ptr %33, align 8
  store i64 %43, ptr %34, align 8
  br label %63

62:                                               ; preds = %29
  br label %63

63:                                               ; preds = %58, %62
  %64 = phi ptr [ %35, %62 ], [ %54, %58 ]
  br label %65

65:                                               ; preds = %63
  %66 = getelementptr double, ptr %64, i64 %31
  store double 9.900000e+01, ptr %66, align 8
  store i64 %32, ptr %30, align 8
  %67 = load ptr, ptr %2, align 8
  %68 = icmp eq ptr %67, null
  br i1 %68, label %69, label %72

69:                                               ; preds = %65
  %70 = call ptr @GC_malloc(i64 32)
  %71 = getelementptr i8, ptr %70, i64 8
  store ptr %71, ptr %2, align 8
  br label %73

72:                                               ; preds = %65
  br label %73

73:                                               ; preds = %69, %72
  %74 = phi ptr [ %67, %72 ], [ %71, %69 ]
  br label %75

75:                                               ; preds = %73
  %76 = getelementptr { ptr, i64, i64 }, ptr %74, i32 0, i32 1
  %77 = load i64, ptr %76, align 8
  %78 = add i64 %77, 1
  %79 = getelementptr { ptr, i64, i64 }, ptr %74, i32 0, i32 0
  %80 = getelementptr { ptr, i64, i64 }, ptr %74, i32 0, i32 2
  %81 = load ptr, ptr %79, align 8
  %82 = load i64, ptr %80, align 8
  %83 = icmp ugt i64 %78, %82
  br i1 %83, label %84, label %108

84:                                               ; preds = %75
  %85 = mul i64 %82, 2
  %86 = icmp uge i64 %85, %78
  %87 = select i1 %86, i64 %85, i64 %78
  %88 = icmp ult i64 %87, 4
  %89 = select i1 %88, i64 4, i64 %87
  %90 = icmp ule i64 %78, 4294967295
  br i1 %90, label %91, label %123

91:                                               ; preds = %84
  %92 = icmp ule i64 %89, 2305843009213693951
  br i1 %92, label %93, label %125

93:                                               ; preds = %91
  %94 = mul i64 8, %89
  %95 = add i64 %94, 8
  %96 = icmp eq ptr %81, null
  %97 = getelementptr i8, ptr %81, i64 -8
  %98 = select i1 %96, ptr null, ptr %97
  %99 = call ptr @GC_realloc(ptr %98, i64 %95)
  %100 = getelementptr i8, ptr %99, i64 8
  %101 = icmp ne ptr %100, null
  %102 = icmp eq i64 %94, 0
  %103 = or i1 %101, %102
  br i1 %103, label %104, label %127

104:                                              ; preds = %93
  %105 = mul i64 8, %82
  %106 = getelementptr i8, ptr %100, i64 %105
  %107 = sub i64 %94, %105
  call void @llvm.memset.p0.i64(ptr %106, i8 0, i64 %107, i1 false)
  store ptr %100, ptr %79, align 8
  store i64 %89, ptr %80, align 8
  br label %109

108:                                              ; preds = %75
  br label %109

109:                                              ; preds = %104, %108
  %110 = phi ptr [ %81, %108 ], [ %100, %104 ]
  br label %111

111:                                              ; preds = %109
  %112 = getelementptr ptr, ptr %110, i64 %77
  store ptr %15, ptr %112, align 8
  store i64 %78, ptr %76, align 8
  %113 = load i32, ptr %1, align 4
  %114 = add i32 %113, 1
  store i32 %114, ptr %1, align 4
  br label %10

115:                                              ; preds = %10
  ret void

116:                                              ; preds = %127, %125, %123, %121, %119, %117
  unreachable

117:                                              ; preds = %38
  %118 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 43, ptr null)
  br label %116

119:                                              ; preds = %45
  %120 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 43, ptr null)
  br label %116

121:                                              ; preds = %47
  %122 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 43, ptr null)
  br label %116

123:                                              ; preds = %84
  %124 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 44, ptr null)
  br label %116

125:                                              ; preds = %91
  %126 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 44, ptr null)
  br label %116

127:                                              ; preds = %93
  %128 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 44, ptr null)
  br label %116
}

; Function Attrs: mustprogress
define double @thrower() #2 {
  %1 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2770870231077248403, i64 8), ptr %1, align 8
  call void @__cxa_throw(ptr %1, ptr @_ZTIPc, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define i1 @viaInterface({ ptr, ptr } %0, double %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca { ptr, ptr }, align 8
  %6 = alloca i1, align 1
  store { ptr, ptr } %0, ptr %5, align 8
  store double %2, ptr %4, align 8
  call void @churn()
  %7 = load { ptr, ptr }, ptr %5, align 8
  %8 = extractvalue { ptr, ptr } %7, 0
  %9 = extractvalue { ptr, ptr } %7, 1
  %10 = getelementptr ptr, ptr %8, i32 2
  %11 = load ptr, ptr %10, align 8
  %12 = ptrtoint ptr %9 to i64
  %13 = ptrtoint ptr %11 to i64
  %14 = add i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = extractvalue { ptr, ptr } %7, 0
  %17 = extractvalue { ptr, ptr } %7, 1
  %18 = getelementptr ptr, ptr %16, i32 2
  %19 = load ptr, ptr %18, align 8
  %20 = ptrtoint ptr %19 to i64
  %21 = and i64 %20, -9223372036854775808
  %22 = icmp ne i64 %21, 0
  br i1 %22, label %23, label %29

23:                                               ; preds = %3
  %24 = sub i64 %20, -9223372036854775808
  %25 = inttoptr i64 %24 to ptr
  %26 = getelementptr { ptr, ptr }, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %26, align 8
  %28 = call double %27(ptr %17)
  br label %31

29:                                               ; preds = %3
  %30 = load double, ptr %15, align 8
  br label %31

31:                                               ; preds = %23, %29
  %32 = phi double [ %30, %29 ], [ %28, %23 ]
  br label %33

33:                                               ; preds = %31
  %34 = load double, ptr %4, align 8
  %35 = fcmp oeq double %32, %34
  br i1 %35, label %36, label %101

36:                                               ; preds = %33
  %37 = load { ptr, ptr }, ptr %5, align 8
  %38 = extractvalue { ptr, ptr } %37, 0
  %39 = extractvalue { ptr, ptr } %37, 1
  %40 = getelementptr ptr, ptr %38, i32 3
  %41 = load ptr, ptr %40, align 8
  %42 = ptrtoint ptr %39 to i64
  %43 = ptrtoint ptr %41 to i64
  %44 = add i64 %42, %43
  %45 = inttoptr i64 %44 to ptr
  %46 = extractvalue { ptr, ptr } %37, 0
  %47 = extractvalue { ptr, ptr } %37, 1
  %48 = getelementptr ptr, ptr %46, i32 3
  %49 = load ptr, ptr %48, align 8
  %50 = ptrtoint ptr %49 to i64
  %51 = and i64 %50, -9223372036854775808
  %52 = icmp ne i64 %51, 0
  br i1 %52, label %53, label %59

53:                                               ; preds = %36
  %54 = sub i64 %50, -9223372036854775808
  %55 = inttoptr i64 %54 to ptr
  %56 = getelementptr { ptr, ptr }, ptr %55, i32 0, i32 0
  %57 = load ptr, ptr %56, align 8
  %58 = call ptr %57(ptr %47)
  br label %61

59:                                               ; preds = %36
  %60 = load ptr, ptr %45, align 8
  br label %61

61:                                               ; preds = %53, %59
  %62 = phi ptr [ %60, %59 ], [ %58, %53 ]
  br label %63

63:                                               ; preds = %61
  %64 = load double, ptr %4, align 8
  %65 = call ptr @GC_malloc_atomic(i64 40)
  %66 = getelementptr i8, ptr %65, i64 8
  call void @__ts_number_to_string(ptr %66, double %64)
  %67 = icmp eq ptr %66, null
  %68 = select i1 %67, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %66
  %69 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %70 = add i64 1, %69
  %71 = call i64 @strlen(ptr %68)
  %72 = add i64 %70, %71
  %73 = add i64 %72, 8
  %74 = call ptr @GC_malloc(i64 %73)
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = getelementptr i8, ptr %75, i64 0
  %77 = sub i64 %72, 0
  %78 = call ptr @__memcpy_chk(ptr %76, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %69, i64 %77)
  %79 = add i64 0, %69
  %80 = getelementptr i8, ptr %75, i64 %79
  %81 = sub i64 %72, %79
  %82 = call ptr @__memcpy_chk(ptr %80, ptr %68, i64 %71, i64 %81)
  %83 = add i64 %79, %71
  %84 = getelementptr i8, ptr %75, i64 %83
  store i8 0, ptr %84, align 1
  %85 = ptrtoint ptr %62 to i64
  %86 = ptrtoint ptr %75 to i64
  %87 = icmp ne i64 %85, 0
  %88 = icmp ne i64 %86, 0
  %89 = zext i1 %87 to i32
  %90 = zext i1 %88 to i32
  %91 = and i32 %89, %90
  %92 = icmp ne i32 %91, 0
  br i1 %92, label %93, label %96

93:                                               ; preds = %63
  %94 = call i32 @strcmp(ptr %62, ptr %75)
  %95 = icmp eq i32 %94, 0
  br label %98

96:                                               ; preds = %63
  %97 = icmp eq i64 %85, %86
  br label %98

98:                                               ; preds = %93, %96
  %99 = phi i1 [ %97, %96 ], [ %95, %93 ]
  br label %100

100:                                              ; preds = %98
  br label %101

101:                                              ; preds = %100, %33
  %102 = phi i1 [ %99, %100 ], [ %35, %33 ]
  store i1 %102, ptr %6, align 1
  %103 = load i1, ptr %6, align 1
  ret i1 %103
}

; Function Attrs: mustprogress
define i1 @viaBase(ptr %0, double %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i1, align 1
  store ptr %0, ptr %5, align 8
  store double %2, ptr %4, align 8
  call void @churn()
  %7 = load ptr, ptr %5, align 8
  %8 = getelementptr %Base, ptr %7, i32 0, i32 2
  %9 = load double, ptr %8, align 8
  %10 = load double, ptr %4, align 8
  %11 = fcmp oeq double %9, %10
  br i1 %11, label %12, label %53

12:                                               ; preds = %3
  %13 = load ptr, ptr %5, align 8
  %14 = getelementptr %Base, ptr %13, i32 0, i32 1
  %15 = load ptr, ptr %14, align 8
  %16 = load double, ptr %4, align 8
  %17 = call ptr @GC_malloc_atomic(i64 40)
  %18 = getelementptr i8, ptr %17, i64 8
  call void @__ts_number_to_string(ptr %18, double %16)
  %19 = icmp eq ptr %18, null
  %20 = select i1 %19, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %18
  %21 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %22 = add i64 1, %21
  %23 = call i64 @strlen(ptr %20)
  %24 = add i64 %22, %23
  %25 = add i64 %24, 8
  %26 = call ptr @GC_malloc(i64 %25)
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr i8, ptr %27, i64 0
  %29 = sub i64 %24, 0
  %30 = call ptr @__memcpy_chk(ptr %28, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %21, i64 %29)
  %31 = add i64 0, %21
  %32 = getelementptr i8, ptr %27, i64 %31
  %33 = sub i64 %24, %31
  %34 = call ptr @__memcpy_chk(ptr %32, ptr %20, i64 %23, i64 %33)
  %35 = add i64 %31, %23
  %36 = getelementptr i8, ptr %27, i64 %35
  store i8 0, ptr %36, align 1
  %37 = ptrtoint ptr %15 to i64
  %38 = ptrtoint ptr %27 to i64
  %39 = icmp ne i64 %37, 0
  %40 = icmp ne i64 %38, 0
  %41 = zext i1 %39 to i32
  %42 = zext i1 %40 to i32
  %43 = and i32 %41, %42
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %45, label %48

45:                                               ; preds = %12
  %46 = call i32 @strcmp(ptr %15, ptr %27)
  %47 = icmp eq i32 %46, 0
  br label %50

48:                                               ; preds = %12
  %49 = icmp eq i64 %37, %38
  br label %50

50:                                               ; preds = %45, %48
  %51 = phi i1 [ %49, %48 ], [ %47, %45 ]
  br label %52

52:                                               ; preds = %50
  br label %53

53:                                               ; preds = %52, %3
  %54 = phi i1 [ %51, %52 ], [ %11, %3 ]
  store i1 %54, ptr %6, align 1
  %55 = load i1, ptr %6, align 1
  ret i1 %55
}

; Function Attrs: mustprogress
define i1 @viaHolder({ ptr, ptr } %0, double %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca { ptr, ptr }, align 8
  %6 = alloca i1, align 1
  store { ptr, ptr } %0, ptr %5, align 8
  store double %2, ptr %4, align 8
  call void @churn()
  %7 = load { ptr, ptr }, ptr %5, align 8
  %8 = extractvalue { ptr, ptr } %7, 0
  %9 = extractvalue { ptr, ptr } %7, 1
  %10 = getelementptr ptr, ptr %8, i32 2
  %11 = load ptr, ptr %10, align 8
  %12 = ptrtoint ptr %9 to i64
  %13 = ptrtoint ptr %11 to i64
  %14 = add i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = extractvalue { ptr, ptr } %7, 0
  %17 = extractvalue { ptr, ptr } %7, 1
  %18 = getelementptr ptr, ptr %16, i32 2
  %19 = load ptr, ptr %18, align 8
  %20 = ptrtoint ptr %19 to i64
  %21 = and i64 %20, -9223372036854775808
  %22 = icmp ne i64 %21, 0
  br i1 %22, label %23, label %29

23:                                               ; preds = %3
  %24 = sub i64 %20, -9223372036854775808
  %25 = inttoptr i64 %24 to ptr
  %26 = getelementptr { ptr, ptr }, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %26, align 8
  %28 = call ptr %27(ptr %17)
  br label %31

29:                                               ; preds = %3
  %30 = load ptr, ptr %15, align 8
  br label %31

31:                                               ; preds = %23, %29
  %32 = phi ptr [ %30, %29 ], [ %28, %23 ]
  br label %33

33:                                               ; preds = %31
  %34 = getelementptr %C, ptr %32, i32 0, i32 0
  %35 = getelementptr %Base, ptr %34, i32 0, i32 2
  %36 = load double, ptr %35, align 8
  %37 = load double, ptr %4, align 8
  %38 = fcmp oeq double %36, %37
  br i1 %38, label %39, label %107

39:                                               ; preds = %33
  %40 = load { ptr, ptr }, ptr %5, align 8
  %41 = extractvalue { ptr, ptr } %40, 0
  %42 = extractvalue { ptr, ptr } %40, 1
  %43 = getelementptr ptr, ptr %41, i32 2
  %44 = load ptr, ptr %43, align 8
  %45 = ptrtoint ptr %42 to i64
  %46 = ptrtoint ptr %44 to i64
  %47 = add i64 %45, %46
  %48 = inttoptr i64 %47 to ptr
  %49 = extractvalue { ptr, ptr } %40, 0
  %50 = extractvalue { ptr, ptr } %40, 1
  %51 = getelementptr ptr, ptr %49, i32 2
  %52 = load ptr, ptr %51, align 8
  %53 = ptrtoint ptr %52 to i64
  %54 = and i64 %53, -9223372036854775808
  %55 = icmp ne i64 %54, 0
  br i1 %55, label %56, label %62

56:                                               ; preds = %39
  %57 = sub i64 %53, -9223372036854775808
  %58 = inttoptr i64 %57 to ptr
  %59 = getelementptr { ptr, ptr }, ptr %58, i32 0, i32 0
  %60 = load ptr, ptr %59, align 8
  %61 = call ptr %60(ptr %50)
  br label %64

62:                                               ; preds = %39
  %63 = load ptr, ptr %48, align 8
  br label %64

64:                                               ; preds = %56, %62
  %65 = phi ptr [ %63, %62 ], [ %61, %56 ]
  br label %66

66:                                               ; preds = %64
  %67 = getelementptr %C, ptr %65, i32 0, i32 0
  %68 = getelementptr %Base, ptr %67, i32 0, i32 1
  %69 = load ptr, ptr %68, align 8
  %70 = load double, ptr %4, align 8
  %71 = call ptr @GC_malloc_atomic(i64 40)
  %72 = getelementptr i8, ptr %71, i64 8
  call void @__ts_number_to_string(ptr %72, double %70)
  %73 = icmp eq ptr %72, null
  %74 = select i1 %73, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %72
  %75 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %76 = add i64 1, %75
  %77 = call i64 @strlen(ptr %74)
  %78 = add i64 %76, %77
  %79 = add i64 %78, 8
  %80 = call ptr @GC_malloc(i64 %79)
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr i8, ptr %81, i64 0
  %83 = sub i64 %78, 0
  %84 = call ptr @__memcpy_chk(ptr %82, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %75, i64 %83)
  %85 = add i64 0, %75
  %86 = getelementptr i8, ptr %81, i64 %85
  %87 = sub i64 %78, %85
  %88 = call ptr @__memcpy_chk(ptr %86, ptr %74, i64 %77, i64 %87)
  %89 = add i64 %85, %77
  %90 = getelementptr i8, ptr %81, i64 %89
  store i8 0, ptr %90, align 1
  %91 = ptrtoint ptr %69 to i64
  %92 = ptrtoint ptr %81 to i64
  %93 = icmp ne i64 %91, 0
  %94 = icmp ne i64 %92, 0
  %95 = zext i1 %93 to i32
  %96 = zext i1 %94 to i32
  %97 = and i32 %95, %96
  %98 = icmp ne i32 %97, 0
  br i1 %98, label %99, label %102

99:                                               ; preds = %66
  %100 = call i32 @strcmp(ptr %69, ptr %81)
  %101 = icmp eq i32 %100, 0
  br label %104

102:                                              ; preds = %66
  %103 = icmp eq i64 %91, %92
  br label %104

104:                                              ; preds = %99, %102
  %105 = phi i1 [ %103, %102 ], [ %101, %99 ]
  br label %106

106:                                              ; preds = %104
  br label %107

107:                                              ; preds = %106, %33
  %108 = phi i1 [ %105, %106 ], [ %38, %33 ]
  br i1 %108, label %109, label %147

109:                                              ; preds = %107
  %110 = load { ptr, ptr }, ptr %5, align 8
  %111 = extractvalue { ptr, ptr } %110, 0
  %112 = extractvalue { ptr, ptr } %110, 1
  %113 = getelementptr ptr, ptr %111, i32 2
  %114 = load ptr, ptr %113, align 8
  %115 = ptrtoint ptr %112 to i64
  %116 = ptrtoint ptr %114 to i64
  %117 = add i64 %115, %116
  %118 = inttoptr i64 %117 to ptr
  %119 = extractvalue { ptr, ptr } %110, 0
  %120 = extractvalue { ptr, ptr } %110, 1
  %121 = getelementptr ptr, ptr %119, i32 2
  %122 = load ptr, ptr %121, align 8
  %123 = ptrtoint ptr %122 to i64
  %124 = and i64 %123, -9223372036854775808
  %125 = icmp ne i64 %124, 0
  br i1 %125, label %126, label %132

126:                                              ; preds = %109
  %127 = sub i64 %123, -9223372036854775808
  %128 = inttoptr i64 %127 to ptr
  %129 = getelementptr { ptr, ptr }, ptr %128, i32 0, i32 0
  %130 = load ptr, ptr %129, align 8
  %131 = call ptr %130(ptr %120)
  br label %134

132:                                              ; preds = %109
  %133 = load ptr, ptr %118, align 8
  br label %134

134:                                              ; preds = %126, %132
  %135 = phi ptr [ %133, %132 ], [ %131, %126 ]
  br label %136

136:                                              ; preds = %134
  %137 = getelementptr %C, ptr %135, i32 0, i32 1
  %138 = load ptr, ptr %137, align 8
  %139 = icmp eq ptr %138, null
  %140 = select i1 %139, ptr @__ts_empty_array, ptr %138
  %141 = getelementptr { ptr, i64, i64 }, ptr %140, i32 0, i32 0
  %142 = load ptr, ptr %141, align 8
  %143 = getelementptr double, ptr %142, i32 0
  %144 = load double, ptr %143, align 8
  %145 = load double, ptr %4, align 8
  %146 = fcmp oeq double %144, %145
  br label %147

147:                                              ; preds = %136, %107
  %148 = phi i1 [ %146, %136 ], [ %108, %107 ]
  store i1 %148, ptr %6, align 1
  %149 = load i1, ptr %6, align 1
  ret i1 %149
}

; Function Attrs: mustprogress
define i1 @viaRecord({ ptr } %0, double %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca { ptr }, align 8
  %6 = alloca i1, align 1
  store { ptr } %0, ptr %5, align 8
  store double %2, ptr %4, align 8
  call void @churn()
  %7 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr %C, ptr %8, i32 0, i32 0
  %10 = getelementptr %Base, ptr %9, i32 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = load double, ptr %4, align 8
  %13 = fcmp oeq double %11, %12
  br i1 %13, label %14, label %57

14:                                               ; preds = %3
  %15 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  %16 = load ptr, ptr %15, align 8
  %17 = getelementptr %C, ptr %16, i32 0, i32 0
  %18 = getelementptr %Base, ptr %17, i32 0, i32 1
  %19 = load ptr, ptr %18, align 8
  %20 = load double, ptr %4, align 8
  %21 = call ptr @GC_malloc_atomic(i64 40)
  %22 = getelementptr i8, ptr %21, i64 8
  call void @__ts_number_to_string(ptr %22, double %20)
  %23 = icmp eq ptr %22, null
  %24 = select i1 %23, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %22
  %25 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
  %26 = add i64 1, %25
  %27 = call i64 @strlen(ptr %24)
  %28 = add i64 %26, %27
  %29 = add i64 %28, 8
  %30 = call ptr @GC_malloc(i64 %29)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr i8, ptr %31, i64 0
  %33 = sub i64 %28, 0
  %34 = call ptr @__memcpy_chk(ptr %32, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8), i64 %25, i64 %33)
  %35 = add i64 0, %25
  %36 = getelementptr i8, ptr %31, i64 %35
  %37 = sub i64 %28, %35
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %24, i64 %27, i64 %37)
  %39 = add i64 %35, %27
  %40 = getelementptr i8, ptr %31, i64 %39
  store i8 0, ptr %40, align 1
  %41 = ptrtoint ptr %19 to i64
  %42 = ptrtoint ptr %31 to i64
  %43 = icmp ne i64 %41, 0
  %44 = icmp ne i64 %42, 0
  %45 = zext i1 %43 to i32
  %46 = zext i1 %44 to i32
  %47 = and i32 %45, %46
  %48 = icmp ne i32 %47, 0
  br i1 %48, label %49, label %52

49:                                               ; preds = %14
  %50 = call i32 @strcmp(ptr %19, ptr %31)
  %51 = icmp eq i32 %50, 0
  br label %54

52:                                               ; preds = %14
  %53 = icmp eq i64 %41, %42
  br label %54

54:                                               ; preds = %49, %52
  %55 = phi i1 [ %53, %52 ], [ %51, %49 ]
  br label %56

56:                                               ; preds = %54
  br label %57

57:                                               ; preds = %56, %3
  %58 = phi i1 [ %55, %56 ], [ %13, %3 ]
  br i1 %58, label %59, label %72

59:                                               ; preds = %57
  %60 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  %61 = load ptr, ptr %60, align 8
  %62 = getelementptr %C, ptr %61, i32 0, i32 1
  %63 = load ptr, ptr %62, align 8
  %64 = icmp eq ptr %63, null
  %65 = select i1 %64, ptr @__ts_empty_array, ptr %63
  %66 = getelementptr { ptr, i64, i64 }, ptr %65, i32 0, i32 0
  %67 = load ptr, ptr %66, align 8
  %68 = getelementptr double, ptr %67, i32 0
  %69 = load double, ptr %68, align 8
  %70 = load double, ptr %4, align 8
  %71 = fcmp oeq double %69, %70
  br label %72

72:                                               ; preds = %59, %57
  %73 = phi i1 [ %71, %59 ], [ %58, %57 ]
  store i1 %73, ptr %6, align 1
  %74 = load i1, ptr %6, align 1
  ret i1 %74
}

; Function Attrs: mustprogress
define internal i1 @C.IC..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  %4 = getelementptr %C, ptr %0, i32 0, i32 0
  %5 = getelementptr %Base, ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = getelementptr ptr, ptr %6, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = call i1 %8(ptr %0, ptr %1)
  store i1 %9, ptr %3, align 1
  %10 = load i1, ptr %3, align 1
  ret i1 %10
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define void @inLoop() #2 {
  %1 = alloca { ptr }, align 8
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %41, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 3
  br i1 %5, label %6, label %44

6:                                                ; preds = %3
  %7 = load ptr, ptr @G, align 8
  %8 = getelementptr %C, ptr %7, i32 0, i32 0
  %9 = getelementptr %Base, ptr %8, i32 0, i32 2
  %10 = load double, ptr %9, align 8
  %11 = load ptr, ptr @G, align 8
  %12 = load i32, ptr %2, align 4
  %13 = add i32 %12, 10
  %14 = sitofp i32 %13 to double
  %15 = call ptr @mk(double %14)
  store ptr %15, ptr @G, align 8
  %16 = ptrtoint ptr %11 to i64
  %17 = icmp ne i64 %16, 0
  br i1 %17, label %18, label %19

18:                                               ; preds = %6
  br label %20

19:                                               ; preds = %6
  br label %20

20:                                               ; preds = %18, %19
  %21 = phi ptr [ @IC.null..vtbl, %19 ], [ @C.IC..vtbl, %18 ]
  %22 = insertvalue { ptr, ptr } undef, ptr %21, 0
  %23 = insertvalue { ptr, ptr } %22, ptr %11, 1
  %24 = call i1 @viaInterface({ ptr, ptr } %23, double 0.000000e+00, double %10)
  br i1 %24, label %25, label %46

25:                                               ; preds = %20
  %26 = load ptr, ptr @G, align 8
  %27 = getelementptr %C, ptr %26, i32 0, i32 0
  %28 = getelementptr %Base, ptr %27, i32 0, i32 2
  %29 = load double, ptr %28, align 8
  %30 = load ptr, ptr @G, align 8
  store { ptr } zeroinitializer, ptr %1, align 8
  %31 = getelementptr { ptr }, ptr %1, i32 0, i32 0
  store ptr %30, ptr %31, align 8
  %32 = load { ptr }, ptr %1, align 8
  %33 = load i32, ptr %2, align 4
  %34 = add i32 %33, 20
  %35 = sitofp i32 %34 to double
  %36 = call ptr @mk(double %35)
  store ptr %36, ptr @G, align 8
  %37 = call ptr @GC_malloc(i64 16)
  %38 = getelementptr i8, ptr %37, i64 8
  store { ptr } %32, ptr %38, align 8
  %39 = insertvalue { ptr, ptr } { ptr @IHolder.1938816268306746371..vtbl, ptr undef }, ptr %38, 1
  %40 = call i1 @viaHolder({ ptr, ptr } %39, double 0.000000e+00, double %29)
  br i1 %40, label %41, label %48

41:                                               ; preds = %25
  %42 = load i32, ptr %2, align 4
  %43 = add i32 %42, 1
  store i32 %43, ptr %2, align 4
  br label %3

44:                                               ; preds = %3
  ret void

45:                                               ; preds = %48, %46
  unreachable

46:                                               ; preds = %20
  %47 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9664403948889819029, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 78, ptr null)
  br label %45

48:                                               ; preds = %25
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6521797870148510405, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 80, ptr null)
  br label %45
}

; Function Attrs: mustprogress
define void @inTryCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = call ptr @mk(double 3.000000e+01)
  store ptr %4, ptr @G, align 8
  store i1 false, ptr %3, align 1
  %5 = load ptr, ptr @G, align 8
  %6 = invoke ptr @mk(double 3.100000e+01)
          to label %7 unwind label %20

7:                                                ; preds = %0
  store ptr %6, ptr @G, align 8
  %8 = invoke double @thrower()
          to label %9 unwind label %20

9:                                                ; preds = %7
  %10 = ptrtoint ptr %5 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %13

12:                                               ; preds = %9
  br label %14

13:                                               ; preds = %9
  br label %14

14:                                               ; preds = %12, %13
  %15 = phi ptr [ @IC.null..vtbl, %13 ], [ @C.IC..vtbl, %12 ]
  %16 = insertvalue { ptr, ptr } undef, ptr %15, 0
  %17 = insertvalue { ptr, ptr } %16, ptr %5, 1
  %18 = invoke i1 @viaInterface({ ptr, ptr } %17, double %8, double 3.000000e+01)
          to label %19 unwind label %20

19:                                               ; preds = %14
  br label %159

20:                                               ; preds = %14, %7, %0
  %21 = landingpad { ptr, i32 }
          catch ptr null
  %22 = extractvalue { ptr, i32 } %21, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %20
  %25 = icmp eq ptr %24, @_ZTIi
  br i1 %25, label %26, label %33

26:                                               ; preds = %.noexc6
  %27 = load i32, ptr %23, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %26
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 0
  store i64 4, ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 2
  store i32 %27, ptr %32, align 4
  br label %156

33:                                               ; preds = %.noexc6
  %34 = icmp eq ptr %24, @_ZTIj
  br i1 %34, label %35, label %42

35:                                               ; preds = %33
  %36 = load i32, ptr %23, align 4
  %37 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %35
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %38, i32 0, i32 0
  store i64 4, ptr %39, align 8
  %40 = getelementptr { i64, ptr, i32 }, ptr %38, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %40, align 8
  %41 = getelementptr { i64, ptr, i32 }, ptr %38, i32 0, i32 2
  store i32 %36, ptr %41, align 4
  br label %153

42:                                               ; preds = %33
  %43 = icmp eq ptr %24, @_ZTId
  br i1 %43, label %44, label %51

44:                                               ; preds = %42
  %45 = load double, ptr %23, align 8
  %46 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %44
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr { i64, ptr, double }, ptr %47, i32 0, i32 0
  store i64 8, ptr %48, align 8
  %49 = getelementptr { i64, ptr, double }, ptr %47, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %49, align 8
  %50 = getelementptr { i64, ptr, double }, ptr %47, i32 0, i32 2
  store double %45, ptr %50, align 8
  br label %150

51:                                               ; preds = %42
  %52 = icmp eq ptr %24, @_ZTIf
  br i1 %52, label %53, label %60

53:                                               ; preds = %51
  %54 = load float, ptr %23, align 4
  %55 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %53
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr { i64, ptr, float }, ptr %56, i32 0, i32 0
  store i64 4, ptr %57, align 8
  %58 = getelementptr { i64, ptr, float }, ptr %56, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %58, align 8
  %59 = getelementptr { i64, ptr, float }, ptr %56, i32 0, i32 2
  store float %54, ptr %59, align 4
  br label %147

60:                                               ; preds = %51
  %61 = icmp eq ptr %24, @_ZTIb
  br i1 %61, label %62, label %69

62:                                               ; preds = %60
  %63 = load i1, ptr %23, align 1
  %64 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %62
  %65 = getelementptr i8, ptr %64, i64 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %65, i32 0, i32 0
  store i64 1, ptr %66, align 8
  %67 = getelementptr { i64, ptr, i1 }, ptr %65, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %67, align 8
  %68 = getelementptr { i64, ptr, i1 }, ptr %65, i32 0, i32 2
  store i1 %63, ptr %68, align 1
  br label %144

69:                                               ; preds = %60
  %70 = icmp eq ptr %24, @_ZTIx
  br i1 %70, label %71, label %78

71:                                               ; preds = %69
  %72 = load i64, ptr %23, align 8
  %73 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %71
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %74, i32 0, i32 0
  store i64 8, ptr %75, align 8
  %76 = getelementptr { i64, ptr, i64 }, ptr %74, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %76, align 8
  %77 = getelementptr { i64, ptr, i64 }, ptr %74, i32 0, i32 2
  store i64 %72, ptr %77, align 8
  br label %141

78:                                               ; preds = %69
  %79 = icmp eq ptr %24, @_ZTIPc
  br i1 %79, label %80, label %87

80:                                               ; preds = %78
  %81 = load ptr, ptr %23, align 8
  %82 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %80
  %83 = getelementptr i8, ptr %82, i64 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %83, i32 0, i32 0
  store i64 8, ptr %84, align 8
  %85 = getelementptr { i64, ptr, ptr }, ptr %83, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %85, align 8
  %86 = getelementptr { i64, ptr, ptr }, ptr %83, i32 0, i32 2
  store ptr %81, ptr %86, align 8
  br label %138

87:                                               ; preds = %78
  %88 = icmp eq ptr %24, @_ZTIPv
  br i1 %88, label %89, label %91

89:                                               ; preds = %87
  %90 = load ptr, ptr %23, align 8
  br label %135

91:                                               ; preds = %87
  %92 = load ptr, ptr %24, align 8
  %93 = icmp eq ptr %92, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %93, label %94, label %125

94:                                               ; preds = %91
  %95 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %24, i32 0, i32 2
  %96 = load i32, ptr %95, align 4
  %97 = and i32 %96, 1073741824
  %98 = icmp ne i32 %97, 0
  br i1 %98, label %99, label %115

99:                                               ; preds = %94
  %100 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %24, i32 0, i32 4
  %101 = load ptr, ptr %100, align 8
  %102 = icmp ne ptr %101, null
  br i1 %102, label %103, label %105

103:                                              ; preds = %99
  invoke void %101(ptr %1, ptr %23)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %103
  %104 = load ptr, ptr %1, align 8
  br label %112

105:                                              ; preds = %99
  %106 = load ptr, ptr %23, align 8
  %107 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %105
  %108 = getelementptr i8, ptr %107, i64 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 0
  store i64 8, ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 2
  store ptr %106, ptr %111, align 8
  br label %112

112:                                              ; preds = %.noexc1, %.noexc5
  %113 = phi ptr [ %108, %.noexc5 ], [ %104, %.noexc1 ]
  br label %114

114:                                              ; preds = %112
  br label %122

115:                                              ; preds = %94
  %116 = load ptr, ptr %23, align 8
  %117 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %115
  %118 = getelementptr i8, ptr %117, i64 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 0
  store i64 8, ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 2
  store ptr %116, ptr %121, align 8
  br label %122

122:                                              ; preds = %114, %.noexc8
  %123 = phi ptr [ %118, %.noexc8 ], [ %113, %114 ]
  br label %124

124:                                              ; preds = %122
  br label %132

125:                                              ; preds = %91
  %126 = load ptr, ptr %23, align 8
  %127 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %125
  %128 = getelementptr i8, ptr %127, i64 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %128, i32 0, i32 0
  store i64 8, ptr %129, align 8
  %130 = getelementptr { i64, ptr, ptr }, ptr %128, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %130, align 8
  %131 = getelementptr { i64, ptr, ptr }, ptr %128, i32 0, i32 2
  store ptr %126, ptr %131, align 8
  br label %132

132:                                              ; preds = %124, %.noexc3
  %133 = phi ptr [ %128, %.noexc3 ], [ %123, %124 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %89, %134
  %136 = phi ptr [ %133, %134 ], [ %90, %89 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc9, %137
  %139 = phi ptr [ %136, %137 ], [ %83, %.noexc9 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc4, %140
  %142 = phi ptr [ %139, %140 ], [ %74, %.noexc4 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc7, %143
  %145 = phi ptr [ %142, %143 ], [ %65, %.noexc7 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc11, %146
  %148 = phi ptr [ %145, %146 ], [ %56, %.noexc11 ]
  br label %149

149:                                              ; preds = %147
  br label %150

150:                                              ; preds = %.noexc10, %149
  %151 = phi ptr [ %148, %149 ], [ %47, %.noexc10 ]
  br label %152

152:                                              ; preds = %150
  br label %153

153:                                              ; preds = %.noexc, %152
  %154 = phi ptr [ %151, %152 ], [ %38, %.noexc ]
  br label %155

155:                                              ; preds = %153
  br label %156

156:                                              ; preds = %.noexc2, %155
  %157 = phi ptr [ %154, %155 ], [ %29, %.noexc2 ]
  br label %158

158:                                              ; preds = %156
  store ptr %157, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %159

159:                                              ; preds = %19, %158
  call void @churn()
  %160 = load i1, ptr %3, align 1
  br i1 %160, label %161, label %167

161:                                              ; preds = %159
  %162 = load ptr, ptr @G, align 8
  %163 = getelementptr %C, ptr %162, i32 0, i32 0
  %164 = getelementptr %Base, ptr %163, i32 0, i32 2
  %165 = load double, ptr %164, align 8
  %166 = fcmp oeq double %165, 3.100000e+01
  br label %167

167:                                              ; preds = %161, %159
  %168 = phi i1 [ %166, %161 ], [ %160, %159 ]
  br i1 %168, label %169, label %187

169:                                              ; preds = %167
  %170 = load ptr, ptr @G, align 8
  %171 = getelementptr %C, ptr %170, i32 0, i32 0
  %172 = getelementptr %Base, ptr %171, i32 0, i32 1
  %173 = load ptr, ptr %172, align 8
  %174 = ptrtoint ptr %173 to i64
  %175 = icmp ne i64 %174, 0
  %176 = zext i1 %175 to i32
  %177 = and i32 %176, 1
  %178 = icmp ne i32 %177, 0
  br i1 %178, label %179, label %182

179:                                              ; preds = %169
  %180 = call i32 @strcmp(ptr %173, ptr getelementptr inbounds nuw (i8, ptr @s_17761974936599606694, i64 8))
  %181 = icmp eq i32 %180, 0
  br label %184

182:                                              ; preds = %169
  %183 = icmp eq i64 %174, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17761974936599606694, i64 8) to i64)
  br label %184

184:                                              ; preds = %179, %182
  %185 = phi i1 [ %183, %182 ], [ %181, %179 ]
  br label %186

186:                                              ; preds = %184
  br label %187

187:                                              ; preds = %186, %167
  %188 = phi i1 [ %185, %186 ], [ %168, %167 ]
  br i1 %188, label %189, label %191

189:                                              ; preds = %187
  ret void

190:                                              ; preds = %191
  unreachable

191:                                              ; preds = %187
  %192 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8973867690343378340, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 94, ptr null)
  br label %190

endcatch.resume:                                  ; preds = %53, %44, %80, %115, %62, %20, %105, %71, %125, %26, %103, %35
  %193 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %193
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca { ptr }, align 8
  %2 = alloca { ptr }, align 8
  %3 = alloca { ptr }, align 8
  %4 = load ptr, ptr @G, align 8
  %5 = call ptr @mk(double 2.000000e+00)
  store ptr %5, ptr @G, align 8
  %6 = ptrtoint ptr %4 to i64
  %7 = icmp ne i64 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %0
  br label %10

9:                                                ; preds = %0
  br label %10

10:                                               ; preds = %8, %9
  %11 = phi ptr [ @IC.null..vtbl, %9 ], [ @C.IC..vtbl, %8 ]
  %12 = insertvalue { ptr, ptr } undef, ptr %11, 0
  %13 = insertvalue { ptr, ptr } %12, ptr %4, 1
  %14 = call i1 @viaInterface({ ptr, ptr } %13, double 0.000000e+00, double 1.000000e+00)
  br i1 %14, label %15, label %81

15:                                               ; preds = %10
  %16 = load ptr, ptr @G, align 8
  %17 = call ptr @mk(double 3.000000e+00)
  store ptr %17, ptr @G, align 8
  %18 = call i1 @viaBase(ptr %16, double 0.000000e+00, double 2.000000e+00)
  br i1 %18, label %19, label %83

19:                                               ; preds = %15
  %20 = load ptr, ptr @GH, align 8
  %21 = getelementptr %H, ptr %20, i32 0, i32 1
  %22 = load ptr, ptr %21, align 8
  %23 = load ptr, ptr @GH, align 8
  %24 = getelementptr %H, ptr %23, i32 0, i32 1
  %25 = call ptr @mk(double 4.000000e+00)
  store ptr %25, ptr %24, align 8
  %26 = ptrtoint ptr %22 to i64
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %29

28:                                               ; preds = %19
  br label %30

29:                                               ; preds = %19
  br label %30

30:                                               ; preds = %28, %29
  %31 = phi ptr [ @IC.null..vtbl, %29 ], [ @C.IC..vtbl, %28 ]
  %32 = insertvalue { ptr, ptr } undef, ptr %31, 0
  %33 = insertvalue { ptr, ptr } %32, ptr %22, 1
  %34 = call i1 @viaInterface({ ptr, ptr } %33, double 0.000000e+00, double 3.000000e+00)
  br i1 %34, label %35, label %85

35:                                               ; preds = %30
  %36 = load ptr, ptr @G, align 8
  store { ptr } zeroinitializer, ptr %3, align 8
  %37 = getelementptr { ptr }, ptr %3, i32 0, i32 0
  store ptr %36, ptr %37, align 8
  %38 = load { ptr }, ptr %3, align 8
  %39 = call ptr @mk(double 5.000000e+00)
  store ptr %39, ptr @G, align 8
  %40 = call ptr @GC_malloc(i64 16)
  %41 = getelementptr i8, ptr %40, i64 8
  store { ptr } %38, ptr %41, align 8
  %42 = insertvalue { ptr, ptr } { ptr @IHolder.1938816268306746371..vtbl, ptr undef }, ptr %41, 1
  %43 = call i1 @viaHolder({ ptr, ptr } %42, double 0.000000e+00, double 3.000000e+00)
  br i1 %43, label %44, label %87

44:                                               ; preds = %35
  %45 = load ptr, ptr @GH, align 8
  %46 = getelementptr %H, ptr %45, i32 0, i32 1
  %47 = load ptr, ptr %46, align 8
  store { ptr } zeroinitializer, ptr %2, align 8
  %48 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  store ptr %47, ptr %48, align 8
  %49 = load { ptr }, ptr %2, align 8
  %50 = load ptr, ptr @GH, align 8
  %51 = getelementptr %H, ptr %50, i32 0, i32 1
  %52 = call ptr @mk(double 6.000000e+00)
  store ptr %52, ptr %51, align 8
  %53 = call ptr @GC_malloc(i64 16)
  %54 = getelementptr i8, ptr %53, i64 8
  store { ptr } %49, ptr %54, align 8
  %55 = insertvalue { ptr, ptr } { ptr @IHolder.1938816268306746371..vtbl, ptr undef }, ptr %54, 1
  %56 = call i1 @viaHolder({ ptr, ptr } %55, double 0.000000e+00, double 4.000000e+00)
  br i1 %56, label %57, label %89

57:                                               ; preds = %44
  %58 = load ptr, ptr @G, align 8
  store { ptr } zeroinitializer, ptr %1, align 8
  %59 = getelementptr { ptr }, ptr %1, i32 0, i32 0
  store ptr %58, ptr %59, align 8
  %60 = load { ptr }, ptr %1, align 8
  %61 = call ptr @mk(double 7.000000e+00)
  store ptr %61, ptr @G, align 8
  %62 = call i1 @viaRecord({ ptr } %60, double 0.000000e+00, double 5.000000e+00)
  br i1 %62, label %63, label %91

63:                                               ; preds = %57
  call void @inLoop()
  call void @inTryCatch()
  call void @churn()
  %64 = load ptr, ptr @G, align 8
  %65 = getelementptr %C, ptr %64, i32 0, i32 0
  %66 = getelementptr %Base, ptr %65, i32 0, i32 2
  %67 = load double, ptr %66, align 8
  %68 = fcmp oeq double %67, 3.100000e+01
  br i1 %68, label %69, label %77

69:                                               ; preds = %63
  %70 = load ptr, ptr @GH, align 8
  %71 = getelementptr %H, ptr %70, i32 0, i32 1
  %72 = load ptr, ptr %71, align 8
  %73 = getelementptr %C, ptr %72, i32 0, i32 0
  %74 = getelementptr %Base, ptr %73, i32 0, i32 2
  %75 = load double, ptr %74, align 8
  %76 = fcmp oeq double %75, 6.000000e+00
  br label %77

77:                                               ; preds = %69, %63
  %78 = phi i1 [ %76, %69 ], [ %68, %63 ]
  br i1 %78, label %79, label %93

79:                                               ; preds = %77
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

80:                                               ; preds = %93, %91, %89, %87, %85, %83, %81
  unreachable

81:                                               ; preds = %10
  %82 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7995604983552920533, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 98, ptr null)
  br label %80

83:                                               ; preds = %15
  %84 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7621023459670631976, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 99, ptr null)
  br label %80

85:                                               ; preds = %30
  %86 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14966932188815007743, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 100, ptr null)
  br label %80

87:                                               ; preds = %35
  %88 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1747183150290012101, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 101, ptr null)
  br label %80

89:                                               ; preds = %44
  %90 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5566661821194917391, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 102, ptr null)
  br label %80

91:                                               ; preds = %57
  %92 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7978875794277107671, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 103, ptr null)
  br label %80

93:                                               ; preds = %77
  %94 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12599839212798849691, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050726357520103011, i64 8), i32 107, ptr null)
  br label %80
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

