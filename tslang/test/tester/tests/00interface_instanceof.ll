; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%B = type { ptr, double }
%D = type { %B, double }
%Other = type { ptr }

$"___unbox<!ts.class<B, !ts.class_storage<B>>>" = comdat any

$B..size = comdat any

$D..size = comdat any

$Other..size = comdat any

@s_12638225683462629381 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFB\00", align 8
@s_12638219086392860115 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFD\00", align 8
@s_14547916283907374261 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOther\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_159832223624545629 = internal constant [112 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00interface_instanceof.ts\00", align 8
@s_3978046374080941754 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNaN\00", align 8
@s_865306855574198937 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInfinity\00", align 8
@s_5514277459305441004 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-Infinity\00", align 8
@s_555417232352566889 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s\00", align 8
@s_12638135523509116079 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF0\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
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
@B..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8)
@B..size = global i64 16, comdat
@B.I..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_781263649003409679_class, i64 40), ptr @B..instanceOf, ptr @B.get }
@td_781263649003409679_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_781263649003409679, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@B.J..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_781263649003409679_class, i64 40), ptr @B..instanceOf, ptr @B.twice }
@B..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @B..instanceOf, ptr @B.I..vtbl, ptr @B.J..vtbl, ptr @B..new, ptr @B.get, ptr @B.twice, ptr @B..rtti, ptr @B..size }
@D..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638219086392860115, i64 8)
@D..size = global i64 24, comdat
@D.I..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_8297331101459474202_class, i64 40), ptr @D..instanceOf, ptr @B.get }
@td_8297331101459474202_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_8297331101459474202, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@D.J..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_8297331101459474202_class, i64 40), ptr @D..instanceOf, ptr @B.twice }
@D..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @D..instanceOf, ptr @D.I..vtbl, ptr @D.J..vtbl, ptr @D..new, ptr @B.get, ptr @B.twice, ptr @B..rtti, ptr @B..size, ptr @D..rtti, ptr @D..size }
@Other..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_14547916283907374261, i64 8)
@Other..size = global i64 8, comdat
@Other.I..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_2680710835251623063_class, i64 40), ptr @Other..instanceOf, ptr @Other.get }
@td_2680710835251623063_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_2680710835251623063, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Other..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Other..instanceOf, ptr @Other.I..vtbl, ptr @Other..new, ptr @Other.get, ptr @Other..rtti, ptr @Other..size }
@_ZTIPc = external constant ptr
@I.null..vtbl = private global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @I.null.get }
@J.null..vtbl = private global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @J.null.twice }
@I.12106813767842140097..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12106813767842140097_object, i64 40), ptr @.instanceOf.none, ptr @.f_main..f_main..objL41C20FH568749565.get }
@td_12106813767842140097_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrel_12106813767842140097, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@td_16873085711952587189_interface = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [10 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 9, i32 -790239321, ptr @tsrel_16873085711952587189, ptr @tsret_16873085711952587189, ptr null, i64 -1 }, [10 x i8] c"interface\00" }, align 8
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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

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

declare void @ts.internal.eh_try(i32)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

define internal void @tsret_16873085711952587189(ptr %0) {
  %2 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %18

6:                                                ; preds = %1
  %7 = getelementptr ptr, ptr %3, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = getelementptr i8, ptr %8, i64 -40
  %12 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %11, i32 0, i32 3
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void %13(ptr %4)
  br label %16

16:                                               ; preds = %15, %10
  br label %17

17:                                               ; preds = %16, %6
  br label %18

18:                                               ; preds = %17, %1
  ret void
}

define internal void @tsrel_16873085711952587189(ptr %0) {
  %2 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %18

6:                                                ; preds = %1
  %7 = getelementptr ptr, ptr %3, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = getelementptr i8, ptr %8, i64 -40
  %12 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %11, i32 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void %13(ptr %4)
  br label %16

16:                                               ; preds = %15, %10
  br label %17

17:                                               ; preds = %16, %6
  br label %18

18:                                               ; preds = %17, %1
  ret void
}

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_12106813767842140097(ptr %0) {
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

define internal void @tsrel_2680710835251623063(ptr %0) {
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

define internal void @tsrel_8297331101459474202(ptr %0) {
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

define internal void @tsrel_781263649003409679(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %12

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %12

6:                                                ; preds = %4
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, @D..vtbl
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

; Function Attrs: mustprogress
define void @B.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %B, ptr %3, i32 0, i32 1
  store double 5.000000e+00, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @B..instanceOf(ptr %0, ptr %1) #1 {
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
define ptr @B..new() #1 {
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
define double @B.get(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %B, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define double @B.twice(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %B, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, 2.000000e+00
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress
define void @D.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %D, ptr %3, i32 0, i32 0
  call void @B.constructor(ptr %4)
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %D, ptr %5, i32 0, i32 1
  store double 1.000000e+00, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @D..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @D..rtti, align 8
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
  %26 = getelementptr %D, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %3, align 8
  %28 = call i1 @B..instanceOf(ptr %26, ptr %27)
  br label %29

29:                                               ; preds = %24, %23
  %30 = phi i1 [ %28, %24 ], [ %22, %23 ]
  store i1 %30, ptr %5, align 1
  %31 = load i1, ptr %5, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define ptr @D..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %D, ptr %3, i32 0, i32 0
  %5 = getelementptr %B, ptr %4, i32 0, i32 0
  store ptr @D..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define i1 @Other..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Other..rtti, align 8
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
define ptr @Other..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Other, ptr %3, i32 0, i32 0
  store ptr @Other..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define double @Other.get(ptr %0) #1 {
  %2 = alloca double, align 8
  store double 9.000000e+00, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  ret double %3
}

; Function Attrs: mustprogress
define double @I.null.get(ptr %0) #2 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define double @J.null.twice(ptr %0) #2 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define double @.f_main..f_main..objL41C20FH568749565.get(ptr %0) #1 {
  %2 = alloca double, align 8
  store double 3.000000e+00, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  ret double %3
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"___unbox<!ts.class<B, !ts.class_storage<B>>>"(ptr %0) #1 comdat {
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
  %39 = load ptr, ptr @B..rtti, align 8
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
  %70 = load ptr, ptr @B..rtti, align 8
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
  %83 = load ptr, ptr @B..rtti, align 8
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
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca { ptr }, align 8
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %B, ptr %6, i32 0, i32 0
  store ptr @B..vtbl, ptr %7, align 8
  call void @B.constructor(ptr %6)
  %8 = ptrtoint ptr %6 to i64
  %9 = icmp ne i64 %8, 0
  br i1 %9, label %10, label %15

10:                                               ; preds = %0
  %11 = getelementptr %B, ptr %6, i32 0, i32 0
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr ptr, ptr %12, i32 1
  %14 = load ptr, ptr %13, align 8
  br label %16

15:                                               ; preds = %0
  br label %16

16:                                               ; preds = %10, %15
  %17 = phi ptr [ @I.null..vtbl, %15 ], [ %14, %10 ]
  %18 = insertvalue { ptr, ptr } undef, ptr %17, 0
  %19 = insertvalue { ptr, ptr } %18, ptr %6, 1
  %20 = call ptr @GC_malloc(i64 24)
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr %B, ptr %21, i32 0, i32 0
  store ptr @B..vtbl, ptr %22, align 8
  call void @B.constructor(ptr %21)
  %23 = ptrtoint ptr %21 to i64
  %24 = icmp ne i64 %23, 0
  br i1 %24, label %25, label %30

25:                                               ; preds = %16
  %26 = getelementptr %B, ptr %21, i32 0, i32 0
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr ptr, ptr %27, i32 2
  %29 = load ptr, ptr %28, align 8
  br label %31

30:                                               ; preds = %16
  br label %31

31:                                               ; preds = %25, %30
  %32 = phi ptr [ @J.null..vtbl, %30 ], [ %29, %25 ]
  %33 = insertvalue { ptr, ptr } undef, ptr %32, 0
  %34 = insertvalue { ptr, ptr } %33, ptr %21, 1
  %35 = call ptr @GC_malloc(i64 32)
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr %D, ptr %36, i32 0, i32 0
  %38 = getelementptr %B, ptr %37, i32 0, i32 0
  store ptr @D..vtbl, ptr %38, align 8
  call void @D.constructor(ptr %36)
  %39 = ptrtoint ptr %36 to i64
  %40 = icmp ne i64 %39, 0
  br i1 %40, label %41, label %42

41:                                               ; preds = %31
  br label %43

42:                                               ; preds = %31
  br label %43

43:                                               ; preds = %41, %42
  %44 = phi ptr [ @I.null..vtbl, %42 ], [ @D.I..vtbl, %41 ]
  %45 = insertvalue { ptr, ptr } undef, ptr %44, 0
  %46 = insertvalue { ptr, ptr } %45, ptr %36, 1
  store { ptr } { ptr @.f_main..f_main..objL41C20FH568749565.get }, ptr %4, align 8
  %47 = load { ptr }, ptr %4, align 8
  %48 = call ptr @GC_malloc(i64 16)
  %49 = getelementptr i8, ptr %48, i64 8
  store { ptr } %47, ptr %49, align 8
  %50 = insertvalue { ptr, ptr } { ptr @I.12106813767842140097..vtbl, ptr undef }, ptr %49, 1
  %51 = extractvalue { ptr, ptr } %19, 1
  %52 = ptrtoint ptr %51 to i64
  %53 = icmp ne i64 %52, 0
  br i1 %53, label %54, label %61

54:                                               ; preds = %43
  %55 = extractvalue { ptr, ptr } %19, 0
  %56 = extractvalue { ptr, ptr } %19, 1
  %57 = getelementptr ptr, ptr %55, i32 1
  %58 = load ptr, ptr %57, align 8
  %59 = load ptr, ptr @B..rtti, align 8
  %60 = call i1 %58(ptr %56, ptr %59)
  br label %61

61:                                               ; preds = %54, %43
  %62 = phi i1 [ %60, %54 ], [ false, %43 ]
  br i1 %62, label %63, label %454

63:                                               ; preds = %61
  %64 = extractvalue { ptr, ptr } %19, 1
  %65 = ptrtoint ptr %64 to i64
  %66 = icmp ne i64 %65, 0
  br i1 %66, label %67, label %74

67:                                               ; preds = %63
  %68 = extractvalue { ptr, ptr } %19, 0
  %69 = extractvalue { ptr, ptr } %19, 1
  %70 = getelementptr ptr, ptr %68, i32 1
  %71 = load ptr, ptr %70, align 8
  %72 = load ptr, ptr @Other..rtti, align 8
  %73 = call i1 %71(ptr %69, ptr %72)
  br label %74

74:                                               ; preds = %67, %63
  %75 = phi i1 [ %73, %67 ], [ false, %63 ]
  %76 = xor i1 %75, true
  br i1 %76, label %77, label %456

77:                                               ; preds = %74
  %78 = extractvalue { ptr, ptr } %19, 1
  %79 = ptrtoint ptr %78 to i64
  %80 = icmp ne i64 %79, 0
  br i1 %80, label %81, label %88

81:                                               ; preds = %77
  %82 = extractvalue { ptr, ptr } %19, 0
  %83 = extractvalue { ptr, ptr } %19, 1
  %84 = getelementptr ptr, ptr %82, i32 1
  %85 = load ptr, ptr %84, align 8
  %86 = load ptr, ptr @D..rtti, align 8
  %87 = call i1 %85(ptr %83, ptr %86)
  br label %88

88:                                               ; preds = %81, %77
  %89 = phi i1 [ %87, %81 ], [ false, %77 ]
  %90 = xor i1 %89, true
  br i1 %90, label %91, label %458

91:                                               ; preds = %88
  %92 = extractvalue { ptr, ptr } %34, 1
  %93 = ptrtoint ptr %92 to i64
  %94 = icmp ne i64 %93, 0
  br i1 %94, label %95, label %102

95:                                               ; preds = %91
  %96 = extractvalue { ptr, ptr } %34, 0
  %97 = extractvalue { ptr, ptr } %34, 1
  %98 = getelementptr ptr, ptr %96, i32 1
  %99 = load ptr, ptr %98, align 8
  %100 = load ptr, ptr @B..rtti, align 8
  %101 = call i1 %99(ptr %97, ptr %100)
  br label %102

102:                                              ; preds = %95, %91
  %103 = phi i1 [ %101, %95 ], [ false, %91 ]
  br i1 %103, label %104, label %460

104:                                              ; preds = %102
  %105 = extractvalue { ptr, ptr } %46, 1
  %106 = ptrtoint ptr %105 to i64
  %107 = icmp ne i64 %106, 0
  br i1 %107, label %108, label %115

108:                                              ; preds = %104
  %109 = extractvalue { ptr, ptr } %46, 0
  %110 = extractvalue { ptr, ptr } %46, 1
  %111 = getelementptr ptr, ptr %109, i32 1
  %112 = load ptr, ptr %111, align 8
  %113 = load ptr, ptr @B..rtti, align 8
  %114 = call i1 %112(ptr %110, ptr %113)
  br label %115

115:                                              ; preds = %108, %104
  %116 = phi i1 [ %114, %108 ], [ false, %104 ]
  br i1 %116, label %117, label %462

117:                                              ; preds = %115
  %118 = extractvalue { ptr, ptr } %46, 1
  %119 = ptrtoint ptr %118 to i64
  %120 = icmp ne i64 %119, 0
  br i1 %120, label %121, label %128

121:                                              ; preds = %117
  %122 = extractvalue { ptr, ptr } %46, 0
  %123 = extractvalue { ptr, ptr } %46, 1
  %124 = getelementptr ptr, ptr %122, i32 1
  %125 = load ptr, ptr %124, align 8
  %126 = load ptr, ptr @D..rtti, align 8
  %127 = call i1 %125(ptr %123, ptr %126)
  br label %128

128:                                              ; preds = %121, %117
  %129 = phi i1 [ %127, %121 ], [ false, %117 ]
  br i1 %129, label %130, label %464

130:                                              ; preds = %128
  %131 = extractvalue { ptr, ptr } %50, 1
  %132 = ptrtoint ptr %131 to i64
  %133 = icmp ne i64 %132, 0
  br i1 %133, label %134, label %141

134:                                              ; preds = %130
  %135 = extractvalue { ptr, ptr } %50, 0
  %136 = extractvalue { ptr, ptr } %50, 1
  %137 = getelementptr ptr, ptr %135, i32 1
  %138 = load ptr, ptr %137, align 8
  %139 = load ptr, ptr @B..rtti, align 8
  %140 = call i1 %138(ptr %136, ptr %139)
  br label %141

141:                                              ; preds = %134, %130
  %142 = phi i1 [ %140, %134 ], [ false, %130 ]
  %143 = xor i1 %142, true
  br i1 %143, label %144, label %466

144:                                              ; preds = %141
  %145 = extractvalue { ptr, ptr } %19, 0
  %146 = extractvalue { ptr, ptr } %19, 1
  %147 = getelementptr ptr, ptr %145, i32 2
  %148 = load ptr, ptr %147, align 8
  %149 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %148, 0
  %150 = insertvalue { ptr, ptr, ptr } %149, ptr %146, 1
  %151 = insertvalue { ptr, ptr, ptr } %150, ptr null, 2
  %152 = extractvalue { ptr, ptr, ptr } %151, 1
  %153 = extractvalue { ptr, ptr, ptr } %151, 0
  %154 = call double %153(ptr %152)
  %155 = fcmp oeq double %154, 5.000000e+00
  br i1 %155, label %156, label %468

156:                                              ; preds = %144
  %157 = extractvalue { ptr, ptr } %34, 0
  %158 = extractvalue { ptr, ptr } %34, 1
  %159 = getelementptr ptr, ptr %157, i32 2
  %160 = load ptr, ptr %159, align 8
  %161 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %160, 0
  %162 = insertvalue { ptr, ptr, ptr } %161, ptr %158, 1
  %163 = insertvalue { ptr, ptr, ptr } %162, ptr null, 2
  %164 = extractvalue { ptr, ptr, ptr } %163, 1
  %165 = extractvalue { ptr, ptr, ptr } %163, 0
  %166 = call double %165(ptr %164)
  %167 = fcmp oeq double %166, 1.000000e+01
  br i1 %167, label %168, label %470

168:                                              ; preds = %156
  %169 = extractvalue { ptr, ptr } %50, 0
  %170 = extractvalue { ptr, ptr } %50, 1
  %171 = getelementptr ptr, ptr %169, i32 2
  %172 = load ptr, ptr %171, align 8
  %173 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %172, 0
  %174 = insertvalue { ptr, ptr, ptr } %173, ptr %170, 1
  %175 = insertvalue { ptr, ptr, ptr } %174, ptr null, 2
  %176 = extractvalue { ptr, ptr, ptr } %175, 1
  %177 = extractvalue { ptr, ptr, ptr } %175, 0
  %178 = call double %177(ptr %176)
  %179 = fcmp oeq double %178, 3.000000e+00
  br i1 %179, label %180, label %472

180:                                              ; preds = %168
  %181 = extractvalue { ptr, ptr } %19, 1
  %182 = ptrtoint ptr %181 to i64
  %183 = icmp ne i64 %182, 0
  br i1 %183, label %184, label %191

184:                                              ; preds = %180
  %185 = extractvalue { ptr, ptr } %19, 0
  %186 = extractvalue { ptr, ptr } %19, 1
  %187 = getelementptr ptr, ptr %185, i32 1
  %188 = load ptr, ptr %187, align 8
  %189 = load ptr, ptr @B..rtti, align 8
  %190 = call i1 %188(ptr %186, ptr %189)
  br label %191

191:                                              ; preds = %184, %180
  %192 = phi i1 [ %190, %184 ], [ false, %180 ]
  br i1 %192, label %193, label %211

193:                                              ; preds = %191
  %194 = extractvalue { ptr, ptr } %19, 1
  %195 = getelementptr %B, ptr %194, i32 0, i32 1
  store double 6.000000e+00, ptr %195, align 8
  %196 = extractvalue { ptr, ptr } %19, 0
  %197 = extractvalue { ptr, ptr } %19, 1
  %198 = getelementptr ptr, ptr %196, i32 2
  %199 = load ptr, ptr %198, align 8
  %200 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %199, 0
  %201 = insertvalue { ptr, ptr, ptr } %200, ptr %197, 1
  %202 = insertvalue { ptr, ptr, ptr } %201, ptr null, 2
  %203 = extractvalue { ptr, ptr, ptr } %202, 1
  %204 = extractvalue { ptr, ptr, ptr } %202, 0
  %205 = call double %204(ptr %203)
  %206 = fcmp oeq double %205, 6.000000e+00
  br i1 %206, label %207, label %474

207:                                              ; preds = %193
  %208 = extractvalue { ptr, ptr } %19, 1
  %209 = ptrtoint ptr %208 to i64
  %210 = icmp ne i64 %209, 0
  br i1 %210, label %213, label %220

211:                                              ; preds = %191
  %212 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %212, align 8
  call void @__cxa_throw(ptr %212, ptr @_ZTIPc, ptr null)
  br label %453

213:                                              ; preds = %207
  %214 = extractvalue { ptr, ptr } %19, 0
  %215 = extractvalue { ptr, ptr } %19, 1
  %216 = getelementptr ptr, ptr %214, i32 1
  %217 = load ptr, ptr %216, align 8
  %218 = load ptr, ptr @B..rtti, align 8
  %219 = call i1 %217(ptr %215, ptr %218)
  br label %220

220:                                              ; preds = %213, %207
  %221 = phi i1 [ %219, %213 ], [ false, %207 ]
  br i1 %221, label %222, label %231

222:                                              ; preds = %220
  %223 = extractvalue { ptr, ptr } %19, 1
  %224 = getelementptr %B, ptr %223, i32 0, i32 0
  %225 = call double @B.twice(ptr %223)
  %226 = fcmp oeq double %225, 1.200000e+01
  br i1 %226, label %227, label %476

227:                                              ; preds = %222
  %228 = extractvalue { ptr, ptr } %46, 1
  %229 = ptrtoint ptr %228 to i64
  %230 = icmp ne i64 %229, 0
  br i1 %230, label %233, label %240

231:                                              ; preds = %220
  %232 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %232, align 8
  call void @__cxa_throw(ptr %232, ptr @_ZTIPc, ptr null)
  br label %453

233:                                              ; preds = %227
  %234 = extractvalue { ptr, ptr } %46, 0
  %235 = extractvalue { ptr, ptr } %46, 1
  %236 = getelementptr ptr, ptr %234, i32 1
  %237 = load ptr, ptr %236, align 8
  %238 = load ptr, ptr @D..rtti, align 8
  %239 = call i1 %237(ptr %235, ptr %238)
  br label %240

240:                                              ; preds = %233, %227
  %241 = phi i1 [ %239, %233 ], [ false, %227 ]
  br i1 %241, label %242, label %263

242:                                              ; preds = %240
  %243 = extractvalue { ptr, ptr } %46, 1
  %244 = ptrtoint ptr %243 to i64
  %245 = icmp ne i64 %244, 0
  br i1 %245, label %246, label %253

246:                                              ; preds = %242
  %247 = extractvalue { ptr, ptr } %46, 0
  %248 = extractvalue { ptr, ptr } %46, 1
  %249 = getelementptr ptr, ptr %247, i32 1
  %250 = load ptr, ptr %249, align 8
  %251 = load ptr, ptr @D..rtti, align 8
  %252 = call i1 %250(ptr %248, ptr %251)
  br label %253

253:                                              ; preds = %246, %242
  %254 = phi i1 [ %252, %246 ], [ false, %242 ]
  br i1 %254, label %255, label %261

255:                                              ; preds = %253
  %256 = extractvalue { ptr, ptr } %46, 1
  %257 = getelementptr %D, ptr %256, i32 0, i32 1
  %258 = load double, ptr %257, align 8
  %259 = fcmp oeq double %258, 1.000000e+00
  br i1 %259, label %260, label %478

260:                                              ; preds = %255
  br label %263

261:                                              ; preds = %253
  %262 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %262, align 8
  call void @__cxa_throw(ptr %262, ptr @_ZTIPc, ptr null)
  br label %453

263:                                              ; preds = %260, %240
  %264 = extractvalue { ptr, ptr } %19, 1
  %265 = ptrtoint ptr %264 to i64
  %266 = icmp ne i64 %265, 0
  br i1 %266, label %267, label %274

267:                                              ; preds = %263
  %268 = extractvalue { ptr, ptr } %19, 0
  %269 = extractvalue { ptr, ptr } %19, 1
  %270 = getelementptr ptr, ptr %268, i32 1
  %271 = load ptr, ptr %270, align 8
  %272 = load ptr, ptr @Other..rtti, align 8
  %273 = call i1 %271(ptr %269, ptr %272)
  br label %274

274:                                              ; preds = %267, %263
  %275 = phi i1 [ %273, %267 ], [ false, %263 ]
  br i1 %275, label %276, label %278

276:                                              ; preds = %274
  %277 = extractvalue { ptr, ptr } %19, 1
  br label %282

278:                                              ; preds = %274
  %279 = call ptr @GC_malloc(i64 16)
  %280 = getelementptr i8, ptr %279, i64 8
  %281 = getelementptr %Other, ptr %280, i32 0, i32 0
  store ptr @Other..vtbl, ptr %281, align 8
  br label %282

282:                                              ; preds = %276, %278
  %283 = phi ptr [ %280, %278 ], [ %277, %276 ]
  %284 = getelementptr %Other, ptr %283, i32 0, i32 0
  %285 = call double @Other.get(ptr %283)
  %286 = fcmp oeq double %285, 9.000000e+00
  br i1 %286, label %287, label %480

287:                                              ; preds = %282
  %288 = call ptr @GC_malloc(i64 40)
  %289 = getelementptr i8, ptr %288, i64 8
  %290 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %289, i32 0, i32 0
  store i64 16, ptr %290, align 8
  %291 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %289, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_16873085711952587189_interface, i64 40), ptr %291, align 8
  %292 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %289, i32 0, i32 2
  store { ptr, ptr } %19, ptr %292, align 8
  %293 = call ptr @"___unbox<!ts.class<B, !ts.class_storage<B>>>"(ptr %289)
  %294 = getelementptr %B, ptr %293, i32 0, i32 1
  %295 = load double, ptr %294, align 8
  %296 = fcmp oeq double %295, 6.000000e+00
  br i1 %296, label %297, label %482

297:                                              ; preds = %287
  %298 = call ptr @GC_malloc(i64 40)
  %299 = getelementptr i8, ptr %298, i64 8
  %300 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %299, i32 0, i32 0
  store i64 16, ptr %300, align 8
  %301 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %299, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_16873085711952587189_interface, i64 40), ptr %301, align 8
  %302 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %299, i32 0, i32 2
  store { ptr, ptr } %50, ptr %302, align 8
  store i1 false, ptr %3, align 1
  %303 = invoke ptr @"___unbox<!ts.class<B, !ts.class_storage<B>>>"(ptr %299)
          to label %304 unwind label %311

304:                                              ; preds = %297
  %305 = getelementptr %B, ptr %303, i32 0, i32 1
  %306 = load double, ptr %305, align 8
  %307 = call ptr @GC_malloc_atomic(i64 40)
  %308 = getelementptr i8, ptr %307, i64 8
  call void @__ts_number_to_string(ptr %308, double %306)
  %309 = icmp eq ptr %308, null
  %310 = select i1 %309, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %308
  call void @puts(ptr %310)
  br label %450

311:                                              ; preds = %297
  %312 = landingpad { ptr, i32 }
          catch ptr null
  %313 = extractvalue { ptr, i32 } %312, 0
  %314 = call ptr @__cxa_begin_catch(ptr %313)
  %315 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %311
  %316 = icmp eq ptr %315, @_ZTIi
  br i1 %316, label %317, label %324

317:                                              ; preds = %.noexc9
  %318 = load i32, ptr %314, align 4
  %319 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %317
  %320 = getelementptr i8, ptr %319, i64 8
  %321 = getelementptr { i64, ptr, i32 }, ptr %320, i32 0, i32 0
  store i64 4, ptr %321, align 8
  %322 = getelementptr { i64, ptr, i32 }, ptr %320, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %322, align 8
  %323 = getelementptr { i64, ptr, i32 }, ptr %320, i32 0, i32 2
  store i32 %318, ptr %323, align 4
  br label %447

324:                                              ; preds = %.noexc9
  %325 = icmp eq ptr %315, @_ZTIj
  br i1 %325, label %326, label %333

326:                                              ; preds = %324
  %327 = load i32, ptr %314, align 4
  %328 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %326
  %329 = getelementptr i8, ptr %328, i64 8
  %330 = getelementptr { i64, ptr, i32 }, ptr %329, i32 0, i32 0
  store i64 4, ptr %330, align 8
  %331 = getelementptr { i64, ptr, i32 }, ptr %329, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %331, align 8
  %332 = getelementptr { i64, ptr, i32 }, ptr %329, i32 0, i32 2
  store i32 %327, ptr %332, align 4
  br label %444

333:                                              ; preds = %324
  %334 = icmp eq ptr %315, @_ZTId
  br i1 %334, label %335, label %342

335:                                              ; preds = %333
  %336 = load double, ptr %314, align 8
  %337 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %335
  %338 = getelementptr i8, ptr %337, i64 8
  %339 = getelementptr { i64, ptr, double }, ptr %338, i32 0, i32 0
  store i64 8, ptr %339, align 8
  %340 = getelementptr { i64, ptr, double }, ptr %338, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %340, align 8
  %341 = getelementptr { i64, ptr, double }, ptr %338, i32 0, i32 2
  store double %336, ptr %341, align 8
  br label %441

342:                                              ; preds = %333
  %343 = icmp eq ptr %315, @_ZTIf
  br i1 %343, label %344, label %351

344:                                              ; preds = %342
  %345 = load float, ptr %314, align 4
  %346 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %344
  %347 = getelementptr i8, ptr %346, i64 8
  %348 = getelementptr { i64, ptr, float }, ptr %347, i32 0, i32 0
  store i64 4, ptr %348, align 8
  %349 = getelementptr { i64, ptr, float }, ptr %347, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %349, align 8
  %350 = getelementptr { i64, ptr, float }, ptr %347, i32 0, i32 2
  store float %345, ptr %350, align 4
  br label %438

351:                                              ; preds = %342
  %352 = icmp eq ptr %315, @_ZTIb
  br i1 %352, label %353, label %360

353:                                              ; preds = %351
  %354 = load i1, ptr %314, align 1
  %355 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %353
  %356 = getelementptr i8, ptr %355, i64 8
  %357 = getelementptr { i64, ptr, i1 }, ptr %356, i32 0, i32 0
  store i64 1, ptr %357, align 8
  %358 = getelementptr { i64, ptr, i1 }, ptr %356, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %358, align 8
  %359 = getelementptr { i64, ptr, i1 }, ptr %356, i32 0, i32 2
  store i1 %354, ptr %359, align 1
  br label %435

360:                                              ; preds = %351
  %361 = icmp eq ptr %315, @_ZTIx
  br i1 %361, label %362, label %369

362:                                              ; preds = %360
  %363 = load i64, ptr %314, align 8
  %364 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %362
  %365 = getelementptr i8, ptr %364, i64 8
  %366 = getelementptr { i64, ptr, i64 }, ptr %365, i32 0, i32 0
  store i64 8, ptr %366, align 8
  %367 = getelementptr { i64, ptr, i64 }, ptr %365, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %367, align 8
  %368 = getelementptr { i64, ptr, i64 }, ptr %365, i32 0, i32 2
  store i64 %363, ptr %368, align 8
  br label %432

369:                                              ; preds = %360
  %370 = icmp eq ptr %315, @_ZTIPc
  br i1 %370, label %371, label %378

371:                                              ; preds = %369
  %372 = load ptr, ptr %314, align 8
  %373 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %371
  %374 = getelementptr i8, ptr %373, i64 8
  %375 = getelementptr { i64, ptr, ptr }, ptr %374, i32 0, i32 0
  store i64 8, ptr %375, align 8
  %376 = getelementptr { i64, ptr, ptr }, ptr %374, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %376, align 8
  %377 = getelementptr { i64, ptr, ptr }, ptr %374, i32 0, i32 2
  store ptr %372, ptr %377, align 8
  br label %429

378:                                              ; preds = %369
  %379 = icmp eq ptr %315, @_ZTIPv
  br i1 %379, label %380, label %382

380:                                              ; preds = %378
  %381 = load ptr, ptr %314, align 8
  br label %426

382:                                              ; preds = %378
  %383 = load ptr, ptr %315, align 8
  %384 = icmp eq ptr %383, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %384, label %385, label %416

385:                                              ; preds = %382
  %386 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %315, i32 0, i32 2
  %387 = load i32, ptr %386, align 4
  %388 = and i32 %387, 1073741824
  %389 = icmp ne i32 %388, 0
  br i1 %389, label %390, label %406

390:                                              ; preds = %385
  %391 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %315, i32 0, i32 4
  %392 = load ptr, ptr %391, align 8
  %393 = icmp ne ptr %392, null
  br i1 %393, label %394, label %396

394:                                              ; preds = %390
  invoke void %392(ptr %1, ptr %314)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %394
  %395 = load ptr, ptr %1, align 8
  br label %403

396:                                              ; preds = %390
  %397 = load ptr, ptr %314, align 8
  %398 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %396
  %399 = getelementptr i8, ptr %398, i64 8
  %400 = getelementptr { i64, ptr, ptr }, ptr %399, i32 0, i32 0
  store i64 8, ptr %400, align 8
  %401 = getelementptr { i64, ptr, ptr }, ptr %399, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %401, align 8
  %402 = getelementptr { i64, ptr, ptr }, ptr %399, i32 0, i32 2
  store ptr %397, ptr %402, align 8
  br label %403

403:                                              ; preds = %.noexc8, %.noexc5
  %404 = phi ptr [ %399, %.noexc5 ], [ %395, %.noexc8 ]
  br label %405

405:                                              ; preds = %403
  br label %413

406:                                              ; preds = %385
  %407 = load ptr, ptr %314, align 8
  %408 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %406
  %409 = getelementptr i8, ptr %408, i64 8
  %410 = getelementptr { i64, ptr, ptr }, ptr %409, i32 0, i32 0
  store i64 8, ptr %410, align 8
  %411 = getelementptr { i64, ptr, ptr }, ptr %409, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %411, align 8
  %412 = getelementptr { i64, ptr, ptr }, ptr %409, i32 0, i32 2
  store ptr %407, ptr %412, align 8
  br label %413

413:                                              ; preds = %405, %.noexc11
  %414 = phi ptr [ %409, %.noexc11 ], [ %404, %405 ]
  br label %415

415:                                              ; preds = %413
  br label %423

416:                                              ; preds = %382
  %417 = load ptr, ptr %314, align 8
  %418 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %416
  %419 = getelementptr i8, ptr %418, i64 8
  %420 = getelementptr { i64, ptr, ptr }, ptr %419, i32 0, i32 0
  store i64 8, ptr %420, align 8
  %421 = getelementptr { i64, ptr, ptr }, ptr %419, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %421, align 8
  %422 = getelementptr { i64, ptr, ptr }, ptr %419, i32 0, i32 2
  store ptr %417, ptr %422, align 8
  br label %423

423:                                              ; preds = %415, %.noexc2
  %424 = phi ptr [ %419, %.noexc2 ], [ %414, %415 ]
  br label %425

425:                                              ; preds = %423
  br label %426

426:                                              ; preds = %380, %425
  %427 = phi ptr [ %424, %425 ], [ %381, %380 ]
  br label %428

428:                                              ; preds = %426
  br label %429

429:                                              ; preds = %.noexc10, %428
  %430 = phi ptr [ %427, %428 ], [ %374, %.noexc10 ]
  br label %431

431:                                              ; preds = %429
  br label %432

432:                                              ; preds = %.noexc6, %431
  %433 = phi ptr [ %430, %431 ], [ %365, %.noexc6 ]
  br label %434

434:                                              ; preds = %432
  br label %435

435:                                              ; preds = %.noexc3, %434
  %436 = phi ptr [ %433, %434 ], [ %356, %.noexc3 ]
  br label %437

437:                                              ; preds = %435
  br label %438

438:                                              ; preds = %.noexc1, %437
  %439 = phi ptr [ %436, %437 ], [ %347, %.noexc1 ]
  br label %440

440:                                              ; preds = %438
  br label %441

441:                                              ; preds = %.noexc7, %440
  %442 = phi ptr [ %439, %440 ], [ %338, %.noexc7 ]
  br label %443

443:                                              ; preds = %441
  br label %444

444:                                              ; preds = %.noexc4, %443
  %445 = phi ptr [ %442, %443 ], [ %329, %.noexc4 ]
  br label %446

446:                                              ; preds = %444
  br label %447

447:                                              ; preds = %.noexc, %446
  %448 = phi ptr [ %445, %446 ], [ %320, %.noexc ]
  br label %449

449:                                              ; preds = %447
  store ptr %448, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %450

450:                                              ; preds = %304, %449
  %451 = load i1, ptr %3, align 1
  br i1 %451, label %452, label %484

452:                                              ; preds = %450
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

453:                                              ; preds = %484, %482, %480, %478, %261, %476, %231, %474, %211, %472, %470, %468, %466, %464, %462, %460, %458, %456, %454
  unreachable

454:                                              ; preds = %61
  %455 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 43, ptr null)
  br label %453

456:                                              ; preds = %74
  %457 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 44, ptr null)
  br label %453

458:                                              ; preds = %88
  %459 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 45, ptr null)
  br label %453

460:                                              ; preds = %102
  %461 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 46, ptr null)
  br label %453

462:                                              ; preds = %115
  %463 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 47, ptr null)
  br label %453

464:                                              ; preds = %128
  %465 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 48, ptr null)
  br label %453

466:                                              ; preds = %141
  %467 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 49, ptr null)
  br label %453

468:                                              ; preds = %144
  %469 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 52, ptr null)
  br label %453

470:                                              ; preds = %156
  %471 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 53, ptr null)
  br label %453

472:                                              ; preds = %168
  %473 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 54, ptr null)
  br label %453

474:                                              ; preds = %193
  %475 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 59, ptr null)
  br label %453

476:                                              ; preds = %222
  %477 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 60, ptr null)
  br label %453

478:                                              ; preds = %255
  %479 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 62, ptr null)
  br label %453

480:                                              ; preds = %282
  %481 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 66, ptr null)
  br label %453

482:                                              ; preds = %287
  %483 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 70, ptr null)
  br label %453

484:                                              ; preds = %450
  %485 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_159832223624545629, i64 8), i32 80, ptr null)
  br label %453

endcatch.resume:                                  ; preds = %406, %371, %311, %394, %335, %362, %396, %326, %353, %416, %344, %317
  %486 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %486
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

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "__null_interface_vtable" }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

