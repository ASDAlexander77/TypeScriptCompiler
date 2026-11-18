; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr }

$"___unbox<!ts.number>" = comdat any

$"___unbox<!ts.string>" = comdat any

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_12638206991764949794 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs\00", align 8
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
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_631772817358541784 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFu8\00", align 8
@s_637547452428770606 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs8\00", align 8
@s_627981701265227156 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFi8\00", align 8
@s_5558799945458185151 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFu16\00", align 8
@s_9334511880624667341 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs16\00", align 8
@s_3092775683940386555 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFi16\00", align 8
@s_9497966886403524235 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFindex\00", align 8
@s_17483980429552467645 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFchar\00", align 8
@s_5563585020063213298 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFu64\00", align 8
@s_9335351907508431320 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs64\00", align 8
@s_3089943341986682694 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFi64\00", align 8
@s_15957463049929925035 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf64\00", align 8
@s_1166204947305017450 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbigint\00", align 8
@s_5560783464435099345 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFu32\00", align 8
@s_9332519565554727459 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs32\00", align 8
@s_3094732814638223685 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFi32\00", align 8
@s_15952682373371409732 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf32\00", align 8
@s_16822532743728177535 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboolean\00", align 8
@s_8091808378024623192 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring\00", align 8
@s_9237349086447201248 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnumber\00", align 8
@s_2495249626774661283 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined\00", align 8
@__false__ = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfalse\00", align 8
@__true__ = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtrue\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@frmt_555419431375823311 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%u\00", align 8
@frmt_1189076927660576410 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-lli\00", align 8
@frmt_15473922249627149015 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%llu\00", align 8
@frmt_12865164881251510284 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-hi\00", align 8
@frmt_11110327606031142231 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%hu\00", align 8
@frmt_3788960132816803490 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-hhi\00", align 8
@frmt_12904420749775593087 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%hhu\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_16852112993072541847 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwrong at\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_11876826921290969655 = internal constant [113 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00catch_ended_on_unwind.ts\00", align 8
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIPc = external constant ptr
@fin = private global i32 0
@disposed = private global i32 0
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
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

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_rethrow()

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

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

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

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: mustprogress
define i32 @f(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp oge double %4, 0.000000e+00
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = load double, ptr %2, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 8)
  store double %7, ptr %8, align 8
  call void @__cxa_throw(ptr %8, ptr @_ZTId, ptr null)
  br label %11

9:                                                ; preds = %1
  store i32 1, ptr %3, align 4
  %10 = load i32, ptr %3, align 4
  ret i32 %10

11:                                               ; preds = %6
  unreachable
}

; Function Attrs: mustprogress
define i32 @fs(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp oge double %4, 0.000000e+00
  br i1 %5, label %6, label %29

6:                                                ; preds = %1
  %7 = load double, ptr %2, align 8
  %8 = call ptr @GC_malloc_atomic(i64 40)
  %9 = getelementptr i8, ptr %8, i64 8
  call void @__ts_number_to_string(ptr %9, double %7)
  %10 = icmp eq ptr %9, null
  %11 = select i1 %10, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %9
  %12 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8))
  %13 = add i64 1, %12
  %14 = call i64 @strlen(ptr %11)
  %15 = add i64 %13, %14
  %16 = add i64 %15, 8
  %17 = call ptr @GC_malloc(i64 %16)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr i8, ptr %18, i64 0
  %20 = sub i64 %15, 0
  %21 = call ptr @__memcpy_chk(ptr %19, ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8), i64 %12, i64 %20)
  %22 = add i64 0, %12
  %23 = getelementptr i8, ptr %18, i64 %22
  %24 = sub i64 %15, %22
  %25 = call ptr @__memcpy_chk(ptr %23, ptr %11, i64 %14, i64 %24)
  %26 = add i64 %22, %14
  %27 = getelementptr i8, ptr %18, i64 %26
  store i8 0, ptr %27, align 1
  %28 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %18, ptr %28, align 8
  call void @__cxa_throw(ptr %28, ptr @_ZTIPc, ptr null)
  br label %31

29:                                               ; preds = %1
  store i32 1, ptr %3, align 4
  %30 = load i32, ptr %3, align 4
  ret i32 %30

31:                                               ; preds = %6
  unreachable
}

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Res..rtti, align 8
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
define ptr @Res..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = load i32, ptr @disposed, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr @disposed, align 4
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr double @"___unbox<!ts.number>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr { i64, ptr, i8 }, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  %9 = zext i1 %8 to i32
  %10 = and i32 %9, 1
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %1
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %24

20:                                               ; preds = %19
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr { i64, ptr, double }, ptr %21, i32 0, i32 2
  %23 = load double, ptr %22, align 8
  store double %23, ptr %3, align 8
  br label %464

24:                                               ; preds = %19
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr { i64, ptr, i8 }, ptr %25, i32 0, i32 1
  %27 = load ptr, ptr %26, align 8
  %28 = ptrtoint ptr %27 to i64
  %29 = icmp ne i64 %28, 0
  %30 = zext i1 %29 to i32
  %31 = and i32 %30, 1
  %32 = icmp ne i32 %31, 0
  br i1 %32, label %33, label %36

33:                                               ; preds = %24
  %34 = call i32 @strcmp(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8))
  %35 = icmp eq i32 %34, 0
  br label %38

36:                                               ; preds = %24
  %37 = icmp eq i64 %28, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8) to i64)
  br label %38

38:                                               ; preds = %33, %36
  %39 = phi i1 [ %37, %36 ], [ %35, %33 ]
  br label %40

40:                                               ; preds = %38
  br i1 %39, label %41, label %45

41:                                               ; preds = %40
  %42 = load ptr, ptr %2, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 2
  %44 = load double, ptr %43, align 8
  store double %44, ptr %3, align 8
  br label %464

45:                                               ; preds = %40
  %46 = load ptr, ptr %2, align 8
  %47 = getelementptr { i64, ptr, i8 }, ptr %46, i32 0, i32 1
  %48 = load ptr, ptr %47, align 8
  %49 = ptrtoint ptr %48 to i64
  %50 = icmp ne i64 %49, 0
  %51 = zext i1 %50 to i32
  %52 = and i32 %51, 1
  %53 = icmp ne i32 %52, 0
  br i1 %53, label %54, label %57

54:                                               ; preds = %45
  %55 = call i32 @strcmp(ptr %48, ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8))
  %56 = icmp eq i32 %55, 0
  br label %59

57:                                               ; preds = %45
  %58 = icmp eq i64 %49, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8) to i64)
  br label %59

59:                                               ; preds = %54, %57
  %60 = phi i1 [ %58, %57 ], [ %56, %54 ]
  br label %61

61:                                               ; preds = %59
  br i1 %60, label %62, label %67

62:                                               ; preds = %61
  %63 = load ptr, ptr %2, align 8
  %64 = getelementptr { i64, ptr, ptr }, ptr %63, i32 0, i32 2
  %65 = load ptr, ptr %64, align 8
  %66 = call double @strtod(ptr %65, ptr null)
  store double %66, ptr %3, align 8
  br label %464

67:                                               ; preds = %61
  %68 = load ptr, ptr %2, align 8
  %69 = getelementptr { i64, ptr, i8 }, ptr %68, i32 0, i32 1
  %70 = load ptr, ptr %69, align 8
  %71 = ptrtoint ptr %70 to i64
  %72 = icmp ne i64 %71, 0
  %73 = zext i1 %72 to i32
  %74 = and i32 %73, 1
  %75 = icmp ne i32 %74, 0
  br i1 %75, label %76, label %79

76:                                               ; preds = %67
  %77 = call i32 @strcmp(ptr %70, ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8))
  %78 = icmp eq i32 %77, 0
  br label %81

79:                                               ; preds = %67
  %80 = icmp eq i64 %71, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8) to i64)
  br label %81

81:                                               ; preds = %76, %79
  %82 = phi i1 [ %80, %79 ], [ %78, %76 ]
  br label %83

83:                                               ; preds = %81
  br i1 %82, label %84, label %89

84:                                               ; preds = %83
  %85 = load ptr, ptr %2, align 8
  %86 = getelementptr { i64, ptr, i1 }, ptr %85, i32 0, i32 2
  %87 = load i1, ptr %86, align 1
  %88 = select i1 %87, double 1.000000e+00, double 0.000000e+00
  store double %88, ptr %3, align 8
  br label %464

89:                                               ; preds = %83
  %90 = load ptr, ptr %2, align 8
  %91 = getelementptr { i64, ptr, i8 }, ptr %90, i32 0, i32 1
  %92 = load ptr, ptr %91, align 8
  %93 = ptrtoint ptr %92 to i64
  %94 = icmp ne i64 %93, 0
  %95 = zext i1 %94 to i32
  %96 = and i32 %95, 1
  %97 = icmp ne i32 %96, 0
  br i1 %97, label %98, label %101

98:                                               ; preds = %89
  %99 = call i32 @strcmp(ptr %92, ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8))
  %100 = icmp eq i32 %99, 0
  br label %103

101:                                              ; preds = %89
  %102 = icmp eq i64 %93, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8) to i64)
  br label %103

103:                                              ; preds = %98, %101
  %104 = phi i1 [ %102, %101 ], [ %100, %98 ]
  br label %105

105:                                              ; preds = %103
  br i1 %104, label %106, label %111

106:                                              ; preds = %105
  %107 = load ptr, ptr %2, align 8
  %108 = getelementptr { i64, ptr, float }, ptr %107, i32 0, i32 2
  %109 = load float, ptr %108, align 4
  %110 = fpext float %109 to double
  store double %110, ptr %3, align 8
  br label %464

111:                                              ; preds = %105
  %112 = load ptr, ptr %2, align 8
  %113 = getelementptr { i64, ptr, i8 }, ptr %112, i32 0, i32 1
  %114 = load ptr, ptr %113, align 8
  %115 = ptrtoint ptr %114 to i64
  %116 = icmp ne i64 %115, 0
  %117 = zext i1 %116 to i32
  %118 = and i32 %117, 1
  %119 = icmp ne i32 %118, 0
  br i1 %119, label %120, label %123

120:                                              ; preds = %111
  %121 = call i32 @strcmp(ptr %114, ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8))
  %122 = icmp eq i32 %121, 0
  br label %125

123:                                              ; preds = %111
  %124 = icmp eq i64 %115, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8) to i64)
  br label %125

125:                                              ; preds = %120, %123
  %126 = phi i1 [ %124, %123 ], [ %122, %120 ]
  br label %127

127:                                              ; preds = %125
  br i1 %126, label %128, label %133

128:                                              ; preds = %127
  %129 = load ptr, ptr %2, align 8
  %130 = getelementptr { i64, ptr, i32 }, ptr %129, i32 0, i32 2
  %131 = load i32, ptr %130, align 4
  %132 = sitofp i32 %131 to double
  store double %132, ptr %3, align 8
  br label %464

133:                                              ; preds = %127
  %134 = load ptr, ptr %2, align 8
  %135 = getelementptr { i64, ptr, i8 }, ptr %134, i32 0, i32 1
  %136 = load ptr, ptr %135, align 8
  %137 = ptrtoint ptr %136 to i64
  %138 = icmp ne i64 %137, 0
  %139 = zext i1 %138 to i32
  %140 = and i32 %139, 1
  %141 = icmp ne i32 %140, 0
  br i1 %141, label %142, label %145

142:                                              ; preds = %133
  %143 = call i32 @strcmp(ptr %136, ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8))
  %144 = icmp eq i32 %143, 0
  br label %147

145:                                              ; preds = %133
  %146 = icmp eq i64 %137, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8) to i64)
  br label %147

147:                                              ; preds = %142, %145
  %148 = phi i1 [ %146, %145 ], [ %144, %142 ]
  br label %149

149:                                              ; preds = %147
  br i1 %148, label %150, label %155

150:                                              ; preds = %149
  %151 = load ptr, ptr %2, align 8
  %152 = getelementptr { i64, ptr, i32 }, ptr %151, i32 0, i32 2
  %153 = load i32, ptr %152, align 4
  %154 = sitofp i32 %153 to double
  store double %154, ptr %3, align 8
  br label %464

155:                                              ; preds = %149
  %156 = load ptr, ptr %2, align 8
  %157 = getelementptr { i64, ptr, i8 }, ptr %156, i32 0, i32 1
  %158 = load ptr, ptr %157, align 8
  %159 = ptrtoint ptr %158 to i64
  %160 = icmp ne i64 %159, 0
  %161 = zext i1 %160 to i32
  %162 = and i32 %161, 1
  %163 = icmp ne i32 %162, 0
  br i1 %163, label %164, label %167

164:                                              ; preds = %155
  %165 = call i32 @strcmp(ptr %158, ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8))
  %166 = icmp eq i32 %165, 0
  br label %169

167:                                              ; preds = %155
  %168 = icmp eq i64 %159, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8) to i64)
  br label %169

169:                                              ; preds = %164, %167
  %170 = phi i1 [ %168, %167 ], [ %166, %164 ]
  br label %171

171:                                              ; preds = %169
  br i1 %170, label %172, label %177

172:                                              ; preds = %171
  %173 = load ptr, ptr %2, align 8
  %174 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 2
  %175 = load i32, ptr %174, align 4
  %176 = uitofp i32 %175 to double
  store double %176, ptr %3, align 8
  br label %464

177:                                              ; preds = %171
  %178 = load ptr, ptr %2, align 8
  %179 = getelementptr { i64, ptr, i8 }, ptr %178, i32 0, i32 1
  %180 = load ptr, ptr %179, align 8
  %181 = ptrtoint ptr %180 to i64
  %182 = icmp ne i64 %181, 0
  %183 = zext i1 %182 to i32
  %184 = and i32 %183, 1
  %185 = icmp ne i32 %184, 0
  br i1 %185, label %186, label %189

186:                                              ; preds = %177
  %187 = call i32 @strcmp(ptr %180, ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8))
  %188 = icmp eq i32 %187, 0
  br label %191

189:                                              ; preds = %177
  %190 = icmp eq i64 %181, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8) to i64)
  br label %191

191:                                              ; preds = %186, %189
  %192 = phi i1 [ %190, %189 ], [ %188, %186 ]
  br label %193

193:                                              ; preds = %191
  br i1 %192, label %194, label %199

194:                                              ; preds = %193
  %195 = load ptr, ptr %2, align 8
  %196 = getelementptr { i64, ptr, i64 }, ptr %195, i32 0, i32 2
  %197 = load i64, ptr %196, align 8
  %198 = sitofp i64 %197 to double
  store double %198, ptr %3, align 8
  br label %464

199:                                              ; preds = %193
  %200 = load ptr, ptr %2, align 8
  %201 = getelementptr { i64, ptr, i8 }, ptr %200, i32 0, i32 1
  %202 = load ptr, ptr %201, align 8
  %203 = ptrtoint ptr %202 to i64
  %204 = icmp ne i64 %203, 0
  %205 = zext i1 %204 to i32
  %206 = and i32 %205, 1
  %207 = icmp ne i32 %206, 0
  br i1 %207, label %208, label %211

208:                                              ; preds = %199
  %209 = call i32 @strcmp(ptr %202, ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8))
  %210 = icmp eq i32 %209, 0
  br label %213

211:                                              ; preds = %199
  %212 = icmp eq i64 %203, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8) to i64)
  br label %213

213:                                              ; preds = %208, %211
  %214 = phi i1 [ %212, %211 ], [ %210, %208 ]
  br label %215

215:                                              ; preds = %213
  br i1 %214, label %216, label %220

216:                                              ; preds = %215
  %217 = load ptr, ptr %2, align 8
  %218 = getelementptr { i64, ptr, double }, ptr %217, i32 0, i32 2
  %219 = load double, ptr %218, align 8
  store double %219, ptr %3, align 8
  br label %464

220:                                              ; preds = %215
  %221 = load ptr, ptr %2, align 8
  %222 = getelementptr { i64, ptr, i8 }, ptr %221, i32 0, i32 1
  %223 = load ptr, ptr %222, align 8
  %224 = ptrtoint ptr %223 to i64
  %225 = icmp ne i64 %224, 0
  %226 = zext i1 %225 to i32
  %227 = and i32 %226, 1
  %228 = icmp ne i32 %227, 0
  br i1 %228, label %229, label %232

229:                                              ; preds = %220
  %230 = call i32 @strcmp(ptr %223, ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8))
  %231 = icmp eq i32 %230, 0
  br label %234

232:                                              ; preds = %220
  %233 = icmp eq i64 %224, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8) to i64)
  br label %234

234:                                              ; preds = %229, %232
  %235 = phi i1 [ %233, %232 ], [ %231, %229 ]
  br label %236

236:                                              ; preds = %234
  br i1 %235, label %237, label %242

237:                                              ; preds = %236
  %238 = load ptr, ptr %2, align 8
  %239 = getelementptr { i64, ptr, i64 }, ptr %238, i32 0, i32 2
  %240 = load i64, ptr %239, align 8
  %241 = sitofp i64 %240 to double
  store double %241, ptr %3, align 8
  br label %464

242:                                              ; preds = %236
  %243 = load ptr, ptr %2, align 8
  %244 = getelementptr { i64, ptr, i8 }, ptr %243, i32 0, i32 1
  %245 = load ptr, ptr %244, align 8
  %246 = ptrtoint ptr %245 to i64
  %247 = icmp ne i64 %246, 0
  %248 = zext i1 %247 to i32
  %249 = and i32 %248, 1
  %250 = icmp ne i32 %249, 0
  br i1 %250, label %251, label %254

251:                                              ; preds = %242
  %252 = call i32 @strcmp(ptr %245, ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8))
  %253 = icmp eq i32 %252, 0
  br label %256

254:                                              ; preds = %242
  %255 = icmp eq i64 %246, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8) to i64)
  br label %256

256:                                              ; preds = %251, %254
  %257 = phi i1 [ %255, %254 ], [ %253, %251 ]
  br label %258

258:                                              ; preds = %256
  br i1 %257, label %259, label %264

259:                                              ; preds = %258
  %260 = load ptr, ptr %2, align 8
  %261 = getelementptr { i64, ptr, i64 }, ptr %260, i32 0, i32 2
  %262 = load i64, ptr %261, align 8
  %263 = sitofp i64 %262 to double
  store double %263, ptr %3, align 8
  br label %464

264:                                              ; preds = %258
  %265 = load ptr, ptr %2, align 8
  %266 = getelementptr { i64, ptr, i8 }, ptr %265, i32 0, i32 1
  %267 = load ptr, ptr %266, align 8
  %268 = ptrtoint ptr %267 to i64
  %269 = icmp ne i64 %268, 0
  %270 = zext i1 %269 to i32
  %271 = and i32 %270, 1
  %272 = icmp ne i32 %271, 0
  br i1 %272, label %273, label %276

273:                                              ; preds = %264
  %274 = call i32 @strcmp(ptr %267, ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8))
  %275 = icmp eq i32 %274, 0
  br label %278

276:                                              ; preds = %264
  %277 = icmp eq i64 %268, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8) to i64)
  br label %278

278:                                              ; preds = %273, %276
  %279 = phi i1 [ %277, %276 ], [ %275, %273 ]
  br label %280

280:                                              ; preds = %278
  br i1 %279, label %281, label %286

281:                                              ; preds = %280
  %282 = load ptr, ptr %2, align 8
  %283 = getelementptr { i64, ptr, i64 }, ptr %282, i32 0, i32 2
  %284 = load i64, ptr %283, align 8
  %285 = uitofp i64 %284 to double
  store double %285, ptr %3, align 8
  br label %464

286:                                              ; preds = %280
  %287 = load ptr, ptr %2, align 8
  %288 = getelementptr { i64, ptr, i8 }, ptr %287, i32 0, i32 1
  %289 = load ptr, ptr %288, align 8
  %290 = ptrtoint ptr %289 to i64
  %291 = icmp ne i64 %290, 0
  %292 = zext i1 %291 to i32
  %293 = and i32 %292, 1
  %294 = icmp ne i32 %293, 0
  br i1 %294, label %295, label %298

295:                                              ; preds = %286
  %296 = call i32 @strcmp(ptr %289, ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8))
  %297 = icmp eq i32 %296, 0
  br label %300

298:                                              ; preds = %286
  %299 = icmp eq i64 %290, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8) to i64)
  br label %300

300:                                              ; preds = %295, %298
  %301 = phi i1 [ %299, %298 ], [ %297, %295 ]
  br label %302

302:                                              ; preds = %300
  br i1 %301, label %303, label %308

303:                                              ; preds = %302
  %304 = load ptr, ptr %2, align 8
  %305 = getelementptr { i64, ptr, i8 }, ptr %304, i32 0, i32 2
  %306 = load i8, ptr %305, align 1
  %307 = sitofp i8 %306 to double
  store double %307, ptr %3, align 8
  br label %464

308:                                              ; preds = %302
  %309 = load ptr, ptr %2, align 8
  %310 = getelementptr { i64, ptr, i8 }, ptr %309, i32 0, i32 1
  %311 = load ptr, ptr %310, align 8
  %312 = ptrtoint ptr %311 to i64
  %313 = icmp ne i64 %312, 0
  %314 = zext i1 %313 to i32
  %315 = and i32 %314, 1
  %316 = icmp ne i32 %315, 0
  br i1 %316, label %317, label %320

317:                                              ; preds = %308
  %318 = call i32 @strcmp(ptr %311, ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8))
  %319 = icmp eq i32 %318, 0
  br label %322

320:                                              ; preds = %308
  %321 = icmp eq i64 %312, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8) to i64)
  br label %322

322:                                              ; preds = %317, %320
  %323 = phi i1 [ %321, %320 ], [ %319, %317 ]
  br label %324

324:                                              ; preds = %322
  br i1 %323, label %325, label %330

325:                                              ; preds = %324
  %326 = load ptr, ptr %2, align 8
  %327 = getelementptr { i64, ptr, i64 }, ptr %326, i32 0, i32 2
  %328 = load i64, ptr %327, align 8
  %329 = sitofp i64 %328 to double
  store double %329, ptr %3, align 8
  br label %464

330:                                              ; preds = %324
  %331 = load ptr, ptr %2, align 8
  %332 = getelementptr { i64, ptr, i8 }, ptr %331, i32 0, i32 1
  %333 = load ptr, ptr %332, align 8
  %334 = ptrtoint ptr %333 to i64
  %335 = icmp ne i64 %334, 0
  %336 = zext i1 %335 to i32
  %337 = and i32 %336, 1
  %338 = icmp ne i32 %337, 0
  br i1 %338, label %339, label %342

339:                                              ; preds = %330
  %340 = call i32 @strcmp(ptr %333, ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8))
  %341 = icmp eq i32 %340, 0
  br label %344

342:                                              ; preds = %330
  %343 = icmp eq i64 %334, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8) to i64)
  br label %344

344:                                              ; preds = %339, %342
  %345 = phi i1 [ %343, %342 ], [ %341, %339 ]
  br label %346

346:                                              ; preds = %344
  br i1 %345, label %347, label %352

347:                                              ; preds = %346
  %348 = load ptr, ptr %2, align 8
  %349 = getelementptr { i64, ptr, i16 }, ptr %348, i32 0, i32 2
  %350 = load i16, ptr %349, align 2
  %351 = sitofp i16 %350 to double
  store double %351, ptr %3, align 8
  br label %464

352:                                              ; preds = %346
  %353 = load ptr, ptr %2, align 8
  %354 = getelementptr { i64, ptr, i8 }, ptr %353, i32 0, i32 1
  %355 = load ptr, ptr %354, align 8
  %356 = ptrtoint ptr %355 to i64
  %357 = icmp ne i64 %356, 0
  %358 = zext i1 %357 to i32
  %359 = and i32 %358, 1
  %360 = icmp ne i32 %359, 0
  br i1 %360, label %361, label %364

361:                                              ; preds = %352
  %362 = call i32 @strcmp(ptr %355, ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8))
  %363 = icmp eq i32 %362, 0
  br label %366

364:                                              ; preds = %352
  %365 = icmp eq i64 %356, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8) to i64)
  br label %366

366:                                              ; preds = %361, %364
  %367 = phi i1 [ %365, %364 ], [ %363, %361 ]
  br label %368

368:                                              ; preds = %366
  br i1 %367, label %369, label %374

369:                                              ; preds = %368
  %370 = load ptr, ptr %2, align 8
  %371 = getelementptr { i64, ptr, i16 }, ptr %370, i32 0, i32 2
  %372 = load i16, ptr %371, align 2
  %373 = sitofp i16 %372 to double
  store double %373, ptr %3, align 8
  br label %464

374:                                              ; preds = %368
  %375 = load ptr, ptr %2, align 8
  %376 = getelementptr { i64, ptr, i8 }, ptr %375, i32 0, i32 1
  %377 = load ptr, ptr %376, align 8
  %378 = ptrtoint ptr %377 to i64
  %379 = icmp ne i64 %378, 0
  %380 = zext i1 %379 to i32
  %381 = and i32 %380, 1
  %382 = icmp ne i32 %381, 0
  br i1 %382, label %383, label %386

383:                                              ; preds = %374
  %384 = call i32 @strcmp(ptr %377, ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8))
  %385 = icmp eq i32 %384, 0
  br label %388

386:                                              ; preds = %374
  %387 = icmp eq i64 %378, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8) to i64)
  br label %388

388:                                              ; preds = %383, %386
  %389 = phi i1 [ %387, %386 ], [ %385, %383 ]
  br label %390

390:                                              ; preds = %388
  br i1 %389, label %391, label %396

391:                                              ; preds = %390
  %392 = load ptr, ptr %2, align 8
  %393 = getelementptr { i64, ptr, i16 }, ptr %392, i32 0, i32 2
  %394 = load i16, ptr %393, align 2
  %395 = uitofp i16 %394 to double
  store double %395, ptr %3, align 8
  br label %464

396:                                              ; preds = %390
  %397 = load ptr, ptr %2, align 8
  %398 = getelementptr { i64, ptr, i8 }, ptr %397, i32 0, i32 1
  %399 = load ptr, ptr %398, align 8
  %400 = ptrtoint ptr %399 to i64
  %401 = icmp ne i64 %400, 0
  %402 = zext i1 %401 to i32
  %403 = and i32 %402, 1
  %404 = icmp ne i32 %403, 0
  br i1 %404, label %405, label %408

405:                                              ; preds = %396
  %406 = call i32 @strcmp(ptr %399, ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8))
  %407 = icmp eq i32 %406, 0
  br label %410

408:                                              ; preds = %396
  %409 = icmp eq i64 %400, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8) to i64)
  br label %410

410:                                              ; preds = %405, %408
  %411 = phi i1 [ %409, %408 ], [ %407, %405 ]
  br label %412

412:                                              ; preds = %410
  br i1 %411, label %413, label %418

413:                                              ; preds = %412
  %414 = load ptr, ptr %2, align 8
  %415 = getelementptr { i64, ptr, i8 }, ptr %414, i32 0, i32 2
  %416 = load i8, ptr %415, align 1
  %417 = sitofp i8 %416 to double
  store double %417, ptr %3, align 8
  br label %464

418:                                              ; preds = %412
  %419 = load ptr, ptr %2, align 8
  %420 = getelementptr { i64, ptr, i8 }, ptr %419, i32 0, i32 1
  %421 = load ptr, ptr %420, align 8
  %422 = ptrtoint ptr %421 to i64
  %423 = icmp ne i64 %422, 0
  %424 = zext i1 %423 to i32
  %425 = and i32 %424, 1
  %426 = icmp ne i32 %425, 0
  br i1 %426, label %427, label %430

427:                                              ; preds = %418
  %428 = call i32 @strcmp(ptr %421, ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8))
  %429 = icmp eq i32 %428, 0
  br label %432

430:                                              ; preds = %418
  %431 = icmp eq i64 %422, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8) to i64)
  br label %432

432:                                              ; preds = %427, %430
  %433 = phi i1 [ %431, %430 ], [ %429, %427 ]
  br label %434

434:                                              ; preds = %432
  br i1 %433, label %435, label %440

435:                                              ; preds = %434
  %436 = load ptr, ptr %2, align 8
  %437 = getelementptr { i64, ptr, i8 }, ptr %436, i32 0, i32 2
  %438 = load i8, ptr %437, align 1
  %439 = sitofp i8 %438 to double
  store double %439, ptr %3, align 8
  br label %464

440:                                              ; preds = %434
  %441 = load ptr, ptr %2, align 8
  %442 = getelementptr { i64, ptr, i8 }, ptr %441, i32 0, i32 1
  %443 = load ptr, ptr %442, align 8
  %444 = ptrtoint ptr %443 to i64
  %445 = icmp ne i64 %444, 0
  %446 = zext i1 %445 to i32
  %447 = and i32 %446, 1
  %448 = icmp ne i32 %447, 0
  br i1 %448, label %449, label %452

449:                                              ; preds = %440
  %450 = call i32 @strcmp(ptr %443, ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8))
  %451 = icmp eq i32 %450, 0
  br label %454

452:                                              ; preds = %440
  %453 = icmp eq i64 %444, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8) to i64)
  br label %454

454:                                              ; preds = %449, %452
  %455 = phi i1 [ %453, %452 ], [ %451, %449 ]
  br label %456

456:                                              ; preds = %454
  br i1 %455, label %457, label %462

457:                                              ; preds = %456
  %458 = load ptr, ptr %2, align 8
  %459 = getelementptr { i64, ptr, i8 }, ptr %458, i32 0, i32 2
  %460 = load i8, ptr %459, align 1
  %461 = uitofp i8 %460 to double
  store double %461, ptr %3, align 8
  br label %464

462:                                              ; preds = %456
  %463 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %463, align 8
  call void @__cxa_throw(ptr %463, ptr @_ZTIPc, ptr null)
  br label %466

464:                                              ; preds = %20, %41, %62, %84, %106, %128, %150, %172, %194, %216, %237, %259, %281, %303, %325, %347, %369, %391, %413, %435, %457
  %465 = load double, ptr %3, align 8
  ret double %465

466:                                              ; preds = %462
  unreachable
}

; Function Attrs: mustprogress
define double @callInCatch(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  store double %0, ptr %6, align 8
  %8 = load double, ptr %6, align 8
  %9 = invoke i32 @f(double %8)
          to label %10 unwind label %11

10:                                               ; preds = %1
  store double -1.000000e+00, ptr %7, align 8
  br label %297

11:                                               ; preds = %1
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc20
  %18 = load i32, ptr %14, align 4
  %19 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %17
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 0
  store i64 4, ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 2
  store i32 %18, ptr %23, align 4
  br label %147

24:                                               ; preds = %.noexc20
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %26
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 0
  store i64 4, ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 2
  store i32 %27, ptr %32, align 4
  br label %144

33:                                               ; preds = %24
  %34 = icmp eq ptr %15, @_ZTId
  br i1 %34, label %35, label %42

35:                                               ; preds = %33
  %36 = load double, ptr %14, align 8
  %37 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %35
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 0
  store i64 8, ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 2
  store double %36, ptr %41, align 8
  br label %141

42:                                               ; preds = %33
  %43 = icmp eq ptr %15, @_ZTIf
  br i1 %43, label %44, label %51

44:                                               ; preds = %42
  %45 = load float, ptr %14, align 4
  %46 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %44
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 0
  store i64 4, ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 2
  store float %45, ptr %50, align 4
  br label %138

51:                                               ; preds = %42
  %52 = icmp eq ptr %15, @_ZTIb
  br i1 %52, label %53, label %60

53:                                               ; preds = %51
  %54 = load i1, ptr %14, align 1
  %55 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %53
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 0
  store i64 1, ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 2
  store i1 %54, ptr %59, align 1
  br label %135

60:                                               ; preds = %51
  %61 = icmp eq ptr %15, @_ZTIx
  br i1 %61, label %62, label %69

62:                                               ; preds = %60
  %63 = load i64, ptr %14, align 8
  %64 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %62
  %65 = getelementptr i8, ptr %64, i64 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 0
  store i64 8, ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 2
  store i64 %63, ptr %68, align 8
  br label %132

69:                                               ; preds = %60
  %70 = icmp eq ptr %15, @_ZTIPc
  br i1 %70, label %71, label %78

71:                                               ; preds = %69
  %72 = load ptr, ptr %14, align 8
  %73 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %71
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 0
  store i64 8, ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 2
  store ptr %72, ptr %77, align 8
  br label %129

78:                                               ; preds = %69
  %79 = icmp eq ptr %15, @_ZTIPv
  br i1 %79, label %80, label %82

80:                                               ; preds = %78
  %81 = load ptr, ptr %14, align 8
  br label %126

82:                                               ; preds = %78
  %83 = load ptr, ptr %15, align 8
  %84 = icmp eq ptr %83, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %84, label %85, label %116

85:                                               ; preds = %82
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 2
  %87 = load i32, ptr %86, align 4
  %88 = and i32 %87, 1073741824
  %89 = icmp ne i32 %88, 0
  br i1 %89, label %90, label %106

90:                                               ; preds = %85
  %91 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 4
  %92 = load ptr, ptr %91, align 8
  %93 = icmp ne ptr %92, null
  br i1 %93, label %94, label %96

94:                                               ; preds = %90
  invoke void %92(ptr %3, ptr %14)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %94
  %95 = load ptr, ptr %3, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc11, %.noexc4
  %104 = phi ptr [ %99, %.noexc4 ], [ %95, %.noexc11 ]
  br label %105

105:                                              ; preds = %103
  br label %113

106:                                              ; preds = %85
  %107 = load ptr, ptr %14, align 8
  %108 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %106
  %109 = getelementptr i8, ptr %108, i64 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 0
  store i64 8, ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 2
  store ptr %107, ptr %112, align 8
  br label %113

113:                                              ; preds = %105, %.noexc
  %114 = phi ptr [ %109, %.noexc ], [ %104, %105 ]
  br label %115

115:                                              ; preds = %113
  br label %123

116:                                              ; preds = %82
  %117 = load ptr, ptr %14, align 8
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc22
  %124 = phi ptr [ %119, %.noexc22 ], [ %114, %115 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %80, %125
  %127 = phi ptr [ %124, %125 ], [ %81, %80 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc9, %128
  %130 = phi ptr [ %127, %128 ], [ %74, %.noexc9 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc16, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc16 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc5, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc5 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc14, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc14 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc15, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc15 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc21, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc21 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc1, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc1 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %4, align 8
  %150 = load double, ptr %6, align 8
  %151 = fadd double %150, 1.000000e+00
  %152 = invoke i32 @f(double %151)
          to label %153 unwind label %endcatch.pad

153:                                              ; preds = %149
  store double 0.000000e+00, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %297

endcatch.pad:                                     ; preds = %149
  %154 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

155:                                              ; No predecessors!
  %156 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %155
  %157 = phi { ptr, i32 } [ %156, %155 ], [ %154, %endcatch.pad ]
  %158 = extractvalue { ptr, i32 } %157, 0
  %159 = call ptr @__cxa_begin_catch(ptr %158)
  %160 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %.caught
  %161 = icmp eq ptr %160, @_ZTIi
  br i1 %161, label %162, label %169

162:                                              ; preds = %.noexc18
  %163 = load i32, ptr %159, align 4
  %164 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %162
  %165 = getelementptr i8, ptr %164, i64 8
  %166 = getelementptr { i64, ptr, i32 }, ptr %165, i32 0, i32 0
  store i64 4, ptr %166, align 8
  %167 = getelementptr { i64, ptr, i32 }, ptr %165, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %167, align 8
  %168 = getelementptr { i64, ptr, i32 }, ptr %165, i32 0, i32 2
  store i32 %163, ptr %168, align 4
  br label %292

169:                                              ; preds = %.noexc18
  %170 = icmp eq ptr %160, @_ZTIj
  br i1 %170, label %171, label %178

171:                                              ; preds = %169
  %172 = load i32, ptr %159, align 4
  %173 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %171
  %174 = getelementptr i8, ptr %173, i64 8
  %175 = getelementptr { i64, ptr, i32 }, ptr %174, i32 0, i32 0
  store i64 4, ptr %175, align 8
  %176 = getelementptr { i64, ptr, i32 }, ptr %174, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %176, align 8
  %177 = getelementptr { i64, ptr, i32 }, ptr %174, i32 0, i32 2
  store i32 %172, ptr %177, align 4
  br label %289

178:                                              ; preds = %169
  %179 = icmp eq ptr %160, @_ZTId
  br i1 %179, label %180, label %187

180:                                              ; preds = %178
  %181 = load double, ptr %159, align 8
  %182 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %180
  %183 = getelementptr i8, ptr %182, i64 8
  %184 = getelementptr { i64, ptr, double }, ptr %183, i32 0, i32 0
  store i64 8, ptr %184, align 8
  %185 = getelementptr { i64, ptr, double }, ptr %183, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %185, align 8
  %186 = getelementptr { i64, ptr, double }, ptr %183, i32 0, i32 2
  store double %181, ptr %186, align 8
  br label %286

187:                                              ; preds = %178
  %188 = icmp eq ptr %160, @_ZTIf
  br i1 %188, label %189, label %196

189:                                              ; preds = %187
  %190 = load float, ptr %159, align 4
  %191 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %189
  %192 = getelementptr i8, ptr %191, i64 8
  %193 = getelementptr { i64, ptr, float }, ptr %192, i32 0, i32 0
  store i64 4, ptr %193, align 8
  %194 = getelementptr { i64, ptr, float }, ptr %192, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %194, align 8
  %195 = getelementptr { i64, ptr, float }, ptr %192, i32 0, i32 2
  store float %190, ptr %195, align 4
  br label %283

196:                                              ; preds = %187
  %197 = icmp eq ptr %160, @_ZTIb
  br i1 %197, label %198, label %205

198:                                              ; preds = %196
  %199 = load i1, ptr %159, align 1
  %200 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %198
  %201 = getelementptr i8, ptr %200, i64 8
  %202 = getelementptr { i64, ptr, i1 }, ptr %201, i32 0, i32 0
  store i64 1, ptr %202, align 8
  %203 = getelementptr { i64, ptr, i1 }, ptr %201, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %203, align 8
  %204 = getelementptr { i64, ptr, i1 }, ptr %201, i32 0, i32 2
  store i1 %199, ptr %204, align 1
  br label %280

205:                                              ; preds = %196
  %206 = icmp eq ptr %160, @_ZTIx
  br i1 %206, label %207, label %214

207:                                              ; preds = %205
  %208 = load i64, ptr %159, align 8
  %209 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %207
  %210 = getelementptr i8, ptr %209, i64 8
  %211 = getelementptr { i64, ptr, i64 }, ptr %210, i32 0, i32 0
  store i64 8, ptr %211, align 8
  %212 = getelementptr { i64, ptr, i64 }, ptr %210, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %212, align 8
  %213 = getelementptr { i64, ptr, i64 }, ptr %210, i32 0, i32 2
  store i64 %208, ptr %213, align 8
  br label %277

214:                                              ; preds = %205
  %215 = icmp eq ptr %160, @_ZTIPc
  br i1 %215, label %216, label %223

216:                                              ; preds = %214
  %217 = load ptr, ptr %159, align 8
  %218 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %216
  %219 = getelementptr i8, ptr %218, i64 8
  %220 = getelementptr { i64, ptr, ptr }, ptr %219, i32 0, i32 0
  store i64 8, ptr %220, align 8
  %221 = getelementptr { i64, ptr, ptr }, ptr %219, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %221, align 8
  %222 = getelementptr { i64, ptr, ptr }, ptr %219, i32 0, i32 2
  store ptr %217, ptr %222, align 8
  br label %274

223:                                              ; preds = %214
  %224 = icmp eq ptr %160, @_ZTIPv
  br i1 %224, label %225, label %227

225:                                              ; preds = %223
  %226 = load ptr, ptr %159, align 8
  br label %271

227:                                              ; preds = %223
  %228 = load ptr, ptr %160, align 8
  %229 = icmp eq ptr %228, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %229, label %230, label %261

230:                                              ; preds = %227
  %231 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %160, i32 0, i32 2
  %232 = load i32, ptr %231, align 4
  %233 = and i32 %232, 1073741824
  %234 = icmp ne i32 %233, 0
  br i1 %234, label %235, label %251

235:                                              ; preds = %230
  %236 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %160, i32 0, i32 4
  %237 = load ptr, ptr %236, align 8
  %238 = icmp ne ptr %237, null
  br i1 %238, label %239, label %241

239:                                              ; preds = %235
  invoke void %237(ptr %2, ptr %159)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %239
  %240 = load ptr, ptr %2, align 8
  br label %248

241:                                              ; preds = %235
  %242 = load ptr, ptr %159, align 8
  %243 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %241
  %244 = getelementptr i8, ptr %243, i64 8
  %245 = getelementptr { i64, ptr, ptr }, ptr %244, i32 0, i32 0
  store i64 8, ptr %245, align 8
  %246 = getelementptr { i64, ptr, ptr }, ptr %244, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %246, align 8
  %247 = getelementptr { i64, ptr, ptr }, ptr %244, i32 0, i32 2
  store ptr %242, ptr %247, align 8
  br label %248

248:                                              ; preds = %.noexc12, %.noexc19
  %249 = phi ptr [ %244, %.noexc19 ], [ %240, %.noexc12 ]
  br label %250

250:                                              ; preds = %248
  br label %258

251:                                              ; preds = %230
  %252 = load ptr, ptr %159, align 8
  %253 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %251
  %254 = getelementptr i8, ptr %253, i64 8
  %255 = getelementptr { i64, ptr, ptr }, ptr %254, i32 0, i32 0
  store i64 8, ptr %255, align 8
  %256 = getelementptr { i64, ptr, ptr }, ptr %254, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %256, align 8
  %257 = getelementptr { i64, ptr, ptr }, ptr %254, i32 0, i32 2
  store ptr %252, ptr %257, align 8
  br label %258

258:                                              ; preds = %250, %.noexc8
  %259 = phi ptr [ %254, %.noexc8 ], [ %249, %250 ]
  br label %260

260:                                              ; preds = %258
  br label %268

261:                                              ; preds = %227
  %262 = load ptr, ptr %159, align 8
  %263 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %261
  %264 = getelementptr i8, ptr %263, i64 8
  %265 = getelementptr { i64, ptr, ptr }, ptr %264, i32 0, i32 0
  store i64 8, ptr %265, align 8
  %266 = getelementptr { i64, ptr, ptr }, ptr %264, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %266, align 8
  %267 = getelementptr { i64, ptr, ptr }, ptr %264, i32 0, i32 2
  store ptr %262, ptr %267, align 8
  br label %268

268:                                              ; preds = %260, %.noexc10
  %269 = phi ptr [ %264, %.noexc10 ], [ %259, %260 ]
  br label %270

270:                                              ; preds = %268
  br label %271

271:                                              ; preds = %225, %270
  %272 = phi ptr [ %269, %270 ], [ %226, %225 ]
  br label %273

273:                                              ; preds = %271
  br label %274

274:                                              ; preds = %.noexc7, %273
  %275 = phi ptr [ %272, %273 ], [ %219, %.noexc7 ]
  br label %276

276:                                              ; preds = %274
  br label %277

277:                                              ; preds = %.noexc17, %276
  %278 = phi ptr [ %275, %276 ], [ %210, %.noexc17 ]
  br label %279

279:                                              ; preds = %277
  br label %280

280:                                              ; preds = %.noexc13, %279
  %281 = phi ptr [ %278, %279 ], [ %201, %.noexc13 ]
  br label %282

282:                                              ; preds = %280
  br label %283

283:                                              ; preds = %.noexc2, %282
  %284 = phi ptr [ %281, %282 ], [ %192, %.noexc2 ]
  br label %285

285:                                              ; preds = %283
  br label %286

286:                                              ; preds = %.noexc24, %285
  %287 = phi ptr [ %284, %285 ], [ %183, %.noexc24 ]
  br label %288

288:                                              ; preds = %286
  br label %289

289:                                              ; preds = %.noexc23, %288
  %290 = phi ptr [ %287, %288 ], [ %174, %.noexc23 ]
  br label %291

291:                                              ; preds = %289
  br label %292

292:                                              ; preds = %.noexc3, %291
  %293 = phi ptr [ %290, %291 ], [ %165, %.noexc3 ]
  br label %294

294:                                              ; preds = %292
  store ptr %293, ptr %5, align 8
  %295 = load ptr, ptr %5, align 8
  %296 = invoke double @"___unbox<!ts.number>"(ptr %295)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %294
  store double %296, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %297

297:                                              ; preds = %10, %153, %.noexc6
  %298 = load double, ptr %7, align 8
  ret double %298

endcatch.resume:                                  ; preds = %180, %171, %116, %26, %11, %241, %.caught, %207, %62, %35, %44, %198, %239, %94, %261, %71, %251, %216, %294, %53, %96, %162, %189, %17, %106
  %299 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %299
}

; Function Attrs: mustprogress
define double @finallyRethrow(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  store double %0, ptr %4, align 8
  %6 = load double, ptr %4, align 8
  %7 = invoke i32 @f(double %6)
          to label %8 unwind label %11

8:                                                ; preds = %1
  %9 = load i32, ptr @fin, align 4
  %10 = add i32 %9, 1
  store i32 %10, ptr @fin, align 4
  store double -1.000000e+00, ptr %5, align 8
  br label %160

11:                                               ; preds = %1
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr @fin, align 4
  %16 = add i32 %15, 1
  store i32 %16, ptr @fin, align 4
  invoke void @__cxa_rethrow()
          to label %162 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %11
  %17 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

18:                                               ; No predecessors!
  %19 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %18
  %20 = phi { ptr, i32 } [ %19, %18 ], [ %17, %endcatch.pad ]
  %21 = extractvalue { ptr, i32 } %20, 0
  %22 = call ptr @__cxa_begin_catch(ptr %21)
  %23 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.caught
  %24 = icmp eq ptr %23, @_ZTIi
  br i1 %24, label %25, label %32

25:                                               ; preds = %.noexc8
  %26 = load i32, ptr %22, align 4
  %27 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %25
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 0
  store i64 4, ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 2
  store i32 %26, ptr %31, align 4
  br label %155

32:                                               ; preds = %.noexc8
  %33 = icmp eq ptr %23, @_ZTIj
  br i1 %33, label %34, label %41

34:                                               ; preds = %32
  %35 = load i32, ptr %22, align 4
  %36 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %34
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 0
  store i64 4, ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %39, align 8
  %40 = getelementptr { i64, ptr, i32 }, ptr %37, i32 0, i32 2
  store i32 %35, ptr %40, align 4
  br label %152

41:                                               ; preds = %32
  %42 = icmp eq ptr %23, @_ZTId
  br i1 %42, label %43, label %50

43:                                               ; preds = %41
  %44 = load double, ptr %22, align 8
  %45 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %43
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr { i64, ptr, double }, ptr %46, i32 0, i32 0
  store i64 8, ptr %47, align 8
  %48 = getelementptr { i64, ptr, double }, ptr %46, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %48, align 8
  %49 = getelementptr { i64, ptr, double }, ptr %46, i32 0, i32 2
  store double %44, ptr %49, align 8
  br label %149

50:                                               ; preds = %41
  %51 = icmp eq ptr %23, @_ZTIf
  br i1 %51, label %52, label %59

52:                                               ; preds = %50
  %53 = load float, ptr %22, align 4
  %54 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = getelementptr { i64, ptr, float }, ptr %55, i32 0, i32 0
  store i64 4, ptr %56, align 8
  %57 = getelementptr { i64, ptr, float }, ptr %55, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %57, align 8
  %58 = getelementptr { i64, ptr, float }, ptr %55, i32 0, i32 2
  store float %53, ptr %58, align 4
  br label %146

59:                                               ; preds = %50
  %60 = icmp eq ptr %23, @_ZTIb
  br i1 %60, label %61, label %68

61:                                               ; preds = %59
  %62 = load i1, ptr %22, align 1
  %63 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %61
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr { i64, ptr, i1 }, ptr %64, i32 0, i32 0
  store i64 1, ptr %65, align 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %64, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %66, align 8
  %67 = getelementptr { i64, ptr, i1 }, ptr %64, i32 0, i32 2
  store i1 %62, ptr %67, align 1
  br label %143

68:                                               ; preds = %59
  %69 = icmp eq ptr %23, @_ZTIx
  br i1 %69, label %70, label %77

70:                                               ; preds = %68
  %71 = load i64, ptr %22, align 8
  %72 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %70
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr { i64, ptr, i64 }, ptr %73, i32 0, i32 0
  store i64 8, ptr %74, align 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %73, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %75, align 8
  %76 = getelementptr { i64, ptr, i64 }, ptr %73, i32 0, i32 2
  store i64 %71, ptr %76, align 8
  br label %140

77:                                               ; preds = %68
  %78 = icmp eq ptr %23, @_ZTIPc
  br i1 %78, label %79, label %86

79:                                               ; preds = %77
  %80 = load ptr, ptr %22, align 8
  %81 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %79
  %82 = getelementptr i8, ptr %81, i64 8
  %83 = getelementptr { i64, ptr, ptr }, ptr %82, i32 0, i32 0
  store i64 8, ptr %83, align 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %82, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %84, align 8
  %85 = getelementptr { i64, ptr, ptr }, ptr %82, i32 0, i32 2
  store ptr %80, ptr %85, align 8
  br label %137

86:                                               ; preds = %77
  %87 = icmp eq ptr %23, @_ZTIPv
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = load ptr, ptr %22, align 8
  br label %134

90:                                               ; preds = %86
  %91 = load ptr, ptr %23, align 8
  %92 = icmp eq ptr %91, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %92, label %93, label %124

93:                                               ; preds = %90
  %94 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %23, i32 0, i32 2
  %95 = load i32, ptr %94, align 4
  %96 = and i32 %95, 1073741824
  %97 = icmp ne i32 %96, 0
  br i1 %97, label %98, label %114

98:                                               ; preds = %93
  %99 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %23, i32 0, i32 4
  %100 = load ptr, ptr %99, align 8
  %101 = icmp ne ptr %100, null
  br i1 %101, label %102, label %104

102:                                              ; preds = %98
  invoke void %100(ptr %2, ptr %22)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %102
  %103 = load ptr, ptr %2, align 8
  br label %111

104:                                              ; preds = %98
  %105 = load ptr, ptr %22, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %.noexc6, %.noexc7
  %112 = phi ptr [ %107, %.noexc7 ], [ %103, %.noexc6 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %93
  %115 = load ptr, ptr %22, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc10
  %122 = phi ptr [ %117, %.noexc10 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %131

124:                                              ; preds = %90
  %125 = load ptr, ptr %22, align 8
  %126 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %124
  %127 = getelementptr i8, ptr %126, i64 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %127, i32 0, i32 0
  store i64 8, ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %127, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %129, align 8
  %130 = getelementptr { i64, ptr, ptr }, ptr %127, i32 0, i32 2
  store ptr %125, ptr %130, align 8
  br label %131

131:                                              ; preds = %123, %.noexc4
  %132 = phi ptr [ %127, %.noexc4 ], [ %122, %123 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %88, %133
  %135 = phi ptr [ %132, %133 ], [ %89, %88 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc3, %136
  %138 = phi ptr [ %135, %136 ], [ %82, %.noexc3 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc, %139
  %141 = phi ptr [ %138, %139 ], [ %73, %.noexc ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc12, %142
  %144 = phi ptr [ %141, %142 ], [ %64, %.noexc12 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc1, %145
  %147 = phi ptr [ %144, %145 ], [ %55, %.noexc1 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc9, %148
  %150 = phi ptr [ %147, %148 ], [ %46, %.noexc9 ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc5, %151
  %153 = phi ptr [ %150, %151 ], [ %37, %.noexc5 ]
  br label %154

154:                                              ; preds = %152
  br label %155

155:                                              ; preds = %.noexc2, %154
  %156 = phi ptr [ %153, %154 ], [ %28, %.noexc2 ]
  br label %157

157:                                              ; preds = %155
  store ptr %156, ptr %3, align 8
  %158 = load ptr, ptr %3, align 8
  %159 = invoke double @"___unbox<!ts.number>"(ptr %158)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %157
  store double %159, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %160

160:                                              ; preds = %8, %.noexc11
  %161 = load double, ptr %5, align 8
  ret double %161

162:                                              ; preds = %11
  unreachable

endcatch.resume:                                  ; preds = %61, %157, %114, %43, %.caught, %104, %102, %34, %124, %79, %25, %52, %70
  %163 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %163
}

; Function Attrs: mustprogress
define double @mismatch(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  store double %0, ptr %5, align 8
  %7 = load double, ptr %5, align 8
  %8 = invoke i32 @f(double %7)
          to label %9 unwind label %10

9:                                                ; preds = %1
  store double -1.000000e+00, ptr %6, align 8
  br label %164

10:                                               ; preds = %1
  %11 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %14 = icmp eq i32 %12, %13
  br i1 %14, label %18, label %15

15:                                               ; preds = %10
  %16 = extractvalue { ptr, i32 } %11, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  invoke void @__cxa_rethrow()
          to label %166 unwind label %endcatch.pad

18:                                               ; preds = %10
  %19 = extractvalue { ptr, i32 } %11, 0
  %20 = call ptr @__cxa_begin_catch(ptr %19)
  store ptr %20, ptr %3, align 8
  store double -2.000000e+00, ptr %6, align 8
  call void @__cxa_end_catch()
  br label %164

endcatch.pad:                                     ; preds = %15
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
  %27 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.caught
  %28 = icmp eq ptr %27, @_ZTIi
  br i1 %28, label %29, label %36

29:                                               ; preds = %.noexc5
  %30 = load i32, ptr %26, align 4
  %31 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %29
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %32, i32 0, i32 0
  store i64 4, ptr %33, align 8
  %34 = getelementptr { i64, ptr, i32 }, ptr %32, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %34, align 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %32, i32 0, i32 2
  store i32 %30, ptr %35, align 4
  br label %159

36:                                               ; preds = %.noexc5
  %37 = icmp eq ptr %27, @_ZTIj
  br i1 %37, label %38, label %45

38:                                               ; preds = %36
  %39 = load i32, ptr %26, align 4
  %40 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %38
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr { i64, ptr, i32 }, ptr %41, i32 0, i32 0
  store i64 4, ptr %42, align 8
  %43 = getelementptr { i64, ptr, i32 }, ptr %41, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %43, align 8
  %44 = getelementptr { i64, ptr, i32 }, ptr %41, i32 0, i32 2
  store i32 %39, ptr %44, align 4
  br label %156

45:                                               ; preds = %36
  %46 = icmp eq ptr %27, @_ZTId
  br i1 %46, label %47, label %54

47:                                               ; preds = %45
  %48 = load double, ptr %26, align 8
  %49 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %47
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr { i64, ptr, double }, ptr %50, i32 0, i32 0
  store i64 8, ptr %51, align 8
  %52 = getelementptr { i64, ptr, double }, ptr %50, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %52, align 8
  %53 = getelementptr { i64, ptr, double }, ptr %50, i32 0, i32 2
  store double %48, ptr %53, align 8
  br label %153

54:                                               ; preds = %45
  %55 = icmp eq ptr %27, @_ZTIf
  br i1 %55, label %56, label %63

56:                                               ; preds = %54
  %57 = load float, ptr %26, align 4
  %58 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %56
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr { i64, ptr, float }, ptr %59, i32 0, i32 0
  store i64 4, ptr %60, align 8
  %61 = getelementptr { i64, ptr, float }, ptr %59, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %61, align 8
  %62 = getelementptr { i64, ptr, float }, ptr %59, i32 0, i32 2
  store float %57, ptr %62, align 4
  br label %150

63:                                               ; preds = %54
  %64 = icmp eq ptr %27, @_ZTIb
  br i1 %64, label %65, label %72

65:                                               ; preds = %63
  %66 = load i1, ptr %26, align 1
  %67 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %65
  %68 = getelementptr i8, ptr %67, i64 8
  %69 = getelementptr { i64, ptr, i1 }, ptr %68, i32 0, i32 0
  store i64 1, ptr %69, align 8
  %70 = getelementptr { i64, ptr, i1 }, ptr %68, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %70, align 8
  %71 = getelementptr { i64, ptr, i1 }, ptr %68, i32 0, i32 2
  store i1 %66, ptr %71, align 1
  br label %147

72:                                               ; preds = %63
  %73 = icmp eq ptr %27, @_ZTIx
  br i1 %73, label %74, label %81

74:                                               ; preds = %72
  %75 = load i64, ptr %26, align 8
  %76 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %74
  %77 = getelementptr i8, ptr %76, i64 8
  %78 = getelementptr { i64, ptr, i64 }, ptr %77, i32 0, i32 0
  store i64 8, ptr %78, align 8
  %79 = getelementptr { i64, ptr, i64 }, ptr %77, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %79, align 8
  %80 = getelementptr { i64, ptr, i64 }, ptr %77, i32 0, i32 2
  store i64 %75, ptr %80, align 8
  br label %144

81:                                               ; preds = %72
  %82 = icmp eq ptr %27, @_ZTIPc
  br i1 %82, label %83, label %90

83:                                               ; preds = %81
  %84 = load ptr, ptr %26, align 8
  %85 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %83
  %86 = getelementptr i8, ptr %85, i64 8
  %87 = getelementptr { i64, ptr, ptr }, ptr %86, i32 0, i32 0
  store i64 8, ptr %87, align 8
  %88 = getelementptr { i64, ptr, ptr }, ptr %86, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %88, align 8
  %89 = getelementptr { i64, ptr, ptr }, ptr %86, i32 0, i32 2
  store ptr %84, ptr %89, align 8
  br label %141

90:                                               ; preds = %81
  %91 = icmp eq ptr %27, @_ZTIPv
  br i1 %91, label %92, label %94

92:                                               ; preds = %90
  %93 = load ptr, ptr %26, align 8
  br label %138

94:                                               ; preds = %90
  %95 = load ptr, ptr %27, align 8
  %96 = icmp eq ptr %95, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %96, label %97, label %128

97:                                               ; preds = %94
  %98 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %27, i32 0, i32 2
  %99 = load i32, ptr %98, align 4
  %100 = and i32 %99, 1073741824
  %101 = icmp ne i32 %100, 0
  br i1 %101, label %102, label %118

102:                                              ; preds = %97
  %103 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %27, i32 0, i32 4
  %104 = load ptr, ptr %103, align 8
  %105 = icmp ne ptr %104, null
  br i1 %105, label %106, label %108

106:                                              ; preds = %102
  invoke void %104(ptr %2, ptr %26)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %106
  %107 = load ptr, ptr %2, align 8
  br label %115

108:                                              ; preds = %102
  %109 = load ptr, ptr %26, align 8
  %110 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %108
  %111 = getelementptr i8, ptr %110, i64 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 0
  store i64 8, ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 2
  store ptr %109, ptr %114, align 8
  br label %115

115:                                              ; preds = %.noexc7, %.noexc12
  %116 = phi ptr [ %111, %.noexc12 ], [ %107, %.noexc7 ]
  br label %117

117:                                              ; preds = %115
  br label %125

118:                                              ; preds = %97
  %119 = load ptr, ptr %26, align 8
  %120 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %118
  %121 = getelementptr i8, ptr %120, i64 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 0
  store i64 8, ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %123, align 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 2
  store ptr %119, ptr %124, align 8
  br label %125

125:                                              ; preds = %117, %.noexc9
  %126 = phi ptr [ %121, %.noexc9 ], [ %116, %117 ]
  br label %127

127:                                              ; preds = %125
  br label %135

128:                                              ; preds = %94
  %129 = load ptr, ptr %26, align 8
  %130 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %128
  %131 = getelementptr i8, ptr %130, i64 8
  %132 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 0
  store i64 8, ptr %132, align 8
  %133 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %133, align 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 2
  store ptr %129, ptr %134, align 8
  br label %135

135:                                              ; preds = %127, %.noexc3
  %136 = phi ptr [ %131, %.noexc3 ], [ %126, %127 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %92, %137
  %139 = phi ptr [ %136, %137 ], [ %93, %92 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc11, %140
  %142 = phi ptr [ %139, %140 ], [ %86, %.noexc11 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc1, %143
  %145 = phi ptr [ %142, %143 ], [ %77, %.noexc1 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc6, %146
  %148 = phi ptr [ %145, %146 ], [ %68, %.noexc6 ]
  br label %149

149:                                              ; preds = %147
  br label %150

150:                                              ; preds = %.noexc10, %149
  %151 = phi ptr [ %148, %149 ], [ %59, %.noexc10 ]
  br label %152

152:                                              ; preds = %150
  br label %153

153:                                              ; preds = %.noexc4, %152
  %154 = phi ptr [ %151, %152 ], [ %50, %.noexc4 ]
  br label %155

155:                                              ; preds = %153
  br label %156

156:                                              ; preds = %.noexc8, %155
  %157 = phi ptr [ %154, %155 ], [ %41, %.noexc8 ]
  br label %158

158:                                              ; preds = %156
  br label %159

159:                                              ; preds = %.noexc, %158
  %160 = phi ptr [ %157, %158 ], [ %32, %.noexc ]
  br label %161

161:                                              ; preds = %159
  store ptr %160, ptr %4, align 8
  %162 = load ptr, ptr %4, align 8
  %163 = invoke double @"___unbox<!ts.number>"(ptr %162)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %161
  store double %163, ptr %6, align 8
  call void @__cxa_end_catch()
  br label %164

164:                                              ; preds = %9, %18, %.noexc2
  %165 = load double, ptr %6, align 8
  ret double %165

166:                                              ; preds = %15
  unreachable

endcatch.resume:                                  ; preds = %108, %83, %56, %118, %38, %106, %65, %.caught, %47, %128, %161, %74, %29
  %167 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %167
}

; Function Attrs: mustprogress
define i32 @mismatchOut(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca i32, align 4
  store double %0, ptr %3, align 8
  %5 = load double, ptr %3, align 8
  %6 = invoke i32 @f(double %5)
          to label %7 unwind label %8

7:                                                ; preds = %1
  store i32 -1, ptr %4, align 4
  br label %19

8:                                                ; preds = %1
  %9 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %10 = extractvalue { ptr, i32 } %9, 1
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %16, label %13

13:                                               ; preds = %8
  %14 = extractvalue { ptr, i32 } %9, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %13
  br label %21

16:                                               ; preds = %8
  %17 = extractvalue { ptr, i32 } %9, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  store ptr %18, ptr %2, align 8
  store i32 -2, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %19

19:                                               ; preds = %7, %16
  %20 = load i32, ptr %4, align 4
  ret i32 %20

21:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %13
  %22 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %22
}

; Function Attrs: mustprogress
define double @callInCatchFinally(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  store double %0, ptr %6, align 8
  %8 = load double, ptr %6, align 8
  %9 = invoke i32 @f(double %8)
          to label %10 unwind label %11

10:                                               ; preds = %1
  br label %162

11:                                               ; preds = %1
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc13
  %18 = load i32, ptr %14, align 4
  %19 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %17
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 0
  store i64 4, ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 2
  store i32 %18, ptr %23, align 4
  br label %147

24:                                               ; preds = %.noexc13
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %26
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 0
  store i64 4, ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 2
  store i32 %27, ptr %32, align 4
  br label %144

33:                                               ; preds = %24
  %34 = icmp eq ptr %15, @_ZTId
  br i1 %34, label %35, label %42

35:                                               ; preds = %33
  %36 = load double, ptr %14, align 8
  %37 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %35
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 0
  store i64 8, ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 2
  store double %36, ptr %41, align 8
  br label %141

42:                                               ; preds = %33
  %43 = icmp eq ptr %15, @_ZTIf
  br i1 %43, label %44, label %51

44:                                               ; preds = %42
  %45 = load float, ptr %14, align 4
  %46 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %44
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 0
  store i64 4, ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 2
  store float %45, ptr %50, align 4
  br label %138

51:                                               ; preds = %42
  %52 = icmp eq ptr %15, @_ZTIb
  br i1 %52, label %53, label %60

53:                                               ; preds = %51
  %54 = load i1, ptr %14, align 1
  %55 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %53
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 0
  store i64 1, ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 2
  store i1 %54, ptr %59, align 1
  br label %135

60:                                               ; preds = %51
  %61 = icmp eq ptr %15, @_ZTIx
  br i1 %61, label %62, label %69

62:                                               ; preds = %60
  %63 = load i64, ptr %14, align 8
  %64 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %62
  %65 = getelementptr i8, ptr %64, i64 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 0
  store i64 8, ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 2
  store i64 %63, ptr %68, align 8
  br label %132

69:                                               ; preds = %60
  %70 = icmp eq ptr %15, @_ZTIPc
  br i1 %70, label %71, label %78

71:                                               ; preds = %69
  %72 = load ptr, ptr %14, align 8
  %73 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %71
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 0
  store i64 8, ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 2
  store ptr %72, ptr %77, align 8
  br label %129

78:                                               ; preds = %69
  %79 = icmp eq ptr %15, @_ZTIPv
  br i1 %79, label %80, label %82

80:                                               ; preds = %78
  %81 = load ptr, ptr %14, align 8
  br label %126

82:                                               ; preds = %78
  %83 = load ptr, ptr %15, align 8
  %84 = icmp eq ptr %83, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %84, label %85, label %116

85:                                               ; preds = %82
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 2
  %87 = load i32, ptr %86, align 4
  %88 = and i32 %87, 1073741824
  %89 = icmp ne i32 %88, 0
  br i1 %89, label %90, label %106

90:                                               ; preds = %85
  %91 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 4
  %92 = load ptr, ptr %91, align 8
  %93 = icmp ne ptr %92, null
  br i1 %93, label %94, label %96

94:                                               ; preds = %90
  invoke void %92(ptr %3, ptr %14)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %94
  %95 = load ptr, ptr %3, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc, %.noexc12
  %104 = phi ptr [ %99, %.noexc12 ], [ %95, %.noexc ]
  br label %105

105:                                              ; preds = %103
  br label %113

106:                                              ; preds = %85
  %107 = load ptr, ptr %14, align 8
  %108 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %106
  %109 = getelementptr i8, ptr %108, i64 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 0
  store i64 8, ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 2
  store ptr %107, ptr %112, align 8
  br label %113

113:                                              ; preds = %105, %.noexc17
  %114 = phi ptr [ %109, %.noexc17 ], [ %104, %105 ]
  br label %115

115:                                              ; preds = %113
  br label %123

116:                                              ; preds = %82
  %117 = load ptr, ptr %14, align 8
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc21
  %124 = phi ptr [ %119, %.noexc21 ], [ %114, %115 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %80, %125
  %127 = phi ptr [ %124, %125 ], [ %81, %80 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc10, %128
  %130 = phi ptr [ %127, %128 ], [ %74, %.noexc10 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc18, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc18 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc11, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc11 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc9, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc9 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc23, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc23 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc14, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc14 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc3, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc3 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %4, align 8
  %150 = load double, ptr %6, align 8
  %151 = fadd double %150, 3.000000e+00
  %152 = invoke i32 @f(double %151)
          to label %153 unwind label %endcatch.pad2

153:                                              ; preds = %149
  call void @__cxa_end_catch()
  br label %162

endcatch.pad2:                                    ; preds = %149
  %154 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

155:                                              ; No predecessors!
  %156 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %155
  %157 = phi { ptr, i32 } [ %156, %155 ], [ %154, %endcatch.pad2 ]
  %158 = extractvalue { ptr, i32 } %157, 0
  %159 = call ptr @__cxa_begin_catch(ptr %158)
  %160 = load i32, ptr @fin, align 4
  %161 = add i32 %160, 1
  store i32 %161, ptr @fin, align 4
  invoke void @__cxa_rethrow()
          to label %310 unwind label %endcatch.pad

162:                                              ; preds = %10, %153
  %163 = load i32, ptr @fin, align 4
  %164 = add i32 %163, 1
  store i32 %164, ptr @fin, align 4
  store double -1.000000e+00, ptr %7, align 8
  br label %308

endcatch.pad:                                     ; preds = %.caught1
  %165 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

166:                                              ; No predecessors!
  %167 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %166
  %168 = phi { ptr, i32 } [ %167, %166 ], [ %165, %endcatch.pad ]
  %169 = extractvalue { ptr, i32 } %168, 0
  %170 = call ptr @__cxa_begin_catch(ptr %169)
  %171 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %.caught
  %172 = icmp eq ptr %171, @_ZTIi
  br i1 %172, label %173, label %180

173:                                              ; preds = %.noexc20
  %174 = load i32, ptr %170, align 4
  %175 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %173
  %176 = getelementptr i8, ptr %175, i64 8
  %177 = getelementptr { i64, ptr, i32 }, ptr %176, i32 0, i32 0
  store i64 4, ptr %177, align 8
  %178 = getelementptr { i64, ptr, i32 }, ptr %176, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %178, align 8
  %179 = getelementptr { i64, ptr, i32 }, ptr %176, i32 0, i32 2
  store i32 %174, ptr %179, align 4
  br label %303

180:                                              ; preds = %.noexc20
  %181 = icmp eq ptr %171, @_ZTIj
  br i1 %181, label %182, label %189

182:                                              ; preds = %180
  %183 = load i32, ptr %170, align 4
  %184 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %182
  %185 = getelementptr i8, ptr %184, i64 8
  %186 = getelementptr { i64, ptr, i32 }, ptr %185, i32 0, i32 0
  store i64 4, ptr %186, align 8
  %187 = getelementptr { i64, ptr, i32 }, ptr %185, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %187, align 8
  %188 = getelementptr { i64, ptr, i32 }, ptr %185, i32 0, i32 2
  store i32 %183, ptr %188, align 4
  br label %300

189:                                              ; preds = %180
  %190 = icmp eq ptr %171, @_ZTId
  br i1 %190, label %191, label %198

191:                                              ; preds = %189
  %192 = load double, ptr %170, align 8
  %193 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %191
  %194 = getelementptr i8, ptr %193, i64 8
  %195 = getelementptr { i64, ptr, double }, ptr %194, i32 0, i32 0
  store i64 8, ptr %195, align 8
  %196 = getelementptr { i64, ptr, double }, ptr %194, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %196, align 8
  %197 = getelementptr { i64, ptr, double }, ptr %194, i32 0, i32 2
  store double %192, ptr %197, align 8
  br label %297

198:                                              ; preds = %189
  %199 = icmp eq ptr %171, @_ZTIf
  br i1 %199, label %200, label %207

200:                                              ; preds = %198
  %201 = load float, ptr %170, align 4
  %202 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %200
  %203 = getelementptr i8, ptr %202, i64 8
  %204 = getelementptr { i64, ptr, float }, ptr %203, i32 0, i32 0
  store i64 4, ptr %204, align 8
  %205 = getelementptr { i64, ptr, float }, ptr %203, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %205, align 8
  %206 = getelementptr { i64, ptr, float }, ptr %203, i32 0, i32 2
  store float %201, ptr %206, align 4
  br label %294

207:                                              ; preds = %198
  %208 = icmp eq ptr %171, @_ZTIb
  br i1 %208, label %209, label %216

209:                                              ; preds = %207
  %210 = load i1, ptr %170, align 1
  %211 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %209
  %212 = getelementptr i8, ptr %211, i64 8
  %213 = getelementptr { i64, ptr, i1 }, ptr %212, i32 0, i32 0
  store i64 1, ptr %213, align 8
  %214 = getelementptr { i64, ptr, i1 }, ptr %212, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %214, align 8
  %215 = getelementptr { i64, ptr, i1 }, ptr %212, i32 0, i32 2
  store i1 %210, ptr %215, align 1
  br label %291

216:                                              ; preds = %207
  %217 = icmp eq ptr %171, @_ZTIx
  br i1 %217, label %218, label %225

218:                                              ; preds = %216
  %219 = load i64, ptr %170, align 8
  %220 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %218
  %221 = getelementptr i8, ptr %220, i64 8
  %222 = getelementptr { i64, ptr, i64 }, ptr %221, i32 0, i32 0
  store i64 8, ptr %222, align 8
  %223 = getelementptr { i64, ptr, i64 }, ptr %221, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %223, align 8
  %224 = getelementptr { i64, ptr, i64 }, ptr %221, i32 0, i32 2
  store i64 %219, ptr %224, align 8
  br label %288

225:                                              ; preds = %216
  %226 = icmp eq ptr %171, @_ZTIPc
  br i1 %226, label %227, label %234

227:                                              ; preds = %225
  %228 = load ptr, ptr %170, align 8
  %229 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %227
  %230 = getelementptr i8, ptr %229, i64 8
  %231 = getelementptr { i64, ptr, ptr }, ptr %230, i32 0, i32 0
  store i64 8, ptr %231, align 8
  %232 = getelementptr { i64, ptr, ptr }, ptr %230, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %232, align 8
  %233 = getelementptr { i64, ptr, ptr }, ptr %230, i32 0, i32 2
  store ptr %228, ptr %233, align 8
  br label %285

234:                                              ; preds = %225
  %235 = icmp eq ptr %171, @_ZTIPv
  br i1 %235, label %236, label %238

236:                                              ; preds = %234
  %237 = load ptr, ptr %170, align 8
  br label %282

238:                                              ; preds = %234
  %239 = load ptr, ptr %171, align 8
  %240 = icmp eq ptr %239, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %240, label %241, label %272

241:                                              ; preds = %238
  %242 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %171, i32 0, i32 2
  %243 = load i32, ptr %242, align 4
  %244 = and i32 %243, 1073741824
  %245 = icmp ne i32 %244, 0
  br i1 %245, label %246, label %262

246:                                              ; preds = %241
  %247 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %171, i32 0, i32 4
  %248 = load ptr, ptr %247, align 8
  %249 = icmp ne ptr %248, null
  br i1 %249, label %250, label %252

250:                                              ; preds = %246
  invoke void %248(ptr %2, ptr %170)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %250
  %251 = load ptr, ptr %2, align 8
  br label %259

252:                                              ; preds = %246
  %253 = load ptr, ptr %170, align 8
  %254 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %252
  %255 = getelementptr i8, ptr %254, i64 8
  %256 = getelementptr { i64, ptr, ptr }, ptr %255, i32 0, i32 0
  store i64 8, ptr %256, align 8
  %257 = getelementptr { i64, ptr, ptr }, ptr %255, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %257, align 8
  %258 = getelementptr { i64, ptr, ptr }, ptr %255, i32 0, i32 2
  store ptr %253, ptr %258, align 8
  br label %259

259:                                              ; preds = %.noexc24, %.noexc6
  %260 = phi ptr [ %255, %.noexc6 ], [ %251, %.noexc24 ]
  br label %261

261:                                              ; preds = %259
  br label %269

262:                                              ; preds = %241
  %263 = load ptr, ptr %170, align 8
  %264 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %262
  %265 = getelementptr i8, ptr %264, i64 8
  %266 = getelementptr { i64, ptr, ptr }, ptr %265, i32 0, i32 0
  store i64 8, ptr %266, align 8
  %267 = getelementptr { i64, ptr, ptr }, ptr %265, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %267, align 8
  %268 = getelementptr { i64, ptr, ptr }, ptr %265, i32 0, i32 2
  store ptr %263, ptr %268, align 8
  br label %269

269:                                              ; preds = %261, %.noexc4
  %270 = phi ptr [ %265, %.noexc4 ], [ %260, %261 ]
  br label %271

271:                                              ; preds = %269
  br label %279

272:                                              ; preds = %238
  %273 = load ptr, ptr %170, align 8
  %274 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %272
  %275 = getelementptr i8, ptr %274, i64 8
  %276 = getelementptr { i64, ptr, ptr }, ptr %275, i32 0, i32 0
  store i64 8, ptr %276, align 8
  %277 = getelementptr { i64, ptr, ptr }, ptr %275, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %277, align 8
  %278 = getelementptr { i64, ptr, ptr }, ptr %275, i32 0, i32 2
  store ptr %273, ptr %278, align 8
  br label %279

279:                                              ; preds = %271, %.noexc22
  %280 = phi ptr [ %275, %.noexc22 ], [ %270, %271 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %236, %281
  %283 = phi ptr [ %280, %281 ], [ %237, %236 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc8, %284
  %286 = phi ptr [ %283, %284 ], [ %230, %.noexc8 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc15, %287
  %289 = phi ptr [ %286, %287 ], [ %221, %.noexc15 ]
  br label %290

290:                                              ; preds = %288
  br label %291

291:                                              ; preds = %.noexc7, %290
  %292 = phi ptr [ %289, %290 ], [ %212, %.noexc7 ]
  br label %293

293:                                              ; preds = %291
  br label %294

294:                                              ; preds = %.noexc16, %293
  %295 = phi ptr [ %292, %293 ], [ %203, %.noexc16 ]
  br label %296

296:                                              ; preds = %294
  br label %297

297:                                              ; preds = %.noexc26, %296
  %298 = phi ptr [ %295, %296 ], [ %194, %.noexc26 ]
  br label %299

299:                                              ; preds = %297
  br label %300

300:                                              ; preds = %.noexc25, %299
  %301 = phi ptr [ %298, %299 ], [ %185, %.noexc25 ]
  br label %302

302:                                              ; preds = %300
  br label %303

303:                                              ; preds = %.noexc5, %302
  %304 = phi ptr [ %301, %302 ], [ %176, %.noexc5 ]
  br label %305

305:                                              ; preds = %303
  store ptr %304, ptr %5, align 8
  %306 = load ptr, ptr %5, align 8
  %307 = invoke double @"___unbox<!ts.number>"(ptr %306)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %305
  store double %307, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %308

308:                                              ; preds = %162, %.noexc19
  %309 = load double, ptr %7, align 8
  ret double %309

310:                                              ; preds = %.caught1
  unreachable

endcatch.resume:                                  ; preds = %191, %182, %250, %35, %272, %116, %.caught, %305, %62, %106, %200, %218, %26, %11, %96, %53, %71, %44, %227, %209, %252, %173, %262, %17, %94
  %311 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %311
}

; Function Attrs: mustprogress
define i32 @callInCatchFinallyOut(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca i32, align 4
  store double %0, ptr %4, align 8
  %6 = load double, ptr %4, align 8
  %7 = invoke i32 @f(double %6)
          to label %8 unwind label %9

8:                                                ; preds = %1
  br label %158

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc11
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc11
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %24
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
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %33
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %42
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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %51
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %60
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %69
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %92
  %93 = load ptr, ptr %2, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc9, %.noexc4
  %102 = phi ptr [ %97, %.noexc4 ], [ %93, %.noexc9 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc1
  %112 = phi ptr [ %107, %.noexc1 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc6
  %122 = phi ptr [ %117, %.noexc6 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc5, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc5 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc10, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc10 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc2, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc2 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc7, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc7 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc3, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc3 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc8, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc8 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %3, align 8
  %148 = load double, ptr %4, align 8
  %149 = fadd double %148, 4.000000e+00
  %150 = invoke i32 @f(double %149)
          to label %151 unwind label %endcatch.pad

151:                                              ; preds = %147
  call void @__cxa_end_catch()
  br label %158

endcatch.pad:                                     ; preds = %147
  %152 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

153:                                              ; No predecessors!
  %154 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %153
  %155 = phi { ptr, i32 } [ %154, %153 ], [ %152, %endcatch.pad ]
  %156 = load i32, ptr @fin, align 4
  %157 = add i32 %156, 1
  store i32 %157, ptr @fin, align 4
  resume { ptr, i32 } %155

158:                                              ; preds = %8, %151
  %159 = load i32, ptr @fin, align 4
  %160 = add i32 %159, 1
  store i32 %160, ptr @fin, align 4
  store i32 -1, ptr %5, align 4
  %161 = load i32, ptr %5, align 4
  ret i32 %161

endcatch.resume:                                  ; preds = %9, %60, %92, %24, %42, %114, %69, %94, %33, %51, %104, %15
  %162 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %162
}

; Function Attrs: mustprogress
define double @nestedInCatch(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca double, align 8
  %9 = alloca double, align 8
  store double %0, ptr %8, align 8
  %10 = load double, ptr %8, align 8
  %11 = invoke i32 @f(double %10)
          to label %12 unwind label %13

12:                                               ; preds = %1
  br label %442

13:                                               ; preds = %1
  %14 = landingpad { ptr, i32 }
          catch ptr null
  %15 = extractvalue { ptr, i32 } %14, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %13
  %18 = icmp eq ptr %17, @_ZTIi
  br i1 %18, label %19, label %26

19:                                               ; preds = %.noexc
  %20 = load i32, ptr %16, align 4
  %21 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %19
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 0
  store i64 4, ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 2
  store i32 %20, ptr %25, align 4
  br label %149

26:                                               ; preds = %.noexc
  %27 = icmp eq ptr %17, @_ZTIj
  br i1 %27, label %28, label %35

28:                                               ; preds = %26
  %29 = load i32, ptr %16, align 4
  %30 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %28
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 0
  store i64 4, ptr %32, align 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %33, align 8
  %34 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 2
  store i32 %29, ptr %34, align 4
  br label %146

35:                                               ; preds = %26
  %36 = icmp eq ptr %17, @_ZTId
  br i1 %36, label %37, label %44

37:                                               ; preds = %35
  %38 = load double, ptr %16, align 8
  %39 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %37
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 0
  store i64 8, ptr %41, align 8
  %42 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %42, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 2
  store double %38, ptr %43, align 8
  br label %143

44:                                               ; preds = %35
  %45 = icmp eq ptr %17, @_ZTIf
  br i1 %45, label %46, label %53

46:                                               ; preds = %44
  %47 = load float, ptr %16, align 4
  %48 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %46
  %49 = getelementptr i8, ptr %48, i64 8
  %50 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 0
  store i64 4, ptr %50, align 8
  %51 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %51, align 8
  %52 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 2
  store float %47, ptr %52, align 4
  br label %140

53:                                               ; preds = %44
  %54 = icmp eq ptr %17, @_ZTIb
  br i1 %54, label %55, label %62

55:                                               ; preds = %53
  %56 = load i1, ptr %16, align 1
  %57 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %55
  %58 = getelementptr i8, ptr %57, i64 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 0
  store i64 1, ptr %59, align 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %60, align 8
  %61 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 2
  store i1 %56, ptr %61, align 1
  br label %137

62:                                               ; preds = %53
  %63 = icmp eq ptr %17, @_ZTIx
  br i1 %63, label %64, label %71

64:                                               ; preds = %62
  %65 = load i64, ptr %16, align 8
  %66 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %64
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 0
  store i64 8, ptr %68, align 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %69, align 8
  %70 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 2
  store i64 %65, ptr %70, align 8
  br label %134

71:                                               ; preds = %62
  %72 = icmp eq ptr %17, @_ZTIPc
  br i1 %72, label %73, label %80

73:                                               ; preds = %71
  %74 = load ptr, ptr %16, align 8
  %75 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %73
  %76 = getelementptr i8, ptr %75, i64 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 0
  store i64 8, ptr %77, align 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %78, align 8
  %79 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 2
  store ptr %74, ptr %79, align 8
  br label %131

80:                                               ; preds = %71
  %81 = icmp eq ptr %17, @_ZTIPv
  br i1 %81, label %82, label %84

82:                                               ; preds = %80
  %83 = load ptr, ptr %16, align 8
  br label %128

84:                                               ; preds = %80
  %85 = load ptr, ptr %17, align 8
  %86 = icmp eq ptr %85, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %86, label %87, label %118

87:                                               ; preds = %84
  %88 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %17, i32 0, i32 2
  %89 = load i32, ptr %88, align 4
  %90 = and i32 %89, 1073741824
  %91 = icmp ne i32 %90, 0
  br i1 %91, label %92, label %108

92:                                               ; preds = %87
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %17, i32 0, i32 4
  %94 = load ptr, ptr %93, align 8
  %95 = icmp ne ptr %94, null
  br i1 %95, label %96, label %98

96:                                               ; preds = %92
  invoke void %94(ptr %5, ptr %16)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %96
  %97 = load ptr, ptr %5, align 8
  br label %105

98:                                               ; preds = %92
  %99 = load ptr, ptr %16, align 8
  %100 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %98
  %101 = getelementptr i8, ptr %100, i64 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 0
  store i64 8, ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %103, align 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 2
  store ptr %99, ptr %104, align 8
  br label %105

105:                                              ; preds = %.noexc13, %.noexc4
  %106 = phi ptr [ %101, %.noexc4 ], [ %97, %.noexc13 ]
  br label %107

107:                                              ; preds = %105
  br label %115

108:                                              ; preds = %87
  %109 = load ptr, ptr %16, align 8
  %110 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %108
  %111 = getelementptr i8, ptr %110, i64 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 0
  store i64 8, ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 2
  store ptr %109, ptr %114, align 8
  br label %115

115:                                              ; preds = %107, %.noexc12
  %116 = phi ptr [ %111, %.noexc12 ], [ %106, %107 ]
  br label %117

117:                                              ; preds = %115
  br label %125

118:                                              ; preds = %84
  %119 = load ptr, ptr %16, align 8
  %120 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %118
  %121 = getelementptr i8, ptr %120, i64 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 0
  store i64 8, ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %123, align 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 2
  store ptr %119, ptr %124, align 8
  br label %125

125:                                              ; preds = %117, %.noexc25
  %126 = phi ptr [ %121, %.noexc25 ], [ %116, %117 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %82, %127
  %129 = phi ptr [ %126, %127 ], [ %83, %82 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc1, %130
  %132 = phi ptr [ %129, %130 ], [ %76, %.noexc1 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc19, %133
  %135 = phi ptr [ %132, %133 ], [ %67, %.noexc19 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc15, %136
  %138 = phi ptr [ %135, %136 ], [ %58, %.noexc15 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc17, %139
  %141 = phi ptr [ %138, %139 ], [ %49, %.noexc17 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc6, %142
  %144 = phi ptr [ %141, %142 ], [ %40, %.noexc6 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc20, %145
  %147 = phi ptr [ %144, %145 ], [ %31, %.noexc20 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc10, %148
  %150 = phi ptr [ %147, %148 ], [ %22, %.noexc10 ]
  br label %151

151:                                              ; preds = %149
  store ptr %150, ptr %6, align 8
  %152 = load double, ptr %8, align 8
  %153 = invoke i32 @f(double %152)
          to label %154 unwind label %155

154:                                              ; preds = %151
  br label %298

155:                                              ; preds = %151
  %156 = landingpad { ptr, i32 }
          catch ptr null
  %157 = extractvalue { ptr, i32 } %156, 0
  %158 = call ptr @__cxa_begin_catch(ptr %157)
  %159 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume2

.noexc8:                                          ; preds = %155
  %160 = icmp eq ptr %159, @_ZTIi
  br i1 %160, label %161, label %168

161:                                              ; preds = %.noexc8
  %162 = load i32, ptr %158, align 4
  %163 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume2

.noexc5:                                          ; preds = %161
  %164 = getelementptr i8, ptr %163, i64 8
  %165 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 0
  store i64 4, ptr %165, align 8
  %166 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %166, align 8
  %167 = getelementptr { i64, ptr, i32 }, ptr %164, i32 0, i32 2
  store i32 %162, ptr %167, align 4
  br label %291

168:                                              ; preds = %.noexc8
  %169 = icmp eq ptr %159, @_ZTIj
  br i1 %169, label %170, label %177

170:                                              ; preds = %168
  %171 = load i32, ptr %158, align 4
  %172 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume2

.noexc23:                                         ; preds = %170
  %173 = getelementptr i8, ptr %172, i64 8
  %174 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 0
  store i64 4, ptr %174, align 8
  %175 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %175, align 8
  %176 = getelementptr { i64, ptr, i32 }, ptr %173, i32 0, i32 2
  store i32 %171, ptr %176, align 4
  br label %288

177:                                              ; preds = %168
  %178 = icmp eq ptr %159, @_ZTId
  br i1 %178, label %179, label %186

179:                                              ; preds = %177
  %180 = load double, ptr %158, align 8
  %181 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume2

.noexc26:                                         ; preds = %179
  %182 = getelementptr i8, ptr %181, i64 8
  %183 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 0
  store i64 8, ptr %183, align 8
  %184 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %184, align 8
  %185 = getelementptr { i64, ptr, double }, ptr %182, i32 0, i32 2
  store double %180, ptr %185, align 8
  br label %285

186:                                              ; preds = %177
  %187 = icmp eq ptr %159, @_ZTIf
  br i1 %187, label %188, label %195

188:                                              ; preds = %186
  %189 = load float, ptr %158, align 4
  %190 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume2

.noexc27:                                         ; preds = %188
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 0
  store i64 4, ptr %192, align 8
  %193 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %193, align 8
  %194 = getelementptr { i64, ptr, float }, ptr %191, i32 0, i32 2
  store float %189, ptr %194, align 4
  br label %282

195:                                              ; preds = %186
  %196 = icmp eq ptr %159, @_ZTIb
  br i1 %196, label %197, label %204

197:                                              ; preds = %195
  %198 = load i1, ptr %158, align 1
  %199 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume2

.noexc3:                                          ; preds = %197
  %200 = getelementptr i8, ptr %199, i64 8
  %201 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 0
  store i64 1, ptr %201, align 8
  %202 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %202, align 8
  %203 = getelementptr { i64, ptr, i1 }, ptr %200, i32 0, i32 2
  store i1 %198, ptr %203, align 1
  br label %279

204:                                              ; preds = %195
  %205 = icmp eq ptr %159, @_ZTIx
  br i1 %205, label %206, label %213

206:                                              ; preds = %204
  %207 = load i64, ptr %158, align 8
  %208 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc28 unwind label %endcatch.resume2

.noexc28:                                         ; preds = %206
  %209 = getelementptr i8, ptr %208, i64 8
  %210 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 0
  store i64 8, ptr %210, align 8
  %211 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %211, align 8
  %212 = getelementptr { i64, ptr, i64 }, ptr %209, i32 0, i32 2
  store i64 %207, ptr %212, align 8
  br label %276

213:                                              ; preds = %204
  %214 = icmp eq ptr %159, @_ZTIPc
  br i1 %214, label %215, label %222

215:                                              ; preds = %213
  %216 = load ptr, ptr %158, align 8
  %217 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc36 unwind label %endcatch.resume2

.noexc36:                                         ; preds = %215
  %218 = getelementptr i8, ptr %217, i64 8
  %219 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 0
  store i64 8, ptr %219, align 8
  %220 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %220, align 8
  %221 = getelementptr { i64, ptr, ptr }, ptr %218, i32 0, i32 2
  store ptr %216, ptr %221, align 8
  br label %273

222:                                              ; preds = %213
  %223 = icmp eq ptr %159, @_ZTIPv
  br i1 %223, label %224, label %226

224:                                              ; preds = %222
  %225 = load ptr, ptr %158, align 8
  br label %270

226:                                              ; preds = %222
  %227 = load ptr, ptr %159, align 8
  %228 = icmp eq ptr %227, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %228, label %229, label %260

229:                                              ; preds = %226
  %230 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %159, i32 0, i32 2
  %231 = load i32, ptr %230, align 4
  %232 = and i32 %231, 1073741824
  %233 = icmp ne i32 %232, 0
  br i1 %233, label %234, label %250

234:                                              ; preds = %229
  %235 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %159, i32 0, i32 4
  %236 = load ptr, ptr %235, align 8
  %237 = icmp ne ptr %236, null
  br i1 %237, label %238, label %240

238:                                              ; preds = %234
  invoke void %236(ptr %3, ptr %158)
          to label %.noexc29 unwind label %endcatch.resume2

.noexc29:                                         ; preds = %238
  %239 = load ptr, ptr %3, align 8
  br label %247

240:                                              ; preds = %234
  %241 = load ptr, ptr %158, align 8
  %242 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc35 unwind label %endcatch.resume2

.noexc35:                                         ; preds = %240
  %243 = getelementptr i8, ptr %242, i64 8
  %244 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 0
  store i64 8, ptr %244, align 8
  %245 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %245, align 8
  %246 = getelementptr { i64, ptr, ptr }, ptr %243, i32 0, i32 2
  store ptr %241, ptr %246, align 8
  br label %247

247:                                              ; preds = %.noexc29, %.noexc35
  %248 = phi ptr [ %243, %.noexc35 ], [ %239, %.noexc29 ]
  br label %249

249:                                              ; preds = %247
  br label %257

250:                                              ; preds = %229
  %251 = load ptr, ptr %158, align 8
  %252 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc34 unwind label %endcatch.resume2

.noexc34:                                         ; preds = %250
  %253 = getelementptr i8, ptr %252, i64 8
  %254 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 0
  store i64 8, ptr %254, align 8
  %255 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %255, align 8
  %256 = getelementptr { i64, ptr, ptr }, ptr %253, i32 0, i32 2
  store ptr %251, ptr %256, align 8
  br label %257

257:                                              ; preds = %249, %.noexc34
  %258 = phi ptr [ %253, %.noexc34 ], [ %248, %249 ]
  br label %259

259:                                              ; preds = %257
  br label %267

260:                                              ; preds = %226
  %261 = load ptr, ptr %158, align 8
  %262 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc31 unwind label %endcatch.resume2

.noexc31:                                         ; preds = %260
  %263 = getelementptr i8, ptr %262, i64 8
  %264 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 0
  store i64 8, ptr %264, align 8
  %265 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %265, align 8
  %266 = getelementptr { i64, ptr, ptr }, ptr %263, i32 0, i32 2
  store ptr %261, ptr %266, align 8
  br label %267

267:                                              ; preds = %259, %.noexc31
  %268 = phi ptr [ %263, %.noexc31 ], [ %258, %259 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %224, %269
  %271 = phi ptr [ %268, %269 ], [ %225, %224 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc36, %272
  %274 = phi ptr [ %271, %272 ], [ %218, %.noexc36 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc28, %275
  %277 = phi ptr [ %274, %275 ], [ %209, %.noexc28 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc3, %278
  %280 = phi ptr [ %277, %278 ], [ %200, %.noexc3 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc27, %281
  %283 = phi ptr [ %280, %281 ], [ %191, %.noexc27 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc26, %284
  %286 = phi ptr [ %283, %284 ], [ %182, %.noexc26 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc23, %287
  %289 = phi ptr [ %286, %287 ], [ %173, %.noexc23 ]
  br label %290

290:                                              ; preds = %288
  br label %291

291:                                              ; preds = %.noexc5, %290
  %292 = phi ptr [ %289, %290 ], [ %164, %.noexc5 ]
  br label %293

293:                                              ; preds = %291
  store ptr %292, ptr %4, align 8
  %294 = load double, ptr %8, align 8
  %295 = fadd double %294, 5.000000e+00
  %296 = invoke i32 @f(double %295)
          to label %297 unwind label %endcatch.pad

297:                                              ; preds = %293
  call void @__cxa_end_catch()
  br label %298

298:                                              ; preds = %154, %297
  call void @__cxa_end_catch()
  br label %442

endcatch.pad:                                     ; preds = %293
  %299 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  br label %.caught

300:                                              ; No predecessors!
  %301 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %300
  %302 = phi { ptr, i32 } [ %301, %300 ], [ %299, %endcatch.pad ]
  %303 = extractvalue { ptr, i32 } %302, 0
  %304 = call ptr @__cxa_begin_catch(ptr %303)
  %305 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %.caught
  %306 = icmp eq ptr %305, @_ZTIi
  br i1 %306, label %307, label %314

307:                                              ; preds = %.noexc24
  %308 = load i32, ptr %304, align 4
  %309 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %307
  %310 = getelementptr i8, ptr %309, i64 8
  %311 = getelementptr { i64, ptr, i32 }, ptr %310, i32 0, i32 0
  store i64 4, ptr %311, align 8
  %312 = getelementptr { i64, ptr, i32 }, ptr %310, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %312, align 8
  %313 = getelementptr { i64, ptr, i32 }, ptr %310, i32 0, i32 2
  store i32 %308, ptr %313, align 4
  br label %437

314:                                              ; preds = %.noexc24
  %315 = icmp eq ptr %305, @_ZTIj
  br i1 %315, label %316, label %323

316:                                              ; preds = %314
  %317 = load i32, ptr %304, align 4
  %318 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %316
  %319 = getelementptr i8, ptr %318, i64 8
  %320 = getelementptr { i64, ptr, i32 }, ptr %319, i32 0, i32 0
  store i64 4, ptr %320, align 8
  %321 = getelementptr { i64, ptr, i32 }, ptr %319, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %321, align 8
  %322 = getelementptr { i64, ptr, i32 }, ptr %319, i32 0, i32 2
  store i32 %317, ptr %322, align 4
  br label %434

323:                                              ; preds = %314
  %324 = icmp eq ptr %305, @_ZTId
  br i1 %324, label %325, label %332

325:                                              ; preds = %323
  %326 = load double, ptr %304, align 8
  %327 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %325
  %328 = getelementptr i8, ptr %327, i64 8
  %329 = getelementptr { i64, ptr, double }, ptr %328, i32 0, i32 0
  store i64 8, ptr %329, align 8
  %330 = getelementptr { i64, ptr, double }, ptr %328, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %330, align 8
  %331 = getelementptr { i64, ptr, double }, ptr %328, i32 0, i32 2
  store double %326, ptr %331, align 8
  br label %431

332:                                              ; preds = %323
  %333 = icmp eq ptr %305, @_ZTIf
  br i1 %333, label %334, label %341

334:                                              ; preds = %332
  %335 = load float, ptr %304, align 4
  %336 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %334
  %337 = getelementptr i8, ptr %336, i64 8
  %338 = getelementptr { i64, ptr, float }, ptr %337, i32 0, i32 0
  store i64 4, ptr %338, align 8
  %339 = getelementptr { i64, ptr, float }, ptr %337, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %339, align 8
  %340 = getelementptr { i64, ptr, float }, ptr %337, i32 0, i32 2
  store float %335, ptr %340, align 4
  br label %428

341:                                              ; preds = %332
  %342 = icmp eq ptr %305, @_ZTIb
  br i1 %342, label %343, label %350

343:                                              ; preds = %341
  %344 = load i1, ptr %304, align 1
  %345 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %343
  %346 = getelementptr i8, ptr %345, i64 8
  %347 = getelementptr { i64, ptr, i1 }, ptr %346, i32 0, i32 0
  store i64 1, ptr %347, align 8
  %348 = getelementptr { i64, ptr, i1 }, ptr %346, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %348, align 8
  %349 = getelementptr { i64, ptr, i1 }, ptr %346, i32 0, i32 2
  store i1 %344, ptr %349, align 1
  br label %425

350:                                              ; preds = %341
  %351 = icmp eq ptr %305, @_ZTIx
  br i1 %351, label %352, label %359

352:                                              ; preds = %350
  %353 = load i64, ptr %304, align 8
  %354 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %352
  %355 = getelementptr i8, ptr %354, i64 8
  %356 = getelementptr { i64, ptr, i64 }, ptr %355, i32 0, i32 0
  store i64 8, ptr %356, align 8
  %357 = getelementptr { i64, ptr, i64 }, ptr %355, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %357, align 8
  %358 = getelementptr { i64, ptr, i64 }, ptr %355, i32 0, i32 2
  store i64 %353, ptr %358, align 8
  br label %422

359:                                              ; preds = %350
  %360 = icmp eq ptr %305, @_ZTIPc
  br i1 %360, label %361, label %368

361:                                              ; preds = %359
  %362 = load ptr, ptr %304, align 8
  %363 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %361
  %364 = getelementptr i8, ptr %363, i64 8
  %365 = getelementptr { i64, ptr, ptr }, ptr %364, i32 0, i32 0
  store i64 8, ptr %365, align 8
  %366 = getelementptr { i64, ptr, ptr }, ptr %364, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %366, align 8
  %367 = getelementptr { i64, ptr, ptr }, ptr %364, i32 0, i32 2
  store ptr %362, ptr %367, align 8
  br label %419

368:                                              ; preds = %359
  %369 = icmp eq ptr %305, @_ZTIPv
  br i1 %369, label %370, label %372

370:                                              ; preds = %368
  %371 = load ptr, ptr %304, align 8
  br label %416

372:                                              ; preds = %368
  %373 = load ptr, ptr %305, align 8
  %374 = icmp eq ptr %373, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %374, label %375, label %406

375:                                              ; preds = %372
  %376 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %305, i32 0, i32 2
  %377 = load i32, ptr %376, align 4
  %378 = and i32 %377, 1073741824
  %379 = icmp ne i32 %378, 0
  br i1 %379, label %380, label %396

380:                                              ; preds = %375
  %381 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %305, i32 0, i32 4
  %382 = load ptr, ptr %381, align 8
  %383 = icmp ne ptr %382, null
  br i1 %383, label %384, label %386

384:                                              ; preds = %380
  invoke void %382(ptr %2, ptr %304)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %384
  %385 = load ptr, ptr %2, align 8
  br label %393

386:                                              ; preds = %380
  %387 = load ptr, ptr %304, align 8
  %388 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %386
  %389 = getelementptr i8, ptr %388, i64 8
  %390 = getelementptr { i64, ptr, ptr }, ptr %389, i32 0, i32 0
  store i64 8, ptr %390, align 8
  %391 = getelementptr { i64, ptr, ptr }, ptr %389, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %391, align 8
  %392 = getelementptr { i64, ptr, ptr }, ptr %389, i32 0, i32 2
  store ptr %387, ptr %392, align 8
  br label %393

393:                                              ; preds = %.noexc11, %.noexc33
  %394 = phi ptr [ %389, %.noexc33 ], [ %385, %.noexc11 ]
  br label %395

395:                                              ; preds = %393
  br label %403

396:                                              ; preds = %375
  %397 = load ptr, ptr %304, align 8
  %398 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %396
  %399 = getelementptr i8, ptr %398, i64 8
  %400 = getelementptr { i64, ptr, ptr }, ptr %399, i32 0, i32 0
  store i64 8, ptr %400, align 8
  %401 = getelementptr { i64, ptr, ptr }, ptr %399, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %401, align 8
  %402 = getelementptr { i64, ptr, ptr }, ptr %399, i32 0, i32 2
  store ptr %397, ptr %402, align 8
  br label %403

403:                                              ; preds = %395, %.noexc18
  %404 = phi ptr [ %399, %.noexc18 ], [ %394, %395 ]
  br label %405

405:                                              ; preds = %403
  br label %413

406:                                              ; preds = %372
  %407 = load ptr, ptr %304, align 8
  %408 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc37 unwind label %endcatch.resume

.noexc37:                                         ; preds = %406
  %409 = getelementptr i8, ptr %408, i64 8
  %410 = getelementptr { i64, ptr, ptr }, ptr %409, i32 0, i32 0
  store i64 8, ptr %410, align 8
  %411 = getelementptr { i64, ptr, ptr }, ptr %409, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %411, align 8
  %412 = getelementptr { i64, ptr, ptr }, ptr %409, i32 0, i32 2
  store ptr %407, ptr %412, align 8
  br label %413

413:                                              ; preds = %405, %.noexc37
  %414 = phi ptr [ %409, %.noexc37 ], [ %404, %405 ]
  br label %415

415:                                              ; preds = %413
  br label %416

416:                                              ; preds = %370, %415
  %417 = phi ptr [ %414, %415 ], [ %371, %370 ]
  br label %418

418:                                              ; preds = %416
  br label %419

419:                                              ; preds = %.noexc32, %418
  %420 = phi ptr [ %417, %418 ], [ %364, %.noexc32 ]
  br label %421

421:                                              ; preds = %419
  br label %422

422:                                              ; preds = %.noexc30, %421
  %423 = phi ptr [ %420, %421 ], [ %355, %.noexc30 ]
  br label %424

424:                                              ; preds = %422
  br label %425

425:                                              ; preds = %.noexc9, %424
  %426 = phi ptr [ %423, %424 ], [ %346, %.noexc9 ]
  br label %427

427:                                              ; preds = %425
  br label %428

428:                                              ; preds = %.noexc16, %427
  %429 = phi ptr [ %426, %427 ], [ %337, %.noexc16 ]
  br label %430

430:                                              ; preds = %428
  br label %431

431:                                              ; preds = %.noexc7, %430
  %432 = phi ptr [ %429, %430 ], [ %328, %.noexc7 ]
  br label %433

433:                                              ; preds = %431
  br label %434

434:                                              ; preds = %.noexc21, %433
  %435 = phi ptr [ %432, %433 ], [ %319, %.noexc21 ]
  br label %436

436:                                              ; preds = %434
  br label %437

437:                                              ; preds = %.noexc22, %436
  %438 = phi ptr [ %435, %436 ], [ %310, %.noexc22 ]
  br label %439

439:                                              ; preds = %437
  store ptr %438, ptr %7, align 8
  %440 = load ptr, ptr %7, align 8
  %441 = invoke double @"___unbox<!ts.number>"(ptr %440)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %439
  store double %441, ptr %9, align 8
  call void @__cxa_end_catch()
  br label %443

442:                                              ; preds = %12, %298
  store double -1.000000e+00, ptr %9, align 8
  br label %443

443:                                              ; preds = %442, %.noexc14
  %444 = load double, ptr %9, align 8
  ret double %444

endcatch.resume:                                  ; preds = %406, %386, %361, %352, %118, %.caught, %307, %316, %28, %64, %396, %46, %334, %55, %439, %96, %108, %384, %19, %343, %325, %37, %98, %73, %13
  %445 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %445

endcatch.resume2:                                 ; preds = %215, %240, %250, %260, %238, %206, %188, %179, %170, %155, %161, %197
  %446 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %446
}

; Function Attrs: mustprogress
define double @throwTakenInCatch(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca double, align 8
  store double %0, ptr %7, align 8
  store double 0.000000e+00, ptr %6, align 8
  %9 = load double, ptr %7, align 8
  %10 = invoke i32 @f(double %9)
          to label %11 unwind label %12

11:                                               ; preds = %1
  br label %297

12:                                               ; preds = %1
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc3 unwind label %endcatch.resume2

.noexc3:                                          ; preds = %12
  %17 = icmp eq ptr %16, @_ZTIi
  br i1 %17, label %18, label %25

18:                                               ; preds = %.noexc3
  %19 = load i32, ptr %15, align 4
  %20 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume2

.noexc16:                                         ; preds = %18
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 0
  store i64 4, ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 2
  store i32 %19, ptr %24, align 4
  br label %148

25:                                               ; preds = %.noexc3
  %26 = icmp eq ptr %16, @_ZTIj
  br i1 %26, label %27, label %34

27:                                               ; preds = %25
  %28 = load i32, ptr %15, align 4
  %29 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume2

.noexc22:                                         ; preds = %27
  %30 = getelementptr i8, ptr %29, i64 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 0
  store i64 4, ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %32, align 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %30, i32 0, i32 2
  store i32 %28, ptr %33, align 4
  br label %145

34:                                               ; preds = %25
  %35 = icmp eq ptr %16, @_ZTId
  br i1 %35, label %36, label %43

36:                                               ; preds = %34
  %37 = load double, ptr %15, align 8
  %38 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume2

.noexc20:                                         ; preds = %36
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 0
  store i64 8, ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %41, align 8
  %42 = getelementptr { i64, ptr, double }, ptr %39, i32 0, i32 2
  store double %37, ptr %42, align 8
  br label %142

43:                                               ; preds = %34
  %44 = icmp eq ptr %16, @_ZTIf
  br i1 %44, label %45, label %52

45:                                               ; preds = %43
  %46 = load float, ptr %15, align 4
  %47 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume2

.noexc15:                                         ; preds = %45
  %48 = getelementptr i8, ptr %47, i64 8
  %49 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 0
  store i64 4, ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %50, align 8
  %51 = getelementptr { i64, ptr, float }, ptr %48, i32 0, i32 2
  store float %46, ptr %51, align 4
  br label %139

52:                                               ; preds = %43
  %53 = icmp eq ptr %16, @_ZTIb
  br i1 %53, label %54, label %61

54:                                               ; preds = %52
  %55 = load i1, ptr %15, align 1
  %56 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume2

.noexc11:                                         ; preds = %54
  %57 = getelementptr i8, ptr %56, i64 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 0
  store i64 1, ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %59, align 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %57, i32 0, i32 2
  store i1 %55, ptr %60, align 1
  br label %136

61:                                               ; preds = %52
  %62 = icmp eq ptr %16, @_ZTIx
  br i1 %62, label %63, label %70

63:                                               ; preds = %61
  %64 = load i64, ptr %15, align 8
  %65 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume2

.noexc19:                                         ; preds = %63
  %66 = getelementptr i8, ptr %65, i64 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 0
  store i64 8, ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %68, align 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %66, i32 0, i32 2
  store i64 %64, ptr %69, align 8
  br label %133

70:                                               ; preds = %61
  %71 = icmp eq ptr %16, @_ZTIPc
  br i1 %71, label %72, label %79

72:                                               ; preds = %70
  %73 = load ptr, ptr %15, align 8
  %74 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume2

.noexc12:                                         ; preds = %72
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 0
  store i64 8, ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %77, align 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %75, i32 0, i32 2
  store ptr %73, ptr %78, align 8
  br label %130

79:                                               ; preds = %70
  %80 = icmp eq ptr %16, @_ZTIPv
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = load ptr, ptr %15, align 8
  br label %127

83:                                               ; preds = %79
  %84 = load ptr, ptr %16, align 8
  %85 = icmp eq ptr %84, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %85, label %86, label %117

86:                                               ; preds = %83
  %87 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %16, i32 0, i32 2
  %88 = load i32, ptr %87, align 4
  %89 = and i32 %88, 1073741824
  %90 = icmp ne i32 %89, 0
  br i1 %90, label %91, label %107

91:                                               ; preds = %86
  %92 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %16, i32 0, i32 4
  %93 = load ptr, ptr %92, align 8
  %94 = icmp ne ptr %93, null
  br i1 %94, label %95, label %97

95:                                               ; preds = %91
  invoke void %93(ptr %4, ptr %15)
          to label %.noexc25 unwind label %endcatch.resume2

.noexc25:                                         ; preds = %95
  %96 = load ptr, ptr %4, align 8
  br label %104

97:                                               ; preds = %91
  %98 = load ptr, ptr %15, align 8
  %99 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume2

.noexc17:                                         ; preds = %97
  %100 = getelementptr i8, ptr %99, i64 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 0
  store i64 8, ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 2
  store ptr %98, ptr %103, align 8
  br label %104

104:                                              ; preds = %.noexc25, %.noexc17
  %105 = phi ptr [ %100, %.noexc17 ], [ %96, %.noexc25 ]
  br label %106

106:                                              ; preds = %104
  br label %114

107:                                              ; preds = %86
  %108 = load ptr, ptr %15, align 8
  %109 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume2

.noexc9:                                          ; preds = %107
  %110 = getelementptr i8, ptr %109, i64 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 0
  store i64 8, ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 2
  store ptr %108, ptr %113, align 8
  br label %114

114:                                              ; preds = %106, %.noexc9
  %115 = phi ptr [ %110, %.noexc9 ], [ %105, %106 ]
  br label %116

116:                                              ; preds = %114
  br label %124

117:                                              ; preds = %83
  %118 = load ptr, ptr %15, align 8
  %119 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume2

.noexc4:                                          ; preds = %117
  %120 = getelementptr i8, ptr %119, i64 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 0
  store i64 8, ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 2
  store ptr %118, ptr %123, align 8
  br label %124

124:                                              ; preds = %116, %.noexc4
  %125 = phi ptr [ %120, %.noexc4 ], [ %115, %116 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %81, %126
  %128 = phi ptr [ %125, %126 ], [ %82, %81 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc12, %129
  %131 = phi ptr [ %128, %129 ], [ %75, %.noexc12 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc19, %132
  %134 = phi ptr [ %131, %132 ], [ %66, %.noexc19 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc11, %135
  %137 = phi ptr [ %134, %135 ], [ %57, %.noexc11 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc15, %138
  %140 = phi ptr [ %137, %138 ], [ %48, %.noexc15 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc20, %141
  %143 = phi ptr [ %140, %141 ], [ %39, %.noexc20 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc22, %144
  %146 = phi ptr [ %143, %144 ], [ %30, %.noexc22 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc16, %147
  %149 = phi ptr [ %146, %147 ], [ %21, %.noexc16 ]
  br label %150

150:                                              ; preds = %148
  store ptr %149, ptr %5, align 8
  %151 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc10 unwind label %endcatch.resume2

.noexc10:                                         ; preds = %150
  store i32 7, ptr %151, align 4
  invoke void @__cxa_throw(ptr %151, ptr @_ZTIi, ptr null)
          to label %300 unwind label %152

152:                                              ; preds = %.noexc10
  %153 = landingpad { ptr, i32 }
          catch ptr null
  %154 = extractvalue { ptr, i32 } %153, 0
  %155 = call ptr @__cxa_begin_catch(ptr %154)
  %156 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %152
  %157 = icmp eq ptr %156, @_ZTIi
  br i1 %157, label %158, label %165

158:                                              ; preds = %.noexc13
  %159 = load i32, ptr %155, align 4
  %160 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %158
  %161 = getelementptr i8, ptr %160, i64 8
  %162 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 0
  store i64 4, ptr %162, align 8
  %163 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %163, align 8
  %164 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 2
  store i32 %159, ptr %164, align 4
  br label %288

165:                                              ; preds = %.noexc13
  %166 = icmp eq ptr %156, @_ZTIj
  br i1 %166, label %167, label %174

167:                                              ; preds = %165
  %168 = load i32, ptr %155, align 4
  %169 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %167
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
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %176
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %185
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %194
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
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %203
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %212
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %235
  %236 = load ptr, ptr %2, align 8
  br label %244

237:                                              ; preds = %231
  %238 = load ptr, ptr %155, align 8
  %239 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %237
  %240 = getelementptr i8, ptr %239, i64 8
  %241 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 0
  store i64 8, ptr %241, align 8
  %242 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %242, align 8
  %243 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 2
  store ptr %238, ptr %243, align 8
  br label %244

244:                                              ; preds = %.noexc7, %.noexc1
  %245 = phi ptr [ %240, %.noexc1 ], [ %236, %.noexc7 ]
  br label %246

246:                                              ; preds = %244
  br label %254

247:                                              ; preds = %226
  %248 = load ptr, ptr %155, align 8
  %249 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %247
  %250 = getelementptr i8, ptr %249, i64 8
  %251 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 0
  store i64 8, ptr %251, align 8
  %252 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %252, align 8
  %253 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 2
  store ptr %248, ptr %253, align 8
  br label %254

254:                                              ; preds = %246, %.noexc
  %255 = phi ptr [ %250, %.noexc ], [ %245, %246 ]
  br label %256

256:                                              ; preds = %254
  br label %264

257:                                              ; preds = %223
  %258 = load ptr, ptr %155, align 8
  %259 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %257
  %260 = getelementptr i8, ptr %259, i64 8
  %261 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 0
  store i64 8, ptr %261, align 8
  %262 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %262, align 8
  %263 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 2
  store ptr %258, ptr %263, align 8
  br label %264

264:                                              ; preds = %256, %.noexc27
  %265 = phi ptr [ %260, %.noexc27 ], [ %255, %256 ]
  br label %266

266:                                              ; preds = %264
  br label %267

267:                                              ; preds = %221, %266
  %268 = phi ptr [ %265, %266 ], [ %222, %221 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %.noexc5, %269
  %271 = phi ptr [ %268, %269 ], [ %215, %.noexc5 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc14, %272
  %274 = phi ptr [ %271, %272 ], [ %206, %.noexc14 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc6, %275
  %277 = phi ptr [ %274, %275 ], [ %197, %.noexc6 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc8, %278
  %280 = phi ptr [ %277, %278 ], [ %188, %.noexc8 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc21, %281
  %283 = phi ptr [ %280, %281 ], [ %179, %.noexc21 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc26, %284
  %286 = phi ptr [ %283, %284 ], [ %170, %.noexc26 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc18, %287
  %289 = phi ptr [ %286, %287 ], [ %161, %.noexc18 ]
  br label %290

290:                                              ; preds = %288
  store ptr %289, ptr %3, align 8
  %291 = load ptr, ptr %3, align 8
  %292 = invoke double @"___unbox<!ts.number>"(ptr %291)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %290
  store double %292, ptr %6, align 8
  call void @__cxa_end_catch()
  %293 = load double, ptr %6, align 8
  %294 = load ptr, ptr %5, align 8
  %295 = invoke double @"___unbox<!ts.number>"(ptr %294)
          to label %.noexc24 unwind label %endcatch.resume2

.noexc24:                                         ; preds = %.noexc23
  %296 = fadd double %293, %295
  store double %296, ptr %6, align 8
  call void @__cxa_end_catch()
  br label %297

297:                                              ; preds = %11, %.noexc24
  %298 = load double, ptr %6, align 8
  store double %298, ptr %8, align 8
  %299 = load double, ptr %8, align 8
  ret double %299

300:                                              ; preds = %.noexc10
  unreachable

endcatch.resume:                                  ; preds = %257, %167, %290, %176, %158, %203, %152, %185, %235, %194, %212, %237, %247
  %301 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %301

endcatch.resume2:                                 ; preds = %95, %.noexc23, %27, %36, %63, %97, %18, %45, %72, %54, %150, %107, %117, %12
  %302 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %302
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"___unbox<!ts.string>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr { i64, ptr, i8 }, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  %9 = zext i1 %8 to i32
  %10 = and i32 %9, 1
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %1
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %24

20:                                               ; preds = %19
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr { i64, ptr, ptr }, ptr %21, i32 0, i32 2
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %3, align 8
  br label %542

24:                                               ; preds = %19
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr { i64, ptr, i8 }, ptr %25, i32 0, i32 1
  %27 = load ptr, ptr %26, align 8
  %28 = ptrtoint ptr %27 to i64
  %29 = icmp ne i64 %28, 0
  %30 = zext i1 %29 to i32
  %31 = and i32 %30, 1
  %32 = icmp ne i32 %31, 0
  br i1 %32, label %33, label %36

33:                                               ; preds = %24
  %34 = call i32 @strcmp(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8))
  %35 = icmp eq i32 %34, 0
  br label %38

36:                                               ; preds = %24
  %37 = icmp eq i64 %28, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8) to i64)
  br label %38

38:                                               ; preds = %33, %36
  %39 = phi i1 [ %37, %36 ], [ %35, %33 ]
  br label %40

40:                                               ; preds = %38
  br i1 %39, label %41, label %47

41:                                               ; preds = %40
  %42 = load ptr, ptr %2, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 2
  %44 = load double, ptr %43, align 8
  %45 = call ptr @GC_malloc_atomic(i64 40)
  %46 = getelementptr i8, ptr %45, i64 8
  call void @__ts_number_to_string(ptr %46, double %44)
  store ptr %46, ptr %3, align 8
  br label %542

47:                                               ; preds = %40
  %48 = load ptr, ptr %2, align 8
  %49 = getelementptr { i64, ptr, i8 }, ptr %48, i32 0, i32 1
  %50 = load ptr, ptr %49, align 8
  %51 = ptrtoint ptr %50 to i64
  %52 = icmp ne i64 %51, 0
  %53 = zext i1 %52 to i32
  %54 = and i32 %53, 1
  %55 = icmp ne i32 %54, 0
  br i1 %55, label %56, label %59

56:                                               ; preds = %47
  %57 = call i32 @strcmp(ptr %50, ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8))
  %58 = icmp eq i32 %57, 0
  br label %61

59:                                               ; preds = %47
  %60 = icmp eq i64 %51, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8) to i64)
  br label %61

61:                                               ; preds = %56, %59
  %62 = phi i1 [ %60, %59 ], [ %58, %56 ]
  br label %63

63:                                               ; preds = %61
  br i1 %62, label %64, label %68

64:                                               ; preds = %63
  %65 = load ptr, ptr %2, align 8
  %66 = getelementptr { i64, ptr, ptr }, ptr %65, i32 0, i32 2
  %67 = load ptr, ptr %66, align 8
  store ptr %67, ptr %3, align 8
  br label %542

68:                                               ; preds = %63
  %69 = load ptr, ptr %2, align 8
  %70 = getelementptr { i64, ptr, i8 }, ptr %69, i32 0, i32 1
  %71 = load ptr, ptr %70, align 8
  %72 = ptrtoint ptr %71 to i64
  %73 = icmp ne i64 %72, 0
  %74 = zext i1 %73 to i32
  %75 = and i32 %74, 1
  %76 = icmp ne i32 %75, 0
  br i1 %76, label %77, label %80

77:                                               ; preds = %68
  %78 = call i32 @strcmp(ptr %71, ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8))
  %79 = icmp eq i32 %78, 0
  br label %82

80:                                               ; preds = %68
  %81 = icmp eq i64 %72, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8) to i64)
  br label %82

82:                                               ; preds = %77, %80
  %83 = phi i1 [ %81, %80 ], [ %79, %77 ]
  br label %84

84:                                               ; preds = %82
  br i1 %83, label %85, label %90

85:                                               ; preds = %84
  %86 = load ptr, ptr %2, align 8
  %87 = getelementptr { i64, ptr, i1 }, ptr %86, i32 0, i32 2
  %88 = load i1, ptr %87, align 1
  %89 = select i1 %88, ptr getelementptr inbounds nuw (i8, ptr @__true__, i64 8), ptr getelementptr inbounds nuw (i8, ptr @__false__, i64 8)
  store ptr %89, ptr %3, align 8
  br label %542

90:                                               ; preds = %84
  %91 = load ptr, ptr %2, align 8
  %92 = getelementptr { i64, ptr, i8 }, ptr %91, i32 0, i32 1
  %93 = load ptr, ptr %92, align 8
  %94 = ptrtoint ptr %93 to i64
  %95 = icmp ne i64 %94, 0
  %96 = zext i1 %95 to i32
  %97 = and i32 %96, 1
  %98 = icmp ne i32 %97, 0
  br i1 %98, label %99, label %102

99:                                               ; preds = %90
  %100 = call i32 @strcmp(ptr %93, ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8))
  %101 = icmp eq i32 %100, 0
  br label %104

102:                                              ; preds = %90
  %103 = icmp eq i64 %94, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8) to i64)
  br label %104

104:                                              ; preds = %99, %102
  %105 = phi i1 [ %103, %102 ], [ %101, %99 ]
  br label %106

106:                                              ; preds = %104
  br i1 %105, label %107, label %114

107:                                              ; preds = %106
  %108 = load ptr, ptr %2, align 8
  %109 = getelementptr { i64, ptr, float }, ptr %108, i32 0, i32 2
  %110 = load float, ptr %109, align 4
  %111 = fpext float %110 to double
  %112 = call ptr @GC_malloc_atomic(i64 40)
  %113 = getelementptr i8, ptr %112, i64 8
  call void @__ts_number_to_string(ptr %113, double %111)
  store ptr %113, ptr %3, align 8
  br label %542

114:                                              ; preds = %106
  %115 = load ptr, ptr %2, align 8
  %116 = getelementptr { i64, ptr, i8 }, ptr %115, i32 0, i32 1
  %117 = load ptr, ptr %116, align 8
  %118 = ptrtoint ptr %117 to i64
  %119 = icmp ne i64 %118, 0
  %120 = zext i1 %119 to i32
  %121 = and i32 %120, 1
  %122 = icmp ne i32 %121, 0
  br i1 %122, label %123, label %126

123:                                              ; preds = %114
  %124 = call i32 @strcmp(ptr %117, ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8))
  %125 = icmp eq i32 %124, 0
  br label %128

126:                                              ; preds = %114
  %127 = icmp eq i64 %118, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8) to i64)
  br label %128

128:                                              ; preds = %123, %126
  %129 = phi i1 [ %127, %126 ], [ %125, %123 ]
  br label %130

130:                                              ; preds = %128
  br i1 %129, label %131, label %138

131:                                              ; preds = %130
  %132 = load ptr, ptr %2, align 8
  %133 = getelementptr { i64, ptr, i32 }, ptr %132, i32 0, i32 2
  %134 = load i32, ptr %133, align 4
  %135 = call ptr @GC_malloc_atomic(i64 58)
  %136 = getelementptr i8, ptr %135, i64 8
  %137 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %136, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %134)
  store ptr %136, ptr %3, align 8
  br label %542

138:                                              ; preds = %130
  %139 = load ptr, ptr %2, align 8
  %140 = getelementptr { i64, ptr, i8 }, ptr %139, i32 0, i32 1
  %141 = load ptr, ptr %140, align 8
  %142 = ptrtoint ptr %141 to i64
  %143 = icmp ne i64 %142, 0
  %144 = zext i1 %143 to i32
  %145 = and i32 %144, 1
  %146 = icmp ne i32 %145, 0
  br i1 %146, label %147, label %150

147:                                              ; preds = %138
  %148 = call i32 @strcmp(ptr %141, ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8))
  %149 = icmp eq i32 %148, 0
  br label %152

150:                                              ; preds = %138
  %151 = icmp eq i64 %142, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8) to i64)
  br label %152

152:                                              ; preds = %147, %150
  %153 = phi i1 [ %151, %150 ], [ %149, %147 ]
  br label %154

154:                                              ; preds = %152
  br i1 %153, label %155, label %162

155:                                              ; preds = %154
  %156 = load ptr, ptr %2, align 8
  %157 = getelementptr { i64, ptr, i32 }, ptr %156, i32 0, i32 2
  %158 = load i32, ptr %157, align 4
  %159 = call ptr @GC_malloc_atomic(i64 58)
  %160 = getelementptr i8, ptr %159, i64 8
  %161 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %160, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %158)
  store ptr %160, ptr %3, align 8
  br label %542

162:                                              ; preds = %154
  %163 = load ptr, ptr %2, align 8
  %164 = getelementptr { i64, ptr, i8 }, ptr %163, i32 0, i32 1
  %165 = load ptr, ptr %164, align 8
  %166 = ptrtoint ptr %165 to i64
  %167 = icmp ne i64 %166, 0
  %168 = zext i1 %167 to i32
  %169 = and i32 %168, 1
  %170 = icmp ne i32 %169, 0
  br i1 %170, label %171, label %174

171:                                              ; preds = %162
  %172 = call i32 @strcmp(ptr %165, ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8))
  %173 = icmp eq i32 %172, 0
  br label %176

174:                                              ; preds = %162
  %175 = icmp eq i64 %166, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8) to i64)
  br label %176

176:                                              ; preds = %171, %174
  %177 = phi i1 [ %175, %174 ], [ %173, %171 ]
  br label %178

178:                                              ; preds = %176
  br i1 %177, label %179, label %186

179:                                              ; preds = %178
  %180 = load ptr, ptr %2, align 8
  %181 = getelementptr { i64, ptr, i32 }, ptr %180, i32 0, i32 2
  %182 = load i32, ptr %181, align 4
  %183 = call ptr @GC_malloc_atomic(i64 58)
  %184 = getelementptr i8, ptr %183, i64 8
  %185 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %184, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_555419431375823311, i64 8), i32 %182)
  store ptr %184, ptr %3, align 8
  br label %542

186:                                              ; preds = %178
  %187 = load ptr, ptr %2, align 8
  %188 = getelementptr { i64, ptr, i8 }, ptr %187, i32 0, i32 1
  %189 = load ptr, ptr %188, align 8
  %190 = ptrtoint ptr %189 to i64
  %191 = icmp ne i64 %190, 0
  %192 = zext i1 %191 to i32
  %193 = and i32 %192, 1
  %194 = icmp ne i32 %193, 0
  br i1 %194, label %195, label %198

195:                                              ; preds = %186
  %196 = call i32 @strcmp(ptr %189, ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8))
  %197 = icmp eq i32 %196, 0
  br label %200

198:                                              ; preds = %186
  %199 = icmp eq i64 %190, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8) to i64)
  br label %200

200:                                              ; preds = %195, %198
  %201 = phi i1 [ %199, %198 ], [ %197, %195 ]
  br label %202

202:                                              ; preds = %200
  br i1 %201, label %203, label %210

203:                                              ; preds = %202
  %204 = load ptr, ptr %2, align 8
  %205 = getelementptr { i64, ptr, i64 }, ptr %204, i32 0, i32 2
  %206 = load i64, ptr %205, align 8
  %207 = call ptr @GC_malloc_atomic(i64 58)
  %208 = getelementptr i8, ptr %207, i64 8
  %209 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %208, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %206)
  store ptr %208, ptr %3, align 8
  br label %542

210:                                              ; preds = %202
  %211 = load ptr, ptr %2, align 8
  %212 = getelementptr { i64, ptr, i8 }, ptr %211, i32 0, i32 1
  %213 = load ptr, ptr %212, align 8
  %214 = ptrtoint ptr %213 to i64
  %215 = icmp ne i64 %214, 0
  %216 = zext i1 %215 to i32
  %217 = and i32 %216, 1
  %218 = icmp ne i32 %217, 0
  br i1 %218, label %219, label %222

219:                                              ; preds = %210
  %220 = call i32 @strcmp(ptr %213, ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8))
  %221 = icmp eq i32 %220, 0
  br label %224

222:                                              ; preds = %210
  %223 = icmp eq i64 %214, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8) to i64)
  br label %224

224:                                              ; preds = %219, %222
  %225 = phi i1 [ %223, %222 ], [ %221, %219 ]
  br label %226

226:                                              ; preds = %224
  br i1 %225, label %227, label %233

227:                                              ; preds = %226
  %228 = load ptr, ptr %2, align 8
  %229 = getelementptr { i64, ptr, double }, ptr %228, i32 0, i32 2
  %230 = load double, ptr %229, align 8
  %231 = call ptr @GC_malloc_atomic(i64 40)
  %232 = getelementptr i8, ptr %231, i64 8
  call void @__ts_number_to_string(ptr %232, double %230)
  store ptr %232, ptr %3, align 8
  br label %542

233:                                              ; preds = %226
  %234 = load ptr, ptr %2, align 8
  %235 = getelementptr { i64, ptr, i8 }, ptr %234, i32 0, i32 1
  %236 = load ptr, ptr %235, align 8
  %237 = ptrtoint ptr %236 to i64
  %238 = icmp ne i64 %237, 0
  %239 = zext i1 %238 to i32
  %240 = and i32 %239, 1
  %241 = icmp ne i32 %240, 0
  br i1 %241, label %242, label %245

242:                                              ; preds = %233
  %243 = call i32 @strcmp(ptr %236, ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8))
  %244 = icmp eq i32 %243, 0
  br label %247

245:                                              ; preds = %233
  %246 = icmp eq i64 %237, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8) to i64)
  br label %247

247:                                              ; preds = %242, %245
  %248 = phi i1 [ %246, %245 ], [ %244, %242 ]
  br label %249

249:                                              ; preds = %247
  br i1 %248, label %250, label %257

250:                                              ; preds = %249
  %251 = load ptr, ptr %2, align 8
  %252 = getelementptr { i64, ptr, i64 }, ptr %251, i32 0, i32 2
  %253 = load i64, ptr %252, align 8
  %254 = call ptr @GC_malloc_atomic(i64 58)
  %255 = getelementptr i8, ptr %254, i64 8
  %256 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %255, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %253)
  store ptr %255, ptr %3, align 8
  br label %542

257:                                              ; preds = %249
  %258 = load ptr, ptr %2, align 8
  %259 = getelementptr { i64, ptr, i8 }, ptr %258, i32 0, i32 1
  %260 = load ptr, ptr %259, align 8
  %261 = ptrtoint ptr %260 to i64
  %262 = icmp ne i64 %261, 0
  %263 = zext i1 %262 to i32
  %264 = and i32 %263, 1
  %265 = icmp ne i32 %264, 0
  br i1 %265, label %266, label %269

266:                                              ; preds = %257
  %267 = call i32 @strcmp(ptr %260, ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8))
  %268 = icmp eq i32 %267, 0
  br label %271

269:                                              ; preds = %257
  %270 = icmp eq i64 %261, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8) to i64)
  br label %271

271:                                              ; preds = %266, %269
  %272 = phi i1 [ %270, %269 ], [ %268, %266 ]
  br label %273

273:                                              ; preds = %271
  br i1 %272, label %274, label %281

274:                                              ; preds = %273
  %275 = load ptr, ptr %2, align 8
  %276 = getelementptr { i64, ptr, i64 }, ptr %275, i32 0, i32 2
  %277 = load i64, ptr %276, align 8
  %278 = call ptr @GC_malloc_atomic(i64 58)
  %279 = getelementptr i8, ptr %278, i64 8
  %280 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %279, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %277)
  store ptr %279, ptr %3, align 8
  br label %542

281:                                              ; preds = %273
  %282 = load ptr, ptr %2, align 8
  %283 = getelementptr { i64, ptr, i8 }, ptr %282, i32 0, i32 1
  %284 = load ptr, ptr %283, align 8
  %285 = ptrtoint ptr %284 to i64
  %286 = icmp ne i64 %285, 0
  %287 = zext i1 %286 to i32
  %288 = and i32 %287, 1
  %289 = icmp ne i32 %288, 0
  br i1 %289, label %290, label %293

290:                                              ; preds = %281
  %291 = call i32 @strcmp(ptr %284, ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8))
  %292 = icmp eq i32 %291, 0
  br label %295

293:                                              ; preds = %281
  %294 = icmp eq i64 %285, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8) to i64)
  br label %295

295:                                              ; preds = %290, %293
  %296 = phi i1 [ %294, %293 ], [ %292, %290 ]
  br label %297

297:                                              ; preds = %295
  br i1 %296, label %298, label %305

298:                                              ; preds = %297
  %299 = load ptr, ptr %2, align 8
  %300 = getelementptr { i64, ptr, i64 }, ptr %299, i32 0, i32 2
  %301 = load i64, ptr %300, align 8
  %302 = call ptr @GC_malloc_atomic(i64 58)
  %303 = getelementptr i8, ptr %302, i64 8
  %304 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %303, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_15473922249627149015, i64 8), i64 %301)
  store ptr %303, ptr %3, align 8
  br label %542

305:                                              ; preds = %297
  %306 = load ptr, ptr %2, align 8
  %307 = getelementptr { i64, ptr, i8 }, ptr %306, i32 0, i32 1
  %308 = load ptr, ptr %307, align 8
  %309 = ptrtoint ptr %308 to i64
  %310 = icmp ne i64 %309, 0
  %311 = zext i1 %310 to i32
  %312 = and i32 %311, 1
  %313 = icmp ne i32 %312, 0
  br i1 %313, label %314, label %317

314:                                              ; preds = %305
  %315 = call i32 @strcmp(ptr %308, ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8))
  %316 = icmp eq i32 %315, 0
  br label %319

317:                                              ; preds = %305
  %318 = icmp eq i64 %309, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8) to i64)
  br label %319

319:                                              ; preds = %314, %317
  %320 = phi i1 [ %318, %317 ], [ %316, %314 ]
  br label %321

321:                                              ; preds = %319
  br i1 %320, label %322, label %330

322:                                              ; preds = %321
  %323 = load ptr, ptr %2, align 8
  %324 = getelementptr { i64, ptr, i8 }, ptr %323, i32 0, i32 2
  %325 = load i8, ptr %324, align 1
  %326 = call ptr @GC_malloc(i64 10)
  %327 = getelementptr i8, ptr %326, i64 8
  %328 = getelementptr i8, ptr %327, i32 0
  store i8 %325, ptr %328, align 1
  %329 = getelementptr i8, ptr %327, i32 1
  store i8 0, ptr %329, align 1
  store ptr %327, ptr %3, align 8
  br label %542

330:                                              ; preds = %321
  %331 = load ptr, ptr %2, align 8
  %332 = getelementptr { i64, ptr, i8 }, ptr %331, i32 0, i32 1
  %333 = load ptr, ptr %332, align 8
  %334 = ptrtoint ptr %333 to i64
  %335 = icmp ne i64 %334, 0
  %336 = zext i1 %335 to i32
  %337 = and i32 %336, 1
  %338 = icmp ne i32 %337, 0
  br i1 %338, label %339, label %342

339:                                              ; preds = %330
  %340 = call i32 @strcmp(ptr %333, ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8))
  %341 = icmp eq i32 %340, 0
  br label %344

342:                                              ; preds = %330
  %343 = icmp eq i64 %334, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8) to i64)
  br label %344

344:                                              ; preds = %339, %342
  %345 = phi i1 [ %343, %342 ], [ %341, %339 ]
  br label %346

346:                                              ; preds = %344
  br i1 %345, label %347, label %354

347:                                              ; preds = %346
  %348 = load ptr, ptr %2, align 8
  %349 = getelementptr { i64, ptr, i64 }, ptr %348, i32 0, i32 2
  %350 = load i64, ptr %349, align 8
  %351 = call ptr @GC_malloc_atomic(i64 58)
  %352 = getelementptr i8, ptr %351, i64 8
  %353 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %352, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %350)
  store ptr %352, ptr %3, align 8
  br label %542

354:                                              ; preds = %346
  %355 = load ptr, ptr %2, align 8
  %356 = getelementptr { i64, ptr, i8 }, ptr %355, i32 0, i32 1
  %357 = load ptr, ptr %356, align 8
  %358 = ptrtoint ptr %357 to i64
  %359 = icmp ne i64 %358, 0
  %360 = zext i1 %359 to i32
  %361 = and i32 %360, 1
  %362 = icmp ne i32 %361, 0
  br i1 %362, label %363, label %366

363:                                              ; preds = %354
  %364 = call i32 @strcmp(ptr %357, ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8))
  %365 = icmp eq i32 %364, 0
  br label %368

366:                                              ; preds = %354
  %367 = icmp eq i64 %358, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8) to i64)
  br label %368

368:                                              ; preds = %363, %366
  %369 = phi i1 [ %367, %366 ], [ %365, %363 ]
  br label %370

370:                                              ; preds = %368
  br i1 %369, label %371, label %379

371:                                              ; preds = %370
  %372 = load ptr, ptr %2, align 8
  %373 = getelementptr { i64, ptr, i16 }, ptr %372, i32 0, i32 2
  %374 = load i16, ptr %373, align 2
  %375 = sext i16 %374 to i32
  %376 = call ptr @GC_malloc_atomic(i64 58)
  %377 = getelementptr i8, ptr %376, i64 8
  %378 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %377, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_12865164881251510284, i64 8), i32 %375)
  store ptr %377, ptr %3, align 8
  br label %542

379:                                              ; preds = %370
  %380 = load ptr, ptr %2, align 8
  %381 = getelementptr { i64, ptr, i8 }, ptr %380, i32 0, i32 1
  %382 = load ptr, ptr %381, align 8
  %383 = ptrtoint ptr %382 to i64
  %384 = icmp ne i64 %383, 0
  %385 = zext i1 %384 to i32
  %386 = and i32 %385, 1
  %387 = icmp ne i32 %386, 0
  br i1 %387, label %388, label %391

388:                                              ; preds = %379
  %389 = call i32 @strcmp(ptr %382, ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8))
  %390 = icmp eq i32 %389, 0
  br label %393

391:                                              ; preds = %379
  %392 = icmp eq i64 %383, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8) to i64)
  br label %393

393:                                              ; preds = %388, %391
  %394 = phi i1 [ %392, %391 ], [ %390, %388 ]
  br label %395

395:                                              ; preds = %393
  br i1 %394, label %396, label %404

396:                                              ; preds = %395
  %397 = load ptr, ptr %2, align 8
  %398 = getelementptr { i64, ptr, i16 }, ptr %397, i32 0, i32 2
  %399 = load i16, ptr %398, align 2
  %400 = sext i16 %399 to i32
  %401 = call ptr @GC_malloc_atomic(i64 58)
  %402 = getelementptr i8, ptr %401, i64 8
  %403 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %402, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_12865164881251510284, i64 8), i32 %400)
  store ptr %402, ptr %3, align 8
  br label %542

404:                                              ; preds = %395
  %405 = load ptr, ptr %2, align 8
  %406 = getelementptr { i64, ptr, i8 }, ptr %405, i32 0, i32 1
  %407 = load ptr, ptr %406, align 8
  %408 = ptrtoint ptr %407 to i64
  %409 = icmp ne i64 %408, 0
  %410 = zext i1 %409 to i32
  %411 = and i32 %410, 1
  %412 = icmp ne i32 %411, 0
  br i1 %412, label %413, label %416

413:                                              ; preds = %404
  %414 = call i32 @strcmp(ptr %407, ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8))
  %415 = icmp eq i32 %414, 0
  br label %418

416:                                              ; preds = %404
  %417 = icmp eq i64 %408, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8) to i64)
  br label %418

418:                                              ; preds = %413, %416
  %419 = phi i1 [ %417, %416 ], [ %415, %413 ]
  br label %420

420:                                              ; preds = %418
  br i1 %419, label %421, label %429

421:                                              ; preds = %420
  %422 = load ptr, ptr %2, align 8
  %423 = getelementptr { i64, ptr, i16 }, ptr %422, i32 0, i32 2
  %424 = load i16, ptr %423, align 2
  %425 = zext i16 %424 to i32
  %426 = call ptr @GC_malloc_atomic(i64 58)
  %427 = getelementptr i8, ptr %426, i64 8
  %428 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %427, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11110327606031142231, i64 8), i32 %425)
  store ptr %427, ptr %3, align 8
  br label %542

429:                                              ; preds = %420
  %430 = load ptr, ptr %2, align 8
  %431 = getelementptr { i64, ptr, i8 }, ptr %430, i32 0, i32 1
  %432 = load ptr, ptr %431, align 8
  %433 = ptrtoint ptr %432 to i64
  %434 = icmp ne i64 %433, 0
  %435 = zext i1 %434 to i32
  %436 = and i32 %435, 1
  %437 = icmp ne i32 %436, 0
  br i1 %437, label %438, label %441

438:                                              ; preds = %429
  %439 = call i32 @strcmp(ptr %432, ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8))
  %440 = icmp eq i32 %439, 0
  br label %443

441:                                              ; preds = %429
  %442 = icmp eq i64 %433, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8) to i64)
  br label %443

443:                                              ; preds = %438, %441
  %444 = phi i1 [ %442, %441 ], [ %440, %438 ]
  br label %445

445:                                              ; preds = %443
  br i1 %444, label %446, label %454

446:                                              ; preds = %445
  %447 = load ptr, ptr %2, align 8
  %448 = getelementptr { i64, ptr, i8 }, ptr %447, i32 0, i32 2
  %449 = load i8, ptr %448, align 1
  %450 = sext i8 %449 to i32
  %451 = call ptr @GC_malloc_atomic(i64 58)
  %452 = getelementptr i8, ptr %451, i64 8
  %453 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %452, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_3788960132816803490, i64 8), i32 %450)
  store ptr %452, ptr %3, align 8
  br label %542

454:                                              ; preds = %445
  %455 = load ptr, ptr %2, align 8
  %456 = getelementptr { i64, ptr, i8 }, ptr %455, i32 0, i32 1
  %457 = load ptr, ptr %456, align 8
  %458 = ptrtoint ptr %457 to i64
  %459 = icmp ne i64 %458, 0
  %460 = zext i1 %459 to i32
  %461 = and i32 %460, 1
  %462 = icmp ne i32 %461, 0
  br i1 %462, label %463, label %466

463:                                              ; preds = %454
  %464 = call i32 @strcmp(ptr %457, ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8))
  %465 = icmp eq i32 %464, 0
  br label %468

466:                                              ; preds = %454
  %467 = icmp eq i64 %458, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8) to i64)
  br label %468

468:                                              ; preds = %463, %466
  %469 = phi i1 [ %467, %466 ], [ %465, %463 ]
  br label %470

470:                                              ; preds = %468
  br i1 %469, label %471, label %479

471:                                              ; preds = %470
  %472 = load ptr, ptr %2, align 8
  %473 = getelementptr { i64, ptr, i8 }, ptr %472, i32 0, i32 2
  %474 = load i8, ptr %473, align 1
  %475 = sext i8 %474 to i32
  %476 = call ptr @GC_malloc_atomic(i64 58)
  %477 = getelementptr i8, ptr %476, i64 8
  %478 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %477, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_3788960132816803490, i64 8), i32 %475)
  store ptr %477, ptr %3, align 8
  br label %542

479:                                              ; preds = %470
  %480 = load ptr, ptr %2, align 8
  %481 = getelementptr { i64, ptr, i8 }, ptr %480, i32 0, i32 1
  %482 = load ptr, ptr %481, align 8
  %483 = ptrtoint ptr %482 to i64
  %484 = icmp ne i64 %483, 0
  %485 = zext i1 %484 to i32
  %486 = and i32 %485, 1
  %487 = icmp ne i32 %486, 0
  br i1 %487, label %488, label %491

488:                                              ; preds = %479
  %489 = call i32 @strcmp(ptr %482, ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8))
  %490 = icmp eq i32 %489, 0
  br label %493

491:                                              ; preds = %479
  %492 = icmp eq i64 %483, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8) to i64)
  br label %493

493:                                              ; preds = %488, %491
  %494 = phi i1 [ %492, %491 ], [ %490, %488 ]
  br label %495

495:                                              ; preds = %493
  br i1 %494, label %496, label %504

496:                                              ; preds = %495
  %497 = load ptr, ptr %2, align 8
  %498 = getelementptr { i64, ptr, i8 }, ptr %497, i32 0, i32 2
  %499 = load i8, ptr %498, align 1
  %500 = zext i8 %499 to i32
  %501 = call ptr @GC_malloc_atomic(i64 58)
  %502 = getelementptr i8, ptr %501, i64 8
  %503 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %502, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_12904420749775593087, i64 8), i32 %500)
  store ptr %502, ptr %3, align 8
  br label %542

504:                                              ; preds = %495
  %505 = load ptr, ptr %2, align 8
  %506 = getelementptr { i64, ptr, i8 }, ptr %505, i32 0, i32 1
  %507 = load ptr, ptr %506, align 8
  %508 = ptrtoint ptr %507 to i64
  %509 = icmp ne i64 %508, 0
  %510 = zext i1 %509 to i32
  %511 = and i32 %510, 1
  %512 = icmp ne i32 %511, 0
  br i1 %512, label %513, label %516

513:                                              ; preds = %504
  %514 = call i32 @strcmp(ptr %507, ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8))
  %515 = icmp eq i32 %514, 0
  br label %518

516:                                              ; preds = %504
  %517 = icmp eq i64 %508, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8) to i64)
  br label %518

518:                                              ; preds = %513, %516
  %519 = phi i1 [ %517, %516 ], [ %515, %513 ]
  br label %520

520:                                              ; preds = %518
  br i1 %519, label %521, label %522

521:                                              ; preds = %520
  store ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8), ptr %3, align 8
  br label %542

522:                                              ; preds = %520
  %523 = load ptr, ptr %2, align 8
  %524 = getelementptr { i64, ptr, i8 }, ptr %523, i32 0, i32 1
  %525 = load ptr, ptr %524, align 8
  %526 = ptrtoint ptr %525 to i64
  %527 = icmp ne i64 %526, 0
  %528 = zext i1 %527 to i32
  %529 = and i32 %528, 1
  %530 = icmp ne i32 %529, 0
  br i1 %530, label %531, label %534

531:                                              ; preds = %522
  %532 = call i32 @strcmp(ptr %525, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8))
  %533 = icmp eq i32 %532, 0
  br label %536

534:                                              ; preds = %522
  %535 = icmp eq i64 %526, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8) to i64)
  br label %536

536:                                              ; preds = %531, %534
  %537 = phi i1 [ %535, %534 ], [ %533, %531 ]
  br label %538

538:                                              ; preds = %536
  br i1 %537, label %539, label %540

539:                                              ; preds = %538
  store ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3, align 8
  br label %542

540:                                              ; preds = %538
  %541 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %541, align 8
  call void @__cxa_throw(ptr %541, ptr @_ZTIPc, ptr null)
  br label %544

542:                                              ; preds = %20, %41, %64, %85, %107, %131, %155, %179, %203, %227, %250, %274, %298, %322, %347, %371, %396, %421, %446, %471, %496, %521, %539
  %543 = load ptr, ptr %3, align 8
  ret ptr %543

544:                                              ; preds = %540
  unreachable
}

; Function Attrs: mustprogress
define i64 @rethrowCaught(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca i64, align 8
  store double %0, ptr %6, align 8
  %8 = load double, ptr %6, align 8
  %9 = invoke i32 @fs(double %8)
          to label %10 unwind label %11

10:                                               ; preds = %1
  store i64 -1, ptr %7, align 8
  br label %294

11:                                               ; preds = %1
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc6
  %18 = load i32, ptr %14, align 4
  %19 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %17
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 0
  store i64 4, ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 2
  store i32 %18, ptr %23, align 4
  br label %147

24:                                               ; preds = %.noexc6
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %26
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 0
  store i64 4, ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %31, align 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %29, i32 0, i32 2
  store i32 %27, ptr %32, align 4
  br label %144

33:                                               ; preds = %24
  %34 = icmp eq ptr %15, @_ZTId
  br i1 %34, label %35, label %42

35:                                               ; preds = %33
  %36 = load double, ptr %14, align 8
  %37 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %35
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 0
  store i64 8, ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %40, align 8
  %41 = getelementptr { i64, ptr, double }, ptr %38, i32 0, i32 2
  store double %36, ptr %41, align 8
  br label %141

42:                                               ; preds = %33
  %43 = icmp eq ptr %15, @_ZTIf
  br i1 %43, label %44, label %51

44:                                               ; preds = %42
  %45 = load float, ptr %14, align 4
  %46 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %44
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 0
  store i64 4, ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %49, align 8
  %50 = getelementptr { i64, ptr, float }, ptr %47, i32 0, i32 2
  store float %45, ptr %50, align 4
  br label %138

51:                                               ; preds = %42
  %52 = icmp eq ptr %15, @_ZTIb
  br i1 %52, label %53, label %60

53:                                               ; preds = %51
  %54 = load i1, ptr %14, align 1
  %55 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %53
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 0
  store i64 1, ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %58, align 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %56, i32 0, i32 2
  store i1 %54, ptr %59, align 1
  br label %135

60:                                               ; preds = %51
  %61 = icmp eq ptr %15, @_ZTIx
  br i1 %61, label %62, label %69

62:                                               ; preds = %60
  %63 = load i64, ptr %14, align 8
  %64 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %62
  %65 = getelementptr i8, ptr %64, i64 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 0
  store i64 8, ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %67, align 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %65, i32 0, i32 2
  store i64 %63, ptr %68, align 8
  br label %132

69:                                               ; preds = %60
  %70 = icmp eq ptr %15, @_ZTIPc
  br i1 %70, label %71, label %78

71:                                               ; preds = %69
  %72 = load ptr, ptr %14, align 8
  %73 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %71
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 0
  store i64 8, ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %76, align 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %74, i32 0, i32 2
  store ptr %72, ptr %77, align 8
  br label %129

78:                                               ; preds = %69
  %79 = icmp eq ptr %15, @_ZTIPv
  br i1 %79, label %80, label %82

80:                                               ; preds = %78
  %81 = load ptr, ptr %14, align 8
  br label %126

82:                                               ; preds = %78
  %83 = load ptr, ptr %15, align 8
  %84 = icmp eq ptr %83, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %84, label %85, label %116

85:                                               ; preds = %82
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 2
  %87 = load i32, ptr %86, align 4
  %88 = and i32 %87, 1073741824
  %89 = icmp ne i32 %88, 0
  br i1 %89, label %90, label %106

90:                                               ; preds = %85
  %91 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %15, i32 0, i32 4
  %92 = load ptr, ptr %91, align 8
  %93 = icmp ne ptr %92, null
  br i1 %93, label %94, label %96

94:                                               ; preds = %90
  invoke void %92(ptr %3, ptr %14)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %94
  %95 = load ptr, ptr %3, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc21, %.noexc12
  %104 = phi ptr [ %99, %.noexc12 ], [ %95, %.noexc21 ]
  br label %105

105:                                              ; preds = %103
  br label %113

106:                                              ; preds = %85
  %107 = load ptr, ptr %14, align 8
  %108 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %106
  %109 = getelementptr i8, ptr %108, i64 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 0
  store i64 8, ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 2
  store ptr %107, ptr %112, align 8
  br label %113

113:                                              ; preds = %105, %.noexc1
  %114 = phi ptr [ %109, %.noexc1 ], [ %104, %105 ]
  br label %115

115:                                              ; preds = %113
  br label %123

116:                                              ; preds = %82
  %117 = load ptr, ptr %14, align 8
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc25
  %124 = phi ptr [ %119, %.noexc25 ], [ %114, %115 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %80, %125
  %127 = phi ptr [ %124, %125 ], [ %81, %80 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc5, %128
  %130 = phi ptr [ %127, %128 ], [ %74, %.noexc5 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc9, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc9 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc16, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc16 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc3, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc3 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc13, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc13 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc2, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc2 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc20, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc20 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %4, align 8
  %150 = load ptr, ptr %4, align 8
  call void @__cxa_end_catch()
  %151 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %150, ptr %151, align 8
  invoke void @__cxa_throw(ptr %151, ptr @_ZTIPv, ptr null)
          to label %296 unwind label %152

152:                                              ; preds = %149
  %153 = landingpad { ptr, i32 }
          catch ptr null
  %154 = extractvalue { ptr, i32 } %153, 0
  %155 = call ptr @__cxa_begin_catch(ptr %154)
  %156 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %152
  %157 = icmp eq ptr %156, @_ZTIi
  br i1 %157, label %158, label %165

158:                                              ; preds = %.noexc4
  %159 = load i32, ptr %155, align 4
  %160 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %158
  %161 = getelementptr i8, ptr %160, i64 8
  %162 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 0
  store i64 4, ptr %162, align 8
  %163 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %163, align 8
  %164 = getelementptr { i64, ptr, i32 }, ptr %161, i32 0, i32 2
  store i32 %159, ptr %164, align 4
  br label %288

165:                                              ; preds = %.noexc4
  %166 = icmp eq ptr %156, @_ZTIj
  br i1 %166, label %167, label %174

167:                                              ; preds = %165
  %168 = load i32, ptr %155, align 4
  %169 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %167
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
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %176
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
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %185
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
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %194
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
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %203
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %212
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %235
  %236 = load ptr, ptr %2, align 8
  br label %244

237:                                              ; preds = %231
  %238 = load ptr, ptr %155, align 8
  %239 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %237
  %240 = getelementptr i8, ptr %239, i64 8
  %241 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 0
  store i64 8, ptr %241, align 8
  %242 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %242, align 8
  %243 = getelementptr { i64, ptr, ptr }, ptr %240, i32 0, i32 2
  store ptr %238, ptr %243, align 8
  br label %244

244:                                              ; preds = %.noexc8, %.noexc11
  %245 = phi ptr [ %240, %.noexc11 ], [ %236, %.noexc8 ]
  br label %246

246:                                              ; preds = %244
  br label %254

247:                                              ; preds = %226
  %248 = load ptr, ptr %155, align 8
  %249 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %247
  %250 = getelementptr i8, ptr %249, i64 8
  %251 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 0
  store i64 8, ptr %251, align 8
  %252 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %252, align 8
  %253 = getelementptr { i64, ptr, ptr }, ptr %250, i32 0, i32 2
  store ptr %248, ptr %253, align 8
  br label %254

254:                                              ; preds = %246, %.noexc14
  %255 = phi ptr [ %250, %.noexc14 ], [ %245, %246 ]
  br label %256

256:                                              ; preds = %254
  br label %264

257:                                              ; preds = %223
  %258 = load ptr, ptr %155, align 8
  %259 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %257
  %260 = getelementptr i8, ptr %259, i64 8
  %261 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 0
  store i64 8, ptr %261, align 8
  %262 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %262, align 8
  %263 = getelementptr { i64, ptr, ptr }, ptr %260, i32 0, i32 2
  store ptr %258, ptr %263, align 8
  br label %264

264:                                              ; preds = %256, %.noexc19
  %265 = phi ptr [ %260, %.noexc19 ], [ %255, %256 ]
  br label %266

266:                                              ; preds = %264
  br label %267

267:                                              ; preds = %221, %266
  %268 = phi ptr [ %265, %266 ], [ %222, %221 ]
  br label %269

269:                                              ; preds = %267
  br label %270

270:                                              ; preds = %.noexc10, %269
  %271 = phi ptr [ %268, %269 ], [ %215, %.noexc10 ]
  br label %272

272:                                              ; preds = %270
  br label %273

273:                                              ; preds = %.noexc18, %272
  %274 = phi ptr [ %271, %272 ], [ %206, %.noexc18 ]
  br label %275

275:                                              ; preds = %273
  br label %276

276:                                              ; preds = %.noexc24, %275
  %277 = phi ptr [ %274, %275 ], [ %197, %.noexc24 ]
  br label %278

278:                                              ; preds = %276
  br label %279

279:                                              ; preds = %.noexc17, %278
  %280 = phi ptr [ %277, %278 ], [ %188, %.noexc17 ]
  br label %281

281:                                              ; preds = %279
  br label %282

282:                                              ; preds = %.noexc15, %281
  %283 = phi ptr [ %280, %281 ], [ %179, %.noexc15 ]
  br label %284

284:                                              ; preds = %282
  br label %285

285:                                              ; preds = %.noexc7, %284
  %286 = phi ptr [ %283, %284 ], [ %170, %.noexc7 ]
  br label %287

287:                                              ; preds = %285
  br label %288

288:                                              ; preds = %.noexc, %287
  %289 = phi ptr [ %286, %287 ], [ %161, %.noexc ]
  br label %290

290:                                              ; preds = %288
  store ptr %289, ptr %5, align 8
  %291 = load ptr, ptr %5, align 8
  %292 = invoke ptr @"___unbox<!ts.string>"(ptr %291)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %290
  %293 = invoke i64 @strlen(ptr %292)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %.noexc22
  store i64 %293, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %294

294:                                              ; preds = %10, %.noexc23
  %295 = load i64, ptr %7, align 8
  ret i64 %295

296:                                              ; preds = %149
  unreachable

endcatch.resume:                                  ; preds = %116, %194, %.noexc22, %290, %94, %17, %257, %203, %185, %53, %176, %247, %35, %96, %237, %212, %62, %235, %167, %11, %71, %152, %44, %26, %106, %158
  %297 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %297
}

; Function Attrs: mustprogress
define double @usingOut(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  store double %0, ptr %6, align 8
  store i1 false, ptr %3, align 1
  %8 = call ptr @GC_malloc(i64 16)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  store ptr %9, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %11 = load double, ptr %6, align 8
  %12 = invoke i32 @f(double %11)
          to label %13 unwind label %20

13:                                               ; preds = %1
  %14 = load i1, ptr %3, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %13
  store i1 false, ptr %3, align 1
  %16 = load ptr, ptr %4, align 8
  %17 = getelementptr %Res, ptr %16, i32 0, i32 0
  invoke void @Res.dispose(ptr %16)
          to label %18 unwind label %20

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %13
  store double -1.000000e+00, ptr %7, align 8
  br label %184

20:                                               ; preds = %15, %1
  %21 = landingpad { ptr, i32 }
          catch ptr null
  %22 = extractvalue { ptr, i32 } %21, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = load i1, ptr %3, align 1
  br i1 %24, label %25, label %29

25:                                               ; preds = %20
  store i1 false, ptr %3, align 1
  %26 = load ptr, ptr %4, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %28 unwind label %endcatch.pad2

28:                                               ; preds = %25
  br label %29

29:                                               ; preds = %28, %20
  invoke void @__cxa_rethrow()
          to label %186 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %25
  %30 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

31:                                               ; No predecessors!
  %32 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %31
  %33 = phi { ptr, i32 } [ %32, %31 ], [ %30, %endcatch.pad2 ]
  %34 = extractvalue { ptr, i32 } %33, 0
  %35 = call ptr @__cxa_begin_catch(ptr %34)
  %36 = load i1, ptr %3, align 1
  br i1 %36, label %37, label %40

37:                                               ; preds = %.caught1
  store i1 false, ptr %3, align 1
  %38 = load ptr, ptr %4, align 8
  %39 = getelementptr %Res, ptr %38, i32 0, i32 0
  invoke void @Res.dispose(ptr %38)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %37
  br label %40

40:                                               ; preds = %.noexc12, %.caught1
  invoke void @__cxa_rethrow()
          to label %186 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %29, %40
  %41 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

42:                                               ; No predecessors!
  %43 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %42
  %44 = phi { ptr, i32 } [ %43, %42 ], [ %41, %endcatch.pad ]
  %45 = extractvalue { ptr, i32 } %44, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  %47 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.caught
  %48 = icmp eq ptr %47, @_ZTIi
  br i1 %48, label %49, label %56

49:                                               ; preds = %.noexc8
  %50 = load i32, ptr %46, align 4
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, i32 }, ptr %52, i32 0, i32 0
  store i64 4, ptr %53, align 8
  %54 = getelementptr { i64, ptr, i32 }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, i32 }, ptr %52, i32 0, i32 2
  store i32 %50, ptr %55, align 4
  br label %179

56:                                               ; preds = %.noexc8
  %57 = icmp eq ptr %47, @_ZTIj
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load i32, ptr %46, align 4
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, i32 }, ptr %61, i32 0, i32 0
  store i64 4, ptr %62, align 8
  %63 = getelementptr { i64, ptr, i32 }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, i32 }, ptr %61, i32 0, i32 2
  store i32 %59, ptr %64, align 4
  br label %176

65:                                               ; preds = %56
  %66 = icmp eq ptr %47, @_ZTId
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load double, ptr %46, align 8
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, double }, ptr %70, i32 0, i32 0
  store i64 8, ptr %71, align 8
  %72 = getelementptr { i64, ptr, double }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, double }, ptr %70, i32 0, i32 2
  store double %68, ptr %73, align 8
  br label %173

74:                                               ; preds = %65
  %75 = icmp eq ptr %47, @_ZTIf
  br i1 %75, label %76, label %83

76:                                               ; preds = %74
  %77 = load float, ptr %46, align 4
  %78 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %76
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr { i64, ptr, float }, ptr %79, i32 0, i32 0
  store i64 4, ptr %80, align 8
  %81 = getelementptr { i64, ptr, float }, ptr %79, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %81, align 8
  %82 = getelementptr { i64, ptr, float }, ptr %79, i32 0, i32 2
  store float %77, ptr %82, align 4
  br label %170

83:                                               ; preds = %74
  %84 = icmp eq ptr %47, @_ZTIb
  br i1 %84, label %85, label %92

85:                                               ; preds = %83
  %86 = load i1, ptr %46, align 1
  %87 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %85
  %88 = getelementptr i8, ptr %87, i64 8
  %89 = getelementptr { i64, ptr, i1 }, ptr %88, i32 0, i32 0
  store i64 1, ptr %89, align 8
  %90 = getelementptr { i64, ptr, i1 }, ptr %88, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %90, align 8
  %91 = getelementptr { i64, ptr, i1 }, ptr %88, i32 0, i32 2
  store i1 %86, ptr %91, align 1
  br label %167

92:                                               ; preds = %83
  %93 = icmp eq ptr %47, @_ZTIx
  br i1 %93, label %94, label %101

94:                                               ; preds = %92
  %95 = load i64, ptr %46, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, i64 }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, i64 }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, i64 }, ptr %97, i32 0, i32 2
  store i64 %95, ptr %100, align 8
  br label %164

101:                                              ; preds = %92
  %102 = icmp eq ptr %47, @_ZTIPc
  br i1 %102, label %103, label %110

103:                                              ; preds = %101
  %104 = load ptr, ptr %46, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %161

110:                                              ; preds = %101
  %111 = icmp eq ptr %47, @_ZTIPv
  br i1 %111, label %112, label %114

112:                                              ; preds = %110
  %113 = load ptr, ptr %46, align 8
  br label %158

114:                                              ; preds = %110
  %115 = load ptr, ptr %47, align 8
  %116 = icmp eq ptr %115, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %116, label %117, label %148

117:                                              ; preds = %114
  %118 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %47, i32 0, i32 2
  %119 = load i32, ptr %118, align 4
  %120 = and i32 %119, 1073741824
  %121 = icmp ne i32 %120, 0
  br i1 %121, label %122, label %138

122:                                              ; preds = %117
  %123 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %47, i32 0, i32 4
  %124 = load ptr, ptr %123, align 8
  %125 = icmp ne ptr %124, null
  br i1 %125, label %126, label %128

126:                                              ; preds = %122
  invoke void %124(ptr %2, ptr %46)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %126
  %127 = load ptr, ptr %2, align 8
  br label %135

128:                                              ; preds = %122
  %129 = load ptr, ptr %46, align 8
  %130 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %128
  %131 = getelementptr i8, ptr %130, i64 8
  %132 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 0
  store i64 8, ptr %132, align 8
  %133 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %133, align 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 2
  store ptr %129, ptr %134, align 8
  br label %135

135:                                              ; preds = %.noexc3, %.noexc5
  %136 = phi ptr [ %131, %.noexc5 ], [ %127, %.noexc3 ]
  br label %137

137:                                              ; preds = %135
  br label %145

138:                                              ; preds = %117
  %139 = load ptr, ptr %46, align 8
  %140 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %138
  %141 = getelementptr i8, ptr %140, i64 8
  %142 = getelementptr { i64, ptr, ptr }, ptr %141, i32 0, i32 0
  store i64 8, ptr %142, align 8
  %143 = getelementptr { i64, ptr, ptr }, ptr %141, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %143, align 8
  %144 = getelementptr { i64, ptr, ptr }, ptr %141, i32 0, i32 2
  store ptr %139, ptr %144, align 8
  br label %145

145:                                              ; preds = %137, %.noexc4
  %146 = phi ptr [ %141, %.noexc4 ], [ %136, %137 ]
  br label %147

147:                                              ; preds = %145
  br label %155

148:                                              ; preds = %114
  %149 = load ptr, ptr %46, align 8
  %150 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %148
  %151 = getelementptr i8, ptr %150, i64 8
  %152 = getelementptr { i64, ptr, ptr }, ptr %151, i32 0, i32 0
  store i64 8, ptr %152, align 8
  %153 = getelementptr { i64, ptr, ptr }, ptr %151, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %153, align 8
  %154 = getelementptr { i64, ptr, ptr }, ptr %151, i32 0, i32 2
  store ptr %149, ptr %154, align 8
  br label %155

155:                                              ; preds = %147, %.noexc15
  %156 = phi ptr [ %151, %.noexc15 ], [ %146, %147 ]
  br label %157

157:                                              ; preds = %155
  br label %158

158:                                              ; preds = %112, %157
  %159 = phi ptr [ %156, %157 ], [ %113, %112 ]
  br label %160

160:                                              ; preds = %158
  br label %161

161:                                              ; preds = %.noexc, %160
  %162 = phi ptr [ %159, %160 ], [ %106, %.noexc ]
  br label %163

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %.noexc9, %163
  %165 = phi ptr [ %162, %163 ], [ %97, %.noexc9 ]
  br label %166

166:                                              ; preds = %164
  br label %167

167:                                              ; preds = %.noexc14, %166
  %168 = phi ptr [ %165, %166 ], [ %88, %.noexc14 ]
  br label %169

169:                                              ; preds = %167
  br label %170

170:                                              ; preds = %.noexc7, %169
  %171 = phi ptr [ %168, %169 ], [ %79, %.noexc7 ]
  br label %172

172:                                              ; preds = %170
  br label %173

173:                                              ; preds = %.noexc13, %172
  %174 = phi ptr [ %171, %172 ], [ %70, %.noexc13 ]
  br label %175

175:                                              ; preds = %173
  br label %176

176:                                              ; preds = %.noexc6, %175
  %177 = phi ptr [ %174, %175 ], [ %61, %.noexc6 ]
  br label %178

178:                                              ; preds = %176
  br label %179

179:                                              ; preds = %.noexc11, %178
  %180 = phi ptr [ %177, %178 ], [ %52, %.noexc11 ]
  br label %181

181:                                              ; preds = %179
  store ptr %180, ptr %5, align 8
  %182 = load ptr, ptr %5, align 8
  %183 = invoke double @"___unbox<!ts.number>"(ptr %182)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %181
  store double %183, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %184

184:                                              ; preds = %19, %.noexc10
  %185 = load double, ptr %7, align 8
  ret double %185

186:                                              ; preds = %29, %40
  unreachable

endcatch.resume:                                  ; preds = %148, %85, %67, %37, %49, %181, %94, %.caught, %76, %58, %128, %138, %126, %103
  %187 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %187
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca i32, align 4
  %7 = alloca double, align 8
  store double 0.000000e+00, ptr %7, align 8
  store i32 0, ptr %6, align 4
  br label %8

8:                                                ; preds = %464, %0
  %9 = load i32, ptr %6, align 4
  %10 = icmp slt i32 %9, 20000
  br i1 %10, label %11, label %470

11:                                               ; preds = %8
  store double 0.000000e+00, ptr %5, align 8
  %12 = load double, ptr %5, align 8
  %13 = load i32, ptr %6, align 4
  %14 = sitofp i32 %13 to double
  %15 = call double @callInCatch(double %14)
  %16 = load i32, ptr %6, align 4
  %17 = add i32 %16, 1
  %18 = sitofp i32 %17 to double
  %19 = fsub double %15, %18
  %20 = fadd double %12, %19
  store double %20, ptr %5, align 8
  %21 = load double, ptr %5, align 8
  %22 = load i32, ptr %6, align 4
  %23 = sitofp i32 %22 to double
  %24 = call double @finallyRethrow(double %23)
  %25 = load i32, ptr %6, align 4
  %26 = sitofp i32 %25 to double
  %27 = fsub double %24, %26
  %28 = fadd double %21, %27
  store double %28, ptr %5, align 8
  %29 = load double, ptr %5, align 8
  %30 = load i32, ptr %6, align 4
  %31 = sitofp i32 %30 to double
  %32 = call double @mismatch(double %31)
  %33 = load i32, ptr %6, align 4
  %34 = sitofp i32 %33 to double
  %35 = fsub double %32, %34
  %36 = fadd double %29, %35
  store double %36, ptr %5, align 8
  %37 = load i32, ptr %6, align 4
  %38 = sitofp i32 %37 to double
  %39 = invoke i32 @mismatchOut(double %38)
          to label %40 unwind label %43

40:                                               ; preds = %11
  %41 = load double, ptr %5, align 8
  %42 = fadd double %41, 1.000000e+00
  store double %42, ptr %5, align 8
  br label %189

43:                                               ; preds = %11
  %44 = landingpad { ptr, i32 }
          catch ptr null
  %45 = extractvalue { ptr, i32 } %44, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  %47 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %43
  %48 = icmp eq ptr %47, @_ZTIi
  br i1 %48, label %49, label %56

49:                                               ; preds = %.noexc20
  %50 = load i32, ptr %46, align 4
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, i32 }, ptr %52, i32 0, i32 0
  store i64 4, ptr %53, align 8
  %54 = getelementptr { i64, ptr, i32 }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, i32 }, ptr %52, i32 0, i32 2
  store i32 %50, ptr %55, align 4
  br label %179

56:                                               ; preds = %.noexc20
  %57 = icmp eq ptr %47, @_ZTIj
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load i32, ptr %46, align 4
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, i32 }, ptr %61, i32 0, i32 0
  store i64 4, ptr %62, align 8
  %63 = getelementptr { i64, ptr, i32 }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, i32 }, ptr %61, i32 0, i32 2
  store i32 %59, ptr %64, align 4
  br label %176

65:                                               ; preds = %56
  %66 = icmp eq ptr %47, @_ZTId
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load double, ptr %46, align 8
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, double }, ptr %70, i32 0, i32 0
  store i64 8, ptr %71, align 8
  %72 = getelementptr { i64, ptr, double }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, double }, ptr %70, i32 0, i32 2
  store double %68, ptr %73, align 8
  br label %173

74:                                               ; preds = %65
  %75 = icmp eq ptr %47, @_ZTIf
  br i1 %75, label %76, label %83

76:                                               ; preds = %74
  %77 = load float, ptr %46, align 4
  %78 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %76
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr { i64, ptr, float }, ptr %79, i32 0, i32 0
  store i64 4, ptr %80, align 8
  %81 = getelementptr { i64, ptr, float }, ptr %79, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %81, align 8
  %82 = getelementptr { i64, ptr, float }, ptr %79, i32 0, i32 2
  store float %77, ptr %82, align 4
  br label %170

83:                                               ; preds = %74
  %84 = icmp eq ptr %47, @_ZTIb
  br i1 %84, label %85, label %92

85:                                               ; preds = %83
  %86 = load i1, ptr %46, align 1
  %87 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %85
  %88 = getelementptr i8, ptr %87, i64 8
  %89 = getelementptr { i64, ptr, i1 }, ptr %88, i32 0, i32 0
  store i64 1, ptr %89, align 8
  %90 = getelementptr { i64, ptr, i1 }, ptr %88, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %90, align 8
  %91 = getelementptr { i64, ptr, i1 }, ptr %88, i32 0, i32 2
  store i1 %86, ptr %91, align 1
  br label %167

92:                                               ; preds = %83
  %93 = icmp eq ptr %47, @_ZTIx
  br i1 %93, label %94, label %101

94:                                               ; preds = %92
  %95 = load i64, ptr %46, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, i64 }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, i64 }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, i64 }, ptr %97, i32 0, i32 2
  store i64 %95, ptr %100, align 8
  br label %164

101:                                              ; preds = %92
  %102 = icmp eq ptr %47, @_ZTIPc
  br i1 %102, label %103, label %110

103:                                              ; preds = %101
  %104 = load ptr, ptr %46, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %161

110:                                              ; preds = %101
  %111 = icmp eq ptr %47, @_ZTIPv
  br i1 %111, label %112, label %114

112:                                              ; preds = %110
  %113 = load ptr, ptr %46, align 8
  br label %158

114:                                              ; preds = %110
  %115 = load ptr, ptr %47, align 8
  %116 = icmp eq ptr %115, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %116, label %117, label %148

117:                                              ; preds = %114
  %118 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %47, i32 0, i32 2
  %119 = load i32, ptr %118, align 4
  %120 = and i32 %119, 1073741824
  %121 = icmp ne i32 %120, 0
  br i1 %121, label %122, label %138

122:                                              ; preds = %117
  %123 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %47, i32 0, i32 4
  %124 = load ptr, ptr %123, align 8
  %125 = icmp ne ptr %124, null
  br i1 %125, label %126, label %128

126:                                              ; preds = %122
  invoke void %124(ptr %1, ptr %46)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %126
  %127 = load ptr, ptr %1, align 8
  br label %135

128:                                              ; preds = %122
  %129 = load ptr, ptr %46, align 8
  %130 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %128
  %131 = getelementptr i8, ptr %130, i64 8
  %132 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 0
  store i64 8, ptr %132, align 8
  %133 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %133, align 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %131, i32 0, i32 2
  store ptr %129, ptr %134, align 8
  br label %135

135:                                              ; preds = %.noexc2, %.noexc17
  %136 = phi ptr [ %131, %.noexc17 ], [ %127, %.noexc2 ]
  br label %137

137:                                              ; preds = %135
  br label %145

138:                                              ; preds = %117
  %139 = load ptr, ptr %46, align 8
  %140 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %138
  %141 = getelementptr i8, ptr %140, i64 8
  %142 = getelementptr { i64, ptr, ptr }, ptr %141, i32 0, i32 0
  store i64 8, ptr %142, align 8
  %143 = getelementptr { i64, ptr, ptr }, ptr %141, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %143, align 8
  %144 = getelementptr { i64, ptr, ptr }, ptr %141, i32 0, i32 2
  store ptr %139, ptr %144, align 8
  br label %145

145:                                              ; preds = %137, %.noexc15
  %146 = phi ptr [ %141, %.noexc15 ], [ %136, %137 ]
  br label %147

147:                                              ; preds = %145
  br label %155

148:                                              ; preds = %114
  %149 = load ptr, ptr %46, align 8
  %150 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %148
  %151 = getelementptr i8, ptr %150, i64 8
  %152 = getelementptr { i64, ptr, ptr }, ptr %151, i32 0, i32 0
  store i64 8, ptr %152, align 8
  %153 = getelementptr { i64, ptr, ptr }, ptr %151, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %153, align 8
  %154 = getelementptr { i64, ptr, ptr }, ptr %151, i32 0, i32 2
  store ptr %149, ptr %154, align 8
  br label %155

155:                                              ; preds = %147, %.noexc18
  %156 = phi ptr [ %151, %.noexc18 ], [ %146, %147 ]
  br label %157

157:                                              ; preds = %155
  br label %158

158:                                              ; preds = %112, %157
  %159 = phi ptr [ %156, %157 ], [ %113, %112 ]
  br label %160

160:                                              ; preds = %158
  br label %161

161:                                              ; preds = %.noexc, %160
  %162 = phi ptr [ %159, %160 ], [ %106, %.noexc ]
  br label %163

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %.noexc25, %163
  %165 = phi ptr [ %162, %163 ], [ %97, %.noexc25 ]
  br label %166

166:                                              ; preds = %164
  br label %167

167:                                              ; preds = %.noexc8, %166
  %168 = phi ptr [ %165, %166 ], [ %88, %.noexc8 ]
  br label %169

169:                                              ; preds = %167
  br label %170

170:                                              ; preds = %.noexc24, %169
  %171 = phi ptr [ %168, %169 ], [ %79, %.noexc24 ]
  br label %172

172:                                              ; preds = %170
  br label %173

173:                                              ; preds = %.noexc9, %172
  %174 = phi ptr [ %171, %172 ], [ %70, %.noexc9 ]
  br label %175

175:                                              ; preds = %173
  br label %176

176:                                              ; preds = %.noexc16, %175
  %177 = phi ptr [ %174, %175 ], [ %61, %.noexc16 ]
  br label %178

178:                                              ; preds = %176
  br label %179

179:                                              ; preds = %.noexc22, %178
  %180 = phi ptr [ %177, %178 ], [ %52, %.noexc22 ]
  br label %181

181:                                              ; preds = %179
  store ptr %180, ptr %4, align 8
  %182 = load double, ptr %5, align 8
  %183 = load ptr, ptr %4, align 8
  %184 = invoke double @"___unbox<!ts.number>"(ptr %183)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %181
  %185 = load i32, ptr %6, align 4
  %186 = sitofp i32 %185 to double
  %187 = fsub double %184, %186
  %188 = fadd double %182, %187
  store double %188, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %189

189:                                              ; preds = %40, %.noexc5
  %190 = load double, ptr %5, align 8
  %191 = load i32, ptr %6, align 4
  %192 = sitofp i32 %191 to double
  %193 = call double @callInCatchFinally(double %192)
  %194 = load i32, ptr %6, align 4
  %195 = add i32 %194, 3
  %196 = sitofp i32 %195 to double
  %197 = fsub double %193, %196
  %198 = fadd double %190, %197
  store double %198, ptr %5, align 8
  %199 = load i32, ptr %6, align 4
  %200 = sitofp i32 %199 to double
  %201 = invoke i32 @callInCatchFinallyOut(double %200)
          to label %202 unwind label %205

202:                                              ; preds = %189
  %203 = load double, ptr %5, align 8
  %204 = fadd double %203, 1.000000e+00
  store double %204, ptr %5, align 8
  br label %352

205:                                              ; preds = %189
  %206 = landingpad { ptr, i32 }
          catch ptr null
  %207 = extractvalue { ptr, i32 } %206, 0
  %208 = call ptr @__cxa_begin_catch(ptr %207)
  %209 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %205
  %210 = icmp eq ptr %209, @_ZTIi
  br i1 %210, label %211, label %218

211:                                              ; preds = %.noexc21
  %212 = load i32, ptr %208, align 4
  %213 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %211
  %214 = getelementptr i8, ptr %213, i64 8
  %215 = getelementptr { i64, ptr, i32 }, ptr %214, i32 0, i32 0
  store i64 4, ptr %215, align 8
  %216 = getelementptr { i64, ptr, i32 }, ptr %214, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %216, align 8
  %217 = getelementptr { i64, ptr, i32 }, ptr %214, i32 0, i32 2
  store i32 %212, ptr %217, align 4
  br label %341

218:                                              ; preds = %.noexc21
  %219 = icmp eq ptr %209, @_ZTIj
  br i1 %219, label %220, label %227

220:                                              ; preds = %218
  %221 = load i32, ptr %208, align 4
  %222 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %220
  %223 = getelementptr i8, ptr %222, i64 8
  %224 = getelementptr { i64, ptr, i32 }, ptr %223, i32 0, i32 0
  store i64 4, ptr %224, align 8
  %225 = getelementptr { i64, ptr, i32 }, ptr %223, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %225, align 8
  %226 = getelementptr { i64, ptr, i32 }, ptr %223, i32 0, i32 2
  store i32 %221, ptr %226, align 4
  br label %338

227:                                              ; preds = %218
  %228 = icmp eq ptr %209, @_ZTId
  br i1 %228, label %229, label %236

229:                                              ; preds = %227
  %230 = load double, ptr %208, align 8
  %231 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %229
  %232 = getelementptr i8, ptr %231, i64 8
  %233 = getelementptr { i64, ptr, double }, ptr %232, i32 0, i32 0
  store i64 8, ptr %233, align 8
  %234 = getelementptr { i64, ptr, double }, ptr %232, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %234, align 8
  %235 = getelementptr { i64, ptr, double }, ptr %232, i32 0, i32 2
  store double %230, ptr %235, align 8
  br label %335

236:                                              ; preds = %227
  %237 = icmp eq ptr %209, @_ZTIf
  br i1 %237, label %238, label %245

238:                                              ; preds = %236
  %239 = load float, ptr %208, align 4
  %240 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %238
  %241 = getelementptr i8, ptr %240, i64 8
  %242 = getelementptr { i64, ptr, float }, ptr %241, i32 0, i32 0
  store i64 4, ptr %242, align 8
  %243 = getelementptr { i64, ptr, float }, ptr %241, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %243, align 8
  %244 = getelementptr { i64, ptr, float }, ptr %241, i32 0, i32 2
  store float %239, ptr %244, align 4
  br label %332

245:                                              ; preds = %236
  %246 = icmp eq ptr %209, @_ZTIb
  br i1 %246, label %247, label %254

247:                                              ; preds = %245
  %248 = load i1, ptr %208, align 1
  %249 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %247
  %250 = getelementptr i8, ptr %249, i64 8
  %251 = getelementptr { i64, ptr, i1 }, ptr %250, i32 0, i32 0
  store i64 1, ptr %251, align 8
  %252 = getelementptr { i64, ptr, i1 }, ptr %250, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %252, align 8
  %253 = getelementptr { i64, ptr, i1 }, ptr %250, i32 0, i32 2
  store i1 %248, ptr %253, align 1
  br label %329

254:                                              ; preds = %245
  %255 = icmp eq ptr %209, @_ZTIx
  br i1 %255, label %256, label %263

256:                                              ; preds = %254
  %257 = load i64, ptr %208, align 8
  %258 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %256
  %259 = getelementptr i8, ptr %258, i64 8
  %260 = getelementptr { i64, ptr, i64 }, ptr %259, i32 0, i32 0
  store i64 8, ptr %260, align 8
  %261 = getelementptr { i64, ptr, i64 }, ptr %259, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %261, align 8
  %262 = getelementptr { i64, ptr, i64 }, ptr %259, i32 0, i32 2
  store i64 %257, ptr %262, align 8
  br label %326

263:                                              ; preds = %254
  %264 = icmp eq ptr %209, @_ZTIPc
  br i1 %264, label %265, label %272

265:                                              ; preds = %263
  %266 = load ptr, ptr %208, align 8
  %267 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %265
  %268 = getelementptr i8, ptr %267, i64 8
  %269 = getelementptr { i64, ptr, ptr }, ptr %268, i32 0, i32 0
  store i64 8, ptr %269, align 8
  %270 = getelementptr { i64, ptr, ptr }, ptr %268, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %270, align 8
  %271 = getelementptr { i64, ptr, ptr }, ptr %268, i32 0, i32 2
  store ptr %266, ptr %271, align 8
  br label %323

272:                                              ; preds = %263
  %273 = icmp eq ptr %209, @_ZTIPv
  br i1 %273, label %274, label %276

274:                                              ; preds = %272
  %275 = load ptr, ptr %208, align 8
  br label %320

276:                                              ; preds = %272
  %277 = load ptr, ptr %209, align 8
  %278 = icmp eq ptr %277, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %278, label %279, label %310

279:                                              ; preds = %276
  %280 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %209, i32 0, i32 2
  %281 = load i32, ptr %280, align 4
  %282 = and i32 %281, 1073741824
  %283 = icmp ne i32 %282, 0
  br i1 %283, label %284, label %300

284:                                              ; preds = %279
  %285 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %209, i32 0, i32 4
  %286 = load ptr, ptr %285, align 8
  %287 = icmp ne ptr %286, null
  br i1 %287, label %288, label %290

288:                                              ; preds = %284
  invoke void %286(ptr %2, ptr %208)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %288
  %289 = load ptr, ptr %2, align 8
  br label %297

290:                                              ; preds = %284
  %291 = load ptr, ptr %208, align 8
  %292 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %290
  %293 = getelementptr i8, ptr %292, i64 8
  %294 = getelementptr { i64, ptr, ptr }, ptr %293, i32 0, i32 0
  store i64 8, ptr %294, align 8
  %295 = getelementptr { i64, ptr, ptr }, ptr %293, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %295, align 8
  %296 = getelementptr { i64, ptr, ptr }, ptr %293, i32 0, i32 2
  store ptr %291, ptr %296, align 8
  br label %297

297:                                              ; preds = %.noexc1, %.noexc3
  %298 = phi ptr [ %293, %.noexc3 ], [ %289, %.noexc1 ]
  br label %299

299:                                              ; preds = %297
  br label %307

300:                                              ; preds = %279
  %301 = load ptr, ptr %208, align 8
  %302 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %300
  %303 = getelementptr i8, ptr %302, i64 8
  %304 = getelementptr { i64, ptr, ptr }, ptr %303, i32 0, i32 0
  store i64 8, ptr %304, align 8
  %305 = getelementptr { i64, ptr, ptr }, ptr %303, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %305, align 8
  %306 = getelementptr { i64, ptr, ptr }, ptr %303, i32 0, i32 2
  store ptr %301, ptr %306, align 8
  br label %307

307:                                              ; preds = %299, %.noexc14
  %308 = phi ptr [ %303, %.noexc14 ], [ %298, %299 ]
  br label %309

309:                                              ; preds = %307
  br label %317

310:                                              ; preds = %276
  %311 = load ptr, ptr %208, align 8
  %312 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %310
  %313 = getelementptr i8, ptr %312, i64 8
  %314 = getelementptr { i64, ptr, ptr }, ptr %313, i32 0, i32 0
  store i64 8, ptr %314, align 8
  %315 = getelementptr { i64, ptr, ptr }, ptr %313, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %315, align 8
  %316 = getelementptr { i64, ptr, ptr }, ptr %313, i32 0, i32 2
  store ptr %311, ptr %316, align 8
  br label %317

317:                                              ; preds = %309, %.noexc12
  %318 = phi ptr [ %313, %.noexc12 ], [ %308, %309 ]
  br label %319

319:                                              ; preds = %317
  br label %320

320:                                              ; preds = %274, %319
  %321 = phi ptr [ %318, %319 ], [ %275, %274 ]
  br label %322

322:                                              ; preds = %320
  br label %323

323:                                              ; preds = %.noexc7, %322
  %324 = phi ptr [ %321, %322 ], [ %268, %.noexc7 ]
  br label %325

325:                                              ; preds = %323
  br label %326

326:                                              ; preds = %.noexc6, %325
  %327 = phi ptr [ %324, %325 ], [ %259, %.noexc6 ]
  br label %328

328:                                              ; preds = %326
  br label %329

329:                                              ; preds = %.noexc10, %328
  %330 = phi ptr [ %327, %328 ], [ %250, %.noexc10 ]
  br label %331

331:                                              ; preds = %329
  br label %332

332:                                              ; preds = %.noexc11, %331
  %333 = phi ptr [ %330, %331 ], [ %241, %.noexc11 ]
  br label %334

334:                                              ; preds = %332
  br label %335

335:                                              ; preds = %.noexc13, %334
  %336 = phi ptr [ %333, %334 ], [ %232, %.noexc13 ]
  br label %337

337:                                              ; preds = %335
  br label %338

338:                                              ; preds = %.noexc4, %337
  %339 = phi ptr [ %336, %337 ], [ %223, %.noexc4 ]
  br label %340

340:                                              ; preds = %338
  br label %341

341:                                              ; preds = %.noexc19, %340
  %342 = phi ptr [ %339, %340 ], [ %214, %.noexc19 ]
  br label %343

343:                                              ; preds = %341
  store ptr %342, ptr %3, align 8
  %344 = load double, ptr %5, align 8
  %345 = load ptr, ptr %3, align 8
  %346 = invoke double @"___unbox<!ts.number>"(ptr %345)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %343
  %347 = load i32, ptr %6, align 4
  %348 = add i32 %347, 4
  %349 = sitofp i32 %348 to double
  %350 = fsub double %346, %349
  %351 = fadd double %344, %350
  store double %351, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %352

352:                                              ; preds = %202, %.noexc23
  %353 = load double, ptr %5, align 8
  %354 = load i32, ptr %6, align 4
  %355 = sitofp i32 %354 to double
  %356 = call double @nestedInCatch(double %355)
  %357 = load i32, ptr %6, align 4
  %358 = add i32 %357, 5
  %359 = sitofp i32 %358 to double
  %360 = fsub double %356, %359
  %361 = fadd double %353, %360
  store double %361, ptr %5, align 8
  %362 = load double, ptr %5, align 8
  %363 = load i32, ptr %6, align 4
  %364 = sitofp i32 %363 to double
  %365 = call double @throwTakenInCatch(double %364)
  %366 = load i32, ptr %6, align 4
  %367 = add i32 %366, 7
  %368 = sitofp i32 %367 to double
  %369 = fsub double %365, %368
  %370 = fadd double %362, %369
  store double %370, ptr %5, align 8
  %371 = load double, ptr %5, align 8
  %372 = load i32, ptr %6, align 4
  %373 = sitofp i32 %372 to double
  %374 = call i64 @rethrowCaught(double %373)
  %375 = load i32, ptr %6, align 4
  %376 = call ptr @GC_malloc_atomic(i64 58)
  %377 = getelementptr i8, ptr %376, i64 8
  %378 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %377, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %375)
  %379 = icmp eq ptr %377, null
  %380 = select i1 %379, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %377
  %381 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8))
  %382 = add i64 1, %381
  %383 = call i64 @strlen(ptr %380)
  %384 = add i64 %382, %383
  %385 = add i64 %384, 8
  %386 = call ptr @GC_malloc(i64 %385)
  %387 = getelementptr i8, ptr %386, i64 8
  %388 = getelementptr i8, ptr %387, i64 0
  %389 = sub i64 %384, 0
  %390 = call ptr @__memcpy_chk(ptr %388, ptr getelementptr inbounds nuw (i8, ptr @s_12638206991764949794, i64 8), i64 %381, i64 %389)
  %391 = add i64 0, %381
  %392 = getelementptr i8, ptr %387, i64 %391
  %393 = sub i64 %384, %391
  %394 = call ptr @__memcpy_chk(ptr %392, ptr %380, i64 %383, i64 %393)
  %395 = add i64 %391, %383
  %396 = getelementptr i8, ptr %387, i64 %395
  store i8 0, ptr %396, align 1
  %397 = call i64 @strlen(ptr %387)
  %398 = sub i64 %374, %397
  %399 = sitofp i64 %398 to double
  %400 = fadd double %371, %399
  store double %400, ptr %5, align 8
  %401 = load i32, ptr @disposed, align 4
  %402 = load double, ptr %5, align 8
  %403 = load i32, ptr %6, align 4
  %404 = sitofp i32 %403 to double
  %405 = call double @usingOut(double %404)
  %406 = load i32, ptr %6, align 4
  %407 = sitofp i32 %406 to double
  %408 = fsub double %405, %407
  %409 = fadd double %402, %408
  store double %409, ptr %5, align 8
  %410 = load double, ptr %5, align 8
  %411 = load i32, ptr @disposed, align 4
  %412 = sub i32 %411, %401
  %413 = sub i32 %412, 1
  %414 = sitofp i32 %413 to double
  %415 = fadd double %410, %414
  store double %415, ptr %5, align 8
  %416 = load double, ptr %5, align 8
  %417 = fcmp une double %416, 0.000000e+00
  br i1 %417, label %418, label %464

418:                                              ; preds = %352
  %419 = load i32, ptr %6, align 4
  %420 = load double, ptr %5, align 8
  %421 = call ptr @GC_malloc_atomic(i64 58)
  %422 = getelementptr i8, ptr %421, i64 8
  %423 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %422, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %419)
  %424 = call ptr @GC_malloc_atomic(i64 40)
  %425 = getelementptr i8, ptr %424, i64 8
  call void @__ts_number_to_string(ptr %425, double %420)
  %426 = call ptr @llvm.stacksave.p0()
  %427 = icmp eq ptr %422, null
  %428 = select i1 %427, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %422
  %429 = icmp eq ptr %425, null
  %430 = select i1 %429, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %425
  %431 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_16852112993072541847, i64 8))
  %432 = add i64 1, %431
  %433 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %434 = add i64 %432, %433
  %435 = call i64 @strlen(ptr %428)
  %436 = add i64 %434, %435
  %437 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %438 = add i64 %436, %437
  %439 = call i64 @strlen(ptr %430)
  %440 = add i64 %438, %439
  %441 = alloca i8, i64 %440, align 1
  %442 = getelementptr i8, ptr %441, i64 0
  %443 = sub i64 %440, 0
  %444 = call ptr @__memcpy_chk(ptr %442, ptr getelementptr inbounds nuw (i8, ptr @s_16852112993072541847, i64 8), i64 %431, i64 %443)
  %445 = add i64 0, %431
  %446 = getelementptr i8, ptr %441, i64 %445
  %447 = sub i64 %440, %445
  %448 = call ptr @__memcpy_chk(ptr %446, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %433, i64 %447)
  %449 = add i64 %445, %433
  %450 = getelementptr i8, ptr %441, i64 %449
  %451 = sub i64 %440, %449
  %452 = call ptr @__memcpy_chk(ptr %450, ptr %428, i64 %435, i64 %451)
  %453 = add i64 %449, %435
  %454 = getelementptr i8, ptr %441, i64 %453
  %455 = sub i64 %440, %453
  %456 = call ptr @__memcpy_chk(ptr %454, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %437, i64 %455)
  %457 = add i64 %453, %437
  %458 = getelementptr i8, ptr %441, i64 %457
  %459 = sub i64 %440, %457
  %460 = call ptr @__memcpy_chk(ptr %458, ptr %430, i64 %439, i64 %459)
  %461 = add i64 %457, %439
  %462 = getelementptr i8, ptr %441, i64 %461
  store i8 0, ptr %462, align 1
  call void @puts(ptr %441)
  call void @llvm.stackrestore.p0(ptr %426)
  br i1 false, label %463, label %481

463:                                              ; preds = %418
  br label %464

464:                                              ; preds = %463, %352
  %465 = load double, ptr %7, align 8
  %466 = load double, ptr %5, align 8
  %467 = fadd double %465, %466
  store double %467, ptr %7, align 8
  %468 = load i32, ptr %6, align 4
  %469 = add i32 %468, 1
  store i32 %469, ptr %6, align 4
  br label %8

470:                                              ; preds = %8
  %471 = load double, ptr %7, align 8
  %472 = fcmp oeq double %471, 0.000000e+00
  br i1 %472, label %473, label %483

473:                                              ; preds = %470
  %474 = load i32, ptr @fin, align 4
  %475 = icmp eq i32 %474, 60000
  br i1 %475, label %476, label %485

476:                                              ; preds = %473
  %477 = load i32, ptr @disposed, align 4
  %478 = icmp eq i32 %477, 20000
  br i1 %478, label %479, label %487

479:                                              ; preds = %476
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

480:                                              ; preds = %481, %487, %485, %483
  unreachable

481:                                              ; preds = %418
  %482 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11876826921290969655, i64 8), i32 134, ptr null)
  br label %480

483:                                              ; preds = %470
  %484 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11876826921290969655, i64 8), i32 140, ptr null)
  br label %480

485:                                              ; preds = %473
  %486 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11876826921290969655, i64 8), i32 141, ptr null)
  br label %480

487:                                              ; preds = %476
  %488 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11876826921290969655, i64 8), i32 142, ptr null)
  br label %480

endcatch.resume:                                  ; preds = %94, %76, %343, %49, %205, %43, %211, %148, %128, %58, %138, %300, %229, %310, %238, %247, %67, %85, %265, %256, %181, %220, %290, %126, %288, %103
  %489 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %489
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

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #5

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #5

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

