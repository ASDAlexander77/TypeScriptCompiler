; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Error = type { ptr, i32 }

$"___unbox<si32>" = comdat any

$Error..size = comdat any

@s_6860172195720241041 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFError\00", align 8
@s_10530820969299836017 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHello \00", align 8
@s_6125926117434577169 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/catch 1\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_8439431954124365208 = internal constant [101 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch.ts\00", align 8
@s_6125922818899692536 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/catch 2\00", align 8
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
@s_4420528118743043111 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHello World\00", align 8
@s_6125923918411320747 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/catch 3\00", align 8
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
@s_3094732814638223685 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFi32\00", align 8
@s_15952682373371409732 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf32\00", align 8
@s_16822532743728177535 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboolean\00", align 8
@s_8091808378024623192 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring\00", align 8
@s_9237349086447201248 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnumber\00", align 8
@s_9332519565554727459 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs32\00", align 8
@s_6125929415969461802 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry/catch 4\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@_ZTIi = external constant ptr
@_ZTId = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@Error..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_6860172195720241041, i64 8)
@Error..size = global i64 16, comdat
@Error..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Error..instanceOf, ptr @Error..new, ptr @Error..rtti, ptr @Error..size }
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

declare i64 @strtoll(ptr, ptr, i32)

declare i32 @strcmp(ptr, ptr)

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

declare ptr @__cxa_current_exception_type()

declare void @__cxa_rethrow()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @snprintf(ptr, i64, ptr, ...)

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @puts(ptr)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define void @main1() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6125926117434577169, i64 8))
  store i32 1, ptr %2, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %47 unwind label %4

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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %9
  br label %47

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %1, align 4
  %16 = load i32, ptr %1, align 4
  %17 = invoke ptr @GC_malloc_atomic(i64 58)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %12
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = invoke i32 (ptr, i64, ptr, ...) @snprintf(ptr %18, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %16)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.noexc
  %20 = call ptr @llvm.stacksave.p0()
  %21 = icmp eq ptr %18, null
  %22 = select i1 %21, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %18
  %23 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_10530820969299836017, i64 8))
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %24 = add i64 1, %23
  %25 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  %26 = add i64 %24, %25
  %27 = invoke i64 @strlen(ptr %22)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %28 = add i64 %26, %27
  %29 = alloca i8, i64 %28, align 1
  %30 = getelementptr i8, ptr %29, i64 0
  %31 = sub i64 %28, 0
  %32 = invoke ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_10530820969299836017, i64 8), i64 %23, i64 %31)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %33 = add i64 0, %23
  %34 = getelementptr i8, ptr %29, i64 %33
  %35 = sub i64 %28, %33
  %36 = invoke ptr @__memcpy_chk(ptr %34, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %25, i64 %35)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %37 = add i64 %33, %25
  %38 = getelementptr i8, ptr %29, i64 %37
  %39 = sub i64 %28, %37
  %40 = invoke ptr @__memcpy_chk(ptr %38, ptr %22, i64 %27, i64 %39)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %41 = add i64 %37, %27
  %42 = getelementptr i8, ptr %29, i64 %41
  store i8 0, ptr %42, align 1
  invoke void @puts(ptr %29)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  call void @llvm.stackrestore.p0(ptr %20)
  %43 = load i32, ptr %1, align 4
  store i32 %43, ptr %2, align 4
  call void @__cxa_end_catch()
  %44 = load i32, ptr %2, align 4
  %45 = icmp eq i32 %44, 2
  br i1 %45, label %46, label %48

46:                                               ; preds = %.noexc8
  ret void

47:                                               ; preds = %48, %.noexc9, %0
  unreachable

48:                                               ; preds = %.noexc8
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8439431954124365208, i64 8), i32 14, ptr null)
  br label %47

endcatch.resume:                                  ; preds = %9, %.noexc7, %.noexc6, %.noexc5, %.noexc4, %.noexc3, %.noexc2, %.noexc1, %.noexc, %12
  %50 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %50
}

; Function Attrs: mustprogress
define void @main2() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6125922818899692536, i64 8))
  store double 1.000000e+00, ptr %2, align 8
  %3 = call ptr @__cxa_allocate_exception(i64 8)
  store double 2.000000e+00, ptr %3, align 8
  invoke void @__cxa_throw(ptr %3, ptr @_ZTId, ptr null)
          to label %83 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %8 = icmp eq i32 %6, %7
  %9 = extractvalue { ptr, i32 } %5, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %11 = icmp eq i32 %9, %10
  %12 = extractvalue { ptr, i32 } %5, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %14 = icmp eq i32 %12, %13
  %15 = extractvalue { ptr, i32 } %5, 1
  %16 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %17 = icmp eq i32 %15, %16
  %18 = or i1 %8, %11
  %19 = or i1 %18, %14
  %20 = or i1 %19, %17
  br i1 %20, label %24, label %21

21:                                               ; preds = %4
  %22 = extractvalue { ptr, i32 } %5, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %21
  br label %83

24:                                               ; preds = %4
  %25 = extractvalue { ptr, i32 } %5, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %24
  %28 = icmp eq ptr %27, @_ZTIi
  br i1 %28, label %29, label %32

29:                                               ; preds = %.noexc10
  %30 = load i32, ptr %26, align 4
  %31 = sitofp i32 %30 to double
  br label %50

32:                                               ; preds = %.noexc10
  %33 = icmp eq ptr %27, @_ZTIj
  br i1 %33, label %34, label %37

34:                                               ; preds = %32
  %35 = load i32, ptr %26, align 4
  %36 = uitofp i32 %35 to double
  br label %47

37:                                               ; preds = %32
  %38 = icmp eq ptr %27, @_ZTIf
  br i1 %38, label %39, label %42

39:                                               ; preds = %37
  %40 = load float, ptr %26, align 4
  %41 = fpext float %40 to double
  br label %44

42:                                               ; preds = %37
  %43 = load double, ptr %26, align 8
  br label %44

44:                                               ; preds = %39, %42
  %45 = phi double [ %43, %42 ], [ %41, %39 ]
  br label %46

46:                                               ; preds = %44
  br label %47

47:                                               ; preds = %34, %46
  %48 = phi double [ %45, %46 ], [ %36, %34 ]
  br label %49

49:                                               ; preds = %47
  br label %50

50:                                               ; preds = %29, %49
  %51 = phi double [ %48, %49 ], [ %31, %29 ]
  br label %52

52:                                               ; preds = %50
  store double %51, ptr %1, align 8
  %53 = load double, ptr %1, align 8
  %54 = invoke ptr @GC_malloc_atomic(i64 40)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  invoke void @__ts_number_to_string(ptr %55, double %53)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %56 = call ptr @llvm.stacksave.p0()
  %57 = icmp eq ptr %55, null
  %58 = select i1 %57, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %55
  %59 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_10530820969299836017, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  %60 = add i64 1, %59
  %61 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %62 = add i64 %60, %61
  %63 = invoke i64 @strlen(ptr %58)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %64 = add i64 %62, %63
  %65 = alloca i8, i64 %64, align 1
  %66 = getelementptr i8, ptr %65, i64 0
  %67 = sub i64 %64, 0
  %68 = invoke ptr @__memcpy_chk(ptr %66, ptr getelementptr inbounds nuw (i8, ptr @s_10530820969299836017, i64 8), i64 %59, i64 %67)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %69 = add i64 0, %59
  %70 = getelementptr i8, ptr %65, i64 %69
  %71 = sub i64 %64, %69
  %72 = invoke ptr @__memcpy_chk(ptr %70, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %61, i64 %71)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %73 = add i64 %69, %61
  %74 = getelementptr i8, ptr %65, i64 %73
  %75 = sub i64 %64, %73
  %76 = invoke ptr @__memcpy_chk(ptr %74, ptr %58, i64 %63, i64 %75)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %77 = add i64 %73, %63
  %78 = getelementptr i8, ptr %65, i64 %77
  store i8 0, ptr %78, align 1
  invoke void @puts(ptr %65)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  call void @llvm.stackrestore.p0(ptr %56)
  %79 = load double, ptr %1, align 8
  store double %79, ptr %2, align 8
  call void @__cxa_end_catch()
  %80 = load double, ptr %2, align 8
  %81 = fcmp oeq double 2.000000e+00, %80
  br i1 %81, label %82, label %84

82:                                               ; preds = %.noexc9
  ret void

83:                                               ; preds = %84, %.noexc, %0
  unreachable

84:                                               ; preds = %.noexc9
  %85 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8439431954124365208, i64 8), i32 29, ptr null)
  br label %83

endcatch.resume:                                  ; preds = %24, %.noexc8, %.noexc7, %.noexc6, %.noexc5, %.noexc4, %.noexc3, %.noexc2, %.noexc1, %52, %21
  %86 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %86
}

; Function Attrs: mustprogress
define void @main3() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6125923918411320747, i64 8))
  store i32 1, ptr %2, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4420528118743043111, i64 8), ptr %3, align 8
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIPc, ptr null)
          to label %21 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %9
  br label %21

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  store ptr %14, ptr %1, align 8
  %15 = load ptr, ptr %1, align 8
  %16 = icmp eq ptr %15, null
  %17 = select i1 %16, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  invoke void @puts(ptr %17)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %12
  store i32 2, ptr %2, align 4
  call void @__cxa_end_catch()
  %18 = load i32, ptr %2, align 4
  %19 = icmp eq i32 %18, 2
  br i1 %19, label %20, label %22

20:                                               ; preds = %.noexc
  ret void

21:                                               ; preds = %22, %.noexc1, %0
  unreachable

22:                                               ; preds = %.noexc
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8439431954124365208, i64 8), i32 44, ptr null)
  br label %21

endcatch.resume:                                  ; preds = %9, %12
  %24 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %24
}

; Function Attrs: mustprogress
define linkonce_odr i32 @"___unbox<si32>"(ptr %0) #3 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
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
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %24

20:                                               ; preds = %19
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 2
  %23 = load i32, ptr %22, align 4
  store i32 %23, ptr %3, align 4
  br label %469

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
  %45 = fptosi double %44 to i64
  %46 = trunc i64 %45 to i32
  store i32 %46, ptr %3, align 4
  br label %469

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
  br i1 %62, label %64, label %72

64:                                               ; preds = %63
  %65 = load ptr, ptr %2, align 8
  %66 = getelementptr { i64, ptr, ptr }, ptr %65, i32 0, i32 2
  %67 = load ptr, ptr %66, align 8
  %68 = call i64 @strtoll(ptr %67, ptr null, i32 10)
  %69 = call i64 @llvm.smin.i64(i64 %68, i64 2147483647)
  %70 = call i64 @llvm.smax.i64(i64 %69, i64 -2147483648)
  %71 = trunc i64 %70 to i32
  store i32 %71, ptr %3, align 4
  br label %469

72:                                               ; preds = %63
  %73 = load ptr, ptr %2, align 8
  %74 = getelementptr { i64, ptr, i8 }, ptr %73, i32 0, i32 1
  %75 = load ptr, ptr %74, align 8
  %76 = ptrtoint ptr %75 to i64
  %77 = icmp ne i64 %76, 0
  %78 = zext i1 %77 to i32
  %79 = and i32 %78, 1
  %80 = icmp ne i32 %79, 0
  br i1 %80, label %81, label %84

81:                                               ; preds = %72
  %82 = call i32 @strcmp(ptr %75, ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8))
  %83 = icmp eq i32 %82, 0
  br label %86

84:                                               ; preds = %72
  %85 = icmp eq i64 %76, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8) to i64)
  br label %86

86:                                               ; preds = %81, %84
  %87 = phi i1 [ %85, %84 ], [ %83, %81 ]
  br label %88

88:                                               ; preds = %86
  br i1 %87, label %89, label %94

89:                                               ; preds = %88
  %90 = load ptr, ptr %2, align 8
  %91 = getelementptr { i64, ptr, i1 }, ptr %90, i32 0, i32 2
  %92 = load i1, ptr %91, align 1
  %93 = zext i1 %92 to i32
  store i32 %93, ptr %3, align 4
  br label %469

94:                                               ; preds = %88
  %95 = load ptr, ptr %2, align 8
  %96 = getelementptr { i64, ptr, i8 }, ptr %95, i32 0, i32 1
  %97 = load ptr, ptr %96, align 8
  %98 = ptrtoint ptr %97 to i64
  %99 = icmp ne i64 %98, 0
  %100 = zext i1 %99 to i32
  %101 = and i32 %100, 1
  %102 = icmp ne i32 %101, 0
  br i1 %102, label %103, label %106

103:                                              ; preds = %94
  %104 = call i32 @strcmp(ptr %97, ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8))
  %105 = icmp eq i32 %104, 0
  br label %108

106:                                              ; preds = %94
  %107 = icmp eq i64 %98, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8) to i64)
  br label %108

108:                                              ; preds = %103, %106
  %109 = phi i1 [ %107, %106 ], [ %105, %103 ]
  br label %110

110:                                              ; preds = %108
  br i1 %109, label %111, label %117

111:                                              ; preds = %110
  %112 = load ptr, ptr %2, align 8
  %113 = getelementptr { i64, ptr, float }, ptr %112, i32 0, i32 2
  %114 = load float, ptr %113, align 4
  %115 = fptosi float %114 to i64
  %116 = trunc i64 %115 to i32
  store i32 %116, ptr %3, align 4
  br label %469

117:                                              ; preds = %110
  %118 = load ptr, ptr %2, align 8
  %119 = getelementptr { i64, ptr, i8 }, ptr %118, i32 0, i32 1
  %120 = load ptr, ptr %119, align 8
  %121 = ptrtoint ptr %120 to i64
  %122 = icmp ne i64 %121, 0
  %123 = zext i1 %122 to i32
  %124 = and i32 %123, 1
  %125 = icmp ne i32 %124, 0
  br i1 %125, label %126, label %129

126:                                              ; preds = %117
  %127 = call i32 @strcmp(ptr %120, ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8))
  %128 = icmp eq i32 %127, 0
  br label %131

129:                                              ; preds = %117
  %130 = icmp eq i64 %121, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8) to i64)
  br label %131

131:                                              ; preds = %126, %129
  %132 = phi i1 [ %130, %129 ], [ %128, %126 ]
  br label %133

133:                                              ; preds = %131
  br i1 %132, label %134, label %138

134:                                              ; preds = %133
  %135 = load ptr, ptr %2, align 8
  %136 = getelementptr { i64, ptr, i32 }, ptr %135, i32 0, i32 2
  %137 = load i32, ptr %136, align 4
  store i32 %137, ptr %3, align 4
  br label %469

138:                                              ; preds = %133
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
  br i1 %153, label %155, label %159

155:                                              ; preds = %154
  %156 = load ptr, ptr %2, align 8
  %157 = getelementptr { i64, ptr, i32 }, ptr %156, i32 0, i32 2
  %158 = load i32, ptr %157, align 4
  store i32 %158, ptr %3, align 4
  br label %469

159:                                              ; preds = %154
  %160 = load ptr, ptr %2, align 8
  %161 = getelementptr { i64, ptr, i8 }, ptr %160, i32 0, i32 1
  %162 = load ptr, ptr %161, align 8
  %163 = ptrtoint ptr %162 to i64
  %164 = icmp ne i64 %163, 0
  %165 = zext i1 %164 to i32
  %166 = and i32 %165, 1
  %167 = icmp ne i32 %166, 0
  br i1 %167, label %168, label %171

168:                                              ; preds = %159
  %169 = call i32 @strcmp(ptr %162, ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8))
  %170 = icmp eq i32 %169, 0
  br label %173

171:                                              ; preds = %159
  %172 = icmp eq i64 %163, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8) to i64)
  br label %173

173:                                              ; preds = %168, %171
  %174 = phi i1 [ %172, %171 ], [ %170, %168 ]
  br label %175

175:                                              ; preds = %173
  br i1 %174, label %176, label %180

176:                                              ; preds = %175
  %177 = load ptr, ptr %2, align 8
  %178 = getelementptr { i64, ptr, i32 }, ptr %177, i32 0, i32 2
  %179 = load i32, ptr %178, align 4
  store i32 %179, ptr %3, align 4
  br label %469

180:                                              ; preds = %175
  %181 = load ptr, ptr %2, align 8
  %182 = getelementptr { i64, ptr, i8 }, ptr %181, i32 0, i32 1
  %183 = load ptr, ptr %182, align 8
  %184 = ptrtoint ptr %183 to i64
  %185 = icmp ne i64 %184, 0
  %186 = zext i1 %185 to i32
  %187 = and i32 %186, 1
  %188 = icmp ne i32 %187, 0
  br i1 %188, label %189, label %192

189:                                              ; preds = %180
  %190 = call i32 @strcmp(ptr %183, ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8))
  %191 = icmp eq i32 %190, 0
  br label %194

192:                                              ; preds = %180
  %193 = icmp eq i64 %184, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8) to i64)
  br label %194

194:                                              ; preds = %189, %192
  %195 = phi i1 [ %193, %192 ], [ %191, %189 ]
  br label %196

196:                                              ; preds = %194
  br i1 %195, label %197, label %202

197:                                              ; preds = %196
  %198 = load ptr, ptr %2, align 8
  %199 = getelementptr { i64, ptr, i64 }, ptr %198, i32 0, i32 2
  %200 = load i64, ptr %199, align 8
  %201 = trunc i64 %200 to i32
  store i32 %201, ptr %3, align 4
  br label %469

202:                                              ; preds = %196
  %203 = load ptr, ptr %2, align 8
  %204 = getelementptr { i64, ptr, i8 }, ptr %203, i32 0, i32 1
  %205 = load ptr, ptr %204, align 8
  %206 = ptrtoint ptr %205 to i64
  %207 = icmp ne i64 %206, 0
  %208 = zext i1 %207 to i32
  %209 = and i32 %208, 1
  %210 = icmp ne i32 %209, 0
  br i1 %210, label %211, label %214

211:                                              ; preds = %202
  %212 = call i32 @strcmp(ptr %205, ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8))
  %213 = icmp eq i32 %212, 0
  br label %216

214:                                              ; preds = %202
  %215 = icmp eq i64 %206, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8) to i64)
  br label %216

216:                                              ; preds = %211, %214
  %217 = phi i1 [ %215, %214 ], [ %213, %211 ]
  br label %218

218:                                              ; preds = %216
  br i1 %217, label %219, label %225

219:                                              ; preds = %218
  %220 = load ptr, ptr %2, align 8
  %221 = getelementptr { i64, ptr, double }, ptr %220, i32 0, i32 2
  %222 = load double, ptr %221, align 8
  %223 = fptosi double %222 to i64
  %224 = trunc i64 %223 to i32
  store i32 %224, ptr %3, align 4
  br label %469

225:                                              ; preds = %218
  %226 = load ptr, ptr %2, align 8
  %227 = getelementptr { i64, ptr, i8 }, ptr %226, i32 0, i32 1
  %228 = load ptr, ptr %227, align 8
  %229 = ptrtoint ptr %228 to i64
  %230 = icmp ne i64 %229, 0
  %231 = zext i1 %230 to i32
  %232 = and i32 %231, 1
  %233 = icmp ne i32 %232, 0
  br i1 %233, label %234, label %237

234:                                              ; preds = %225
  %235 = call i32 @strcmp(ptr %228, ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8))
  %236 = icmp eq i32 %235, 0
  br label %239

237:                                              ; preds = %225
  %238 = icmp eq i64 %229, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8) to i64)
  br label %239

239:                                              ; preds = %234, %237
  %240 = phi i1 [ %238, %237 ], [ %236, %234 ]
  br label %241

241:                                              ; preds = %239
  br i1 %240, label %242, label %247

242:                                              ; preds = %241
  %243 = load ptr, ptr %2, align 8
  %244 = getelementptr { i64, ptr, i64 }, ptr %243, i32 0, i32 2
  %245 = load i64, ptr %244, align 8
  %246 = trunc i64 %245 to i32
  store i32 %246, ptr %3, align 4
  br label %469

247:                                              ; preds = %241
  %248 = load ptr, ptr %2, align 8
  %249 = getelementptr { i64, ptr, i8 }, ptr %248, i32 0, i32 1
  %250 = load ptr, ptr %249, align 8
  %251 = ptrtoint ptr %250 to i64
  %252 = icmp ne i64 %251, 0
  %253 = zext i1 %252 to i32
  %254 = and i32 %253, 1
  %255 = icmp ne i32 %254, 0
  br i1 %255, label %256, label %259

256:                                              ; preds = %247
  %257 = call i32 @strcmp(ptr %250, ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8))
  %258 = icmp eq i32 %257, 0
  br label %261

259:                                              ; preds = %247
  %260 = icmp eq i64 %251, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8) to i64)
  br label %261

261:                                              ; preds = %256, %259
  %262 = phi i1 [ %260, %259 ], [ %258, %256 ]
  br label %263

263:                                              ; preds = %261
  br i1 %262, label %264, label %269

264:                                              ; preds = %263
  %265 = load ptr, ptr %2, align 8
  %266 = getelementptr { i64, ptr, i64 }, ptr %265, i32 0, i32 2
  %267 = load i64, ptr %266, align 8
  %268 = trunc i64 %267 to i32
  store i32 %268, ptr %3, align 4
  br label %469

269:                                              ; preds = %263
  %270 = load ptr, ptr %2, align 8
  %271 = getelementptr { i64, ptr, i8 }, ptr %270, i32 0, i32 1
  %272 = load ptr, ptr %271, align 8
  %273 = ptrtoint ptr %272 to i64
  %274 = icmp ne i64 %273, 0
  %275 = zext i1 %274 to i32
  %276 = and i32 %275, 1
  %277 = icmp ne i32 %276, 0
  br i1 %277, label %278, label %281

278:                                              ; preds = %269
  %279 = call i32 @strcmp(ptr %272, ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8))
  %280 = icmp eq i32 %279, 0
  br label %283

281:                                              ; preds = %269
  %282 = icmp eq i64 %273, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8) to i64)
  br label %283

283:                                              ; preds = %278, %281
  %284 = phi i1 [ %282, %281 ], [ %280, %278 ]
  br label %285

285:                                              ; preds = %283
  br i1 %284, label %286, label %291

286:                                              ; preds = %285
  %287 = load ptr, ptr %2, align 8
  %288 = getelementptr { i64, ptr, i64 }, ptr %287, i32 0, i32 2
  %289 = load i64, ptr %288, align 8
  %290 = trunc i64 %289 to i32
  store i32 %290, ptr %3, align 4
  br label %469

291:                                              ; preds = %285
  %292 = load ptr, ptr %2, align 8
  %293 = getelementptr { i64, ptr, i8 }, ptr %292, i32 0, i32 1
  %294 = load ptr, ptr %293, align 8
  %295 = ptrtoint ptr %294 to i64
  %296 = icmp ne i64 %295, 0
  %297 = zext i1 %296 to i32
  %298 = and i32 %297, 1
  %299 = icmp ne i32 %298, 0
  br i1 %299, label %300, label %303

300:                                              ; preds = %291
  %301 = call i32 @strcmp(ptr %294, ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8))
  %302 = icmp eq i32 %301, 0
  br label %305

303:                                              ; preds = %291
  %304 = icmp eq i64 %295, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8) to i64)
  br label %305

305:                                              ; preds = %300, %303
  %306 = phi i1 [ %304, %303 ], [ %302, %300 ]
  br label %307

307:                                              ; preds = %305
  br i1 %306, label %308, label %313

308:                                              ; preds = %307
  %309 = load ptr, ptr %2, align 8
  %310 = getelementptr { i64, ptr, i8 }, ptr %309, i32 0, i32 2
  %311 = load i8, ptr %310, align 1
  %312 = zext i8 %311 to i32
  store i32 %312, ptr %3, align 4
  br label %469

313:                                              ; preds = %307
  %314 = load ptr, ptr %2, align 8
  %315 = getelementptr { i64, ptr, i8 }, ptr %314, i32 0, i32 1
  %316 = load ptr, ptr %315, align 8
  %317 = ptrtoint ptr %316 to i64
  %318 = icmp ne i64 %317, 0
  %319 = zext i1 %318 to i32
  %320 = and i32 %319, 1
  %321 = icmp ne i32 %320, 0
  br i1 %321, label %322, label %325

322:                                              ; preds = %313
  %323 = call i32 @strcmp(ptr %316, ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8))
  %324 = icmp eq i32 %323, 0
  br label %327

325:                                              ; preds = %313
  %326 = icmp eq i64 %317, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8) to i64)
  br label %327

327:                                              ; preds = %322, %325
  %328 = phi i1 [ %326, %325 ], [ %324, %322 ]
  br label %329

329:                                              ; preds = %327
  br i1 %328, label %330, label %335

330:                                              ; preds = %329
  %331 = load ptr, ptr %2, align 8
  %332 = getelementptr { i64, ptr, i64 }, ptr %331, i32 0, i32 2
  %333 = load i64, ptr %332, align 8
  %334 = trunc i64 %333 to i32
  store i32 %334, ptr %3, align 4
  br label %469

335:                                              ; preds = %329
  %336 = load ptr, ptr %2, align 8
  %337 = getelementptr { i64, ptr, i8 }, ptr %336, i32 0, i32 1
  %338 = load ptr, ptr %337, align 8
  %339 = ptrtoint ptr %338 to i64
  %340 = icmp ne i64 %339, 0
  %341 = zext i1 %340 to i32
  %342 = and i32 %341, 1
  %343 = icmp ne i32 %342, 0
  br i1 %343, label %344, label %347

344:                                              ; preds = %335
  %345 = call i32 @strcmp(ptr %338, ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8))
  %346 = icmp eq i32 %345, 0
  br label %349

347:                                              ; preds = %335
  %348 = icmp eq i64 %339, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8) to i64)
  br label %349

349:                                              ; preds = %344, %347
  %350 = phi i1 [ %348, %347 ], [ %346, %344 ]
  br label %351

351:                                              ; preds = %349
  br i1 %350, label %352, label %357

352:                                              ; preds = %351
  %353 = load ptr, ptr %2, align 8
  %354 = getelementptr { i64, ptr, i16 }, ptr %353, i32 0, i32 2
  %355 = load i16, ptr %354, align 2
  %356 = sext i16 %355 to i32
  store i32 %356, ptr %3, align 4
  br label %469

357:                                              ; preds = %351
  %358 = load ptr, ptr %2, align 8
  %359 = getelementptr { i64, ptr, i8 }, ptr %358, i32 0, i32 1
  %360 = load ptr, ptr %359, align 8
  %361 = ptrtoint ptr %360 to i64
  %362 = icmp ne i64 %361, 0
  %363 = zext i1 %362 to i32
  %364 = and i32 %363, 1
  %365 = icmp ne i32 %364, 0
  br i1 %365, label %366, label %369

366:                                              ; preds = %357
  %367 = call i32 @strcmp(ptr %360, ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8))
  %368 = icmp eq i32 %367, 0
  br label %371

369:                                              ; preds = %357
  %370 = icmp eq i64 %361, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8) to i64)
  br label %371

371:                                              ; preds = %366, %369
  %372 = phi i1 [ %370, %369 ], [ %368, %366 ]
  br label %373

373:                                              ; preds = %371
  br i1 %372, label %374, label %379

374:                                              ; preds = %373
  %375 = load ptr, ptr %2, align 8
  %376 = getelementptr { i64, ptr, i16 }, ptr %375, i32 0, i32 2
  %377 = load i16, ptr %376, align 2
  %378 = sext i16 %377 to i32
  store i32 %378, ptr %3, align 4
  br label %469

379:                                              ; preds = %373
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
  %389 = call i32 @strcmp(ptr %382, ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8))
  %390 = icmp eq i32 %389, 0
  br label %393

391:                                              ; preds = %379
  %392 = icmp eq i64 %383, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8) to i64)
  br label %393

393:                                              ; preds = %388, %391
  %394 = phi i1 [ %392, %391 ], [ %390, %388 ]
  br label %395

395:                                              ; preds = %393
  br i1 %394, label %396, label %401

396:                                              ; preds = %395
  %397 = load ptr, ptr %2, align 8
  %398 = getelementptr { i64, ptr, i16 }, ptr %397, i32 0, i32 2
  %399 = load i16, ptr %398, align 2
  %400 = zext i16 %399 to i32
  store i32 %400, ptr %3, align 4
  br label %469

401:                                              ; preds = %395
  %402 = load ptr, ptr %2, align 8
  %403 = getelementptr { i64, ptr, i8 }, ptr %402, i32 0, i32 1
  %404 = load ptr, ptr %403, align 8
  %405 = ptrtoint ptr %404 to i64
  %406 = icmp ne i64 %405, 0
  %407 = zext i1 %406 to i32
  %408 = and i32 %407, 1
  %409 = icmp ne i32 %408, 0
  br i1 %409, label %410, label %413

410:                                              ; preds = %401
  %411 = call i32 @strcmp(ptr %404, ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8))
  %412 = icmp eq i32 %411, 0
  br label %415

413:                                              ; preds = %401
  %414 = icmp eq i64 %405, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8) to i64)
  br label %415

415:                                              ; preds = %410, %413
  %416 = phi i1 [ %414, %413 ], [ %412, %410 ]
  br label %417

417:                                              ; preds = %415
  br i1 %416, label %418, label %423

418:                                              ; preds = %417
  %419 = load ptr, ptr %2, align 8
  %420 = getelementptr { i64, ptr, i8 }, ptr %419, i32 0, i32 2
  %421 = load i8, ptr %420, align 1
  %422 = sext i8 %421 to i32
  store i32 %422, ptr %3, align 4
  br label %469

423:                                              ; preds = %417
  %424 = load ptr, ptr %2, align 8
  %425 = getelementptr { i64, ptr, i8 }, ptr %424, i32 0, i32 1
  %426 = load ptr, ptr %425, align 8
  %427 = ptrtoint ptr %426 to i64
  %428 = icmp ne i64 %427, 0
  %429 = zext i1 %428 to i32
  %430 = and i32 %429, 1
  %431 = icmp ne i32 %430, 0
  br i1 %431, label %432, label %435

432:                                              ; preds = %423
  %433 = call i32 @strcmp(ptr %426, ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8))
  %434 = icmp eq i32 %433, 0
  br label %437

435:                                              ; preds = %423
  %436 = icmp eq i64 %427, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8) to i64)
  br label %437

437:                                              ; preds = %432, %435
  %438 = phi i1 [ %436, %435 ], [ %434, %432 ]
  br label %439

439:                                              ; preds = %437
  br i1 %438, label %440, label %445

440:                                              ; preds = %439
  %441 = load ptr, ptr %2, align 8
  %442 = getelementptr { i64, ptr, i8 }, ptr %441, i32 0, i32 2
  %443 = load i8, ptr %442, align 1
  %444 = sext i8 %443 to i32
  store i32 %444, ptr %3, align 4
  br label %469

445:                                              ; preds = %439
  %446 = load ptr, ptr %2, align 8
  %447 = getelementptr { i64, ptr, i8 }, ptr %446, i32 0, i32 1
  %448 = load ptr, ptr %447, align 8
  %449 = ptrtoint ptr %448 to i64
  %450 = icmp ne i64 %449, 0
  %451 = zext i1 %450 to i32
  %452 = and i32 %451, 1
  %453 = icmp ne i32 %452, 0
  br i1 %453, label %454, label %457

454:                                              ; preds = %445
  %455 = call i32 @strcmp(ptr %448, ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8))
  %456 = icmp eq i32 %455, 0
  br label %459

457:                                              ; preds = %445
  %458 = icmp eq i64 %449, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8) to i64)
  br label %459

459:                                              ; preds = %454, %457
  %460 = phi i1 [ %458, %457 ], [ %456, %454 ]
  br label %461

461:                                              ; preds = %459
  br i1 %460, label %462, label %467

462:                                              ; preds = %461
  %463 = load ptr, ptr %2, align 8
  %464 = getelementptr { i64, ptr, i8 }, ptr %463, i32 0, i32 2
  %465 = load i8, ptr %464, align 1
  %466 = zext i8 %465 to i32
  store i32 %466, ptr %3, align 4
  br label %469

467:                                              ; preds = %461
  %468 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %468, align 8
  call void @__cxa_throw(ptr %468, ptr @_ZTIPc, ptr null)
  br label %471

469:                                              ; preds = %20, %41, %64, %89, %111, %134, %155, %176, %197, %219, %242, %264, %286, %308, %330, %352, %374, %396, %418, %440, %462
  %470 = load i32, ptr %3, align 4
  ret i32 %470

471:                                              ; preds = %467
  unreachable
}

; Function Attrs: mustprogress
define void @main4() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6125929415969461802, i64 8))
  store i32 1, ptr %3, align 4
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, i32 }, ptr %5, i32 0, i32 0
  store i64 4, ptr %6, align 8
  %7 = getelementptr { i64, ptr, i32 }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, i32 }, ptr %5, i32 0, i32 2
  store i32 123, ptr %8, align 4
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %5, ptr %9, align 8
  invoke void @__cxa_throw(ptr %9, ptr @_ZTIPv, ptr null)
          to label %161 unwind label %10

10:                                               ; preds = %0
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %10
  %15 = icmp eq ptr %14, @_ZTIi
  br i1 %15, label %16, label %23

16:                                               ; preds = %.noexc13
  %17 = load i32, ptr %13, align 4
  %18 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %16
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 0
  store i64 4, ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 2
  store i32 %17, ptr %22, align 4
  br label %146

23:                                               ; preds = %.noexc13
  %24 = icmp eq ptr %14, @_ZTIj
  br i1 %24, label %25, label %32

25:                                               ; preds = %23
  %26 = load i32, ptr %13, align 4
  %27 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %25
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 0
  store i64 4, ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 2
  store i32 %26, ptr %31, align 4
  br label %143

32:                                               ; preds = %23
  %33 = icmp eq ptr %14, @_ZTId
  br i1 %33, label %34, label %41

34:                                               ; preds = %32
  %35 = load double, ptr %13, align 8
  %36 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %34
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 0
  store i64 8, ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 2
  store double %35, ptr %40, align 8
  br label %140

41:                                               ; preds = %32
  %42 = icmp eq ptr %14, @_ZTIf
  br i1 %42, label %43, label %50

43:                                               ; preds = %41
  %44 = load float, ptr %13, align 4
  %45 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %43
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 0
  store i64 4, ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 2
  store float %44, ptr %49, align 4
  br label %137

50:                                               ; preds = %41
  %51 = icmp eq ptr %14, @_ZTIb
  br i1 %51, label %52, label %59

52:                                               ; preds = %50
  %53 = load i1, ptr %13, align 1
  %54 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 0
  store i64 1, ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 2
  store i1 %53, ptr %58, align 1
  br label %134

59:                                               ; preds = %50
  %60 = icmp eq ptr %14, @_ZTIx
  br i1 %60, label %61, label %68

61:                                               ; preds = %59
  %62 = load i64, ptr %13, align 8
  %63 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %61
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 0
  store i64 8, ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 2
  store i64 %62, ptr %67, align 8
  br label %131

68:                                               ; preds = %59
  %69 = icmp eq ptr %14, @_ZTIPc
  br i1 %69, label %70, label %77

70:                                               ; preds = %68
  %71 = load ptr, ptr %13, align 8
  %72 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %70
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 0
  store i64 8, ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 2
  store ptr %71, ptr %76, align 8
  br label %128

77:                                               ; preds = %68
  %78 = icmp eq ptr %14, @_ZTIPv
  br i1 %78, label %79, label %81

79:                                               ; preds = %77
  %80 = load ptr, ptr %13, align 8
  br label %125

81:                                               ; preds = %77
  %82 = load ptr, ptr %14, align 8
  %83 = icmp eq ptr %82, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %83, label %84, label %115

84:                                               ; preds = %81
  %85 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %14, i32 0, i32 2
  %86 = load i32, ptr %85, align 4
  %87 = and i32 %86, 1073741824
  %88 = icmp ne i32 %87, 0
  br i1 %88, label %89, label %105

89:                                               ; preds = %84
  %90 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %14, i32 0, i32 4
  %91 = load ptr, ptr %90, align 8
  %92 = icmp ne ptr %91, null
  br i1 %92, label %93, label %95

93:                                               ; preds = %89
  invoke void %91(ptr %1, ptr %13)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %93
  %94 = load ptr, ptr %1, align 8
  br label %102

95:                                               ; preds = %89
  %96 = load ptr, ptr %13, align 8
  %97 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %95
  %98 = getelementptr i8, ptr %97, i64 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 0
  store i64 8, ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 2
  store ptr %96, ptr %101, align 8
  br label %102

102:                                              ; preds = %.noexc2, %.noexc7
  %103 = phi ptr [ %98, %.noexc7 ], [ %94, %.noexc2 ]
  br label %104

104:                                              ; preds = %102
  br label %112

105:                                              ; preds = %84
  %106 = load ptr, ptr %13, align 8
  %107 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %105
  %108 = getelementptr i8, ptr %107, i64 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 0
  store i64 8, ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 2
  store ptr %106, ptr %111, align 8
  br label %112

112:                                              ; preds = %104, %.noexc15
  %113 = phi ptr [ %108, %.noexc15 ], [ %103, %104 ]
  br label %114

114:                                              ; preds = %112
  br label %122

115:                                              ; preds = %81
  %116 = load ptr, ptr %13, align 8
  %117 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %115
  %118 = getelementptr i8, ptr %117, i64 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 0
  store i64 8, ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 2
  store ptr %116, ptr %121, align 8
  br label %122

122:                                              ; preds = %114, %.noexc
  %123 = phi ptr [ %118, %.noexc ], [ %113, %114 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %79, %124
  %126 = phi ptr [ %123, %124 ], [ %80, %79 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc4, %127
  %129 = phi ptr [ %126, %127 ], [ %73, %.noexc4 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc16, %130
  %132 = phi ptr [ %129, %130 ], [ %64, %.noexc16 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc1, %133
  %135 = phi ptr [ %132, %133 ], [ %55, %.noexc1 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc14, %136
  %138 = phi ptr [ %135, %136 ], [ %46, %.noexc14 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc3, %139
  %141 = phi ptr [ %138, %139 ], [ %37, %.noexc3 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc6, %142
  %144 = phi ptr [ %141, %142 ], [ %28, %.noexc6 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc5, %145
  %147 = phi ptr [ %144, %145 ], [ %19, %.noexc5 ]
  br label %148

148:                                              ; preds = %146
  store ptr %147, ptr %2, align 8
  %149 = load ptr, ptr %2, align 8
  %150 = invoke i32 @"___unbox<si32>"(ptr %149)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %148
  %151 = invoke ptr @GC_malloc_atomic(i64 58)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %152 = getelementptr i8, ptr %151, i64 8
  %153 = invoke i32 (ptr, i64, ptr, ...) @snprintf(ptr %152, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %150)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %.noexc9
  %154 = icmp eq ptr %152, null
  %155 = select i1 %154, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %152
  invoke void @puts(ptr %155)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  %156 = load ptr, ptr %2, align 8
  %157 = invoke i32 @"___unbox<si32>"(ptr %156)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %.noexc11
  store i32 %157, ptr %3, align 4
  call void @__cxa_end_catch()
  %158 = load i32, ptr %3, align 4
  %159 = icmp eq i32 %158, 123
  br i1 %159, label %160, label %162

160:                                              ; preds = %.noexc12
  ret void

161:                                              ; preds = %162, %0
  unreachable

162:                                              ; preds = %.noexc12
  %163 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8439431954124365208, i64 8), i32 59, ptr null)
  br label %161

endcatch.resume:                                  ; preds = %61, %105, %43, %10, %.noexc11, %.noexc10, %.noexc9, %.noexc8, %148, %95, %25, %16, %70, %34, %93, %52, %115
  %164 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %164
}

; Function Attrs: mustprogress
define void @Error.constructor(ptr %0) #3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Error, ptr %3, i32 0, i32 1
  store i32 10, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Error..instanceOf(ptr %0, ptr %1) #3 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Error..rtti, align 8
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
define ptr @Error..new() #3 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Error, ptr %3, i32 0, i32 0
  store ptr @Error..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i32 @main() #3 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @main1()
  call void @main2()
  call void @main3()
  call void @main4()
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #5

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress "personality" }
attributes #3 = { mustprogress }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

