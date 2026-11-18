; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%E = type { ptr, double, ptr }
%H = type { ptr, double }

$"___unbox<!ts.string>" = comdat any

$"___unbox<!ts.class<E, !ts.class_storage<E>>>" = comdat any

$E..size = comdat any

$H..size = comdat any

$_ZTSP1E = comdat any

$_ZTIP1E = comdat any

$_ZTS1E = comdat any

$_ZTI1E = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638217986881231904 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFE\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@E..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638217986881231904, i64 8)
@E..size = global i64 24, comdat
@E..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @E..instanceOf, ptr @E..new, ptr @E..rtti, ptr @E..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 16, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@_ZTIPc = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP1E = linkonce_odr constant [4 x i8] c"P1E\00", comdat
@_ZTIP1E = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP1E, i32 1073741824, ptr @_ZTI1E, ptr @.eh.copy.box.P1E }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS1E = linkonce_odr constant [3 x i8] c"1E\00", comdat
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_17464517255946554849 = internal constant [110 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\own\\own_caught_value_owned.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
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
@s_12638191598602154840 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFm\00", align 8
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_2495249626774661283 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined\00", align 8
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
@s_9237349086447201248 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnumber\00", align 8
@s_8091808378024623192 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring\00", align 8
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
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_13770579504406577721 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx99\00", align 8
@m_11601984928555610496 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa caught value kept past its clause\00", align 8
@m_15069869055152812943 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery value read after churn\00", align 8
@_ZTI1E = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1E }, comdat
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@kept = private global ptr undef
@a_0_13605027861770455342 = internal constant [0 x ptr] undef
@a_0_7891295797979919183 = internal constant [0 x ptr] undef
@td_7632308648875476676_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_7632308648875476676, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_14527407094101050897_null = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [5 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 13, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [5 x i8] c"null\00" }, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @kept__cctor, ptr null }]

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

define internal void @tsrel_7632308648875476676(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %E, ptr %2, i32 0, i32 2
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

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

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

define internal void @kept__cctor() {
  %1 = call ptr @GC_malloc(i64 8)
  %2 = getelementptr i8, ptr %1, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %2, ptr @a_0_13605027861770455342, i64 0, i1 false)
  %3 = call ptr @GC_malloc(i64 32)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr { ptr, i64, i64 }, ptr %4, i32 0, i32 0
  store ptr %2, ptr %5, align 8
  %6 = getelementptr { ptr, i64, i64 }, ptr %4, i32 0, i32 1
  store i64 0, ptr %6, align 8
  %7 = getelementptr { ptr, i64, i64 }, ptr %4, i32 0, i32 2
  store i64 0, ptr %7, align 8
  store ptr %4, ptr @kept, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @E..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @E..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %E, ptr %3, i32 0, i32 0
  store ptr @E..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @E.constructor(ptr %0, double %1, ptr %2) #2 {
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %E, ptr %7, i32 0, i32 1
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %E, ptr %10, i32 0, i32 2
  %12 = load ptr, ptr %4, align 8
  store ptr %12, ptr %11, align 8
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
define void @H.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %H, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define i64 @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = call ptr @GC_malloc(i64 8)
  %5 = getelementptr i8, ptr %4, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %5, ptr @a_0_7891295797979919183, i64 0, i1 false)
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

11:                                               ; preds = %62, %0
  %12 = load i32, ptr %1, align 4
  %13 = icmp slt i32 %12, 20
  br i1 %13, label %14, label %66

14:                                               ; preds = %11
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %H, ptr %16, i32 0, i32 0
  store ptr @H..vtbl, ptr %17, align 8
  call void @H.constructor(ptr %16, double 7.777700e+04)
  %18 = load ptr, ptr %2, align 8
  %19 = icmp eq ptr %18, null
  br i1 %19, label %20, label %23

20:                                               ; preds = %14
  %21 = call ptr @GC_malloc(i64 32)
  %22 = getelementptr i8, ptr %21, i64 8
  store ptr %22, ptr %2, align 8
  br label %24

23:                                               ; preds = %14
  br label %24

24:                                               ; preds = %20, %23
  %25 = phi ptr [ %18, %23 ], [ %22, %20 ]
  br label %26

26:                                               ; preds = %24
  %27 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 1
  %28 = load i64, ptr %27, align 8
  %29 = add i64 %28, 1
  %30 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 0
  %31 = getelementptr { ptr, i64, i64 }, ptr %25, i32 0, i32 2
  %32 = load ptr, ptr %30, align 8
  %33 = load i64, ptr %31, align 8
  %34 = icmp ugt i64 %29, %33
  br i1 %34, label %35, label %59

35:                                               ; preds = %26
  %36 = mul i64 %33, 2
  %37 = icmp uge i64 %36, %29
  %38 = select i1 %37, i64 %36, i64 %29
  %39 = icmp ult i64 %38, 4
  %40 = select i1 %39, i64 4, i64 %38
  %41 = icmp ule i64 %29, 4294967295
  br i1 %41, label %42, label %74

42:                                               ; preds = %35
  %43 = icmp ule i64 %40, 2305843009213693951
  br i1 %43, label %44, label %76

44:                                               ; preds = %42
  %45 = mul i64 8, %40
  %46 = add i64 %45, 8
  %47 = icmp eq ptr %32, null
  %48 = getelementptr i8, ptr %32, i64 -8
  %49 = select i1 %47, ptr null, ptr %48
  %50 = call ptr @GC_realloc(ptr %49, i64 %46)
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = icmp ne ptr %51, null
  %53 = icmp eq i64 %45, 0
  %54 = or i1 %52, %53
  br i1 %54, label %55, label %78

55:                                               ; preds = %44
  %56 = mul i64 8, %33
  %57 = getelementptr i8, ptr %51, i64 %56
  %58 = sub i64 %45, %56
  call void @llvm.memset.p0.i64(ptr %57, i8 0, i64 %58, i1 false)
  store ptr %51, ptr %30, align 8
  store i64 %40, ptr %31, align 8
  br label %60

59:                                               ; preds = %26
  br label %60

60:                                               ; preds = %55, %59
  %61 = phi ptr [ %32, %59 ], [ %51, %55 ]
  br label %62

62:                                               ; preds = %60
  %63 = getelementptr ptr, ptr %61, i64 %28
  store ptr %16, ptr %63, align 8
  store i64 %29, ptr %27, align 8
  %64 = load i32, ptr %1, align 4
  %65 = add i32 %64, 1
  store i32 %65, ptr %1, align 4
  br label %11

66:                                               ; preds = %11
  %67 = load ptr, ptr %2, align 8
  %68 = icmp eq ptr %67, null
  %69 = select i1 %68, ptr @__ts_empty_array, ptr %67
  %70 = getelementptr { ptr, i64, i64 }, ptr %69, i32 0, i32 1
  %71 = load i64, ptr %70, align 8
  store i64 %71, ptr %3, align 8
  %72 = load i64, ptr %3, align 8
  ret i64 %72

73:                                               ; preds = %78, %76, %74
  unreachable

74:                                               ; preds = %35
  %75 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 18, ptr null)
  br label %73

76:                                               ; preds = %42
  %77 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 18, ptr null)
  br label %73

78:                                               ; preds = %44
  %79 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 18, ptr null)
  br label %73
}

; Function Attrs: mustprogress
define i32 @throwString(double %0) #2 {
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
  %12 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %13 = add i64 1, %12
  %14 = call i64 @strlen(ptr %11)
  %15 = add i64 %13, %14
  %16 = add i64 %15, 8
  %17 = call ptr @GC_malloc(i64 %16)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr i8, ptr %18, i64 0
  %20 = sub i64 %15, 0
  %21 = call ptr @__memcpy_chk(ptr %19, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), i64 %12, i64 %20)
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
define internal void @.eh.copy.box.P1E(ptr %0, ptr %1) #2 {
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
define i32 @throwClass(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp oge double %4, 0.000000e+00
  br i1 %5, label %6, label %33

6:                                                ; preds = %1
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %E, ptr %8, i32 0, i32 0
  store ptr @E..vtbl, ptr %9, align 8
  %10 = load double, ptr %2, align 8
  %11 = load double, ptr %2, align 8
  %12 = call ptr @GC_malloc_atomic(i64 40)
  %13 = getelementptr i8, ptr %12, i64 8
  call void @__ts_number_to_string(ptr %13, double %11)
  %14 = icmp eq ptr %13, null
  %15 = select i1 %14, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %13
  %16 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638191598602154840, i64 8))
  %17 = add i64 1, %16
  %18 = call i64 @strlen(ptr %15)
  %19 = add i64 %17, %18
  %20 = add i64 %19, 8
  %21 = call ptr @GC_malloc(i64 %20)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr i8, ptr %22, i64 0
  %24 = sub i64 %19, 0
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12638191598602154840, i64 8), i64 %16, i64 %24)
  %26 = add i64 0, %16
  %27 = getelementptr i8, ptr %22, i64 %26
  %28 = sub i64 %19, %26
  %29 = call ptr @__memcpy_chk(ptr %27, ptr %15, i64 %18, i64 %28)
  %30 = add i64 %26, %18
  %31 = getelementptr i8, ptr %22, i64 %30
  store i8 0, ptr %31, align 1
  call void @E.constructor(ptr %8, double %10, ptr %22)
  %32 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %8, ptr %32, align 8
  call void @__cxa_throw(ptr %32, ptr @_ZTIP1E, ptr null)
  br label %35

33:                                               ; preds = %1
  store i32 1, ptr %3, align 4
  %34 = load i32, ptr %3, align 4
  ret i32 %34

35:                                               ; preds = %6
  unreachable
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
define i64 @caughtString(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca i64, align 8
  store double %0, ptr %4, align 8
  %6 = load double, ptr %4, align 8
  %7 = invoke i32 @throwString(double %6)
          to label %8 unwind label %9

8:                                                ; preds = %1
  store i64 0, ptr %5, align 8
  br label %152

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc10
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc10
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %24
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %51
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %60
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
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %92
  %93 = load ptr, ptr %2, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc4, %.noexc
  %102 = phi ptr [ %97, %.noexc ], [ %93, %.noexc4 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc11
  %112 = phi ptr [ %107, %.noexc11 ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc14
  %122 = phi ptr [ %117, %.noexc14 ], [ %112, %113 ]
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

130:                                              ; preds = %.noexc6, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc6 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc8, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc8 ]
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

142:                                              ; preds = %.noexc12, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc12 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc13, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc13 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %3, align 8
  %148 = invoke i64 @churn()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %147
  %149 = load ptr, ptr %3, align 8
  %150 = invoke ptr @"___unbox<!ts.string>"(ptr %149)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %151 = invoke i64 @strlen(ptr %150)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  store i64 %151, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %152

152:                                              ; preds = %8, %.noexc3
  %153 = load i64, ptr %5, align 8
  ret i64 %153

endcatch.resume:                                  ; preds = %114, %15, %24, %104, %9, %42, %51, %33, %60, %69, %92, %.noexc2, %.noexc1, %147, %94
  %154 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %154
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"___unbox<!ts.class<E, !ts.class_storage<E>>>"(ptr %0) #2 comdat {
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
  %39 = load ptr, ptr @E..rtti, align 8
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
  %70 = load ptr, ptr @E..rtti, align 8
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
  %83 = load ptr, ptr @E..rtti, align 8
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
define double @caughtClass(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  store double %0, ptr %4, align 8
  %6 = load double, ptr %4, align 8
  %7 = invoke i32 @throwClass(double %6)
          to label %8 unwind label %9

8:                                                ; preds = %1
  store double 0.000000e+00, ptr %5, align 8
  br label %160

9:                                                ; preds = %1
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %9
  %14 = icmp eq ptr %13, @_ZTIi
  br i1 %14, label %15, label %22

15:                                               ; preds = %.noexc10
  %16 = load i32, ptr %12, align 4
  %17 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %15
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 0
  store i64 4, ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %18, i32 0, i32 2
  store i32 %16, ptr %21, align 4
  br label %145

22:                                               ; preds = %.noexc10
  %23 = icmp eq ptr %13, @_ZTIj
  br i1 %23, label %24, label %31

24:                                               ; preds = %22
  %25 = load i32, ptr %12, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %24
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
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %33
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %42
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
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %51
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %60
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
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %69
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %92
  %93 = load ptr, ptr %2, align 8
  br label %101

94:                                               ; preds = %88
  %95 = load ptr, ptr %12, align 8
  %96 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %94
  %97 = getelementptr i8, ptr %96, i64 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 0
  store i64 8, ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %97, i32 0, i32 2
  store ptr %95, ptr %100, align 8
  br label %101

101:                                              ; preds = %.noexc7, %.noexc5
  %102 = phi ptr [ %97, %.noexc5 ], [ %93, %.noexc7 ]
  br label %103

103:                                              ; preds = %101
  br label %111

104:                                              ; preds = %83
  %105 = load ptr, ptr %12, align 8
  %106 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %104
  %107 = getelementptr i8, ptr %106, i64 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 0
  store i64 8, ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %107, i32 0, i32 2
  store ptr %105, ptr %110, align 8
  br label %111

111:                                              ; preds = %103, %.noexc
  %112 = phi ptr [ %107, %.noexc ], [ %102, %103 ]
  br label %113

113:                                              ; preds = %111
  br label %121

114:                                              ; preds = %80
  %115 = load ptr, ptr %12, align 8
  %116 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %114
  %117 = getelementptr i8, ptr %116, i64 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 0
  store i64 8, ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %117, i32 0, i32 2
  store ptr %115, ptr %120, align 8
  br label %121

121:                                              ; preds = %113, %.noexc11
  %122 = phi ptr [ %117, %.noexc11 ], [ %112, %113 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %78, %123
  %125 = phi ptr [ %122, %123 ], [ %79, %78 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc14, %126
  %128 = phi ptr [ %125, %126 ], [ %72, %.noexc14 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc9, %129
  %131 = phi ptr [ %128, %129 ], [ %63, %.noexc9 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc15, %132
  %134 = phi ptr [ %131, %132 ], [ %54, %.noexc15 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc8, %135
  %137 = phi ptr [ %134, %135 ], [ %45, %.noexc8 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc12, %138
  %140 = phi ptr [ %137, %138 ], [ %36, %.noexc12 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc13, %141
  %143 = phi ptr [ %140, %141 ], [ %27, %.noexc13 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc6, %144
  %146 = phi ptr [ %143, %144 ], [ %18, %.noexc6 ]
  br label %147

147:                                              ; preds = %145
  store ptr %146, ptr %3, align 8
  %148 = invoke i64 @churn()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %147
  %149 = load ptr, ptr %3, align 8
  %150 = invoke ptr @"___unbox<!ts.class<E, !ts.class_storage<E>>>"(ptr %149)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %151 = getelementptr %E, ptr %150, i32 0, i32 1
  %152 = load double, ptr %151, align 8
  %153 = load ptr, ptr %3, align 8
  %154 = invoke ptr @"___unbox<!ts.class<E, !ts.class_storage<E>>>"(ptr %153)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  %155 = getelementptr %E, ptr %154, i32 0, i32 2
  %156 = load ptr, ptr %155, align 8
  %157 = invoke i64 @strlen(ptr %156)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %158 = sitofp i64 %157 to double
  %159 = fadd double %152, %158
  store double %159, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %160

160:                                              ; preds = %8, %.noexc4
  %161 = load double, ptr %5, align 8
  ret double %161

endcatch.resume:                                  ; preds = %51, %69, %24, %33, %114, %9, %60, %42, %92, %15, %94, %.noexc3, %.noexc2, %.noexc1, %147, %104
  %162 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %162
}

; Function Attrs: mustprogress
define i32 @noVariable(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = invoke i32 @throwClass(double %4)
          to label %6 unwind label %7

6:                                                ; preds = %1
  store i32 0, ptr %3, align 4
  br label %11

7:                                                ; preds = %1
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %11

11:                                               ; preds = %6, %7
  %12 = load i32, ptr %3, align 4
  ret i32 %12
}

; Function Attrs: mustprogress
define i32 @inLoop(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca double, align 8
  %7 = alloca i32, align 4
  store double %0, ptr %6, align 8
  store i32 0, ptr %5, align 4
  store i32 0, ptr %4, align 4
  br label %8

8:                                                ; preds = %168, %1
  %9 = load i32, ptr %4, align 4
  %10 = load double, ptr %6, align 8
  %11 = sitofp i32 %9 to double
  %12 = fcmp olt double %11, %10
  br i1 %12, label %13, label %171

13:                                               ; preds = %8
  %14 = load i32, ptr %4, align 4
  %15 = sitofp i32 %14 to double
  %16 = invoke i32 @throwString(double %15)
          to label %17 unwind label %18

17:                                               ; preds = %13
  br label %168

18:                                               ; preds = %13
  %19 = landingpad { ptr, i32 }
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  %22 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %18
  %23 = icmp eq ptr %22, @_ZTIi
  br i1 %23, label %24, label %31

24:                                               ; preds = %.noexc5
  %25 = load i32, ptr %21, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %154

31:                                               ; preds = %.noexc5
  %32 = icmp eq ptr %22, @_ZTIj
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load i32, ptr %21, align 4
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %33
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 0
  store i64 4, ptr %37, align 8
  %38 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %38, align 8
  %39 = getelementptr { i64, ptr, i32 }, ptr %36, i32 0, i32 2
  store i32 %34, ptr %39, align 4
  br label %151

40:                                               ; preds = %31
  %41 = icmp eq ptr %22, @_ZTId
  br i1 %41, label %42, label %49

42:                                               ; preds = %40
  %43 = load double, ptr %21, align 8
  %44 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %42
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 0
  store i64 8, ptr %46, align 8
  %47 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %47, align 8
  %48 = getelementptr { i64, ptr, double }, ptr %45, i32 0, i32 2
  store double %43, ptr %48, align 8
  br label %148

49:                                               ; preds = %40
  %50 = icmp eq ptr %22, @_ZTIf
  br i1 %50, label %51, label %58

51:                                               ; preds = %49
  %52 = load float, ptr %21, align 4
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 0
  store i64 4, ptr %55, align 8
  %56 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, float }, ptr %54, i32 0, i32 2
  store float %52, ptr %57, align 4
  br label %145

58:                                               ; preds = %49
  %59 = icmp eq ptr %22, @_ZTIb
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i1, ptr %21, align 1
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 0
  store i64 1, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i1 }, ptr %63, i32 0, i32 2
  store i1 %61, ptr %66, align 1
  br label %142

67:                                               ; preds = %58
  %68 = icmp eq ptr %22, @_ZTIx
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load i64, ptr %21, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, i64 }, ptr %72, i32 0, i32 2
  store i64 %70, ptr %75, align 8
  br label %139

76:                                               ; preds = %67
  %77 = icmp eq ptr %22, @_ZTIPc
  br i1 %77, label %78, label %85

78:                                               ; preds = %76
  %79 = load ptr, ptr %21, align 8
  %80 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %78
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 0
  store i64 8, ptr %82, align 8
  %83 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %83, align 8
  %84 = getelementptr { i64, ptr, ptr }, ptr %81, i32 0, i32 2
  store ptr %79, ptr %84, align 8
  br label %136

85:                                               ; preds = %76
  %86 = icmp eq ptr %22, @_ZTIPv
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = load ptr, ptr %21, align 8
  br label %133

89:                                               ; preds = %85
  %90 = load ptr, ptr %22, align 8
  %91 = icmp eq ptr %90, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %91, label %92, label %123

92:                                               ; preds = %89
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 2
  %94 = load i32, ptr %93, align 4
  %95 = and i32 %94, 1073741824
  %96 = icmp ne i32 %95, 0
  br i1 %96, label %97, label %113

97:                                               ; preds = %92
  %98 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %22, i32 0, i32 4
  %99 = load ptr, ptr %98, align 8
  %100 = icmp ne ptr %99, null
  br i1 %100, label %101, label %103

101:                                              ; preds = %97
  invoke void %99(ptr %2, ptr %21)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %101
  %102 = load ptr, ptr %2, align 8
  br label %110

103:                                              ; preds = %97
  %104 = load ptr, ptr %21, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %.noexc9, %.noexc7
  %111 = phi ptr [ %106, %.noexc7 ], [ %102, %.noexc9 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %92
  %114 = load ptr, ptr %21, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc3
  %121 = phi ptr [ %116, %.noexc3 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %89
  %124 = load ptr, ptr %21, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc12
  %131 = phi ptr [ %126, %.noexc12 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %87, %132
  %134 = phi ptr [ %131, %132 ], [ %88, %87 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc4, %135
  %137 = phi ptr [ %134, %135 ], [ %81, %.noexc4 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc, %138
  %140 = phi ptr [ %137, %138 ], [ %72, %.noexc ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc6, %141
  %143 = phi ptr [ %140, %141 ], [ %63, %.noexc6 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc1, %144
  %146 = phi ptr [ %143, %144 ], [ %54, %.noexc1 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc13, %147
  %149 = phi ptr [ %146, %147 ], [ %45, %.noexc13 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc2, %150
  %152 = phi ptr [ %149, %150 ], [ %36, %.noexc2 ]
  br label %153

153:                                              ; preds = %151
  br label %154

154:                                              ; preds = %.noexc8, %153
  %155 = phi ptr [ %152, %153 ], [ %27, %.noexc8 ]
  br label %156

156:                                              ; preds = %154
  store ptr %155, ptr %3, align 8
  %157 = load i32, ptr %4, align 4
  %158 = icmp eq i32 %157, 2
  br i1 %158, label %159, label %160

159:                                              ; preds = %156
  call void @__cxa_end_catch()
  br label %171

160:                                              ; preds = %156
  %161 = load i32, ptr %5, align 4
  %162 = load ptr, ptr %3, align 8
  %163 = invoke ptr @"___unbox<!ts.string>"(ptr %162)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %160
  %164 = invoke i64 @strlen(ptr %163)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  %165 = sext i32 %161 to i64
  %166 = add i64 %165, %164
  %167 = trunc i64 %166 to i32
  store i32 %167, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %168

168:                                              ; preds = %17, %.noexc11
  %169 = load i32, ptr %4, align 4
  %170 = add i32 %169, 1
  store i32 %170, ptr %4, align 4
  br label %8

171:                                              ; preds = %159, %8
  %172 = load i32, ptr %5, align 4
  store i32 %172, ptr %7, align 4
  %173 = load i32, ptr %7, align 4
  ret i32 %173

endcatch.resume:                                  ; preds = %42, %123, %.noexc10, %160, %101, %24, %103, %60, %18, %78, %113, %33, %51, %69
  %174 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %174
}

; Function Attrs: mustprogress
define void @rethrow(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store double %0, ptr %4, align 8
  %5 = load double, ptr %4, align 8
  %6 = invoke i32 @throwString(double %5)
          to label %7 unwind label %8

7:                                                ; preds = %1
  ret void

8:                                                ; preds = %1
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %8
  %13 = icmp eq ptr %12, @_ZTIi
  br i1 %13, label %14, label %21

14:                                               ; preds = %.noexc9
  %15 = load i32, ptr %11, align 4
  %16 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %14
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 0
  store i64 4, ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 2
  store i32 %15, ptr %20, align 4
  br label %144

21:                                               ; preds = %.noexc9
  %22 = icmp eq ptr %12, @_ZTIj
  br i1 %22, label %23, label %30

23:                                               ; preds = %21
  %24 = load i32, ptr %11, align 4
  %25 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %23
  %26 = getelementptr i8, ptr %25, i64 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %26, i32 0, i32 0
  store i64 4, ptr %27, align 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %26, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %26, i32 0, i32 2
  store i32 %24, ptr %29, align 4
  br label %141

30:                                               ; preds = %21
  %31 = icmp eq ptr %12, @_ZTId
  br i1 %31, label %32, label %39

32:                                               ; preds = %30
  %33 = load double, ptr %11, align 8
  %34 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %32
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr { i64, ptr, double }, ptr %35, i32 0, i32 0
  store i64 8, ptr %36, align 8
  %37 = getelementptr { i64, ptr, double }, ptr %35, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %37, align 8
  %38 = getelementptr { i64, ptr, double }, ptr %35, i32 0, i32 2
  store double %33, ptr %38, align 8
  br label %138

39:                                               ; preds = %30
  %40 = icmp eq ptr %12, @_ZTIf
  br i1 %40, label %41, label %48

41:                                               ; preds = %39
  %42 = load float, ptr %11, align 4
  %43 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %41
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = getelementptr { i64, ptr, float }, ptr %44, i32 0, i32 0
  store i64 4, ptr %45, align 8
  %46 = getelementptr { i64, ptr, float }, ptr %44, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %46, align 8
  %47 = getelementptr { i64, ptr, float }, ptr %44, i32 0, i32 2
  store float %42, ptr %47, align 4
  br label %135

48:                                               ; preds = %39
  %49 = icmp eq ptr %12, @_ZTIb
  br i1 %49, label %50, label %57

50:                                               ; preds = %48
  %51 = load i1, ptr %11, align 1
  %52 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %50
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %53, i32 0, i32 0
  store i64 1, ptr %54, align 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %53, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %55, align 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %53, i32 0, i32 2
  store i1 %51, ptr %56, align 1
  br label %132

57:                                               ; preds = %48
  %58 = icmp eq ptr %12, @_ZTIx
  br i1 %58, label %59, label %66

59:                                               ; preds = %57
  %60 = load i64, ptr %11, align 8
  %61 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %59
  %62 = getelementptr i8, ptr %61, i64 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %62, i32 0, i32 0
  store i64 8, ptr %63, align 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %62, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %64, align 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %62, i32 0, i32 2
  store i64 %60, ptr %65, align 8
  br label %129

66:                                               ; preds = %57
  %67 = icmp eq ptr %12, @_ZTIPc
  br i1 %67, label %68, label %75

68:                                               ; preds = %66
  %69 = load ptr, ptr %11, align 8
  %70 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %68
  %71 = getelementptr i8, ptr %70, i64 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %71, i32 0, i32 0
  store i64 8, ptr %72, align 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %71, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %73, align 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %71, i32 0, i32 2
  store ptr %69, ptr %74, align 8
  br label %126

75:                                               ; preds = %66
  %76 = icmp eq ptr %12, @_ZTIPv
  br i1 %76, label %77, label %79

77:                                               ; preds = %75
  %78 = load ptr, ptr %11, align 8
  br label %123

79:                                               ; preds = %75
  %80 = load ptr, ptr %12, align 8
  %81 = icmp eq ptr %80, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %81, label %82, label %113

82:                                               ; preds = %79
  %83 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %12, i32 0, i32 2
  %84 = load i32, ptr %83, align 4
  %85 = and i32 %84, 1073741824
  %86 = icmp ne i32 %85, 0
  br i1 %86, label %87, label %103

87:                                               ; preds = %82
  %88 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %12, i32 0, i32 4
  %89 = load ptr, ptr %88, align 8
  %90 = icmp ne ptr %89, null
  br i1 %90, label %91, label %93

91:                                               ; preds = %87
  invoke void %89(ptr %2, ptr %11)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %91
  %92 = load ptr, ptr %2, align 8
  br label %100

93:                                               ; preds = %87
  %94 = load ptr, ptr %11, align 8
  %95 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %93
  %96 = getelementptr i8, ptr %95, i64 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 0
  store i64 8, ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 2
  store ptr %94, ptr %99, align 8
  br label %100

100:                                              ; preds = %.noexc11, %.noexc5
  %101 = phi ptr [ %96, %.noexc5 ], [ %92, %.noexc11 ]
  br label %102

102:                                              ; preds = %100
  br label %110

103:                                              ; preds = %82
  %104 = load ptr, ptr %11, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %102, %.noexc7
  %111 = phi ptr [ %106, %.noexc7 ], [ %101, %102 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %79
  %114 = load ptr, ptr %11, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc2
  %121 = phi ptr [ %116, %.noexc2 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %123

123:                                              ; preds = %77, %122
  %124 = phi ptr [ %121, %122 ], [ %78, %77 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %.noexc10, %125
  %127 = phi ptr [ %124, %125 ], [ %71, %.noexc10 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc1, %128
  %130 = phi ptr [ %127, %128 ], [ %62, %.noexc1 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc8, %131
  %133 = phi ptr [ %130, %131 ], [ %53, %.noexc8 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc, %134
  %136 = phi ptr [ %133, %134 ], [ %44, %.noexc ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc4, %137
  %139 = phi ptr [ %136, %137 ], [ %35, %.noexc4 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc6, %140
  %142 = phi ptr [ %139, %140 ], [ %26, %.noexc6 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc3, %143
  %145 = phi ptr [ %142, %143 ], [ %17, %.noexc3 ]
  br label %146

146:                                              ; preds = %144
  store ptr %145, ptr %3, align 8
  %147 = load ptr, ptr %3, align 8
  call void @__cxa_end_catch()
  %148 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %147, ptr %148, align 8
  call void @__cxa_throw(ptr %148, ptr @_ZTIPv, ptr null)
  br label %149

149:                                              ; preds = %146
  unreachable

endcatch.resume:                                  ; preds = %91, %68, %8, %50, %103, %23, %93, %32, %14, %113, %59, %41
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define ptr @returned(double %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  store double %0, ptr %4, align 8
  %6 = load double, ptr %4, align 8
  %7 = invoke i32 @throwString(double %6)
          to label %8 unwind label %14

8:                                                ; preds = %1
  %9 = call ptr @GC_malloc(i64 32)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr { i64, ptr, ptr }, ptr %10, i32 0, i32 0
  store i64 8, ptr %11, align 8
  %12 = getelementptr { i64, ptr, ptr }, ptr %10, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_14527407094101050897_null, i64 40), ptr %12, align 8
  %13 = getelementptr { i64, ptr, ptr }, ptr %10, i32 0, i32 2
  store ptr null, ptr %13, align 8
  store ptr %10, ptr %5, align 8
  br label %154

14:                                               ; preds = %1
  %15 = landingpad { ptr, i32 }
          catch ptr null
  %16 = extractvalue { ptr, i32 } %15, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %14
  %19 = icmp eq ptr %18, @_ZTIi
  br i1 %19, label %20, label %27

20:                                               ; preds = %.noexc
  %21 = load i32, ptr %17, align 4
  %22 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %20
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 0
  store i64 4, ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 2
  store i32 %21, ptr %26, align 4
  br label %150

27:                                               ; preds = %.noexc
  %28 = icmp eq ptr %18, @_ZTIj
  br i1 %28, label %29, label %36

29:                                               ; preds = %27
  %30 = load i32, ptr %17, align 4
  %31 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %29
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %32, i32 0, i32 0
  store i64 4, ptr %33, align 8
  %34 = getelementptr { i64, ptr, i32 }, ptr %32, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %34, align 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %32, i32 0, i32 2
  store i32 %30, ptr %35, align 4
  br label %147

36:                                               ; preds = %27
  %37 = icmp eq ptr %18, @_ZTId
  br i1 %37, label %38, label %45

38:                                               ; preds = %36
  %39 = load double, ptr %17, align 8
  %40 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %38
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr { i64, ptr, double }, ptr %41, i32 0, i32 0
  store i64 8, ptr %42, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %41, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %43, align 8
  %44 = getelementptr { i64, ptr, double }, ptr %41, i32 0, i32 2
  store double %39, ptr %44, align 8
  br label %144

45:                                               ; preds = %36
  %46 = icmp eq ptr %18, @_ZTIf
  br i1 %46, label %47, label %54

47:                                               ; preds = %45
  %48 = load float, ptr %17, align 4
  %49 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %47
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr { i64, ptr, float }, ptr %50, i32 0, i32 0
  store i64 4, ptr %51, align 8
  %52 = getelementptr { i64, ptr, float }, ptr %50, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %52, align 8
  %53 = getelementptr { i64, ptr, float }, ptr %50, i32 0, i32 2
  store float %48, ptr %53, align 4
  br label %141

54:                                               ; preds = %45
  %55 = icmp eq ptr %18, @_ZTIb
  br i1 %55, label %56, label %63

56:                                               ; preds = %54
  %57 = load i1, ptr %17, align 1
  %58 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %56
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %59, i32 0, i32 0
  store i64 1, ptr %60, align 8
  %61 = getelementptr { i64, ptr, i1 }, ptr %59, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %61, align 8
  %62 = getelementptr { i64, ptr, i1 }, ptr %59, i32 0, i32 2
  store i1 %57, ptr %62, align 1
  br label %138

63:                                               ; preds = %54
  %64 = icmp eq ptr %18, @_ZTIx
  br i1 %64, label %65, label %72

65:                                               ; preds = %63
  %66 = load i64, ptr %17, align 8
  %67 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %65
  %68 = getelementptr i8, ptr %67, i64 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %68, i32 0, i32 0
  store i64 8, ptr %69, align 8
  %70 = getelementptr { i64, ptr, i64 }, ptr %68, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %70, align 8
  %71 = getelementptr { i64, ptr, i64 }, ptr %68, i32 0, i32 2
  store i64 %66, ptr %71, align 8
  br label %135

72:                                               ; preds = %63
  %73 = icmp eq ptr %18, @_ZTIPc
  br i1 %73, label %74, label %81

74:                                               ; preds = %72
  %75 = load ptr, ptr %17, align 8
  %76 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %74
  %77 = getelementptr i8, ptr %76, i64 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %77, i32 0, i32 0
  store i64 8, ptr %78, align 8
  %79 = getelementptr { i64, ptr, ptr }, ptr %77, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %79, align 8
  %80 = getelementptr { i64, ptr, ptr }, ptr %77, i32 0, i32 2
  store ptr %75, ptr %80, align 8
  br label %132

81:                                               ; preds = %72
  %82 = icmp eq ptr %18, @_ZTIPv
  br i1 %82, label %83, label %85

83:                                               ; preds = %81
  %84 = load ptr, ptr %17, align 8
  br label %129

85:                                               ; preds = %81
  %86 = load ptr, ptr %18, align 8
  %87 = icmp eq ptr %86, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %87, label %88, label %119

88:                                               ; preds = %85
  %89 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %18, i32 0, i32 2
  %90 = load i32, ptr %89, align 4
  %91 = and i32 %90, 1073741824
  %92 = icmp ne i32 %91, 0
  br i1 %92, label %93, label %109

93:                                               ; preds = %88
  %94 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %18, i32 0, i32 4
  %95 = load ptr, ptr %94, align 8
  %96 = icmp ne ptr %95, null
  br i1 %96, label %97, label %99

97:                                               ; preds = %93
  invoke void %95(ptr %2, ptr %17)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %97
  %98 = load ptr, ptr %2, align 8
  br label %106

99:                                               ; preds = %93
  %100 = load ptr, ptr %17, align 8
  %101 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %99
  %102 = getelementptr i8, ptr %101, i64 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 0
  store i64 8, ptr %103, align 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %104, align 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %102, i32 0, i32 2
  store ptr %100, ptr %105, align 8
  br label %106

106:                                              ; preds = %.noexc11, %.noexc3
  %107 = phi ptr [ %102, %.noexc3 ], [ %98, %.noexc11 ]
  br label %108

108:                                              ; preds = %106
  br label %116

109:                                              ; preds = %88
  %110 = load ptr, ptr %17, align 8
  %111 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %109
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 0
  store i64 8, ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %112, i32 0, i32 2
  store ptr %110, ptr %115, align 8
  br label %116

116:                                              ; preds = %108, %.noexc6
  %117 = phi ptr [ %112, %.noexc6 ], [ %107, %108 ]
  br label %118

118:                                              ; preds = %116
  br label %126

119:                                              ; preds = %85
  %120 = load ptr, ptr %17, align 8
  %121 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %119
  %122 = getelementptr i8, ptr %121, i64 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %122, i32 0, i32 0
  store i64 8, ptr %123, align 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %122, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %124, align 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %122, i32 0, i32 2
  store ptr %120, ptr %125, align 8
  br label %126

126:                                              ; preds = %118, %.noexc2
  %127 = phi ptr [ %122, %.noexc2 ], [ %117, %118 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %83, %128
  %130 = phi ptr [ %127, %128 ], [ %84, %83 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc8, %131
  %133 = phi ptr [ %130, %131 ], [ %77, %.noexc8 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc10, %134
  %136 = phi ptr [ %133, %134 ], [ %68, %.noexc10 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc7, %137
  %139 = phi ptr [ %136, %137 ], [ %59, %.noexc7 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc1, %140
  %142 = phi ptr [ %139, %140 ], [ %50, %.noexc1 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc4, %143
  %145 = phi ptr [ %142, %143 ], [ %41, %.noexc4 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc5, %146
  %148 = phi ptr [ %145, %146 ], [ %32, %.noexc5 ]
  br label %149

149:                                              ; preds = %147
  br label %150

150:                                              ; preds = %.noexc9, %149
  %151 = phi ptr [ %148, %149 ], [ %23, %.noexc9 ]
  br label %152

152:                                              ; preds = %150
  store ptr %151, ptr %3, align 8
  %153 = load ptr, ptr %3, align 8
  store ptr %153, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %154

154:                                              ; preds = %8, %152
  %155 = load ptr, ptr %5, align 8
  ret ptr %155

endcatch.resume:                                  ; preds = %97, %65, %20, %74, %56, %109, %29, %38, %99, %119, %47, %14
  %156 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %156
}

; Function Attrs: mustprogress
define i32 @localNumber() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 42, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %184 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr null
  %7 = extractvalue { ptr, i32 } %6, 0
  %8 = call ptr @__cxa_begin_catch(ptr %7)
  %9 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %5
  %10 = icmp eq ptr %9, @_ZTIi
  br i1 %10, label %11, label %18

11:                                               ; preds = %.noexc9
  %12 = load i32, ptr %8, align 4
  %13 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %11
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr { i64, ptr, i32 }, ptr %14, i32 0, i32 0
  store i64 4, ptr %15, align 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %14, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %14, i32 0, i32 2
  store i32 %12, ptr %17, align 4
  br label %141

18:                                               ; preds = %.noexc9
  %19 = icmp eq ptr %9, @_ZTIj
  br i1 %19, label %20, label %27

20:                                               ; preds = %18
  %21 = load i32, ptr %8, align 4
  %22 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %20
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 0
  store i64 4, ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %23, i32 0, i32 2
  store i32 %21, ptr %26, align 4
  br label %138

27:                                               ; preds = %18
  %28 = icmp eq ptr %9, @_ZTId
  br i1 %28, label %29, label %36

29:                                               ; preds = %27
  %30 = load double, ptr %8, align 8
  %31 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %29
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr { i64, ptr, double }, ptr %32, i32 0, i32 0
  store i64 8, ptr %33, align 8
  %34 = getelementptr { i64, ptr, double }, ptr %32, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %32, i32 0, i32 2
  store double %30, ptr %35, align 8
  br label %135

36:                                               ; preds = %27
  %37 = icmp eq ptr %9, @_ZTIf
  br i1 %37, label %38, label %45

38:                                               ; preds = %36
  %39 = load float, ptr %8, align 4
  %40 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %38
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr { i64, ptr, float }, ptr %41, i32 0, i32 0
  store i64 4, ptr %42, align 8
  %43 = getelementptr { i64, ptr, float }, ptr %41, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %41, i32 0, i32 2
  store float %39, ptr %44, align 4
  br label %132

45:                                               ; preds = %36
  %46 = icmp eq ptr %9, @_ZTIb
  br i1 %46, label %47, label %54

47:                                               ; preds = %45
  %48 = load i1, ptr %8, align 1
  %49 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %47
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr { i64, ptr, i1 }, ptr %50, i32 0, i32 0
  store i64 1, ptr %51, align 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %50, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %50, i32 0, i32 2
  store i1 %48, ptr %53, align 1
  br label %129

54:                                               ; preds = %45
  %55 = icmp eq ptr %9, @_ZTIx
  br i1 %55, label %56, label %63

56:                                               ; preds = %54
  %57 = load i64, ptr %8, align 8
  %58 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %56
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr { i64, ptr, i64 }, ptr %59, i32 0, i32 0
  store i64 8, ptr %60, align 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %59, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %59, i32 0, i32 2
  store i64 %57, ptr %62, align 8
  br label %126

63:                                               ; preds = %54
  %64 = icmp eq ptr %9, @_ZTIPc
  br i1 %64, label %65, label %72

65:                                               ; preds = %63
  %66 = load ptr, ptr %8, align 8
  %67 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %65
  %68 = getelementptr i8, ptr %67, i64 8
  %69 = getelementptr { i64, ptr, ptr }, ptr %68, i32 0, i32 0
  store i64 8, ptr %69, align 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %68, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %68, i32 0, i32 2
  store ptr %66, ptr %71, align 8
  br label %123

72:                                               ; preds = %63
  %73 = icmp eq ptr %9, @_ZTIPv
  br i1 %73, label %74, label %76

74:                                               ; preds = %72
  %75 = load ptr, ptr %8, align 8
  br label %120

76:                                               ; preds = %72
  %77 = load ptr, ptr %9, align 8
  %78 = icmp eq ptr %77, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %78, label %79, label %110

79:                                               ; preds = %76
  %80 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %9, i32 0, i32 2
  %81 = load i32, ptr %80, align 4
  %82 = and i32 %81, 1073741824
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %100

84:                                               ; preds = %79
  %85 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %9, i32 0, i32 4
  %86 = load ptr, ptr %85, align 8
  %87 = icmp ne ptr %86, null
  br i1 %87, label %88, label %90

88:                                               ; preds = %84
  invoke void %86(ptr %1, ptr %8)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %88
  %89 = load ptr, ptr %1, align 8
  br label %97

90:                                               ; preds = %84
  %91 = load ptr, ptr %8, align 8
  %92 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %90
  %93 = getelementptr i8, ptr %92, i64 8
  %94 = getelementptr { i64, ptr, ptr }, ptr %93, i32 0, i32 0
  store i64 8, ptr %94, align 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %93, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %93, i32 0, i32 2
  store ptr %91, ptr %96, align 8
  br label %97

97:                                               ; preds = %.noexc5, %.noexc3
  %98 = phi ptr [ %93, %.noexc3 ], [ %89, %.noexc5 ]
  br label %99

99:                                               ; preds = %97
  br label %107

100:                                              ; preds = %79
  %101 = load ptr, ptr %8, align 8
  %102 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %100
  %103 = getelementptr i8, ptr %102, i64 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 0
  store i64 8, ptr %104, align 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %103, i32 0, i32 2
  store ptr %101, ptr %106, align 8
  br label %107

107:                                              ; preds = %99, %.noexc
  %108 = phi ptr [ %103, %.noexc ], [ %98, %99 ]
  br label %109

109:                                              ; preds = %107
  br label %117

110:                                              ; preds = %76
  %111 = load ptr, ptr %8, align 8
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 0
  store i64 8, ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 2
  store ptr %111, ptr %116, align 8
  br label %117

117:                                              ; preds = %109, %.noexc10
  %118 = phi ptr [ %113, %.noexc10 ], [ %108, %109 ]
  br label %119

119:                                              ; preds = %117
  br label %120

120:                                              ; preds = %74, %119
  %121 = phi ptr [ %118, %119 ], [ %75, %74 ]
  br label %122

122:                                              ; preds = %120
  br label %123

123:                                              ; preds = %.noexc13, %122
  %124 = phi ptr [ %121, %122 ], [ %68, %.noexc13 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %.noexc8, %125
  %127 = phi ptr [ %124, %125 ], [ %59, %.noexc8 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc7, %128
  %130 = phi ptr [ %127, %128 ], [ %50, %.noexc7 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc4, %131
  %133 = phi ptr [ %130, %131 ], [ %41, %.noexc4 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc6, %134
  %136 = phi ptr [ %133, %134 ], [ %32, %.noexc6 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc11, %137
  %139 = phi ptr [ %136, %137 ], [ %23, %.noexc11 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc1, %140
  %142 = phi ptr [ %139, %140 ], [ %14, %.noexc1 ]
  br label %143

143:                                              ; preds = %141
  store ptr %142, ptr %2, align 8
  %144 = load ptr, ptr %2, align 8
  %145 = getelementptr { i64, ptr, i8 }, ptr %144, i32 0, i32 1
  %146 = load ptr, ptr %145, align 8
  %147 = ptrtoint ptr %146 to i64
  %148 = icmp ne i64 %147, 0
  %149 = zext i1 %148 to i32
  %150 = and i32 %149, 1
  %151 = icmp ne i32 %150, 0
  br i1 %151, label %152, label %155

152:                                              ; preds = %143
  %153 = invoke i32 @strcmp(ptr %146, ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8))
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %152
  %154 = icmp eq i32 %153, 0
  br label %157

155:                                              ; preds = %143
  %156 = icmp eq i64 %147, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8) to i64)
  br label %157

157:                                              ; preds = %.noexc2, %155
  %158 = phi i1 [ %156, %155 ], [ %154, %.noexc2 ]
  br label %159

159:                                              ; preds = %157
  br i1 %158, label %177, label %160

160:                                              ; preds = %159
  %161 = load ptr, ptr %2, align 8
  %162 = getelementptr { i64, ptr, i8 }, ptr %161, i32 0, i32 1
  %163 = load ptr, ptr %162, align 8
  %164 = ptrtoint ptr %163 to i64
  %165 = icmp ne i64 %164, 0
  %166 = zext i1 %165 to i32
  %167 = and i32 %166, 1
  %168 = icmp ne i32 %167, 0
  br i1 %168, label %169, label %172

169:                                              ; preds = %160
  %170 = invoke i32 @strcmp(ptr %163, ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8))
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %169
  %171 = icmp eq i32 %170, 0
  br label %174

172:                                              ; preds = %160
  %173 = icmp eq i64 %164, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9237349086447201248, i64 8) to i64)
  br label %174

174:                                              ; preds = %.noexc12, %172
  %175 = phi i1 [ %173, %172 ], [ %171, %.noexc12 ]
  br label %176

176:                                              ; preds = %174
  br label %177

177:                                              ; preds = %176, %159
  %178 = phi i1 [ %175, %176 ], [ %158, %159 ]
  br i1 %178, label %179, label %180

179:                                              ; preds = %177
  br label %181

180:                                              ; preds = %177
  br label %181

181:                                              ; preds = %179, %180
  %182 = phi i32 [ 0, %180 ], [ 1, %179 ]
  store i32 %182, ptr %3, align 4
  call void @__cxa_end_catch()
  %183 = load i32, ptr %3, align 4
  ret i32 %183

184:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %65, %169, %20, %110, %5, %56, %47, %29, %88, %38, %90, %152, %11, %100
  %185 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %185
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca double, align 8
  store double 0.000000e+00, ptr %6, align 8
  store i32 0, ptr %5, align 4
  br label %7

7:                                                ; preds = %378, %0
  %8 = load i32, ptr %5, align 4
  %9 = icmp slt i32 %8, 100
  br i1 %9, label %10, label %385

10:                                               ; preds = %7
  %11 = load double, ptr %6, align 8
  %12 = load i32, ptr %5, align 4
  %13 = sitofp i32 %12 to double
  %14 = call i64 @caughtString(double %13)
  %15 = load i32, ptr %5, align 4
  %16 = sitofp i32 %15 to double
  %17 = call double @caughtClass(double %16)
  %18 = sitofp i64 %14 to double
  %19 = fadd double %18, %17
  %20 = load i32, ptr %5, align 4
  %21 = sitofp i32 %20 to double
  %22 = call i32 @noVariable(double %21)
  %23 = sitofp i32 %22 to double
  %24 = fadd double %19, %23
  %25 = call i32 @inLoop(double 5.000000e+00)
  %26 = sitofp i32 %25 to double
  %27 = fadd double %24, %26
  %28 = fadd double %11, %27
  store double %28, ptr %6, align 8
  %29 = load i32, ptr %5, align 4
  %30 = sitofp i32 %29 to double
  invoke void @rethrow(double %30)
          to label %31 unwind label %32

31:                                               ; preds = %10
  br label %178

32:                                               ; preds = %10
  %33 = landingpad { ptr, i32 }
          catch ptr null
  %34 = extractvalue { ptr, i32 } %33, 0
  %35 = call ptr @__cxa_begin_catch(ptr %34)
  %36 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %32
  %37 = icmp eq ptr %36, @_ZTIi
  br i1 %37, label %38, label %45

38:                                               ; preds = %.noexc24
  %39 = load i32, ptr %35, align 4
  %40 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %38
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = getelementptr { i64, ptr, i32 }, ptr %41, i32 0, i32 0
  store i64 4, ptr %42, align 8
  %43 = getelementptr { i64, ptr, i32 }, ptr %41, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %43, align 8
  %44 = getelementptr { i64, ptr, i32 }, ptr %41, i32 0, i32 2
  store i32 %39, ptr %44, align 4
  br label %168

45:                                               ; preds = %.noexc24
  %46 = icmp eq ptr %36, @_ZTIj
  br i1 %46, label %47, label %54

47:                                               ; preds = %45
  %48 = load i32, ptr %35, align 4
  %49 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %47
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = getelementptr { i64, ptr, i32 }, ptr %50, i32 0, i32 0
  store i64 4, ptr %51, align 8
  %52 = getelementptr { i64, ptr, i32 }, ptr %50, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %52, align 8
  %53 = getelementptr { i64, ptr, i32 }, ptr %50, i32 0, i32 2
  store i32 %48, ptr %53, align 4
  br label %165

54:                                               ; preds = %45
  %55 = icmp eq ptr %36, @_ZTId
  br i1 %55, label %56, label %63

56:                                               ; preds = %54
  %57 = load double, ptr %35, align 8
  %58 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %56
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = getelementptr { i64, ptr, double }, ptr %59, i32 0, i32 0
  store i64 8, ptr %60, align 8
  %61 = getelementptr { i64, ptr, double }, ptr %59, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %61, align 8
  %62 = getelementptr { i64, ptr, double }, ptr %59, i32 0, i32 2
  store double %57, ptr %62, align 8
  br label %162

63:                                               ; preds = %54
  %64 = icmp eq ptr %36, @_ZTIf
  br i1 %64, label %65, label %72

65:                                               ; preds = %63
  %66 = load float, ptr %35, align 4
  %67 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %65
  %68 = getelementptr i8, ptr %67, i64 8
  %69 = getelementptr { i64, ptr, float }, ptr %68, i32 0, i32 0
  store i64 4, ptr %69, align 8
  %70 = getelementptr { i64, ptr, float }, ptr %68, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %70, align 8
  %71 = getelementptr { i64, ptr, float }, ptr %68, i32 0, i32 2
  store float %66, ptr %71, align 4
  br label %159

72:                                               ; preds = %63
  %73 = icmp eq ptr %36, @_ZTIb
  br i1 %73, label %74, label %81

74:                                               ; preds = %72
  %75 = load i1, ptr %35, align 1
  %76 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %74
  %77 = getelementptr i8, ptr %76, i64 8
  %78 = getelementptr { i64, ptr, i1 }, ptr %77, i32 0, i32 0
  store i64 1, ptr %78, align 8
  %79 = getelementptr { i64, ptr, i1 }, ptr %77, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %79, align 8
  %80 = getelementptr { i64, ptr, i1 }, ptr %77, i32 0, i32 2
  store i1 %75, ptr %80, align 1
  br label %156

81:                                               ; preds = %72
  %82 = icmp eq ptr %36, @_ZTIx
  br i1 %82, label %83, label %90

83:                                               ; preds = %81
  %84 = load i64, ptr %35, align 8
  %85 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %83
  %86 = getelementptr i8, ptr %85, i64 8
  %87 = getelementptr { i64, ptr, i64 }, ptr %86, i32 0, i32 0
  store i64 8, ptr %87, align 8
  %88 = getelementptr { i64, ptr, i64 }, ptr %86, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %88, align 8
  %89 = getelementptr { i64, ptr, i64 }, ptr %86, i32 0, i32 2
  store i64 %84, ptr %89, align 8
  br label %153

90:                                               ; preds = %81
  %91 = icmp eq ptr %36, @_ZTIPc
  br i1 %91, label %92, label %99

92:                                               ; preds = %90
  %93 = load ptr, ptr %35, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %150

99:                                               ; preds = %90
  %100 = icmp eq ptr %36, @_ZTIPv
  br i1 %100, label %101, label %103

101:                                              ; preds = %99
  %102 = load ptr, ptr %35, align 8
  br label %147

103:                                              ; preds = %99
  %104 = load ptr, ptr %36, align 8
  %105 = icmp eq ptr %104, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %105, label %106, label %137

106:                                              ; preds = %103
  %107 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %36, i32 0, i32 2
  %108 = load i32, ptr %107, align 4
  %109 = and i32 %108, 1073741824
  %110 = icmp ne i32 %109, 0
  br i1 %110, label %111, label %127

111:                                              ; preds = %106
  %112 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %36, i32 0, i32 4
  %113 = load ptr, ptr %112, align 8
  %114 = icmp ne ptr %113, null
  br i1 %114, label %115, label %117

115:                                              ; preds = %111
  invoke void %113(ptr %1, ptr %35)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %115
  %116 = load ptr, ptr %1, align 8
  br label %124

117:                                              ; preds = %111
  %118 = load ptr, ptr %35, align 8
  %119 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %117
  %120 = getelementptr i8, ptr %119, i64 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 0
  store i64 8, ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 2
  store ptr %118, ptr %123, align 8
  br label %124

124:                                              ; preds = %.noexc5, %.noexc9
  %125 = phi ptr [ %120, %.noexc9 ], [ %116, %.noexc5 ]
  br label %126

126:                                              ; preds = %124
  br label %134

127:                                              ; preds = %106
  %128 = load ptr, ptr %35, align 8
  %129 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc29 unwind label %endcatch.resume

.noexc29:                                         ; preds = %127
  %130 = getelementptr i8, ptr %129, i64 8
  %131 = getelementptr { i64, ptr, ptr }, ptr %130, i32 0, i32 0
  store i64 8, ptr %131, align 8
  %132 = getelementptr { i64, ptr, ptr }, ptr %130, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %132, align 8
  %133 = getelementptr { i64, ptr, ptr }, ptr %130, i32 0, i32 2
  store ptr %128, ptr %133, align 8
  br label %134

134:                                              ; preds = %126, %.noexc29
  %135 = phi ptr [ %130, %.noexc29 ], [ %125, %126 ]
  br label %136

136:                                              ; preds = %134
  br label %144

137:                                              ; preds = %103
  %138 = load ptr, ptr %35, align 8
  %139 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %137
  %140 = getelementptr i8, ptr %139, i64 8
  %141 = getelementptr { i64, ptr, ptr }, ptr %140, i32 0, i32 0
  store i64 8, ptr %141, align 8
  %142 = getelementptr { i64, ptr, ptr }, ptr %140, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %142, align 8
  %143 = getelementptr { i64, ptr, ptr }, ptr %140, i32 0, i32 2
  store ptr %138, ptr %143, align 8
  br label %144

144:                                              ; preds = %136, %.noexc19
  %145 = phi ptr [ %140, %.noexc19 ], [ %135, %136 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %101, %146
  %148 = phi ptr [ %145, %146 ], [ %102, %101 ]
  br label %149

149:                                              ; preds = %147
  br label %150

150:                                              ; preds = %.noexc16, %149
  %151 = phi ptr [ %148, %149 ], [ %95, %.noexc16 ]
  br label %152

152:                                              ; preds = %150
  br label %153

153:                                              ; preds = %.noexc18, %152
  %154 = phi ptr [ %151, %152 ], [ %86, %.noexc18 ]
  br label %155

155:                                              ; preds = %153
  br label %156

156:                                              ; preds = %.noexc, %155
  %157 = phi ptr [ %154, %155 ], [ %77, %.noexc ]
  br label %158

158:                                              ; preds = %156
  br label %159

159:                                              ; preds = %.noexc21, %158
  %160 = phi ptr [ %157, %158 ], [ %68, %.noexc21 ]
  br label %161

161:                                              ; preds = %159
  br label %162

162:                                              ; preds = %.noexc14, %161
  %163 = phi ptr [ %160, %161 ], [ %59, %.noexc14 ]
  br label %164

164:                                              ; preds = %162
  br label %165

165:                                              ; preds = %.noexc22, %164
  %166 = phi ptr [ %163, %164 ], [ %50, %.noexc22 ]
  br label %167

167:                                              ; preds = %165
  br label %168

168:                                              ; preds = %.noexc2, %167
  %169 = phi ptr [ %166, %167 ], [ %41, %.noexc2 ]
  br label %170

170:                                              ; preds = %168
  store ptr %169, ptr %4, align 8
  %171 = invoke i64 @churn()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %170
  %172 = load double, ptr %6, align 8
  %173 = load ptr, ptr %4, align 8
  %174 = invoke ptr @"___unbox<!ts.string>"(ptr %173)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  %175 = invoke i64 @strlen(ptr %174)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %.noexc11
  %176 = sitofp i64 %175 to double
  %177 = fadd double %172, %176
  store double %177, ptr %6, align 8
  call void @__cxa_end_catch()
  br label %178

178:                                              ; preds = %31, %.noexc12
  %179 = load i32, ptr %5, align 4
  %180 = sitofp i32 %179 to double
  %181 = call ptr @returned(double %180)
  %182 = call i64 @churn()
  %183 = load double, ptr %6, align 8
  %184 = call ptr @"___unbox<!ts.string>"(ptr %181)
  %185 = call i64 @strlen(ptr %184)
  %186 = sitofp i64 %185 to double
  %187 = fadd double %183, %186
  store double %187, ptr %6, align 8
  %188 = load i32, ptr %5, align 4
  %189 = sitofp i32 %188 to double
  %190 = invoke i32 @throwString(double %189)
          to label %191 unwind label %192

191:                                              ; preds = %178
  br label %378

192:                                              ; preds = %178
  %193 = landingpad { ptr, i32 }
          catch ptr null
  %194 = extractvalue { ptr, i32 } %193, 0
  %195 = call ptr @__cxa_begin_catch(ptr %194)
  %196 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %192
  %197 = icmp eq ptr %196, @_ZTIi
  br i1 %197, label %198, label %205

198:                                              ; preds = %.noexc20
  %199 = load i32, ptr %195, align 4
  %200 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %198
  %201 = getelementptr i8, ptr %200, i64 8
  %202 = getelementptr { i64, ptr, i32 }, ptr %201, i32 0, i32 0
  store i64 4, ptr %202, align 8
  %203 = getelementptr { i64, ptr, i32 }, ptr %201, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %203, align 8
  %204 = getelementptr { i64, ptr, i32 }, ptr %201, i32 0, i32 2
  store i32 %199, ptr %204, align 4
  br label %328

205:                                              ; preds = %.noexc20
  %206 = icmp eq ptr %196, @_ZTIj
  br i1 %206, label %207, label %214

207:                                              ; preds = %205
  %208 = load i32, ptr %195, align 4
  %209 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %207
  %210 = getelementptr i8, ptr %209, i64 8
  %211 = getelementptr { i64, ptr, i32 }, ptr %210, i32 0, i32 0
  store i64 4, ptr %211, align 8
  %212 = getelementptr { i64, ptr, i32 }, ptr %210, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %212, align 8
  %213 = getelementptr { i64, ptr, i32 }, ptr %210, i32 0, i32 2
  store i32 %208, ptr %213, align 4
  br label %325

214:                                              ; preds = %205
  %215 = icmp eq ptr %196, @_ZTId
  br i1 %215, label %216, label %223

216:                                              ; preds = %214
  %217 = load double, ptr %195, align 8
  %218 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %216
  %219 = getelementptr i8, ptr %218, i64 8
  %220 = getelementptr { i64, ptr, double }, ptr %219, i32 0, i32 0
  store i64 8, ptr %220, align 8
  %221 = getelementptr { i64, ptr, double }, ptr %219, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %221, align 8
  %222 = getelementptr { i64, ptr, double }, ptr %219, i32 0, i32 2
  store double %217, ptr %222, align 8
  br label %322

223:                                              ; preds = %214
  %224 = icmp eq ptr %196, @_ZTIf
  br i1 %224, label %225, label %232

225:                                              ; preds = %223
  %226 = load float, ptr %195, align 4
  %227 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %225
  %228 = getelementptr i8, ptr %227, i64 8
  %229 = getelementptr { i64, ptr, float }, ptr %228, i32 0, i32 0
  store i64 4, ptr %229, align 8
  %230 = getelementptr { i64, ptr, float }, ptr %228, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %230, align 8
  %231 = getelementptr { i64, ptr, float }, ptr %228, i32 0, i32 2
  store float %226, ptr %231, align 4
  br label %319

232:                                              ; preds = %223
  %233 = icmp eq ptr %196, @_ZTIb
  br i1 %233, label %234, label %241

234:                                              ; preds = %232
  %235 = load i1, ptr %195, align 1
  %236 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %234
  %237 = getelementptr i8, ptr %236, i64 8
  %238 = getelementptr { i64, ptr, i1 }, ptr %237, i32 0, i32 0
  store i64 1, ptr %238, align 8
  %239 = getelementptr { i64, ptr, i1 }, ptr %237, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %239, align 8
  %240 = getelementptr { i64, ptr, i1 }, ptr %237, i32 0, i32 2
  store i1 %235, ptr %240, align 1
  br label %316

241:                                              ; preds = %232
  %242 = icmp eq ptr %196, @_ZTIx
  br i1 %242, label %243, label %250

243:                                              ; preds = %241
  %244 = load i64, ptr %195, align 8
  %245 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %243
  %246 = getelementptr i8, ptr %245, i64 8
  %247 = getelementptr { i64, ptr, i64 }, ptr %246, i32 0, i32 0
  store i64 8, ptr %247, align 8
  %248 = getelementptr { i64, ptr, i64 }, ptr %246, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %248, align 8
  %249 = getelementptr { i64, ptr, i64 }, ptr %246, i32 0, i32 2
  store i64 %244, ptr %249, align 8
  br label %313

250:                                              ; preds = %241
  %251 = icmp eq ptr %196, @_ZTIPc
  br i1 %251, label %252, label %259

252:                                              ; preds = %250
  %253 = load ptr, ptr %195, align 8
  %254 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %252
  %255 = getelementptr i8, ptr %254, i64 8
  %256 = getelementptr { i64, ptr, ptr }, ptr %255, i32 0, i32 0
  store i64 8, ptr %256, align 8
  %257 = getelementptr { i64, ptr, ptr }, ptr %255, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %257, align 8
  %258 = getelementptr { i64, ptr, ptr }, ptr %255, i32 0, i32 2
  store ptr %253, ptr %258, align 8
  br label %310

259:                                              ; preds = %250
  %260 = icmp eq ptr %196, @_ZTIPv
  br i1 %260, label %261, label %263

261:                                              ; preds = %259
  %262 = load ptr, ptr %195, align 8
  br label %307

263:                                              ; preds = %259
  %264 = load ptr, ptr %196, align 8
  %265 = icmp eq ptr %264, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %265, label %266, label %297

266:                                              ; preds = %263
  %267 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %196, i32 0, i32 2
  %268 = load i32, ptr %267, align 4
  %269 = and i32 %268, 1073741824
  %270 = icmp ne i32 %269, 0
  br i1 %270, label %271, label %287

271:                                              ; preds = %266
  %272 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %196, i32 0, i32 4
  %273 = load ptr, ptr %272, align 8
  %274 = icmp ne ptr %273, null
  br i1 %274, label %275, label %277

275:                                              ; preds = %271
  invoke void %273(ptr %2, ptr %195)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %275
  %276 = load ptr, ptr %2, align 8
  br label %284

277:                                              ; preds = %271
  %278 = load ptr, ptr %195, align 8
  %279 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %277
  %280 = getelementptr i8, ptr %279, i64 8
  %281 = getelementptr { i64, ptr, ptr }, ptr %280, i32 0, i32 0
  store i64 8, ptr %281, align 8
  %282 = getelementptr { i64, ptr, ptr }, ptr %280, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %282, align 8
  %283 = getelementptr { i64, ptr, ptr }, ptr %280, i32 0, i32 2
  store ptr %278, ptr %283, align 8
  br label %284

284:                                              ; preds = %.noexc15, %.noexc32
  %285 = phi ptr [ %280, %.noexc32 ], [ %276, %.noexc15 ]
  br label %286

286:                                              ; preds = %284
  br label %294

287:                                              ; preds = %266
  %288 = load ptr, ptr %195, align 8
  %289 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc31 unwind label %endcatch.resume

.noexc31:                                         ; preds = %287
  %290 = getelementptr i8, ptr %289, i64 8
  %291 = getelementptr { i64, ptr, ptr }, ptr %290, i32 0, i32 0
  store i64 8, ptr %291, align 8
  %292 = getelementptr { i64, ptr, ptr }, ptr %290, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %292, align 8
  %293 = getelementptr { i64, ptr, ptr }, ptr %290, i32 0, i32 2
  store ptr %288, ptr %293, align 8
  br label %294

294:                                              ; preds = %286, %.noexc31
  %295 = phi ptr [ %290, %.noexc31 ], [ %285, %286 ]
  br label %296

296:                                              ; preds = %294
  br label %304

297:                                              ; preds = %263
  %298 = load ptr, ptr %195, align 8
  %299 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %297
  %300 = getelementptr i8, ptr %299, i64 8
  %301 = getelementptr { i64, ptr, ptr }, ptr %300, i32 0, i32 0
  store i64 8, ptr %301, align 8
  %302 = getelementptr { i64, ptr, ptr }, ptr %300, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %302, align 8
  %303 = getelementptr { i64, ptr, ptr }, ptr %300, i32 0, i32 2
  store ptr %298, ptr %303, align 8
  br label %304

304:                                              ; preds = %296, %.noexc30
  %305 = phi ptr [ %300, %.noexc30 ], [ %295, %296 ]
  br label %306

306:                                              ; preds = %304
  br label %307

307:                                              ; preds = %261, %306
  %308 = phi ptr [ %305, %306 ], [ %262, %261 ]
  br label %309

309:                                              ; preds = %307
  br label %310

310:                                              ; preds = %.noexc1, %309
  %311 = phi ptr [ %308, %309 ], [ %255, %.noexc1 ]
  br label %312

312:                                              ; preds = %310
  br label %313

313:                                              ; preds = %.noexc26, %312
  %314 = phi ptr [ %311, %312 ], [ %246, %.noexc26 ]
  br label %315

315:                                              ; preds = %313
  br label %316

316:                                              ; preds = %.noexc33, %315
  %317 = phi ptr [ %314, %315 ], [ %237, %.noexc33 ]
  br label %318

318:                                              ; preds = %316
  br label %319

319:                                              ; preds = %.noexc23, %318
  %320 = phi ptr [ %317, %318 ], [ %228, %.noexc23 ]
  br label %321

321:                                              ; preds = %319
  br label %322

322:                                              ; preds = %.noexc13, %321
  %323 = phi ptr [ %320, %321 ], [ %219, %.noexc13 ]
  br label %324

324:                                              ; preds = %322
  br label %325

325:                                              ; preds = %.noexc17, %324
  %326 = phi ptr [ %323, %324 ], [ %210, %.noexc17 ]
  br label %327

327:                                              ; preds = %325
  br label %328

328:                                              ; preds = %.noexc25, %327
  %329 = phi ptr [ %326, %327 ], [ %201, %.noexc25 ]
  br label %330

330:                                              ; preds = %328
  store ptr %329, ptr %3, align 8
  %331 = load ptr, ptr %3, align 8
  %332 = load ptr, ptr @kept, align 8
  %333 = icmp eq ptr %332, null
  br i1 %333, label %334, label %337

334:                                              ; preds = %330
  %335 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %334
  %336 = getelementptr i8, ptr %335, i64 8
  store ptr %336, ptr @kept, align 8
  br label %338

337:                                              ; preds = %330
  br label %338

338:                                              ; preds = %.noexc6, %337
  %339 = phi ptr [ %332, %337 ], [ %336, %.noexc6 ]
  br label %340

340:                                              ; preds = %338
  %341 = getelementptr { ptr, i64, i64 }, ptr %339, i32 0, i32 1
  %342 = load i64, ptr %341, align 8
  %343 = add i64 %342, 1
  %344 = getelementptr { ptr, i64, i64 }, ptr %339, i32 0, i32 0
  %345 = getelementptr { ptr, i64, i64 }, ptr %339, i32 0, i32 2
  %346 = load ptr, ptr %344, align 8
  %347 = load i64, ptr %345, align 8
  %348 = icmp ugt i64 %343, %347
  br i1 %348, label %349, label %373

349:                                              ; preds = %340
  %350 = mul i64 %347, 2
  %351 = icmp uge i64 %350, %343
  %352 = select i1 %351, i64 %350, i64 %343
  %353 = icmp ult i64 %352, 4
  %354 = select i1 %353, i64 4, i64 %352
  %355 = icmp ule i64 %343, 4294967295
  br i1 %355, label %356, label %413

356:                                              ; preds = %349
  %357 = icmp ule i64 %354, 2305843009213693951
  br i1 %357, label %358, label %415

358:                                              ; preds = %356
  %359 = mul i64 8, %354
  %360 = add i64 %359, 8
  %361 = icmp eq ptr %346, null
  %362 = getelementptr i8, ptr %346, i64 -8
  %363 = select i1 %361, ptr null, ptr %362
  %364 = invoke ptr @GC_realloc(ptr %363, i64 %360)
          to label %.noexc34 unwind label %endcatch.resume

.noexc34:                                         ; preds = %358
  %365 = getelementptr i8, ptr %364, i64 8
  %366 = icmp ne ptr %365, null
  %367 = icmp eq i64 %359, 0
  %368 = or i1 %366, %367
  br i1 %368, label %369, label %417

369:                                              ; preds = %.noexc34
  %370 = mul i64 8, %347
  %371 = getelementptr i8, ptr %365, i64 %370
  %372 = sub i64 %359, %370
  call void @llvm.memset.p0.i64(ptr %371, i8 0, i64 %372, i1 false)
  store ptr %365, ptr %344, align 8
  store i64 %354, ptr %345, align 8
  br label %374

373:                                              ; preds = %340
  br label %374

374:                                              ; preds = %369, %373
  %375 = phi ptr [ %346, %373 ], [ %365, %369 ]
  br label %376

376:                                              ; preds = %374
  %377 = getelementptr ptr, ptr %375, i64 %342
  store ptr %331, ptr %377, align 8
  store i64 %343, ptr %341, align 8
  call void @__cxa_end_catch()
  br label %378

378:                                              ; preds = %191, %376
  %379 = load double, ptr %6, align 8
  %380 = call i32 @localNumber()
  %381 = sitofp i32 %380 to double
  %382 = fadd double %379, %381
  store double %382, ptr %6, align 8
  %383 = load i32, ptr %5, align 4
  %384 = add i32 %383, 1
  store i32 %384, ptr %5, align 4
  br label %7

385:                                              ; preds = %7
  %386 = call i64 @churn()
  %387 = load ptr, ptr @kept, align 8
  %388 = icmp eq ptr %387, null
  %389 = select i1 %388, ptr @__ts_empty_array, ptr %387
  %390 = getelementptr { ptr, i64, i64 }, ptr %389, i32 0, i32 0
  %391 = load ptr, ptr %390, align 8
  %392 = getelementptr ptr, ptr %391, i32 99
  %393 = load ptr, ptr %392, align 8
  %394 = call ptr @"___unbox<!ts.string>"(ptr %393)
  %395 = ptrtoint ptr %394 to i64
  %396 = icmp ne i64 %395, 0
  %397 = zext i1 %396 to i32
  %398 = and i32 %397, 1
  %399 = icmp ne i32 %398, 0
  br i1 %399, label %400, label %403

400:                                              ; preds = %385
  %401 = call i32 @strcmp(ptr %394, ptr getelementptr inbounds nuw (i8, ptr @s_13770579504406577721, i64 8))
  %402 = icmp eq i32 %401, 0
  br label %405

403:                                              ; preds = %385
  %404 = icmp eq i64 %395, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_13770579504406577721, i64 8) to i64)
  br label %405

405:                                              ; preds = %400, %403
  %406 = phi i1 [ %404, %403 ], [ %402, %400 ]
  br label %407

407:                                              ; preds = %405
  br i1 %406, label %408, label %419

408:                                              ; preds = %407
  %409 = load double, ptr %6, align 8
  %410 = fcmp oeq double %409, 6.710000e+03
  br i1 %410, label %411, label %421

411:                                              ; preds = %408
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

412:                                              ; preds = %.noexc4, %.noexc8, %.noexc28, %421, %419
  unreachable

413:                                              ; preds = %349
  %414 = invoke i32 @fflush(ptr null)
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %413
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 124, ptr null)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %.noexc27
  br label %412

415:                                              ; preds = %356
  %416 = invoke i32 @fflush(ptr null)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %415
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 124, ptr null)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  br label %412

417:                                              ; preds = %.noexc34
  %418 = invoke i32 @fflush(ptr null)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %417
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 124, ptr null)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  br label %412

419:                                              ; preds = %407
  %420 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11601984928555610496, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 131, ptr null)
  br label %412

421:                                              ; preds = %408
  %422 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15069869055152812943, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17464517255946554849, i64 8), i32 133, ptr null)
  br label %412

endcatch.resume:                                  ; preds = %358, %234, %277, %287, %297, %127, %.noexc27, %413, %243, %198, %32, %225, %47, %65, %192, %137, %83, %207, %92, %275, %56, %216, %.noexc11, %.noexc10, %170, %117, %.noexc7, %415, %334, %115, %.noexc3, %417, %38, %252, %74
  %423 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %423
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

