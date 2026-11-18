; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Failure = type { ptr, i32 }
%Derived = type { %Failure, i32 }

$"___unbox<!ts.number>" = comdat any

$"___unbox<si32>" = comdat any

$"___unbox<!ts.string>" = comdat any

$"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>" = comdat any

$"___unbox<!ts.class<Derived, !ts.class_storage<Derived>>>" = comdat any

$Failure..size = comdat any

$Derived..size = comdat any

$_ZTSP7Failure = comdat any

$_ZTIP7Failure = comdat any

$_ZTS7Failure = comdat any

$_ZTI7Failure = comdat any

$_ZTSP7Derived = comdat any

$_ZTIP7Derived = comdat any

$_ZTS7Derived = comdat any

$_ZTI7Derived = comdat any

@s_9133338390725077489 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFailure\00", align 8
@s_4179427500551276790 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFDerived\00", align 8
@Failure..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_9133338390725077489, i64 8)
@Failure..size = global i64 16, comdat
@Failure..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Failure..instanceOf, ptr @Failure..new, ptr @Failure..rtti, ptr @Failure..size }
@Derived..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_4179427500551276790, i64 8)
@Derived..size = global i64 24, comdat
@Derived..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Derived..instanceOf, ptr @Derived..new, ptr @Failure..rtti, ptr @Failure..size, ptr @Derived..rtti, ptr @Derived..size }
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP7Failure = linkonce_odr constant [10 x i8] c"P7Failure\00", comdat
@_ZTIP7Failure = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP7Failure, i32 1073741824, ptr @_ZTI7Failure, ptr @.eh.copy.box.P7Failure }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS7Failure = linkonce_odr constant [9 x i8] c"7Failure\00", comdat
@_ZTI7Failure = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS7Failure }, comdat
@_ZTSP7Derived = linkonce_odr constant [10 x i8] c"P7Derived\00", comdat
@_ZTIP7Derived = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP7Derived, i32 1073741824, ptr @_ZTI7Derived, ptr @.eh.copy.box.P7Derived }, comdat
@_ZTVN10__cxxabiv120__si_class_type_infoE = external constant ptr
@_ZTS7Derived = linkonce_odr constant [9 x i8] c"7Derived\00", comdat
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
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_2495249626774661283 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined\00", align 8
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
@s_14785268767687447306 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboom\00", align 8
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_4044607399571746621 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown number\00", align 8
@f_5920818698833515450 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00catch_untyped_value.ts\00", align 8
@m_12069005346676613813 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown int\00", align 8
@m_8864937869768531853 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown string\00", align 8
@m_5530502136322933383 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown class instance\00", align 8
@m_5903054358280398641 = internal constant [73 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown derived instance as either class\00", align 8
@m_7504679140485314426 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa catch (e: any) reads a thrown int\00", align 8
@m_9404695902159035447 = internal constant [39 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa thrown any is bound as it is\00", align 8
@m_4558786565675528182 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: number) catches a thrown integer literal\00", align 8
@m_7614099512854692056 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtyped catches still bind their own types\00", align 8
@_ZTI7Derived = linkonce_odr constant { ptr, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTS7Derived, ptr @_ZTI7Failure }, comdat
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_2515608230574705588_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_2515608230574705588, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_856374508671476337_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_856374508671476337, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

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

define internal void @tsrel_2515608230574705588(ptr %0) {
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

declare i32 @snprintf(ptr, i64, ptr, ...)

declare i64 @strtoll(ptr, ptr, i32)

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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare double @strtod(ptr, ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define void @Failure.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Failure, ptr %3, i32 0, i32 1
  store i32 42, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Failure..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @Failure..new() #2 {
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
define void @Derived.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Derived, ptr %3, i32 0, i32 0
  call void @Failure.constructor(ptr %4)
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Derived, ptr %5, i32 0, i32 1
  store i32 1, ptr %6, align 4
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
  %28 = call i1 @Failure..instanceOf(ptr %26, ptr %27)
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
  %5 = getelementptr %Failure, ptr %4, i32 0, i32 0
  store ptr @Derived..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
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
define double @untypedNumber() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double 0.000000e+00, ptr %3, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store double 2.500000e+00, ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTId, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc10
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc10
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc2, %.noexc11
  %99 = phi ptr [ %94, %.noexc11 ], [ %90, %.noexc2 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc12
  %109 = phi ptr [ %104, %.noexc12 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc8
  %119 = phi ptr [ %114, %.noexc8 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc6, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc6 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc4, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc4 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc7, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc7 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc3, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc3 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc5, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc5 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc9, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc9 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke double @"___unbox<!ts.number>"(ptr %145)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %144
  store double %146, ptr %3, align 8
  call void @__cxa_end_catch()
  %147 = load double, ptr %3, align 8
  store double %147, ptr %4, align 8
  %148 = load double, ptr %4, align 8
  ret double %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %101, %91, %6, %12, %111, %48, %66, %21, %57, %39, %89, %144, %30
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define linkonce_odr i32 @"___unbox<si32>"(ptr %0) #2 comdat {
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
define i32 @untypedInt() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 7, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc5
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc5
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc10, %.noexc8
  %99 = phi ptr [ %94, %.noexc8 ], [ %90, %.noexc10 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc9
  %109 = phi ptr [ %104, %.noexc9 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
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
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc1, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc1 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc6, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc6 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc2, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc2 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc12, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc12 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc7, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc7 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc11, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc11 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke i32 @"___unbox<si32>"(ptr %145)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %144
  store i32 %146, ptr %3, align 4
  call void @__cxa_end_catch()
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %30, %12, %89, %101, %91, %21, %57, %6, %111, %144, %48, %66, %39
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
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
define ptr @untypedString() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %3, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8), ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIPc, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc11
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc11
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc6, %.noexc4
  %99 = phi ptr [ %94, %.noexc4 ], [ %90, %.noexc6 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc5
  %109 = phi ptr [ %104, %.noexc5 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc10
  %119 = phi ptr [ %114, %.noexc10 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc12, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc12 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc9, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc9 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc2, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc2 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc3, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc3 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc7, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc7 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc8, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc8 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke ptr @"___unbox<!ts.string>"(ptr %145)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %144
  store ptr %146, ptr %3, align 8
  call void @__cxa_end_catch()
  %147 = load ptr, ptr %3, align 8
  store ptr %147, ptr %4, align 8
  %148 = load ptr, ptr %4, align 8
  ret ptr %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %66, %6, %111, %57, %12, %21, %89, %101, %91, %39, %48, %144, %30
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P7Failure(ptr %0, ptr %1) #2 {
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
define linkonce_odr ptr @"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>"(ptr %0) #2 comdat {
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
define i32 @untypedClass() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Failure, ptr %6, i32 0, i32 0
  store ptr @Failure..vtbl, ptr %7, align 8
  invoke void @Failure.constructor(ptr %6)
          to label %8 unwind label %10

8:                                                ; preds = %0
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %6, ptr %9, align 8
  invoke void @__cxa_throw(ptr %9, ptr @_ZTIP7Failure, ptr null)
          to label %155 unwind label %10

10:                                               ; preds = %8, %0
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %10
  %15 = icmp eq ptr %14, @_ZTIi
  br i1 %15, label %16, label %23

16:                                               ; preds = %.noexc4
  %17 = load i32, ptr %13, align 4
  %18 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %16
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 0
  store i64 4, ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 2
  store i32 %17, ptr %22, align 4
  br label %146

23:                                               ; preds = %.noexc4
  %24 = icmp eq ptr %14, @_ZTIj
  br i1 %24, label %25, label %32

25:                                               ; preds = %23
  %26 = load i32, ptr %13, align 4
  %27 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %25
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %34
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %43
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %52
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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %61
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %70
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
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %93
  %94 = load ptr, ptr %1, align 8
  br label %102

95:                                               ; preds = %89
  %96 = load ptr, ptr %13, align 8
  %97 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %95
  %98 = getelementptr i8, ptr %97, i64 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 0
  store i64 8, ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 2
  store ptr %96, ptr %101, align 8
  br label %102

102:                                              ; preds = %.noexc12, %.noexc3
  %103 = phi ptr [ %98, %.noexc3 ], [ %94, %.noexc12 ]
  br label %104

104:                                              ; preds = %102
  br label %112

105:                                              ; preds = %84
  %106 = load ptr, ptr %13, align 8
  %107 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %105
  %108 = getelementptr i8, ptr %107, i64 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 0
  store i64 8, ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 2
  store ptr %106, ptr %111, align 8
  br label %112

112:                                              ; preds = %104, %.noexc
  %113 = phi ptr [ %108, %.noexc ], [ %103, %104 ]
  br label %114

114:                                              ; preds = %112
  br label %122

115:                                              ; preds = %81
  %116 = load ptr, ptr %13, align 8
  %117 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %115
  %118 = getelementptr i8, ptr %117, i64 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 0
  store i64 8, ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 2
  store ptr %116, ptr %121, align 8
  br label %122

122:                                              ; preds = %114, %.noexc1
  %123 = phi ptr [ %118, %.noexc1 ], [ %113, %114 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %79, %124
  %126 = phi ptr [ %123, %124 ], [ %80, %79 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc6, %127
  %129 = phi ptr [ %126, %127 ], [ %73, %.noexc6 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc2, %130
  %132 = phi ptr [ %129, %130 ], [ %64, %.noexc2 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc11, %133
  %135 = phi ptr [ %132, %133 ], [ %55, %.noexc11 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc10, %136
  %138 = phi ptr [ %135, %136 ], [ %46, %.noexc10 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc5, %139
  %141 = phi ptr [ %138, %139 ], [ %37, %.noexc5 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc8, %142
  %144 = phi ptr [ %141, %142 ], [ %28, %.noexc8 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc9, %145
  %147 = phi ptr [ %144, %145 ], [ %19, %.noexc9 ]
  br label %148

148:                                              ; preds = %146
  store ptr %147, ptr %2, align 8
  %149 = load ptr, ptr %2, align 8
  %150 = invoke ptr @"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>"(ptr %149)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %148
  %151 = getelementptr %Failure, ptr %150, i32 0, i32 1
  %152 = load i32, ptr %151, align 4
  store i32 %152, ptr %3, align 4
  call void @__cxa_end_catch()
  %153 = load i32, ptr %3, align 4
  store i32 %153, ptr %4, align 4
  %154 = load i32, ptr %4, align 4
  ret i32 %154

155:                                              ; preds = %8
  unreachable

endcatch.resume:                                  ; preds = %93, %52, %43, %16, %25, %148, %70, %34, %10, %95, %61, %115, %105
  %156 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %156
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
define linkonce_odr ptr @"___unbox<!ts.class<Derived, !ts.class_storage<Derived>>>"(ptr %0) #2 comdat {
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
  %39 = load ptr, ptr @Derived..rtti, align 8
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
  %70 = load ptr, ptr @Derived..rtti, align 8
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
  %83 = load ptr, ptr @Derived..rtti, align 8
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
define i32 @untypedDerivedAsBase() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Derived, ptr %6, i32 0, i32 0
  %8 = getelementptr %Failure, ptr %7, i32 0, i32 0
  store ptr @Derived..vtbl, ptr %8, align 8
  invoke void @Derived.constructor(ptr %6)
          to label %9 unwind label %11

9:                                                ; preds = %0
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %6, ptr %10, align 8
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIP7Derived, ptr null)
          to label %161 unwind label %11

11:                                               ; preds = %9, %0
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc7
  %18 = load i32, ptr %14, align 4
  %19 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %17
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 0
  store i64 4, ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 2
  store i32 %18, ptr %23, align 4
  br label %147

24:                                               ; preds = %.noexc7
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %26
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %35
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %44
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
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %53
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
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %62
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %71
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
  invoke void %92(ptr %1, ptr %14)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %94
  %95 = load ptr, ptr %1, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc2, %.noexc9
  %104 = phi ptr [ %99, %.noexc9 ], [ %95, %.noexc2 ]
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc
  %124 = phi ptr [ %119, %.noexc ], [ %114, %115 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %80, %125
  %127 = phi ptr [ %124, %125 ], [ %81, %80 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc8, %128
  %130 = phi ptr [ %127, %128 ], [ %74, %.noexc8 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc13, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc13 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc4, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc4 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc10, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc10 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc11, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc11 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc3, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc3 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc12, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc12 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %2, align 8
  %150 = load ptr, ptr %2, align 8
  %151 = invoke ptr @"___unbox<!ts.class<Failure, !ts.class_storage<Failure>>>"(ptr %150)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %149
  %152 = getelementptr %Failure, ptr %151, i32 0, i32 1
  %153 = load i32, ptr %152, align 4
  %154 = load ptr, ptr %2, align 8
  %155 = invoke ptr @"___unbox<!ts.class<Derived, !ts.class_storage<Derived>>>"(ptr %154)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %156 = getelementptr %Derived, ptr %155, i32 0, i32 1
  %157 = load i32, ptr %156, align 4
  %158 = add i32 %153, %157
  store i32 %158, ptr %3, align 4
  call void @__cxa_end_catch()
  %159 = load i32, ptr %3, align 4
  store i32 %159, ptr %4, align 4
  %160 = load i32, ptr %4, align 4
  ret i32 %160

161:                                              ; preds = %9
  unreachable

endcatch.resume:                                  ; preds = %62, %17, %35, %44, %96, %71, %11, %.noexc5, %149, %53, %26, %94, %106, %116
  %162 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %162
}

; Function Attrs: mustprogress
define i32 @anyTyped() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 11, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc2
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc2
  %20 = icmp eq ptr %10, @_ZTIj
  br i1 %20, label %21, label %28

21:                                               ; preds = %19
  %22 = load i32, ptr %9, align 4
  %23 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %21
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 0
  store i64 4, ptr %25, align 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %24, i32 0, i32 2
  store i32 %22, ptr %27, align 4
  br label %139

28:                                               ; preds = %19
  %29 = icmp eq ptr %10, @_ZTId
  br i1 %29, label %30, label %37

30:                                               ; preds = %28
  %31 = load double, ptr %9, align 8
  %32 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %30
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 0
  store i64 8, ptr %34, align 8
  %35 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %33, i32 0, i32 2
  store double %31, ptr %36, align 8
  br label %136

37:                                               ; preds = %28
  %38 = icmp eq ptr %10, @_ZTIf
  br i1 %38, label %39, label %46

39:                                               ; preds = %37
  %40 = load float, ptr %9, align 4
  %41 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %39
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 0
  store i64 4, ptr %43, align 8
  %44 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %42, i32 0, i32 2
  store float %40, ptr %45, align 4
  br label %133

46:                                               ; preds = %37
  %47 = icmp eq ptr %10, @_ZTIb
  br i1 %47, label %48, label %55

48:                                               ; preds = %46
  %49 = load i1, ptr %9, align 1
  %50 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %48
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 0
  store i64 1, ptr %52, align 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %51, i32 0, i32 2
  store i1 %49, ptr %54, align 1
  br label %130

55:                                               ; preds = %46
  %56 = icmp eq ptr %10, @_ZTIx
  br i1 %56, label %57, label %64

57:                                               ; preds = %55
  %58 = load i64, ptr %9, align 8
  %59 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %57
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 0
  store i64 8, ptr %61, align 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %60, i32 0, i32 2
  store i64 %58, ptr %63, align 8
  br label %127

64:                                               ; preds = %55
  %65 = icmp eq ptr %10, @_ZTIPc
  br i1 %65, label %66, label %73

66:                                               ; preds = %64
  %67 = load ptr, ptr %9, align 8
  %68 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %66
  %69 = getelementptr i8, ptr %68, i64 8
  %70 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 0
  store i64 8, ptr %70, align 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %69, i32 0, i32 2
  store ptr %67, ptr %72, align 8
  br label %124

73:                                               ; preds = %64
  %74 = icmp eq ptr %10, @_ZTIPv
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  %76 = load ptr, ptr %9, align 8
  br label %121

77:                                               ; preds = %73
  %78 = load ptr, ptr %10, align 8
  %79 = icmp eq ptr %78, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %79, label %80, label %111

80:                                               ; preds = %77
  %81 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 2
  %82 = load i32, ptr %81, align 4
  %83 = and i32 %82, 1073741824
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %101

85:                                               ; preds = %80
  %86 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %10, i32 0, i32 4
  %87 = load ptr, ptr %86, align 8
  %88 = icmp ne ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void %87(ptr %1, ptr %9)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc8, %.noexc4
  %99 = phi ptr [ %94, %.noexc4 ], [ %90, %.noexc8 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc3
  %109 = phi ptr [ %104, %.noexc3 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc
  %119 = phi ptr [ %114, %.noexc ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc11, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc11 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc12, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc12 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc10, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc10 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc7, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc7 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc6, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc6 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc9, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc9 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc5, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc5 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke i32 @"___unbox<si32>"(ptr %145)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %144
  store i32 %146, ptr %3, align 4
  call void @__cxa_end_catch()
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %57, %66, %48, %21, %89, %39, %30, %12, %91, %101, %6, %144, %111
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i32 @alreadyAny() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr { i64, ptr, i32 }, ptr %6, i32 0, i32 0
  store i64 4, ptr %7, align 8
  %8 = getelementptr { i64, ptr, i32 }, ptr %6, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %8, align 8
  %9 = getelementptr { i64, ptr, i32 }, ptr %6, i32 0, i32 2
  store i32 123, ptr %9, align 4
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %6, ptr %10, align 8
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIPv, ptr null)
          to label %154 unwind label %11

11:                                               ; preds = %0
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc1
  %18 = load i32, ptr %14, align 4
  %19 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %17
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 0
  store i64 4, ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %20, i32 0, i32 2
  store i32 %18, ptr %23, align 4
  br label %147

24:                                               ; preds = %.noexc1
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %26
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %35
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
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %62
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %71
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
  invoke void %92(ptr %1, ptr %14)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %94
  %95 = load ptr, ptr %1, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc8, %.noexc
  %104 = phi ptr [ %99, %.noexc ], [ %95, %.noexc8 ]
  br label %105

105:                                              ; preds = %103
  br label %113

106:                                              ; preds = %85
  %107 = load ptr, ptr %14, align 8
  %108 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %106
  %109 = getelementptr i8, ptr %108, i64 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 0
  store i64 8, ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 2
  store ptr %107, ptr %112, align 8
  br label %113

113:                                              ; preds = %105, %.noexc10
  %114 = phi ptr [ %109, %.noexc10 ], [ %104, %105 ]
  br label %115

115:                                              ; preds = %113
  br label %123

116:                                              ; preds = %82
  %117 = load ptr, ptr %14, align 8
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc3
  %124 = phi ptr [ %119, %.noexc3 ], [ %114, %115 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %80, %125
  %127 = phi ptr [ %124, %125 ], [ %81, %80 ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc7, %128
  %130 = phi ptr [ %127, %128 ], [ %74, %.noexc7 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc12, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc12 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc5, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc5 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc9, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc9 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc11, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc11 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc4, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc4 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc2, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc2 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %2, align 8
  %150 = load ptr, ptr %2, align 8
  %151 = invoke i32 @"___unbox<si32>"(ptr %150)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %149
  store i32 %151, ptr %3, align 4
  call void @__cxa_end_catch()
  %152 = load i32, ptr %3, align 4
  store i32 %152, ptr %4, align 4
  %153 = load i32, ptr %4, align 4
  ret i32 %153

154:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %62, %35, %106, %44, %94, %71, %149, %53, %26, %116, %17, %11, %96
  %155 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %155
}

; Function Attrs: mustprogress
define double @numberCatchesInt() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 5, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %58 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %9 = icmp eq i32 %7, %8
  %10 = extractvalue { ptr, i32 } %6, 1
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %12 = icmp eq i32 %10, %11
  %13 = extractvalue { ptr, i32 } %6, 1
  %14 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %15 = icmp eq i32 %13, %14
  %16 = extractvalue { ptr, i32 } %6, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %18 = icmp eq i32 %16, %17
  %19 = or i1 %9, %12
  %20 = or i1 %19, %15
  %21 = or i1 %20, %18
  br i1 %21, label %25, label %22

22:                                               ; preds = %5
  %23 = extractvalue { ptr, i32 } %6, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %22
  br label %58

25:                                               ; preds = %5
  %26 = extractvalue { ptr, i32 } %6, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  %28 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %25
  %29 = icmp eq ptr %28, @_ZTIi
  br i1 %29, label %30, label %33

30:                                               ; preds = %.noexc1
  %31 = load i32, ptr %27, align 4
  %32 = sitofp i32 %31 to double
  br label %51

33:                                               ; preds = %.noexc1
  %34 = icmp eq ptr %28, @_ZTIj
  br i1 %34, label %35, label %38

35:                                               ; preds = %33
  %36 = load i32, ptr %27, align 4
  %37 = uitofp i32 %36 to double
  br label %48

38:                                               ; preds = %33
  %39 = icmp eq ptr %28, @_ZTIf
  br i1 %39, label %40, label %43

40:                                               ; preds = %38
  %41 = load float, ptr %27, align 4
  %42 = fpext float %41 to double
  br label %45

43:                                               ; preds = %38
  %44 = load double, ptr %27, align 8
  br label %45

45:                                               ; preds = %40, %43
  %46 = phi double [ %44, %43 ], [ %42, %40 ]
  br label %47

47:                                               ; preds = %45
  br label %48

48:                                               ; preds = %35, %47
  %49 = phi double [ %46, %47 ], [ %37, %35 ]
  br label %50

50:                                               ; preds = %48
  br label %51

51:                                               ; preds = %30, %50
  %52 = phi double [ %49, %50 ], [ %32, %30 ]
  br label %53

53:                                               ; preds = %51
  store double %52, ptr %1, align 8
  %54 = load double, ptr %1, align 8
  %55 = fadd double %54, 5.000000e-01
  store double %55, ptr %2, align 8
  call void @__cxa_end_catch()
  %56 = load double, ptr %2, align 8
  store double %56, ptr %3, align 8
  %57 = load double, ptr %3, align 8
  ret double %57

58:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %25, %22
  %59 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %59
}

; Function Attrs: mustprogress
define double @typedCatchesUnchanged() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca double, align 8
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  store double 0.000000e+00, ptr %4, align 8
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 3, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %112 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = extractvalue { ptr, i32 } %8, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %12
  br label %112

15:                                               ; preds = %7
  %16 = extractvalue { ptr, i32 } %8, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = load i32, ptr %17, align 4
  store i32 %18, ptr %3, align 4
  %19 = load double, ptr %4, align 8
  %20 = load i32, ptr %3, align 4
  %21 = sitofp i32 %20 to double
  %22 = fadd double %19, %21
  store double %22, ptr %4, align 8
  call void @__cxa_end_catch()
  %23 = call ptr @__cxa_allocate_exception(i64 8)
  store double 1.500000e+00, ptr %23, align 8
  invoke void @__cxa_throw(ptr %23, ptr @_ZTId, ptr null)
          to label %112 unwind label %24

24:                                               ; preds = %15
  %25 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %26 = extractvalue { ptr, i32 } %25, 1
  %27 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %28 = icmp eq i32 %26, %27
  %29 = extractvalue { ptr, i32 } %25, 1
  %30 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %31 = icmp eq i32 %29, %30
  %32 = extractvalue { ptr, i32 } %25, 1
  %33 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %34 = icmp eq i32 %32, %33
  %35 = extractvalue { ptr, i32 } %25, 1
  %36 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %37 = icmp eq i32 %35, %36
  %38 = or i1 %28, %31
  %39 = or i1 %38, %34
  %40 = or i1 %39, %37
  br i1 %40, label %44, label %41

41:                                               ; preds = %24
  %42 = extractvalue { ptr, i32 } %25, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %41
  br label %112

44:                                               ; preds = %24
  %45 = extractvalue { ptr, i32 } %25, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  %47 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %44
  %48 = icmp eq ptr %47, @_ZTIi
  br i1 %48, label %49, label %52

49:                                               ; preds = %.noexc4
  %50 = load i32, ptr %46, align 4
  %51 = sitofp i32 %50 to double
  br label %70

52:                                               ; preds = %.noexc4
  %53 = icmp eq ptr %47, @_ZTIj
  br i1 %53, label %54, label %57

54:                                               ; preds = %52
  %55 = load i32, ptr %46, align 4
  %56 = uitofp i32 %55 to double
  br label %67

57:                                               ; preds = %52
  %58 = icmp eq ptr %47, @_ZTIf
  br i1 %58, label %59, label %62

59:                                               ; preds = %57
  %60 = load float, ptr %46, align 4
  %61 = fpext float %60 to double
  br label %64

62:                                               ; preds = %57
  %63 = load double, ptr %46, align 8
  br label %64

64:                                               ; preds = %59, %62
  %65 = phi double [ %63, %62 ], [ %61, %59 ]
  br label %66

66:                                               ; preds = %64
  br label %67

67:                                               ; preds = %54, %66
  %68 = phi double [ %65, %66 ], [ %56, %54 ]
  br label %69

69:                                               ; preds = %67
  br label %70

70:                                               ; preds = %49, %69
  %71 = phi double [ %68, %69 ], [ %51, %49 ]
  br label %72

72:                                               ; preds = %70
  store double %71, ptr %2, align 8
  %73 = load double, ptr %4, align 8
  %74 = load double, ptr %2, align 8
  %75 = fmul double %74, 2.000000e+00
  %76 = fadd double %73, %75
  store double %76, ptr %4, align 8
  call void @__cxa_end_catch()
  %77 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %77, align 8
  invoke void @__cxa_throw(ptr %77, ptr @_ZTIPc, ptr null)
          to label %112 unwind label %78

78:                                               ; preds = %72
  %79 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %80 = extractvalue { ptr, i32 } %79, 1
  %81 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %82 = icmp eq i32 %80, %81
  br i1 %82, label %86, label %83

83:                                               ; preds = %78
  %84 = extractvalue { ptr, i32 } %79, 0
  %85 = call ptr @__cxa_begin_catch(ptr %84)
  invoke void @__cxa_rethrow()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %83
  br label %112

86:                                               ; preds = %78
  %87 = extractvalue { ptr, i32 } %79, 0
  %88 = call ptr @__cxa_begin_catch(ptr %87)
  store ptr %88, ptr %1, align 8
  %89 = load double, ptr %4, align 8
  %90 = load ptr, ptr %1, align 8
  %91 = ptrtoint ptr %90 to i64
  %92 = icmp ne i64 %91, 0
  %93 = zext i1 %92 to i32
  %94 = and i32 %93, 1
  %95 = icmp ne i32 %94, 0
  br i1 %95, label %96, label %99

96:                                               ; preds = %86
  %97 = invoke i32 @strcmp(ptr %90, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %96
  %98 = icmp eq i32 %97, 0
  br label %101

99:                                               ; preds = %86
  %100 = icmp eq i64 %91, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8) to i64)
  br label %101

101:                                              ; preds = %.noexc2, %99
  %102 = phi i1 [ %100, %99 ], [ %98, %.noexc2 ]
  br label %103

103:                                              ; preds = %101
  br i1 %102, label %104, label %105

104:                                              ; preds = %103
  br label %106

105:                                              ; preds = %103
  br label %106

106:                                              ; preds = %104, %105
  %107 = phi i32 [ 0, %105 ], [ 10, %104 ]
  %108 = sitofp i32 %107 to double
  %109 = fadd double %89, %108
  store double %109, ptr %4, align 8
  call void @__cxa_end_catch()
  %110 = load double, ptr %4, align 8
  store double %110, ptr %5, align 8
  %111 = load double, ptr %5, align 8
  ret double %111

112:                                              ; preds = %.noexc3, %72, %.noexc, %15, %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %44, %83, %96, %12, %41
  %113 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %113
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call double @untypedNumber()
  %2 = fcmp oeq double %1, 2.500000e+00
  br i1 %2, label %3, label %41

3:                                                ; preds = %0
  %4 = call i32 @untypedInt()
  %5 = icmp eq i32 %4, 7
  br i1 %5, label %6, label %43

6:                                                ; preds = %3
  %7 = call ptr @untypedString()
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %6
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %6
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14785268767687447306, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %45

21:                                               ; preds = %20
  %22 = call i32 @untypedClass()
  %23 = icmp eq i32 %22, 42
  br i1 %23, label %24, label %47

24:                                               ; preds = %21
  %25 = call i32 @untypedDerivedAsBase()
  %26 = icmp eq i32 %25, 43
  br i1 %26, label %27, label %49

27:                                               ; preds = %24
  %28 = call i32 @anyTyped()
  %29 = icmp eq i32 %28, 11
  br i1 %29, label %30, label %51

30:                                               ; preds = %27
  %31 = call i32 @alreadyAny()
  %32 = icmp eq i32 %31, 123
  br i1 %32, label %33, label %53

33:                                               ; preds = %30
  %34 = call double @numberCatchesInt()
  %35 = fcmp oeq double %34, 5.500000e+00
  br i1 %35, label %36, label %55

36:                                               ; preds = %33
  %37 = call double @typedCatchesUnchanged()
  %38 = fcmp oeq double %37, 1.600000e+01
  br i1 %38, label %39, label %57

39:                                               ; preds = %36
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

40:                                               ; preds = %57, %55, %53, %51, %49, %47, %45, %43, %41
  unreachable

41:                                               ; preds = %0
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4044607399571746621, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 136, ptr null)
  br label %40

43:                                               ; preds = %3
  %44 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12069005346676613813, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 137, ptr null)
  br label %40

45:                                               ; preds = %20
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8864937869768531853, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 138, ptr null)
  br label %40

47:                                               ; preds = %21
  %48 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5530502136322933383, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 139, ptr null)
  br label %40

49:                                               ; preds = %24
  %50 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5903054358280398641, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 140, ptr null)
  br label %40

51:                                               ; preds = %27
  %52 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7504679140485314426, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 141, ptr null)
  br label %40

53:                                               ; preds = %30
  %54 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9404695902159035447, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 142, ptr null)
  br label %40

55:                                               ; preds = %33
  %56 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4558786565675528182, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 143, ptr null)
  br label %40

57:                                               ; preds = %36
  %58 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7614099512854692056, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_5920818698833515450, i64 8), i32 144, ptr null)
  br label %40
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

