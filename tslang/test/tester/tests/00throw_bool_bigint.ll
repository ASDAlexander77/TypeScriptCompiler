; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$"___unbox<!ts.boolean>" = comdat any

$"___unbox<!ts.bigint>" = comdat any

@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
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
@s_8091808378024623192 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring\00", align 8
@s_9237349086447201248 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnumber\00", align 8
@s_16822532743728177535 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboolean\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_9186415445472728809 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv=-123456789012\00", align 8
@s_571532774284038691 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF42\00", align 8
@s_634733802172746332 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv=\00", align 8
@frmt_1189076927660576410 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-lli\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_10471070127234361712 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: boolean) catches a thrown boolean\00", align 8
@f_7484941213777721143 = internal constant [109 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_bool_bigint.ts\00", align 8
@m_6943169351122162168 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown boolean\00", align 8
@m_2677683196513557903 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa boolean variable can be thrown\00", align 8
@m_8637983907680937160 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: bigint) catches a thrown bigint\00", align 8
@m_5702300165540205539 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown bigint\00", align 8
@m_43741921607073719 = internal constant [71 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboolean and bigint catches let each other's exceptions through\00", align 8
@m_14407052258144518178 = internal constant [57 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa bigint boxes, unboxes and converts to a string\00", align 8
@m_1607514885731154834 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbigint values compare as signed integers\00", align 8
@_ZTIb = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
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

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare i64 @strtoll(ptr, ptr, i32)

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

declare i32 @strcmp(ptr, ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define i1 @typedBool() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca i1, align 1
  %3 = alloca i1, align 1
  store i1 false, ptr %2, align 1
  %4 = call ptr @__cxa_allocate_exception(i64 1)
  store i1 true, ptr %4, align 1
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIb, ptr null)
          to label %20 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIb
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIb)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %10
  br label %20

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i1, ptr %15, align 1
  store i1 %16, ptr %1, align 1
  %17 = load i1, ptr %1, align 1
  store i1 %17, ptr %2, align 1
  call void @__cxa_end_catch()
  %18 = load i1, ptr %2, align 1
  store i1 %18, ptr %3, align 1
  %19 = load i1, ptr %3, align 1
  ret i1 %19

20:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %21 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %21
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"___unbox<!ts.boolean>"(ptr %0) #3 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
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
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %24

20:                                               ; preds = %19
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr { i64, ptr, i1 }, ptr %21, i32 0, i32 2
  %23 = load i1, ptr %22, align 1
  store i1 %23, ptr %3, align 1
  br label %503

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
  br i1 %39, label %41, label %46

41:                                               ; preds = %40
  %42 = load ptr, ptr %2, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 2
  %44 = load double, ptr %43, align 8
  %45 = fcmp one double %44, 0.000000e+00
  store i1 %45, ptr %3, align 1
  br label %503

46:                                               ; preds = %40
  %47 = load ptr, ptr %2, align 8
  %48 = getelementptr { i64, ptr, i8 }, ptr %47, i32 0, i32 1
  %49 = load ptr, ptr %48, align 8
  %50 = ptrtoint ptr %49 to i64
  %51 = icmp ne i64 %50, 0
  %52 = zext i1 %51 to i32
  %53 = and i32 %52, 1
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %55, label %58

55:                                               ; preds = %46
  %56 = call i32 @strcmp(ptr %49, ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8))
  %57 = icmp eq i32 %56, 0
  br label %60

58:                                               ; preds = %46
  %59 = icmp eq i64 %50, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8) to i64)
  br label %60

60:                                               ; preds = %55, %58
  %61 = phi i1 [ %59, %58 ], [ %57, %55 ]
  br label %62

62:                                               ; preds = %60
  br i1 %61, label %63, label %71

63:                                               ; preds = %62
  %64 = load ptr, ptr %2, align 8
  %65 = getelementptr { i64, ptr, ptr }, ptr %64, i32 0, i32 2
  %66 = load ptr, ptr %65, align 8
  %67 = icmp eq ptr %66, null
  %68 = select i1 %67, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %66
  %69 = load i8, ptr %68, align 1
  %70 = icmp ne i8 %69, 0
  store i1 %70, ptr %3, align 1
  br label %503

71:                                               ; preds = %62
  %72 = load ptr, ptr %2, align 8
  %73 = getelementptr { i64, ptr, i8 }, ptr %72, i32 0, i32 1
  %74 = load ptr, ptr %73, align 8
  %75 = ptrtoint ptr %74 to i64
  %76 = icmp ne i64 %75, 0
  %77 = zext i1 %76 to i32
  %78 = and i32 %77, 1
  %79 = icmp ne i32 %78, 0
  br i1 %79, label %80, label %83

80:                                               ; preds = %71
  %81 = call i32 @strcmp(ptr %74, ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8))
  %82 = icmp eq i32 %81, 0
  br label %85

83:                                               ; preds = %71
  %84 = icmp eq i64 %75, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8) to i64)
  br label %85

85:                                               ; preds = %80, %83
  %86 = phi i1 [ %84, %83 ], [ %82, %80 ]
  br label %87

87:                                               ; preds = %85
  br i1 %86, label %88, label %92

88:                                               ; preds = %87
  %89 = load ptr, ptr %2, align 8
  %90 = getelementptr { i64, ptr, i1 }, ptr %89, i32 0, i32 2
  %91 = load i1, ptr %90, align 1
  store i1 %91, ptr %3, align 1
  br label %503

92:                                               ; preds = %87
  %93 = load ptr, ptr %2, align 8
  %94 = getelementptr { i64, ptr, i8 }, ptr %93, i32 0, i32 1
  %95 = load ptr, ptr %94, align 8
  %96 = ptrtoint ptr %95 to i64
  %97 = icmp ne i64 %96, 0
  %98 = zext i1 %97 to i32
  %99 = and i32 %98, 1
  %100 = icmp ne i32 %99, 0
  br i1 %100, label %101, label %104

101:                                              ; preds = %92
  %102 = call i32 @strcmp(ptr %95, ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8))
  %103 = icmp eq i32 %102, 0
  br label %106

104:                                              ; preds = %92
  %105 = icmp eq i64 %96, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8) to i64)
  br label %106

106:                                              ; preds = %101, %104
  %107 = phi i1 [ %105, %104 ], [ %103, %101 ]
  br label %108

108:                                              ; preds = %106
  br i1 %107, label %109, label %114

109:                                              ; preds = %108
  %110 = load ptr, ptr %2, align 8
  %111 = getelementptr { i64, ptr, float }, ptr %110, i32 0, i32 2
  %112 = load float, ptr %111, align 4
  %113 = fcmp one float %112, 0.000000e+00
  store i1 %113, ptr %3, align 1
  br label %503

114:                                              ; preds = %108
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
  br i1 %129, label %131, label %136

131:                                              ; preds = %130
  %132 = load ptr, ptr %2, align 8
  %133 = getelementptr { i64, ptr, i32 }, ptr %132, i32 0, i32 2
  %134 = load i32, ptr %133, align 4
  %135 = icmp ne i32 %134, 0
  store i1 %135, ptr %3, align 1
  br label %503

136:                                              ; preds = %130
  %137 = load ptr, ptr %2, align 8
  %138 = getelementptr { i64, ptr, i8 }, ptr %137, i32 0, i32 1
  %139 = load ptr, ptr %138, align 8
  %140 = ptrtoint ptr %139 to i64
  %141 = icmp ne i64 %140, 0
  %142 = zext i1 %141 to i32
  %143 = and i32 %142, 1
  %144 = icmp ne i32 %143, 0
  br i1 %144, label %145, label %148

145:                                              ; preds = %136
  %146 = call i32 @strcmp(ptr %139, ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8))
  %147 = icmp eq i32 %146, 0
  br label %150

148:                                              ; preds = %136
  %149 = icmp eq i64 %140, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8) to i64)
  br label %150

150:                                              ; preds = %145, %148
  %151 = phi i1 [ %149, %148 ], [ %147, %145 ]
  br label %152

152:                                              ; preds = %150
  br i1 %151, label %153, label %158

153:                                              ; preds = %152
  %154 = load ptr, ptr %2, align 8
  %155 = getelementptr { i64, ptr, i32 }, ptr %154, i32 0, i32 2
  %156 = load i32, ptr %155, align 4
  %157 = icmp ne i32 %156, 0
  store i1 %157, ptr %3, align 1
  br label %503

158:                                              ; preds = %152
  %159 = load ptr, ptr %2, align 8
  %160 = getelementptr { i64, ptr, i8 }, ptr %159, i32 0, i32 1
  %161 = load ptr, ptr %160, align 8
  %162 = ptrtoint ptr %161 to i64
  %163 = icmp ne i64 %162, 0
  %164 = zext i1 %163 to i32
  %165 = and i32 %164, 1
  %166 = icmp ne i32 %165, 0
  br i1 %166, label %167, label %170

167:                                              ; preds = %158
  %168 = call i32 @strcmp(ptr %161, ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8))
  %169 = icmp eq i32 %168, 0
  br label %172

170:                                              ; preds = %158
  %171 = icmp eq i64 %162, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8) to i64)
  br label %172

172:                                              ; preds = %167, %170
  %173 = phi i1 [ %171, %170 ], [ %169, %167 ]
  br label %174

174:                                              ; preds = %172
  br i1 %173, label %175, label %180

175:                                              ; preds = %174
  %176 = load ptr, ptr %2, align 8
  %177 = getelementptr { i64, ptr, i32 }, ptr %176, i32 0, i32 2
  %178 = load i32, ptr %177, align 4
  %179 = icmp ne i32 %178, 0
  store i1 %179, ptr %3, align 1
  br label %503

180:                                              ; preds = %174
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
  %201 = icmp ne i64 %200, 0
  store i1 %201, ptr %3, align 1
  br label %503

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
  br i1 %217, label %219, label %224

219:                                              ; preds = %218
  %220 = load ptr, ptr %2, align 8
  %221 = getelementptr { i64, ptr, double }, ptr %220, i32 0, i32 2
  %222 = load double, ptr %221, align 8
  %223 = fcmp one double %222, 0.000000e+00
  store i1 %223, ptr %3, align 1
  br label %503

224:                                              ; preds = %218
  %225 = load ptr, ptr %2, align 8
  %226 = getelementptr { i64, ptr, i8 }, ptr %225, i32 0, i32 1
  %227 = load ptr, ptr %226, align 8
  %228 = ptrtoint ptr %227 to i64
  %229 = icmp ne i64 %228, 0
  %230 = zext i1 %229 to i32
  %231 = and i32 %230, 1
  %232 = icmp ne i32 %231, 0
  br i1 %232, label %233, label %236

233:                                              ; preds = %224
  %234 = call i32 @strcmp(ptr %227, ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8))
  %235 = icmp eq i32 %234, 0
  br label %238

236:                                              ; preds = %224
  %237 = icmp eq i64 %228, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8) to i64)
  br label %238

238:                                              ; preds = %233, %236
  %239 = phi i1 [ %237, %236 ], [ %235, %233 ]
  br label %240

240:                                              ; preds = %238
  br i1 %239, label %241, label %246

241:                                              ; preds = %240
  %242 = load ptr, ptr %2, align 8
  %243 = getelementptr { i64, ptr, i64 }, ptr %242, i32 0, i32 2
  %244 = load i64, ptr %243, align 8
  %245 = icmp ne i64 %244, 0
  store i1 %245, ptr %3, align 1
  br label %503

246:                                              ; preds = %240
  %247 = load ptr, ptr %2, align 8
  %248 = getelementptr { i64, ptr, i8 }, ptr %247, i32 0, i32 1
  %249 = load ptr, ptr %248, align 8
  %250 = ptrtoint ptr %249 to i64
  %251 = icmp ne i64 %250, 0
  %252 = zext i1 %251 to i32
  %253 = and i32 %252, 1
  %254 = icmp ne i32 %253, 0
  br i1 %254, label %255, label %258

255:                                              ; preds = %246
  %256 = call i32 @strcmp(ptr %249, ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8))
  %257 = icmp eq i32 %256, 0
  br label %260

258:                                              ; preds = %246
  %259 = icmp eq i64 %250, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8) to i64)
  br label %260

260:                                              ; preds = %255, %258
  %261 = phi i1 [ %259, %258 ], [ %257, %255 ]
  br label %262

262:                                              ; preds = %260
  br i1 %261, label %263, label %268

263:                                              ; preds = %262
  %264 = load ptr, ptr %2, align 8
  %265 = getelementptr { i64, ptr, i64 }, ptr %264, i32 0, i32 2
  %266 = load i64, ptr %265, align 8
  %267 = icmp ne i64 %266, 0
  store i1 %267, ptr %3, align 1
  br label %503

268:                                              ; preds = %262
  %269 = load ptr, ptr %2, align 8
  %270 = getelementptr { i64, ptr, i8 }, ptr %269, i32 0, i32 1
  %271 = load ptr, ptr %270, align 8
  %272 = ptrtoint ptr %271 to i64
  %273 = icmp ne i64 %272, 0
  %274 = zext i1 %273 to i32
  %275 = and i32 %274, 1
  %276 = icmp ne i32 %275, 0
  br i1 %276, label %277, label %280

277:                                              ; preds = %268
  %278 = call i32 @strcmp(ptr %271, ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8))
  %279 = icmp eq i32 %278, 0
  br label %282

280:                                              ; preds = %268
  %281 = icmp eq i64 %272, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8) to i64)
  br label %282

282:                                              ; preds = %277, %280
  %283 = phi i1 [ %281, %280 ], [ %279, %277 ]
  br label %284

284:                                              ; preds = %282
  br i1 %283, label %285, label %290

285:                                              ; preds = %284
  %286 = load ptr, ptr %2, align 8
  %287 = getelementptr { i64, ptr, i64 }, ptr %286, i32 0, i32 2
  %288 = load i64, ptr %287, align 8
  %289 = icmp ne i64 %288, 0
  store i1 %289, ptr %3, align 1
  br label %503

290:                                              ; preds = %284
  %291 = load ptr, ptr %2, align 8
  %292 = getelementptr { i64, ptr, i8 }, ptr %291, i32 0, i32 1
  %293 = load ptr, ptr %292, align 8
  %294 = ptrtoint ptr %293 to i64
  %295 = icmp ne i64 %294, 0
  %296 = zext i1 %295 to i32
  %297 = and i32 %296, 1
  %298 = icmp ne i32 %297, 0
  br i1 %298, label %299, label %302

299:                                              ; preds = %290
  %300 = call i32 @strcmp(ptr %293, ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8))
  %301 = icmp eq i32 %300, 0
  br label %304

302:                                              ; preds = %290
  %303 = icmp eq i64 %294, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8) to i64)
  br label %304

304:                                              ; preds = %299, %302
  %305 = phi i1 [ %303, %302 ], [ %301, %299 ]
  br label %306

306:                                              ; preds = %304
  br i1 %305, label %307, label %312

307:                                              ; preds = %306
  %308 = load ptr, ptr %2, align 8
  %309 = getelementptr { i64, ptr, i8 }, ptr %308, i32 0, i32 2
  %310 = load i8, ptr %309, align 1
  %311 = icmp ne i8 %310, 0
  store i1 %311, ptr %3, align 1
  br label %503

312:                                              ; preds = %306
  %313 = load ptr, ptr %2, align 8
  %314 = getelementptr { i64, ptr, i8 }, ptr %313, i32 0, i32 1
  %315 = load ptr, ptr %314, align 8
  %316 = ptrtoint ptr %315 to i64
  %317 = icmp ne i64 %316, 0
  %318 = zext i1 %317 to i32
  %319 = and i32 %318, 1
  %320 = icmp ne i32 %319, 0
  br i1 %320, label %321, label %324

321:                                              ; preds = %312
  %322 = call i32 @strcmp(ptr %315, ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8))
  %323 = icmp eq i32 %322, 0
  br label %326

324:                                              ; preds = %312
  %325 = icmp eq i64 %316, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8) to i64)
  br label %326

326:                                              ; preds = %321, %324
  %327 = phi i1 [ %325, %324 ], [ %323, %321 ]
  br label %328

328:                                              ; preds = %326
  br i1 %327, label %329, label %334

329:                                              ; preds = %328
  %330 = load ptr, ptr %2, align 8
  %331 = getelementptr { i64, ptr, i64 }, ptr %330, i32 0, i32 2
  %332 = load i64, ptr %331, align 8
  %333 = icmp ne i64 %332, 0
  store i1 %333, ptr %3, align 1
  br label %503

334:                                              ; preds = %328
  %335 = load ptr, ptr %2, align 8
  %336 = getelementptr { i64, ptr, i8 }, ptr %335, i32 0, i32 1
  %337 = load ptr, ptr %336, align 8
  %338 = ptrtoint ptr %337 to i64
  %339 = icmp ne i64 %338, 0
  %340 = zext i1 %339 to i32
  %341 = and i32 %340, 1
  %342 = icmp ne i32 %341, 0
  br i1 %342, label %343, label %346

343:                                              ; preds = %334
  %344 = call i32 @strcmp(ptr %337, ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8))
  %345 = icmp eq i32 %344, 0
  br label %348

346:                                              ; preds = %334
  %347 = icmp eq i64 %338, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8) to i64)
  br label %348

348:                                              ; preds = %343, %346
  %349 = phi i1 [ %347, %346 ], [ %345, %343 ]
  br label %350

350:                                              ; preds = %348
  br i1 %349, label %351, label %356

351:                                              ; preds = %350
  %352 = load ptr, ptr %2, align 8
  %353 = getelementptr { i64, ptr, i16 }, ptr %352, i32 0, i32 2
  %354 = load i16, ptr %353, align 2
  %355 = icmp ne i16 %354, 0
  store i1 %355, ptr %3, align 1
  br label %503

356:                                              ; preds = %350
  %357 = load ptr, ptr %2, align 8
  %358 = getelementptr { i64, ptr, i8 }, ptr %357, i32 0, i32 1
  %359 = load ptr, ptr %358, align 8
  %360 = ptrtoint ptr %359 to i64
  %361 = icmp ne i64 %360, 0
  %362 = zext i1 %361 to i32
  %363 = and i32 %362, 1
  %364 = icmp ne i32 %363, 0
  br i1 %364, label %365, label %368

365:                                              ; preds = %356
  %366 = call i32 @strcmp(ptr %359, ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8))
  %367 = icmp eq i32 %366, 0
  br label %370

368:                                              ; preds = %356
  %369 = icmp eq i64 %360, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8) to i64)
  br label %370

370:                                              ; preds = %365, %368
  %371 = phi i1 [ %369, %368 ], [ %367, %365 ]
  br label %372

372:                                              ; preds = %370
  br i1 %371, label %373, label %378

373:                                              ; preds = %372
  %374 = load ptr, ptr %2, align 8
  %375 = getelementptr { i64, ptr, i16 }, ptr %374, i32 0, i32 2
  %376 = load i16, ptr %375, align 2
  %377 = icmp ne i16 %376, 0
  store i1 %377, ptr %3, align 1
  br label %503

378:                                              ; preds = %372
  %379 = load ptr, ptr %2, align 8
  %380 = getelementptr { i64, ptr, i8 }, ptr %379, i32 0, i32 1
  %381 = load ptr, ptr %380, align 8
  %382 = ptrtoint ptr %381 to i64
  %383 = icmp ne i64 %382, 0
  %384 = zext i1 %383 to i32
  %385 = and i32 %384, 1
  %386 = icmp ne i32 %385, 0
  br i1 %386, label %387, label %390

387:                                              ; preds = %378
  %388 = call i32 @strcmp(ptr %381, ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8))
  %389 = icmp eq i32 %388, 0
  br label %392

390:                                              ; preds = %378
  %391 = icmp eq i64 %382, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8) to i64)
  br label %392

392:                                              ; preds = %387, %390
  %393 = phi i1 [ %391, %390 ], [ %389, %387 ]
  br label %394

394:                                              ; preds = %392
  br i1 %393, label %395, label %400

395:                                              ; preds = %394
  %396 = load ptr, ptr %2, align 8
  %397 = getelementptr { i64, ptr, i16 }, ptr %396, i32 0, i32 2
  %398 = load i16, ptr %397, align 2
  %399 = icmp ne i16 %398, 0
  store i1 %399, ptr %3, align 1
  br label %503

400:                                              ; preds = %394
  %401 = load ptr, ptr %2, align 8
  %402 = getelementptr { i64, ptr, i8 }, ptr %401, i32 0, i32 1
  %403 = load ptr, ptr %402, align 8
  %404 = ptrtoint ptr %403 to i64
  %405 = icmp ne i64 %404, 0
  %406 = zext i1 %405 to i32
  %407 = and i32 %406, 1
  %408 = icmp ne i32 %407, 0
  br i1 %408, label %409, label %412

409:                                              ; preds = %400
  %410 = call i32 @strcmp(ptr %403, ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8))
  %411 = icmp eq i32 %410, 0
  br label %414

412:                                              ; preds = %400
  %413 = icmp eq i64 %404, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8) to i64)
  br label %414

414:                                              ; preds = %409, %412
  %415 = phi i1 [ %413, %412 ], [ %411, %409 ]
  br label %416

416:                                              ; preds = %414
  br i1 %415, label %417, label %422

417:                                              ; preds = %416
  %418 = load ptr, ptr %2, align 8
  %419 = getelementptr { i64, ptr, i8 }, ptr %418, i32 0, i32 2
  %420 = load i8, ptr %419, align 1
  %421 = icmp ne i8 %420, 0
  store i1 %421, ptr %3, align 1
  br label %503

422:                                              ; preds = %416
  %423 = load ptr, ptr %2, align 8
  %424 = getelementptr { i64, ptr, i8 }, ptr %423, i32 0, i32 1
  %425 = load ptr, ptr %424, align 8
  %426 = ptrtoint ptr %425 to i64
  %427 = icmp ne i64 %426, 0
  %428 = zext i1 %427 to i32
  %429 = and i32 %428, 1
  %430 = icmp ne i32 %429, 0
  br i1 %430, label %431, label %434

431:                                              ; preds = %422
  %432 = call i32 @strcmp(ptr %425, ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8))
  %433 = icmp eq i32 %432, 0
  br label %436

434:                                              ; preds = %422
  %435 = icmp eq i64 %426, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8) to i64)
  br label %436

436:                                              ; preds = %431, %434
  %437 = phi i1 [ %435, %434 ], [ %433, %431 ]
  br label %438

438:                                              ; preds = %436
  br i1 %437, label %439, label %444

439:                                              ; preds = %438
  %440 = load ptr, ptr %2, align 8
  %441 = getelementptr { i64, ptr, i8 }, ptr %440, i32 0, i32 2
  %442 = load i8, ptr %441, align 1
  %443 = icmp ne i8 %442, 0
  store i1 %443, ptr %3, align 1
  br label %503

444:                                              ; preds = %438
  %445 = load ptr, ptr %2, align 8
  %446 = getelementptr { i64, ptr, i8 }, ptr %445, i32 0, i32 1
  %447 = load ptr, ptr %446, align 8
  %448 = ptrtoint ptr %447 to i64
  %449 = icmp ne i64 %448, 0
  %450 = zext i1 %449 to i32
  %451 = and i32 %450, 1
  %452 = icmp ne i32 %451, 0
  br i1 %452, label %453, label %456

453:                                              ; preds = %444
  %454 = call i32 @strcmp(ptr %447, ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8))
  %455 = icmp eq i32 %454, 0
  br label %458

456:                                              ; preds = %444
  %457 = icmp eq i64 %448, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8) to i64)
  br label %458

458:                                              ; preds = %453, %456
  %459 = phi i1 [ %457, %456 ], [ %455, %453 ]
  br label %460

460:                                              ; preds = %458
  br i1 %459, label %461, label %466

461:                                              ; preds = %460
  %462 = load ptr, ptr %2, align 8
  %463 = getelementptr { i64, ptr, i8 }, ptr %462, i32 0, i32 2
  %464 = load i8, ptr %463, align 1
  %465 = icmp ne i8 %464, 0
  store i1 %465, ptr %3, align 1
  br label %503

466:                                              ; preds = %460
  %467 = load ptr, ptr %2, align 8
  %468 = getelementptr { i64, ptr, i8 }, ptr %467, i32 0, i32 1
  %469 = load ptr, ptr %468, align 8
  %470 = ptrtoint ptr %469 to i64
  %471 = icmp ne i64 %470, 0
  %472 = zext i1 %471 to i32
  %473 = and i32 %472, 1
  %474 = icmp ne i32 %473, 0
  br i1 %474, label %475, label %478

475:                                              ; preds = %466
  %476 = call i32 @strcmp(ptr %469, ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8))
  %477 = icmp eq i32 %476, 0
  br label %480

478:                                              ; preds = %466
  %479 = icmp eq i64 %470, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8) to i64)
  br label %480

480:                                              ; preds = %475, %478
  %481 = phi i1 [ %479, %478 ], [ %477, %475 ]
  br label %482

482:                                              ; preds = %480
  br i1 %481, label %483, label %484

483:                                              ; preds = %482
  store i1 false, ptr %3, align 1
  br label %503

484:                                              ; preds = %482
  %485 = load ptr, ptr %2, align 8
  %486 = getelementptr { i64, ptr, i8 }, ptr %485, i32 0, i32 1
  %487 = load ptr, ptr %486, align 8
  %488 = ptrtoint ptr %487 to i64
  %489 = icmp ne i64 %488, 0
  %490 = zext i1 %489 to i32
  %491 = and i32 %490, 1
  %492 = icmp ne i32 %491, 0
  br i1 %492, label %493, label %496

493:                                              ; preds = %484
  %494 = call i32 @strcmp(ptr %487, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8))
  %495 = icmp eq i32 %494, 0
  br label %498

496:                                              ; preds = %484
  %497 = icmp eq i64 %488, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8) to i64)
  br label %498

498:                                              ; preds = %493, %496
  %499 = phi i1 [ %497, %496 ], [ %495, %493 ]
  br label %500

500:                                              ; preds = %498
  br i1 %499, label %501, label %502

501:                                              ; preds = %500
  store i1 false, ptr %3, align 1
  br label %503

502:                                              ; preds = %500
  store i1 true, ptr %3, align 1
  br label %503

503:                                              ; preds = %20, %41, %63, %88, %109, %131, %153, %175, %197, %219, %241, %263, %285, %307, %329, %351, %373, %395, %417, %439, %461, %483, %501, %502
  %504 = load i1, ptr %3, align 1
  ret i1 %504
}

; Function Attrs: mustprogress
define i1 @untypedBool() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca i1, align 1
  store i1 true, ptr %3, align 1
  %5 = call ptr @__cxa_allocate_exception(i64 1)
  store i1 false, ptr %5, align 1
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIb, ptr null)
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %21
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %30
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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %39
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
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %66
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc11, %.noexc7
  %99 = phi ptr [ %94, %.noexc7 ], [ %90, %.noexc11 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc4
  %109 = phi ptr [ %104, %.noexc4 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc6
  %119 = phi ptr [ %114, %.noexc6 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc3, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc3 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc12, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc12 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc2, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc2 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc1, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc1 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc5, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc5 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc9, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc9 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke i1 @"___unbox<!ts.boolean>"(ptr %145)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %144
  store i1 %146, ptr %3, align 1
  call void @__cxa_end_catch()
  %147 = load i1, ptr %3, align 1
  store i1 %147, ptr %4, align 1
  %148 = load i1, ptr %4, align 1
  ret i1 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %57, %89, %6, %12, %144, %91, %111, %30, %101, %66, %48, %39, %21
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i1 @boolVariable() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca i1, align 1
  %3 = alloca i1, align 1
  %4 = alloca i1, align 1
  store i1 true, ptr %3, align 1
  store i1 false, ptr %2, align 1
  %5 = load i1, ptr %3, align 1
  %6 = call ptr @__cxa_allocate_exception(i64 1)
  store i1 %5, ptr %6, align 1
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIb, ptr null)
          to label %22 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIb
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIb)
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = extractvalue { ptr, i32 } %8, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %12
  br label %22

15:                                               ; preds = %7
  %16 = extractvalue { ptr, i32 } %8, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = load i1, ptr %17, align 1
  store i1 %18, ptr %1, align 1
  %19 = load i1, ptr %1, align 1
  store i1 %19, ptr %2, align 1
  call void @__cxa_end_catch()
  %20 = load i1, ptr %2, align 1
  store i1 %20, ptr %4, align 1
  %21 = load i1, ptr %4, align 1
  ret i1 %21

22:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %12
  %23 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %23
}

; Function Attrs: mustprogress
define i64 @typedBigInt() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i64, align 8
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 0, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store i64 123456789012, ptr %4, align 8
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIx, ptr null)
          to label %20 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIx
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIx)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %10
  br label %20

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i64, ptr %15, align 8
  store i64 %16, ptr %1, align 8
  %17 = load i64, ptr %1, align 8
  store i64 %17, ptr %2, align 8
  call void @__cxa_end_catch()
  %18 = load i64, ptr %2, align 8
  store i64 %18, ptr %3, align 8
  %19 = load i64, ptr %3, align 8
  ret i64 %19

20:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %21 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %21
}

; Function Attrs: mustprogress
define linkonce_odr i64 @"___unbox<!ts.bigint>"(ptr %0) #3 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
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
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %24

20:                                               ; preds = %19
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr { i64, ptr, i64 }, ptr %21, i32 0, i32 2
  %23 = load i64, ptr %22, align 8
  store i64 %23, ptr %3, align 8
  br label %461

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
  br i1 %39, label %41, label %46

41:                                               ; preds = %40
  %42 = load ptr, ptr %2, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %42, i32 0, i32 2
  %44 = load double, ptr %43, align 8
  %45 = fptosi double %44 to i64
  store i64 %45, ptr %3, align 8
  br label %461

46:                                               ; preds = %40
  %47 = load ptr, ptr %2, align 8
  %48 = getelementptr { i64, ptr, i8 }, ptr %47, i32 0, i32 1
  %49 = load ptr, ptr %48, align 8
  %50 = ptrtoint ptr %49 to i64
  %51 = icmp ne i64 %50, 0
  %52 = zext i1 %51 to i32
  %53 = and i32 %52, 1
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %55, label %58

55:                                               ; preds = %46
  %56 = call i32 @strcmp(ptr %49, ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8))
  %57 = icmp eq i32 %56, 0
  br label %60

58:                                               ; preds = %46
  %59 = icmp eq i64 %50, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_8091808378024623192, i64 8) to i64)
  br label %60

60:                                               ; preds = %55, %58
  %61 = phi i1 [ %59, %58 ], [ %57, %55 ]
  br label %62

62:                                               ; preds = %60
  br i1 %61, label %63, label %68

63:                                               ; preds = %62
  %64 = load ptr, ptr %2, align 8
  %65 = getelementptr { i64, ptr, ptr }, ptr %64, i32 0, i32 2
  %66 = load ptr, ptr %65, align 8
  %67 = call i64 @strtoll(ptr %66, ptr null, i32 10)
  store i64 %67, ptr %3, align 8
  br label %461

68:                                               ; preds = %62
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
  %89 = zext i1 %88 to i64
  store i64 %89, ptr %3, align 8
  br label %461

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
  br i1 %105, label %107, label %112

107:                                              ; preds = %106
  %108 = load ptr, ptr %2, align 8
  %109 = getelementptr { i64, ptr, float }, ptr %108, i32 0, i32 2
  %110 = load float, ptr %109, align 4
  %111 = fptosi float %110 to i64
  store i64 %111, ptr %3, align 8
  br label %461

112:                                              ; preds = %106
  %113 = load ptr, ptr %2, align 8
  %114 = getelementptr { i64, ptr, i8 }, ptr %113, i32 0, i32 1
  %115 = load ptr, ptr %114, align 8
  %116 = ptrtoint ptr %115 to i64
  %117 = icmp ne i64 %116, 0
  %118 = zext i1 %117 to i32
  %119 = and i32 %118, 1
  %120 = icmp ne i32 %119, 0
  br i1 %120, label %121, label %124

121:                                              ; preds = %112
  %122 = call i32 @strcmp(ptr %115, ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8))
  %123 = icmp eq i32 %122, 0
  br label %126

124:                                              ; preds = %112
  %125 = icmp eq i64 %116, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3094732814638223685, i64 8) to i64)
  br label %126

126:                                              ; preds = %121, %124
  %127 = phi i1 [ %125, %124 ], [ %123, %121 ]
  br label %128

128:                                              ; preds = %126
  br i1 %127, label %129, label %134

129:                                              ; preds = %128
  %130 = load ptr, ptr %2, align 8
  %131 = getelementptr { i64, ptr, i32 }, ptr %130, i32 0, i32 2
  %132 = load i32, ptr %131, align 4
  %133 = zext i32 %132 to i64
  store i64 %133, ptr %3, align 8
  br label %461

134:                                              ; preds = %128
  %135 = load ptr, ptr %2, align 8
  %136 = getelementptr { i64, ptr, i8 }, ptr %135, i32 0, i32 1
  %137 = load ptr, ptr %136, align 8
  %138 = ptrtoint ptr %137 to i64
  %139 = icmp ne i64 %138, 0
  %140 = zext i1 %139 to i32
  %141 = and i32 %140, 1
  %142 = icmp ne i32 %141, 0
  br i1 %142, label %143, label %146

143:                                              ; preds = %134
  %144 = call i32 @strcmp(ptr %137, ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8))
  %145 = icmp eq i32 %144, 0
  br label %148

146:                                              ; preds = %134
  %147 = icmp eq i64 %138, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9332519565554727459, i64 8) to i64)
  br label %148

148:                                              ; preds = %143, %146
  %149 = phi i1 [ %147, %146 ], [ %145, %143 ]
  br label %150

150:                                              ; preds = %148
  br i1 %149, label %151, label %156

151:                                              ; preds = %150
  %152 = load ptr, ptr %2, align 8
  %153 = getelementptr { i64, ptr, i32 }, ptr %152, i32 0, i32 2
  %154 = load i32, ptr %153, align 4
  %155 = zext i32 %154 to i64
  store i64 %155, ptr %3, align 8
  br label %461

156:                                              ; preds = %150
  %157 = load ptr, ptr %2, align 8
  %158 = getelementptr { i64, ptr, i8 }, ptr %157, i32 0, i32 1
  %159 = load ptr, ptr %158, align 8
  %160 = ptrtoint ptr %159 to i64
  %161 = icmp ne i64 %160, 0
  %162 = zext i1 %161 to i32
  %163 = and i32 %162, 1
  %164 = icmp ne i32 %163, 0
  br i1 %164, label %165, label %168

165:                                              ; preds = %156
  %166 = call i32 @strcmp(ptr %159, ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8))
  %167 = icmp eq i32 %166, 0
  br label %170

168:                                              ; preds = %156
  %169 = icmp eq i64 %160, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8) to i64)
  br label %170

170:                                              ; preds = %165, %168
  %171 = phi i1 [ %169, %168 ], [ %167, %165 ]
  br label %172

172:                                              ; preds = %170
  br i1 %171, label %173, label %178

173:                                              ; preds = %172
  %174 = load ptr, ptr %2, align 8
  %175 = getelementptr { i64, ptr, i32 }, ptr %174, i32 0, i32 2
  %176 = load i32, ptr %175, align 4
  %177 = zext i32 %176 to i64
  store i64 %177, ptr %3, align 8
  br label %461

178:                                              ; preds = %172
  %179 = load ptr, ptr %2, align 8
  %180 = getelementptr { i64, ptr, i8 }, ptr %179, i32 0, i32 1
  %181 = load ptr, ptr %180, align 8
  %182 = ptrtoint ptr %181 to i64
  %183 = icmp ne i64 %182, 0
  %184 = zext i1 %183 to i32
  %185 = and i32 %184, 1
  %186 = icmp ne i32 %185, 0
  br i1 %186, label %187, label %190

187:                                              ; preds = %178
  %188 = call i32 @strcmp(ptr %181, ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8))
  %189 = icmp eq i32 %188, 0
  br label %192

190:                                              ; preds = %178
  %191 = icmp eq i64 %182, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8) to i64)
  br label %192

192:                                              ; preds = %187, %190
  %193 = phi i1 [ %191, %190 ], [ %189, %187 ]
  br label %194

194:                                              ; preds = %192
  br i1 %193, label %195, label %199

195:                                              ; preds = %194
  %196 = load ptr, ptr %2, align 8
  %197 = getelementptr { i64, ptr, i64 }, ptr %196, i32 0, i32 2
  %198 = load i64, ptr %197, align 8
  store i64 %198, ptr %3, align 8
  br label %461

199:                                              ; preds = %194
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
  br i1 %214, label %216, label %221

216:                                              ; preds = %215
  %217 = load ptr, ptr %2, align 8
  %218 = getelementptr { i64, ptr, double }, ptr %217, i32 0, i32 2
  %219 = load double, ptr %218, align 8
  %220 = fptosi double %219 to i64
  store i64 %220, ptr %3, align 8
  br label %461

221:                                              ; preds = %215
  %222 = load ptr, ptr %2, align 8
  %223 = getelementptr { i64, ptr, i8 }, ptr %222, i32 0, i32 1
  %224 = load ptr, ptr %223, align 8
  %225 = ptrtoint ptr %224 to i64
  %226 = icmp ne i64 %225, 0
  %227 = zext i1 %226 to i32
  %228 = and i32 %227, 1
  %229 = icmp ne i32 %228, 0
  br i1 %229, label %230, label %233

230:                                              ; preds = %221
  %231 = call i32 @strcmp(ptr %224, ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8))
  %232 = icmp eq i32 %231, 0
  br label %235

233:                                              ; preds = %221
  %234 = icmp eq i64 %225, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8) to i64)
  br label %235

235:                                              ; preds = %230, %233
  %236 = phi i1 [ %234, %233 ], [ %232, %230 ]
  br label %237

237:                                              ; preds = %235
  br i1 %236, label %238, label %242

238:                                              ; preds = %237
  %239 = load ptr, ptr %2, align 8
  %240 = getelementptr { i64, ptr, i64 }, ptr %239, i32 0, i32 2
  %241 = load i64, ptr %240, align 8
  store i64 %241, ptr %3, align 8
  br label %461

242:                                              ; preds = %237
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
  br i1 %257, label %259, label %263

259:                                              ; preds = %258
  %260 = load ptr, ptr %2, align 8
  %261 = getelementptr { i64, ptr, i64 }, ptr %260, i32 0, i32 2
  %262 = load i64, ptr %261, align 8
  store i64 %262, ptr %3, align 8
  br label %461

263:                                              ; preds = %258
  %264 = load ptr, ptr %2, align 8
  %265 = getelementptr { i64, ptr, i8 }, ptr %264, i32 0, i32 1
  %266 = load ptr, ptr %265, align 8
  %267 = ptrtoint ptr %266 to i64
  %268 = icmp ne i64 %267, 0
  %269 = zext i1 %268 to i32
  %270 = and i32 %269, 1
  %271 = icmp ne i32 %270, 0
  br i1 %271, label %272, label %275

272:                                              ; preds = %263
  %273 = call i32 @strcmp(ptr %266, ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8))
  %274 = icmp eq i32 %273, 0
  br label %277

275:                                              ; preds = %263
  %276 = icmp eq i64 %267, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8) to i64)
  br label %277

277:                                              ; preds = %272, %275
  %278 = phi i1 [ %276, %275 ], [ %274, %272 ]
  br label %279

279:                                              ; preds = %277
  br i1 %278, label %280, label %284

280:                                              ; preds = %279
  %281 = load ptr, ptr %2, align 8
  %282 = getelementptr { i64, ptr, i64 }, ptr %281, i32 0, i32 2
  %283 = load i64, ptr %282, align 8
  store i64 %283, ptr %3, align 8
  br label %461

284:                                              ; preds = %279
  %285 = load ptr, ptr %2, align 8
  %286 = getelementptr { i64, ptr, i8 }, ptr %285, i32 0, i32 1
  %287 = load ptr, ptr %286, align 8
  %288 = ptrtoint ptr %287 to i64
  %289 = icmp ne i64 %288, 0
  %290 = zext i1 %289 to i32
  %291 = and i32 %290, 1
  %292 = icmp ne i32 %291, 0
  br i1 %292, label %293, label %296

293:                                              ; preds = %284
  %294 = call i32 @strcmp(ptr %287, ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8))
  %295 = icmp eq i32 %294, 0
  br label %298

296:                                              ; preds = %284
  %297 = icmp eq i64 %288, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8) to i64)
  br label %298

298:                                              ; preds = %293, %296
  %299 = phi i1 [ %297, %296 ], [ %295, %293 ]
  br label %300

300:                                              ; preds = %298
  br i1 %299, label %301, label %306

301:                                              ; preds = %300
  %302 = load ptr, ptr %2, align 8
  %303 = getelementptr { i64, ptr, i8 }, ptr %302, i32 0, i32 2
  %304 = load i8, ptr %303, align 1
  %305 = zext i8 %304 to i64
  store i64 %305, ptr %3, align 8
  br label %461

306:                                              ; preds = %300
  %307 = load ptr, ptr %2, align 8
  %308 = getelementptr { i64, ptr, i8 }, ptr %307, i32 0, i32 1
  %309 = load ptr, ptr %308, align 8
  %310 = ptrtoint ptr %309 to i64
  %311 = icmp ne i64 %310, 0
  %312 = zext i1 %311 to i32
  %313 = and i32 %312, 1
  %314 = icmp ne i32 %313, 0
  br i1 %314, label %315, label %318

315:                                              ; preds = %306
  %316 = call i32 @strcmp(ptr %309, ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8))
  %317 = icmp eq i32 %316, 0
  br label %320

318:                                              ; preds = %306
  %319 = icmp eq i64 %310, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8) to i64)
  br label %320

320:                                              ; preds = %315, %318
  %321 = phi i1 [ %319, %318 ], [ %317, %315 ]
  br label %322

322:                                              ; preds = %320
  br i1 %321, label %323, label %327

323:                                              ; preds = %322
  %324 = load ptr, ptr %2, align 8
  %325 = getelementptr { i64, ptr, i64 }, ptr %324, i32 0, i32 2
  %326 = load i64, ptr %325, align 8
  store i64 %326, ptr %3, align 8
  br label %461

327:                                              ; preds = %322
  %328 = load ptr, ptr %2, align 8
  %329 = getelementptr { i64, ptr, i8 }, ptr %328, i32 0, i32 1
  %330 = load ptr, ptr %329, align 8
  %331 = ptrtoint ptr %330 to i64
  %332 = icmp ne i64 %331, 0
  %333 = zext i1 %332 to i32
  %334 = and i32 %333, 1
  %335 = icmp ne i32 %334, 0
  br i1 %335, label %336, label %339

336:                                              ; preds = %327
  %337 = call i32 @strcmp(ptr %330, ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8))
  %338 = icmp eq i32 %337, 0
  br label %341

339:                                              ; preds = %327
  %340 = icmp eq i64 %331, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8) to i64)
  br label %341

341:                                              ; preds = %336, %339
  %342 = phi i1 [ %340, %339 ], [ %338, %336 ]
  br label %343

343:                                              ; preds = %341
  br i1 %342, label %344, label %349

344:                                              ; preds = %343
  %345 = load ptr, ptr %2, align 8
  %346 = getelementptr { i64, ptr, i16 }, ptr %345, i32 0, i32 2
  %347 = load i16, ptr %346, align 2
  %348 = zext i16 %347 to i64
  store i64 %348, ptr %3, align 8
  br label %461

349:                                              ; preds = %343
  %350 = load ptr, ptr %2, align 8
  %351 = getelementptr { i64, ptr, i8 }, ptr %350, i32 0, i32 1
  %352 = load ptr, ptr %351, align 8
  %353 = ptrtoint ptr %352 to i64
  %354 = icmp ne i64 %353, 0
  %355 = zext i1 %354 to i32
  %356 = and i32 %355, 1
  %357 = icmp ne i32 %356, 0
  br i1 %357, label %358, label %361

358:                                              ; preds = %349
  %359 = call i32 @strcmp(ptr %352, ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8))
  %360 = icmp eq i32 %359, 0
  br label %363

361:                                              ; preds = %349
  %362 = icmp eq i64 %353, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8) to i64)
  br label %363

363:                                              ; preds = %358, %361
  %364 = phi i1 [ %362, %361 ], [ %360, %358 ]
  br label %365

365:                                              ; preds = %363
  br i1 %364, label %366, label %371

366:                                              ; preds = %365
  %367 = load ptr, ptr %2, align 8
  %368 = getelementptr { i64, ptr, i16 }, ptr %367, i32 0, i32 2
  %369 = load i16, ptr %368, align 2
  %370 = zext i16 %369 to i64
  store i64 %370, ptr %3, align 8
  br label %461

371:                                              ; preds = %365
  %372 = load ptr, ptr %2, align 8
  %373 = getelementptr { i64, ptr, i8 }, ptr %372, i32 0, i32 1
  %374 = load ptr, ptr %373, align 8
  %375 = ptrtoint ptr %374 to i64
  %376 = icmp ne i64 %375, 0
  %377 = zext i1 %376 to i32
  %378 = and i32 %377, 1
  %379 = icmp ne i32 %378, 0
  br i1 %379, label %380, label %383

380:                                              ; preds = %371
  %381 = call i32 @strcmp(ptr %374, ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8))
  %382 = icmp eq i32 %381, 0
  br label %385

383:                                              ; preds = %371
  %384 = icmp eq i64 %375, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8) to i64)
  br label %385

385:                                              ; preds = %380, %383
  %386 = phi i1 [ %384, %383 ], [ %382, %380 ]
  br label %387

387:                                              ; preds = %385
  br i1 %386, label %388, label %393

388:                                              ; preds = %387
  %389 = load ptr, ptr %2, align 8
  %390 = getelementptr { i64, ptr, i16 }, ptr %389, i32 0, i32 2
  %391 = load i16, ptr %390, align 2
  %392 = zext i16 %391 to i64
  store i64 %392, ptr %3, align 8
  br label %461

393:                                              ; preds = %387
  %394 = load ptr, ptr %2, align 8
  %395 = getelementptr { i64, ptr, i8 }, ptr %394, i32 0, i32 1
  %396 = load ptr, ptr %395, align 8
  %397 = ptrtoint ptr %396 to i64
  %398 = icmp ne i64 %397, 0
  %399 = zext i1 %398 to i32
  %400 = and i32 %399, 1
  %401 = icmp ne i32 %400, 0
  br i1 %401, label %402, label %405

402:                                              ; preds = %393
  %403 = call i32 @strcmp(ptr %396, ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8))
  %404 = icmp eq i32 %403, 0
  br label %407

405:                                              ; preds = %393
  %406 = icmp eq i64 %397, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8) to i64)
  br label %407

407:                                              ; preds = %402, %405
  %408 = phi i1 [ %406, %405 ], [ %404, %402 ]
  br label %409

409:                                              ; preds = %407
  br i1 %408, label %410, label %415

410:                                              ; preds = %409
  %411 = load ptr, ptr %2, align 8
  %412 = getelementptr { i64, ptr, i8 }, ptr %411, i32 0, i32 2
  %413 = load i8, ptr %412, align 1
  %414 = zext i8 %413 to i64
  store i64 %414, ptr %3, align 8
  br label %461

415:                                              ; preds = %409
  %416 = load ptr, ptr %2, align 8
  %417 = getelementptr { i64, ptr, i8 }, ptr %416, i32 0, i32 1
  %418 = load ptr, ptr %417, align 8
  %419 = ptrtoint ptr %418 to i64
  %420 = icmp ne i64 %419, 0
  %421 = zext i1 %420 to i32
  %422 = and i32 %421, 1
  %423 = icmp ne i32 %422, 0
  br i1 %423, label %424, label %427

424:                                              ; preds = %415
  %425 = call i32 @strcmp(ptr %418, ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8))
  %426 = icmp eq i32 %425, 0
  br label %429

427:                                              ; preds = %415
  %428 = icmp eq i64 %419, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8) to i64)
  br label %429

429:                                              ; preds = %424, %427
  %430 = phi i1 [ %428, %427 ], [ %426, %424 ]
  br label %431

431:                                              ; preds = %429
  br i1 %430, label %432, label %437

432:                                              ; preds = %431
  %433 = load ptr, ptr %2, align 8
  %434 = getelementptr { i64, ptr, i8 }, ptr %433, i32 0, i32 2
  %435 = load i8, ptr %434, align 1
  %436 = zext i8 %435 to i64
  store i64 %436, ptr %3, align 8
  br label %461

437:                                              ; preds = %431
  %438 = load ptr, ptr %2, align 8
  %439 = getelementptr { i64, ptr, i8 }, ptr %438, i32 0, i32 1
  %440 = load ptr, ptr %439, align 8
  %441 = ptrtoint ptr %440 to i64
  %442 = icmp ne i64 %441, 0
  %443 = zext i1 %442 to i32
  %444 = and i32 %443, 1
  %445 = icmp ne i32 %444, 0
  br i1 %445, label %446, label %449

446:                                              ; preds = %437
  %447 = call i32 @strcmp(ptr %440, ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8))
  %448 = icmp eq i32 %447, 0
  br label %451

449:                                              ; preds = %437
  %450 = icmp eq i64 %441, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8) to i64)
  br label %451

451:                                              ; preds = %446, %449
  %452 = phi i1 [ %450, %449 ], [ %448, %446 ]
  br label %453

453:                                              ; preds = %451
  br i1 %452, label %454, label %459

454:                                              ; preds = %453
  %455 = load ptr, ptr %2, align 8
  %456 = getelementptr { i64, ptr, i8 }, ptr %455, i32 0, i32 2
  %457 = load i8, ptr %456, align 1
  %458 = zext i8 %457 to i64
  store i64 %458, ptr %3, align 8
  br label %461

459:                                              ; preds = %453
  %460 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %460, align 8
  call void @__cxa_throw(ptr %460, ptr @_ZTIPc, ptr null)
  br label %463

461:                                              ; preds = %20, %41, %63, %85, %107, %129, %151, %173, %195, %216, %238, %259, %280, %301, %323, %344, %366, %388, %410, %432, %454
  %462 = load i64, ptr %3, align 8
  ret i64 %462

463:                                              ; preds = %459
  unreachable
}

; Function Attrs: mustprogress
define i64 @untypedBigInt() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  store i64 0, ptr %3, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store i64 -5, ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIx, ptr null)
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %12
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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %21
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %30
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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %48
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %57
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
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %66
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc8, %.noexc9
  %99 = phi ptr [ %94, %.noexc9 ], [ %90, %.noexc8 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc7
  %109 = phi ptr [ %104, %.noexc7 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc12
  %119 = phi ptr [ %114, %.noexc12 ], [ %109, %110 ]
  br label %120

120:                                              ; preds = %118
  br label %121

121:                                              ; preds = %75, %120
  %122 = phi ptr [ %119, %120 ], [ %76, %75 ]
  br label %123

123:                                              ; preds = %121
  br label %124

124:                                              ; preds = %.noexc4, %123
  %125 = phi ptr [ %122, %123 ], [ %69, %.noexc4 ]
  br label %126

126:                                              ; preds = %124
  br label %127

127:                                              ; preds = %.noexc, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc1, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc1 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc3, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc3 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc11, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc11 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc2, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc2 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc6, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc6 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke i64 @"___unbox<!ts.bigint>"(ptr %145)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %144
  store i64 %146, ptr %3, align 8
  call void @__cxa_end_catch()
  %147 = load i64, ptr %3, align 8
  store i64 %147, ptr %4, align 8
  %148 = load i64, ptr %4, align 8
  ret i64 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %111, %30, %6, %91, %89, %101, %12, %144, %66, %39, %21, %48, %57
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define i32 @mismatchPassesOn() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i64, align 8
  %2 = alloca i1, align 1
  %3 = alloca i1, align 1
  %4 = alloca i64, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  %7 = call ptr @__cxa_allocate_exception(i64 8)
  store i64 7, ptr %7, align 8
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIx, ptr null)
          to label %84 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr @_ZTIb
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 1
  %11 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIb)
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %16, label %13

13:                                               ; preds = %8
  %14 = extractvalue { ptr, i32 } %9, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  invoke void @__cxa_rethrow()
          to label %84 unwind label %endcatch.pad

16:                                               ; preds = %8
  %17 = extractvalue { ptr, i32 } %9, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  %19 = load i1, ptr %18, align 1
  store i1 %19, ptr %3, align 1
  %20 = load i32, ptr %5, align 4
  %21 = add i32 %20, 1
  store i32 %21, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %44

endcatch.pad:                                     ; preds = %13
  %22 = landingpad { ptr, i32 }
          catch ptr @_ZTIx
  call void @__cxa_end_catch()
  br label %.caught

23:                                               ; No predecessors!
  %24 = landingpad { ptr, i32 }
          catch ptr @_ZTIx
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %23
  %25 = phi { ptr, i32 } [ %24, %23 ], [ %22, %endcatch.pad ]
  %26 = extractvalue { ptr, i32 } %25, 1
  %27 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIx)
  %28 = icmp eq i32 %26, %27
  br i1 %28, label %32, label %29

29:                                               ; preds = %.caught
  %30 = extractvalue { ptr, i32 } %25, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  invoke void @__cxa_rethrow()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %29
  br label %84

32:                                               ; preds = %.caught
  %33 = extractvalue { ptr, i32 } %25, 0
  %34 = call ptr @__cxa_begin_catch(ptr %33)
  %35 = load i64, ptr %34, align 8
  store i64 %35, ptr %4, align 8
  %36 = load i32, ptr %5, align 4
  %37 = load i64, ptr %4, align 8
  %38 = icmp eq i64 %37, 7
  br i1 %38, label %39, label %40

39:                                               ; preds = %32
  br label %41

40:                                               ; preds = %32
  br label %41

41:                                               ; preds = %39, %40
  %42 = phi i32 [ 100, %40 ], [ 10, %39 ]
  %43 = add i32 %36, %42
  store i32 %43, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %44

44:                                               ; preds = %16, %41
  %45 = call ptr @__cxa_allocate_exception(i64 1)
  store i1 true, ptr %45, align 1
  invoke void @__cxa_throw(ptr %45, ptr @_ZTIb, ptr null)
          to label %84 unwind label %46

46:                                               ; preds = %44
  %47 = landingpad { ptr, i32 }
          catch ptr @_ZTIx
          catch ptr null
  %48 = extractvalue { ptr, i32 } %47, 1
  %49 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIx)
  %50 = icmp eq i32 %48, %49
  br i1 %50, label %54, label %51

51:                                               ; preds = %46
  %52 = extractvalue { ptr, i32 } %47, 0
  %53 = call ptr @__cxa_begin_catch(ptr %52)
  invoke void @__cxa_rethrow()
          to label %84 unwind label %endcatch.pad2

54:                                               ; preds = %46
  %55 = extractvalue { ptr, i32 } %47, 0
  %56 = call ptr @__cxa_begin_catch(ptr %55)
  %57 = load i64, ptr %56, align 8
  store i64 %57, ptr %1, align 8
  %58 = load i32, ptr %5, align 4
  %59 = add i32 %58, 1000
  store i32 %59, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %81

endcatch.pad2:                                    ; preds = %51
  %60 = landingpad { ptr, i32 }
          catch ptr @_ZTIb
  call void @__cxa_end_catch()
  br label %.caught1

61:                                               ; No predecessors!
  %62 = landingpad { ptr, i32 }
          catch ptr @_ZTIb
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %61
  %63 = phi { ptr, i32 } [ %62, %61 ], [ %60, %endcatch.pad2 ]
  %64 = extractvalue { ptr, i32 } %63, 1
  %65 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIb)
  %66 = icmp eq i32 %64, %65
  br i1 %66, label %70, label %67

67:                                               ; preds = %.caught1
  %68 = extractvalue { ptr, i32 } %63, 0
  %69 = call ptr @__cxa_begin_catch(ptr %68)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %67
  br label %84

70:                                               ; preds = %.caught1
  %71 = extractvalue { ptr, i32 } %63, 0
  %72 = call ptr @__cxa_begin_catch(ptr %71)
  %73 = load i1, ptr %72, align 1
  store i1 %73, ptr %2, align 1
  %74 = load i32, ptr %5, align 4
  %75 = load i1, ptr %2, align 1
  br i1 %75, label %76, label %77

76:                                               ; preds = %70
  br label %78

77:                                               ; preds = %70
  br label %78

78:                                               ; preds = %76, %77
  %79 = phi i32 [ 200, %77 ], [ 20, %76 ]
  %80 = add i32 %74, %79
  store i32 %80, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %81

81:                                               ; preds = %54, %78
  %82 = load i32, ptr %5, align 4
  store i32 %82, ptr %6, align 4
  %83 = load i32, ptr %6, align 4
  ret i32 %83

84:                                               ; preds = %.noexc, %51, %44, %.noexc3, %13, %0
  unreachable

endcatch.resume:                                  ; preds = %29, %67
  %85 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %85
}

; Function Attrs: mustprogress
define i1 @bigIntWithoutExceptions() #3 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  %5 = alloca i1, align 1
  store i64 -123456789012, ptr %4, align 8
  %6 = load i64, ptr %4, align 8
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr { i64, ptr, i64 }, ptr %8, i32 0, i32 0
  store i64 8, ptr %9, align 8
  %10 = getelementptr { i64, ptr, i64 }, ptr %8, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %10, align 8
  %11 = getelementptr { i64, ptr, i64 }, ptr %8, i32 0, i32 2
  store i64 %6, ptr %11, align 8
  store ptr %8, ptr %3, align 8
  %12 = load i64, ptr %4, align 8
  %13 = call ptr @GC_malloc_atomic(i64 58)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %14, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_1189076927660576410, i64 8), i64 %12)
  %16 = icmp eq ptr %14, null
  %17 = select i1 %16, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %18 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_634733802172746332, i64 8))
  %19 = add i64 1, %18
  %20 = call i64 @strlen(ptr %17)
  %21 = add i64 %19, %20
  %22 = add i64 %21, 8
  %23 = call ptr @GC_malloc(i64 %22)
  %24 = getelementptr i8, ptr %23, i64 8
  %25 = getelementptr i8, ptr %24, i64 0
  %26 = sub i64 %21, 0
  %27 = call ptr @__memcpy_chk(ptr %25, ptr getelementptr inbounds nuw (i8, ptr @s_634733802172746332, i64 8), i64 %18, i64 %26)
  %28 = add i64 0, %18
  %29 = getelementptr i8, ptr %24, i64 %28
  %30 = sub i64 %21, %28
  %31 = call ptr @__memcpy_chk(ptr %29, ptr %17, i64 %20, i64 %30)
  %32 = add i64 %28, %20
  %33 = getelementptr i8, ptr %24, i64 %32
  store i8 0, ptr %33, align 1
  store ptr %24, ptr %2, align 8
  %34 = call ptr @GC_malloc(i64 32)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr { i64, ptr, ptr }, ptr %35, i32 0, i32 0
  store i64 8, ptr %36, align 8
  %37 = getelementptr { i64, ptr, ptr }, ptr %35, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %37, align 8
  %38 = getelementptr { i64, ptr, ptr }, ptr %35, i32 0, i32 2
  store ptr getelementptr inbounds nuw (i8, ptr @s_571532774284038691, i64 8), ptr %38, align 8
  store ptr %35, ptr %1, align 8
  %39 = load ptr, ptr %3, align 8
  %40 = getelementptr { i64, ptr, i8 }, ptr %39, i32 0, i32 1
  %41 = load ptr, ptr %40, align 8
  %42 = ptrtoint ptr %41 to i64
  %43 = icmp ne i64 %42, 0
  %44 = zext i1 %43 to i32
  %45 = and i32 %44, 1
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %50

47:                                               ; preds = %0
  %48 = call i32 @strcmp(ptr %41, ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8))
  %49 = icmp eq i32 %48, 0
  br label %52

50:                                               ; preds = %0
  %51 = icmp eq i64 %42, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1166204947305017450, i64 8) to i64)
  br label %52

52:                                               ; preds = %47, %50
  %53 = phi i1 [ %51, %50 ], [ %49, %47 ]
  br label %54

54:                                               ; preds = %52
  br i1 %53, label %55, label %61

55:                                               ; preds = %54
  %56 = load ptr, ptr %3, align 8
  %57 = getelementptr { i64, ptr, i64 }, ptr %56, i32 0, i32 2
  %58 = load i64, ptr %57, align 8
  %59 = load i64, ptr %4, align 8
  %60 = icmp eq i64 %58, %59
  br label %61

61:                                               ; preds = %55, %54
  %62 = phi i1 [ %60, %55 ], [ %53, %54 ]
  br i1 %62, label %63, label %78

63:                                               ; preds = %61
  %64 = load ptr, ptr %2, align 8
  %65 = ptrtoint ptr %64 to i64
  %66 = icmp ne i64 %65, 0
  %67 = zext i1 %66 to i32
  %68 = and i32 %67, 1
  %69 = icmp ne i32 %68, 0
  br i1 %69, label %70, label %73

70:                                               ; preds = %63
  %71 = call i32 @strcmp(ptr %64, ptr getelementptr inbounds nuw (i8, ptr @s_9186415445472728809, i64 8))
  %72 = icmp eq i32 %71, 0
  br label %75

73:                                               ; preds = %63
  %74 = icmp eq i64 %65, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9186415445472728809, i64 8) to i64)
  br label %75

75:                                               ; preds = %70, %73
  %76 = phi i1 [ %74, %73 ], [ %72, %70 ]
  br label %77

77:                                               ; preds = %75
  br label %78

78:                                               ; preds = %77, %61
  %79 = phi i1 [ %76, %77 ], [ %62, %61 ]
  br i1 %79, label %80, label %84

80:                                               ; preds = %78
  %81 = load ptr, ptr %1, align 8
  %82 = call i64 @"___unbox<!ts.bigint>"(ptr %81)
  %83 = icmp eq i64 %82, 42
  br label %84

84:                                               ; preds = %80, %78
  %85 = phi i1 [ %83, %80 ], [ %79, %78 ]
  store i1 %85, ptr %5, align 1
  %86 = load i1, ptr %5, align 1
  ret i1 %86
}

; Function Attrs: mustprogress
define i1 @bigIntComparisons() #3 {
  %1 = alloca i64, align 8
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  %4 = alloca i1, align 1
  store i64 -123456789012, ptr %3, align 8
  store i64 -123456789012, ptr %2, align 8
  store i64 7, ptr %1, align 8
  %5 = load i64, ptr %3, align 8
  %6 = load i64, ptr %2, align 8
  %7 = icmp eq i64 %5, %6
  br i1 %7, label %8, label %12

8:                                                ; preds = %0
  %9 = load i64, ptr %3, align 8
  %10 = load i64, ptr %2, align 8
  %11 = icmp eq i64 %9, %10
  br label %12

12:                                               ; preds = %8, %0
  %13 = phi i1 [ %11, %8 ], [ %7, %0 ]
  br i1 %13, label %14, label %19

14:                                               ; preds = %12
  %15 = load i64, ptr %3, align 8
  %16 = load i64, ptr %2, align 8
  %17 = icmp ne i64 %15, %16
  %18 = xor i1 %17, true
  br label %19

19:                                               ; preds = %14, %12
  %20 = phi i1 [ %18, %14 ], [ %13, %12 ]
  br i1 %20, label %21, label %25

21:                                               ; preds = %19
  %22 = load i64, ptr %3, align 8
  %23 = load i64, ptr %1, align 8
  %24 = icmp slt i64 %22, %23
  br label %25

25:                                               ; preds = %21, %19
  %26 = phi i1 [ %24, %21 ], [ %20, %19 ]
  br i1 %26, label %27, label %31

27:                                               ; preds = %25
  %28 = load i64, ptr %1, align 8
  %29 = load i64, ptr %3, align 8
  %30 = icmp sgt i64 %28, %29
  br label %31

31:                                               ; preds = %27, %25
  %32 = phi i1 [ %30, %27 ], [ %26, %25 ]
  br i1 %32, label %33, label %37

33:                                               ; preds = %31
  %34 = load i64, ptr %3, align 8
  %35 = load i64, ptr %2, align 8
  %36 = icmp sle i64 %34, %35
  br label %37

37:                                               ; preds = %33, %31
  %38 = phi i1 [ %36, %33 ], [ %32, %31 ]
  br i1 %38, label %39, label %43

39:                                               ; preds = %37
  %40 = load i64, ptr %1, align 8
  %41 = load i64, ptr %3, align 8
  %42 = icmp sge i64 %40, %41
  br label %43

43:                                               ; preds = %39, %37
  %44 = phi i1 [ %42, %39 ], [ %38, %37 ]
  br i1 %44, label %45, label %50

45:                                               ; preds = %43
  %46 = load i64, ptr %1, align 8
  %47 = load i64, ptr %3, align 8
  %48 = icmp slt i64 %46, %47
  %49 = xor i1 %48, true
  br label %50

50:                                               ; preds = %45, %43
  %51 = phi i1 [ %49, %45 ], [ %44, %43 ]
  store i1 %51, ptr %4, align 1
  %52 = load i1, ptr %4, align 1
  ret i1 %52
}

; Function Attrs: mustprogress
define i32 @main() #3 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call i1 @typedBool()
  br i1 %1, label %2, label %22

2:                                                ; preds = %0
  %3 = call i1 @untypedBool()
  %4 = xor i1 %3, true
  br i1 %4, label %5, label %24

5:                                                ; preds = %2
  %6 = call i1 @boolVariable()
  br i1 %6, label %7, label %26

7:                                                ; preds = %5
  %8 = call i64 @typedBigInt()
  %9 = icmp eq i64 %8, 123456789012
  br i1 %9, label %10, label %28

10:                                               ; preds = %7
  %11 = call i64 @untypedBigInt()
  %12 = icmp eq i64 %11, -5
  br i1 %12, label %13, label %30

13:                                               ; preds = %10
  %14 = call i32 @mismatchPassesOn()
  %15 = icmp eq i32 %14, 30
  br i1 %15, label %16, label %32

16:                                               ; preds = %13
  %17 = call i1 @bigIntWithoutExceptions()
  br i1 %17, label %18, label %34

18:                                               ; preds = %16
  %19 = call i1 @bigIntComparisons()
  br i1 %19, label %20, label %36

20:                                               ; preds = %18
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

21:                                               ; preds = %36, %34, %32, %30, %28, %26, %24, %22
  unreachable

22:                                               ; preds = %0
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10471070127234361712, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 115, ptr null)
  br label %21

24:                                               ; preds = %2
  %25 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6943169351122162168, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 116, ptr null)
  br label %21

26:                                               ; preds = %5
  %27 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2677683196513557903, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 117, ptr null)
  br label %21

28:                                               ; preds = %7
  %29 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8637983907680937160, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 118, ptr null)
  br label %21

30:                                               ; preds = %10
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5702300165540205539, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 119, ptr null)
  br label %21

32:                                               ; preds = %13
  %33 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_43741921607073719, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 120, ptr null)
  br label %21

34:                                               ; preds = %16
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14407052258144518178, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 121, ptr null)
  br label %21

36:                                               ; preds = %18
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1607514885731154834, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7484941213777721143, i64 8), i32 122, ptr null)
  br label %21
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
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress "personality" }
attributes #3 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

