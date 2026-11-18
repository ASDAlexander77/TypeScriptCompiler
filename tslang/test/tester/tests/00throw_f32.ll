; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$"___unbox<f32>" = comdat any

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
@s_16822532743728177535 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboolean\00", align 8
@s_8091808378024623192 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring\00", align 8
@s_9237349086447201248 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnumber\00", align 8
@s_15952682373371409732 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf32\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_4304588356623647936 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: f32) catches a thrown f32\00", align 8
@f_17003361720708021055 = internal constant [101 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_f32.ts\00", align 8
@m_15125558611850858047 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown f32\00", align 8
@m_3429671353266982435 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: number) widens a thrown f32\00", align 8
@m_2124737424241911831 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa thrown number passes an f32 catch\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIf = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIb = external constant ptr
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

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

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

declare double @strtod(ptr, ptr)

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
define float @typedF32() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca float, align 4
  %2 = alloca float, align 4
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float 1.500000e+00, ptr %3, align 4
  store float 0.000000e+00, ptr %2, align 4
  %5 = load float, ptr %3, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store float %5, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIf, ptr null)
          to label %22 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIf
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
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
  %18 = load float, ptr %17, align 4
  store float %18, ptr %1, align 4
  %19 = load float, ptr %1, align 4
  store float %19, ptr %2, align 4
  call void @__cxa_end_catch()
  %20 = load float, ptr %2, align 4
  store float %20, ptr %4, align 4
  %21 = load float, ptr %4, align 4
  ret float %21

22:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %12
  %23 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %23
}

; Function Attrs: mustprogress
define linkonce_odr float @"___unbox<f32>"(ptr %0) #3 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca float, align 4
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
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %24

20:                                               ; preds = %19
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr { i64, ptr, float }, ptr %21, i32 0, i32 2
  %23 = load float, ptr %22, align 4
  store float %23, ptr %3, align 4
  br label %466

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
  %45 = fptrunc double %44 to float
  store float %45, ptr %3, align 4
  br label %466

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
  br i1 %61, label %63, label %69

63:                                               ; preds = %62
  %64 = load ptr, ptr %2, align 8
  %65 = getelementptr { i64, ptr, ptr }, ptr %64, i32 0, i32 2
  %66 = load ptr, ptr %65, align 8
  %67 = call double @strtod(ptr %66, ptr null)
  %68 = fptrunc double %67 to float
  store float %68, ptr %3, align 4
  br label %466

69:                                               ; preds = %62
  %70 = load ptr, ptr %2, align 8
  %71 = getelementptr { i64, ptr, i8 }, ptr %70, i32 0, i32 1
  %72 = load ptr, ptr %71, align 8
  %73 = ptrtoint ptr %72 to i64
  %74 = icmp ne i64 %73, 0
  %75 = zext i1 %74 to i32
  %76 = and i32 %75, 1
  %77 = icmp ne i32 %76, 0
  br i1 %77, label %78, label %81

78:                                               ; preds = %69
  %79 = call i32 @strcmp(ptr %72, ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8))
  %80 = icmp eq i32 %79, 0
  br label %83

81:                                               ; preds = %69
  %82 = icmp eq i64 %73, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16822532743728177535, i64 8) to i64)
  br label %83

83:                                               ; preds = %78, %81
  %84 = phi i1 [ %82, %81 ], [ %80, %78 ]
  br label %85

85:                                               ; preds = %83
  br i1 %84, label %86, label %91

86:                                               ; preds = %85
  %87 = load ptr, ptr %2, align 8
  %88 = getelementptr { i64, ptr, i1 }, ptr %87, i32 0, i32 2
  %89 = load i1, ptr %88, align 1
  %90 = uitofp i1 %89 to float
  store float %90, ptr %3, align 4
  br label %466

91:                                               ; preds = %85
  %92 = load ptr, ptr %2, align 8
  %93 = getelementptr { i64, ptr, i8 }, ptr %92, i32 0, i32 1
  %94 = load ptr, ptr %93, align 8
  %95 = ptrtoint ptr %94 to i64
  %96 = icmp ne i64 %95, 0
  %97 = zext i1 %96 to i32
  %98 = and i32 %97, 1
  %99 = icmp ne i32 %98, 0
  br i1 %99, label %100, label %103

100:                                              ; preds = %91
  %101 = call i32 @strcmp(ptr %94, ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8))
  %102 = icmp eq i32 %101, 0
  br label %105

103:                                              ; preds = %91
  %104 = icmp eq i64 %95, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15952682373371409732, i64 8) to i64)
  br label %105

105:                                              ; preds = %100, %103
  %106 = phi i1 [ %104, %103 ], [ %102, %100 ]
  br label %107

107:                                              ; preds = %105
  br i1 %106, label %108, label %112

108:                                              ; preds = %107
  %109 = load ptr, ptr %2, align 8
  %110 = getelementptr { i64, ptr, float }, ptr %109, i32 0, i32 2
  %111 = load float, ptr %110, align 4
  store float %111, ptr %3, align 4
  br label %466

112:                                              ; preds = %107
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
  %133 = sitofp i32 %132 to float
  store float %133, ptr %3, align 4
  br label %466

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
  %155 = sitofp i32 %154 to float
  store float %155, ptr %3, align 4
  br label %466

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
  %177 = uitofp i32 %176 to float
  store float %177, ptr %3, align 4
  br label %466

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
  br i1 %193, label %195, label %200

195:                                              ; preds = %194
  %196 = load ptr, ptr %2, align 8
  %197 = getelementptr { i64, ptr, i64 }, ptr %196, i32 0, i32 2
  %198 = load i64, ptr %197, align 8
  %199 = sitofp i64 %198 to float
  store float %199, ptr %3, align 4
  br label %466

200:                                              ; preds = %194
  %201 = load ptr, ptr %2, align 8
  %202 = getelementptr { i64, ptr, i8 }, ptr %201, i32 0, i32 1
  %203 = load ptr, ptr %202, align 8
  %204 = ptrtoint ptr %203 to i64
  %205 = icmp ne i64 %204, 0
  %206 = zext i1 %205 to i32
  %207 = and i32 %206, 1
  %208 = icmp ne i32 %207, 0
  br i1 %208, label %209, label %212

209:                                              ; preds = %200
  %210 = call i32 @strcmp(ptr %203, ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8))
  %211 = icmp eq i32 %210, 0
  br label %214

212:                                              ; preds = %200
  %213 = icmp eq i64 %204, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15957463049929925035, i64 8) to i64)
  br label %214

214:                                              ; preds = %209, %212
  %215 = phi i1 [ %213, %212 ], [ %211, %209 ]
  br label %216

216:                                              ; preds = %214
  br i1 %215, label %217, label %222

217:                                              ; preds = %216
  %218 = load ptr, ptr %2, align 8
  %219 = getelementptr { i64, ptr, double }, ptr %218, i32 0, i32 2
  %220 = load double, ptr %219, align 8
  %221 = fptrunc double %220 to float
  store float %221, ptr %3, align 4
  br label %466

222:                                              ; preds = %216
  %223 = load ptr, ptr %2, align 8
  %224 = getelementptr { i64, ptr, i8 }, ptr %223, i32 0, i32 1
  %225 = load ptr, ptr %224, align 8
  %226 = ptrtoint ptr %225 to i64
  %227 = icmp ne i64 %226, 0
  %228 = zext i1 %227 to i32
  %229 = and i32 %228, 1
  %230 = icmp ne i32 %229, 0
  br i1 %230, label %231, label %234

231:                                              ; preds = %222
  %232 = call i32 @strcmp(ptr %225, ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8))
  %233 = icmp eq i32 %232, 0
  br label %236

234:                                              ; preds = %222
  %235 = icmp eq i64 %226, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3089943341986682694, i64 8) to i64)
  br label %236

236:                                              ; preds = %231, %234
  %237 = phi i1 [ %235, %234 ], [ %233, %231 ]
  br label %238

238:                                              ; preds = %236
  br i1 %237, label %239, label %244

239:                                              ; preds = %238
  %240 = load ptr, ptr %2, align 8
  %241 = getelementptr { i64, ptr, i64 }, ptr %240, i32 0, i32 2
  %242 = load i64, ptr %241, align 8
  %243 = sitofp i64 %242 to float
  store float %243, ptr %3, align 4
  br label %466

244:                                              ; preds = %238
  %245 = load ptr, ptr %2, align 8
  %246 = getelementptr { i64, ptr, i8 }, ptr %245, i32 0, i32 1
  %247 = load ptr, ptr %246, align 8
  %248 = ptrtoint ptr %247 to i64
  %249 = icmp ne i64 %248, 0
  %250 = zext i1 %249 to i32
  %251 = and i32 %250, 1
  %252 = icmp ne i32 %251, 0
  br i1 %252, label %253, label %256

253:                                              ; preds = %244
  %254 = call i32 @strcmp(ptr %247, ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8))
  %255 = icmp eq i32 %254, 0
  br label %258

256:                                              ; preds = %244
  %257 = icmp eq i64 %248, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9335351907508431320, i64 8) to i64)
  br label %258

258:                                              ; preds = %253, %256
  %259 = phi i1 [ %257, %256 ], [ %255, %253 ]
  br label %260

260:                                              ; preds = %258
  br i1 %259, label %261, label %266

261:                                              ; preds = %260
  %262 = load ptr, ptr %2, align 8
  %263 = getelementptr { i64, ptr, i64 }, ptr %262, i32 0, i32 2
  %264 = load i64, ptr %263, align 8
  %265 = sitofp i64 %264 to float
  store float %265, ptr %3, align 4
  br label %466

266:                                              ; preds = %260
  %267 = load ptr, ptr %2, align 8
  %268 = getelementptr { i64, ptr, i8 }, ptr %267, i32 0, i32 1
  %269 = load ptr, ptr %268, align 8
  %270 = ptrtoint ptr %269 to i64
  %271 = icmp ne i64 %270, 0
  %272 = zext i1 %271 to i32
  %273 = and i32 %272, 1
  %274 = icmp ne i32 %273, 0
  br i1 %274, label %275, label %278

275:                                              ; preds = %266
  %276 = call i32 @strcmp(ptr %269, ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8))
  %277 = icmp eq i32 %276, 0
  br label %280

278:                                              ; preds = %266
  %279 = icmp eq i64 %270, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5563585020063213298, i64 8) to i64)
  br label %280

280:                                              ; preds = %275, %278
  %281 = phi i1 [ %279, %278 ], [ %277, %275 ]
  br label %282

282:                                              ; preds = %280
  br i1 %281, label %283, label %288

283:                                              ; preds = %282
  %284 = load ptr, ptr %2, align 8
  %285 = getelementptr { i64, ptr, i64 }, ptr %284, i32 0, i32 2
  %286 = load i64, ptr %285, align 8
  %287 = uitofp i64 %286 to float
  store float %287, ptr %3, align 4
  br label %466

288:                                              ; preds = %282
  %289 = load ptr, ptr %2, align 8
  %290 = getelementptr { i64, ptr, i8 }, ptr %289, i32 0, i32 1
  %291 = load ptr, ptr %290, align 8
  %292 = ptrtoint ptr %291 to i64
  %293 = icmp ne i64 %292, 0
  %294 = zext i1 %293 to i32
  %295 = and i32 %294, 1
  %296 = icmp ne i32 %295, 0
  br i1 %296, label %297, label %300

297:                                              ; preds = %288
  %298 = call i32 @strcmp(ptr %291, ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8))
  %299 = icmp eq i32 %298, 0
  br label %302

300:                                              ; preds = %288
  %301 = icmp eq i64 %292, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17483980429552467645, i64 8) to i64)
  br label %302

302:                                              ; preds = %297, %300
  %303 = phi i1 [ %301, %300 ], [ %299, %297 ]
  br label %304

304:                                              ; preds = %302
  br i1 %303, label %305, label %310

305:                                              ; preds = %304
  %306 = load ptr, ptr %2, align 8
  %307 = getelementptr { i64, ptr, i8 }, ptr %306, i32 0, i32 2
  %308 = load i8, ptr %307, align 1
  %309 = sitofp i8 %308 to float
  store float %309, ptr %3, align 4
  br label %466

310:                                              ; preds = %304
  %311 = load ptr, ptr %2, align 8
  %312 = getelementptr { i64, ptr, i8 }, ptr %311, i32 0, i32 1
  %313 = load ptr, ptr %312, align 8
  %314 = ptrtoint ptr %313 to i64
  %315 = icmp ne i64 %314, 0
  %316 = zext i1 %315 to i32
  %317 = and i32 %316, 1
  %318 = icmp ne i32 %317, 0
  br i1 %318, label %319, label %322

319:                                              ; preds = %310
  %320 = call i32 @strcmp(ptr %313, ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8))
  %321 = icmp eq i32 %320, 0
  br label %324

322:                                              ; preds = %310
  %323 = icmp eq i64 %314, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9497966886403524235, i64 8) to i64)
  br label %324

324:                                              ; preds = %319, %322
  %325 = phi i1 [ %323, %322 ], [ %321, %319 ]
  br label %326

326:                                              ; preds = %324
  br i1 %325, label %327, label %332

327:                                              ; preds = %326
  %328 = load ptr, ptr %2, align 8
  %329 = getelementptr { i64, ptr, i64 }, ptr %328, i32 0, i32 2
  %330 = load i64, ptr %329, align 8
  %331 = sitofp i64 %330 to float
  store float %331, ptr %3, align 4
  br label %466

332:                                              ; preds = %326
  %333 = load ptr, ptr %2, align 8
  %334 = getelementptr { i64, ptr, i8 }, ptr %333, i32 0, i32 1
  %335 = load ptr, ptr %334, align 8
  %336 = ptrtoint ptr %335 to i64
  %337 = icmp ne i64 %336, 0
  %338 = zext i1 %337 to i32
  %339 = and i32 %338, 1
  %340 = icmp ne i32 %339, 0
  br i1 %340, label %341, label %344

341:                                              ; preds = %332
  %342 = call i32 @strcmp(ptr %335, ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8))
  %343 = icmp eq i32 %342, 0
  br label %346

344:                                              ; preds = %332
  %345 = icmp eq i64 %336, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_3092775683940386555, i64 8) to i64)
  br label %346

346:                                              ; preds = %341, %344
  %347 = phi i1 [ %345, %344 ], [ %343, %341 ]
  br label %348

348:                                              ; preds = %346
  br i1 %347, label %349, label %354

349:                                              ; preds = %348
  %350 = load ptr, ptr %2, align 8
  %351 = getelementptr { i64, ptr, i16 }, ptr %350, i32 0, i32 2
  %352 = load i16, ptr %351, align 2
  %353 = sitofp i16 %352 to float
  store float %353, ptr %3, align 4
  br label %466

354:                                              ; preds = %348
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
  %364 = call i32 @strcmp(ptr %357, ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8))
  %365 = icmp eq i32 %364, 0
  br label %368

366:                                              ; preds = %354
  %367 = icmp eq i64 %358, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9334511880624667341, i64 8) to i64)
  br label %368

368:                                              ; preds = %363, %366
  %369 = phi i1 [ %367, %366 ], [ %365, %363 ]
  br label %370

370:                                              ; preds = %368
  br i1 %369, label %371, label %376

371:                                              ; preds = %370
  %372 = load ptr, ptr %2, align 8
  %373 = getelementptr { i64, ptr, i16 }, ptr %372, i32 0, i32 2
  %374 = load i16, ptr %373, align 2
  %375 = sitofp i16 %374 to float
  store float %375, ptr %3, align 4
  br label %466

376:                                              ; preds = %370
  %377 = load ptr, ptr %2, align 8
  %378 = getelementptr { i64, ptr, i8 }, ptr %377, i32 0, i32 1
  %379 = load ptr, ptr %378, align 8
  %380 = ptrtoint ptr %379 to i64
  %381 = icmp ne i64 %380, 0
  %382 = zext i1 %381 to i32
  %383 = and i32 %382, 1
  %384 = icmp ne i32 %383, 0
  br i1 %384, label %385, label %388

385:                                              ; preds = %376
  %386 = call i32 @strcmp(ptr %379, ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8))
  %387 = icmp eq i32 %386, 0
  br label %390

388:                                              ; preds = %376
  %389 = icmp eq i64 %380, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5558799945458185151, i64 8) to i64)
  br label %390

390:                                              ; preds = %385, %388
  %391 = phi i1 [ %389, %388 ], [ %387, %385 ]
  br label %392

392:                                              ; preds = %390
  br i1 %391, label %393, label %398

393:                                              ; preds = %392
  %394 = load ptr, ptr %2, align 8
  %395 = getelementptr { i64, ptr, i16 }, ptr %394, i32 0, i32 2
  %396 = load i16, ptr %395, align 2
  %397 = uitofp i16 %396 to float
  store float %397, ptr %3, align 4
  br label %466

398:                                              ; preds = %392
  %399 = load ptr, ptr %2, align 8
  %400 = getelementptr { i64, ptr, i8 }, ptr %399, i32 0, i32 1
  %401 = load ptr, ptr %400, align 8
  %402 = ptrtoint ptr %401 to i64
  %403 = icmp ne i64 %402, 0
  %404 = zext i1 %403 to i32
  %405 = and i32 %404, 1
  %406 = icmp ne i32 %405, 0
  br i1 %406, label %407, label %410

407:                                              ; preds = %398
  %408 = call i32 @strcmp(ptr %401, ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8))
  %409 = icmp eq i32 %408, 0
  br label %412

410:                                              ; preds = %398
  %411 = icmp eq i64 %402, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_627981701265227156, i64 8) to i64)
  br label %412

412:                                              ; preds = %407, %410
  %413 = phi i1 [ %411, %410 ], [ %409, %407 ]
  br label %414

414:                                              ; preds = %412
  br i1 %413, label %415, label %420

415:                                              ; preds = %414
  %416 = load ptr, ptr %2, align 8
  %417 = getelementptr { i64, ptr, i8 }, ptr %416, i32 0, i32 2
  %418 = load i8, ptr %417, align 1
  %419 = sitofp i8 %418 to float
  store float %419, ptr %3, align 4
  br label %466

420:                                              ; preds = %414
  %421 = load ptr, ptr %2, align 8
  %422 = getelementptr { i64, ptr, i8 }, ptr %421, i32 0, i32 1
  %423 = load ptr, ptr %422, align 8
  %424 = ptrtoint ptr %423 to i64
  %425 = icmp ne i64 %424, 0
  %426 = zext i1 %425 to i32
  %427 = and i32 %426, 1
  %428 = icmp ne i32 %427, 0
  br i1 %428, label %429, label %432

429:                                              ; preds = %420
  %430 = call i32 @strcmp(ptr %423, ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8))
  %431 = icmp eq i32 %430, 0
  br label %434

432:                                              ; preds = %420
  %433 = icmp eq i64 %424, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_637547452428770606, i64 8) to i64)
  br label %434

434:                                              ; preds = %429, %432
  %435 = phi i1 [ %433, %432 ], [ %431, %429 ]
  br label %436

436:                                              ; preds = %434
  br i1 %435, label %437, label %442

437:                                              ; preds = %436
  %438 = load ptr, ptr %2, align 8
  %439 = getelementptr { i64, ptr, i8 }, ptr %438, i32 0, i32 2
  %440 = load i8, ptr %439, align 1
  %441 = sitofp i8 %440 to float
  store float %441, ptr %3, align 4
  br label %466

442:                                              ; preds = %436
  %443 = load ptr, ptr %2, align 8
  %444 = getelementptr { i64, ptr, i8 }, ptr %443, i32 0, i32 1
  %445 = load ptr, ptr %444, align 8
  %446 = ptrtoint ptr %445 to i64
  %447 = icmp ne i64 %446, 0
  %448 = zext i1 %447 to i32
  %449 = and i32 %448, 1
  %450 = icmp ne i32 %449, 0
  br i1 %450, label %451, label %454

451:                                              ; preds = %442
  %452 = call i32 @strcmp(ptr %445, ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8))
  %453 = icmp eq i32 %452, 0
  br label %456

454:                                              ; preds = %442
  %455 = icmp eq i64 %446, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_631772817358541784, i64 8) to i64)
  br label %456

456:                                              ; preds = %451, %454
  %457 = phi i1 [ %455, %454 ], [ %453, %451 ]
  br label %458

458:                                              ; preds = %456
  br i1 %457, label %459, label %464

459:                                              ; preds = %458
  %460 = load ptr, ptr %2, align 8
  %461 = getelementptr { i64, ptr, i8 }, ptr %460, i32 0, i32 2
  %462 = load i8, ptr %461, align 1
  %463 = uitofp i8 %462 to float
  store float %463, ptr %3, align 4
  br label %466

464:                                              ; preds = %458
  %465 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %465, align 8
  call void @__cxa_throw(ptr %465, ptr @_ZTIPc, ptr null)
  br label %468

466:                                              ; preds = %20, %41, %63, %86, %108, %129, %151, %173, %195, %217, %239, %261, %283, %305, %327, %349, %371, %393, %415, %437, %459
  %467 = load float, ptr %3, align 4
  ret float %467

468:                                              ; preds = %464
  unreachable
}

; Function Attrs: mustprogress
define float @untypedF32() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  %5 = alloca float, align 4
  store float 2.500000e+00, ptr %4, align 4
  store float 0.000000e+00, ptr %3, align 4
  %6 = load float, ptr %4, align 4
  %7 = call ptr @__cxa_allocate_exception(i64 4)
  store float %6, ptr %7, align 4
  invoke void @__cxa_throw(ptr %7, ptr @_ZTIf, ptr null)
          to label %151 unwind label %8

8:                                                ; preds = %0
  %9 = landingpad { ptr, i32 }
          catch ptr null
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  %12 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %8
  %13 = icmp eq ptr %12, @_ZTIi
  br i1 %13, label %14, label %21

14:                                               ; preds = %.noexc5
  %15 = load i32, ptr %11, align 4
  %16 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %14
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 0
  store i64 4, ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %19, align 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %17, i32 0, i32 2
  store i32 %15, ptr %20, align 4
  br label %144

21:                                               ; preds = %.noexc5
  %22 = icmp eq ptr %12, @_ZTIj
  br i1 %22, label %23, label %30

23:                                               ; preds = %21
  %24 = load i32, ptr %11, align 4
  %25 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %23
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %32
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
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %41
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %50
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %59
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %68
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
  invoke void %89(ptr %1, ptr %11)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %91
  %92 = load ptr, ptr %1, align 8
  br label %100

93:                                               ; preds = %87
  %94 = load ptr, ptr %11, align 8
  %95 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %93
  %96 = getelementptr i8, ptr %95, i64 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 0
  store i64 8, ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %98, align 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %96, i32 0, i32 2
  store ptr %94, ptr %99, align 8
  br label %100

100:                                              ; preds = %.noexc1, %.noexc4
  %101 = phi ptr [ %96, %.noexc4 ], [ %92, %.noexc1 ]
  br label %102

102:                                              ; preds = %100
  br label %110

103:                                              ; preds = %82
  %104 = load ptr, ptr %11, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %102, %.noexc3
  %111 = phi ptr [ %106, %.noexc3 ], [ %101, %102 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %79
  %114 = load ptr, ptr %11, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc12
  %121 = phi ptr [ %116, %.noexc12 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %123

123:                                              ; preds = %77, %122
  %124 = phi ptr [ %121, %122 ], [ %78, %77 ]
  br label %125

125:                                              ; preds = %123
  br label %126

126:                                              ; preds = %.noexc, %125
  %127 = phi ptr [ %124, %125 ], [ %71, %.noexc ]
  br label %128

128:                                              ; preds = %126
  br label %129

129:                                              ; preds = %.noexc8, %128
  %130 = phi ptr [ %127, %128 ], [ %62, %.noexc8 ]
  br label %131

131:                                              ; preds = %129
  br label %132

132:                                              ; preds = %.noexc9, %131
  %133 = phi ptr [ %130, %131 ], [ %53, %.noexc9 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc6, %134
  %136 = phi ptr [ %133, %134 ], [ %44, %.noexc6 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc10, %137
  %139 = phi ptr [ %136, %137 ], [ %35, %.noexc10 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc11, %140
  %142 = phi ptr [ %139, %140 ], [ %26, %.noexc11 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc7, %143
  %145 = phi ptr [ %142, %143 ], [ %17, %.noexc7 ]
  br label %146

146:                                              ; preds = %144
  store ptr %145, ptr %2, align 8
  %147 = load ptr, ptr %2, align 8
  %148 = invoke float @"___unbox<f32>"(ptr %147)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %146
  store float %148, ptr %3, align 4
  call void @__cxa_end_catch()
  %149 = load float, ptr %3, align 4
  store float %149, ptr %5, align 4
  %150 = load float, ptr %5, align 4
  ret float %150

151:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %113, %23, %32, %50, %59, %14, %41, %8, %93, %103, %146, %91, %68
  %152 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %152
}

; Function Attrs: mustprogress
define double @f32AsNumber() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca float, align 4
  %4 = alloca double, align 8
  store float 1.500000e+00, ptr %3, align 4
  store double 0.000000e+00, ptr %2, align 8
  %5 = load float, ptr %3, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store float %5, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIf, ptr null)
          to label %59 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %11 = icmp eq i32 %9, %10
  %12 = extractvalue { ptr, i32 } %8, 1
  %13 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %14 = icmp eq i32 %12, %13
  %15 = extractvalue { ptr, i32 } %8, 1
  %16 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %17 = icmp eq i32 %15, %16
  %18 = extractvalue { ptr, i32 } %8, 1
  %19 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %20 = icmp eq i32 %18, %19
  %21 = or i1 %11, %14
  %22 = or i1 %21, %17
  %23 = or i1 %22, %20
  br i1 %23, label %27, label %24

24:                                               ; preds = %7
  %25 = extractvalue { ptr, i32 } %8, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %24
  br label %59

27:                                               ; preds = %7
  %28 = extractvalue { ptr, i32 } %8, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  %30 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %27
  %31 = icmp eq ptr %30, @_ZTIi
  br i1 %31, label %32, label %35

32:                                               ; preds = %.noexc
  %33 = load i32, ptr %29, align 4
  %34 = sitofp i32 %33 to double
  br label %53

35:                                               ; preds = %.noexc
  %36 = icmp eq ptr %30, @_ZTIj
  br i1 %36, label %37, label %40

37:                                               ; preds = %35
  %38 = load i32, ptr %29, align 4
  %39 = uitofp i32 %38 to double
  br label %50

40:                                               ; preds = %35
  %41 = icmp eq ptr %30, @_ZTIf
  br i1 %41, label %42, label %45

42:                                               ; preds = %40
  %43 = load float, ptr %29, align 4
  %44 = fpext float %43 to double
  br label %47

45:                                               ; preds = %40
  %46 = load double, ptr %29, align 8
  br label %47

47:                                               ; preds = %42, %45
  %48 = phi double [ %46, %45 ], [ %44, %42 ]
  br label %49

49:                                               ; preds = %47
  br label %50

50:                                               ; preds = %37, %49
  %51 = phi double [ %48, %49 ], [ %39, %37 ]
  br label %52

52:                                               ; preds = %50
  br label %53

53:                                               ; preds = %32, %52
  %54 = phi double [ %51, %52 ], [ %34, %32 ]
  br label %55

55:                                               ; preds = %53
  store double %54, ptr %1, align 8
  %56 = load double, ptr %1, align 8
  store double %56, ptr %2, align 8
  call void @__cxa_end_catch()
  %57 = load double, ptr %2, align 8
  store double %57, ptr %4, align 8
  %58 = load double, ptr %4, align 8
  ret double %58

59:                                               ; preds = %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %24, %27
  %60 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %60
}

; Function Attrs: mustprogress
define double @numberPassesF32Catch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca float, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double 0.000000e+00, ptr %3, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store double 2.250000e+00, ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTId, ptr null)
          to label %73 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIf
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %73 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load float, ptr %16, align 4
  store float %17, ptr %1, align 4
  store double -1.000000e+00, ptr %3, align 8
  call void @__cxa_end_catch()
  br label %70

endcatch.pad:                                     ; preds = %11
  %18 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  call void @__cxa_end_catch()
  br label %.caught

19:                                               ; No predecessors!
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %19
  %21 = phi { ptr, i32 } [ %20, %19 ], [ %18, %endcatch.pad ]
  %22 = extractvalue { ptr, i32 } %21, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %24 = icmp eq i32 %22, %23
  %25 = extractvalue { ptr, i32 } %21, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %27 = icmp eq i32 %25, %26
  %28 = extractvalue { ptr, i32 } %21, 1
  %29 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %30 = icmp eq i32 %28, %29
  %31 = extractvalue { ptr, i32 } %21, 1
  %32 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %33 = icmp eq i32 %31, %32
  %34 = or i1 %24, %27
  %35 = or i1 %34, %30
  %36 = or i1 %35, %33
  br i1 %36, label %40, label %37

37:                                               ; preds = %.caught
  %38 = extractvalue { ptr, i32 } %21, 0
  %39 = call ptr @__cxa_begin_catch(ptr %38)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %37
  br label %73

40:                                               ; preds = %.caught
  %41 = extractvalue { ptr, i32 } %21, 0
  %42 = call ptr @__cxa_begin_catch(ptr %41)
  %43 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %40
  %44 = icmp eq ptr %43, @_ZTIi
  br i1 %44, label %45, label %48

45:                                               ; preds = %.noexc1
  %46 = load i32, ptr %42, align 4
  %47 = sitofp i32 %46 to double
  br label %66

48:                                               ; preds = %.noexc1
  %49 = icmp eq ptr %43, @_ZTIj
  br i1 %49, label %50, label %53

50:                                               ; preds = %48
  %51 = load i32, ptr %42, align 4
  %52 = uitofp i32 %51 to double
  br label %63

53:                                               ; preds = %48
  %54 = icmp eq ptr %43, @_ZTIf
  br i1 %54, label %55, label %58

55:                                               ; preds = %53
  %56 = load float, ptr %42, align 4
  %57 = fpext float %56 to double
  br label %60

58:                                               ; preds = %53
  %59 = load double, ptr %42, align 8
  br label %60

60:                                               ; preds = %55, %58
  %61 = phi double [ %59, %58 ], [ %57, %55 ]
  br label %62

62:                                               ; preds = %60
  br label %63

63:                                               ; preds = %50, %62
  %64 = phi double [ %61, %62 ], [ %52, %50 ]
  br label %65

65:                                               ; preds = %63
  br label %66

66:                                               ; preds = %45, %65
  %67 = phi double [ %64, %65 ], [ %47, %45 ]
  br label %68

68:                                               ; preds = %66
  store double %67, ptr %2, align 8
  %69 = load double, ptr %2, align 8
  store double %69, ptr %3, align 8
  call void @__cxa_end_catch()
  br label %70

70:                                               ; preds = %14, %68
  %71 = load double, ptr %3, align 8
  store double %71, ptr %4, align 8
  %72 = load double, ptr %4, align 8
  ret double %72

73:                                               ; preds = %.noexc, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %40, %37
  %74 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %74
}

; Function Attrs: mustprogress
define i32 @main() #3 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call float @typedF32()
  %2 = fpext float %1 to double
  %3 = fcmp oeq double %2, 1.500000e+00
  br i1 %3, label %4, label %16

4:                                                ; preds = %0
  %5 = call float @untypedF32()
  %6 = fpext float %5 to double
  %7 = fcmp oeq double %6, 2.500000e+00
  br i1 %7, label %8, label %18

8:                                                ; preds = %4
  %9 = call double @f32AsNumber()
  %10 = fcmp oeq double %9, 1.500000e+00
  br i1 %10, label %11, label %20

11:                                               ; preds = %8
  %12 = call double @numberPassesF32Catch()
  %13 = fcmp oeq double %12, 2.250000e+00
  br i1 %13, label %14, label %22

14:                                               ; preds = %11
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

15:                                               ; preds = %22, %20, %18, %16
  unreachable

16:                                               ; preds = %0
  %17 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4304588356623647936, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17003361720708021055, i64 8), i32 60, ptr null)
  br label %15

18:                                               ; preds = %4
  %19 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15125558611850858047, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17003361720708021055, i64 8), i32 61, ptr null)
  br label %15

20:                                               ; preds = %8
  %21 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3429671353266982435, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17003361720708021055, i64 8), i32 62, ptr null)
  br label %15

22:                                               ; preds = %11
  %23 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2124737424241911831, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_17003361720708021055, i64 8), i32 63, ptr null)
  br label %15
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

