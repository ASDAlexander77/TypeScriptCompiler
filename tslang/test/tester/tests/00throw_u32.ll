; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$"___unbox<ui32>" = comdat any

$"___unbox<si32>" = comdat any

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
@s_9332519565554727459 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFs32\00", align 8
@s_3094732814638223685 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFi32\00", align 8
@s_15952682373371409732 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFf32\00", align 8
@s_16822532743728177535 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboolean\00", align 8
@s_8091808378024623192 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring\00", align 8
@s_9237349086447201248 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnumber\00", align 8
@s_5560783464435099345 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFu32\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_11381805876829376184 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFu32 to number is unsigned\00", align 8
@f_4050998935155952585 = internal constant [101 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00throw_u32.ts\00", align 8
@m_5576006961491055044 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: u32) catches a thrown u32\00", align 8
@m_7192102582270308793 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch reads a thrown u32 unsigned\00", align 8
@m_528062667432010869 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: number) widens a thrown u32 unsigned\00", align 8
@m_4425280313596494542 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcatch (e: number) still widens a thrown s32 signed\00", align 8
@m_11759106845429461338 = internal constant [57 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan untyped catch still reads a thrown s32 signed\00", align 8
@m_16106415213500427201 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa thrown s32 passes a u32 catch\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@big = private constant i32 -1
@_ZTIj = external constant ptr
@_ZTIi = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
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

declare i64 @strtoll(ptr, ptr, i32)

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
define i32 @typedU32() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  %4 = load i32, ptr @big, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 %4, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIj, ptr null)
          to label %21 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIj
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %11
  br label %21

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %1, align 4
  %18 = load i32, ptr %1, align 4
  store i32 %18, ptr %2, align 4
  call void @__cxa_end_catch()
  %19 = load i32, ptr %2, align 4
  store i32 %19, ptr %3, align 4
  %20 = load i32, ptr %3, align 4
  ret i32 %20

21:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %11
  %22 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %22
}

; Function Attrs: mustprogress
define linkonce_odr i32 @"___unbox<ui32>"(ptr %0) #3 comdat {
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
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5560783464435099345, i64 8) to i64)
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
define double @untypedU32() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double 0.000000e+00, ptr %3, align 8
  %5 = load i32, ptr @big, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 %5, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIj, ptr null)
          to label %151 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  %10 = call ptr @__cxa_begin_catch(ptr %9)
  %11 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %7
  %12 = icmp eq ptr %11, @_ZTIi
  br i1 %12, label %13, label %20

13:                                               ; preds = %.noexc9
  %14 = load i32, ptr %10, align 4
  %15 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %13
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 0
  store i64 4, ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %18, align 8
  %19 = getelementptr { i64, ptr, i32 }, ptr %16, i32 0, i32 2
  store i32 %14, ptr %19, align 4
  br label %143

20:                                               ; preds = %.noexc9
  %21 = icmp eq ptr %11, @_ZTIj
  br i1 %21, label %22, label %29

22:                                               ; preds = %20
  %23 = load i32, ptr %10, align 4
  %24 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %22
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 0
  store i64 4, ptr %26, align 8
  %27 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %27, align 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %25, i32 0, i32 2
  store i32 %23, ptr %28, align 4
  br label %140

29:                                               ; preds = %20
  %30 = icmp eq ptr %11, @_ZTId
  br i1 %30, label %31, label %38

31:                                               ; preds = %29
  %32 = load double, ptr %10, align 8
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 0
  store i64 8, ptr %35, align 8
  %36 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, double }, ptr %34, i32 0, i32 2
  store double %32, ptr %37, align 8
  br label %137

38:                                               ; preds = %29
  %39 = icmp eq ptr %11, @_ZTIf
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load float, ptr %10, align 4
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 0
  store i64 4, ptr %44, align 8
  %45 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, float }, ptr %43, i32 0, i32 2
  store float %41, ptr %46, align 4
  br label %134

47:                                               ; preds = %38
  %48 = icmp eq ptr %11, @_ZTIb
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load i1, ptr %10, align 1
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 0
  store i64 1, ptr %53, align 8
  %54 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, i1 }, ptr %52, i32 0, i32 2
  store i1 %50, ptr %55, align 1
  br label %131

56:                                               ; preds = %47
  %57 = icmp eq ptr %11, @_ZTIx
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load i64, ptr %10, align 8
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 0
  store i64 8, ptr %62, align 8
  %63 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, i64 }, ptr %61, i32 0, i32 2
  store i64 %59, ptr %64, align 8
  br label %128

65:                                               ; preds = %56
  %66 = icmp eq ptr %11, @_ZTIPc
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load ptr, ptr %10, align 8
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 0
  store i64 8, ptr %71, align 8
  %72 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, ptr }, ptr %70, i32 0, i32 2
  store ptr %68, ptr %73, align 8
  br label %125

74:                                               ; preds = %65
  %75 = icmp eq ptr %11, @_ZTIPv
  br i1 %75, label %76, label %78

76:                                               ; preds = %74
  %77 = load ptr, ptr %10, align 8
  br label %122

78:                                               ; preds = %74
  %79 = load ptr, ptr %11, align 8
  %80 = icmp eq ptr %79, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %80, label %81, label %112

81:                                               ; preds = %78
  %82 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %11, i32 0, i32 2
  %83 = load i32, ptr %82, align 4
  %84 = and i32 %83, 1073741824
  %85 = icmp ne i32 %84, 0
  br i1 %85, label %86, label %102

86:                                               ; preds = %81
  %87 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %11, i32 0, i32 4
  %88 = load ptr, ptr %87, align 8
  %89 = icmp ne ptr %88, null
  br i1 %89, label %90, label %92

90:                                               ; preds = %86
  invoke void %88(ptr %1, ptr %10)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %90
  %91 = load ptr, ptr %1, align 8
  br label %99

92:                                               ; preds = %86
  %93 = load ptr, ptr %10, align 8
  %94 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %92
  %95 = getelementptr i8, ptr %94, i64 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 0
  store i64 8, ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %97, align 8
  %98 = getelementptr { i64, ptr, ptr }, ptr %95, i32 0, i32 2
  store ptr %93, ptr %98, align 8
  br label %99

99:                                               ; preds = %.noexc10, %.noexc6
  %100 = phi ptr [ %95, %.noexc6 ], [ %91, %.noexc10 ]
  br label %101

101:                                              ; preds = %99
  br label %109

102:                                              ; preds = %81
  %103 = load ptr, ptr %10, align 8
  %104 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %102
  %105 = getelementptr i8, ptr %104, i64 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 0
  store i64 8, ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %105, i32 0, i32 2
  store ptr %103, ptr %108, align 8
  br label %109

109:                                              ; preds = %101, %.noexc11
  %110 = phi ptr [ %105, %.noexc11 ], [ %100, %101 ]
  br label %111

111:                                              ; preds = %109
  br label %119

112:                                              ; preds = %78
  %113 = load ptr, ptr %10, align 8
  %114 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %112
  %115 = getelementptr i8, ptr %114, i64 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 0
  store i64 8, ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %115, i32 0, i32 2
  store ptr %113, ptr %118, align 8
  br label %119

119:                                              ; preds = %111, %.noexc
  %120 = phi ptr [ %115, %.noexc ], [ %110, %111 ]
  br label %121

121:                                              ; preds = %119
  br label %122

122:                                              ; preds = %76, %121
  %123 = phi ptr [ %120, %121 ], [ %77, %76 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %.noexc7, %124
  %126 = phi ptr [ %123, %124 ], [ %70, %.noexc7 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc8, %127
  %129 = phi ptr [ %126, %127 ], [ %61, %.noexc8 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc3, %130
  %132 = phi ptr [ %129, %130 ], [ %52, %.noexc3 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc12, %133
  %135 = phi ptr [ %132, %133 ], [ %43, %.noexc12 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc4, %136
  %138 = phi ptr [ %135, %136 ], [ %34, %.noexc4 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc1, %139
  %141 = phi ptr [ %138, %139 ], [ %25, %.noexc1 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc5, %142
  %144 = phi ptr [ %141, %142 ], [ %16, %.noexc5 ]
  br label %145

145:                                              ; preds = %143
  store ptr %144, ptr %2, align 8
  %146 = load ptr, ptr %2, align 8
  %147 = invoke i32 @"___unbox<ui32>"(ptr %146)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %145
  %148 = uitofp i32 %147 to double
  store double %148, ptr %3, align 8
  call void @__cxa_end_catch()
  %149 = load double, ptr %3, align 8
  store double %149, ptr %4, align 8
  %150 = load double, ptr %4, align 8
  ret double %150

151:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %40, %102, %90, %7, %58, %67, %92, %13, %31, %49, %145, %22, %112
  %152 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %152
}

; Function Attrs: mustprogress
define double @u32AsNumber() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  %4 = load i32, ptr @big, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 %4, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIj, ptr null)
          to label %58 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %10 = icmp eq i32 %8, %9
  %11 = extractvalue { ptr, i32 } %7, 1
  %12 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %13 = icmp eq i32 %11, %12
  %14 = extractvalue { ptr, i32 } %7, 1
  %15 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %16 = icmp eq i32 %14, %15
  %17 = extractvalue { ptr, i32 } %7, 1
  %18 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %19 = icmp eq i32 %17, %18
  %20 = or i1 %10, %13
  %21 = or i1 %20, %16
  %22 = or i1 %21, %19
  br i1 %22, label %26, label %23

23:                                               ; preds = %6
  %24 = extractvalue { ptr, i32 } %7, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %23
  br label %58

26:                                               ; preds = %6
  %27 = extractvalue { ptr, i32 } %7, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %26
  %30 = icmp eq ptr %29, @_ZTIi
  br i1 %30, label %31, label %34

31:                                               ; preds = %.noexc1
  %32 = load i32, ptr %28, align 4
  %33 = sitofp i32 %32 to double
  br label %52

34:                                               ; preds = %.noexc1
  %35 = icmp eq ptr %29, @_ZTIj
  br i1 %35, label %36, label %39

36:                                               ; preds = %34
  %37 = load i32, ptr %28, align 4
  %38 = uitofp i32 %37 to double
  br label %49

39:                                               ; preds = %34
  %40 = icmp eq ptr %29, @_ZTIf
  br i1 %40, label %41, label %44

41:                                               ; preds = %39
  %42 = load float, ptr %28, align 4
  %43 = fpext float %42 to double
  br label %46

44:                                               ; preds = %39
  %45 = load double, ptr %28, align 8
  br label %46

46:                                               ; preds = %41, %44
  %47 = phi double [ %45, %44 ], [ %43, %41 ]
  br label %48

48:                                               ; preds = %46
  br label %49

49:                                               ; preds = %36, %48
  %50 = phi double [ %47, %48 ], [ %38, %36 ]
  br label %51

51:                                               ; preds = %49
  br label %52

52:                                               ; preds = %31, %51
  %53 = phi double [ %50, %51 ], [ %33, %31 ]
  br label %54

54:                                               ; preds = %52
  store double %53, ptr %1, align 8
  %55 = load double, ptr %1, align 8
  store double %55, ptr %2, align 8
  call void @__cxa_end_catch()
  %56 = load double, ptr %2, align 8
  store double %56, ptr %3, align 8
  %57 = load double, ptr %3, align 8
  ret double %57

58:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %26, %23
  %59 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %59
}

; Function Attrs: mustprogress
define double @s32AsNumber() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double 0.000000e+00, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 -1, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %57 unwind label %5

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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %22
  br label %57

25:                                               ; preds = %5
  %26 = extractvalue { ptr, i32 } %6, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  %28 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %25
  %29 = icmp eq ptr %28, @_ZTIi
  br i1 %29, label %30, label %33

30:                                               ; preds = %.noexc
  %31 = load i32, ptr %27, align 4
  %32 = sitofp i32 %31 to double
  br label %51

33:                                               ; preds = %.noexc
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
  store double %54, ptr %2, align 8
  call void @__cxa_end_catch()
  %55 = load double, ptr %2, align 8
  store double %55, ptr %3, align 8
  %56 = load double, ptr %3, align 8
  ret double %56

57:                                               ; preds = %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %22, %25
  %58 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %58
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
define i32 @untypedS32() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 -1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %149 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = call ptr @__cxa_begin_catch(ptr %8)
  %10 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %6
  %11 = icmp eq ptr %10, @_ZTIi
  br i1 %11, label %12, label %19

12:                                               ; preds = %.noexc4
  %13 = load i32, ptr %9, align 4
  %14 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %12
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 0
  store i64 4, ptr %16, align 8
  %17 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %17, align 8
  %18 = getelementptr { i64, ptr, i32 }, ptr %15, i32 0, i32 2
  store i32 %13, ptr %18, align 4
  br label %142

19:                                               ; preds = %.noexc4
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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %30
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
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %39
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %48
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
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %57
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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %89
  %90 = load ptr, ptr %1, align 8
  br label %98

91:                                               ; preds = %85
  %92 = load ptr, ptr %9, align 8
  %93 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %91
  %94 = getelementptr i8, ptr %93, i64 8
  %95 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 0
  store i64 8, ptr %95, align 8
  %96 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %96, align 8
  %97 = getelementptr { i64, ptr, ptr }, ptr %94, i32 0, i32 2
  store ptr %92, ptr %97, align 8
  br label %98

98:                                               ; preds = %.noexc1, %.noexc6
  %99 = phi ptr [ %94, %.noexc6 ], [ %90, %.noexc1 ]
  br label %100

100:                                              ; preds = %98
  br label %108

101:                                              ; preds = %80
  %102 = load ptr, ptr %9, align 8
  %103 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %101
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 0
  store i64 8, ptr %105, align 8
  %106 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %106, align 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %104, i32 0, i32 2
  store ptr %102, ptr %107, align 8
  br label %108

108:                                              ; preds = %100, %.noexc2
  %109 = phi ptr [ %104, %.noexc2 ], [ %99, %100 ]
  br label %110

110:                                              ; preds = %108
  br label %118

111:                                              ; preds = %77
  %112 = load ptr, ptr %9, align 8
  %113 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %111
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 0
  store i64 8, ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %116, align 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %114, i32 0, i32 2
  store ptr %112, ptr %117, align 8
  br label %118

118:                                              ; preds = %110, %.noexc9
  %119 = phi ptr [ %114, %.noexc9 ], [ %109, %110 ]
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

127:                                              ; preds = %.noexc3, %126
  %128 = phi ptr [ %125, %126 ], [ %60, %.noexc3 ]
  br label %129

129:                                              ; preds = %127
  br label %130

130:                                              ; preds = %.noexc, %129
  %131 = phi ptr [ %128, %129 ], [ %51, %.noexc ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc12, %132
  %134 = phi ptr [ %131, %132 ], [ %42, %.noexc12 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc8, %135
  %137 = phi ptr [ %134, %135 ], [ %33, %.noexc8 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc7, %138
  %140 = phi ptr [ %137, %138 ], [ %24, %.noexc7 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc10, %141
  %143 = phi ptr [ %140, %141 ], [ %15, %.noexc10 ]
  br label %144

144:                                              ; preds = %142
  store ptr %143, ptr %2, align 8
  %145 = load ptr, ptr %2, align 8
  %146 = invoke i32 @"___unbox<si32>"(ptr %145)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %144
  store i32 %146, ptr %3, align 4
  call void @__cxa_end_catch()
  %147 = load i32, ptr %3, align 4
  store i32 %147, ptr %4, align 4
  %148 = load i32, ptr %4, align 4
  ret i32 %148

149:                                              ; preds = %0
  unreachable

endcatch.resume:                                  ; preds = %39, %66, %12, %111, %30, %21, %91, %144, %6, %57, %101, %89, %48
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %150
}

; Function Attrs: mustprogress
define double @s32PassesU32Catch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double 0.000000e+00, ptr %3, align 8
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 -2, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %73 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIj
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
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
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %1, align 4
  store double 1.000000e+00, ptr %3, align 8
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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %37
  br label %73

40:                                               ; preds = %.caught
  %41 = extractvalue { ptr, i32 } %21, 0
  %42 = call ptr @__cxa_begin_catch(ptr %41)
  %43 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %40
  %44 = icmp eq ptr %43, @_ZTIi
  br i1 %44, label %45, label %48

45:                                               ; preds = %.noexc
  %46 = load i32, ptr %42, align 4
  %47 = sitofp i32 %46 to double
  br label %66

48:                                               ; preds = %.noexc
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

73:                                               ; preds = %.noexc1, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %37, %40
  %74 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %74
}

; Function Attrs: mustprogress
define i32 @main() #3 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = load i32, ptr @big, align 4
  %2 = uitofp i32 %1 to double
  %3 = fcmp ogt double %2, 0.000000e+00
  br i1 %3, label %4, label %25

4:                                                ; preds = %0
  %5 = call i32 @typedU32()
  %6 = load i32, ptr @big, align 4
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %8, label %27

8:                                                ; preds = %4
  %9 = call double @untypedU32()
  %10 = fcmp ogt double %9, 0.000000e+00
  br i1 %10, label %11, label %29

11:                                               ; preds = %8
  %12 = call double @u32AsNumber()
  %13 = fcmp ogt double %12, 0.000000e+00
  br i1 %13, label %14, label %31

14:                                               ; preds = %11
  %15 = call double @s32AsNumber()
  %16 = fcmp olt double %15, 0.000000e+00
  br i1 %16, label %17, label %33

17:                                               ; preds = %14
  %18 = call i32 @untypedS32()
  %19 = icmp slt i32 %18, 0
  br i1 %19, label %20, label %35

20:                                               ; preds = %17
  %21 = call double @s32PassesU32Catch()
  %22 = fcmp oeq double %21, -2.000000e+00
  br i1 %22, label %23, label %37

23:                                               ; preds = %20
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

24:                                               ; preds = %37, %35, %33, %31, %29, %27, %25
  unreachable

25:                                               ; preds = %0
  %26 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11381805876829376184, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 87, ptr null)
  br label %24

27:                                               ; preds = %4
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5576006961491055044, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 89, ptr null)
  br label %24

29:                                               ; preds = %8
  %30 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7192102582270308793, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 90, ptr null)
  br label %24

31:                                               ; preds = %11
  %32 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_528062667432010869, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 91, ptr null)
  br label %24

33:                                               ; preds = %14
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4425280313596494542, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 92, ptr null)
  br label %24

35:                                               ; preds = %17
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11759106845429461338, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 93, ptr null)
  br label %24

37:                                               ; preds = %20
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16106415213500427201, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4050998935155952585, i64 8), i32 94, ptr null)
  br label %24
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
declare i64 @llvm.smin.i64(i64, i64) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress "personality" }
attributes #3 = { mustprogress }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

