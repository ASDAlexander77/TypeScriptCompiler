; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%V = type { ptr, ptr, double }
%H = type { ptr, ptr }
%Junk = type { ptr, i32, i32, ptr }

$"___unbox<!ts.number>" = comdat any

$V..size = comdat any

$H..size = comdat any

$Junk..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_12638238877602167913 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFV\00", align 8
@s_12638232280532398647 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFH\00", align 8
@s_1560590634776335115 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFJunk\00", align 8
@s_12638203693230065161 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFv\00", align 8
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
@s_16940945161124618027 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFjunk\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_4377004417030018898 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00argument_slot_assigned.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_13796550140266830968 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe argument's block\00", align 8
@m_13005919702631549433 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe arguments' blocks\00", align 8
@m_4015499365481866607 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local\00", align 8
@m_12306812985014937222 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe local's new value\00", align 8
@m_2519180544061476713 = internal constant [36 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local, then its new value\00", align 8
@m_2959860279860419253 = internal constant [47 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local, assigned in the next argument\00", align 8
@m_12630436860726440249 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local in a loop\00", align 8
@m_12439984521663929121 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field of a local\00", align 8
@m_3161810430787740801 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field of a const\00", align 8
@m_4081330473648082071 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan element of a local\00", align 8
@m_10202604351289301259 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa conditional argument\00", align 8
@m_15252316600109684411 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa call in a try body\00", align 8
@m_8798840678818865255 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnothing throws\00", align 8
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
@m_234323632491049957 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe call throws\00", align 8
@m_4443436482793291723 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa parameter\00", align 8
@m_9712415665295363832 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa captured local, assigned by a closure\00", align 8
@m_12745201741415506076 = internal constant [39 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe captured local's new value\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@V..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638238877602167913, i64 8)
@V..size = global i64 24, comdat
@V..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @V..instanceOf, ptr @V..new, ptr @V..rtti, ptr @V..size }
@H..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638232280532398647, i64 8)
@H..size = global i64 16, comdat
@H..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @H..instanceOf, ptr @H..new, ptr @H..rtti, ptr @H..size }
@Junk..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1560590634776335115, i64 8)
@Junk..size = global i64 24, comdat
@Junk..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Junk..instanceOf, ptr @Junk..new, ptr @Junk..rtti, ptr @Junk..size }
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@a_0_6706924739130738156 = internal constant [0 x ptr] undef
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_1071659150439645108_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_1071659150439645108, ptr @tsretcb_1071659150439645108, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

define internal void @tsretcb_1071659150439645108(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrel_1375346448667278364(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %V, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrelcb_1071659150439645108(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %14

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %14

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %13

10:                                               ; preds = %6
  %11 = call i1 @__tslang_dec_ref(ptr %8)
  br i1 %11, label %12, label %13

12:                                               ; preds = %10
  call void @tsrel_1375346448667278364(ptr %8)
  call void @__tslang_free_block(ptr %8)
  br label %13

13:                                               ; preds = %12, %10, %6
  call void @__tslang_free_block(ptr %2)
  br label %14

14:                                               ; preds = %13, %4, %1
  ret void
}

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

; Function Attrs: mustprogress
define i1 @V..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @V..rtti, align 8
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
define ptr @V..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %V, ptr %3, i32 0, i32 0
  store ptr @V..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @V.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %V, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %V, ptr %7, i32 0, i32 2
  %9 = load double, ptr %3, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr %V, ptr %10, i32 0, i32 1
  %12 = load double, ptr %3, align 8
  %13 = call ptr @GC_malloc_atomic(i64 40)
  %14 = getelementptr i8, ptr %13, i64 8
  call void @__ts_number_to_string(ptr %14, double %12)
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr %16)
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %17, i64 %25)
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
define void @H.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %H, ptr %3, i32 0, i32 1
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %V, ptr %6, i32 0, i32 0
  store ptr @V..vtbl, ptr %7, align 8
  call void @V.constructor(ptr %6, double 0.000000e+00)
  store ptr %6, ptr %4, align 8
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
define void @Junk.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Junk, ptr %3, i32 0, i32 1
  store i32 -1, ptr %4, align 4
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Junk, ptr %5, i32 0, i32 2
  store i32 -2, ptr %6, align 4
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %Junk, ptr %7, i32 0, i32 3
  store ptr getelementptr inbounds nuw (i8, ptr @s_16940945161124618027, i64 8), ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Junk..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Junk..rtti, align 8
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
define ptr @Junk..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Junk, ptr %3, i32 0, i32 0
  store ptr @Junk..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i64 @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = call ptr @GC_malloc(i64 8)
  %5 = getelementptr i8, ptr %4, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %5, ptr @a_0_6706924739130738156, i64 0, i1 false)
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
  %13 = icmp slt i32 %12, 1000
  br i1 %13, label %14, label %66

14:                                               ; preds = %11
  %15 = call ptr @GC_malloc(i64 32)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr %Junk, ptr %16, i32 0, i32 0
  store ptr @Junk..vtbl, ptr %17, align 8
  call void @Junk.constructor(ptr %16)
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
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 26, ptr null)
  br label %73

76:                                               ; preds = %42
  %77 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 26, ptr null)
  br label %73

78:                                               ; preds = %44
  %79 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 26, ptr null)
  br label %73
}

; Function Attrs: mustprogress
define ptr @mk(double %0) #2 {
  %2 = alloca double, align 8
  %3 = alloca ptr, align 8
  store double %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %V, ptr %5, i32 0, i32 0
  store ptr @V..vtbl, ptr %6, align 8
  %7 = load double, ptr %2, align 8
  call void @V.constructor(ptr %5, double %7)
  store ptr %5, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define double @g(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = call i64 @churn()
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %V, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr %V, ptr %10, i32 0, i32 2
  %12 = load double, ptr %11, align 8
  %13 = call ptr @GC_malloc_atomic(i64 40)
  %14 = getelementptr i8, ptr %13, i64 8
  call void @__ts_number_to_string(ptr %14, double %12)
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr %16)
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %20, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr %16, i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  %33 = ptrtoint ptr %9 to i64
  %34 = ptrtoint ptr %23 to i64
  %35 = icmp ne i64 %33, 0
  %36 = icmp ne i64 %34, 0
  %37 = zext i1 %35 to i32
  %38 = zext i1 %36 to i32
  %39 = and i32 %37, %38
  %40 = icmp ne i32 %39, 0
  br i1 %40, label %41, label %44

41:                                               ; preds = %2
  %42 = call i32 @strcmp(ptr %9, ptr %23)
  %43 = icmp eq i32 %42, 0
  br label %46

44:                                               ; preds = %2
  %45 = icmp eq i64 %33, %34
  br label %46

46:                                               ; preds = %41, %44
  %47 = phi i1 [ %45, %44 ], [ %43, %41 ]
  br label %48

48:                                               ; preds = %46
  br i1 %47, label %49, label %58

49:                                               ; preds = %48
  %50 = load ptr, ptr %4, align 8
  %51 = getelementptr %V, ptr %50, i32 0, i32 2
  %52 = load double, ptr %51, align 8
  %53 = fmul double %52, 1.000000e+01
  %54 = load double, ptr %3, align 8
  %55 = fadd double %53, %54
  store double %55, ptr %5, align 8
  %56 = load double, ptr %5, align 8
  ret double %56

57:                                               ; preds = %58
  unreachable

58:                                               ; preds = %48
  %59 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13796550140266830968, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 36, ptr null)
  br label %57
}

; Function Attrs: mustprogress
define double @g2(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = call i64 @churn()
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %V, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr %V, ptr %10, i32 0, i32 2
  %12 = load double, ptr %11, align 8
  %13 = call ptr @GC_malloc_atomic(i64 40)
  %14 = getelementptr i8, ptr %13, i64 8
  call void @__ts_number_to_string(ptr %14, double %12)
  %15 = icmp eq ptr %14, null
  %16 = select i1 %15, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %18 = add i64 1, %17
  %19 = call i64 @strlen(ptr %16)
  %20 = add i64 %18, %19
  %21 = add i64 %20, 8
  %22 = call ptr @GC_malloc(i64 %21)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr i8, ptr %23, i64 0
  %25 = sub i64 %20, 0
  %26 = call ptr @__memcpy_chk(ptr %24, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %17, i64 %25)
  %27 = add i64 0, %17
  %28 = getelementptr i8, ptr %23, i64 %27
  %29 = sub i64 %20, %27
  %30 = call ptr @__memcpy_chk(ptr %28, ptr %16, i64 %19, i64 %29)
  %31 = add i64 %27, %19
  %32 = getelementptr i8, ptr %23, i64 %31
  store i8 0, ptr %32, align 1
  %33 = ptrtoint ptr %9 to i64
  %34 = ptrtoint ptr %23 to i64
  %35 = icmp ne i64 %33, 0
  %36 = icmp ne i64 %34, 0
  %37 = zext i1 %35 to i32
  %38 = zext i1 %36 to i32
  %39 = and i32 %37, %38
  %40 = icmp ne i32 %39, 0
  br i1 %40, label %41, label %44

41:                                               ; preds = %2
  %42 = call i32 @strcmp(ptr %9, ptr %23)
  %43 = icmp eq i32 %42, 0
  br label %46

44:                                               ; preds = %2
  %45 = icmp eq i64 %33, %34
  br label %46

46:                                               ; preds = %41, %44
  %47 = phi i1 [ %45, %44 ], [ %43, %41 ]
  br label %48

48:                                               ; preds = %46
  br i1 %47, label %49, label %92

49:                                               ; preds = %48
  %50 = load ptr, ptr %3, align 8
  %51 = getelementptr %V, ptr %50, i32 0, i32 1
  %52 = load ptr, ptr %51, align 8
  %53 = load ptr, ptr %3, align 8
  %54 = getelementptr %V, ptr %53, i32 0, i32 2
  %55 = load double, ptr %54, align 8
  %56 = call ptr @GC_malloc_atomic(i64 40)
  %57 = getelementptr i8, ptr %56, i64 8
  call void @__ts_number_to_string(ptr %57, double %55)
  %58 = icmp eq ptr %57, null
  %59 = select i1 %58, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %57
  %60 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %61 = add i64 1, %60
  %62 = call i64 @strlen(ptr %59)
  %63 = add i64 %61, %62
  %64 = add i64 %63, 8
  %65 = call ptr @GC_malloc(i64 %64)
  %66 = getelementptr i8, ptr %65, i64 8
  %67 = getelementptr i8, ptr %66, i64 0
  %68 = sub i64 %63, 0
  %69 = call ptr @__memcpy_chk(ptr %67, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %60, i64 %68)
  %70 = add i64 0, %60
  %71 = getelementptr i8, ptr %66, i64 %70
  %72 = sub i64 %63, %70
  %73 = call ptr @__memcpy_chk(ptr %71, ptr %59, i64 %62, i64 %72)
  %74 = add i64 %70, %62
  %75 = getelementptr i8, ptr %66, i64 %74
  store i8 0, ptr %75, align 1
  %76 = ptrtoint ptr %52 to i64
  %77 = ptrtoint ptr %66 to i64
  %78 = icmp ne i64 %76, 0
  %79 = icmp ne i64 %77, 0
  %80 = zext i1 %78 to i32
  %81 = zext i1 %79 to i32
  %82 = and i32 %80, %81
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %87

84:                                               ; preds = %49
  %85 = call i32 @strcmp(ptr %52, ptr %66)
  %86 = icmp eq i32 %85, 0
  br label %89

87:                                               ; preds = %49
  %88 = icmp eq i64 %76, %77
  br label %89

89:                                               ; preds = %84, %87
  %90 = phi i1 [ %88, %87 ], [ %86, %84 ]
  br label %91

91:                                               ; preds = %89
  br label %92

92:                                               ; preds = %91, %48
  %93 = phi i1 [ %90, %91 ], [ %47, %48 ]
  br i1 %93, label %94, label %105

94:                                               ; preds = %92
  %95 = load ptr, ptr %4, align 8
  %96 = getelementptr %V, ptr %95, i32 0, i32 2
  %97 = load double, ptr %96, align 8
  %98 = fmul double %97, 1.000000e+01
  %99 = load ptr, ptr %3, align 8
  %100 = getelementptr %V, ptr %99, i32 0, i32 2
  %101 = load double, ptr %100, align 8
  %102 = fadd double %98, %101
  store double %102, ptr %5, align 8
  %103 = load double, ptr %5, align 8
  ret double %103

104:                                              ; preds = %105
  unreachable

105:                                              ; preds = %92
  %106 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13005919702631549433, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 42, ptr null)
  br label %104
}

; Function Attrs: mustprogress
define double @gThrows(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = call i64 @churn()
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr %V, ptr %6, i32 0, i32 1
  %8 = load ptr, ptr %7, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr %V, ptr %9, i32 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = call ptr @GC_malloc_atomic(i64 40)
  %13 = getelementptr i8, ptr %12, i64 8
  call void @__ts_number_to_string(ptr %13, double %11)
  %14 = icmp eq ptr %13, null
  %15 = select i1 %14, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %13
  %16 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8))
  %17 = add i64 1, %16
  %18 = call i64 @strlen(ptr %15)
  %19 = add i64 %17, %18
  %20 = add i64 %19, 8
  %21 = call ptr @GC_malloc(i64 %20)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr i8, ptr %22, i64 0
  %24 = sub i64 %19, 0
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12638203693230065161, i64 8), i64 %16, i64 %24)
  %26 = add i64 0, %16
  %27 = getelementptr i8, ptr %22, i64 %26
  %28 = sub i64 %19, %26
  %29 = call ptr @__memcpy_chk(ptr %27, ptr %15, i64 %18, i64 %28)
  %30 = add i64 %26, %18
  %31 = getelementptr i8, ptr %22, i64 %30
  store i8 0, ptr %31, align 1
  %32 = ptrtoint ptr %8 to i64
  %33 = ptrtoint ptr %22 to i64
  %34 = icmp ne i64 %32, 0
  %35 = icmp ne i64 %33, 0
  %36 = zext i1 %34 to i32
  %37 = zext i1 %35 to i32
  %38 = and i32 %36, %37
  %39 = icmp ne i32 %38, 0
  br i1 %39, label %40, label %43

40:                                               ; preds = %2
  %41 = call i32 @strcmp(ptr %8, ptr %22)
  %42 = icmp eq i32 %41, 0
  br label %45

43:                                               ; preds = %2
  %44 = icmp eq i64 %32, %33
  br label %45

45:                                               ; preds = %40, %43
  %46 = phi i1 [ %44, %43 ], [ %42, %40 ]
  br label %47

47:                                               ; preds = %45
  br i1 %46, label %48, label %57

48:                                               ; preds = %47
  %49 = load ptr, ptr %4, align 8
  %50 = getelementptr %V, ptr %49, i32 0, i32 2
  %51 = load double, ptr %50, align 8
  %52 = fmul double %51, 1.000000e+01
  %53 = load double, ptr %3, align 8
  %54 = fadd double %52, %53
  %55 = call ptr @__cxa_allocate_exception(i64 8)
  store double %54, ptr %55, align 8
  call void @__cxa_throw(ptr %55, ptr @_ZTId, ptr null)
  br label %56

56:                                               ; preds = %48, %57
  unreachable

57:                                               ; preds = %47
  %58 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13796550140266830968, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 48, ptr null)
  br label %56
}

; Function Attrs: mustprogress
define void @local() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @mk(double 1.000000e+00)
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @mk(double 2.000000e+00)
  store ptr %4, ptr %1, align 8
  %5 = call double @g(ptr %3, double 0.000000e+00)
  %6 = fcmp oeq double %5, 1.000000e+01
  br i1 %6, label %7, label %14

7:                                                ; preds = %0
  %8 = load ptr, ptr %1, align 8
  %9 = getelementptr %V, ptr %8, i32 0, i32 2
  %10 = load double, ptr %9, align 8
  %11 = fcmp oeq double %10, 2.000000e+00
  br i1 %11, label %12, label %16

12:                                               ; preds = %7
  ret void

13:                                               ; preds = %16, %14
  unreachable

14:                                               ; preds = %0
  %15 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4015499365481866607, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 54, ptr null)
  br label %13

16:                                               ; preds = %7
  %17 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12306812985014937222, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 55, ptr null)
  br label %13
}

; Function Attrs: mustprogress
define void @localTwice() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @mk(double 1.000000e+00)
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @mk(double 2.000000e+00)
  store ptr %4, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  %6 = call double @g2(ptr %3, ptr %5)
  %7 = fcmp oeq double %6, 1.200000e+01
  br i1 %7, label %8, label %10

8:                                                ; preds = %0
  ret void

9:                                                ; preds = %10
  unreachable

10:                                               ; preds = %0
  %11 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2519180544061476713, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 60, ptr null)
  br label %9
}

; Function Attrs: mustprogress
define void @localAssignment() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @mk(double 1.000000e+00)
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @mk(double 3.000000e+00)
  store ptr %4, ptr %1, align 8
  %5 = call double @g2(ptr %3, ptr %4)
  %6 = fcmp oeq double %5, 1.300000e+01
  br i1 %6, label %7, label %9

7:                                                ; preds = %0
  ret void

8:                                                ; preds = %9
  unreachable

9:                                                ; preds = %0
  %10 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2959860279860419253, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 65, ptr null)
  br label %8
}

; Function Attrs: mustprogress
define void @inLoop() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  br label %4

4:                                                ; preds = %7, %0
  %5 = load i32, ptr %2, align 4
  %6 = icmp slt i32 %5, 100
  br i1 %6, label %7, label %24

7:                                                ; preds = %4
  %8 = load i32, ptr %2, align 4
  %9 = sitofp i32 %8 to double
  %10 = call ptr @mk(double %9)
  store ptr %10, ptr %1, align 8
  %11 = load i32, ptr %3, align 4
  %12 = load ptr, ptr %1, align 8
  %13 = load i32, ptr %2, align 4
  %14 = add i32 %13, 1
  %15 = sitofp i32 %14 to double
  %16 = call ptr @mk(double %15)
  store ptr %16, ptr %1, align 8
  %17 = call double @g(ptr %12, double 0.000000e+00)
  %18 = sitofp i32 %11 to double
  %19 = fadd double %18, %17
  %20 = fptosi double %19 to i64
  %21 = trunc i64 %20 to i32
  store i32 %21, ptr %3, align 4
  %22 = load i32, ptr %2, align 4
  %23 = add i32 %22, 1
  store i32 %23, ptr %2, align 4
  br label %4

24:                                               ; preds = %4
  %25 = load i32, ptr %3, align 4
  %26 = icmp eq i32 %25, 49500
  br i1 %26, label %27, label %29

27:                                               ; preds = %24
  ret void

28:                                               ; preds = %29
  unreachable

29:                                               ; preds = %24
  %30 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12630436860726440249, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 75, ptr null)
  br label %28
}

; Function Attrs: mustprogress
define void @field() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca <{ i64, %H }>, align 16
  %3 = alloca <{ i64, %H }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %H zeroinitializer, ptr %4, align 8
  store i64 -1, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 8
  store %H zeroinitializer, ptr %5, align 8
  store %H zeroinitializer, ptr %5, align 8
  %6 = getelementptr %H, ptr %5, i32 0, i32 0
  store ptr @H..vtbl, ptr %6, align 8
  call void @H.constructor(ptr %5)
  store ptr %5, ptr %1, align 8
  %7 = load ptr, ptr %1, align 8
  %8 = getelementptr %H, ptr %7, i32 0, i32 1
  %9 = call ptr @mk(double 1.000000e+00)
  store ptr %9, ptr %8, align 8
  %10 = load ptr, ptr %1, align 8
  %11 = getelementptr %H, ptr %10, i32 0, i32 1
  %12 = load ptr, ptr %11, align 8
  %13 = load ptr, ptr %1, align 8
  %14 = getelementptr %H, ptr %13, i32 0, i32 1
  %15 = call ptr @mk(double 2.000000e+00)
  store ptr %15, ptr %14, align 8
  %16 = call double @g(ptr %12, double 0.000000e+00)
  %17 = fcmp oeq double %16, 1.000000e+01
  br i1 %17, label %18, label %30

18:                                               ; preds = %0
  store %H zeroinitializer, ptr %4, align 8
  %19 = getelementptr %H, ptr %4, i32 0, i32 0
  store ptr @H..vtbl, ptr %19, align 8
  call void @H.constructor(ptr %4)
  %20 = getelementptr %H, ptr %4, i32 0, i32 1
  %21 = call ptr @mk(double 3.000000e+00)
  store ptr %21, ptr %20, align 8
  %22 = getelementptr %H, ptr %4, i32 0, i32 1
  %23 = load ptr, ptr %22, align 8
  %24 = getelementptr %H, ptr %4, i32 0, i32 1
  %25 = call ptr @mk(double 4.000000e+00)
  store ptr %25, ptr %24, align 8
  %26 = call double @g(ptr %23, double 0.000000e+00)
  %27 = fcmp oeq double %26, 3.000000e+01
  br i1 %27, label %28, label %32

28:                                               ; preds = %18
  ret void

29:                                               ; preds = %32, %30
  unreachable

30:                                               ; preds = %0
  %31 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12439984521663929121, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 81, ptr null)
  br label %29

32:                                               ; preds = %18
  %33 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3161810430787740801, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 84, ptr null)
  br label %29
}

; Function Attrs: mustprogress
define void @element() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %3 = alloca <{ i64, [1 x ptr] }>, align 16
  %4 = call ptr @mk(double 1.000000e+00)
  store i64 -1, ptr %3, align 8
  %5 = getelementptr i8, ptr %3, i64 8
  store [1 x ptr] zeroinitializer, ptr %5, align 8
  %6 = getelementptr ptr, ptr %5, i64 0
  store ptr %4, ptr %6, align 8
  store i64 -1, ptr %2, align 8
  %7 = getelementptr i8, ptr %2, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %7, align 8
  %8 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 0
  store ptr %5, ptr %8, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 1
  store i64 1, ptr %9, align 8
  %10 = getelementptr { ptr, i64, i64 }, ptr %7, i32 0, i32 2
  store i64 1, ptr %10, align 8
  store ptr %7, ptr %1, align 8
  %11 = load ptr, ptr %1, align 8
  %12 = icmp eq ptr %11, null
  %13 = select i1 %12, ptr @__ts_empty_array, ptr %11
  %14 = getelementptr { ptr, i64, i64 }, ptr %13, i32 0, i32 0
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr ptr, ptr %15, i32 0
  %17 = load ptr, ptr %16, align 8
  %18 = load ptr, ptr %1, align 8
  %19 = icmp eq ptr %18, null
  %20 = select i1 %19, ptr @__ts_empty_array, ptr %18
  %21 = getelementptr { ptr, i64, i64 }, ptr %20, i32 0, i32 0
  %22 = load ptr, ptr %21, align 8
  %23 = getelementptr ptr, ptr %22, i32 0
  %24 = call ptr @mk(double 2.000000e+00)
  %25 = icmp eq ptr %18, null
  %26 = select i1 %25, ptr @__ts_empty_array, ptr %18
  %27 = getelementptr { ptr, i64, i64 }, ptr %26, i32 0, i32 0
  %28 = load ptr, ptr %27, align 8
  %29 = getelementptr ptr, ptr %28, i32 0
  store ptr %24, ptr %29, align 8
  %30 = call double @g(ptr %17, double 0.000000e+00)
  %31 = fcmp oeq double %30, 1.000000e+01
  br i1 %31, label %32, label %34

32:                                               ; preds = %0
  ret void

33:                                               ; preds = %34
  unreachable

34:                                               ; preds = %0
  %35 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4081330473648082071, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 89, ptr null)
  br label %33
}

; Function Attrs: mustprogress
define void @conditional() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @mk(double 1.000000e+00)
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = getelementptr %V, ptr %3, i32 0, i32 2
  %5 = load double, ptr %4, align 8
  %6 = fcmp ogt double %5, 0.000000e+00
  %7 = load ptr, ptr %1, align 8
  br i1 %6, label %8, label %10

8:                                                ; preds = %0
  %9 = call ptr @mk(double 2.000000e+00)
  store ptr %9, ptr %1, align 8
  br label %11

10:                                               ; preds = %0
  br label %11

11:                                               ; preds = %8, %10
  %12 = phi i32 [ 1, %10 ], [ 0, %8 ]
  %13 = sitofp i32 %12 to double
  %14 = call double @g(ptr %7, double %13)
  %15 = fcmp oeq double %14, 1.000000e+01
  br i1 %15, label %16, label %18

16:                                               ; preds = %11
  ret void

17:                                               ; preds = %18
  unreachable

18:                                               ; preds = %11
  %19 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10202604351289301259, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 95, ptr null)
  br label %17
}

; Function Attrs: mustprogress
define void @inTry() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = call ptr @mk(double 1.000000e+00)
  store ptr %4, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = invoke ptr @mk(double 2.000000e+00)
          to label %7 unwind label %12

7:                                                ; preds = %0
  store ptr %6, ptr %3, align 8
  %8 = invoke double @g(ptr %5, double 0.000000e+00)
          to label %9 unwind label %12

9:                                                ; preds = %7
  %10 = fcmp oeq double %8, 1.000000e+01
  br i1 %10, label %11, label %154

11:                                               ; preds = %9
  br label %152

12:                                               ; preds = %7, %0
  %13 = landingpad { ptr, i32 }
          catch ptr null
  %14 = extractvalue { ptr, i32 } %13, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %12
  %17 = icmp eq ptr %16, @_ZTIi
  br i1 %17, label %18, label %25

18:                                               ; preds = %.noexc11
  %19 = load i32, ptr %15, align 4
  %20 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %18
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 0
  store i64 4, ptr %22, align 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %21, i32 0, i32 2
  store i32 %19, ptr %24, align 4
  br label %148

25:                                               ; preds = %.noexc11
  %26 = icmp eq ptr %16, @_ZTIj
  br i1 %26, label %27, label %34

27:                                               ; preds = %25
  %28 = load i32, ptr %15, align 4
  %29 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %27
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
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %36
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
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %45
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %54
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %63
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %72
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
  invoke void %93(ptr %1, ptr %15)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %95
  %96 = load ptr, ptr %1, align 8
  br label %104

97:                                               ; preds = %91
  %98 = load ptr, ptr %15, align 8
  %99 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %97
  %100 = getelementptr i8, ptr %99, i64 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 0
  store i64 8, ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %100, i32 0, i32 2
  store ptr %98, ptr %103, align 8
  br label %104

104:                                              ; preds = %.noexc, %.noexc6
  %105 = phi ptr [ %100, %.noexc6 ], [ %96, %.noexc ]
  br label %106

106:                                              ; preds = %104
  br label %114

107:                                              ; preds = %86
  %108 = load ptr, ptr %15, align 8
  %109 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %107
  %110 = getelementptr i8, ptr %109, i64 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 0
  store i64 8, ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %110, i32 0, i32 2
  store ptr %108, ptr %113, align 8
  br label %114

114:                                              ; preds = %106, %.noexc3
  %115 = phi ptr [ %110, %.noexc3 ], [ %105, %106 ]
  br label %116

116:                                              ; preds = %114
  br label %124

117:                                              ; preds = %83
  %118 = load ptr, ptr %15, align 8
  %119 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

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

130:                                              ; preds = %.noexc5, %129
  %131 = phi ptr [ %128, %129 ], [ %75, %.noexc5 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %.noexc9, %132
  %134 = phi ptr [ %131, %132 ], [ %66, %.noexc9 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc10, %135
  %137 = phi ptr [ %134, %135 ], [ %57, %.noexc10 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc12, %138
  %140 = phi ptr [ %137, %138 ], [ %48, %.noexc12 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc13, %141
  %143 = phi ptr [ %140, %141 ], [ %39, %.noexc13 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc7, %144
  %146 = phi ptr [ %143, %144 ], [ %30, %.noexc7 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc8, %147
  %149 = phi ptr [ %146, %147 ], [ %21, %.noexc8 ]
  br label %150

150:                                              ; preds = %148
  store ptr %149, ptr %2, align 8
  br i1 false, label %151, label %156

151:                                              ; preds = %150
  call void @__cxa_end_catch()
  br label %152

152:                                              ; preds = %11, %151
  ret void

153:                                              ; preds = %154, %.noexc2
  unreachable

154:                                              ; preds = %9
  %155 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15252316600109684411, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 101, ptr null)
  br label %153

156:                                              ; preds = %150
  %157 = invoke i32 @fflush(ptr null)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %156
  invoke void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8798840678818865255, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 103, ptr null)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  br label %153

endcatch.resume:                                  ; preds = %36, %45, %12, %54, %63, %18, %27, %97, %72, %117, %107, %.noexc1, %156, %95
  %158 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %158
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
define void @thrown() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = call ptr @mk(double 1.000000e+00)
  store ptr %5, ptr %4, align 8
  store i32 0, ptr %3, align 4
  %6 = load ptr, ptr %4, align 8
  %7 = invoke ptr @mk(double 2.000000e+00)
          to label %8 unwind label %11

8:                                                ; preds = %0
  store ptr %7, ptr %4, align 8
  %9 = invoke double @gThrows(ptr %6, double 0.000000e+00)
          to label %10 unwind label %11

10:                                               ; preds = %8
  br label %154

11:                                               ; preds = %8, %0
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %11
  %16 = icmp eq ptr %15, @_ZTIi
  br i1 %16, label %17, label %24

17:                                               ; preds = %.noexc9
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

24:                                               ; preds = %.noexc9
  %25 = icmp eq ptr %15, @_ZTIj
  br i1 %25, label %26, label %33

26:                                               ; preds = %24
  %27 = load i32, ptr %14, align 4
  %28 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %26
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
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %35
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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %44
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
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %53
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
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %62
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
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %94
  %95 = load ptr, ptr %1, align 8
  br label %103

96:                                               ; preds = %90
  %97 = load ptr, ptr %14, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %99, i32 0, i32 2
  store ptr %97, ptr %102, align 8
  br label %103

103:                                              ; preds = %.noexc3, %.noexc11
  %104 = phi ptr [ %99, %.noexc11 ], [ %95, %.noexc3 ]
  br label %105

105:                                              ; preds = %103
  br label %113

106:                                              ; preds = %85
  %107 = load ptr, ptr %14, align 8
  %108 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %106
  %109 = getelementptr i8, ptr %108, i64 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 0
  store i64 8, ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %111, align 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %109, i32 0, i32 2
  store ptr %107, ptr %112, align 8
  br label %113

113:                                              ; preds = %105, %.noexc8
  %114 = phi ptr [ %109, %.noexc8 ], [ %104, %105 ]
  br label %115

115:                                              ; preds = %113
  br label %123

116:                                              ; preds = %82
  %117 = load ptr, ptr %14, align 8
  %118 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %116
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 0
  store i64 8, ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %119, i32 0, i32 2
  store ptr %117, ptr %122, align 8
  br label %123

123:                                              ; preds = %115, %.noexc1
  %124 = phi ptr [ %119, %.noexc1 ], [ %114, %115 ]
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

132:                                              ; preds = %.noexc5, %131
  %133 = phi ptr [ %130, %131 ], [ %65, %.noexc5 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %.noexc, %134
  %136 = phi ptr [ %133, %134 ], [ %56, %.noexc ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %.noexc2, %137
  %139 = phi ptr [ %136, %137 ], [ %47, %.noexc2 ]
  br label %140

140:                                              ; preds = %138
  br label %141

141:                                              ; preds = %.noexc4, %140
  %142 = phi ptr [ %139, %140 ], [ %38, %.noexc4 ]
  br label %143

143:                                              ; preds = %141
  br label %144

144:                                              ; preds = %.noexc10, %143
  %145 = phi ptr [ %142, %143 ], [ %29, %.noexc10 ]
  br label %146

146:                                              ; preds = %144
  br label %147

147:                                              ; preds = %.noexc12, %146
  %148 = phi ptr [ %145, %146 ], [ %20, %.noexc12 ]
  br label %149

149:                                              ; preds = %147
  store ptr %148, ptr %2, align 8
  %150 = load ptr, ptr %2, align 8
  %151 = invoke double @"___unbox<!ts.number>"(ptr %150)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %149
  %152 = fptosi double %151 to i64
  %153 = trunc i64 %152 to i32
  store i32 %153, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %154

154:                                              ; preds = %10, %.noexc6
  %155 = call i64 @churn()
  %156 = load i32, ptr %3, align 4
  %157 = icmp eq i32 %156, 10
  br i1 %157, label %158, label %163

158:                                              ; preds = %154
  %159 = load ptr, ptr %4, align 8
  %160 = getelementptr %V, ptr %159, i32 0, i32 2
  %161 = load double, ptr %160, align 8
  %162 = fcmp oeq double %161, 2.000000e+00
  br label %163

163:                                              ; preds = %158, %154
  %164 = phi i1 [ %162, %158 ], [ %157, %154 ]
  br i1 %164, label %165, label %167

165:                                              ; preds = %163
  ret void

166:                                              ; preds = %167
  unreachable

167:                                              ; preds = %163
  %168 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_234323632491049957, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 117, ptr null)
  br label %166

endcatch.resume:                                  ; preds = %17, %96, %26, %11, %106, %71, %149, %62, %35, %94, %44, %116, %53
  %169 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %169
}

; Function Attrs: mustprogress
define void @param(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @mk(double 2.000000e+00)
  store ptr %4, ptr %2, align 8
  %5 = call double @g(ptr %3, double 0.000000e+00)
  %6 = fcmp oeq double %5, 1.000000e+01
  br i1 %6, label %7, label %9

7:                                                ; preds = %1
  ret void

8:                                                ; preds = %9
  unreachable

9:                                                ; preds = %1
  %10 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4443436482793291723, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 121, ptr null)
  br label %8
}

; Function Attrs: mustprogress
define i32 @.f_captured..afL126C17FH3593357200(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = call ptr @mk(double 2.000000e+00)
  store ptr %5, ptr %4, align 8
  store i32 0, ptr %2, align 4
  %6 = load i32, ptr %2, align 4
  ret i32 %6
}

; Function Attrs: mustprogress
define void @captured() #2 {
  %1 = alloca <{ i64, { ptr } }>, align 16
  %2 = call ptr @mk(double 1.000000e+00)
  %3 = call ptr @GC_malloc(i64 16)
  %4 = getelementptr i8, ptr %3, i64 8
  store ptr %2, ptr %4, align 8
  store i64 -1, ptr %1, align 8
  %5 = getelementptr i8, ptr %1, i64 8
  store { ptr } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %4, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_captured..afL126C17FH3593357200, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_1071659150439645108_object, i64 40), 2
  %9 = load ptr, ptr %4, align 8
  %10 = call i32 @.f_captured..afL126C17FH3593357200(ptr %5)
  %11 = sitofp i32 %10 to double
  %12 = call double @g(ptr %9, double %11)
  %13 = fcmp oeq double %12, 1.000000e+01
  br i1 %13, label %14, label %21

14:                                               ; preds = %0
  %15 = load ptr, ptr %4, align 8
  %16 = getelementptr %V, ptr %15, i32 0, i32 2
  %17 = load double, ptr %16, align 8
  %18 = fcmp oeq double %17, 2.000000e+00
  br i1 %18, label %19, label %23

19:                                               ; preds = %14
  ret void

20:                                               ; preds = %23, %21
  unreachable

21:                                               ; preds = %0
  %22 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9712415665295363832, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 130, ptr null)
  br label %20

23:                                               ; preds = %14
  %24 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12745201741415506076, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4377004417030018898, i64 8), i32 131, ptr null)
  br label %20
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @local()
  call void @localTwice()
  call void @localAssignment()
  call void @inLoop()
  call void @field()
  call void @element()
  call void @conditional()
  call void @inTry()
  call void @thrown()
  %1 = call ptr @mk(double 1.000000e+00)
  call void @param(ptr %1)
  call void @captured()
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

