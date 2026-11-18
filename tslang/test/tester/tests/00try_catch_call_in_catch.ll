; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Fail = type { ptr, ptr }

$Fail..size = comdat any

$_ZTSP4Fail = comdat any

$_ZTIP4Fail = comdat any

$_ZTS4Fail = comdat any

$_ZTI4Fail = comdat any

@s_1841804593947372813 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFail\00", align 8
@Fail..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1841804593947372813, i64 8)
@Fail..size = global i64 16, comdat
@Fail..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Fail..instanceOf, ptr @Fail..new, ptr @Fail..rtti, ptr @Fail..size }
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP4Fail = linkonce_odr constant [7 x i8] c"P4Fail\00", comdat
@_ZTIP4Fail = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP4Fail, i32 1073741824, ptr @_ZTI4Fail, ptr @.eh.copy.box.P4Fail }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS4Fail = linkonce_odr constant [6 x i8] c"4Fail\00", comdat
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_3523764787506715604 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFouter\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_11860658356491264189 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFsecond\00", align 8
@s_9932668635479743809 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfirst\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_3263991042789642887 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner\00", align 8
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
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_3034936240246819029 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa number\00", align 8
@f_13214604348754279962 = internal constant [115 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch_call_in_catch.ts\00", align 8
@m_924000406299335852 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class\00", align 8
@m_15968135012252376645 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa string\00", align 8
@m_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@_ZTI4Fail = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS4Fail }, comdat
@_ZTIPc = external constant ptr
@td_16282675435063983936_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_16282675435063983936, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__cxa_rethrow()

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

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

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

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

define internal void @tsrel_16282675435063983936(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Fail, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Fail..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Fail..rtti, align 8
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
define ptr @Fail..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Fail, ptr %3, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Fail.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Fail, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @throwNumber(double %0) #2 {
  %2 = alloca double, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store double %3, ptr %4, align 8
  call void @__cxa_throw(ptr %4, ptr @_ZTId, ptr null)
  br label %5

5:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P4Fail(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_16282675435063983936_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define void @throwFail(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = call ptr @GC_malloc(i64 24)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %Fail, ptr %4, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %5, align 8
  %6 = load ptr, ptr %2, align 8
  call void @Fail.constructor(ptr %4, ptr %6)
  %7 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %4, ptr %7, align 8
  call void @__cxa_throw(ptr %7, ptr @_ZTIP4Fail, ptr null)
  br label %8

8:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define void @throwString(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %3, ptr %4, align 8
  call void @__cxa_throw(ptr %4, ptr @_ZTIPc, ptr null)
  br label %5

5:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca double, align 8
  %9 = alloca double, align 8
  %10 = alloca double, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  store double 0.000000e+00, ptr %10, align 8
  %13 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %13, align 4
  invoke void @__cxa_throw(ptr %13, ptr @_ZTIi, ptr null)
          to label %405 unwind label %14

14:                                               ; preds = %0
  %15 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %16 = extractvalue { ptr, i32 } %15, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %18 = icmp eq i32 %16, %17
  %19 = extractvalue { ptr, i32 } %15, 1
  %20 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %21 = icmp eq i32 %19, %20
  %22 = extractvalue { ptr, i32 } %15, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %24 = icmp eq i32 %22, %23
  %25 = extractvalue { ptr, i32 } %15, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %27 = icmp eq i32 %25, %26
  %28 = or i1 %18, %21
  %29 = or i1 %28, %24
  %30 = or i1 %29, %27
  br i1 %30, label %34, label %31

31:                                               ; preds = %14
  %32 = extractvalue { ptr, i32 } %15, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad6

34:                                               ; preds = %14
  %35 = extractvalue { ptr, i32 } %15, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  %37 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %34
  %38 = icmp eq ptr %37, @_ZTIi
  br i1 %38, label %39, label %42

39:                                               ; preds = %.noexc
  %40 = load i32, ptr %36, align 4
  %41 = sitofp i32 %40 to double
  br label %60

42:                                               ; preds = %.noexc
  %43 = icmp eq ptr %37, @_ZTIj
  br i1 %43, label %44, label %47

44:                                               ; preds = %42
  %45 = load i32, ptr %36, align 4
  %46 = uitofp i32 %45 to double
  br label %57

47:                                               ; preds = %42
  %48 = icmp eq ptr %37, @_ZTIf
  br i1 %48, label %49, label %52

49:                                               ; preds = %47
  %50 = load float, ptr %36, align 4
  %51 = fpext float %50 to double
  br label %54

52:                                               ; preds = %47
  %53 = load double, ptr %36, align 8
  br label %54

54:                                               ; preds = %49, %52
  %55 = phi double [ %53, %52 ], [ %51, %49 ]
  br label %56

56:                                               ; preds = %54
  br label %57

57:                                               ; preds = %44, %56
  %58 = phi double [ %55, %56 ], [ %46, %44 ]
  br label %59

59:                                               ; preds = %57
  br label %60

60:                                               ; preds = %39, %59
  %61 = phi double [ %58, %59 ], [ %41, %39 ]
  br label %62

62:                                               ; preds = %60
  store double %61, ptr %8, align 8
  %63 = load double, ptr %8, align 8
  %64 = invoke ptr @GC_malloc_atomic(i64 40)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %62
  %65 = getelementptr i8, ptr %64, i64 8
  invoke void @__ts_number_to_string(ptr %65, double %63)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %.noexc18
  %66 = call ptr @llvm.stacksave.p0()
  %67 = icmp eq ptr %65, null
  %68 = select i1 %67, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %65
  %69 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8))
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %.noexc19
  %70 = add i64 1, %69
  %71 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %.noexc20
  %72 = add i64 %70, %71
  %73 = invoke i64 @strlen(ptr %68)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %.noexc21
  %74 = add i64 %72, %73
  %75 = alloca i8, i64 %74, align 1
  %76 = getelementptr i8, ptr %75, i64 0
  %77 = sub i64 %74, 0
  %78 = invoke ptr @__memcpy_chk(ptr %76, ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8), i64 %69, i64 %77)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %.noexc22
  %79 = add i64 0, %69
  %80 = getelementptr i8, ptr %75, i64 %79
  %81 = sub i64 %74, %79
  %82 = invoke ptr @__memcpy_chk(ptr %80, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %71, i64 %81)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %.noexc23
  %83 = add i64 %79, %71
  %84 = getelementptr i8, ptr %75, i64 %83
  %85 = sub i64 %74, %83
  %86 = invoke ptr @__memcpy_chk(ptr %84, ptr %68, i64 %73, i64 %85)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %.noexc24
  %87 = add i64 %83, %73
  %88 = getelementptr i8, ptr %75, i64 %87
  store i8 0, ptr %88, align 1
  invoke void @puts(ptr %75)
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %.noexc25
  call void @llvm.stackrestore.p0(ptr %66)
  invoke void @throwNumber(double 2.500000e+00)
          to label %89 unwind label %endcatch.pad6

89:                                               ; preds = %.noexc26
  call void @__cxa_end_catch()
  br label %142

endcatch.pad6:                                    ; preds = %31, %.noexc26
  %90 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught5

91:                                               ; No predecessors!
  %92 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught5

.caught5:                                         ; preds = %endcatch.pad6, %91
  %93 = phi { ptr, i32 } [ %92, %91 ], [ %90, %endcatch.pad6 ]
  %94 = extractvalue { ptr, i32 } %93, 1
  %95 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %96 = icmp eq i32 %94, %95
  %97 = extractvalue { ptr, i32 } %93, 1
  %98 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %99 = icmp eq i32 %97, %98
  %100 = extractvalue { ptr, i32 } %93, 1
  %101 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %102 = icmp eq i32 %100, %101
  %103 = extractvalue { ptr, i32 } %93, 1
  %104 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %105 = icmp eq i32 %103, %104
  %106 = or i1 %96, %99
  %107 = or i1 %106, %102
  %108 = or i1 %107, %105
  br i1 %108, label %112, label %109

109:                                              ; preds = %.caught5
  %110 = extractvalue { ptr, i32 } %93, 0
  %111 = call ptr @__cxa_begin_catch(ptr %110)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad2

112:                                              ; preds = %.caught5
  %113 = extractvalue { ptr, i32 } %93, 0
  %114 = call ptr @__cxa_begin_catch(ptr %113)
  %115 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc36 unwind label %endcatch.resume

.noexc36:                                         ; preds = %112
  %116 = icmp eq ptr %115, @_ZTIi
  br i1 %116, label %117, label %120

117:                                              ; preds = %.noexc36
  %118 = load i32, ptr %114, align 4
  %119 = sitofp i32 %118 to double
  br label %138

120:                                              ; preds = %.noexc36
  %121 = icmp eq ptr %115, @_ZTIj
  br i1 %121, label %122, label %125

122:                                              ; preds = %120
  %123 = load i32, ptr %114, align 4
  %124 = uitofp i32 %123 to double
  br label %135

125:                                              ; preds = %120
  %126 = icmp eq ptr %115, @_ZTIf
  br i1 %126, label %127, label %130

127:                                              ; preds = %125
  %128 = load float, ptr %114, align 4
  %129 = fpext float %128 to double
  br label %132

130:                                              ; preds = %125
  %131 = load double, ptr %114, align 8
  br label %132

132:                                              ; preds = %127, %130
  %133 = phi double [ %131, %130 ], [ %129, %127 ]
  br label %134

134:                                              ; preds = %132
  br label %135

135:                                              ; preds = %122, %134
  %136 = phi double [ %133, %134 ], [ %124, %122 ]
  br label %137

137:                                              ; preds = %135
  br label %138

138:                                              ; preds = %117, %137
  %139 = phi double [ %136, %137 ], [ %119, %117 ]
  br label %140

140:                                              ; preds = %138
  store double %139, ptr %9, align 8
  %141 = load double, ptr %9, align 8
  store double %141, ptr %10, align 8
  call void @__cxa_end_catch()
  br label %142

142:                                              ; preds = %89, %140
  %143 = load double, ptr %10, align 8
  %144 = fcmp oeq double %143, 2.500000e+00
  br i1 %144, label %145, label %406

145:                                              ; preds = %142
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %12, align 8
  invoke void @throwFail(ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8))
          to label %146 unwind label %147

146:                                              ; preds = %145
  br label %201

147:                                              ; preds = %145
  %148 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %149 = extractvalue { ptr, i32 } %148, 1
  %150 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %151 = icmp eq i32 %149, %150
  br i1 %151, label %155, label %152

152:                                              ; preds = %147
  %153 = extractvalue { ptr, i32 } %148, 0
  %154 = call ptr @__cxa_begin_catch(ptr %153)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad4

155:                                              ; preds = %147
  %156 = extractvalue { ptr, i32 } %148, 0
  %157 = call ptr @__cxa_begin_catch(ptr %156)
  store ptr %157, ptr %6, align 8
  %158 = load ptr, ptr %6, align 8
  %159 = getelementptr %Fail, ptr %158, i32 0, i32 1
  %160 = load ptr, ptr %159, align 8
  %161 = call ptr @llvm.stacksave.p0()
  %162 = icmp eq ptr %160, null
  %163 = select i1 %162, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %160
  %164 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8))
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %155
  %165 = add i64 1, %164
  %166 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %.noexc11
  %167 = add i64 %165, %166
  %168 = invoke i64 @strlen(ptr %163)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %.noexc12
  %169 = add i64 %167, %168
  %170 = alloca i8, i64 %169, align 1
  %171 = getelementptr i8, ptr %170, i64 0
  %172 = sub i64 %169, 0
  %173 = invoke ptr @__memcpy_chk(ptr %171, ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8), i64 %164, i64 %172)
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %.noexc13
  %174 = add i64 0, %164
  %175 = getelementptr i8, ptr %170, i64 %174
  %176 = sub i64 %169, %174
  %177 = invoke ptr @__memcpy_chk(ptr %175, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %166, i64 %176)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %.noexc14
  %178 = add i64 %174, %166
  %179 = getelementptr i8, ptr %170, i64 %178
  %180 = sub i64 %169, %178
  %181 = invoke ptr @__memcpy_chk(ptr %179, ptr %163, i64 %168, i64 %180)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %.noexc15
  %182 = add i64 %178, %168
  %183 = getelementptr i8, ptr %170, i64 %182
  store i8 0, ptr %183, align 1
  invoke void @puts(ptr %170)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %.noexc16
  call void @llvm.stackrestore.p0(ptr %161)
  invoke void @throwFail(ptr getelementptr inbounds nuw (i8, ptr @s_11860658356491264189, i64 8))
          to label %184 unwind label %endcatch.pad4

184:                                              ; preds = %.noexc17
  call void @__cxa_end_catch()
  br label %201

endcatch.pad4:                                    ; preds = %152, %.noexc17
  %185 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

186:                                              ; No predecessors!
  %187 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %186
  %188 = phi { ptr, i32 } [ %187, %186 ], [ %185, %endcatch.pad4 ]
  %189 = extractvalue { ptr, i32 } %188, 1
  %190 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %191 = icmp eq i32 %189, %190
  br i1 %191, label %195, label %192

192:                                              ; preds = %.caught3
  %193 = extractvalue { ptr, i32 } %188, 0
  %194 = call ptr @__cxa_begin_catch(ptr %193)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad2

195:                                              ; preds = %.caught3
  %196 = extractvalue { ptr, i32 } %188, 0
  %197 = call ptr @__cxa_begin_catch(ptr %196)
  store ptr %197, ptr %7, align 8
  %198 = load ptr, ptr %7, align 8
  %199 = getelementptr %Fail, ptr %198, i32 0, i32 1
  %200 = load ptr, ptr %199, align 8
  store ptr %200, ptr %12, align 8
  call void @__cxa_end_catch()
  br label %201

201:                                              ; preds = %146, %184, %195
  %202 = load ptr, ptr %12, align 8
  %203 = ptrtoint ptr %202 to i64
  %204 = icmp ne i64 %203, 0
  %205 = zext i1 %204 to i32
  %206 = and i32 %205, 1
  %207 = icmp ne i32 %206, 0
  br i1 %207, label %208, label %211

208:                                              ; preds = %201
  %209 = call i32 @strcmp(ptr %202, ptr getelementptr inbounds nuw (i8, ptr @s_11860658356491264189, i64 8))
  %210 = icmp eq i32 %209, 0
  br label %213

211:                                              ; preds = %201
  %212 = icmp eq i64 %203, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11860658356491264189, i64 8) to i64)
  br label %213

213:                                              ; preds = %208, %211
  %214 = phi i1 [ %212, %211 ], [ %210, %208 ]
  br label %215

215:                                              ; preds = %213
  br i1 %214, label %216, label %408

216:                                              ; preds = %215
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %11, align 8
  %217 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), ptr %217, align 8
  invoke void @__cxa_throw(ptr %217, ptr @_ZTIPc, ptr null)
          to label %405 unwind label %218

218:                                              ; preds = %216
  %219 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %220 = extractvalue { ptr, i32 } %219, 1
  %221 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %222 = icmp eq i32 %220, %221
  br i1 %222, label %226, label %223

223:                                              ; preds = %218
  %224 = extractvalue { ptr, i32 } %219, 0
  %225 = call ptr @__cxa_begin_catch(ptr %224)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad8

226:                                              ; preds = %218
  %227 = extractvalue { ptr, i32 } %219, 0
  %228 = call ptr @__cxa_begin_catch(ptr %227)
  store ptr %228, ptr %4, align 8
  invoke void @throwString(ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %229 unwind label %endcatch.pad8

229:                                              ; preds = %226
  call void @__cxa_end_catch()
  br label %268

endcatch.pad8:                                    ; preds = %226, %223
  %230 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught7

231:                                              ; No predecessors!
  %232 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught7

.caught7:                                         ; preds = %endcatch.pad8, %231
  %233 = phi { ptr, i32 } [ %232, %231 ], [ %230, %endcatch.pad8 ]
  %234 = extractvalue { ptr, i32 } %233, 1
  %235 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %236 = icmp eq i32 %234, %235
  br i1 %236, label %240, label %237

237:                                              ; preds = %.caught7
  %238 = extractvalue { ptr, i32 } %233, 0
  %239 = call ptr @__cxa_begin_catch(ptr %238)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad2

240:                                              ; preds = %.caught7
  %241 = extractvalue { ptr, i32 } %233, 0
  %242 = call ptr @__cxa_begin_catch(ptr %241)
  store ptr %242, ptr %5, align 8
  %243 = load ptr, ptr %5, align 8
  %244 = call ptr @llvm.stacksave.p0()
  %245 = icmp eq ptr %243, null
  %246 = select i1 %245, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %243
  %247 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3523764787506715604, i64 8))
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %240
  %248 = add i64 1, %247
  %249 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc29 unwind label %endcatch.resume

.noexc29:                                         ; preds = %.noexc28
  %250 = add i64 %248, %249
  %251 = invoke i64 @strlen(ptr %246)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %.noexc29
  %252 = add i64 %250, %251
  %253 = alloca i8, i64 %252, align 1
  %254 = getelementptr i8, ptr %253, i64 0
  %255 = sub i64 %252, 0
  %256 = invoke ptr @__memcpy_chk(ptr %254, ptr getelementptr inbounds nuw (i8, ptr @s_3523764787506715604, i64 8), i64 %247, i64 %255)
          to label %.noexc31 unwind label %endcatch.resume

.noexc31:                                         ; preds = %.noexc30
  %257 = add i64 0, %247
  %258 = getelementptr i8, ptr %253, i64 %257
  %259 = sub i64 %252, %257
  %260 = invoke ptr @__memcpy_chk(ptr %258, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %249, i64 %259)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %.noexc31
  %261 = add i64 %257, %249
  %262 = getelementptr i8, ptr %253, i64 %261
  %263 = sub i64 %252, %261
  %264 = invoke ptr @__memcpy_chk(ptr %262, ptr %246, i64 %251, i64 %263)
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %.noexc32
  %265 = add i64 %261, %251
  %266 = getelementptr i8, ptr %253, i64 %265
  store i8 0, ptr %266, align 1
  invoke void @puts(ptr %253)
          to label %.noexc34 unwind label %endcatch.resume

.noexc34:                                         ; preds = %.noexc33
  call void @llvm.stackrestore.p0(ptr %244)
  %267 = load ptr, ptr %5, align 8
  store ptr %267, ptr %11, align 8
  call void @__cxa_end_catch()
  br label %268

268:                                              ; preds = %229, %.noexc34
  %269 = load ptr, ptr %11, align 8
  %270 = ptrtoint ptr %269 to i64
  %271 = icmp ne i64 %270, 0
  %272 = zext i1 %271 to i32
  %273 = and i32 %272, 1
  %274 = icmp ne i32 %273, 0
  br i1 %274, label %275, label %278

275:                                              ; preds = %268
  %276 = call i32 @strcmp(ptr %269, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
  %277 = icmp eq i32 %276, 0
  br label %280

278:                                              ; preds = %268
  %279 = icmp eq i64 %270, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8) to i64)
  br label %280

280:                                              ; preds = %275, %278
  %281 = phi i1 [ %279, %278 ], [ %277, %275 ]
  br label %282

282:                                              ; preds = %280
  br i1 %281, label %283, label %410

283:                                              ; preds = %282
  store i1 false, ptr %3, align 1
  store double 0.000000e+00, ptr %10, align 8
  %284 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %284, align 4
  invoke void @__cxa_throw(ptr %284, ptr @_ZTIi, ptr null)
          to label %405 unwind label %285

285:                                              ; preds = %283
  %286 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %287 = extractvalue { ptr, i32 } %286, 1
  %288 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %289 = icmp eq i32 %287, %288
  %290 = extractvalue { ptr, i32 } %286, 1
  %291 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %292 = icmp eq i32 %290, %291
  %293 = extractvalue { ptr, i32 } %286, 1
  %294 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %295 = icmp eq i32 %293, %294
  %296 = extractvalue { ptr, i32 } %286, 1
  %297 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %298 = icmp eq i32 %296, %297
  %299 = or i1 %289, %292
  %300 = or i1 %299, %295
  %301 = or i1 %300, %298
  br i1 %301, label %305, label %302

302:                                              ; preds = %285
  %303 = extractvalue { ptr, i32 } %286, 0
  %304 = call ptr @__cxa_begin_catch(ptr %303)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad

305:                                              ; preds = %285
  %306 = extractvalue { ptr, i32 } %286, 0
  %307 = call ptr @__cxa_begin_catch(ptr %306)
  %308 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc35 unwind label %endcatch.resume

.noexc35:                                         ; preds = %305
  %309 = icmp eq ptr %308, @_ZTIi
  br i1 %309, label %310, label %313

310:                                              ; preds = %.noexc35
  %311 = load i32, ptr %307, align 4
  %312 = sitofp i32 %311 to double
  br label %331

313:                                              ; preds = %.noexc35
  %314 = icmp eq ptr %308, @_ZTIj
  br i1 %314, label %315, label %318

315:                                              ; preds = %313
  %316 = load i32, ptr %307, align 4
  %317 = uitofp i32 %316 to double
  br label %328

318:                                              ; preds = %313
  %319 = icmp eq ptr %308, @_ZTIf
  br i1 %319, label %320, label %323

320:                                              ; preds = %318
  %321 = load float, ptr %307, align 4
  %322 = fpext float %321 to double
  br label %325

323:                                              ; preds = %318
  %324 = load double, ptr %307, align 8
  br label %325

325:                                              ; preds = %320, %323
  %326 = phi double [ %324, %323 ], [ %322, %320 ]
  br label %327

327:                                              ; preds = %325
  br label %328

328:                                              ; preds = %315, %327
  %329 = phi double [ %326, %327 ], [ %317, %315 ]
  br label %330

330:                                              ; preds = %328
  br label %331

331:                                              ; preds = %310, %330
  %332 = phi double [ %329, %330 ], [ %312, %310 ]
  br label %333

333:                                              ; preds = %331
  store double %332, ptr %1, align 8
  invoke void @throwNumber(double 3.000000e+00)
          to label %334 unwind label %endcatch.pad

334:                                              ; preds = %333
  call void @__cxa_end_catch()
  store i1 true, ptr %3, align 1
  br label %393

endcatch.pad:                                     ; preds = %333, %302
  %335 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

336:                                              ; No predecessors!
  %337 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %336
  %338 = phi { ptr, i32 } [ %337, %336 ], [ %335, %endcatch.pad ]
  %339 = extractvalue { ptr, i32 } %338, 0
  %340 = call ptr @__cxa_begin_catch(ptr %339)
  store i1 true, ptr %3, align 1
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad10

endcatch.pad10:                                   ; preds = %.caught
  %341 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught9

342:                                              ; No predecessors!
  %343 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught9

.caught9:                                         ; preds = %endcatch.pad10, %342
  %344 = phi { ptr, i32 } [ %343, %342 ], [ %341, %endcatch.pad10 ]
  %345 = extractvalue { ptr, i32 } %344, 1
  %346 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %347 = icmp eq i32 %345, %346
  %348 = extractvalue { ptr, i32 } %344, 1
  %349 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %350 = icmp eq i32 %348, %349
  %351 = extractvalue { ptr, i32 } %344, 1
  %352 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %353 = icmp eq i32 %351, %352
  %354 = extractvalue { ptr, i32 } %344, 1
  %355 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %356 = icmp eq i32 %354, %355
  %357 = or i1 %347, %350
  %358 = or i1 %357, %353
  %359 = or i1 %358, %356
  br i1 %359, label %363, label %360

360:                                              ; preds = %.caught9
  %361 = extractvalue { ptr, i32 } %344, 0
  %362 = call ptr @__cxa_begin_catch(ptr %361)
  invoke void @__cxa_rethrow()
          to label %405 unwind label %endcatch.pad2

363:                                              ; preds = %.caught9
  %364 = extractvalue { ptr, i32 } %344, 0
  %365 = call ptr @__cxa_begin_catch(ptr %364)
  %366 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %363
  %367 = icmp eq ptr %366, @_ZTIi
  br i1 %367, label %368, label %371

368:                                              ; preds = %.noexc27
  %369 = load i32, ptr %365, align 4
  %370 = sitofp i32 %369 to double
  br label %389

371:                                              ; preds = %.noexc27
  %372 = icmp eq ptr %366, @_ZTIj
  br i1 %372, label %373, label %376

373:                                              ; preds = %371
  %374 = load i32, ptr %365, align 4
  %375 = uitofp i32 %374 to double
  br label %386

376:                                              ; preds = %371
  %377 = icmp eq ptr %366, @_ZTIf
  br i1 %377, label %378, label %381

378:                                              ; preds = %376
  %379 = load float, ptr %365, align 4
  %380 = fpext float %379 to double
  br label %383

381:                                              ; preds = %376
  %382 = load double, ptr %365, align 8
  br label %383

383:                                              ; preds = %378, %381
  %384 = phi double [ %382, %381 ], [ %380, %378 ]
  br label %385

385:                                              ; preds = %383
  br label %386

386:                                              ; preds = %373, %385
  %387 = phi double [ %384, %385 ], [ %375, %373 ]
  br label %388

388:                                              ; preds = %386
  br label %389

389:                                              ; preds = %368, %388
  %390 = phi double [ %387, %388 ], [ %370, %368 ]
  br label %391

391:                                              ; preds = %389
  store double %390, ptr %2, align 8
  %392 = load double, ptr %2, align 8
  store double %392, ptr %10, align 8
  call void @__cxa_end_catch()
  br label %393

393:                                              ; preds = %334, %391
  %394 = load i1, ptr %3, align 1
  br i1 %394, label %395, label %398

395:                                              ; preds = %393
  %396 = load double, ptr %10, align 8
  %397 = fcmp oeq double %396, 3.000000e+00
  br label %398

398:                                              ; preds = %395, %393
  %399 = phi i1 [ %397, %395 ], [ %394, %393 ]
  br i1 %399, label %400, label %412

400:                                              ; preds = %398
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

endcatch.pad2:                                    ; preds = %237, %192, %109, %360
  %401 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught1

402:                                              ; No predecessors!
  %403 = landingpad { ptr, i32 }
          cleanup
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %402
  %404 = phi { ptr, i32 } [ %403, %402 ], [ %401, %endcatch.pad2 ]
  resume { ptr, i32 } %404

405:                                              ; preds = %412, %360, %.caught, %302, %283, %410, %237, %223, %216, %408, %192, %152, %406, %109, %31, %0
  unreachable

406:                                              ; preds = %142
  %407 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3034936240246819029, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13214604348754279962, i64 8), i32 20, ptr null)
  br label %405

408:                                              ; preds = %215
  %409 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_924000406299335852, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13214604348754279962, i64 8), i32 28, ptr null)
  br label %405

410:                                              ; preds = %282
  %411 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15968135012252376645, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13214604348754279962, i64 8), i32 37, ptr null)
  br label %405

412:                                              ; preds = %398
  %413 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18175057562723240456, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13214604348754279962, i64 8), i32 47, ptr null)
  br label %405

endcatch.resume:                                  ; preds = %112, %305, %.noexc33, %.noexc32, %.noexc31, %.noexc30, %.noexc29, %.noexc28, %240, %363, %.noexc25, %.noexc24, %.noexc23, %.noexc22, %.noexc21, %.noexc20, %.noexc19, %.noexc18, %62, %.noexc16, %.noexc15, %.noexc14, %.noexc13, %.noexc12, %.noexc11, %155, %34
  %414 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %414
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

