; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%W = type { ptr, double }

$"___check_tuple<!ts.tuple<{\22n\22,!ts.number}>>" = comdat any

$"___check_tuple<!ts.tuple<{\22p\22,!ts.tuple<{\22n\22,!ts.number}>},{\22s\22,!ts.string}>>" = comdat any

$W..size = comdat any

@s_12638237778090539702 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFW\00", align 8
@s_11548266869007507528 = internal constant [84 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the object's fields are not those of type '{n:number}'\00", align 8
@s_15305194920199280056 = internal constant [77 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the value is not an object of type '{n:number}'\00", align 8
@s_10458455444534216913 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtuple\00", align 8
@s_1712190022454804100 = internal constant [98 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the object's fields are not those of type '{p:{n:number}, s:string}'\00", align 8
@s_187150928315272468 = internal constant [91 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the value is not an object of type '{p:{n:number}, s:string}'\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_8982223085663264368 = internal constant [104 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00any_to_tuple.ts\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@W..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638237778090539702, i64 8)
@W..size = global i64 16, comdat
@W..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @W..instanceOf, ptr @W..new, ptr @W..rtti, ptr @W..size }
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@td_4617940393880614542_tuple = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 6, i32 1094197076, ptr null, ptr null, ptr null, i64 -1 }, [6 x i8] c"tuple\00" }, align 8
@td_16546078158687492637_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_16546078158687492637, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_11850163807121388224_tuple = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 6, i32 1289496979, ptr null, ptr null, ptr null, i64 -1 }, [6 x i8] c"tuple\00" }, align 8
@td_6300196387968623321_tuple = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 6, i32 -1471841672, ptr null, ptr null, ptr null, i64 -1 }, [6 x i8] c"tuple\00" }, align 8
@td_12845129401128444698_tuple = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 6, i32 1094197076, ptr null, ptr null, ptr null, i64 -1 }, [6 x i8] c"tuple\00" }, align 8
@td_4706355282782734589_tuple = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 6, i32 640346249, ptr @tsrel_4706355282782734589, ptr @tsret_4706355282782734589, ptr null, i64 -1 }, [6 x i8] c"tuple\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @__cxa_end_catch()

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @puts(ptr)

define internal void @tsret_4706355282782734589(ptr %0) {
  %2 = getelementptr { { double }, ptr }, ptr %0, i32 0, i32 1
  call void @__tslang_retain_block_slot(ptr %2)
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

define internal void @tsrel_4706355282782734589(ptr %0) {
  %2 = getelementptr { { double }, ptr }, ptr %0, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %2)
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

define internal void @tsrel_16546078158687492637(ptr %0) {
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

declare void @ts.internal.eh_try(i32)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @W..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @W..rtti, align 8
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
define ptr @W..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %W, ptr %3, i32 0, i32 0
  store ptr @W..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @W.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %W, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"___check_tuple<!ts.tuple<{\22n\22,!ts.number}>>"(ptr %0, i1 %1) #1 comdat {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i1 %1, ptr %3, align 1
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr { i64, ptr, i8 }, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %2
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8))
  %15 = icmp ne i32 %14, 0
  br label %18

16:                                               ; preds = %2
  %17 = icmp ne i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %23

21:                                               ; preds = %20
  %22 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_15305194920199280056, i64 8), ptr %22, align 8
  call void @__cxa_throw(ptr %22, ptr @_ZTIPc, ptr null)
  br label %29

23:                                               ; preds = %20
  %24 = load i1, ptr %3, align 1
  %25 = xor i1 %24, true
  br i1 %25, label %26, label %28

26:                                               ; preds = %23
  %27 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_11548266869007507528, i64 8), ptr %27, align 8
  call void @__cxa_throw(ptr %27, ptr @_ZTIPc, ptr null)
  br label %29

28:                                               ; preds = %23
  ret void

29:                                               ; preds = %21, %26
  unreachable
}

; Function Attrs: mustprogress
define { double } @asTuple(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca { double }, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr { i64, ptr, i8 }, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = getelementptr i8, ptr %6, i64 -40
  %8 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = icmp eq i32 %9, 1094197076
  call void @"___check_tuple<!ts.tuple<{\22n\22,!ts.number}>>"(ptr %4, i1 %10)
  %11 = getelementptr { i64, ptr, { double } }, ptr %4, i32 0, i32 2
  %12 = load { double }, ptr %11, align 8
  store { double } %12, ptr %3, align 8
  %13 = load { double }, ptr %3, align 8
  ret { double } %13
}

; Function Attrs: mustprogress
define linkonce_odr void @"___check_tuple<!ts.tuple<{\22p\22,!ts.tuple<{\22n\22,!ts.number}>},{\22s\22,!ts.string}>>"(ptr %0, i1 %1) #1 comdat {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i1 %1, ptr %3, align 1
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr { i64, ptr, i8 }, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %2
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8))
  %15 = icmp ne i32 %14, 0
  br label %18

16:                                               ; preds = %2
  %17 = icmp ne i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %23

21:                                               ; preds = %20
  %22 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_187150928315272468, i64 8), ptr %22, align 8
  call void @__cxa_throw(ptr %22, ptr @_ZTIPc, ptr null)
  br label %29

23:                                               ; preds = %20
  %24 = load i1, ptr %3, align 1
  %25 = xor i1 %24, true
  br i1 %25, label %26, label %28

26:                                               ; preds = %23
  %27 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_1712190022454804100, i64 8), ptr %27, align 8
  call void @__cxa_throw(ptr %27, ptr @_ZTIPc, ptr null)
  br label %29

28:                                               ; preds = %23
  ret void

29:                                               ; preds = %21, %26
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca { { double }, ptr }, align 8
  %6 = alloca { double }, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca i1, align 1
  %12 = alloca { double }, align 8
  store { double } zeroinitializer, ptr %12, align 8
  %13 = getelementptr { double }, ptr %12, i32 0, i32 0
  store double 5.000000e+00, ptr %13, align 8
  %14 = load { double }, ptr %12, align 8
  %15 = call ptr @GC_malloc(i64 32)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { i64, ptr, { double } }, ptr %16, i32 0, i32 0
  store i64 8, ptr %17, align 8
  %18 = getelementptr { i64, ptr, { double } }, ptr %16, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4617940393880614542_tuple, i64 40), ptr %18, align 8
  %19 = getelementptr { i64, ptr, { double } }, ptr %16, i32 0, i32 2
  store { double } %14, ptr %19, align 8
  %20 = call { double } @asTuple(ptr %16)
  %21 = extractvalue { double } %20, 0
  %22 = fcmp oeq double %21, 5.000000e+00
  br i1 %22, label %23, label %672

23:                                               ; preds = %0
  store i1 false, ptr %11, align 1
  %24 = call ptr @GC_malloc(i64 24)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr %W, ptr %25, i32 0, i32 0
  store ptr @W..vtbl, ptr %26, align 8
  invoke void @W.constructor(ptr %25, double 7.000000e+00)
          to label %27 unwind label %35

27:                                               ; preds = %23
  %28 = call ptr @GC_malloc(i64 32)
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = getelementptr { i64, ptr, ptr }, ptr %29, i32 0, i32 0
  store i64 8, ptr %30, align 8
  %31 = getelementptr { i64, ptr, ptr }, ptr %29, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_16546078158687492637_class, i64 40), ptr %31, align 8
  %32 = getelementptr { i64, ptr, ptr }, ptr %29, i32 0, i32 2
  store ptr %25, ptr %32, align 8
  %33 = invoke { double } @asTuple(ptr %29)
          to label %34 unwind label %35

34:                                               ; preds = %27
  br label %174

35:                                               ; preds = %27, %23
  %36 = landingpad { ptr, i32 }
          catch ptr null
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  %39 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc27 unwind label %endcatch.resume

.noexc27:                                         ; preds = %35
  %40 = icmp eq ptr %39, @_ZTIi
  br i1 %40, label %41, label %48

41:                                               ; preds = %.noexc27
  %42 = load i32, ptr %38, align 4
  %43 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc28 unwind label %endcatch.resume

.noexc28:                                         ; preds = %41
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = getelementptr { i64, ptr, i32 }, ptr %44, i32 0, i32 0
  store i64 4, ptr %45, align 8
  %46 = getelementptr { i64, ptr, i32 }, ptr %44, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %46, align 8
  %47 = getelementptr { i64, ptr, i32 }, ptr %44, i32 0, i32 2
  store i32 %42, ptr %47, align 4
  br label %171

48:                                               ; preds = %.noexc27
  %49 = icmp eq ptr %39, @_ZTIj
  br i1 %49, label %50, label %57

50:                                               ; preds = %48
  %51 = load i32, ptr %38, align 4
  %52 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc45 unwind label %endcatch.resume

.noexc45:                                         ; preds = %50
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = getelementptr { i64, ptr, i32 }, ptr %53, i32 0, i32 0
  store i64 4, ptr %54, align 8
  %55 = getelementptr { i64, ptr, i32 }, ptr %53, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %55, align 8
  %56 = getelementptr { i64, ptr, i32 }, ptr %53, i32 0, i32 2
  store i32 %51, ptr %56, align 4
  br label %168

57:                                               ; preds = %48
  %58 = icmp eq ptr %39, @_ZTId
  br i1 %58, label %59, label %66

59:                                               ; preds = %57
  %60 = load double, ptr %38, align 8
  %61 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %59
  %62 = getelementptr i8, ptr %61, i64 8
  %63 = getelementptr { i64, ptr, double }, ptr %62, i32 0, i32 0
  store i64 8, ptr %63, align 8
  %64 = getelementptr { i64, ptr, double }, ptr %62, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %64, align 8
  %65 = getelementptr { i64, ptr, double }, ptr %62, i32 0, i32 2
  store double %60, ptr %65, align 8
  br label %165

66:                                               ; preds = %57
  %67 = icmp eq ptr %39, @_ZTIf
  br i1 %67, label %68, label %75

68:                                               ; preds = %66
  %69 = load float, ptr %38, align 4
  %70 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc38 unwind label %endcatch.resume

.noexc38:                                         ; preds = %68
  %71 = getelementptr i8, ptr %70, i64 8
  %72 = getelementptr { i64, ptr, float }, ptr %71, i32 0, i32 0
  store i64 4, ptr %72, align 8
  %73 = getelementptr { i64, ptr, float }, ptr %71, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %73, align 8
  %74 = getelementptr { i64, ptr, float }, ptr %71, i32 0, i32 2
  store float %69, ptr %74, align 4
  br label %162

75:                                               ; preds = %66
  %76 = icmp eq ptr %39, @_ZTIb
  br i1 %76, label %77, label %84

77:                                               ; preds = %75
  %78 = load i1, ptr %38, align 1
  %79 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %77
  %80 = getelementptr i8, ptr %79, i64 8
  %81 = getelementptr { i64, ptr, i1 }, ptr %80, i32 0, i32 0
  store i64 1, ptr %81, align 8
  %82 = getelementptr { i64, ptr, i1 }, ptr %80, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %82, align 8
  %83 = getelementptr { i64, ptr, i1 }, ptr %80, i32 0, i32 2
  store i1 %78, ptr %83, align 1
  br label %159

84:                                               ; preds = %75
  %85 = icmp eq ptr %39, @_ZTIx
  br i1 %85, label %86, label %93

86:                                               ; preds = %84
  %87 = load i64, ptr %38, align 8
  %88 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc32 unwind label %endcatch.resume

.noexc32:                                         ; preds = %86
  %89 = getelementptr i8, ptr %88, i64 8
  %90 = getelementptr { i64, ptr, i64 }, ptr %89, i32 0, i32 0
  store i64 8, ptr %90, align 8
  %91 = getelementptr { i64, ptr, i64 }, ptr %89, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %91, align 8
  %92 = getelementptr { i64, ptr, i64 }, ptr %89, i32 0, i32 2
  store i64 %87, ptr %92, align 8
  br label %156

93:                                               ; preds = %84
  %94 = icmp eq ptr %39, @_ZTIPc
  br i1 %94, label %95, label %102

95:                                               ; preds = %93
  %96 = load ptr, ptr %38, align 8
  %97 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc35 unwind label %endcatch.resume

.noexc35:                                         ; preds = %95
  %98 = getelementptr i8, ptr %97, i64 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 0
  store i64 8, ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 2
  store ptr %96, ptr %101, align 8
  br label %153

102:                                              ; preds = %93
  %103 = icmp eq ptr %39, @_ZTIPv
  br i1 %103, label %104, label %106

104:                                              ; preds = %102
  %105 = load ptr, ptr %38, align 8
  br label %150

106:                                              ; preds = %102
  %107 = load ptr, ptr %39, align 8
  %108 = icmp eq ptr %107, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %108, label %109, label %140

109:                                              ; preds = %106
  %110 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %39, i32 0, i32 2
  %111 = load i32, ptr %110, align 4
  %112 = and i32 %111, 1073741824
  %113 = icmp ne i32 %112, 0
  br i1 %113, label %114, label %130

114:                                              ; preds = %109
  %115 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %39, i32 0, i32 4
  %116 = load ptr, ptr %115, align 8
  %117 = icmp ne ptr %116, null
  br i1 %117, label %118, label %120

118:                                              ; preds = %114
  invoke void %116(ptr %1, ptr %38)
          to label %.noexc42 unwind label %endcatch.resume

.noexc42:                                         ; preds = %118
  %119 = load ptr, ptr %1, align 8
  br label %127

120:                                              ; preds = %114
  %121 = load ptr, ptr %38, align 8
  %122 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %120
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 0
  store i64 8, ptr %124, align 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %125, align 8
  %126 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 2
  store ptr %121, ptr %126, align 8
  br label %127

127:                                              ; preds = %.noexc42, %.noexc8
  %128 = phi ptr [ %123, %.noexc8 ], [ %119, %.noexc42 ]
  br label %129

129:                                              ; preds = %127
  br label %137

130:                                              ; preds = %109
  %131 = load ptr, ptr %38, align 8
  %132 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %130
  %133 = getelementptr i8, ptr %132, i64 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 0
  store i64 8, ptr %134, align 8
  %135 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %135, align 8
  %136 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 2
  store ptr %131, ptr %136, align 8
  br label %137

137:                                              ; preds = %129, %.noexc11
  %138 = phi ptr [ %133, %.noexc11 ], [ %128, %129 ]
  br label %139

139:                                              ; preds = %137
  br label %147

140:                                              ; preds = %106
  %141 = load ptr, ptr %38, align 8
  %142 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %140
  %143 = getelementptr i8, ptr %142, i64 8
  %144 = getelementptr { i64, ptr, ptr }, ptr %143, i32 0, i32 0
  store i64 8, ptr %144, align 8
  %145 = getelementptr { i64, ptr, ptr }, ptr %143, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %145, align 8
  %146 = getelementptr { i64, ptr, ptr }, ptr %143, i32 0, i32 2
  store ptr %141, ptr %146, align 8
  br label %147

147:                                              ; preds = %139, %.noexc9
  %148 = phi ptr [ %143, %.noexc9 ], [ %138, %139 ]
  br label %149

149:                                              ; preds = %147
  br label %150

150:                                              ; preds = %104, %149
  %151 = phi ptr [ %148, %149 ], [ %105, %104 ]
  br label %152

152:                                              ; preds = %150
  br label %153

153:                                              ; preds = %.noexc35, %152
  %154 = phi ptr [ %151, %152 ], [ %98, %.noexc35 ]
  br label %155

155:                                              ; preds = %153
  br label %156

156:                                              ; preds = %.noexc32, %155
  %157 = phi ptr [ %154, %155 ], [ %89, %.noexc32 ]
  br label %158

158:                                              ; preds = %156
  br label %159

159:                                              ; preds = %.noexc, %158
  %160 = phi ptr [ %157, %158 ], [ %80, %.noexc ]
  br label %161

161:                                              ; preds = %159
  br label %162

162:                                              ; preds = %.noexc38, %161
  %163 = phi ptr [ %160, %161 ], [ %71, %.noexc38 ]
  br label %164

164:                                              ; preds = %162
  br label %165

165:                                              ; preds = %.noexc2, %164
  %166 = phi ptr [ %163, %164 ], [ %62, %.noexc2 ]
  br label %167

167:                                              ; preds = %165
  br label %168

168:                                              ; preds = %.noexc45, %167
  %169 = phi ptr [ %166, %167 ], [ %53, %.noexc45 ]
  br label %170

170:                                              ; preds = %168
  br label %171

171:                                              ; preds = %.noexc28, %170
  %172 = phi ptr [ %169, %170 ], [ %44, %.noexc28 ]
  br label %173

173:                                              ; preds = %171
  store ptr %172, ptr %10, align 8
  store i1 true, ptr %11, align 1
  call void @__cxa_end_catch()
  br label %174

174:                                              ; preds = %34, %173
  %175 = load i1, ptr %11, align 1
  br i1 %175, label %176, label %674

176:                                              ; preds = %174
  store i1 false, ptr %11, align 1
  %177 = call ptr @GC_malloc(i64 32)
  %178 = getelementptr i8, ptr %177, i64 8
  %179 = getelementptr { i64, ptr, double }, ptr %178, i32 0, i32 0
  store i64 8, ptr %179, align 8
  %180 = getelementptr { i64, ptr, double }, ptr %178, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %180, align 8
  %181 = getelementptr { i64, ptr, double }, ptr %178, i32 0, i32 2
  store double 3.000000e+00, ptr %181, align 8
  %182 = invoke { double } @asTuple(ptr %178)
          to label %183 unwind label %184

183:                                              ; preds = %176
  br label %323

184:                                              ; preds = %176
  %185 = landingpad { ptr, i32 }
          catch ptr null
  %186 = extractvalue { ptr, i32 } %185, 0
  %187 = call ptr @__cxa_begin_catch(ptr %186)
  %188 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc14 unwind label %endcatch.resume

.noexc14:                                         ; preds = %184
  %189 = icmp eq ptr %188, @_ZTIi
  br i1 %189, label %190, label %197

190:                                              ; preds = %.noexc14
  %191 = load i32, ptr %187, align 4
  %192 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc17 unwind label %endcatch.resume

.noexc17:                                         ; preds = %190
  %193 = getelementptr i8, ptr %192, i64 8
  %194 = getelementptr { i64, ptr, i32 }, ptr %193, i32 0, i32 0
  store i64 4, ptr %194, align 8
  %195 = getelementptr { i64, ptr, i32 }, ptr %193, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %195, align 8
  %196 = getelementptr { i64, ptr, i32 }, ptr %193, i32 0, i32 2
  store i32 %191, ptr %196, align 4
  br label %320

197:                                              ; preds = %.noexc14
  %198 = icmp eq ptr %188, @_ZTIj
  br i1 %198, label %199, label %206

199:                                              ; preds = %197
  %200 = load i32, ptr %187, align 4
  %201 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc19 unwind label %endcatch.resume

.noexc19:                                         ; preds = %199
  %202 = getelementptr i8, ptr %201, i64 8
  %203 = getelementptr { i64, ptr, i32 }, ptr %202, i32 0, i32 0
  store i64 4, ptr %203, align 8
  %204 = getelementptr { i64, ptr, i32 }, ptr %202, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %204, align 8
  %205 = getelementptr { i64, ptr, i32 }, ptr %202, i32 0, i32 2
  store i32 %200, ptr %205, align 4
  br label %317

206:                                              ; preds = %197
  %207 = icmp eq ptr %188, @_ZTId
  br i1 %207, label %208, label %215

208:                                              ; preds = %206
  %209 = load double, ptr %187, align 8
  %210 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc15 unwind label %endcatch.resume

.noexc15:                                         ; preds = %208
  %211 = getelementptr i8, ptr %210, i64 8
  %212 = getelementptr { i64, ptr, double }, ptr %211, i32 0, i32 0
  store i64 8, ptr %212, align 8
  %213 = getelementptr { i64, ptr, double }, ptr %211, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %213, align 8
  %214 = getelementptr { i64, ptr, double }, ptr %211, i32 0, i32 2
  store double %209, ptr %214, align 8
  br label %314

215:                                              ; preds = %206
  %216 = icmp eq ptr %188, @_ZTIf
  br i1 %216, label %217, label %224

217:                                              ; preds = %215
  %218 = load float, ptr %187, align 4
  %219 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc46 unwind label %endcatch.resume

.noexc46:                                         ; preds = %217
  %220 = getelementptr i8, ptr %219, i64 8
  %221 = getelementptr { i64, ptr, float }, ptr %220, i32 0, i32 0
  store i64 4, ptr %221, align 8
  %222 = getelementptr { i64, ptr, float }, ptr %220, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %222, align 8
  %223 = getelementptr { i64, ptr, float }, ptr %220, i32 0, i32 2
  store float %218, ptr %223, align 4
  br label %311

224:                                              ; preds = %215
  %225 = icmp eq ptr %188, @_ZTIb
  br i1 %225, label %226, label %233

226:                                              ; preds = %224
  %227 = load i1, ptr %187, align 1
  %228 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc33 unwind label %endcatch.resume

.noexc33:                                         ; preds = %226
  %229 = getelementptr i8, ptr %228, i64 8
  %230 = getelementptr { i64, ptr, i1 }, ptr %229, i32 0, i32 0
  store i64 1, ptr %230, align 8
  %231 = getelementptr { i64, ptr, i1 }, ptr %229, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %231, align 8
  %232 = getelementptr { i64, ptr, i1 }, ptr %229, i32 0, i32 2
  store i1 %227, ptr %232, align 1
  br label %308

233:                                              ; preds = %224
  %234 = icmp eq ptr %188, @_ZTIx
  br i1 %234, label %235, label %242

235:                                              ; preds = %233
  %236 = load i64, ptr %187, align 8
  %237 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc25 unwind label %endcatch.resume

.noexc25:                                         ; preds = %235
  %238 = getelementptr i8, ptr %237, i64 8
  %239 = getelementptr { i64, ptr, i64 }, ptr %238, i32 0, i32 0
  store i64 8, ptr %239, align 8
  %240 = getelementptr { i64, ptr, i64 }, ptr %238, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %240, align 8
  %241 = getelementptr { i64, ptr, i64 }, ptr %238, i32 0, i32 2
  store i64 %236, ptr %241, align 8
  br label %305

242:                                              ; preds = %233
  %243 = icmp eq ptr %188, @_ZTIPc
  br i1 %243, label %244, label %251

244:                                              ; preds = %242
  %245 = load ptr, ptr %187, align 8
  %246 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc44 unwind label %endcatch.resume

.noexc44:                                         ; preds = %244
  %247 = getelementptr i8, ptr %246, i64 8
  %248 = getelementptr { i64, ptr, ptr }, ptr %247, i32 0, i32 0
  store i64 8, ptr %248, align 8
  %249 = getelementptr { i64, ptr, ptr }, ptr %247, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %249, align 8
  %250 = getelementptr { i64, ptr, ptr }, ptr %247, i32 0, i32 2
  store ptr %245, ptr %250, align 8
  br label %302

251:                                              ; preds = %242
  %252 = icmp eq ptr %188, @_ZTIPv
  br i1 %252, label %253, label %255

253:                                              ; preds = %251
  %254 = load ptr, ptr %187, align 8
  br label %299

255:                                              ; preds = %251
  %256 = load ptr, ptr %188, align 8
  %257 = icmp eq ptr %256, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %257, label %258, label %289

258:                                              ; preds = %255
  %259 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %188, i32 0, i32 2
  %260 = load i32, ptr %259, align 4
  %261 = and i32 %260, 1073741824
  %262 = icmp ne i32 %261, 0
  br i1 %262, label %263, label %279

263:                                              ; preds = %258
  %264 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %188, i32 0, i32 4
  %265 = load ptr, ptr %264, align 8
  %266 = icmp ne ptr %265, null
  br i1 %266, label %267, label %269

267:                                              ; preds = %263
  invoke void %265(ptr %2, ptr %187)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %267
  %268 = load ptr, ptr %2, align 8
  br label %276

269:                                              ; preds = %263
  %270 = load ptr, ptr %187, align 8
  %271 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc24 unwind label %endcatch.resume

.noexc24:                                         ; preds = %269
  %272 = getelementptr i8, ptr %271, i64 8
  %273 = getelementptr { i64, ptr, ptr }, ptr %272, i32 0, i32 0
  store i64 8, ptr %273, align 8
  %274 = getelementptr { i64, ptr, ptr }, ptr %272, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %274, align 8
  %275 = getelementptr { i64, ptr, ptr }, ptr %272, i32 0, i32 2
  store ptr %270, ptr %275, align 8
  br label %276

276:                                              ; preds = %.noexc3, %.noexc24
  %277 = phi ptr [ %272, %.noexc24 ], [ %268, %.noexc3 ]
  br label %278

278:                                              ; preds = %276
  br label %286

279:                                              ; preds = %258
  %280 = load ptr, ptr %187, align 8
  %281 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc41 unwind label %endcatch.resume

.noexc41:                                         ; preds = %279
  %282 = getelementptr i8, ptr %281, i64 8
  %283 = getelementptr { i64, ptr, ptr }, ptr %282, i32 0, i32 0
  store i64 8, ptr %283, align 8
  %284 = getelementptr { i64, ptr, ptr }, ptr %282, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %284, align 8
  %285 = getelementptr { i64, ptr, ptr }, ptr %282, i32 0, i32 2
  store ptr %280, ptr %285, align 8
  br label %286

286:                                              ; preds = %278, %.noexc41
  %287 = phi ptr [ %282, %.noexc41 ], [ %277, %278 ]
  br label %288

288:                                              ; preds = %286
  br label %296

289:                                              ; preds = %255
  %290 = load ptr, ptr %187, align 8
  %291 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc21 unwind label %endcatch.resume

.noexc21:                                         ; preds = %289
  %292 = getelementptr i8, ptr %291, i64 8
  %293 = getelementptr { i64, ptr, ptr }, ptr %292, i32 0, i32 0
  store i64 8, ptr %293, align 8
  %294 = getelementptr { i64, ptr, ptr }, ptr %292, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %294, align 8
  %295 = getelementptr { i64, ptr, ptr }, ptr %292, i32 0, i32 2
  store ptr %290, ptr %295, align 8
  br label %296

296:                                              ; preds = %288, %.noexc21
  %297 = phi ptr [ %292, %.noexc21 ], [ %287, %288 ]
  br label %298

298:                                              ; preds = %296
  br label %299

299:                                              ; preds = %253, %298
  %300 = phi ptr [ %297, %298 ], [ %254, %253 ]
  br label %301

301:                                              ; preds = %299
  br label %302

302:                                              ; preds = %.noexc44, %301
  %303 = phi ptr [ %300, %301 ], [ %247, %.noexc44 ]
  br label %304

304:                                              ; preds = %302
  br label %305

305:                                              ; preds = %.noexc25, %304
  %306 = phi ptr [ %303, %304 ], [ %238, %.noexc25 ]
  br label %307

307:                                              ; preds = %305
  br label %308

308:                                              ; preds = %.noexc33, %307
  %309 = phi ptr [ %306, %307 ], [ %229, %.noexc33 ]
  br label %310

310:                                              ; preds = %308
  br label %311

311:                                              ; preds = %.noexc46, %310
  %312 = phi ptr [ %309, %310 ], [ %220, %.noexc46 ]
  br label %313

313:                                              ; preds = %311
  br label %314

314:                                              ; preds = %.noexc15, %313
  %315 = phi ptr [ %312, %313 ], [ %211, %.noexc15 ]
  br label %316

316:                                              ; preds = %314
  br label %317

317:                                              ; preds = %.noexc19, %316
  %318 = phi ptr [ %315, %316 ], [ %202, %.noexc19 ]
  br label %319

319:                                              ; preds = %317
  br label %320

320:                                              ; preds = %.noexc17, %319
  %321 = phi ptr [ %318, %319 ], [ %193, %.noexc17 ]
  br label %322

322:                                              ; preds = %320
  store ptr %321, ptr %9, align 8
  store i1 true, ptr %11, align 1
  call void @__cxa_end_catch()
  br label %323

323:                                              ; preds = %183, %322
  %324 = load i1, ptr %11, align 1
  br i1 %324, label %325, label %676

325:                                              ; preds = %323
  store i1 false, ptr %11, align 1
  %326 = call ptr @GC_malloc(i64 32)
  %327 = getelementptr i8, ptr %326, i64 8
  %328 = getelementptr { i64, ptr, { i32 } }, ptr %327, i32 0, i32 0
  store i64 4, ptr %328, align 8
  %329 = getelementptr { i64, ptr, { i32 } }, ptr %327, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11850163807121388224_tuple, i64 40), ptr %329, align 8
  %330 = getelementptr { i64, ptr, { i32 } }, ptr %327, i32 0, i32 2
  store { i32 } { i32 5 }, ptr %330, align 4
  %331 = invoke { double } @asTuple(ptr %327)
          to label %332 unwind label %333

332:                                              ; preds = %325
  br label %472

333:                                              ; preds = %325
  %334 = landingpad { ptr, i32 }
          catch ptr null
  %335 = extractvalue { ptr, i32 } %334, 0
  %336 = call ptr @__cxa_begin_catch(ptr %335)
  %337 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %333
  %338 = icmp eq ptr %337, @_ZTIi
  br i1 %338, label %339, label %346

339:                                              ; preds = %.noexc10
  %340 = load i32, ptr %336, align 4
  %341 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc13 unwind label %endcatch.resume

.noexc13:                                         ; preds = %339
  %342 = getelementptr i8, ptr %341, i64 8
  %343 = getelementptr { i64, ptr, i32 }, ptr %342, i32 0, i32 0
  store i64 4, ptr %343, align 8
  %344 = getelementptr { i64, ptr, i32 }, ptr %342, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %344, align 8
  %345 = getelementptr { i64, ptr, i32 }, ptr %342, i32 0, i32 2
  store i32 %340, ptr %345, align 4
  br label %469

346:                                              ; preds = %.noexc10
  %347 = icmp eq ptr %337, @_ZTIj
  br i1 %347, label %348, label %355

348:                                              ; preds = %346
  %349 = load i32, ptr %336, align 4
  %350 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc22 unwind label %endcatch.resume

.noexc22:                                         ; preds = %348
  %351 = getelementptr i8, ptr %350, i64 8
  %352 = getelementptr { i64, ptr, i32 }, ptr %351, i32 0, i32 0
  store i64 4, ptr %352, align 8
  %353 = getelementptr { i64, ptr, i32 }, ptr %351, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %353, align 8
  %354 = getelementptr { i64, ptr, i32 }, ptr %351, i32 0, i32 2
  store i32 %349, ptr %354, align 4
  br label %466

355:                                              ; preds = %346
  %356 = icmp eq ptr %337, @_ZTId
  br i1 %356, label %357, label %364

357:                                              ; preds = %355
  %358 = load double, ptr %336, align 8
  %359 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc23 unwind label %endcatch.resume

.noexc23:                                         ; preds = %357
  %360 = getelementptr i8, ptr %359, i64 8
  %361 = getelementptr { i64, ptr, double }, ptr %360, i32 0, i32 0
  store i64 8, ptr %361, align 8
  %362 = getelementptr { i64, ptr, double }, ptr %360, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %362, align 8
  %363 = getelementptr { i64, ptr, double }, ptr %360, i32 0, i32 2
  store double %358, ptr %363, align 8
  br label %463

364:                                              ; preds = %355
  %365 = icmp eq ptr %337, @_ZTIf
  br i1 %365, label %366, label %373

366:                                              ; preds = %364
  %367 = load float, ptr %336, align 4
  %368 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %366
  %369 = getelementptr i8, ptr %368, i64 8
  %370 = getelementptr { i64, ptr, float }, ptr %369, i32 0, i32 0
  store i64 4, ptr %370, align 8
  %371 = getelementptr { i64, ptr, float }, ptr %369, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %371, align 8
  %372 = getelementptr { i64, ptr, float }, ptr %369, i32 0, i32 2
  store float %367, ptr %372, align 4
  br label %460

373:                                              ; preds = %364
  %374 = icmp eq ptr %337, @_ZTIb
  br i1 %374, label %375, label %382

375:                                              ; preds = %373
  %376 = load i1, ptr %336, align 1
  %377 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %375
  %378 = getelementptr i8, ptr %377, i64 8
  %379 = getelementptr { i64, ptr, i1 }, ptr %378, i32 0, i32 0
  store i64 1, ptr %379, align 8
  %380 = getelementptr { i64, ptr, i1 }, ptr %378, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %380, align 8
  %381 = getelementptr { i64, ptr, i1 }, ptr %378, i32 0, i32 2
  store i1 %376, ptr %381, align 1
  br label %457

382:                                              ; preds = %373
  %383 = icmp eq ptr %337, @_ZTIx
  br i1 %383, label %384, label %391

384:                                              ; preds = %382
  %385 = load i64, ptr %336, align 8
  %386 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %384
  %387 = getelementptr i8, ptr %386, i64 8
  %388 = getelementptr { i64, ptr, i64 }, ptr %387, i32 0, i32 0
  store i64 8, ptr %388, align 8
  %389 = getelementptr { i64, ptr, i64 }, ptr %387, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %389, align 8
  %390 = getelementptr { i64, ptr, i64 }, ptr %387, i32 0, i32 2
  store i64 %385, ptr %390, align 8
  br label %454

391:                                              ; preds = %382
  %392 = icmp eq ptr %337, @_ZTIPc
  br i1 %392, label %393, label %400

393:                                              ; preds = %391
  %394 = load ptr, ptr %336, align 8
  %395 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %393
  %396 = getelementptr i8, ptr %395, i64 8
  %397 = getelementptr { i64, ptr, ptr }, ptr %396, i32 0, i32 0
  store i64 8, ptr %397, align 8
  %398 = getelementptr { i64, ptr, ptr }, ptr %396, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %398, align 8
  %399 = getelementptr { i64, ptr, ptr }, ptr %396, i32 0, i32 2
  store ptr %394, ptr %399, align 8
  br label %451

400:                                              ; preds = %391
  %401 = icmp eq ptr %337, @_ZTIPv
  br i1 %401, label %402, label %404

402:                                              ; preds = %400
  %403 = load ptr, ptr %336, align 8
  br label %448

404:                                              ; preds = %400
  %405 = load ptr, ptr %337, align 8
  %406 = icmp eq ptr %405, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %406, label %407, label %438

407:                                              ; preds = %404
  %408 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %337, i32 0, i32 2
  %409 = load i32, ptr %408, align 4
  %410 = and i32 %409, 1073741824
  %411 = icmp ne i32 %410, 0
  br i1 %411, label %412, label %428

412:                                              ; preds = %407
  %413 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %337, i32 0, i32 4
  %414 = load ptr, ptr %413, align 8
  %415 = icmp ne ptr %414, null
  br i1 %415, label %416, label %418

416:                                              ; preds = %412
  invoke void %414(ptr %3, ptr %336)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %416
  %417 = load ptr, ptr %3, align 8
  br label %425

418:                                              ; preds = %412
  %419 = load ptr, ptr %336, align 8
  %420 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc20 unwind label %endcatch.resume

.noexc20:                                         ; preds = %418
  %421 = getelementptr i8, ptr %420, i64 8
  %422 = getelementptr { i64, ptr, ptr }, ptr %421, i32 0, i32 0
  store i64 8, ptr %422, align 8
  %423 = getelementptr { i64, ptr, ptr }, ptr %421, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %423, align 8
  %424 = getelementptr { i64, ptr, ptr }, ptr %421, i32 0, i32 2
  store ptr %419, ptr %424, align 8
  br label %425

425:                                              ; preds = %.noexc6, %.noexc20
  %426 = phi ptr [ %421, %.noexc20 ], [ %417, %.noexc6 ]
  br label %427

427:                                              ; preds = %425
  br label %435

428:                                              ; preds = %407
  %429 = load ptr, ptr %336, align 8
  %430 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc16 unwind label %endcatch.resume

.noexc16:                                         ; preds = %428
  %431 = getelementptr i8, ptr %430, i64 8
  %432 = getelementptr { i64, ptr, ptr }, ptr %431, i32 0, i32 0
  store i64 8, ptr %432, align 8
  %433 = getelementptr { i64, ptr, ptr }, ptr %431, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %433, align 8
  %434 = getelementptr { i64, ptr, ptr }, ptr %431, i32 0, i32 2
  store ptr %429, ptr %434, align 8
  br label %435

435:                                              ; preds = %427, %.noexc16
  %436 = phi ptr [ %431, %.noexc16 ], [ %426, %427 ]
  br label %437

437:                                              ; preds = %435
  br label %445

438:                                              ; preds = %404
  %439 = load ptr, ptr %336, align 8
  %440 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc18 unwind label %endcatch.resume

.noexc18:                                         ; preds = %438
  %441 = getelementptr i8, ptr %440, i64 8
  %442 = getelementptr { i64, ptr, ptr }, ptr %441, i32 0, i32 0
  store i64 8, ptr %442, align 8
  %443 = getelementptr { i64, ptr, ptr }, ptr %441, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %443, align 8
  %444 = getelementptr { i64, ptr, ptr }, ptr %441, i32 0, i32 2
  store ptr %439, ptr %444, align 8
  br label %445

445:                                              ; preds = %437, %.noexc18
  %446 = phi ptr [ %441, %.noexc18 ], [ %436, %437 ]
  br label %447

447:                                              ; preds = %445
  br label %448

448:                                              ; preds = %402, %447
  %449 = phi ptr [ %446, %447 ], [ %403, %402 ]
  br label %450

450:                                              ; preds = %448
  br label %451

451:                                              ; preds = %.noexc5, %450
  %452 = phi ptr [ %449, %450 ], [ %396, %.noexc5 ]
  br label %453

453:                                              ; preds = %451
  br label %454

454:                                              ; preds = %.noexc7, %453
  %455 = phi ptr [ %452, %453 ], [ %387, %.noexc7 ]
  br label %456

456:                                              ; preds = %454
  br label %457

457:                                              ; preds = %.noexc12, %456
  %458 = phi ptr [ %455, %456 ], [ %378, %.noexc12 ]
  br label %459

459:                                              ; preds = %457
  br label %460

460:                                              ; preds = %.noexc1, %459
  %461 = phi ptr [ %458, %459 ], [ %369, %.noexc1 ]
  br label %462

462:                                              ; preds = %460
  br label %463

463:                                              ; preds = %.noexc23, %462
  %464 = phi ptr [ %461, %462 ], [ %360, %.noexc23 ]
  br label %465

465:                                              ; preds = %463
  br label %466

466:                                              ; preds = %.noexc22, %465
  %467 = phi ptr [ %464, %465 ], [ %351, %.noexc22 ]
  br label %468

468:                                              ; preds = %466
  br label %469

469:                                              ; preds = %.noexc13, %468
  %470 = phi ptr [ %467, %468 ], [ %342, %.noexc13 ]
  br label %471

471:                                              ; preds = %469
  store ptr %470, ptr %8, align 8
  store i1 true, ptr %11, align 1
  call void @__cxa_end_catch()
  br label %472

472:                                              ; preds = %332, %471
  %473 = load i1, ptr %11, align 1
  br i1 %473, label %474, label %678

474:                                              ; preds = %472
  store i1 false, ptr %11, align 1
  %475 = call ptr @GC_malloc(i64 32)
  %476 = getelementptr i8, ptr %475, i64 8
  %477 = getelementptr { i64, ptr, { double } }, ptr %476, i32 0, i32 0
  store i64 8, ptr %477, align 8
  %478 = getelementptr { i64, ptr, { double } }, ptr %476, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6300196387968623321_tuple, i64 40), ptr %478, align 8
  %479 = getelementptr { i64, ptr, { double } }, ptr %476, i32 0, i32 2
  store { double } { double 1.500000e+00 }, ptr %479, align 8
  %480 = invoke { double } @asTuple(ptr %476)
          to label %481 unwind label %482

481:                                              ; preds = %474
  br label %621

482:                                              ; preds = %474
  %483 = landingpad { ptr, i32 }
          catch ptr null
  %484 = extractvalue { ptr, i32 } %483, 0
  %485 = call ptr @__cxa_begin_catch(ptr %484)
  %486 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc26 unwind label %endcatch.resume

.noexc26:                                         ; preds = %482
  %487 = icmp eq ptr %486, @_ZTIi
  br i1 %487, label %488, label %495

488:                                              ; preds = %.noexc26
  %489 = load i32, ptr %485, align 4
  %490 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc39 unwind label %endcatch.resume

.noexc39:                                         ; preds = %488
  %491 = getelementptr i8, ptr %490, i64 8
  %492 = getelementptr { i64, ptr, i32 }, ptr %491, i32 0, i32 0
  store i64 4, ptr %492, align 8
  %493 = getelementptr { i64, ptr, i32 }, ptr %491, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %493, align 8
  %494 = getelementptr { i64, ptr, i32 }, ptr %491, i32 0, i32 2
  store i32 %489, ptr %494, align 4
  br label %618

495:                                              ; preds = %.noexc26
  %496 = icmp eq ptr %486, @_ZTIj
  br i1 %496, label %497, label %504

497:                                              ; preds = %495
  %498 = load i32, ptr %485, align 4
  %499 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc29 unwind label %endcatch.resume

.noexc29:                                         ; preds = %497
  %500 = getelementptr i8, ptr %499, i64 8
  %501 = getelementptr { i64, ptr, i32 }, ptr %500, i32 0, i32 0
  store i64 4, ptr %501, align 8
  %502 = getelementptr { i64, ptr, i32 }, ptr %500, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %502, align 8
  %503 = getelementptr { i64, ptr, i32 }, ptr %500, i32 0, i32 2
  store i32 %498, ptr %503, align 4
  br label %615

504:                                              ; preds = %495
  %505 = icmp eq ptr %486, @_ZTId
  br i1 %505, label %506, label %513

506:                                              ; preds = %504
  %507 = load double, ptr %485, align 8
  %508 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc40 unwind label %endcatch.resume

.noexc40:                                         ; preds = %506
  %509 = getelementptr i8, ptr %508, i64 8
  %510 = getelementptr { i64, ptr, double }, ptr %509, i32 0, i32 0
  store i64 8, ptr %510, align 8
  %511 = getelementptr { i64, ptr, double }, ptr %509, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %511, align 8
  %512 = getelementptr { i64, ptr, double }, ptr %509, i32 0, i32 2
  store double %507, ptr %512, align 8
  br label %612

513:                                              ; preds = %504
  %514 = icmp eq ptr %486, @_ZTIf
  br i1 %514, label %515, label %522

515:                                              ; preds = %513
  %516 = load float, ptr %485, align 4
  %517 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc34 unwind label %endcatch.resume

.noexc34:                                         ; preds = %515
  %518 = getelementptr i8, ptr %517, i64 8
  %519 = getelementptr { i64, ptr, float }, ptr %518, i32 0, i32 0
  store i64 4, ptr %519, align 8
  %520 = getelementptr { i64, ptr, float }, ptr %518, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %520, align 8
  %521 = getelementptr { i64, ptr, float }, ptr %518, i32 0, i32 2
  store float %516, ptr %521, align 4
  br label %609

522:                                              ; preds = %513
  %523 = icmp eq ptr %486, @_ZTIb
  br i1 %523, label %524, label %531

524:                                              ; preds = %522
  %525 = load i1, ptr %485, align 1
  %526 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc31 unwind label %endcatch.resume

.noexc31:                                         ; preds = %524
  %527 = getelementptr i8, ptr %526, i64 8
  %528 = getelementptr { i64, ptr, i1 }, ptr %527, i32 0, i32 0
  store i64 1, ptr %528, align 8
  %529 = getelementptr { i64, ptr, i1 }, ptr %527, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %529, align 8
  %530 = getelementptr { i64, ptr, i1 }, ptr %527, i32 0, i32 2
  store i1 %525, ptr %530, align 1
  br label %606

531:                                              ; preds = %522
  %532 = icmp eq ptr %486, @_ZTIx
  br i1 %532, label %533, label %540

533:                                              ; preds = %531
  %534 = load i64, ptr %485, align 8
  %535 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc36 unwind label %endcatch.resume

.noexc36:                                         ; preds = %533
  %536 = getelementptr i8, ptr %535, i64 8
  %537 = getelementptr { i64, ptr, i64 }, ptr %536, i32 0, i32 0
  store i64 8, ptr %537, align 8
  %538 = getelementptr { i64, ptr, i64 }, ptr %536, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %538, align 8
  %539 = getelementptr { i64, ptr, i64 }, ptr %536, i32 0, i32 2
  store i64 %534, ptr %539, align 8
  br label %603

540:                                              ; preds = %531
  %541 = icmp eq ptr %486, @_ZTIPc
  br i1 %541, label %542, label %549

542:                                              ; preds = %540
  %543 = load ptr, ptr %485, align 8
  %544 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %542
  %545 = getelementptr i8, ptr %544, i64 8
  %546 = getelementptr { i64, ptr, ptr }, ptr %545, i32 0, i32 0
  store i64 8, ptr %546, align 8
  %547 = getelementptr { i64, ptr, ptr }, ptr %545, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %547, align 8
  %548 = getelementptr { i64, ptr, ptr }, ptr %545, i32 0, i32 2
  store ptr %543, ptr %548, align 8
  br label %600

549:                                              ; preds = %540
  %550 = icmp eq ptr %486, @_ZTIPv
  br i1 %550, label %551, label %553

551:                                              ; preds = %549
  %552 = load ptr, ptr %485, align 8
  br label %597

553:                                              ; preds = %549
  %554 = load ptr, ptr %486, align 8
  %555 = icmp eq ptr %554, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %555, label %556, label %587

556:                                              ; preds = %553
  %557 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %486, i32 0, i32 2
  %558 = load i32, ptr %557, align 4
  %559 = and i32 %558, 1073741824
  %560 = icmp ne i32 %559, 0
  br i1 %560, label %561, label %577

561:                                              ; preds = %556
  %562 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %486, i32 0, i32 4
  %563 = load ptr, ptr %562, align 8
  %564 = icmp ne ptr %563, null
  br i1 %564, label %565, label %567

565:                                              ; preds = %561
  invoke void %563(ptr %4, ptr %485)
          to label %.noexc37 unwind label %endcatch.resume

.noexc37:                                         ; preds = %565
  %566 = load ptr, ptr %4, align 8
  br label %574

567:                                              ; preds = %561
  %568 = load ptr, ptr %485, align 8
  %569 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc30 unwind label %endcatch.resume

.noexc30:                                         ; preds = %567
  %570 = getelementptr i8, ptr %569, i64 8
  %571 = getelementptr { i64, ptr, ptr }, ptr %570, i32 0, i32 0
  store i64 8, ptr %571, align 8
  %572 = getelementptr { i64, ptr, ptr }, ptr %570, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %572, align 8
  %573 = getelementptr { i64, ptr, ptr }, ptr %570, i32 0, i32 2
  store ptr %568, ptr %573, align 8
  br label %574

574:                                              ; preds = %.noexc37, %.noexc30
  %575 = phi ptr [ %570, %.noexc30 ], [ %566, %.noexc37 ]
  br label %576

576:                                              ; preds = %574
  br label %584

577:                                              ; preds = %556
  %578 = load ptr, ptr %485, align 8
  %579 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc43 unwind label %endcatch.resume

.noexc43:                                         ; preds = %577
  %580 = getelementptr i8, ptr %579, i64 8
  %581 = getelementptr { i64, ptr, ptr }, ptr %580, i32 0, i32 0
  store i64 8, ptr %581, align 8
  %582 = getelementptr { i64, ptr, ptr }, ptr %580, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %582, align 8
  %583 = getelementptr { i64, ptr, ptr }, ptr %580, i32 0, i32 2
  store ptr %578, ptr %583, align 8
  br label %584

584:                                              ; preds = %576, %.noexc43
  %585 = phi ptr [ %580, %.noexc43 ], [ %575, %576 ]
  br label %586

586:                                              ; preds = %584
  br label %594

587:                                              ; preds = %553
  %588 = load ptr, ptr %485, align 8
  %589 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc47 unwind label %endcatch.resume

.noexc47:                                         ; preds = %587
  %590 = getelementptr i8, ptr %589, i64 8
  %591 = getelementptr { i64, ptr, ptr }, ptr %590, i32 0, i32 0
  store i64 8, ptr %591, align 8
  %592 = getelementptr { i64, ptr, ptr }, ptr %590, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %592, align 8
  %593 = getelementptr { i64, ptr, ptr }, ptr %590, i32 0, i32 2
  store ptr %588, ptr %593, align 8
  br label %594

594:                                              ; preds = %586, %.noexc47
  %595 = phi ptr [ %590, %.noexc47 ], [ %585, %586 ]
  br label %596

596:                                              ; preds = %594
  br label %597

597:                                              ; preds = %551, %596
  %598 = phi ptr [ %595, %596 ], [ %552, %551 ]
  br label %599

599:                                              ; preds = %597
  br label %600

600:                                              ; preds = %.noexc4, %599
  %601 = phi ptr [ %598, %599 ], [ %545, %.noexc4 ]
  br label %602

602:                                              ; preds = %600
  br label %603

603:                                              ; preds = %.noexc36, %602
  %604 = phi ptr [ %601, %602 ], [ %536, %.noexc36 ]
  br label %605

605:                                              ; preds = %603
  br label %606

606:                                              ; preds = %.noexc31, %605
  %607 = phi ptr [ %604, %605 ], [ %527, %.noexc31 ]
  br label %608

608:                                              ; preds = %606
  br label %609

609:                                              ; preds = %.noexc34, %608
  %610 = phi ptr [ %607, %608 ], [ %518, %.noexc34 ]
  br label %611

611:                                              ; preds = %609
  br label %612

612:                                              ; preds = %.noexc40, %611
  %613 = phi ptr [ %610, %611 ], [ %509, %.noexc40 ]
  br label %614

614:                                              ; preds = %612
  br label %615

615:                                              ; preds = %.noexc29, %614
  %616 = phi ptr [ %613, %614 ], [ %500, %.noexc29 ]
  br label %617

617:                                              ; preds = %615
  br label %618

618:                                              ; preds = %.noexc39, %617
  %619 = phi ptr [ %616, %617 ], [ %491, %.noexc39 ]
  br label %620

620:                                              ; preds = %618
  store ptr %619, ptr %7, align 8
  store i1 true, ptr %11, align 1
  call void @__cxa_end_catch()
  br label %621

621:                                              ; preds = %481, %620
  %622 = load i1, ptr %11, align 1
  br i1 %622, label %623, label %680

623:                                              ; preds = %621
  %624 = call ptr @GC_malloc(i64 32)
  %625 = getelementptr i8, ptr %624, i64 8
  %626 = getelementptr { i64, ptr, { double } }, ptr %625, i32 0, i32 0
  store i64 8, ptr %626, align 8
  %627 = getelementptr { i64, ptr, { double } }, ptr %625, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_12845129401128444698_tuple, i64 40), ptr %627, align 8
  %628 = getelementptr { i64, ptr, { double } }, ptr %625, i32 0, i32 2
  store { double } { double 2.500000e+00 }, ptr %628, align 8
  %629 = call { double } @asTuple(ptr %625)
  %630 = extractvalue { double } %629, 0
  %631 = fcmp oeq double %630, 2.500000e+00
  br i1 %631, label %632, label %682

632:                                              ; preds = %623
  store { double } zeroinitializer, ptr %6, align 8
  %633 = getelementptr { double }, ptr %6, i32 0, i32 0
  store double 5.000000e+00, ptr %633, align 8
  %634 = load { double }, ptr %6, align 8
  store { { double }, ptr } { { double } zeroinitializer, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8) }, ptr %5, align 8
  %635 = getelementptr { { double }, ptr }, ptr %5, i32 0, i32 0
  store { double } %634, ptr %635, align 8
  %636 = load { { double }, ptr }, ptr %5, align 8
  %637 = call ptr @GC_malloc(i64 40)
  %638 = getelementptr i8, ptr %637, i64 8
  %639 = getelementptr { i64, ptr, { { double }, ptr } }, ptr %638, i32 0, i32 0
  store i64 16, ptr %639, align 8
  %640 = getelementptr { i64, ptr, { { double }, ptr } }, ptr %638, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4706355282782734589_tuple, i64 40), ptr %640, align 8
  %641 = getelementptr { i64, ptr, { { double }, ptr } }, ptr %638, i32 0, i32 2
  store { { double }, ptr } %636, ptr %641, align 8
  %642 = getelementptr { i64, ptr, i8 }, ptr %638, i32 0, i32 1
  %643 = load ptr, ptr %642, align 8
  %644 = getelementptr i8, ptr %643, i64 -40
  %645 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %644, i32 0, i32 1
  %646 = load i32, ptr %645, align 4
  %647 = icmp eq i32 %646, 640346249
  call void @"___check_tuple<!ts.tuple<{\22p\22,!ts.tuple<{\22n\22,!ts.number}>},{\22s\22,!ts.string}>>"(ptr %638, i1 %647)
  %648 = getelementptr { i64, ptr, { { double }, ptr } }, ptr %638, i32 0, i32 2
  %649 = load { { double }, ptr }, ptr %648, align 8
  %650 = extractvalue { { double }, ptr } %649, 0
  %651 = extractvalue { double } %650, 0
  %652 = fcmp oeq double %651, 5.000000e+00
  br i1 %652, label %653, label %668

653:                                              ; preds = %632
  %654 = extractvalue { { double }, ptr } %649, 1
  %655 = ptrtoint ptr %654 to i64
  %656 = icmp ne i64 %655, 0
  %657 = zext i1 %656 to i32
  %658 = and i32 %657, 1
  %659 = icmp ne i32 %658, 0
  br i1 %659, label %660, label %663

660:                                              ; preds = %653
  %661 = call i32 @strcmp(ptr %654, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %662 = icmp eq i32 %661, 0
  br label %665

663:                                              ; preds = %653
  %664 = icmp eq i64 %655, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8) to i64)
  br label %665

665:                                              ; preds = %660, %663
  %666 = phi i1 [ %664, %663 ], [ %662, %660 ]
  br label %667

667:                                              ; preds = %665
  br label %668

668:                                              ; preds = %667, %632
  %669 = phi i1 [ %666, %667 ], [ %652, %632 ]
  br i1 %669, label %670, label %684

670:                                              ; preds = %668
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

671:                                              ; preds = %684, %682, %680, %678, %676, %674, %672
  unreachable

672:                                              ; preds = %0
  %673 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 15, ptr null)
  br label %671

674:                                              ; preds = %174
  %675 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 24, ptr null)
  br label %671

676:                                              ; preds = %323
  %677 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 34, ptr null)
  br label %671

678:                                              ; preds = %472
  %679 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 46, ptr null)
  br label %671

680:                                              ; preds = %621
  %681 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 56, ptr null)
  br label %671

682:                                              ; preds = %623
  %683 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 60, ptr null)
  br label %671

684:                                              ; preds = %668
  %685 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8982223085663264368, i64 8), i32 64, ptr null)
  br label %671

endcatch.resume:                                  ; preds = %587, %217, %50, %244, %577, %118, %279, %506, %488, %68, %565, %533, %95, %515, %226, %86, %524, %567, %497, %41, %35, %482, %235, %269, %357, %348, %289, %418, %199, %438, %190, %428, %208, %184, %339, %375, %130, %333, %140, %120, %384, %416, %393, %542, %267, %59, %366, %77
  %686 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %686
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
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

