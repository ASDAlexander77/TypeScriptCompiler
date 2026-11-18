; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Color = type { ptr, double, double }
%Boxed = type { ptr, double }
%Owner = type { ptr, double }

$Color..size = comdat any

$Boxed..size = comdat any

$Owner..size = comdat any

@s_3769135706557701272 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFColor\00", align 8
@s_13188565116058713315 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBoxed\00", align 8
@s_13362706469355457364 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOwner\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_1177635204362931792 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa captured parameter read only in `new` arguments\00", align 8
@f_12536250055124535363 = internal constant [116 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00capture_in_new_arguments.ts\00", align 8
@m_16719205305887358728 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa captured object's field in `new` arguments\00", align 8
@m_17325684577615475286 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa captured local in `new` arguments\00", align 8
@m_7406480972552250788 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa captured mutable local in `new` arguments\00", align 8
@m_11473572904547755511 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFseveral captures across several arguments\00", align 8
@m_8310380780962149797 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa `new` inside another `new`'s arguments\00", align 8
@m_14565991458262343450 = internal constant [61 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa lambda inside a lambda reaching the outer function\00", align 8
@m_10636315116849094793 = internal constant [48 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF`this` captured by a lambda in a method\00", align 8
@m_15985146957802969304 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe same capture read outside the `new` too\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Color..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_3769135706557701272, i64 8)
@Color..size = global i64 24, comdat
@Color..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Color..instanceOf, ptr @Color..new, ptr @Color..rtti, ptr @Color..size }
@Boxed..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_13188565116058713315, i64 8)
@Boxed..size = global i64 16, comdat
@Boxed..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Boxed..instanceOf, ptr @Boxed..new, ptr @Boxed..rtti, ptr @Boxed..size }
@Owner..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_13362706469355457364, i64 8)
@Owner..size = global i64 16, comdat
@Owner..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Owner..instanceOf, ptr @Owner..new, ptr @Owner.build, ptr @Owner..rtti, ptr @Owner..size }
@td_17318990632888470625_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_17318990632888470625, ptr @tsretcb_17318990632888470625, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_11394132933189114885_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_11394132933189114885, ptr @tsretcb_11394132933189114885, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_4854122611388280841_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_4854122611388280841, ptr @tsretcb_4854122611388280841, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_7702282631036391141_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_7702282631036391141, ptr @tsretcb_7702282631036391141, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_13749874419300764099_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_13749874419300764099, ptr @tsretcb_13749874419300764099, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

define internal void @tsretcb_13749874419300764099(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrel_15264642789954933751(ptr %0) {
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

define internal void @tsrelcb_13749874419300764099(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr, double }, ptr %2, i32 0, i32 0
  call void @tsrel_15264642789954933751(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsretcb_7702282631036391141(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_7702282631036391141(ptr %0) {
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

define internal void @tsretcb_4854122611388280841(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_4854122611388280841(ptr %0) {
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
  call void @__tslang_free_block(ptr %8)
  br label %13

13:                                               ; preds = %12, %10, %6
  call void @__tslang_free_block(ptr %2)
  br label %14

14:                                               ; preds = %13, %4, %1
  ret void
}

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

define internal void @tsretcb_11394132933189114885(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrel_4765636457781556658(ptr %0) {
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

define internal void @tsrelcb_11394132933189114885(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_4765636457781556658(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
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

define internal void @tsretcb_17318990632888470625(ptr %0) {
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

define internal void @tsrelcb_17318990632888470625(ptr %0) {
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

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Color..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Color..rtti, align 8
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
define ptr @Color..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Color, ptr %3, i32 0, i32 0
  store ptr @Color..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Color.constructor(ptr %0, double %1, double %2) #1 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %Color, ptr %7, i32 0, i32 1
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %Color, ptr %10, i32 0, i32 2
  %12 = load double, ptr %4, align 8
  store double %12, ptr %11, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Boxed..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Boxed..rtti, align 8
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
define ptr @Boxed..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Boxed, ptr %3, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Boxed.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Boxed, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @.f_capturedParam..afL34C18FH2918425925(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store double %1, ptr %3, align 8
  %5 = getelementptr { double }, ptr %0, i32 0, i32 0
  %6 = load double, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Boxed, ptr %8, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %9, align 8
  %10 = load double, ptr %3, align 8
  %11 = fadd double %6, %10
  call void @Boxed.constructor(ptr %8, double %11)
  store ptr %8, ptr %4, align 8
  %12 = load ptr, ptr %4, align 8
  ret ptr %12
}

; Function Attrs: mustprogress
define double @capturedParam(double %0) #1 {
  %2 = alloca <{ i64, { double } }>, align 16
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store i64 -1, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 8
  store { double } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { double }, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @.f_capturedParam..afL34C18FH2918425925, ptr null, ptr null }, ptr %5, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_17318990632888470625_object, i64 40), 2
  %10 = call ptr @.f_capturedParam..afL34C18FH2918425925(ptr %5, double 1.000000e+00)
  %11 = getelementptr %Boxed, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  store double %12, ptr %4, align 8
  %13 = load double, ptr %4, align 8
  ret double %13
}

; Function Attrs: mustprogress
define ptr @.f_capturedField..afL40C18FH2918425925(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store double %1, ptr %3, align 8
  %5 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Boxed, ptr %8, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %9, align 8
  %10 = getelementptr %Color, ptr %6, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = load double, ptr %3, align 8
  %13 = fadd double %11, %12
  call void @Boxed.constructor(ptr %8, double %13)
  store ptr %8, ptr %4, align 8
  %14 = load ptr, ptr %4, align 8
  ret ptr %14
}

; Function Attrs: mustprogress
define double @capturedField(ptr %0) #1 {
  %2 = alloca <{ i64, { ptr } }>, align 16
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  store i64 -1, ptr %2, align 8
  %6 = getelementptr i8, ptr %2, i64 8
  store { ptr } zeroinitializer, ptr %6, align 8
  %7 = getelementptr { ptr }, ptr %6, i32 0, i32 0
  %8 = load ptr, ptr %3, align 8
  store ptr %8, ptr %7, align 8
  %9 = insertvalue { ptr, ptr, ptr } { ptr @.f_capturedField..afL40C18FH2918425925, ptr null, ptr null }, ptr %6, 1
  %10 = insertvalue { ptr, ptr, ptr } %9, ptr getelementptr inbounds nuw (i8, ptr @td_11394132933189114885_object, i64 40), 2
  %11 = call ptr @.f_capturedField..afL40C18FH2918425925(ptr %6, double 1.000000e+00)
  %12 = getelementptr %Boxed, ptr %11, i32 0, i32 1
  %13 = load double, ptr %12, align 8
  store double %13, ptr %4, align 8
  %14 = load double, ptr %4, align 8
  ret double %14
}

; Function Attrs: mustprogress
define ptr @.f_capturedLocal..afL49C18FH2918425925(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store double %1, ptr %3, align 8
  %5 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Boxed, ptr %8, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %9, align 8
  %10 = load double, ptr %6, align 8
  %11 = load double, ptr %3, align 8
  %12 = fadd double %10, %11
  call void @Boxed.constructor(ptr %8, double %12)
  store ptr %8, ptr %4, align 8
  %13 = load ptr, ptr %4, align 8
  ret ptr %13
}

; Function Attrs: mustprogress
define double @capturedLocal() #1 {
  %1 = alloca <{ i64, { ptr } }>, align 16
  %2 = alloca double, align 8
  %3 = call ptr @GC_malloc(i64 16)
  %4 = getelementptr i8, ptr %3, i64 8
  store double 1.000000e+01, ptr %4, align 8
  store i64 -1, ptr %1, align 8
  %5 = getelementptr i8, ptr %1, i64 8
  store { ptr } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %4, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_capturedLocal..afL49C18FH2918425925, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_4854122611388280841_object, i64 40), 2
  %9 = call ptr @.f_capturedLocal..afL49C18FH2918425925(ptr %5, double 1.000000e+00)
  %10 = getelementptr %Boxed, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  store double %11, ptr %2, align 8
  %12 = load double, ptr %2, align 8
  ret double %12
}

; Function Attrs: mustprogress
define ptr @.f_capturedMutableLocal..afL59C18FH2918425925(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store double %1, ptr %3, align 8
  %5 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Boxed, ptr %8, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %9, align 8
  %10 = load double, ptr %6, align 8
  %11 = load double, ptr %3, align 8
  %12 = fadd double %10, %11
  call void @Boxed.constructor(ptr %8, double %12)
  store ptr %8, ptr %4, align 8
  %13 = load ptr, ptr %4, align 8
  ret ptr %13
}

; Function Attrs: mustprogress
define double @capturedMutableLocal() #1 {
  %1 = alloca <{ i64, { ptr } }>, align 16
  %2 = alloca double, align 8
  %3 = call ptr @GC_malloc(i64 16)
  %4 = getelementptr i8, ptr %3, i64 8
  store double 1.000000e+01, ptr %4, align 8
  store i64 -1, ptr %1, align 8
  %5 = getelementptr i8, ptr %1, i64 8
  store { ptr } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { ptr }, ptr %5, i32 0, i32 0
  store ptr %4, ptr %6, align 8
  %7 = insertvalue { ptr, ptr, ptr } { ptr @.f_capturedMutableLocal..afL59C18FH2918425925, ptr null, ptr null }, ptr %5, 1
  %8 = insertvalue { ptr, ptr, ptr } %7, ptr getelementptr inbounds nuw (i8, ptr @td_4854122611388280841_object, i64 40), 2
  store double 2.000000e+01, ptr %4, align 8
  %9 = call ptr @.f_capturedMutableLocal..afL59C18FH2918425925(ptr %5, double 1.000000e+00)
  %10 = getelementptr %Boxed, ptr %9, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  store double %11, ptr %2, align 8
  %12 = load double, ptr %2, align 8
  ret double %12
}

; Function Attrs: mustprogress
define ptr @.f_severalArgumentsAndCaptures..afL68C18FH2918425925(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = getelementptr { double, double }, ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8
  %5 = getelementptr { double, double }, ptr %0, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Color, ptr %8, i32 0, i32 0
  store ptr @Color..vtbl, ptr %9, align 8
  call void @Color.constructor(ptr %8, double %4, double %6)
  store ptr %8, ptr %2, align 8
  %10 = load ptr, ptr %2, align 8
  ret ptr %10
}

; Function Attrs: mustprogress
define double @severalArgumentsAndCaptures(double %0, double %1) #1 {
  %3 = alloca <{ i64, { double, double } }>, align 16
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  store double %0, ptr %5, align 8
  store double %1, ptr %4, align 8
  store i64 -1, ptr %3, align 8
  %7 = getelementptr i8, ptr %3, i64 8
  store { double, double } zeroinitializer, ptr %7, align 8
  %8 = getelementptr { double, double }, ptr %7, i32 0, i32 0
  %9 = load double, ptr %5, align 8
  store double %9, ptr %8, align 8
  %10 = getelementptr { double, double }, ptr %7, i32 0, i32 1
  %11 = load double, ptr %4, align 8
  store double %11, ptr %10, align 8
  %12 = insertvalue { ptr, ptr, ptr } { ptr @.f_severalArgumentsAndCaptures..afL68C18FH2918425925, ptr null, ptr null }, ptr %7, 1
  %13 = insertvalue { ptr, ptr, ptr } %12, ptr getelementptr inbounds nuw (i8, ptr @td_7702282631036391141_object, i64 40), 2
  %14 = call ptr @.f_severalArgumentsAndCaptures..afL68C18FH2918425925(ptr %7)
  %15 = getelementptr %Color, ptr %14, i32 0, i32 1
  %16 = load double, ptr %15, align 8
  %17 = fmul double %16, 1.000000e+01
  %18 = getelementptr %Color, ptr %14, i32 0, i32 2
  %19 = load double, ptr %18, align 8
  %20 = fadd double %17, %19
  store double %20, ptr %6, align 8
  %21 = load double, ptr %6, align 8
  ret double %21
}

; Function Attrs: mustprogress
define ptr @.f_nestedNew..afL77C18FH2918425925(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, %Boxed }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Boxed zeroinitializer, ptr %4, align 8
  %5 = getelementptr { double }, ptr %0, i32 0, i32 0
  %6 = load double, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Boxed, ptr %8, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %9, align 8
  store %Boxed zeroinitializer, ptr %4, align 8
  %10 = getelementptr %Boxed, ptr %4, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %10, align 8
  call void @Boxed.constructor(ptr %4, double %6)
  %11 = getelementptr %Boxed, ptr %4, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  %13 = fadd double %12, 1.000000e+00
  call void @Boxed.constructor(ptr %8, double %13)
  store ptr %8, ptr %2, align 8
  %14 = load ptr, ptr %2, align 8
  ret ptr %14
}

; Function Attrs: mustprogress
define double @nestedNew(double %0) #1 {
  %2 = alloca <{ i64, { double } }>, align 16
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store i64 -1, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 8
  store { double } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { double }, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @.f_nestedNew..afL77C18FH2918425925, ptr null, ptr null }, ptr %5, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_17318990632888470625_object, i64 40), 2
  %10 = call ptr @.f_nestedNew..afL77C18FH2918425925(ptr %5)
  %11 = getelementptr %Boxed, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  store double %12, ptr %4, align 8
  %13 = load double, ptr %4, align 8
  ret double %13
}

; Function Attrs: mustprogress
define ptr @.f_nestedLambdas..f_.afL85C19FH2918425925..afL86C23FH2918425925(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = getelementptr { double }, ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Boxed, ptr %6, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %7, align 8
  %8 = fadd double %4, 1.000000e+00
  call void @Boxed.constructor(ptr %6, double %8)
  store ptr %6, ptr %2, align 8
  %9 = load ptr, ptr %2, align 8
  ret ptr %9
}

; Function Attrs: mustprogress
define double @.f_nestedLambdas..afL85C19FH2918425925(ptr %0) #1 {
  %2 = alloca <{ i64, { double } }>, align 16
  %3 = alloca double, align 8
  %4 = getelementptr { double }, ptr %0, i32 0, i32 0
  %5 = load double, ptr %4, align 8
  store i64 -1, ptr %2, align 8
  %6 = getelementptr i8, ptr %2, i64 8
  store { double } zeroinitializer, ptr %6, align 8
  %7 = getelementptr { double }, ptr %6, i32 0, i32 0
  store double %5, ptr %7, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @.f_nestedLambdas..f_.afL85C19FH2918425925..afL86C23FH2918425925, ptr null, ptr null }, ptr %6, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_17318990632888470625_object, i64 40), 2
  %10 = call ptr @.f_nestedLambdas..f_.afL85C19FH2918425925..afL86C23FH2918425925(ptr %6)
  %11 = getelementptr %Boxed, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  store double %12, ptr %3, align 8
  %13 = load double, ptr %3, align 8
  ret double %13
}

; Function Attrs: mustprogress
define double @nestedLambdas(double %0) #1 {
  %2 = alloca <{ i64, { double } }>, align 16
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store i64 -1, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 8
  store { double } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { double }, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @.f_nestedLambdas..afL85C19FH2918425925, ptr null, ptr null }, ptr %5, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_17318990632888470625_object, i64 40), 2
  %10 = call double @.f_nestedLambdas..afL85C19FH2918425925(ptr %5)
  store double %10, ptr %4, align 8
  %11 = load double, ptr %4, align 8
  ret double %11
}

; Function Attrs: mustprogress
define i1 @Owner..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Owner..rtti, align 8
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
define ptr @Owner..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Owner, ptr %3, i32 0, i32 0
  store ptr @Owner..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Owner.constructor(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Owner, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define ptr @.f_Owner.build..afL102C22FH2918425925(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = getelementptr { ptr, double }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr { ptr, double }, ptr %0, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Boxed, ptr %8, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %9, align 8
  %10 = getelementptr %Owner, ptr %4, i32 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = fmul double %11, %6
  call void @Boxed.constructor(ptr %8, double %12)
  store ptr %8, ptr %2, align 8
  %13 = load ptr, ptr %2, align 8
  ret ptr %13
}

; Function Attrs: mustprogress
define double @Owner.build(ptr %0, double %1) #1 {
  %3 = alloca <{ i64, { ptr, double } }>, align 16
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca double, align 8
  store ptr %0, ptr %5, align 8
  store double %1, ptr %4, align 8
  %7 = load ptr, ptr %5, align 8
  store i64 -1, ptr %3, align 8
  %8 = getelementptr i8, ptr %3, i64 8
  store { ptr, double } zeroinitializer, ptr %8, align 8
  %9 = getelementptr { ptr, double }, ptr %8, i32 0, i32 0
  %10 = load ptr, ptr %5, align 8
  store ptr %10, ptr %9, align 8
  %11 = getelementptr { ptr, double }, ptr %8, i32 0, i32 1
  %12 = load double, ptr %4, align 8
  store double %12, ptr %11, align 8
  %13 = insertvalue { ptr, ptr, ptr } { ptr @.f_Owner.build..afL102C22FH2918425925, ptr null, ptr null }, ptr %8, 1
  %14 = insertvalue { ptr, ptr, ptr } %13, ptr getelementptr inbounds nuw (i8, ptr @td_13749874419300764099_object, i64 40), 2
  %15 = call ptr @.f_Owner.build..afL102C22FH2918425925(ptr %8)
  %16 = getelementptr %Boxed, ptr %15, i32 0, i32 1
  %17 = load double, ptr %16, align 8
  store double %17, ptr %6, align 8
  %18 = load double, ptr %6, align 8
  ret double %18
}

; Function Attrs: mustprogress
define ptr @.f_alsoReadOutsideTheNew..afL111C18FH2918425925(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  store double %1, ptr %4, align 8
  %6 = getelementptr { double }, ptr %0, i32 0, i32 0
  %7 = load double, ptr %6, align 8
  store double %7, ptr %3, align 8
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Boxed, ptr %9, i32 0, i32 0
  store ptr @Boxed..vtbl, ptr %10, align 8
  %11 = load double, ptr %4, align 8
  %12 = fadd double %7, %11
  %13 = load double, ptr %3, align 8
  %14 = fmul double %13, 0.000000e+00
  %15 = fadd double %12, %14
  call void @Boxed.constructor(ptr %9, double %15)
  store ptr %9, ptr %5, align 8
  %16 = load ptr, ptr %5, align 8
  ret ptr %16
}

; Function Attrs: mustprogress
define double @alsoReadOutsideTheNew(double %0) #1 {
  %2 = alloca <{ i64, { double } }>, align 16
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store i64 -1, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 8
  store { double } zeroinitializer, ptr %5, align 8
  %6 = getelementptr { double }, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @.f_alsoReadOutsideTheNew..afL111C18FH2918425925, ptr null, ptr null }, ptr %5, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_17318990632888470625_object, i64 40), 2
  %10 = call ptr @.f_alsoReadOutsideTheNew..afL111C18FH2918425925(ptr %5, double 1.000000e+00)
  %11 = getelementptr %Boxed, ptr %10, i32 0, i32 1
  %12 = load double, ptr %11, align 8
  store double %12, ptr %4, align 8
  %13 = load double, ptr %4, align 8
  ret double %13
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call double @capturedParam(double 2.000000e+00)
  %2 = fcmp oeq double %1, 3.000000e+00
  br i1 %2, label %3, label %36

3:                                                ; preds = %0
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Color, ptr %5, i32 0, i32 0
  store ptr @Color..vtbl, ptr %6, align 8
  call void @Color.constructor(ptr %5, double 2.000000e+00, double 0.000000e+00)
  %7 = call double @capturedField(ptr %5)
  %8 = fcmp oeq double %7, 3.000000e+00
  br i1 %8, label %9, label %38

9:                                                ; preds = %3
  %10 = call double @capturedLocal()
  %11 = fcmp oeq double %10, 1.100000e+01
  br i1 %11, label %12, label %40

12:                                               ; preds = %9
  %13 = call double @capturedMutableLocal()
  %14 = fcmp oeq double %13, 2.100000e+01
  br i1 %14, label %15, label %42

15:                                               ; preds = %12
  %16 = call double @severalArgumentsAndCaptures(double 3.000000e+00, double 4.000000e+00)
  %17 = fcmp oeq double %16, 3.400000e+01
  br i1 %17, label %18, label %44

18:                                               ; preds = %15
  %19 = call double @nestedNew(double 2.000000e+00)
  %20 = fcmp oeq double %19, 3.000000e+00
  br i1 %20, label %21, label %46

21:                                               ; preds = %18
  %22 = call double @nestedLambdas(double 2.000000e+00)
  %23 = fcmp oeq double %22, 3.000000e+00
  br i1 %23, label %24, label %48

24:                                               ; preds = %21
  %25 = call ptr @GC_malloc(i64 24)
  %26 = getelementptr i8, ptr %25, i64 8
  %27 = getelementptr %Owner, ptr %26, i32 0, i32 0
  store ptr @Owner..vtbl, ptr %27, align 8
  call void @Owner.constructor(ptr %26, double 3.000000e+00)
  %28 = getelementptr %Owner, ptr %26, i32 0, i32 0
  %29 = call double @Owner.build(ptr %26, double 2.000000e+00)
  %30 = fcmp oeq double %29, 6.000000e+00
  br i1 %30, label %31, label %50

31:                                               ; preds = %24
  %32 = call double @alsoReadOutsideTheNew(double 2.000000e+00)
  %33 = fcmp oeq double %32, 3.000000e+00
  br i1 %33, label %34, label %52

34:                                               ; preds = %31
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

35:                                               ; preds = %52, %50, %48, %46, %44, %42, %40, %38, %36
  unreachable

36:                                               ; preds = %0
  %37 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1177635204362931792, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 120, ptr null)
  br label %35

38:                                               ; preds = %3
  %39 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16719205305887358728, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 121, ptr null)
  br label %35

40:                                               ; preds = %9
  %41 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17325684577615475286, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 122, ptr null)
  br label %35

42:                                               ; preds = %12
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7406480972552250788, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 123, ptr null)
  br label %35

44:                                               ; preds = %15
  %45 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11473572904547755511, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 124, ptr null)
  br label %35

46:                                               ; preds = %18
  %47 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8310380780962149797, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 125, ptr null)
  br label %35

48:                                               ; preds = %21
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14565991458262343450, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 126, ptr null)
  br label %35

50:                                               ; preds = %24
  %51 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10636315116849094793, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 127, ptr null)
  br label %35

52:                                               ; preds = %31
  %53 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15985146957802969304, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12536250055124535363, i64 8), i32 128, ptr null)
  br label %35
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

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

