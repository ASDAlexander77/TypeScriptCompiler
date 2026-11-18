; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Plain = type { ptr }
%"Bag<si32>" = type { ptr, ptr }
%"Late<si32>" = type { ptr, ptr }

$"Bag<si32>..instanceOf" = comdat any

$"Bag<si32>..new" = comdat any

$"Bag<si32>.constructor" = comdat any

$"Bag<si32>.sum" = comdat any

$"Bag<si32>.scaled" = comdat any

$".f_Bag<si32>.adder..afL29C16FH2337409642" = comdat any

$"Bag<si32>.adder" = comdat any

$"sumOf<si32>" = comdat any

$"scaledBy<si32>" = comdat any

$"Plain.sum<si32>" = comdat any

$".afL48C18FH2337409642<si32>" = comdat any

$"Late<si32>..instanceOf" = comdat any

$"Late<si32>..new" = comdat any

$"Late<si32>.constructor" = comdat any

$"Late<si32>.sum" = comdat any

$"lateSum<si32>" = comdat any

$".f_main.times<si32>" = comdat any

$Plain..size = comdat any

$"Bag<si32>..rtti" = comdat any

$"Bag<si32>..size" = comdat any

$"Bag<si32>..vtbl" = comdat any

$"Late<si32>..rtti" = comdat any

$"Late<si32>..size" = comdat any

$"Late<si32>..vtbl" = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_1410190387630922087 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFPlain\00", align 8
@s_6188926167365535444 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBag<si32>\00", align 8
@s_15658622148512784356 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFLate<si32>\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_5644151482248669270 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic class method calls the module's function\00", align 8
@f_6317610793784083762 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00tslang_issue_30.ts\00", align 8
@m_7947706393692433882 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFand reads the module's variable\00", align 8
@m_7414518026257394668 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa lambda in a generic captures the generic's local\00", align 8
@m_14896294745222502455 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic function calls the module's function\00", align 8
@m_3117495602205247823 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic function reads the module's variable\00", align 8
@m_5531655364543160778 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic method\00", align 8
@m_1084503071927299744 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic arrow\00", align 8
@m_15172081542469865651 = internal constant [50 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic class declared after the caller\00", align 8
@m_2926673886819752697 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa generic function declared after the caller\00", align 8
@m_11252426188826558099 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe caller's locals are its own\00", align 8
@m_18432555915240551822 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa local generic reads its function's local\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@scale = private global i32 10
@Plain..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1410190387630922087, i64 8)
@Plain..size = global i64 8, comdat
@Plain..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Plain..instanceOf, ptr @Plain..new, ptr @Plain..rtti, ptr @Plain..size }
@"Bag<si32>..rtti" = linkonce_odr global ptr getelementptr inbounds nuw (i8, ptr @s_6188926167365535444, i64 8), comdat
@"Bag<si32>..size" = global i64 16, comdat
@"Bag<si32>..vtbl" = linkonce_odr global { ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @"Bag<si32>..instanceOf", ptr @"Bag<si32>..new", ptr @"Bag<si32>.sum", ptr @"Bag<si32>.scaled", ptr @"Bag<si32>.adder", ptr @"Bag<si32>..rtti", ptr @"Bag<si32>..size" }, comdat
@"Late<si32>..rtti" = linkonce_odr global ptr getelementptr inbounds nuw (i8, ptr @s_15658622148512784356, i64 8), comdat
@"Late<si32>..size" = global i64 16, comdat
@"Late<si32>..vtbl" = linkonce_odr global { ptr, ptr, ptr, ptr, ptr } { ptr @"Late<si32>..instanceOf", ptr @"Late<si32>..new", ptr @"Late<si32>.sum", ptr @"Late<si32>..rtti", ptr @"Late<si32>..size" }, comdat
@td_12934533156908281318_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_12934533156908281318, ptr @tsretcb_12934533156908281318, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_12811134088920325202_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_12811134088920325202, ptr @tsretcb_12811134088920325202, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

define internal void @tsretcb_12811134088920325202(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_12811134088920325202(ptr %0) {
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

define internal void @tsretcb_12934533156908281318(ptr %0) {
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

define internal void @tsrelcb_12934533156908281318(ptr %0) {
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

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define double @total(ptr %0) #1 {
  %2 = alloca i32, align 4
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store double 0.000000e+00, ptr %3, align 8
  store i32 0, ptr %2, align 4
  br label %6

6:                                                ; preds = %15, %1
  %7 = load i32, ptr %2, align 4
  %8 = load ptr, ptr %4, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr @__ts_empty_array, ptr %8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 1
  %12 = load i64, ptr %11, align 8
  %13 = sext i32 %7 to i64
  %14 = icmp slt i64 %13, %12
  br i1 %14, label %15, label %28

15:                                               ; preds = %6
  %16 = load i32, ptr %2, align 4
  %17 = load ptr, ptr %4, align 8
  %18 = icmp eq ptr %17, null
  %19 = select i1 %18, ptr @__ts_empty_array, ptr %17
  %20 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 0
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr double, ptr %21, i32 %16
  %23 = load double, ptr %22, align 8
  %24 = load double, ptr %3, align 8
  %25 = fadd double %24, %23
  store double %25, ptr %3, align 8
  %26 = load i32, ptr %2, align 4
  %27 = add i32 %26, 1
  store i32 %27, ptr %2, align 4
  br label %6

28:                                               ; preds = %6
  %29 = load double, ptr %3, align 8
  store double %29, ptr %5, align 8
  %30 = load double, ptr %5, align 8
  ret double %30
}

; Function Attrs: mustprogress
define i1 @Plain..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Plain..rtti, align 8
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
define ptr @Plain..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Plain, ptr %3, i32 0, i32 0
  store ptr @Plain..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"Bag<si32>..instanceOf"(ptr %0, ptr %1) #1 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @"Bag<si32>..rtti", align 8
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
define linkonce_odr ptr @"Bag<si32>..new"() #1 comdat {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %"Bag<si32>", ptr %3, i32 0, i32 0
  store ptr @"Bag<si32>..vtbl", ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr void @"Bag<si32>.constructor"(ptr %0, ptr %1) #1 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Bag<si32>", ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr double @"Bag<si32>.sum"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %"Bag<si32>", ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = call double @total(ptr %6)
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress
define linkonce_odr double @"Bag<si32>.scaled"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %"Bag<si32>", ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = call double @total(ptr %6)
  %8 = load i32, ptr @scale, align 4
  %9 = sitofp i32 %8 to double
  %10 = fmul double %7, %9
  store double %10, ptr %3, align 8
  %11 = load double, ptr %3, align 8
  ret double %11
}

; Function Attrs: mustprogress
define linkonce_odr double @".f_Bag<si32>.adder..afL29C16FH2337409642"(ptr %0, double %1) #1 comdat {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %1, ptr %3, align 8
  %5 = getelementptr { i32 }, ptr %0, i32 0, i32 0
  %6 = load i32, ptr %5, align 4
  %7 = load double, ptr %3, align 8
  %8 = sitofp i32 %6 to double
  %9 = fadd double %7, %8
  store double %9, ptr %4, align 8
  %10 = load double, ptr %4, align 8
  ret double %10
}

; Function Attrs: mustprogress
define linkonce_odr { ptr, ptr, ptr } @"Bag<si32>.adder"(ptr %0) #1 comdat {
  %2 = alloca { ptr, ptr, ptr }, align 8
  %3 = call ptr @GC_malloc(i64 12)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr { i32 }, ptr %4, i32 0, i32 0
  store i32 100, ptr %5, align 4
  %6 = insertvalue { ptr, ptr, ptr } { ptr @".f_Bag<si32>.adder..afL29C16FH2337409642", ptr null, ptr null }, ptr %4, 1
  %7 = insertvalue { ptr, ptr, ptr } %6, ptr getelementptr inbounds nuw (i8, ptr @td_12934533156908281318_object, i64 40), 2
  store { ptr, ptr, ptr } %7, ptr %2, align 8
  %8 = load { ptr, ptr, ptr }, ptr %2, align 8
  ret { ptr, ptr, ptr } %8
}

; Function Attrs: mustprogress
define linkonce_odr double @"sumOf<si32>"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call double @total(ptr %4)
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  ret double %6
}

; Function Attrs: mustprogress
define linkonce_odr double @"scaledBy<si32>"(double %0) #1 comdat {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = load i32, ptr @scale, align 4
  %6 = sitofp i32 %5 to double
  %7 = fmul double %4, %6
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress
define linkonce_odr double @"Plain.sum<si32>"(ptr %0, ptr %1) #1 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca double, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = call double @total(ptr %5)
  store double %6, ptr %4, align 8
  %7 = load double, ptr %4, align 8
  ret double %7
}

; Function Attrs: mustprogress
define linkonce_odr double @".afL48C18FH2337409642<si32>"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call double @total(ptr %4)
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  ret double %6
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"Late<si32>..instanceOf"(ptr %0, ptr %1) #1 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @"Late<si32>..rtti", align 8
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
define linkonce_odr ptr @"Late<si32>..new"() #1 comdat {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %"Late<si32>", ptr %3, i32 0, i32 0
  store ptr @"Late<si32>..vtbl", ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr void @"Late<si32>.constructor"(ptr %0, ptr %1) #1 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Late<si32>", ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr double @"Late<si32>.sum"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %"Late<si32>", ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = call double @total(ptr %6)
  store double %7, ptr %3, align 8
  %8 = load double, ptr %3, align 8
  ret double %8
}

; Function Attrs: mustprogress
define linkonce_odr double @"lateSum<si32>"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call double @total(ptr %4)
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  ret double %6
}

; Function Attrs: mustprogress
define linkonce_odr double @".f_main.times<si32>"(ptr %0, double %1) #1 comdat {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %1, ptr %3, align 8
  %5 = getelementptr { i32 }, ptr %0, i32 0, i32 0
  %6 = load i32, ptr %5, align 4
  %7 = load double, ptr %3, align 8
  %8 = sitofp i32 %6 to double
  %9 = fmul double %7, %8
  store double %9, ptr %4, align 8
  %10 = load double, ptr %4, align 8
  ret double %10
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca <{ i64, { i32 } }>, align 16
  %2 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %3 = alloca <{ i64, [2 x double] }>, align 16
  %4 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %5 = alloca <{ i64, [2 x double] }>, align 16
  %6 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %7 = alloca <{ i64, [2 x double] }>, align 16
  %8 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %9 = alloca <{ i64, [2 x double] }>, align 16
  %10 = alloca <{ i64, %"Bag<si32>" }>, align 16
  %11 = alloca <{ i64, %Plain }>, align 16
  %12 = alloca <{ i64, %"Late<si32>" }>, align 16
  store i64 -1, ptr %12, align 8
  %13 = getelementptr i8, ptr %12, i64 8
  store %"Late<si32>" zeroinitializer, ptr %13, align 8
  store i64 -1, ptr %11, align 8
  %14 = getelementptr i8, ptr %11, i64 8
  store %Plain zeroinitializer, ptr %14, align 8
  store i64 -1, ptr %10, align 8
  %15 = getelementptr i8, ptr %10, i64 8
  store %"Bag<si32>" zeroinitializer, ptr %15, align 8
  store %"Bag<si32>" zeroinitializer, ptr %15, align 8
  %16 = getelementptr %"Bag<si32>", ptr %15, i32 0, i32 0
  store ptr @"Bag<si32>..vtbl", ptr %16, align 8
  %17 = call ptr @GC_malloc(i64 32)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr double, ptr %18, i64 0
  store double 1.000000e+00, ptr %19, align 8
  %20 = getelementptr double, ptr %18, i64 1
  store double 2.000000e+00, ptr %20, align 8
  %21 = getelementptr double, ptr %18, i64 2
  store double 3.000000e+00, ptr %21, align 8
  %22 = call ptr @GC_malloc(i64 32)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr { ptr, i64, i64 }, ptr %23, i32 0, i32 0
  store ptr %18, ptr %24, align 8
  %25 = getelementptr { ptr, i64, i64 }, ptr %23, i32 0, i32 1
  store i64 3, ptr %25, align 8
  %26 = getelementptr { ptr, i64, i64 }, ptr %23, i32 0, i32 2
  store i64 3, ptr %26, align 8
  call void @"Bag<si32>.constructor"(ptr %15, ptr %23)
  %27 = getelementptr %"Bag<si32>", ptr %15, i32 0, i32 0
  %28 = call double @"Bag<si32>.sum"(ptr %15)
  %29 = fcmp oeq double %28, 6.000000e+00
  br i1 %29, label %30, label %115

30:                                               ; preds = %0
  %31 = getelementptr %"Bag<si32>", ptr %15, i32 0, i32 0
  %32 = call double @"Bag<si32>.scaled"(ptr %15)
  %33 = fcmp oeq double %32, 6.000000e+01
  br i1 %33, label %34, label %117

34:                                               ; preds = %30
  %35 = getelementptr %"Bag<si32>", ptr %15, i32 0, i32 0
  %36 = call { ptr, ptr, ptr } @"Bag<si32>.adder"(ptr %15)
  %37 = extractvalue { ptr, ptr, ptr } %36, 1
  %38 = extractvalue { ptr, ptr, ptr } %36, 0
  %39 = call double %38(ptr %37, double 1.000000e+00)
  %40 = fcmp oeq double %39, 1.010000e+02
  br i1 %40, label %41, label %119

41:                                               ; preds = %34
  store i64 -1, ptr %9, align 8
  %42 = getelementptr i8, ptr %9, i64 8
  store [2 x double] zeroinitializer, ptr %42, align 8
  %43 = getelementptr double, ptr %42, i64 0
  store double 4.000000e+00, ptr %43, align 8
  %44 = getelementptr double, ptr %42, i64 1
  store double 5.000000e+00, ptr %44, align 8
  store i64 -1, ptr %8, align 8
  %45 = getelementptr i8, ptr %8, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %45, align 8
  %46 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 0
  store ptr %42, ptr %46, align 8
  %47 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 1
  store i64 2, ptr %47, align 8
  %48 = getelementptr { ptr, i64, i64 }, ptr %45, i32 0, i32 2
  store i64 2, ptr %48, align 8
  %49 = call double @"sumOf<si32>"(ptr %45)
  %50 = fcmp oeq double %49, 9.000000e+00
  br i1 %50, label %51, label %121

51:                                               ; preds = %41
  %52 = call double @"scaledBy<si32>"(double 3.000000e+00)
  %53 = fcmp oeq double %52, 3.000000e+01
  br i1 %53, label %54, label %123

54:                                               ; preds = %51
  store %Plain zeroinitializer, ptr %14, align 8
  %55 = getelementptr %Plain, ptr %14, i32 0, i32 0
  store ptr @Plain..vtbl, ptr %55, align 8
  store i64 -1, ptr %7, align 8
  %56 = getelementptr i8, ptr %7, i64 8
  store [2 x double] zeroinitializer, ptr %56, align 8
  %57 = getelementptr double, ptr %56, i64 0
  store double 5.000000e+00, ptr %57, align 8
  %58 = getelementptr double, ptr %56, i64 1
  store double 6.000000e+00, ptr %58, align 8
  store i64 -1, ptr %6, align 8
  %59 = getelementptr i8, ptr %6, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %59, align 8
  %60 = getelementptr { ptr, i64, i64 }, ptr %59, i32 0, i32 0
  store ptr %56, ptr %60, align 8
  %61 = getelementptr { ptr, i64, i64 }, ptr %59, i32 0, i32 1
  store i64 2, ptr %61, align 8
  %62 = getelementptr { ptr, i64, i64 }, ptr %59, i32 0, i32 2
  store i64 2, ptr %62, align 8
  %63 = call double @"Plain.sum<si32>"(ptr %14, ptr %59)
  %64 = fcmp oeq double %63, 1.100000e+01
  br i1 %64, label %65, label %125

65:                                               ; preds = %54
  store i64 -1, ptr %5, align 8
  %66 = getelementptr i8, ptr %5, i64 8
  store [2 x double] zeroinitializer, ptr %66, align 8
  %67 = getelementptr double, ptr %66, i64 0
  store double 7.000000e+00, ptr %67, align 8
  %68 = getelementptr double, ptr %66, i64 1
  store double 8.000000e+00, ptr %68, align 8
  store i64 -1, ptr %4, align 8
  %69 = getelementptr i8, ptr %4, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %69, align 8
  %70 = getelementptr { ptr, i64, i64 }, ptr %69, i32 0, i32 0
  store ptr %66, ptr %70, align 8
  %71 = getelementptr { ptr, i64, i64 }, ptr %69, i32 0, i32 1
  store i64 2, ptr %71, align 8
  %72 = getelementptr { ptr, i64, i64 }, ptr %69, i32 0, i32 2
  store i64 2, ptr %72, align 8
  %73 = call double @".afL48C18FH2337409642<si32>"(ptr %69)
  %74 = fcmp oeq double %73, 1.500000e+01
  br i1 %74, label %75, label %127

75:                                               ; preds = %65
  store %"Late<si32>" zeroinitializer, ptr %13, align 8
  %76 = getelementptr %"Late<si32>", ptr %13, i32 0, i32 0
  store ptr @"Late<si32>..vtbl", ptr %76, align 8
  %77 = call ptr @GC_malloc(i64 24)
  %78 = getelementptr i8, ptr %77, i64 8
  %79 = getelementptr double, ptr %78, i64 0
  store double 1.000000e+00, ptr %79, align 8
  %80 = getelementptr double, ptr %78, i64 1
  store double 2.000000e+00, ptr %80, align 8
  %81 = call ptr @GC_malloc(i64 32)
  %82 = getelementptr i8, ptr %81, i64 8
  %83 = getelementptr { ptr, i64, i64 }, ptr %82, i32 0, i32 0
  store ptr %78, ptr %83, align 8
  %84 = getelementptr { ptr, i64, i64 }, ptr %82, i32 0, i32 1
  store i64 2, ptr %84, align 8
  %85 = getelementptr { ptr, i64, i64 }, ptr %82, i32 0, i32 2
  store i64 2, ptr %85, align 8
  call void @"Late<si32>.constructor"(ptr %13, ptr %82)
  %86 = getelementptr %"Late<si32>", ptr %13, i32 0, i32 0
  %87 = call double @"Late<si32>.sum"(ptr %13)
  %88 = fcmp oeq double %87, 3.000000e+00
  br i1 %88, label %89, label %129

89:                                               ; preds = %75
  store i64 -1, ptr %3, align 8
  %90 = getelementptr i8, ptr %3, i64 8
  store [2 x double] zeroinitializer, ptr %90, align 8
  %91 = getelementptr double, ptr %90, i64 0
  store double 3.000000e+00, ptr %91, align 8
  %92 = getelementptr double, ptr %90, i64 1
  store double 4.000000e+00, ptr %92, align 8
  store i64 -1, ptr %2, align 8
  %93 = getelementptr i8, ptr %2, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %93, align 8
  %94 = getelementptr { ptr, i64, i64 }, ptr %93, i32 0, i32 0
  store ptr %90, ptr %94, align 8
  %95 = getelementptr { ptr, i64, i64 }, ptr %93, i32 0, i32 1
  store i64 2, ptr %95, align 8
  %96 = getelementptr { ptr, i64, i64 }, ptr %93, i32 0, i32 2
  store i64 2, ptr %96, align 8
  %97 = call double @"lateSum<si32>"(ptr %93)
  %98 = fcmp oeq double %97, 7.000000e+00
  br i1 %98, label %99, label %131

99:                                               ; preds = %89
  br i1 true, label %100, label %101

100:                                              ; preds = %99
  br label %101

101:                                              ; preds = %100, %99
  %102 = phi i1 [ true, %100 ], [ true, %99 ]
  br i1 %102, label %103, label %104

103:                                              ; preds = %101
  br label %104

104:                                              ; preds = %103, %101
  %105 = phi i1 [ true, %103 ], [ %102, %101 ]
  br i1 %105, label %106, label %133

106:                                              ; preds = %104
  store i64 -1, ptr %1, align 8
  %107 = getelementptr i8, ptr %1, i64 8
  store { i32 } zeroinitializer, ptr %107, align 4
  %108 = getelementptr { i32 }, ptr %107, i32 0, i32 0
  store i32 7, ptr %108, align 4
  %109 = insertvalue { ptr, ptr, ptr } { ptr @".f_main.times<si32>", ptr null, ptr null }, ptr %107, 1
  %110 = insertvalue { ptr, ptr, ptr } %109, ptr getelementptr inbounds nuw (i8, ptr @td_12811134088920325202_object, i64 40), 2
  %111 = call double @".f_main.times<si32>"(ptr %107, double 3.000000e+00)
  %112 = fcmp oeq double %111, 2.100000e+01
  br i1 %112, label %113, label %135

113:                                              ; preds = %106
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

114:                                              ; preds = %135, %133, %131, %129, %127, %125, %123, %121, %119, %117, %115
  unreachable

115:                                              ; preds = %0
  %116 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5644151482248669270, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 57, ptr null)
  br label %114

117:                                              ; preds = %30
  %118 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7947706393692433882, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 58, ptr null)
  br label %114

119:                                              ; preds = %34
  %120 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7414518026257394668, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 59, ptr null)
  br label %114

121:                                              ; preds = %41
  %122 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14896294745222502455, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 60, ptr null)
  br label %114

123:                                              ; preds = %51
  %124 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3117495602205247823, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 61, ptr null)
  br label %114

125:                                              ; preds = %54
  %126 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5531655364543160778, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 62, ptr null)
  br label %114

127:                                              ; preds = %65
  %128 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1084503071927299744, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 63, ptr null)
  br label %114

129:                                              ; preds = %75
  %130 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15172081542469865651, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 64, ptr null)
  br label %114

131:                                              ; preds = %89
  %132 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2926673886819752697, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 65, ptr null)
  br label %114

133:                                              ; preds = %104
  %134 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11252426188826558099, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 66, ptr null)
  br label %114

135:                                              ; preds = %106
  %136 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18432555915240551822, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_6317610793784083762, i64 8), i32 73, ptr null)
  br label %114
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

