; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Strip = type { ptr, ptr, double }
%Frame = type { %Square, double }
%Square = type { ptr, ptr, double }
%Disc = type { ptr, ptr, double }
%"ts.final:Shape" = type { ptr }

$Square..size = comdat any

$Disc..size = comdat any

$Frame..size = comdat any

$Strip..size = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_7368384151532411686 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFSquare\00", align 8
@s_17641053234615172556 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFDisc\00", align 8
@s_17228831225763461510 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFrame\00", align 8
@s_2174293125769639825 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFStrip\00", align 8
@s_8684338226774033990 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFsquare\00", align 8
@s_14574945162230725868 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdisc\00", align 8
@s_15624169452398137894 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFframe\00", align 8
@s_6804020590496539953 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstrip\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_13374536976187281016 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00final_interface.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_4336704786460529575 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFsum over every implementer\00", align 8
@m_9676383991544082989 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class that declares it\00", align 8
@m_12719141601850817940 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFanother\00", align 8
@m_4479399679609512457 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa derived class's override\00", align 8
@m_835931038001039502 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa class that does not declare it\00", align 8
@m_12702471791481906645 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull throws\00", align 8
@m_11562344213036158520 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Square..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7368384151532411686, i64 8)
@Square..size = global i64 24, comdat
@Square.Shape..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_1571629035032042005_class, i64 40), ptr @Square..instanceOf, ptr @Square.area, ptr @Square.scaled, ptr inttoptr (i64 8 to ptr) }
@td_1571629035032042005_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_1571629035032042005, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Square..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Square..instanceOf, ptr @Square.Shape..vtbl, ptr @Square..new, ptr @Square.area, ptr @Square.scaled, ptr @Square..rtti, ptr @Square..size }
@Disc..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_17641053234615172556, i64 8)
@Disc..size = global i64 24, comdat
@Disc.Shape..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_15245147577379482130_class, i64 40), ptr @Disc..instanceOf, ptr @Disc.area, ptr @Disc.scaled, ptr inttoptr (i64 8 to ptr) }
@td_15245147577379482130_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_15245147577379482130, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Disc..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Disc..instanceOf, ptr @Disc.Shape..vtbl, ptr @Disc..new, ptr @Disc.area, ptr @Disc.scaled, ptr @Disc..rtti, ptr @Disc..size }
@Frame..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_17228831225763461510, i64 8)
@Frame..size = global i64 32, comdat
@Frame.Shape..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_8880372917725371992_class, i64 40), ptr @Frame..instanceOf, ptr @Frame.area, ptr @Square.scaled, ptr inttoptr (i64 8 to ptr) }
@td_8880372917725371992_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_8880372917725371992, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Frame..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Frame..instanceOf, ptr @Frame.Shape..vtbl, ptr @Frame..new, ptr @Frame.area, ptr @Square.scaled, ptr @Square..rtti, ptr @Square..size, ptr @Frame..rtti, ptr @Frame..size }
@Strip..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_2174293125769639825, i64 8)
@Strip..size = global i64 24, comdat
@Strip..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Strip..instanceOf, ptr @Strip..new, ptr @Strip.area, ptr @Strip.scaled, ptr @Strip..rtti, ptr @Strip..size }
@_ZTIPc = external constant ptr
@Shape.null..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @Shape.null.area, ptr @Shape.null.scaled, ptr null }
@Strip.Shape..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_4080704330802167619_class, i64 40), ptr @Strip.Shape..vtbl..instanceOf.dispatch, ptr @Strip.Shape..vtbl.area.dispatch, ptr @Strip.Shape..vtbl.scaled.dispatch, ptr inttoptr (i64 8 to ptr) }
@td_4080704330802167619_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_4080704330802167619, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
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
@a_0_2683940359019178161 = internal constant [0 x { ptr, ptr }] undef
@"ts.final:Shape:Square.Shape..vtbl" = internal constant ptr @Square.Shape..vtbl
@"ts.final:Shape:Disc.Shape..vtbl" = internal constant ptr @Disc.Shape..vtbl
@"ts.final:Shape:Frame.Shape..vtbl" = internal constant ptr @Frame.Shape..vtbl
@"ts.final:Shape:Shape.null..vtbl" = internal constant ptr @Shape.null..vtbl
@"ts.final:Shape:Strip.Shape..vtbl" = internal constant ptr @Strip.Shape..vtbl

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

declare void @__cxa_end_catch()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_4080704330802167619(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Strip, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrel_8880372917725371992(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Frame, ptr %2, i32 0, i32 0
  call void @tsrel_14325905248704939953(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrel_15245147577379482130(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Disc, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
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

define internal void @tsrel_14325905248704939953(ptr %0) {
  %2 = getelementptr %Square, ptr %0, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %2)
  ret void
}

define internal void @tsrel_1571629035032042005(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %14

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %14

6:                                                ; preds = %4
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, @Frame..vtbl
  br i1 %8, label %9, label %11

9:                                                ; preds = %6
  %10 = getelementptr %Frame, ptr %2, i32 0, i32 0
  call void @tsrel_14325905248704939953(ptr %10)
  br label %13

11:                                               ; preds = %6
  %12 = getelementptr %Square, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %12)
  br label %13

13:                                               ; preds = %9, %11
  call void @__tslang_free_block(ptr %2)
  br label %14

14:                                               ; preds = %13, %4, %1
  ret void
}

; Function Attrs: mustprogress
define i1 @Square..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Square..rtti, align 8
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
define ptr @Square..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Square, ptr %3, i32 0, i32 0
  store ptr @Square..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Square.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Square, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_8684338226774033990, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Square, ptr %7, i32 0, i32 2
  %9 = load double, ptr %3, align 8
  store double %9, ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Square.area(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Square, ptr %4, i32 0, i32 2
  %6 = load double, ptr %5, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %Square, ptr %7, i32 0, i32 2
  %9 = load double, ptr %8, align 8
  %10 = fmul double %6, %9
  store double %10, ptr %3, align 8
  %11 = load double, ptr %3, align 8
  ret double %11
}

; Function Attrs: mustprogress
define double @Square.scaled(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr %Square, ptr %6, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr ptr, ptr %8, i32 3
  %10 = load ptr, ptr %9, align 8
  %11 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %10, 0
  %12 = insertvalue { ptr, ptr, ptr } %11, ptr %6, 1
  %13 = insertvalue { ptr, ptr, ptr } %12, ptr null, 2
  %14 = extractvalue { ptr, ptr, ptr } %13, 1
  %15 = extractvalue { ptr, ptr, ptr } %13, 0
  %16 = call double %15(ptr %14)
  %17 = load double, ptr %3, align 8
  %18 = fmul double %16, %17
  store double %18, ptr %5, align 8
  %19 = load double, ptr %5, align 8
  ret double %19
}

; Function Attrs: mustprogress
define i1 @Disc..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Disc..rtti, align 8
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
define ptr @Disc..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Disc, ptr %3, i32 0, i32 0
  store ptr @Disc..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Disc.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Disc, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_14574945162230725868, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Disc, ptr %7, i32 0, i32 2
  %9 = load double, ptr %3, align 8
  store double %9, ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Disc.area(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Disc, ptr %4, i32 0, i32 2
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, 3.000000e+00
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr %Disc, ptr %8, i32 0, i32 2
  %10 = load double, ptr %9, align 8
  %11 = fmul double %7, %10
  store double %11, ptr %3, align 8
  %12 = load double, ptr %3, align 8
  ret double %12
}

; Function Attrs: mustprogress
define double @Disc.scaled(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr %Disc, ptr %6, i32 0, i32 0
  %8 = call double @Disc.area(ptr %6)
  %9 = load double, ptr %3, align 8
  %10 = fmul double %8, %9
  store double %10, ptr %5, align 8
  %11 = load double, ptr %5, align 8
  ret double %11
}

; Function Attrs: mustprogress
define i1 @Frame..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @Frame..rtti, align 8
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
  %26 = getelementptr %Frame, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %3, align 8
  %28 = call i1 @Square..instanceOf(ptr %26, ptr %27)
  br label %29

29:                                               ; preds = %24, %23
  %30 = phi i1 [ %28, %24 ], [ %22, %23 ]
  store i1 %30, ptr %5, align 1
  %31 = load i1, ptr %5, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define ptr @Frame..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Frame, ptr %3, i32 0, i32 0
  %5 = getelementptr %Square, ptr %4, i32 0, i32 0
  store ptr @Frame..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define void @Frame.constructor(ptr %0, double %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store double %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %Frame, ptr %7, i32 0, i32 0
  %9 = load double, ptr %5, align 8
  call void @Square.constructor(ptr %8, double %9)
  %10 = load ptr, ptr %6, align 8
  %11 = getelementptr %Frame, ptr %10, i32 0, i32 1
  %12 = load double, ptr %4, align 8
  store double %12, ptr %11, align 8
  %13 = load ptr, ptr %6, align 8
  %14 = getelementptr %Frame, ptr %13, i32 0, i32 0
  %15 = getelementptr %Square, ptr %14, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_15624169452398137894, i64 8), ptr %15, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Frame.area(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Frame, ptr %4, i32 0, i32 0
  %6 = getelementptr %Square, ptr %5, i32 0, i32 2
  %7 = load double, ptr %6, align 8
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr %Frame, ptr %8, i32 0, i32 0
  %10 = getelementptr %Square, ptr %9, i32 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = fmul double %7, %11
  %13 = load ptr, ptr %2, align 8
  %14 = getelementptr %Frame, ptr %13, i32 0, i32 1
  %15 = load double, ptr %14, align 8
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Frame, ptr %16, i32 0, i32 1
  %18 = load double, ptr %17, align 8
  %19 = fmul double %15, %18
  %20 = fsub double %12, %19
  store double %20, ptr %3, align 8
  %21 = load double, ptr %3, align 8
  ret double %21
}

; Function Attrs: mustprogress
define i1 @Strip..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Strip..rtti, align 8
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
define ptr @Strip..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 32)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Strip, ptr %3, i32 0, i32 0
  store ptr @Strip..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Strip.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Strip, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_6804020590496539953, i64 8), ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Strip, ptr %7, i32 0, i32 2
  %9 = load double, ptr %3, align 8
  store double %9, ptr %8, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Strip.area(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Strip, ptr %4, i32 0, i32 2
  %6 = load double, ptr %5, align 8
  store double %6, ptr %3, align 8
  %7 = load double, ptr %3, align 8
  ret double %7
}

; Function Attrs: mustprogress
define double @Strip.scaled(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr %Strip, ptr %6, i32 0, i32 2
  %8 = load double, ptr %7, align 8
  %9 = load double, ptr %3, align 8
  %10 = fmul double %8, %9
  store double %10, ptr %5, align 8
  %11 = load double, ptr %5, align 8
  ret double %11
}

; Function Attrs: mustprogress
define double @Shape.null.area(ptr %0) #3 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define double @Shape.null.scaled(ptr %0, double %1) #3 {
  %3 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2519477130935904862, i64 8), ptr %3, align 8
  call void @__cxa_throw(ptr %3, ptr @_ZTIPc, ptr null)
  br label %4

4:                                                ; preds = %2
  unreachable
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define internal i1 @Strip.Shape..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  %4 = getelementptr %Strip, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = call i1 %7(ptr %0, ptr %1)
  store i1 %8, ptr %3, align 1
  %9 = load i1, ptr %3, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define internal double @Strip.Shape..vtbl.area.dispatch(ptr %0) #2 {
  %2 = alloca double, align 8
  %3 = getelementptr %Strip, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr ptr, ptr %4, i32 2
  %6 = load ptr, ptr %5, align 8
  %7 = call double %6(ptr %0)
  store double %7, ptr %2, align 8
  %8 = load double, ptr %2, align 8
  ret double %8
}

; Function Attrs: mustprogress
define internal double @Strip.Shape..vtbl.scaled.dispatch(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = getelementptr %Strip, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 3
  %7 = load ptr, ptr %6, align 8
  %8 = call double %7(ptr %0, double %1)
  store double %8, ptr %3, align 8
  %9 = load double, ptr %3, align 8
  ret double %9
}

; Function Attrs: mustprogress noinline
define { ptr, ptr } @make(double %0) #4 {
  %2 = alloca double, align 8
  %3 = alloca { ptr, ptr }, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fcmp oeq double %4, 0.000000e+00
  br i1 %5, label %6, label %22

6:                                                ; preds = %1
  %7 = call ptr @GC_malloc(i64 32)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Square, ptr %8, i32 0, i32 0
  store ptr @Square..vtbl, ptr %9, align 8
  call void @Square.constructor(ptr %8, double 2.000000e+00)
  %10 = ptrtoint ptr %8 to i64
  %11 = icmp ne i64 %10, 0
  br i1 %11, label %12, label %17

12:                                               ; preds = %6
  %13 = getelementptr %Square, ptr %8, i32 0, i32 0
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr ptr, ptr %14, i32 1
  %16 = load ptr, ptr %15, align 8
  br label %18

17:                                               ; preds = %6
  br label %18

18:                                               ; preds = %12, %17
  %19 = phi ptr [ @Shape.null..vtbl, %17 ], [ %16, %12 ]
  %20 = insertvalue { ptr, ptr } undef, ptr %19, 0
  %21 = insertvalue { ptr, ptr } %20, ptr %8, 1
  store { ptr, ptr } %21, ptr %3, align 8
  br label %73

22:                                               ; preds = %1
  %23 = load double, ptr %2, align 8
  %24 = fcmp oeq double %23, 1.000000e+00
  br i1 %24, label %25, label %41

25:                                               ; preds = %22
  %26 = call ptr @GC_malloc(i64 32)
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr %Disc, ptr %27, i32 0, i32 0
  store ptr @Disc..vtbl, ptr %28, align 8
  call void @Disc.constructor(ptr %27, double 1.000000e+00)
  %29 = ptrtoint ptr %27 to i64
  %30 = icmp ne i64 %29, 0
  br i1 %30, label %31, label %36

31:                                               ; preds = %25
  %32 = getelementptr %Disc, ptr %27, i32 0, i32 0
  %33 = load ptr, ptr %32, align 8
  %34 = getelementptr ptr, ptr %33, i32 1
  %35 = load ptr, ptr %34, align 8
  br label %37

36:                                               ; preds = %25
  br label %37

37:                                               ; preds = %31, %36
  %38 = phi ptr [ @Shape.null..vtbl, %36 ], [ %35, %31 ]
  %39 = insertvalue { ptr, ptr } undef, ptr %38, 0
  %40 = insertvalue { ptr, ptr } %39, ptr %27, 1
  store { ptr, ptr } %40, ptr %3, align 8
  br label %73

41:                                               ; preds = %22
  %42 = load double, ptr %2, align 8
  %43 = fcmp oeq double %42, 2.000000e+00
  br i1 %43, label %44, label %57

44:                                               ; preds = %41
  %45 = call ptr @GC_malloc(i64 40)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr %Frame, ptr %46, i32 0, i32 0
  %48 = getelementptr %Square, ptr %47, i32 0, i32 0
  store ptr @Frame..vtbl, ptr %48, align 8
  call void @Frame.constructor(ptr %46, double 3.000000e+00, double 1.000000e+00)
  %49 = ptrtoint ptr %46 to i64
  %50 = icmp ne i64 %49, 0
  br i1 %50, label %51, label %52

51:                                               ; preds = %44
  br label %53

52:                                               ; preds = %44
  br label %53

53:                                               ; preds = %51, %52
  %54 = phi ptr [ @Shape.null..vtbl, %52 ], [ @Frame.Shape..vtbl, %51 ]
  %55 = insertvalue { ptr, ptr } undef, ptr %54, 0
  %56 = insertvalue { ptr, ptr } %55, ptr %46, 1
  store { ptr, ptr } %56, ptr %3, align 8
  br label %73

57:                                               ; preds = %41
  %58 = load double, ptr %2, align 8
  %59 = fcmp oeq double %58, 3.000000e+00
  br i1 %59, label %60, label %72

60:                                               ; preds = %57
  %61 = call ptr @GC_malloc(i64 32)
  %62 = getelementptr i8, ptr %61, i64 8
  %63 = getelementptr %Strip, ptr %62, i32 0, i32 0
  store ptr @Strip..vtbl, ptr %63, align 8
  call void @Strip.constructor(ptr %62, double 5.000000e+00)
  %64 = ptrtoint ptr %62 to i64
  %65 = icmp ne i64 %64, 0
  br i1 %65, label %66, label %67

66:                                               ; preds = %60
  br label %68

67:                                               ; preds = %60
  br label %68

68:                                               ; preds = %66, %67
  %69 = phi ptr [ @Shape.null..vtbl, %67 ], [ @Strip.Shape..vtbl, %66 ]
  %70 = insertvalue { ptr, ptr } undef, ptr %69, 0
  %71 = insertvalue { ptr, ptr } %70, ptr %62, 1
  store { ptr, ptr } %71, ptr %3, align 8
  br label %73

72:                                               ; preds = %57
  store { ptr, ptr } { ptr @Shape.null..vtbl, ptr null }, ptr %3, align 8
  br label %73

73:                                               ; preds = %18, %37, %53, %68, %72
  %74 = load { ptr, ptr }, ptr %3, align 8
  ret { ptr, ptr } %74
}

; Function Attrs: mustprogress noinline
define double @total(ptr %0) #4 {
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
  br i1 %14, label %15, label %49

15:                                               ; preds = %6
  %16 = load i32, ptr %2, align 4
  %17 = load ptr, ptr %4, align 8
  %18 = icmp eq ptr %17, null
  %19 = select i1 %18, ptr @__ts_empty_array, ptr %17
  %20 = getelementptr { ptr, i64, i64 }, ptr %19, i32 0, i32 0
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr { ptr, ptr }, ptr %21, i32 %16
  %23 = load { ptr, ptr }, ptr %22, align 8
  %24 = load double, ptr %3, align 8
  %25 = extractvalue { ptr, ptr } %23, 0
  %26 = extractvalue { ptr, ptr } %23, 1
  %27 = getelementptr %"ts.final:Shape", ptr %25, i32 2
  %28 = load ptr, ptr %27, align 8
  %29 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %28, 0
  %30 = insertvalue { ptr, ptr, ptr } %29, ptr %26, 1
  %31 = insertvalue { ptr, ptr, ptr } %30, ptr null, 2
  %32 = extractvalue { ptr, ptr, ptr } %31, 1
  %33 = extractvalue { ptr, ptr, ptr } %31, 0
  %34 = call double %33(ptr %32)
  %35 = extractvalue { ptr, ptr } %23, 0
  %36 = extractvalue { ptr, ptr } %23, 1
  %37 = getelementptr %"ts.final:Shape", ptr %35, i32 3
  %38 = load ptr, ptr %37, align 8
  %39 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %38, 0
  %40 = insertvalue { ptr, ptr, ptr } %39, ptr %36, 1
  %41 = insertvalue { ptr, ptr, ptr } %40, ptr null, 2
  %42 = extractvalue { ptr, ptr, ptr } %41, 1
  %43 = extractvalue { ptr, ptr, ptr } %41, 0
  %44 = call double %43(ptr %42, double 1.000000e+01)
  %45 = fadd double %34, %44
  %46 = fadd double %24, %45
  store double %46, ptr %3, align 8
  %47 = load i32, ptr %2, align 4
  %48 = add i32 %47, 1
  store i32 %48, ptr %2, align 4
  br label %6

49:                                               ; preds = %6
  %50 = load double, ptr %3, align 8
  store double %50, ptr %5, align 8
  %51 = load double, ptr %5, align 8
  ret double %51
}

; Function Attrs: mustprogress noinline
define double @areaOf({ ptr, ptr } %0) #5 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca { ptr, ptr }, align 8
  %5 = alloca double, align 8
  store { ptr, ptr } %0, ptr %4, align 8
  %6 = load { ptr, ptr }, ptr %4, align 8
  %7 = extractvalue { ptr, ptr } %6, 0
  %8 = extractvalue { ptr, ptr } %6, 1
  %9 = getelementptr %"ts.final:Shape", ptr %7, i32 2
  %10 = load ptr, ptr %9, align 8
  %11 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %10, 0
  %12 = insertvalue { ptr, ptr, ptr } %11, ptr %8, 1
  %13 = insertvalue { ptr, ptr, ptr } %12, ptr null, 2
  %14 = extractvalue { ptr, ptr, ptr } %13, 1
  %15 = extractvalue { ptr, ptr, ptr } %13, 0
  %16 = invoke double %15(ptr %14)
          to label %17 unwind label %18

17:                                               ; preds = %1
  store double %16, ptr %5, align 8
  br label %157

18:                                               ; preds = %1
  %19 = landingpad { ptr, i32 }
          catch ptr null
  %20 = extractvalue { ptr, i32 } %19, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  %22 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %18
  %23 = icmp eq ptr %22, @_ZTIi
  br i1 %23, label %24, label %31

24:                                               ; preds = %.noexc4
  %25 = load i32, ptr %21, align 4
  %26 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %24
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 0
  store i64 4, ptr %28, align 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %27, i32 0, i32 2
  store i32 %25, ptr %30, align 4
  br label %154

31:                                               ; preds = %.noexc4
  %32 = icmp eq ptr %22, @_ZTIj
  br i1 %32, label %33, label %40

33:                                               ; preds = %31
  %34 = load i32, ptr %21, align 4
  %35 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %33
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
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %42
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
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %51
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
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %60
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
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %69
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
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %78
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
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %101
  %102 = load ptr, ptr %2, align 8
  br label %110

103:                                              ; preds = %97
  %104 = load ptr, ptr %21, align 8
  %105 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %103
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 0
  store i64 8, ptr %107, align 8
  %108 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %108, align 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %106, i32 0, i32 2
  store ptr %104, ptr %109, align 8
  br label %110

110:                                              ; preds = %.noexc10, %.noexc8
  %111 = phi ptr [ %106, %.noexc8 ], [ %102, %.noexc10 ]
  br label %112

112:                                              ; preds = %110
  br label %120

113:                                              ; preds = %92
  %114 = load ptr, ptr %21, align 8
  %115 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %113
  %116 = getelementptr i8, ptr %115, i64 8
  %117 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 0
  store i64 8, ptr %117, align 8
  %118 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %118, align 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %116, i32 0, i32 2
  store ptr %114, ptr %119, align 8
  br label %120

120:                                              ; preds = %112, %.noexc5
  %121 = phi ptr [ %116, %.noexc5 ], [ %111, %112 ]
  br label %122

122:                                              ; preds = %120
  br label %130

123:                                              ; preds = %89
  %124 = load ptr, ptr %21, align 8
  %125 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %123
  %126 = getelementptr i8, ptr %125, i64 8
  %127 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 0
  store i64 8, ptr %127, align 8
  %128 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %128, align 8
  %129 = getelementptr { i64, ptr, ptr }, ptr %126, i32 0, i32 2
  store ptr %124, ptr %129, align 8
  br label %130

130:                                              ; preds = %122, %.noexc3
  %131 = phi ptr [ %126, %.noexc3 ], [ %121, %122 ]
  br label %132

132:                                              ; preds = %130
  br label %133

133:                                              ; preds = %87, %132
  %134 = phi ptr [ %131, %132 ], [ %88, %87 ]
  br label %135

135:                                              ; preds = %133
  br label %136

136:                                              ; preds = %.noexc9, %135
  %137 = phi ptr [ %134, %135 ], [ %81, %.noexc9 ]
  br label %138

138:                                              ; preds = %136
  br label %139

139:                                              ; preds = %.noexc1, %138
  %140 = phi ptr [ %137, %138 ], [ %72, %.noexc1 ]
  br label %141

141:                                              ; preds = %139
  br label %142

142:                                              ; preds = %.noexc2, %141
  %143 = phi ptr [ %140, %141 ], [ %63, %.noexc2 ]
  br label %144

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %.noexc7, %144
  %146 = phi ptr [ %143, %144 ], [ %54, %.noexc7 ]
  br label %147

147:                                              ; preds = %145
  br label %148

148:                                              ; preds = %.noexc11, %147
  %149 = phi ptr [ %146, %147 ], [ %45, %.noexc11 ]
  br label %150

150:                                              ; preds = %148
  br label %151

151:                                              ; preds = %.noexc, %150
  %152 = phi ptr [ %149, %150 ], [ %36, %.noexc ]
  br label %153

153:                                              ; preds = %151
  br label %154

154:                                              ; preds = %.noexc6, %153
  %155 = phi ptr [ %152, %153 ], [ %27, %.noexc6 ]
  br label %156

156:                                              ; preds = %154
  store ptr %155, ptr %3, align 8
  store double -1.000000e+00, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %157

157:                                              ; preds = %17, %156
  %158 = load double, ptr %5, align 8
  ret double %158

endcatch.resume:                                  ; preds = %42, %101, %78, %103, %51, %24, %113, %18, %123, %60, %69, %33
  %159 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %159
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 8)
  %4 = getelementptr i8, ptr %3, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_2683940359019178161, i64 0, i1 false)
  %5 = call ptr @GC_malloc(i64 32)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 0
  store ptr %4, ptr %7, align 8
  %8 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 1
  store i64 0, ptr %8, align 8
  %9 = getelementptr { ptr, i64, i64 }, ptr %6, i32 0, i32 2
  store i64 0, ptr %9, align 8
  store ptr %6, ptr %2, align 8
  store i32 0, ptr %1, align 4
  br label %10

10:                                               ; preds = %61, %0
  %11 = load i32, ptr %1, align 4
  %12 = icmp slt i32 %11, 4
  br i1 %12, label %13, label %65

13:                                               ; preds = %10
  %14 = load i32, ptr %1, align 4
  %15 = sitofp i32 %14 to double
  %16 = call { ptr, ptr } @make(double %15)
  %17 = load ptr, ptr %2, align 8
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %22

19:                                               ; preds = %13
  %20 = call ptr @GC_malloc(i64 32)
  %21 = getelementptr i8, ptr %20, i64 8
  store ptr %21, ptr %2, align 8
  br label %23

22:                                               ; preds = %13
  br label %23

23:                                               ; preds = %19, %22
  %24 = phi ptr [ %17, %22 ], [ %21, %19 ]
  br label %25

25:                                               ; preds = %23
  %26 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 1
  %27 = load i64, ptr %26, align 8
  %28 = add i64 %27, 1
  %29 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 0
  %30 = getelementptr { ptr, i64, i64 }, ptr %24, i32 0, i32 2
  %31 = load ptr, ptr %29, align 8
  %32 = load i64, ptr %30, align 8
  %33 = icmp ugt i64 %28, %32
  br i1 %33, label %34, label %58

34:                                               ; preds = %25
  %35 = mul i64 %32, 2
  %36 = icmp uge i64 %35, %28
  %37 = select i1 %36, i64 %35, i64 %28
  %38 = icmp ult i64 %37, 4
  %39 = select i1 %38, i64 4, i64 %37
  %40 = icmp ule i64 %28, 4294967295
  br i1 %40, label %41, label %132

41:                                               ; preds = %34
  %42 = icmp ule i64 %39, 1152921504606846975
  br i1 %42, label %43, label %134

43:                                               ; preds = %41
  %44 = mul i64 16, %39
  %45 = add i64 %44, 8
  %46 = icmp eq ptr %31, null
  %47 = getelementptr i8, ptr %31, i64 -8
  %48 = select i1 %46, ptr null, ptr %47
  %49 = call ptr @GC_realloc(ptr %48, i64 %45)
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = icmp ne ptr %50, null
  %52 = icmp eq i64 %44, 0
  %53 = or i1 %51, %52
  br i1 %53, label %54, label %136

54:                                               ; preds = %43
  %55 = mul i64 16, %32
  %56 = getelementptr i8, ptr %50, i64 %55
  %57 = sub i64 %44, %55
  call void @llvm.memset.p0.i64(ptr %56, i8 0, i64 %57, i1 false)
  store ptr %50, ptr %29, align 8
  store i64 %39, ptr %30, align 8
  br label %59

58:                                               ; preds = %25
  br label %59

59:                                               ; preds = %54, %58
  %60 = phi ptr [ %31, %58 ], [ %50, %54 ]
  br label %61

61:                                               ; preds = %59
  %62 = getelementptr { ptr, ptr }, ptr %60, i64 %27
  store { ptr, ptr } %16, ptr %62, align 8
  store i64 %28, ptr %26, align 8
  %63 = load i32, ptr %1, align 4
  %64 = add i32 %63, 1
  store i32 %64, ptr %1, align 4
  br label %10

65:                                               ; preds = %10
  %66 = load ptr, ptr %2, align 8
  %67 = call double @total(ptr %66)
  %68 = fcmp oeq double %67, 2.200000e+02
  br i1 %68, label %69, label %138

69:                                               ; preds = %65
  %70 = call { ptr, ptr } @make(double 0.000000e+00)
  %71 = call double @areaOf({ ptr, ptr } %70)
  %72 = fcmp oeq double %71, 4.000000e+00
  br i1 %72, label %73, label %140

73:                                               ; preds = %69
  %74 = call { ptr, ptr } @make(double 1.000000e+00)
  %75 = call double @areaOf({ ptr, ptr } %74)
  %76 = fcmp oeq double %75, 3.000000e+00
  br i1 %76, label %77, label %142

77:                                               ; preds = %73
  %78 = call { ptr, ptr } @make(double 2.000000e+00)
  %79 = call double @areaOf({ ptr, ptr } %78)
  %80 = fcmp oeq double %79, 8.000000e+00
  br i1 %80, label %81, label %144

81:                                               ; preds = %77
  %82 = call { ptr, ptr } @make(double 3.000000e+00)
  %83 = call double @areaOf({ ptr, ptr } %82)
  %84 = fcmp oeq double %83, 5.000000e+00
  br i1 %84, label %85, label %146

85:                                               ; preds = %81
  %86 = call { ptr, ptr } @make(double 4.000000e+00)
  %87 = call double @areaOf({ ptr, ptr } %86)
  %88 = fcmp oeq double %87, -1.000000e+00
  br i1 %88, label %89, label %148

89:                                               ; preds = %85
  %90 = call { ptr, ptr } @make(double 2.000000e+00)
  %91 = extractvalue { ptr, ptr } %90, 0
  %92 = extractvalue { ptr, ptr } %90, 1
  %93 = getelementptr ptr, ptr %91, i32 4
  %94 = load ptr, ptr %93, align 8
  %95 = ptrtoint ptr %92 to i64
  %96 = ptrtoint ptr %94 to i64
  %97 = add i64 %95, %96
  %98 = inttoptr i64 %97 to ptr
  %99 = extractvalue { ptr, ptr } %90, 0
  %100 = extractvalue { ptr, ptr } %90, 1
  %101 = getelementptr ptr, ptr %99, i32 4
  %102 = load ptr, ptr %101, align 8
  %103 = ptrtoint ptr %102 to i64
  %104 = and i64 %103, -9223372036854775808
  %105 = icmp ne i64 %104, 0
  br i1 %105, label %106, label %112

106:                                              ; preds = %89
  %107 = sub i64 %103, -9223372036854775808
  %108 = inttoptr i64 %107 to ptr
  %109 = getelementptr { ptr, ptr }, ptr %108, i32 0, i32 0
  %110 = load ptr, ptr %109, align 8
  %111 = call ptr %110(ptr %100)
  br label %114

112:                                              ; preds = %89
  %113 = load ptr, ptr %98, align 8
  br label %114

114:                                              ; preds = %106, %112
  %115 = phi ptr [ %113, %112 ], [ %111, %106 ]
  br label %116

116:                                              ; preds = %114
  %117 = ptrtoint ptr %115 to i64
  %118 = icmp ne i64 %117, 0
  %119 = zext i1 %118 to i32
  %120 = and i32 %119, 1
  %121 = icmp ne i32 %120, 0
  br i1 %121, label %122, label %125

122:                                              ; preds = %116
  %123 = call i32 @strcmp(ptr %115, ptr getelementptr inbounds nuw (i8, ptr @s_15624169452398137894, i64 8))
  %124 = icmp eq i32 %123, 0
  br label %127

125:                                              ; preds = %116
  %126 = icmp eq i64 %117, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15624169452398137894, i64 8) to i64)
  br label %127

127:                                              ; preds = %122, %125
  %128 = phi i1 [ %126, %125 ], [ %124, %122 ]
  br label %129

129:                                              ; preds = %127
  br i1 %128, label %130, label %150

130:                                              ; preds = %129
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

131:                                              ; preds = %136, %134, %132, %150, %148, %146, %144, %142, %140, %138
  unreachable

132:                                              ; preds = %34
  %133 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 72, ptr null)
  br label %131

134:                                              ; preds = %41
  %135 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 72, ptr null)
  br label %131

136:                                              ; preds = %43
  %137 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 72, ptr null)
  br label %131

138:                                              ; preds = %65
  %139 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4336704786460529575, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 76, ptr null)
  br label %131

140:                                              ; preds = %69
  %141 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9676383991544082989, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 78, ptr null)
  br label %131

142:                                              ; preds = %73
  %143 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12719141601850817940, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 79, ptr null)
  br label %131

144:                                              ; preds = %77
  %145 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4479399679609512457, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 80, ptr null)
  br label %131

146:                                              ; preds = %81
  %147 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_835931038001039502, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 81, ptr null)
  br label %131

148:                                              ; preds = %85
  %149 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12702471791481906645, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 82, ptr null)
  br label %131

150:                                              ; preds = %129
  %151 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11562344213036158520, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_13374536976187281016, i64 8), i32 84, ptr null)
  br label %131
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "__null_interface_vtable" }
attributes #4 = { mustprogress noinline }
attributes #5 = { mustprogress noinline "personality" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

