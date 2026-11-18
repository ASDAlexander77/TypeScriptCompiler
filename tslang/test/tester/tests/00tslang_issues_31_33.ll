; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Box = type { ptr, i32 }
%Keeper = type { ptr, { ptr, ptr, ptr } }
%"Start<!ts.class<Box, !ts.class_storage<Box>>>" = type { ptr, ptr, { ptr, ptr, ptr } }
%"Start<!ts.string>" = type { ptr, ptr, { ptr, ptr, ptr } }
%"Start<si32>" = type { ptr, i32, { ptr, ptr, ptr } }

$"Start<!ts.class<Box, !ts.class_storage<Box>>>..instanceOf" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>..new" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>.constructor" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>.run" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>.runInTry" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>.drop" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>.get" = comdat any

$"spawn<!ts.class<Box, !ts.class_storage<Box>>>" = comdat any

$"Start<!ts.string>..instanceOf" = comdat any

$"Start<!ts.string>..new" = comdat any

$"Start<!ts.string>.constructor" = comdat any

$"Start<!ts.string>.run" = comdat any

$"Start<!ts.string>.runInTry" = comdat any

$"Start<!ts.string>.drop" = comdat any

$"Start<!ts.string>.get" = comdat any

$"spawn<!ts.string>" = comdat any

$"Start<si32>..instanceOf" = comdat any

$"Start<si32>..new" = comdat any

$"Start<si32>.constructor" = comdat any

$"Start<si32>.run" = comdat any

$"Start<si32>.runInTry" = comdat any

$"Start<si32>.drop" = comdat any

$"Start<si32>.get" = comdat any

$"spawn<si32>" = comdat any

$Box..size = comdat any

$Keeper..size = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>..rtti" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>..size" = comdat any

$"Start<!ts.class<Box, !ts.class_storage<Box>>>..vtbl" = comdat any

$"Start<!ts.string>..rtti" = comdat any

$"Start<!ts.string>..size" = comdat any

$"Start<!ts.string>..vtbl" = comdat any

$"Start<si32>..rtti" = comdat any

$"Start<si32>..size" = comdat any

$"Start<si32>..vtbl" = comdat any

@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_1609559789082440018 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBox\00", align 8
@s_10708658430971434591 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFKeeper\00", align 8
@s_11820158475841152797 = internal constant [54 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFStart<!ts.class<Box, !ts.class_storage<Box>>>\00", align 8
@s_6295632040551916886 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFStart<!ts.string>\00", align 8
@s_3012435020349344922 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFStart<si32>\00", align 8
@m_12143668946091774359 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInvalid array length\00", align 8
@f_11817493874169425794 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00tslang_issues_31_33.ts\00", align 8
@m_2964340536813307715 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFOut of memory\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@s_9690130352454037036 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnot reached\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_16602643279055812603 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa41\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@Box..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1609559789082440018, i64 8)
@Box..size = global i64 16, comdat
@Box..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Box..instanceOf, ptr @Box..new, ptr @Box..rtti, ptr @Box..size }
@Keeper..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_10708658430971434591, i64 8)
@Keeper..size = global i64 32, comdat
@Keeper..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Keeper..instanceOf, ptr @Keeper..new, ptr @Keeper..rtti, ptr @Keeper..size }
@"Start<!ts.class<Box, !ts.class_storage<Box>>>..rtti" = linkonce_odr global ptr getelementptr inbounds nuw (i8, ptr @s_11820158475841152797, i64 8), comdat
@"Start<!ts.class<Box, !ts.class_storage<Box>>>..size" = global i64 40, comdat
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPc = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@"Start<!ts.class<Box, !ts.class_storage<Box>>>..vtbl" = linkonce_odr global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..instanceOf", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..new", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.run", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.runInTry", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.drop", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.get", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..rtti", ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..size" }, comdat
@"Start<!ts.string>..rtti" = linkonce_odr global ptr getelementptr inbounds nuw (i8, ptr @s_6295632040551916886, i64 8), comdat
@"Start<!ts.string>..size" = global i64 40, comdat
@"Start<!ts.string>..vtbl" = linkonce_odr global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @"Start<!ts.string>..instanceOf", ptr @"Start<!ts.string>..new", ptr @"Start<!ts.string>.run", ptr @"Start<!ts.string>.runInTry", ptr @"Start<!ts.string>.drop", ptr @"Start<!ts.string>.get", ptr @"Start<!ts.string>..rtti", ptr @"Start<!ts.string>..size" }, comdat
@"Start<si32>..rtti" = linkonce_odr global ptr getelementptr inbounds nuw (i8, ptr @s_3012435020349344922, i64 8), comdat
@"Start<si32>..size" = global i64 40, comdat
@"Start<si32>..vtbl" = linkonce_odr global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @"Start<si32>..instanceOf", ptr @"Start<si32>..new", ptr @"Start<si32>.run", ptr @"Start<si32>.runInTry", ptr @"Start<si32>.drop", ptr @"Start<si32>.get", ptr @"Start<si32>..rtti", ptr @"Start<si32>..size" }, comdat
@a_0_5448230598113101985 = internal constant [0 x ptr] undef
@td_2135040168780653229_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_2135040168780653229, ptr @tsretcb_2135040168780653229, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_6051753898719660491_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_6051753898719660491, ptr @tsretcb_6051753898719660491, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@a_0_14601883450416671780 = internal constant [0 x ptr] undef
@td_276211409953694424_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_276211409953694424, ptr @tsretcb_276211409953694424, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_12216996078527908883_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_12216996078527908883, ptr @tsretcb_12216996078527908883, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_7267384957113499010_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_7267384957113499010, ptr @tsretcb_7267384957113499010, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_15336564547564024679_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_15336564547564024679, ptr @tsretcb_15336564547564024679, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

define internal void @tsretcb_15336564547564024679(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_15336564547564024679(ptr %0) {
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

define internal void @tsretcb_7267384957113499010(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_7267384957113499010(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr, i32 }, ptr %2, i32 0, i32 0
  call void @tsrel_17239760223775721853(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @__cxa_end_catch()

declare void @puts(ptr)

define internal void @__tslang_retain_block_slot(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
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

define internal void @tsretcb_12216996078527908883(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_12216996078527908883(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %15

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %15

6:                                                ; preds = %4
  %7 = getelementptr { ptr, ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_12614831220088641735(ptr %7)
  %8 = getelementptr { ptr, ptr }, ptr %2, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = icmp ne ptr %9, null
  br i1 %10, label %11, label %14

11:                                               ; preds = %6
  %12 = call i1 @__tslang_dec_ref(ptr %9)
  br i1 %12, label %13, label %14

13:                                               ; preds = %11
  call void @__tslang_free_block(ptr %9)
  br label %14

14:                                               ; preds = %13, %11, %6
  call void @__tslang_free_block(ptr %2)
  br label %15

15:                                               ; preds = %14, %4, %1
  ret void
}

define internal void @tsretcb_276211409953694424(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_276211409953694424(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_12614831220088641735(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsretcb_6051753898719660491(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrelcb_6051753898719660491(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_12614831220088641735(ptr %7)
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

define internal void @tsretcb_2135040168780653229(ptr %0) {
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

define internal void @tsrel_17239760223775721853(ptr %0) {
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

define internal void @tsrel_12614831220088641735(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  call void @tsrel_17239760223775721853(ptr %2)
  call void @__tslang_free_block(ptr %2)
  br label %7

7:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrelcb_2135040168780653229(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_12614831220088641735(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare noalias ptr @GC_realloc(ptr, i64)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Box..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Box..rtti, align 8
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
define ptr @Box..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Box, ptr %3, i32 0, i32 0
  store ptr @Box..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Box.constructor(ptr %0, i32 %1) #2 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store i32 %1, ptr %3, align 4
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %3, align 4
  store i32 %7, ptr %6, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Keeper..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Keeper..rtti, align 8
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
define ptr @Keeper..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Keeper, ptr %3, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Keeper.constructor(ptr %0, { ptr, ptr, ptr } %1) #2 {
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store { ptr, ptr, ptr } %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Keeper, ptr %5, i32 0, i32 1
  %7 = load { ptr, ptr, ptr }, ptr %3, align 8
  store { ptr, ptr, ptr } %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @churn() #2 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 8)
  %4 = getelementptr i8, ptr %3, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %4, ptr @a_0_5448230598113101985, i64 0, i1 false)
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
  %12 = icmp slt i32 %11, 16
  br i1 %12, label %13, label %65

13:                                               ; preds = %10
  %14 = call ptr @GC_malloc(i64 24)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr %Box, ptr %15, i32 0, i32 0
  store ptr @Box..vtbl, ptr %16, align 8
  call void @Box.constructor(ptr %15, i32 -1)
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
  br i1 %40, label %41, label %78

41:                                               ; preds = %34
  %42 = icmp ule i64 %39, 2305843009213693951
  br i1 %42, label %43, label %80

43:                                               ; preds = %41
  %44 = mul i64 8, %39
  %45 = add i64 %44, 8
  %46 = icmp eq ptr %31, null
  %47 = getelementptr i8, ptr %31, i64 -8
  %48 = select i1 %46, ptr null, ptr %47
  %49 = call ptr @GC_realloc(ptr %48, i64 %45)
  %50 = getelementptr i8, ptr %49, i64 8
  %51 = icmp ne ptr %50, null
  %52 = icmp eq i64 %44, 0
  %53 = or i1 %51, %52
  br i1 %53, label %54, label %82

54:                                               ; preds = %43
  %55 = mul i64 8, %32
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
  %62 = getelementptr ptr, ptr %60, i64 %27
  store ptr %15, ptr %62, align 8
  store i64 %28, ptr %26, align 8
  %63 = load i32, ptr %1, align 4
  %64 = add i32 %63, 1
  store i32 %64, ptr %1, align 4
  br label %10

65:                                               ; preds = %10
  %66 = load ptr, ptr %2, align 8
  %67 = icmp eq ptr %66, null
  %68 = select i1 %67, ptr @__ts_empty_array, ptr %66
  %69 = getelementptr { ptr, i64, i64 }, ptr %68, i32 0, i32 0
  %70 = load ptr, ptr %69, align 8
  %71 = getelementptr ptr, ptr %70, i32 15
  %72 = load ptr, ptr %71, align 8
  %73 = getelementptr %Box, ptr %72, i32 0, i32 1
  %74 = load i32, ptr %73, align 4
  %75 = icmp eq i32 %74, -1
  br i1 %75, label %76, label %84

76:                                               ; preds = %65
  ret void

77:                                               ; preds = %82, %80, %78, %84
  unreachable

78:                                               ; preds = %34
  %79 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 49, ptr null)
  br label %77

80:                                               ; preds = %41
  %81 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 49, ptr null)
  br label %77

82:                                               ; preds = %43
  %83 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 49, ptr null)
  br label %77

84:                                               ; preds = %65
  %85 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 52, ptr null)
  br label %77
}

; Function Attrs: mustprogress
define void @expectCount(double %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load double, ptr %4, align 8
  %6 = fcmp oeq double %5, -1.000000e+00
  br i1 %6, label %11, label %7

7:                                                ; preds = %2
  %8 = load double, ptr %4, align 8
  %9 = load double, ptr %3, align 8
  %10 = fcmp oeq double %8, %9
  br label %11

11:                                               ; preds = %7, %2
  %12 = phi i1 [ %10, %7 ], [ %6, %2 ]
  br i1 %12, label %13, label %15

13:                                               ; preds = %11
  ret void

14:                                               ; preds = %15
  unreachable

15:                                               ; preds = %11
  %16 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 57, ptr null)
  br label %14
}

; Function Attrs: mustprogress
define void @runBoth(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Keeper, ptr %5, i32 0, i32 1
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %2
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = call i32 %12(ptr %8)
  br label %17

14:                                               ; preds = %2
  %15 = extractvalue { ptr, ptr, ptr } %7, 0
  %16 = call i32 %15()
  br label %17

17:                                               ; preds = %11, %14
  %18 = phi i32 [ %16, %14 ], [ %13, %11 ]
  br label %19

19:                                               ; preds = %17
  %20 = load ptr, ptr %3, align 8
  %21 = getelementptr %Keeper, ptr %20, i32 0, i32 1
  %22 = load { ptr, ptr, ptr }, ptr %21, align 8
  %23 = extractvalue { ptr, ptr, ptr } %22, 1
  %24 = ptrtoint ptr %23 to i64
  %25 = icmp ne i64 %24, 0
  br i1 %25, label %26, label %29

26:                                               ; preds = %19
  %27 = extractvalue { ptr, ptr, ptr } %22, 0
  %28 = call i32 %27(ptr %23)
  br label %32

29:                                               ; preds = %19
  %30 = extractvalue { ptr, ptr, ptr } %22, 0
  %31 = call i32 %30()
  br label %32

32:                                               ; preds = %26, %29
  %33 = phi i32 [ %31, %29 ], [ %28, %26 ]
  br label %34

34:                                               ; preds = %32
  ret void
}

; Function Attrs: mustprogress
define i32 @.f_twoClosures..afL67C26FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = add i32 %9, 1
  store i32 %10, ptr %6, align 4
  store i32 0, ptr %2, align 4
  %11 = load i32, ptr %2, align 4
  ret i32 %11
}

; Function Attrs: mustprogress
define i32 @.f_twoClosures..afL68C26FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = add i32 %9, 1
  store i32 %10, ptr %6, align 4
  store i32 0, ptr %2, align 4
  %11 = load i32, ptr %2, align 4
  ret i32 %11
}

; Function Attrs: mustprogress
define void @twoClosures(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca <{ i64, %Keeper }>, align 16
  %5 = alloca <{ i64, %Keeper }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Keeper zeroinitializer, ptr %6, align 8
  store i64 -1, ptr %4, align 8
  %7 = getelementptr i8, ptr %4, i64 8
  store %Keeper zeroinitializer, ptr %7, align 8
  store ptr %0, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  store ptr %8, ptr %2, align 8
  store %Keeper zeroinitializer, ptr %7, align 8
  %9 = getelementptr %Keeper, ptr %7, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %9, align 8
  %10 = load ptr, ptr %2, align 8
  %11 = call ptr @GC_malloc(i64 16)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr { ptr }, ptr %12, i32 0, i32 0
  %14 = load ptr, ptr %2, align 8
  store ptr %14, ptr %13, align 8
  %15 = insertvalue { ptr, ptr, ptr } { ptr @.f_twoClosures..afL67C26FH3418460370, ptr null, ptr null }, ptr %12, 1
  %16 = insertvalue { ptr, ptr, ptr } %15, ptr getelementptr inbounds nuw (i8, ptr @td_2135040168780653229_object, i64 40), 2
  call void @Keeper.constructor(ptr %7, { ptr, ptr, ptr } %16)
  store %Keeper zeroinitializer, ptr %6, align 8
  %17 = getelementptr %Keeper, ptr %6, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %17, align 8
  %18 = load ptr, ptr %2, align 8
  %19 = call ptr @GC_malloc(i64 16)
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr { ptr }, ptr %20, i32 0, i32 0
  %22 = load ptr, ptr %2, align 8
  store ptr %22, ptr %21, align 8
  %23 = insertvalue { ptr, ptr, ptr } { ptr @.f_twoClosures..afL68C26FH3418460370, ptr null, ptr null }, ptr %20, 1
  %24 = insertvalue { ptr, ptr, ptr } %23, ptr getelementptr inbounds nuw (i8, ptr @td_2135040168780653229_object, i64 40), 2
  call void @Keeper.constructor(ptr %6, { ptr, ptr, ptr } %24)
  %25 = load ptr, ptr %3, align 8
  call void @expectCount(double -1.000000e+00, double 4.000000e+00)
  call void @runBoth(ptr %7, ptr %6)
  call void @churn()
  %26 = load ptr, ptr %2, align 8
  %27 = load ptr, ptr %26, align 8
  %28 = getelementptr %Box, ptr %27, i32 0, i32 1
  %29 = load i32, ptr %28, align 4
  %30 = icmp eq i32 %29, 2
  br i1 %30, label %31, label %33

31:                                               ; preds = %1
  ret void

32:                                               ; preds = %33
  unreachable

33:                                               ; preds = %1
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 72, ptr null)
  br label %32
}

; Function Attrs: mustprogress
define i32 @.f_closuresShareOne..afL81C26FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  %8 = load i32, ptr %2, align 4
  ret i32 %8
}

; Function Attrs: mustprogress
define i32 @.f_closuresShareOne..afL82C26FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  %8 = load i32, ptr %2, align 4
  ret i32 %8
}

; Function Attrs: mustprogress
define void @closuresShareOne() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca <{ i64, %Keeper }>, align 16
  %3 = alloca <{ i64, %Keeper }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Keeper zeroinitializer, ptr %4, align 8
  store i64 -1, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 8
  store %Keeper zeroinitializer, ptr %5, align 8
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 0
  store ptr @Box..vtbl, ptr %8, align 8
  call void @Box.constructor(ptr %7, i32 0)
  %9 = call ptr @GC_malloc(i64 16)
  %10 = getelementptr i8, ptr %9, i64 8
  store ptr %7, ptr %10, align 8
  store ptr %10, ptr %1, align 8
  %11 = load ptr, ptr %1, align 8
  call void @twoClosures(ptr %11)
  %12 = load ptr, ptr %1, align 8
  call void @expectCount(double -1.000000e+00, double 1.000000e+00)
  %13 = load ptr, ptr %1, align 8
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr %Box, ptr %14, i32 0, i32 1
  %16 = load i32, ptr %15, align 4
  %17 = icmp eq i32 %16, 2
  br i1 %17, label %18, label %68

18:                                               ; preds = %0
  store %Keeper zeroinitializer, ptr %5, align 8
  %19 = getelementptr %Keeper, ptr %5, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %19, align 8
  %20 = load ptr, ptr %1, align 8
  %21 = call ptr @GC_malloc(i64 16)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr { ptr }, ptr %22, i32 0, i32 0
  %24 = load ptr, ptr %1, align 8
  store ptr %24, ptr %23, align 8
  %25 = insertvalue { ptr, ptr, ptr } { ptr @.f_closuresShareOne..afL81C26FH3418460370, ptr null, ptr null }, ptr %22, 1
  %26 = insertvalue { ptr, ptr, ptr } %25, ptr getelementptr inbounds nuw (i8, ptr @td_6051753898719660491_object, i64 40), 2
  call void @Keeper.constructor(ptr %5, { ptr, ptr, ptr } %26)
  store %Keeper zeroinitializer, ptr %4, align 8
  %27 = getelementptr %Keeper, ptr %4, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %27, align 8
  %28 = load ptr, ptr %1, align 8
  %29 = call ptr @GC_malloc(i64 16)
  %30 = getelementptr i8, ptr %29, i64 8
  %31 = getelementptr { ptr }, ptr %30, i32 0, i32 0
  %32 = load ptr, ptr %1, align 8
  store ptr %32, ptr %31, align 8
  %33 = insertvalue { ptr, ptr, ptr } { ptr @.f_closuresShareOne..afL82C26FH3418460370, ptr null, ptr null }, ptr %30, 1
  %34 = insertvalue { ptr, ptr, ptr } %33, ptr getelementptr inbounds nuw (i8, ptr @td_6051753898719660491_object, i64 40), 2
  call void @Keeper.constructor(ptr %4, { ptr, ptr, ptr } %34)
  %35 = load ptr, ptr %1, align 8
  call void @expectCount(double -1.000000e+00, double 3.000000e+00)
  %36 = getelementptr %Keeper, ptr %5, i32 0, i32 1
  %37 = load { ptr, ptr, ptr }, ptr %36, align 8
  %38 = extractvalue { ptr, ptr, ptr } %37, 1
  %39 = ptrtoint ptr %38 to i64
  %40 = icmp ne i64 %39, 0
  br i1 %40, label %41, label %44

41:                                               ; preds = %18
  %42 = extractvalue { ptr, ptr, ptr } %37, 0
  %43 = call i32 %42(ptr %38)
  br label %47

44:                                               ; preds = %18
  %45 = extractvalue { ptr, ptr, ptr } %37, 0
  %46 = call i32 %45()
  br label %47

47:                                               ; preds = %41, %44
  %48 = phi i32 [ %46, %44 ], [ %43, %41 ]
  br label %49

49:                                               ; preds = %47
  %50 = getelementptr %Keeper, ptr %4, i32 0, i32 1
  %51 = load { ptr, ptr, ptr }, ptr %50, align 8
  %52 = extractvalue { ptr, ptr, ptr } %51, 1
  %53 = ptrtoint ptr %52 to i64
  %54 = icmp ne i64 %53, 0
  br i1 %54, label %55, label %58

55:                                               ; preds = %49
  %56 = extractvalue { ptr, ptr, ptr } %51, 0
  %57 = call i32 %56(ptr %52)
  br label %61

58:                                               ; preds = %49
  %59 = extractvalue { ptr, ptr, ptr } %51, 0
  %60 = call i32 %59()
  br label %61

61:                                               ; preds = %55, %58
  %62 = phi i32 [ %60, %58 ], [ %57, %55 ]
  br label %63

63:                                               ; preds = %61
  %64 = add i32 %48, %62
  %65 = icmp eq i32 %64, 4
  br i1 %65, label %66, label %70

66:                                               ; preds = %63
  ret void

67:                                               ; preds = %70, %68
  unreachable

68:                                               ; preds = %0
  %69 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 79, ptr null)
  br label %67

70:                                               ; preds = %63
  %71 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 84, ptr null)
  br label %67
}

; Function Attrs: mustprogress
define i32 @.f_closuresInLoop..afL91C33FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  %8 = load i32, ptr %2, align 4
  ret i32 %8
}

; Function Attrs: mustprogress
define void @closuresInLoop() #2 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 0
  store ptr @Box..vtbl, ptr %8, align 8
  call void @Box.constructor(ptr %7, i32 5)
  %9 = call ptr @GC_malloc(i64 16)
  %10 = getelementptr i8, ptr %9, i64 8
  store ptr %7, ptr %10, align 8
  store ptr %10, ptr %5, align 8
  %11 = call ptr @GC_malloc(i64 8)
  %12 = getelementptr i8, ptr %11, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %12, ptr @a_0_14601883450416671780, i64 0, i1 false)
  %13 = call ptr @GC_malloc(i64 32)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr { ptr, i64, i64 }, ptr %14, i32 0, i32 0
  store ptr %12, ptr %15, align 8
  %16 = getelementptr { ptr, i64, i64 }, ptr %14, i32 0, i32 1
  store i64 0, ptr %16, align 8
  %17 = getelementptr { ptr, i64, i64 }, ptr %14, i32 0, i32 2
  store i64 0, ptr %17, align 8
  store ptr %14, ptr %4, align 8
  store i32 0, ptr %3, align 4
  br label %18

18:                                               ; preds = %76, %0
  %19 = load i32, ptr %3, align 4
  %20 = icmp slt i32 %19, 4
  br i1 %20, label %21, label %80

21:                                               ; preds = %18
  %22 = call ptr @GC_malloc(i64 40)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr %Keeper, ptr %23, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %24, align 8
  %25 = load ptr, ptr %5, align 8
  %26 = call ptr @GC_malloc(i64 16)
  %27 = getelementptr i8, ptr %26, i64 8
  %28 = getelementptr { ptr }, ptr %27, i32 0, i32 0
  %29 = load ptr, ptr %5, align 8
  store ptr %29, ptr %28, align 8
  %30 = insertvalue { ptr, ptr, ptr } { ptr @.f_closuresInLoop..afL91C33FH3418460370, ptr null, ptr null }, ptr %27, 1
  %31 = insertvalue { ptr, ptr, ptr } %30, ptr getelementptr inbounds nuw (i8, ptr @td_6051753898719660491_object, i64 40), 2
  call void @Keeper.constructor(ptr %23, { ptr, ptr, ptr } %31)
  %32 = load ptr, ptr %4, align 8
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %37

34:                                               ; preds = %21
  %35 = call ptr @GC_malloc(i64 32)
  %36 = getelementptr i8, ptr %35, i64 8
  store ptr %36, ptr %4, align 8
  br label %38

37:                                               ; preds = %21
  br label %38

38:                                               ; preds = %34, %37
  %39 = phi ptr [ %32, %37 ], [ %36, %34 ]
  br label %40

40:                                               ; preds = %38
  %41 = getelementptr { ptr, i64, i64 }, ptr %39, i32 0, i32 1
  %42 = load i64, ptr %41, align 8
  %43 = add i64 %42, 1
  %44 = getelementptr { ptr, i64, i64 }, ptr %39, i32 0, i32 0
  %45 = getelementptr { ptr, i64, i64 }, ptr %39, i32 0, i32 2
  %46 = load ptr, ptr %44, align 8
  %47 = load i64, ptr %45, align 8
  %48 = icmp ugt i64 %43, %47
  br i1 %48, label %49, label %73

49:                                               ; preds = %40
  %50 = mul i64 %47, 2
  %51 = icmp uge i64 %50, %43
  %52 = select i1 %51, i64 %50, i64 %43
  %53 = icmp ult i64 %52, 4
  %54 = select i1 %53, i64 4, i64 %52
  %55 = icmp ule i64 %43, 4294967295
  br i1 %55, label %56, label %123

56:                                               ; preds = %49
  %57 = icmp ule i64 %54, 2305843009213693951
  br i1 %57, label %58, label %125

58:                                               ; preds = %56
  %59 = mul i64 8, %54
  %60 = add i64 %59, 8
  %61 = icmp eq ptr %46, null
  %62 = getelementptr i8, ptr %46, i64 -8
  %63 = select i1 %61, ptr null, ptr %62
  %64 = call ptr @GC_realloc(ptr %63, i64 %60)
  %65 = getelementptr i8, ptr %64, i64 8
  %66 = icmp ne ptr %65, null
  %67 = icmp eq i64 %59, 0
  %68 = or i1 %66, %67
  br i1 %68, label %69, label %127

69:                                               ; preds = %58
  %70 = mul i64 8, %47
  %71 = getelementptr i8, ptr %65, i64 %70
  %72 = sub i64 %59, %70
  call void @llvm.memset.p0.i64(ptr %71, i8 0, i64 %72, i1 false)
  store ptr %65, ptr %44, align 8
  store i64 %54, ptr %45, align 8
  br label %74

73:                                               ; preds = %40
  br label %74

74:                                               ; preds = %69, %73
  %75 = phi ptr [ %46, %73 ], [ %65, %69 ]
  br label %76

76:                                               ; preds = %74
  %77 = getelementptr ptr, ptr %75, i64 %42
  store ptr %23, ptr %77, align 8
  store i64 %43, ptr %41, align 8
  %78 = load i32, ptr %3, align 4
  %79 = add i32 %78, 1
  store i32 %79, ptr %3, align 4
  br label %18

80:                                               ; preds = %18
  %81 = load ptr, ptr %5, align 8
  call void @expectCount(double -1.000000e+00, double 5.000000e+00)
  call void @churn()
  store i32 0, ptr %2, align 4
  store i32 0, ptr %1, align 4
  br label %82

82:                                               ; preds = %114, %80
  %83 = load i32, ptr %1, align 4
  %84 = load ptr, ptr %4, align 8
  %85 = icmp eq ptr %84, null
  %86 = select i1 %85, ptr @__ts_empty_array, ptr %84
  %87 = getelementptr { ptr, i64, i64 }, ptr %86, i32 0, i32 1
  %88 = load i64, ptr %87, align 8
  %89 = sext i32 %83 to i64
  %90 = icmp slt i64 %89, %88
  br i1 %90, label %91, label %118

91:                                               ; preds = %82
  %92 = load i32, ptr %1, align 4
  %93 = load ptr, ptr %4, align 8
  %94 = icmp eq ptr %93, null
  %95 = select i1 %94, ptr @__ts_empty_array, ptr %93
  %96 = getelementptr { ptr, i64, i64 }, ptr %95, i32 0, i32 0
  %97 = load ptr, ptr %96, align 8
  %98 = getelementptr ptr, ptr %97, i32 %92
  %99 = load ptr, ptr %98, align 8
  %100 = load i32, ptr %2, align 4
  %101 = getelementptr %Keeper, ptr %99, i32 0, i32 1
  %102 = load { ptr, ptr, ptr }, ptr %101, align 8
  %103 = extractvalue { ptr, ptr, ptr } %102, 1
  %104 = ptrtoint ptr %103 to i64
  %105 = icmp ne i64 %104, 0
  br i1 %105, label %106, label %109

106:                                              ; preds = %91
  %107 = extractvalue { ptr, ptr, ptr } %102, 0
  %108 = call i32 %107(ptr %103)
  br label %112

109:                                              ; preds = %91
  %110 = extractvalue { ptr, ptr, ptr } %102, 0
  %111 = call i32 %110()
  br label %112

112:                                              ; preds = %106, %109
  %113 = phi i32 [ %111, %109 ], [ %108, %106 ]
  br label %114

114:                                              ; preds = %112
  %115 = add i32 %100, %113
  store i32 %115, ptr %2, align 4
  %116 = load i32, ptr %1, align 4
  %117 = add i32 %116, 1
  store i32 %117, ptr %1, align 4
  br label %82

118:                                              ; preds = %82
  %119 = load i32, ptr %2, align 4
  %120 = icmp eq i32 %119, 20
  br i1 %120, label %121, label %129

121:                                              ; preds = %118
  ret void

122:                                              ; preds = %127, %125, %123, %129
  unreachable

123:                                              ; preds = %49
  %124 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 91, ptr null)
  br label %122

125:                                              ; preds = %56
  %126 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12143668946091774359, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 91, ptr null)
  br label %122

127:                                              ; preds = %58
  %128 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2964340536813307715, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 91, ptr null)
  br label %122

129:                                              ; preds = %118
  %130 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 101, ptr null)
  br label %122
}

; Function Attrs: mustprogress
define i32 @.f_closureOutlivesTheLocal..afL106C23FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  %8 = load i32, ptr %2, align 4
  ret i32 %8
}

; Function Attrs: mustprogress
define ptr @closureOutlivesTheLocal() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = call ptr @GC_malloc(i64 24)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %Box, ptr %4, i32 0, i32 0
  store ptr @Box..vtbl, ptr %5, align 8
  call void @Box.constructor(ptr %4, i32 7)
  %6 = call ptr @GC_malloc(i64 16)
  %7 = getelementptr i8, ptr %6, i64 8
  store ptr %4, ptr %7, align 8
  store ptr %7, ptr %1, align 8
  %8 = call ptr @GC_malloc(i64 40)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Keeper, ptr %9, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %10, align 8
  %11 = load ptr, ptr %1, align 8
  %12 = call ptr @GC_malloc(i64 16)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr { ptr }, ptr %13, i32 0, i32 0
  %15 = load ptr, ptr %1, align 8
  store ptr %15, ptr %14, align 8
  %16 = insertvalue { ptr, ptr, ptr } { ptr @.f_closureOutlivesTheLocal..afL106C23FH3418460370, ptr null, ptr null }, ptr %13, 1
  %17 = insertvalue { ptr, ptr, ptr } %16, ptr getelementptr inbounds nuw (i8, ptr @td_6051753898719660491_object, i64 40), 2
  call void @Keeper.constructor(ptr %9, { ptr, ptr, ptr } %17)
  store ptr %9, ptr %2, align 8
  %18 = load ptr, ptr %2, align 8
  ret ptr %18
}

; Function Attrs: mustprogress
define i32 @callClosure({ ptr, ptr, ptr } %0) #2 {
  %2 = alloca { ptr, ptr, ptr }, align 8
  %3 = alloca i32, align 4
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  %4 = load { ptr, ptr, ptr }, ptr %2, align 8
  %5 = extractvalue { ptr, ptr, ptr } %4, 1
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %1
  %9 = extractvalue { ptr, ptr, ptr } %4, 0
  %10 = call i32 %9(ptr %5)
  br label %14

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %4, 0
  %13 = call i32 %12()
  br label %14

14:                                               ; preds = %8, %11
  %15 = phi i32 [ %13, %11 ], [ %10, %8 ]
  br label %16

16:                                               ; preds = %14
  store i32 %15, ptr %3, align 4
  %17 = load i32, ptr %3, align 4
  ret i32 %17
}

; Function Attrs: mustprogress
define i32 @.f_borrowingBoxes..afL115C24FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  %8 = load i32, ptr %2, align 4
  ret i32 %8
}

; Function Attrs: mustprogress
define i32 @.f_borrowingBoxes..afL118C24FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr %Box, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = load i32, ptr %6, align 4
  %11 = add i32 %9, %10
  store i32 %11, ptr %2, align 4
  %12 = load i32, ptr %2, align 4
  ret i32 %12
}

; Function Attrs: mustprogress
define i32 @.f_borrowingBoxes..f_.afL119C31FH3418460370..afL120C23FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Box, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  %8 = mul i32 %7, 2
  store i32 %8, ptr %2, align 4
  %9 = load i32, ptr %2, align 4
  ret i32 %9
}

; Function Attrs: mustprogress
define i32 @.f_borrowingBoxes..afL119C31FH3418460370(ptr %0) #2 {
  %2 = alloca <{ i64, { ptr } }>, align 16
  %3 = alloca i32, align 4
  %4 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  store i64 -1, ptr %2, align 8
  %6 = getelementptr i8, ptr %2, i64 8
  store { ptr } zeroinitializer, ptr %6, align 8
  %7 = getelementptr { ptr }, ptr %6, i32 0, i32 0
  store ptr %5, ptr %7, align 8
  %8 = insertvalue { ptr, ptr, ptr } { ptr @.f_borrowingBoxes..f_.afL119C31FH3418460370..afL120C23FH3418460370, ptr null, ptr null }, ptr %6, 1
  %9 = insertvalue { ptr, ptr, ptr } %8, ptr getelementptr inbounds nuw (i8, ptr @td_276211409953694424_object, i64 40), 2
  %10 = call i32 @.f_borrowingBoxes..f_.afL119C31FH3418460370..afL120C23FH3418460370(ptr %6)
  store i32 %10, ptr %3, align 4
  %11 = load i32, ptr %3, align 4
  ret i32 %11
}

; Function Attrs: mustprogress
define void @borrowingBoxes(ptr %0) #2 {
  %2 = alloca <{ i64, { ptr, ptr } }>, align 16
  %3 = alloca <{ i64, { ptr } }>, align 16
  %4 = alloca ptr, align 8
  %5 = alloca <{ i64, %Keeper }>, align 16
  store i64 -1, ptr %5, align 8
  %6 = getelementptr i8, ptr %5, i64 8
  store %Keeper zeroinitializer, ptr %6, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  store i64 -1, ptr %3, align 8
  %8 = getelementptr i8, ptr %3, i64 8
  store { ptr } zeroinitializer, ptr %8, align 8
  %9 = getelementptr { ptr }, ptr %8, i32 0, i32 0
  %10 = load ptr, ptr %4, align 8
  store ptr %10, ptr %9, align 8
  %11 = insertvalue { ptr, ptr, ptr } { ptr @.f_borrowingBoxes..afL115C24FH3418460370, ptr null, ptr null }, ptr %8, 1
  %12 = insertvalue { ptr, ptr, ptr } %11, ptr getelementptr inbounds nuw (i8, ptr @td_276211409953694424_object, i64 40), 2
  %13 = call i32 @callClosure({ ptr, ptr, ptr } %12)
  %14 = icmp eq i32 %13, 3
  br i1 %14, label %15, label %53

15:                                               ; preds = %1
  %16 = call ptr @GC_malloc(i64 12)
  %17 = getelementptr i8, ptr %16, i64 8
  store i32 1, ptr %17, align 4
  store i32 2, ptr %17, align 4
  %18 = load ptr, ptr %4, align 8
  store i64 -1, ptr %2, align 8
  %19 = getelementptr i8, ptr %2, i64 8
  store { ptr, ptr } zeroinitializer, ptr %19, align 8
  %20 = getelementptr { ptr, ptr }, ptr %19, i32 0, i32 0
  %21 = load ptr, ptr %4, align 8
  store ptr %21, ptr %20, align 8
  %22 = getelementptr { ptr, ptr }, ptr %19, i32 0, i32 1
  store ptr %17, ptr %22, align 8
  %23 = insertvalue { ptr, ptr, ptr } { ptr @.f_borrowingBoxes..afL118C24FH3418460370, ptr null, ptr null }, ptr %19, 1
  %24 = insertvalue { ptr, ptr, ptr } %23, ptr getelementptr inbounds nuw (i8, ptr @td_12216996078527908883_object, i64 40), 2
  %25 = call i32 @callClosure({ ptr, ptr, ptr } %24)
  %26 = icmp eq i32 %25, 5
  br i1 %26, label %27, label %55

27:                                               ; preds = %15
  store %Keeper zeroinitializer, ptr %6, align 8
  %28 = getelementptr %Keeper, ptr %6, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %28, align 8
  %29 = load ptr, ptr %4, align 8
  %30 = call ptr @GC_malloc(i64 16)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr { ptr }, ptr %31, i32 0, i32 0
  %33 = load ptr, ptr %4, align 8
  store ptr %33, ptr %32, align 8
  %34 = insertvalue { ptr, ptr, ptr } { ptr @.f_borrowingBoxes..afL119C31FH3418460370, ptr null, ptr null }, ptr %31, 1
  %35 = insertvalue { ptr, ptr, ptr } %34, ptr getelementptr inbounds nuw (i8, ptr @td_276211409953694424_object, i64 40), 2
  call void @Keeper.constructor(ptr %6, { ptr, ptr, ptr } %35)
  %36 = getelementptr %Keeper, ptr %6, i32 0, i32 1
  %37 = load { ptr, ptr, ptr }, ptr %36, align 8
  %38 = extractvalue { ptr, ptr, ptr } %37, 1
  %39 = ptrtoint ptr %38 to i64
  %40 = icmp ne i64 %39, 0
  br i1 %40, label %41, label %44

41:                                               ; preds = %27
  %42 = extractvalue { ptr, ptr, ptr } %37, 0
  %43 = call i32 %42(ptr %38)
  br label %47

44:                                               ; preds = %27
  %45 = extractvalue { ptr, ptr, ptr } %37, 0
  %46 = call i32 %45()
  br label %47

47:                                               ; preds = %41, %44
  %48 = phi i32 [ %46, %44 ], [ %43, %41 ]
  br label %49

49:                                               ; preds = %47
  %50 = icmp eq i32 %48, 6
  br i1 %50, label %51, label %57

51:                                               ; preds = %49
  ret void

52:                                               ; preds = %57, %55, %53
  unreachable

53:                                               ; preds = %1
  %54 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 115, ptr null)
  br label %52

55:                                               ; preds = %15
  %56 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 118, ptr null)
  br label %52

57:                                               ; preds = %49
  %58 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 123, ptr null)
  br label %52
}

; Function Attrs: mustprogress
define ptr @makeBox() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Box, ptr %3, i32 0, i32 0
  store ptr @Box..vtbl, ptr %4, align 8
  call void @Box.constructor(ptr %3, i32 1)
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"Start<!ts.class<Box, !ts.class_storage<Box>>>..instanceOf"(ptr %0, ptr %1) #2 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..rtti", align 8
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
define linkonce_odr ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..new"() #2 comdat {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 48)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 0
  store ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..vtbl", ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.constructor"(ptr %0, { ptr, ptr, ptr } %1) #2 comdat {
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store { ptr, ptr, ptr } %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %3, align 8
  store { ptr, ptr, ptr } %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.run"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = call ptr %12(ptr %8)
  br label %17

14:                                               ; preds = %1
  %15 = extractvalue { ptr, ptr, ptr } %7, 0
  %16 = call ptr %15()
  br label %17

17:                                               ; preds = %11, %14
  %18 = phi ptr [ %16, %14 ], [ %13, %11 ]
  br label %19

19:                                               ; preds = %17
  store ptr %18, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.runInTry"(ptr %0) #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %15

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = invoke ptr %12(ptr %8)
          to label %14 unwind label %25

14:                                               ; preds = %11
  br label %19

15:                                               ; preds = %1
  %16 = extractvalue { ptr, ptr, ptr } %7, 0
  %17 = invoke ptr %16()
          to label %18 unwind label %25

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %14, %18
  %20 = phi ptr [ %17, %18 ], [ %13, %14 ]
  br label %21

21:                                               ; preds = %19
  br label %22

22:                                               ; preds = %21
  %23 = load ptr, ptr %4, align 8
  %24 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %23, i32 0, i32 1
  store ptr %20, ptr %24, align 8
  br label %164

25:                                               ; preds = %11, %15
  %26 = landingpad { ptr, i32 }
          catch ptr null
  %27 = extractvalue { ptr, i32 } %26, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %25
  %30 = icmp eq ptr %29, @_ZTIi
  br i1 %30, label %31, label %38

31:                                               ; preds = %.noexc3
  %32 = load i32, ptr %28, align 4
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 0
  store i64 4, ptr %35, align 8
  %36 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 2
  store i32 %32, ptr %37, align 4
  br label %161

38:                                               ; preds = %.noexc3
  %39 = icmp eq ptr %29, @_ZTIj
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load i32, ptr %28, align 4
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 0
  store i64 4, ptr %44, align 8
  %45 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 2
  store i32 %41, ptr %46, align 4
  br label %158

47:                                               ; preds = %38
  %48 = icmp eq ptr %29, @_ZTId
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load double, ptr %28, align 8
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 0
  store i64 8, ptr %53, align 8
  %54 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 2
  store double %50, ptr %55, align 8
  br label %155

56:                                               ; preds = %47
  %57 = icmp eq ptr %29, @_ZTIf
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load float, ptr %28, align 4
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 0
  store i64 4, ptr %62, align 8
  %63 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 2
  store float %59, ptr %64, align 4
  br label %152

65:                                               ; preds = %56
  %66 = icmp eq ptr %29, @_ZTIb
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load i1, ptr %28, align 1
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 0
  store i64 1, ptr %71, align 8
  %72 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 2
  store i1 %68, ptr %73, align 1
  br label %149

74:                                               ; preds = %65
  %75 = icmp eq ptr %29, @_ZTIx
  br i1 %75, label %76, label %83

76:                                               ; preds = %74
  %77 = load i64, ptr %28, align 8
  %78 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %76
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 0
  store i64 8, ptr %80, align 8
  %81 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %81, align 8
  %82 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 2
  store i64 %77, ptr %82, align 8
  br label %146

83:                                               ; preds = %74
  %84 = icmp eq ptr %29, @_ZTIPc
  br i1 %84, label %85, label %92

85:                                               ; preds = %83
  %86 = load ptr, ptr %28, align 8
  %87 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %85
  %88 = getelementptr i8, ptr %87, i64 8
  %89 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 0
  store i64 8, ptr %89, align 8
  %90 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %90, align 8
  %91 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 2
  store ptr %86, ptr %91, align 8
  br label %143

92:                                               ; preds = %83
  %93 = icmp eq ptr %29, @_ZTIPv
  br i1 %93, label %94, label %96

94:                                               ; preds = %92
  %95 = load ptr, ptr %28, align 8
  br label %140

96:                                               ; preds = %92
  %97 = load ptr, ptr %29, align 8
  %98 = icmp eq ptr %97, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %98, label %99, label %130

99:                                               ; preds = %96
  %100 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %29, i32 0, i32 2
  %101 = load i32, ptr %100, align 4
  %102 = and i32 %101, 1073741824
  %103 = icmp ne i32 %102, 0
  br i1 %103, label %104, label %120

104:                                              ; preds = %99
  %105 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %29, i32 0, i32 4
  %106 = load ptr, ptr %105, align 8
  %107 = icmp ne ptr %106, null
  br i1 %107, label %108, label %110

108:                                              ; preds = %104
  invoke void %106(ptr %2, ptr %28)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %108
  %109 = load ptr, ptr %2, align 8
  br label %117

110:                                              ; preds = %104
  %111 = load ptr, ptr %28, align 8
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 0
  store i64 8, ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 2
  store ptr %111, ptr %116, align 8
  br label %117

117:                                              ; preds = %.noexc9, %.noexc10
  %118 = phi ptr [ %113, %.noexc10 ], [ %109, %.noexc9 ]
  br label %119

119:                                              ; preds = %117
  br label %127

120:                                              ; preds = %99
  %121 = load ptr, ptr %28, align 8
  %122 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %120
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 0
  store i64 8, ptr %124, align 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %125, align 8
  %126 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 2
  store ptr %121, ptr %126, align 8
  br label %127

127:                                              ; preds = %119, %.noexc4
  %128 = phi ptr [ %123, %.noexc4 ], [ %118, %119 ]
  br label %129

129:                                              ; preds = %127
  br label %137

130:                                              ; preds = %96
  %131 = load ptr, ptr %28, align 8
  %132 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %130
  %133 = getelementptr i8, ptr %132, i64 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 0
  store i64 8, ptr %134, align 8
  %135 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %135, align 8
  %136 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 2
  store ptr %131, ptr %136, align 8
  br label %137

137:                                              ; preds = %129, %.noexc5
  %138 = phi ptr [ %133, %.noexc5 ], [ %128, %129 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %94, %139
  %141 = phi ptr [ %138, %139 ], [ %95, %94 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc1, %142
  %144 = phi ptr [ %141, %142 ], [ %88, %.noexc1 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc12, %145
  %147 = phi ptr [ %144, %145 ], [ %79, %.noexc12 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc2, %148
  %150 = phi ptr [ %147, %148 ], [ %70, %.noexc2 ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc8, %151
  %153 = phi ptr [ %150, %151 ], [ %61, %.noexc8 ]
  br label %154

154:                                              ; preds = %152
  br label %155

155:                                              ; preds = %.noexc11, %154
  %156 = phi ptr [ %153, %154 ], [ %52, %.noexc11 ]
  br label %157

157:                                              ; preds = %155
  br label %158

158:                                              ; preds = %.noexc7, %157
  %159 = phi ptr [ %156, %157 ], [ %43, %.noexc7 ]
  br label %160

160:                                              ; preds = %158
  br label %161

161:                                              ; preds = %.noexc6, %160
  %162 = phi ptr [ %159, %160 ], [ %34, %.noexc6 ]
  br label %163

163:                                              ; preds = %161
  store ptr %162, ptr %3, align 8
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8))
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %163
  call void @__cxa_end_catch()
  br label %164

164:                                              ; preds = %22, %.noexc
  ret void

endcatch.resume:                                  ; preds = %76, %49, %110, %108, %58, %40, %31, %130, %120, %25, %67, %85, %163
  %165 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %165
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.drop"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 2
  %5 = load { ptr, ptr, ptr }, ptr %4, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  %11 = call ptr %10(ptr %6)
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  %14 = call ptr %13()
  br label %15

15:                                               ; preds = %9, %12
  %16 = phi ptr [ %14, %12 ], [ %11, %9 ]
  br label %17

17:                                               ; preds = %15
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.get"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  ret ptr %7
}

; Function Attrs: mustprogress
define ptr @.f_resultsKept..afL138C25FH3418460370(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = getelementptr { ptr, i32 }, ptr %0, i32 0, i32 0
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr { ptr, i32 }, ptr %0, i32 0, i32 1
  %6 = load i32, ptr %5, align 4
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Box, ptr %8, i32 0, i32 0
  store ptr @Box..vtbl, ptr %9, align 8
  %10 = getelementptr %Box, ptr %4, i32 0, i32 1
  %11 = load i32, ptr %10, align 4
  %12 = add i32 %11, %6
  call void @Box.constructor(ptr %8, i32 %12)
  store ptr %8, ptr %2, align 8
  %13 = load ptr, ptr %2, align 8
  ret ptr %13
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"spawn<!ts.class<Box, !ts.class_storage<Box>>>"({ ptr, ptr, ptr } %0) #2 comdat {
  %2 = alloca { ptr, ptr, ptr }, align 8
  %3 = alloca ptr, align 8
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 48)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %5, i32 0, i32 0
  store ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..vtbl", ptr %6, align 8
  %7 = load { ptr, ptr, ptr }, ptr %2, align 8
  call void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.constructor"(ptr %5, { ptr, ptr, ptr } %7)
  store ptr %5, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define ptr @.f_resultsKept..afL152C24FH3418460370(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = getelementptr { i32 }, ptr %0, i32 0, i32 0
  %4 = load i32, ptr %3, align 4
  %5 = call ptr @GC_malloc_atomic(i64 58)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %6, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %4)
  %8 = icmp eq ptr %6, null
  %9 = select i1 %8, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %10 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
  %11 = add i64 1, %10
  %12 = call i64 @strlen(ptr %9)
  %13 = add i64 %11, %12
  %14 = add i64 %13, 8
  %15 = call ptr @GC_malloc(i64 %14)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr i8, ptr %16, i64 0
  %18 = sub i64 %13, 0
  %19 = call ptr @__memcpy_chk(ptr %17, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8), i64 %10, i64 %18)
  %20 = add i64 0, %10
  %21 = getelementptr i8, ptr %16, i64 %20
  %22 = sub i64 %13, %20
  %23 = call ptr @__memcpy_chk(ptr %21, ptr %9, i64 %12, i64 %22)
  %24 = add i64 %20, %12
  %25 = getelementptr i8, ptr %16, i64 %24
  store i8 0, ptr %25, align 1
  store ptr %16, ptr %2, align 8
  %26 = load ptr, ptr %2, align 8
  ret ptr %26
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"Start<!ts.string>..instanceOf"(ptr %0, ptr %1) #2 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @"Start<!ts.string>..rtti", align 8
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
define linkonce_odr ptr @"Start<!ts.string>..new"() #2 comdat {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 48)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %"Start<!ts.string>", ptr %3, i32 0, i32 0
  store ptr @"Start<!ts.string>..vtbl", ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.string>.constructor"(ptr %0, { ptr, ptr, ptr } %1) #2 comdat {
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store { ptr, ptr, ptr } %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Start<!ts.string>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %3, align 8
  store { ptr, ptr, ptr } %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.string>.run"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %"Start<!ts.string>", ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %"Start<!ts.string>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = call ptr %12(ptr %8)
  br label %17

14:                                               ; preds = %1
  %15 = extractvalue { ptr, ptr, ptr } %7, 0
  %16 = call ptr %15()
  br label %17

17:                                               ; preds = %11, %14
  %18 = phi ptr [ %16, %14 ], [ %13, %11 ]
  br label %19

19:                                               ; preds = %17
  store ptr %18, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.string>.runInTry"(ptr %0) #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Start<!ts.string>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %15

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = invoke ptr %12(ptr %8)
          to label %14 unwind label %25

14:                                               ; preds = %11
  br label %19

15:                                               ; preds = %1
  %16 = extractvalue { ptr, ptr, ptr } %7, 0
  %17 = invoke ptr %16()
          to label %18 unwind label %25

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %14, %18
  %20 = phi ptr [ %17, %18 ], [ %13, %14 ]
  br label %21

21:                                               ; preds = %19
  br label %22

22:                                               ; preds = %21
  %23 = load ptr, ptr %4, align 8
  %24 = getelementptr %"Start<!ts.string>", ptr %23, i32 0, i32 1
  store ptr %20, ptr %24, align 8
  br label %164

25:                                               ; preds = %11, %15
  %26 = landingpad { ptr, i32 }
          catch ptr null
  %27 = extractvalue { ptr, i32 } %26, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %25
  %30 = icmp eq ptr %29, @_ZTIi
  br i1 %30, label %31, label %38

31:                                               ; preds = %.noexc12
  %32 = load i32, ptr %28, align 4
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 0
  store i64 4, ptr %35, align 8
  %36 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 2
  store i32 %32, ptr %37, align 4
  br label %161

38:                                               ; preds = %.noexc12
  %39 = icmp eq ptr %29, @_ZTIj
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load i32, ptr %28, align 4
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 0
  store i64 4, ptr %44, align 8
  %45 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 2
  store i32 %41, ptr %46, align 4
  br label %158

47:                                               ; preds = %38
  %48 = icmp eq ptr %29, @_ZTId
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load double, ptr %28, align 8
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 0
  store i64 8, ptr %53, align 8
  %54 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 2
  store double %50, ptr %55, align 8
  br label %155

56:                                               ; preds = %47
  %57 = icmp eq ptr %29, @_ZTIf
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load float, ptr %28, align 4
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 0
  store i64 4, ptr %62, align 8
  %63 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 2
  store float %59, ptr %64, align 4
  br label %152

65:                                               ; preds = %56
  %66 = icmp eq ptr %29, @_ZTIb
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load i1, ptr %28, align 1
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 0
  store i64 1, ptr %71, align 8
  %72 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 2
  store i1 %68, ptr %73, align 1
  br label %149

74:                                               ; preds = %65
  %75 = icmp eq ptr %29, @_ZTIx
  br i1 %75, label %76, label %83

76:                                               ; preds = %74
  %77 = load i64, ptr %28, align 8
  %78 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %76
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 0
  store i64 8, ptr %80, align 8
  %81 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %81, align 8
  %82 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 2
  store i64 %77, ptr %82, align 8
  br label %146

83:                                               ; preds = %74
  %84 = icmp eq ptr %29, @_ZTIPc
  br i1 %84, label %85, label %92

85:                                               ; preds = %83
  %86 = load ptr, ptr %28, align 8
  %87 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %85
  %88 = getelementptr i8, ptr %87, i64 8
  %89 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 0
  store i64 8, ptr %89, align 8
  %90 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %90, align 8
  %91 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 2
  store ptr %86, ptr %91, align 8
  br label %143

92:                                               ; preds = %83
  %93 = icmp eq ptr %29, @_ZTIPv
  br i1 %93, label %94, label %96

94:                                               ; preds = %92
  %95 = load ptr, ptr %28, align 8
  br label %140

96:                                               ; preds = %92
  %97 = load ptr, ptr %29, align 8
  %98 = icmp eq ptr %97, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %98, label %99, label %130

99:                                               ; preds = %96
  %100 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %29, i32 0, i32 2
  %101 = load i32, ptr %100, align 4
  %102 = and i32 %101, 1073741824
  %103 = icmp ne i32 %102, 0
  br i1 %103, label %104, label %120

104:                                              ; preds = %99
  %105 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %29, i32 0, i32 4
  %106 = load ptr, ptr %105, align 8
  %107 = icmp ne ptr %106, null
  br i1 %107, label %108, label %110

108:                                              ; preds = %104
  invoke void %106(ptr %2, ptr %28)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %108
  %109 = load ptr, ptr %2, align 8
  br label %117

110:                                              ; preds = %104
  %111 = load ptr, ptr %28, align 8
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 0
  store i64 8, ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 2
  store ptr %111, ptr %116, align 8
  br label %117

117:                                              ; preds = %.noexc9, %.noexc4
  %118 = phi ptr [ %113, %.noexc4 ], [ %109, %.noexc9 ]
  br label %119

119:                                              ; preds = %117
  br label %127

120:                                              ; preds = %99
  %121 = load ptr, ptr %28, align 8
  %122 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %120
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 0
  store i64 8, ptr %124, align 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %125, align 8
  %126 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 2
  store ptr %121, ptr %126, align 8
  br label %127

127:                                              ; preds = %119, %.noexc2
  %128 = phi ptr [ %123, %.noexc2 ], [ %118, %119 ]
  br label %129

129:                                              ; preds = %127
  br label %137

130:                                              ; preds = %96
  %131 = load ptr, ptr %28, align 8
  %132 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %130
  %133 = getelementptr i8, ptr %132, i64 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 0
  store i64 8, ptr %134, align 8
  %135 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %135, align 8
  %136 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 2
  store ptr %131, ptr %136, align 8
  br label %137

137:                                              ; preds = %129, %.noexc8
  %138 = phi ptr [ %133, %.noexc8 ], [ %128, %129 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %94, %139
  %141 = phi ptr [ %138, %139 ], [ %95, %94 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc7, %142
  %144 = phi ptr [ %141, %142 ], [ %88, %.noexc7 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc10, %145
  %147 = phi ptr [ %144, %145 ], [ %79, %.noexc10 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc, %148
  %150 = phi ptr [ %147, %148 ], [ %70, %.noexc ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc6, %151
  %153 = phi ptr [ %150, %151 ], [ %61, %.noexc6 ]
  br label %154

154:                                              ; preds = %152
  br label %155

155:                                              ; preds = %.noexc1, %154
  %156 = phi ptr [ %153, %154 ], [ %52, %.noexc1 ]
  br label %157

157:                                              ; preds = %155
  br label %158

158:                                              ; preds = %.noexc3, %157
  %159 = phi ptr [ %156, %157 ], [ %43, %.noexc3 ]
  br label %160

160:                                              ; preds = %158
  br label %161

161:                                              ; preds = %.noexc5, %160
  %162 = phi ptr [ %159, %160 ], [ %34, %.noexc5 ]
  br label %163

163:                                              ; preds = %161
  store ptr %162, ptr %3, align 8
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8))
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %163
  call void @__cxa_end_catch()
  br label %164

164:                                              ; preds = %22, %.noexc11
  ret void

endcatch.resume:                                  ; preds = %25, %163, %76, %108, %130, %85, %58, %31, %110, %40, %120, %49, %67
  %165 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %165
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<!ts.string>.drop"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %"Start<!ts.string>", ptr %3, i32 0, i32 2
  %5 = load { ptr, ptr, ptr }, ptr %4, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  %11 = call ptr %10(ptr %6)
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  %14 = call ptr %13()
  br label %15

15:                                               ; preds = %9, %12
  %16 = phi ptr [ %14, %12 ], [ %11, %9 ]
  br label %17

17:                                               ; preds = %15
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"Start<!ts.string>.get"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %"Start<!ts.string>", ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %3, align 8
  ret ptr %7
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"spawn<!ts.string>"({ ptr, ptr, ptr } %0) #2 comdat {
  %2 = alloca { ptr, ptr, ptr }, align 8
  %3 = alloca ptr, align 8
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 48)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %"Start<!ts.string>", ptr %5, i32 0, i32 0
  store ptr @"Start<!ts.string>..vtbl", ptr %6, align 8
  %7 = load { ptr, ptr, ptr }, ptr %2, align 8
  call void @"Start<!ts.string>.constructor"(ptr %5, { ptr, ptr, ptr } %7)
  store ptr %5, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define i32 @.f_resultsKept..afL156C26FH3418460370(ptr %0) #2 {
  %2 = alloca i32, align 4
  %3 = getelementptr { i32 }, ptr %0, i32 0, i32 0
  %4 = load i32, ptr %3, align 4
  %5 = add i32 %4, 1
  store i32 %5, ptr %2, align 4
  %6 = load i32, ptr %2, align 4
  ret i32 %6
}

; Function Attrs: mustprogress
define linkonce_odr i1 @"Start<si32>..instanceOf"(ptr %0, ptr %1) #2 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @"Start<si32>..rtti", align 8
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
define linkonce_odr ptr @"Start<si32>..new"() #2 comdat {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 48)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %"Start<si32>", ptr %3, i32 0, i32 0
  store ptr @"Start<si32>..vtbl", ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<si32>.constructor"(ptr %0, { ptr, ptr, ptr } %1) #2 comdat {
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store { ptr, ptr, ptr } %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Start<si32>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %3, align 8
  store { ptr, ptr, ptr } %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<si32>.run"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %"Start<si32>", ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %"Start<si32>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = call i32 %12(ptr %8)
  br label %17

14:                                               ; preds = %1
  %15 = extractvalue { ptr, ptr, ptr } %7, 0
  %16 = call i32 %15()
  br label %17

17:                                               ; preds = %11, %14
  %18 = phi i32 [ %16, %14 ], [ %13, %11 ]
  br label %19

19:                                               ; preds = %17
  store i32 %18, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<si32>.runInTry"(ptr %0) #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %"Start<si32>", ptr %5, i32 0, i32 2
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %15

11:                                               ; preds = %1
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  %13 = invoke i32 %12(ptr %8)
          to label %14 unwind label %25

14:                                               ; preds = %11
  br label %19

15:                                               ; preds = %1
  %16 = extractvalue { ptr, ptr, ptr } %7, 0
  %17 = invoke i32 %16()
          to label %18 unwind label %25

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %14, %18
  %20 = phi i32 [ %17, %18 ], [ %13, %14 ]
  br label %21

21:                                               ; preds = %19
  br label %22

22:                                               ; preds = %21
  %23 = load ptr, ptr %4, align 8
  %24 = getelementptr %"Start<si32>", ptr %23, i32 0, i32 1
  store i32 %20, ptr %24, align 4
  br label %164

25:                                               ; preds = %11, %15
  %26 = landingpad { ptr, i32 }
          catch ptr null
  %27 = extractvalue { ptr, i32 } %26, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %25
  %30 = icmp eq ptr %29, @_ZTIi
  br i1 %30, label %31, label %38

31:                                               ; preds = %.noexc4
  %32 = load i32, ptr %28, align 4
  %33 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc12 unwind label %endcatch.resume

.noexc12:                                         ; preds = %31
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 0
  store i64 4, ptr %35, align 8
  %36 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %36, align 8
  %37 = getelementptr { i64, ptr, i32 }, ptr %34, i32 0, i32 2
  store i32 %32, ptr %37, align 4
  br label %161

38:                                               ; preds = %.noexc4
  %39 = icmp eq ptr %29, @_ZTIj
  br i1 %39, label %40, label %47

40:                                               ; preds = %38
  %41 = load i32, ptr %28, align 4
  %42 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %40
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 0
  store i64 4, ptr %44, align 8
  %45 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, i32 }, ptr %43, i32 0, i32 2
  store i32 %41, ptr %46, align 4
  br label %158

47:                                               ; preds = %38
  %48 = icmp eq ptr %29, @_ZTId
  br i1 %48, label %49, label %56

49:                                               ; preds = %47
  %50 = load double, ptr %28, align 8
  %51 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %49
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 0
  store i64 8, ptr %53, align 8
  %54 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %54, align 8
  %55 = getelementptr { i64, ptr, double }, ptr %52, i32 0, i32 2
  store double %50, ptr %55, align 8
  br label %155

56:                                               ; preds = %47
  %57 = icmp eq ptr %29, @_ZTIf
  br i1 %57, label %58, label %65

58:                                               ; preds = %56
  %59 = load float, ptr %28, align 4
  %60 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %58
  %61 = getelementptr i8, ptr %60, i64 8
  %62 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 0
  store i64 4, ptr %62, align 8
  %63 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %63, align 8
  %64 = getelementptr { i64, ptr, float }, ptr %61, i32 0, i32 2
  store float %59, ptr %64, align 4
  br label %152

65:                                               ; preds = %56
  %66 = icmp eq ptr %29, @_ZTIb
  br i1 %66, label %67, label %74

67:                                               ; preds = %65
  %68 = load i1, ptr %28, align 1
  %69 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %67
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 0
  store i64 1, ptr %71, align 8
  %72 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %72, align 8
  %73 = getelementptr { i64, ptr, i1 }, ptr %70, i32 0, i32 2
  store i1 %68, ptr %73, align 1
  br label %149

74:                                               ; preds = %65
  %75 = icmp eq ptr %29, @_ZTIx
  br i1 %75, label %76, label %83

76:                                               ; preds = %74
  %77 = load i64, ptr %28, align 8
  %78 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %76
  %79 = getelementptr i8, ptr %78, i64 8
  %80 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 0
  store i64 8, ptr %80, align 8
  %81 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %81, align 8
  %82 = getelementptr { i64, ptr, i64 }, ptr %79, i32 0, i32 2
  store i64 %77, ptr %82, align 8
  br label %146

83:                                               ; preds = %74
  %84 = icmp eq ptr %29, @_ZTIPc
  br i1 %84, label %85, label %92

85:                                               ; preds = %83
  %86 = load ptr, ptr %28, align 8
  %87 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %85
  %88 = getelementptr i8, ptr %87, i64 8
  %89 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 0
  store i64 8, ptr %89, align 8
  %90 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %90, align 8
  %91 = getelementptr { i64, ptr, ptr }, ptr %88, i32 0, i32 2
  store ptr %86, ptr %91, align 8
  br label %143

92:                                               ; preds = %83
  %93 = icmp eq ptr %29, @_ZTIPv
  br i1 %93, label %94, label %96

94:                                               ; preds = %92
  %95 = load ptr, ptr %28, align 8
  br label %140

96:                                               ; preds = %92
  %97 = load ptr, ptr %29, align 8
  %98 = icmp eq ptr %97, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %98, label %99, label %130

99:                                               ; preds = %96
  %100 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %29, i32 0, i32 2
  %101 = load i32, ptr %100, align 4
  %102 = and i32 %101, 1073741824
  %103 = icmp ne i32 %102, 0
  br i1 %103, label %104, label %120

104:                                              ; preds = %99
  %105 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %29, i32 0, i32 4
  %106 = load ptr, ptr %105, align 8
  %107 = icmp ne ptr %106, null
  br i1 %107, label %108, label %110

108:                                              ; preds = %104
  invoke void %106(ptr %2, ptr %28)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %108
  %109 = load ptr, ptr %2, align 8
  br label %117

110:                                              ; preds = %104
  %111 = load ptr, ptr %28, align 8
  %112 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %110
  %113 = getelementptr i8, ptr %112, i64 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 0
  store i64 8, ptr %114, align 8
  %115 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %115, align 8
  %116 = getelementptr { i64, ptr, ptr }, ptr %113, i32 0, i32 2
  store ptr %111, ptr %116, align 8
  br label %117

117:                                              ; preds = %.noexc, %.noexc8
  %118 = phi ptr [ %113, %.noexc8 ], [ %109, %.noexc ]
  br label %119

119:                                              ; preds = %117
  br label %127

120:                                              ; preds = %99
  %121 = load ptr, ptr %28, align 8
  %122 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %120
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 0
  store i64 8, ptr %124, align 8
  %125 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %125, align 8
  %126 = getelementptr { i64, ptr, ptr }, ptr %123, i32 0, i32 2
  store ptr %121, ptr %126, align 8
  br label %127

127:                                              ; preds = %119, %.noexc11
  %128 = phi ptr [ %123, %.noexc11 ], [ %118, %119 ]
  br label %129

129:                                              ; preds = %127
  br label %137

130:                                              ; preds = %96
  %131 = load ptr, ptr %28, align 8
  %132 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %130
  %133 = getelementptr i8, ptr %132, i64 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 0
  store i64 8, ptr %134, align 8
  %135 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %135, align 8
  %136 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 2
  store ptr %131, ptr %136, align 8
  br label %137

137:                                              ; preds = %129, %.noexc7
  %138 = phi ptr [ %133, %.noexc7 ], [ %128, %129 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %94, %139
  %141 = phi ptr [ %138, %139 ], [ %95, %94 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc3, %142
  %144 = phi ptr [ %141, %142 ], [ %88, %.noexc3 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc6, %145
  %147 = phi ptr [ %144, %145 ], [ %79, %.noexc6 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc2, %148
  %150 = phi ptr [ %147, %148 ], [ %70, %.noexc2 ]
  br label %151

151:                                              ; preds = %149
  br label %152

152:                                              ; preds = %.noexc9, %151
  %153 = phi ptr [ %150, %151 ], [ %61, %.noexc9 ]
  br label %154

154:                                              ; preds = %152
  br label %155

155:                                              ; preds = %.noexc10, %154
  %156 = phi ptr [ %153, %154 ], [ %52, %.noexc10 ]
  br label %157

157:                                              ; preds = %155
  br label %158

158:                                              ; preds = %.noexc1, %157
  %159 = phi ptr [ %156, %157 ], [ %43, %.noexc1 ]
  br label %160

160:                                              ; preds = %158
  br label %161

161:                                              ; preds = %.noexc12, %160
  %162 = phi ptr [ %159, %160 ], [ %34, %.noexc12 ]
  br label %163

163:                                              ; preds = %161
  store ptr %162, ptr %3, align 8
  invoke void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8))
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %163
  call void @__cxa_end_catch()
  br label %164

164:                                              ; preds = %22, %.noexc5
  ret void

endcatch.resume:                                  ; preds = %31, %120, %49, %58, %110, %130, %76, %163, %25, %85, %67, %40, %108
  %165 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %165
}

; Function Attrs: mustprogress
define linkonce_odr void @"Start<si32>.drop"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %"Start<si32>", ptr %3, i32 0, i32 2
  %5 = load { ptr, ptr, ptr }, ptr %4, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  %11 = call i32 %10(ptr %6)
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  %14 = call i32 %13()
  br label %15

15:                                               ; preds = %9, %12
  %16 = phi i32 [ %14, %12 ], [ %11, %9 ]
  br label %17

17:                                               ; preds = %15
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr i32 @"Start<si32>.get"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %"Start<si32>", ptr %4, i32 0, i32 1
  %6 = load i32, ptr %5, align 4
  store i32 %6, ptr %3, align 4
  %7 = load i32, ptr %3, align 4
  ret i32 %7
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"spawn<si32>"({ ptr, ptr, ptr } %0) #2 comdat {
  %2 = alloca { ptr, ptr, ptr }, align 8
  %3 = alloca ptr, align 8
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  %4 = call ptr @GC_malloc(i64 48)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %"Start<si32>", ptr %5, i32 0, i32 0
  store ptr @"Start<si32>..vtbl", ptr %6, align 8
  %7 = load { ptr, ptr, ptr }, ptr %2, align 8
  call void @"Start<si32>.constructor"(ptr %5, { ptr, ptr, ptr } %7)
  store ptr %5, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  ret ptr %8
}

; Function Attrs: mustprogress
define void @resultsKept() #2 {
  %1 = alloca i32, align 4
  %2 = alloca <{ i64, %"Start<!ts.class<Box, !ts.class_storage<Box>>>" }>, align 16
  store i64 -1, ptr %2, align 8
  %3 = getelementptr i8, ptr %2, i64 8
  store %"Start<!ts.class<Box, !ts.class_storage<Box>>>" zeroinitializer, ptr %3, align 8
  store %"Start<!ts.class<Box, !ts.class_storage<Box>>>" zeroinitializer, ptr %3, align 8
  %4 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 0
  store ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>..vtbl", ptr %4, align 8
  call void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.constructor"(ptr %3, { ptr, ptr, ptr } { ptr @makeBox, ptr null, ptr null })
  %5 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 0
  call void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.run"(ptr %3)
  call void @churn()
  %6 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 0
  %7 = call ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.get"(ptr %3)
  %8 = getelementptr %Box, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = icmp eq i32 %9, 1
  br i1 %10, label %11, label %73

11:                                               ; preds = %0
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %Box, ptr %13, i32 0, i32 0
  store ptr @Box..vtbl, ptr %14, align 8
  call void @Box.constructor(ptr %13, i32 1)
  %15 = call ptr @GC_malloc(i64 24)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr { ptr, i32 }, ptr %16, i32 0, i32 0
  store ptr %13, ptr %17, align 8
  %18 = getelementptr { ptr, i32 }, ptr %16, i32 0, i32 1
  store i32 41, ptr %18, align 4
  %19 = insertvalue { ptr, ptr, ptr } { ptr @.f_resultsKept..afL138C25FH3418460370, ptr null, ptr null }, ptr %16, 1
  %20 = insertvalue { ptr, ptr, ptr } %19, ptr getelementptr inbounds nuw (i8, ptr @td_7267384957113499010_object, i64 40), 2
  %21 = call ptr @"spawn<!ts.class<Box, !ts.class_storage<Box>>>"({ ptr, ptr, ptr } %20)
  %22 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %21, i32 0, i32 0
  call void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.run"(ptr %21)
  %23 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %21, i32 0, i32 0
  call void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.runInTry"(ptr %21)
  call void @churn()
  %24 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %21, i32 0, i32 0
  %25 = call ptr @"Start<!ts.class<Box, !ts.class_storage<Box>>>.get"(ptr %21)
  %26 = getelementptr %Box, ptr %25, i32 0, i32 1
  %27 = load i32, ptr %26, align 4
  %28 = icmp eq i32 %27, 42
  br i1 %28, label %29, label %75

29:                                               ; preds = %11
  store i32 0, ptr %1, align 4
  br label %30

30:                                               ; preds = %33, %29
  %31 = load i32, ptr %1, align 4
  %32 = icmp slt i32 %31, 1000
  br i1 %32, label %33, label %37

33:                                               ; preds = %30
  %34 = getelementptr %"Start<!ts.class<Box, !ts.class_storage<Box>>>", ptr %3, i32 0, i32 0
  call void @"Start<!ts.class<Box, !ts.class_storage<Box>>>.drop"(ptr %3)
  %35 = load i32, ptr %1, align 4
  %36 = add i32 %35, 1
  store i32 %36, ptr %1, align 4
  br label %30

37:                                               ; preds = %30
  %38 = call ptr @GC_malloc(i64 12)
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr { i32 }, ptr %39, i32 0, i32 0
  store i32 41, ptr %40, align 4
  %41 = insertvalue { ptr, ptr, ptr } { ptr @.f_resultsKept..afL152C24FH3418460370, ptr null, ptr null }, ptr %39, 1
  %42 = insertvalue { ptr, ptr, ptr } %41, ptr getelementptr inbounds nuw (i8, ptr @td_15336564547564024679_object, i64 40), 2
  %43 = call ptr @"spawn<!ts.string>"({ ptr, ptr, ptr } %42)
  %44 = getelementptr %"Start<!ts.string>", ptr %43, i32 0, i32 0
  call void @"Start<!ts.string>.run"(ptr %43)
  %45 = getelementptr %"Start<!ts.string>", ptr %43, i32 0, i32 0
  %46 = call ptr @"Start<!ts.string>.get"(ptr %43)
  %47 = ptrtoint ptr %46 to i64
  %48 = icmp ne i64 %47, 0
  %49 = zext i1 %48 to i32
  %50 = and i32 %49, 1
  %51 = icmp ne i32 %50, 0
  br i1 %51, label %52, label %55

52:                                               ; preds = %37
  %53 = call i32 @strcmp(ptr %46, ptr getelementptr inbounds nuw (i8, ptr @s_16602643279055812603, i64 8))
  %54 = icmp eq i32 %53, 0
  br label %57

55:                                               ; preds = %37
  %56 = icmp eq i64 %47, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16602643279055812603, i64 8) to i64)
  br label %57

57:                                               ; preds = %52, %55
  %58 = phi i1 [ %56, %55 ], [ %54, %52 ]
  br label %59

59:                                               ; preds = %57
  br i1 %58, label %60, label %77

60:                                               ; preds = %59
  %61 = call ptr @GC_malloc(i64 12)
  %62 = getelementptr i8, ptr %61, i64 8
  %63 = getelementptr { i32 }, ptr %62, i32 0, i32 0
  store i32 41, ptr %63, align 4
  %64 = insertvalue { ptr, ptr, ptr } { ptr @.f_resultsKept..afL156C26FH3418460370, ptr null, ptr null }, ptr %62, 1
  %65 = insertvalue { ptr, ptr, ptr } %64, ptr getelementptr inbounds nuw (i8, ptr @td_15336564547564024679_object, i64 40), 2
  %66 = call ptr @"spawn<si32>"({ ptr, ptr, ptr } %65)
  %67 = getelementptr %"Start<si32>", ptr %66, i32 0, i32 0
  call void @"Start<si32>.run"(ptr %66)
  %68 = getelementptr %"Start<si32>", ptr %66, i32 0, i32 0
  %69 = call i32 @"Start<si32>.get"(ptr %66)
  %70 = icmp eq i32 %69, 42
  br i1 %70, label %71, label %79

71:                                               ; preds = %60
  ret void

72:                                               ; preds = %79, %77, %75, %73
  unreachable

73:                                               ; preds = %0
  %74 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 134, ptr null)
  br label %72

75:                                               ; preds = %11
  %76 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 143, ptr null)
  br label %72

77:                                               ; preds = %59
  %78 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 154, ptr null)
  br label %72

79:                                               ; preds = %60
  %80 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 158, ptr null)
  br label %72
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  call void @closuresShareOne()
  call void @closuresInLoop()
  %2 = call ptr @closureOutlivesTheLocal()
  call void @churn()
  %3 = getelementptr %Keeper, ptr %2, i32 0, i32 1
  %4 = load { ptr, ptr, ptr }, ptr %3, align 8
  %5 = extractvalue { ptr, ptr, ptr } %4, 1
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %0
  %9 = extractvalue { ptr, ptr, ptr } %4, 0
  %10 = call i32 %9(ptr %5)
  br label %14

11:                                               ; preds = %0
  %12 = extractvalue { ptr, ptr, ptr } %4, 0
  %13 = call i32 %12()
  br label %14

14:                                               ; preds = %8, %11
  %15 = phi i32 [ %13, %11 ], [ %10, %8 ]
  br label %16

16:                                               ; preds = %14
  %17 = icmp eq i32 %15, 7
  br i1 %17, label %18, label %27

18:                                               ; preds = %16
  %19 = call ptr @GC_malloc(i64 24)
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr %Box, ptr %20, i32 0, i32 0
  store ptr @Box..vtbl, ptr %21, align 8
  call void @Box.constructor(ptr %20, i32 3)
  %22 = call ptr @GC_malloc(i64 16)
  %23 = getelementptr i8, ptr %22, i64 8
  store ptr %20, ptr %23, align 8
  store ptr %23, ptr %1, align 8
  %24 = load ptr, ptr %1, align 8
  call void @borrowingBoxes(ptr %24)
  %25 = load ptr, ptr %1, align 8
  call void @expectCount(double -1.000000e+00, double 1.000000e+00)
  call void @resultsKept()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

26:                                               ; preds = %27
  unreachable

27:                                               ; preds = %16
  %28 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_11817493874169425794, i64 8), i32 167, ptr null)
  br label %26
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
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

