; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Later = type { ptr, { ptr, ptr, ptr } }
%Keeper = type { ptr, { ptr, ptr, ptr } }
%Walker = type { ptr, { ptr, ptr, ptr } }
%Item = type { ptr, double }

$Later.Holder..vtbl.f.thunk = comdat any

$"___unbox_interface<!ts.iface<Holder>>" = comdat any

$"___unbox_interface<!ts.iface<Runner>>" = comdat any

$Item..size = comdat any

$Walker..size = comdat any

$Keeper..size = comdat any

$Later..size = comdat any

$Later.Holder..vtbl.f.method = comdat any

$Later.Holder..vtbl = comdat any

$Holder.null..vtbl = comdat any

@s_5967165952864249862 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFItem\00", align 8
@s_16222276863397847335 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFWalker\00", align 8
@s_10708658430971434591 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFKeeper\00", align 8
@s_13193572148966979017 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFLater\00", align 8
@s_2519477130935904862 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmethod is abstract\00", align 8
@s_2495249626774661283 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_5656299461393339625 = internal constant [84 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: an object literal has no vtable for interface 'Holder'\00", align 8
@s_10458455444534216913 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtuple\00", align 8
@s_10231808476453998586 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFobject\00", align 8
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_4216450727644127646 = internal constant [83 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the class object has no vtable for interface 'Holder'\00", align 8
@s_15633735204921351509 = internal constant [82 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: class 'Walker' does not implement interface 'Holder'\00", align 8
@s_927594501218240418 = internal constant [80 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: class 'Item' does not implement interface 'Holder'\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@m_10917319126229082950 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa Runner over a Walker is no Holder\00", align 8
@f_12745439577603844570 = internal constant [108 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00any_to_interface.ts\00", align 8
@m_2179998208988451237 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa Walker is no Holder\00", align 8
@s_14806728300850859021 = internal constant [84 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: an object literal has no vtable for interface 'Runner'\00", align 8
@s_1157645605329739794 = internal constant [83 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the class object has no vtable for interface 'Runner'\00", align 8
@s_6007013098191354486 = internal constant [80 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: class 'Item' does not implement interface 'Runner'\00", align 8
@s_10753497107086709977 = internal constant [82 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: class 'Keeper' does not implement interface 'Runner'\00", align 8
@s_7324454017586831841 = internal constant [81 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: class 'Later' does not implement interface 'Runner'\00", align 8
@m_13538094400678585102 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFround trip\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_3166235262791310685 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery Holder reached show\00", align 8
@Item..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_5967165952864249862, i64 8)
@Item..size = global i64 16, comdat
@Item..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Item..instanceOf, ptr @Item..new, ptr @Item..rtti, ptr @Item..size }
@Walker..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_16222276863397847335, i64 8)
@Walker..size = global i64 32, comdat
@Walker.Runner..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_7966825562501435538_class, i64 40), ptr @Walker..instanceOf, ptr @Walker.walk }
@td_7966825562501435538_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_7966825562501435538, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Walker..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Walker..instanceOf, ptr @Walker.Runner..vtbl, ptr @Walker..new, ptr @Walker.walk, ptr @Walker..rtti, ptr @Walker..size }
@Keeper..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_10708658430971434591, i64 8)
@Keeper..size = global i64 32, comdat
@Keeper.Holder..vtbl.f.method = private global { ptr, i64, i64 } { ptr @Keeper.Holder..vtbl.f.thunk, i64 -1, i64 8 }
@Keeper.Holder..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12419312426083759605_class, i64 40), ptr @Keeper..instanceOf, ptr inttoptr (i64 add (i64 ptrtoint (ptr @Keeper.Holder..vtbl.f.method to i64), i64 1) to ptr) }
@td_12419312426083759605_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_12419312426083759605, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Keeper..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Keeper..instanceOf, ptr @Keeper.Holder..vtbl, ptr @Keeper..new, ptr @Keeper..rtti, ptr @Keeper..size }
@Later..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_13193572148966979017, i64 8)
@Later..size = global i64 32, comdat
@Later..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Later..instanceOf, ptr @Later..new, ptr @Later..rtti, ptr @Later..size }
@shown = private global double 0.000000e+00
@_ZTIPc = external constant ptr
@Runner.null..vtbl = private global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr @Runner.null.walk }
@Later.Holder..vtbl.f.method = linkonce_odr global { ptr, i64, i64 } { ptr @Later.Holder..vtbl.f.thunk, i64 -1, i64 8 }, comdat
@Later.Holder..vtbl = linkonce_odr global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_1841389987134824288_class, i64 40), ptr @Later.Holder..vtbl..instanceOf.dispatch, ptr inttoptr (i64 add (i64 ptrtoint (ptr @Later.Holder..vtbl.f.method to i64), i64 1) to ptr) }, comdat
@td_1841389987134824288_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_1841389987134824288, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Holder.null..vtbl = linkonce_odr global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr null }, comdat
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@td_14063698103718993911_interface = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [10 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 9, i32 -178799487, ptr @tsrel_14063698103718993911, ptr @tsret_14063698103718993911, ptr null, i64 -1 }, [10 x i8] c"interface\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_9375918230080828099_interface = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [10 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 9, i32 -658052112, ptr @tsrel_9375918230080828099, ptr @tsret_9375918230080828099, ptr null, i64 -1 }, [10 x i8] c"interface\00" }, align 8
@a_0_3230914496259700488 = internal constant [0 x ptr] undef

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

define internal void @tsret_9375918230080828099(ptr %0) {
  %2 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %18

6:                                                ; preds = %1
  %7 = getelementptr ptr, ptr %3, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = getelementptr i8, ptr %8, i64 -40
  %12 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %11, i32 0, i32 3
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void %13(ptr %4)
  br label %16

16:                                               ; preds = %15, %10
  br label %17

17:                                               ; preds = %16, %6
  br label %18

18:                                               ; preds = %17, %1
  ret void
}

define internal void @tsrel_9375918230080828099(ptr %0) {
  %2 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %18

6:                                                ; preds = %1
  %7 = getelementptr ptr, ptr %3, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = getelementptr i8, ptr %8, i64 -40
  %12 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %11, i32 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void %13(ptr %4)
  br label %16

16:                                               ; preds = %15, %10
  br label %17

17:                                               ; preds = %16, %6
  br label %18

18:                                               ; preds = %17, %1
  ret void
}

declare void @__cxa_end_catch()

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

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

define internal void @tsret_14063698103718993911(ptr %0) {
  %2 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %18

6:                                                ; preds = %1
  %7 = getelementptr ptr, ptr %3, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = getelementptr i8, ptr %8, i64 -40
  %12 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %11, i32 0, i32 3
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void %13(ptr %4)
  br label %16

16:                                               ; preds = %15, %10
  br label %17

17:                                               ; preds = %16, %6
  br label %18

18:                                               ; preds = %17, %1
  ret void
}

define internal void @tsrel_14063698103718993911(ptr %0) {
  %2 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %18

6:                                                ; preds = %1
  %7 = getelementptr ptr, ptr %3, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = getelementptr i8, ptr %8, i64 -40
  %12 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %11, i32 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = icmp ne ptr %13, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void %13(ptr %4)
  br label %16

16:                                               ; preds = %15, %10
  br label %17

17:                                               ; preds = %16, %6
  br label %18

18:                                               ; preds = %17, %1
  ret void
}

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_1841389987134824288(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Later, ptr %2, i32 0, i32 1
  call void @tsrel_553310009159819399(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @tsrel_12419312426083759605(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Keeper, ptr %2, i32 0, i32 1
  call void @tsrel_553310009159819399(ptr %7)
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

define internal void @tsrel_553310009159819399(ptr %0) {
  %2 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %5 = icmp ne ptr %3, null
  br i1 %5, label %6, label %13

6:                                                ; preds = %1
  %7 = getelementptr i8, ptr %3, i64 -40
  %8 = getelementptr { i32, i32, ptr, ptr, ptr, i64 }, ptr %7, i32 0, i32 2
  %9 = load ptr, ptr %8, align 8
  %10 = icmp ne ptr %9, null
  br i1 %10, label %11, label %12

11:                                               ; preds = %6
  call void %9(ptr %4)
  br label %12

12:                                               ; preds = %11, %6
  br label %13

13:                                               ; preds = %12, %1
  ret void
}

define internal void @tsrel_7966825562501435538(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Walker, ptr %2, i32 0, i32 1
  call void @tsrel_553310009159819399(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

; Function Attrs: mustprogress
define i1 @Item..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Item..rtti, align 8
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
define ptr @Item..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Item, ptr %3, i32 0, i32 0
  store ptr @Item..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Item.constructor(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Item, ptr %5, i32 0, i32 1
  %7 = load double, ptr %3, align 8
  store double %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Walker..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Walker..rtti, align 8
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
define ptr @Walker..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Walker, ptr %3, i32 0, i32 0
  store ptr @Walker..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Walker.constructor(ptr %0, { ptr, ptr, ptr } %1) #2 {
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store { ptr, ptr, ptr } %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Walker, ptr %5, i32 0, i32 1
  %7 = load { ptr, ptr, ptr }, ptr %3, align 8
  store { ptr, ptr, ptr } %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Walker.walk(ptr %0, ptr %1) #2 {
  %3 = alloca double, align 8
  store double 1.000000e+00, ptr %3, align 8
  %4 = load double, ptr %3, align 8
  ret double %4
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
define void @Keeper.Holder..vtbl.f.thunk(ptr %0, ptr %1) #2 {
  %3 = getelementptr %Keeper, ptr %0, i32 0, i32 1
  %4 = load { ptr, ptr, ptr }, ptr %3, align 8
  %5 = extractvalue { ptr, ptr, ptr } %4, 1
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  br i1 %7, label %8, label %10

8:                                                ; preds = %2
  %9 = extractvalue { ptr, ptr, ptr } %4, 0
  call void %9(ptr %5, ptr %1)
  br label %12

10:                                               ; preds = %2
  %11 = extractvalue { ptr, ptr, ptr } %4, 0
  call void %11(ptr %1)
  br label %12

12:                                               ; preds = %8, %10
  br label %13

13:                                               ; preds = %12
  ret void
}

; Function Attrs: mustprogress
define i1 @Later..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Later..rtti, align 8
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
define ptr @Later..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 40)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Later, ptr %3, i32 0, i32 0
  store ptr @Later..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Later.constructor(ptr %0, { ptr, ptr, ptr } %1) #2 {
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store { ptr, ptr, ptr } %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Later, ptr %5, i32 0, i32 1
  %7 = load { ptr, ptr, ptr }, ptr %3, align 8
  store { ptr, ptr, ptr } %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @show(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load double, ptr @shown, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Item, ptr %4, i32 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = fadd double %3, %6
  store double %7, ptr @shown, align 8
  ret void
}

; Function Attrs: mustprogress
define double @Runner.null.walk(ptr %0, ptr %1) #3 {
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
define linkonce_odr void @Later.Holder..vtbl.f.thunk(ptr %0, ptr %1) #2 comdat {
  %3 = getelementptr %Later, ptr %0, i32 0, i32 1
  %4 = load { ptr, ptr, ptr }, ptr %3, align 8
  %5 = extractvalue { ptr, ptr, ptr } %4, 1
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  br i1 %7, label %8, label %10

8:                                                ; preds = %2
  %9 = extractvalue { ptr, ptr, ptr } %4, 0
  call void %9(ptr %5, ptr %1)
  br label %12

10:                                               ; preds = %2
  %11 = extractvalue { ptr, ptr, ptr } %4, 0
  call void %11(ptr %1)
  br label %12

12:                                               ; preds = %8, %10
  br label %13

13:                                               ; preds = %12
  ret void
}

; Function Attrs: mustprogress
define internal i1 @Later.Holder..vtbl..instanceOf.dispatch(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  %4 = getelementptr %Later, ptr %0, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 0
  %7 = load ptr, ptr %6, align 8
  %8 = call i1 %7(ptr %0, ptr %1)
  store i1 %8, ptr %3, align 1
  %9 = load i1, ptr %3, align 1
  ret i1 %9
}

; Function Attrs: mustprogress
define linkonce_odr { ptr, ptr } @"___unbox_interface<!ts.iface<Holder>>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, ptr }, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr { i64, ptr, i8 }, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %1
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %1
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %90

21:                                               ; preds = %20
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 2
  %24 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 0
  %25 = load i64, ptr %24, align 8
  %26 = load i64, ptr %23, align 8
  %27 = icmp uge i64 %25, 8
  %28 = select i1 %27, i64 0, i64 %25
  %29 = mul i64 %28, 8
  %30 = shl i64 1, %29
  %31 = sub i64 %30, 1
  %32 = and i64 %26, %31
  %33 = select i1 %27, i64 %26, i64 %32
  %34 = inttoptr i64 %33 to ptr
  %35 = getelementptr ptr, ptr %34, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr ptr, ptr %36, i32 0
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr @Later..rtti, align 8
  %40 = call i1 %38(ptr %34, ptr %39)
  br i1 %40, label %41, label %50

41:                                               ; preds = %21
  %42 = ptrtoint ptr %34 to i64
  %43 = icmp ne i64 %42, 0
  br i1 %43, label %44, label %45

44:                                               ; preds = %41
  br label %46

45:                                               ; preds = %41
  br label %46

46:                                               ; preds = %44, %45
  %47 = phi ptr [ @Holder.null..vtbl, %45 ], [ @Later.Holder..vtbl, %44 ]
  %48 = insertvalue { ptr, ptr } undef, ptr %47, 0
  %49 = insertvalue { ptr, ptr } %48, ptr %34, 1
  store { ptr, ptr } %49, ptr %3, align 8
  br label %328

50:                                               ; preds = %21
  %51 = getelementptr ptr, ptr %34, i32 0
  %52 = load ptr, ptr %51, align 8
  %53 = getelementptr ptr, ptr %52, i32 0
  %54 = load ptr, ptr %53, align 8
  %55 = load ptr, ptr @Keeper..rtti, align 8
  %56 = call i1 %54(ptr %34, ptr %55)
  br i1 %56, label %57, label %70

57:                                               ; preds = %50
  %58 = ptrtoint ptr %34 to i64
  %59 = icmp ne i64 %58, 0
  br i1 %59, label %60, label %65

60:                                               ; preds = %57
  %61 = getelementptr %Keeper, ptr %34, i32 0, i32 0
  %62 = load ptr, ptr %61, align 8
  %63 = getelementptr ptr, ptr %62, i32 1
  %64 = load ptr, ptr %63, align 8
  br label %66

65:                                               ; preds = %57
  br label %66

66:                                               ; preds = %60, %65
  %67 = phi ptr [ @Holder.null..vtbl, %65 ], [ %64, %60 ]
  %68 = insertvalue { ptr, ptr } undef, ptr %67, 0
  %69 = insertvalue { ptr, ptr } %68, ptr %34, 1
  store { ptr, ptr } %69, ptr %3, align 8
  br label %328

70:                                               ; preds = %50
  %71 = getelementptr ptr, ptr %34, i32 0
  %72 = load ptr, ptr %71, align 8
  %73 = getelementptr ptr, ptr %72, i32 0
  %74 = load ptr, ptr %73, align 8
  %75 = load ptr, ptr @Item..rtti, align 8
  %76 = call i1 %74(ptr %34, ptr %75)
  br i1 %76, label %77, label %79

77:                                               ; preds = %70
  %78 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_927594501218240418, i64 8), ptr %78, align 8
  call void @__cxa_throw(ptr %78, ptr @_ZTIPc, ptr null)
  br label %330

79:                                               ; preds = %70
  %80 = getelementptr ptr, ptr %34, i32 0
  %81 = load ptr, ptr %80, align 8
  %82 = getelementptr ptr, ptr %81, i32 0
  %83 = load ptr, ptr %82, align 8
  %84 = load ptr, ptr @Walker..rtti, align 8
  %85 = call i1 %83(ptr %34, ptr %84)
  br i1 %85, label %86, label %88

86:                                               ; preds = %79
  %87 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_15633735204921351509, i64 8), ptr %87, align 8
  call void @__cxa_throw(ptr %87, ptr @_ZTIPc, ptr null)
  br label %330

88:                                               ; preds = %79
  %89 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4216450727644127646, i64 8), ptr %89, align 8
  call void @__cxa_throw(ptr %89, ptr @_ZTIPc, ptr null)
  br label %330

90:                                               ; preds = %20
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
  %100 = call i32 @strcmp(ptr %93, ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8))
  %101 = icmp eq i32 %100, 0
  br label %104

102:                                              ; preds = %90
  %103 = icmp eq i64 %94, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8) to i64)
  br label %104

104:                                              ; preds = %99, %102
  %105 = phi i1 [ %103, %102 ], [ %101, %99 ]
  br label %106

106:                                              ; preds = %104
  br i1 %105, label %107, label %249

107:                                              ; preds = %106
  %108 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %4, i32 0, i32 2
  %109 = load { ptr, ptr }, ptr %108, align 8
  %110 = extractvalue { ptr, ptr } %109, 1
  %111 = ptrtoint ptr %110 to i64
  %112 = icmp ne i64 %111, 0
  br i1 %112, label %113, label %120

113:                                              ; preds = %107
  %114 = extractvalue { ptr, ptr } %109, 0
  %115 = extractvalue { ptr, ptr } %109, 1
  %116 = getelementptr ptr, ptr %114, i32 1
  %117 = load ptr, ptr %116, align 8
  %118 = load ptr, ptr @Later..rtti, align 8
  %119 = call i1 %117(ptr %115, ptr %118)
  br label %120

120:                                              ; preds = %113, %107
  %121 = phi i1 [ %119, %113 ], [ false, %107 ]
  br i1 %121, label %122, label %147

122:                                              ; preds = %120
  %123 = extractvalue { ptr, ptr } %109, 1
  %124 = ptrtoint ptr %123 to i64
  %125 = icmp ne i64 %124, 0
  br i1 %125, label %126, label %133

126:                                              ; preds = %122
  %127 = extractvalue { ptr, ptr } %109, 0
  %128 = extractvalue { ptr, ptr } %109, 1
  %129 = getelementptr ptr, ptr %127, i32 1
  %130 = load ptr, ptr %129, align 8
  %131 = load ptr, ptr @Later..rtti, align 8
  %132 = call i1 %130(ptr %128, ptr %131)
  br label %133

133:                                              ; preds = %126, %122
  %134 = phi i1 [ %132, %126 ], [ false, %122 ]
  br i1 %134, label %135, label %139

135:                                              ; preds = %133
  %136 = extractvalue { ptr, ptr } %109, 1
  %137 = ptrtoint ptr %136 to i64
  %138 = icmp ne i64 %137, 0
  br i1 %138, label %141, label %142

139:                                              ; preds = %133
  %140 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %140, align 8
  call void @__cxa_throw(ptr %140, ptr @_ZTIPc, ptr null)
  br label %330

141:                                              ; preds = %135
  br label %143

142:                                              ; preds = %135
  br label %143

143:                                              ; preds = %141, %142
  %144 = phi ptr [ @Holder.null..vtbl, %142 ], [ @Later.Holder..vtbl, %141 ]
  %145 = insertvalue { ptr, ptr } undef, ptr %144, 0
  %146 = insertvalue { ptr, ptr } %145, ptr %136, 1
  store { ptr, ptr } %146, ptr %3, align 8
  br label %328

147:                                              ; preds = %120
  %148 = extractvalue { ptr, ptr } %109, 1
  %149 = ptrtoint ptr %148 to i64
  %150 = icmp ne i64 %149, 0
  br i1 %150, label %151, label %158

151:                                              ; preds = %147
  %152 = extractvalue { ptr, ptr } %109, 0
  %153 = extractvalue { ptr, ptr } %109, 1
  %154 = getelementptr ptr, ptr %152, i32 1
  %155 = load ptr, ptr %154, align 8
  %156 = load ptr, ptr @Keeper..rtti, align 8
  %157 = call i1 %155(ptr %153, ptr %156)
  br label %158

158:                                              ; preds = %151, %147
  %159 = phi i1 [ %157, %151 ], [ false, %147 ]
  br i1 %159, label %160, label %189

160:                                              ; preds = %158
  %161 = extractvalue { ptr, ptr } %109, 1
  %162 = ptrtoint ptr %161 to i64
  %163 = icmp ne i64 %162, 0
  br i1 %163, label %164, label %171

164:                                              ; preds = %160
  %165 = extractvalue { ptr, ptr } %109, 0
  %166 = extractvalue { ptr, ptr } %109, 1
  %167 = getelementptr ptr, ptr %165, i32 1
  %168 = load ptr, ptr %167, align 8
  %169 = load ptr, ptr @Keeper..rtti, align 8
  %170 = call i1 %168(ptr %166, ptr %169)
  br label %171

171:                                              ; preds = %164, %160
  %172 = phi i1 [ %170, %164 ], [ false, %160 ]
  br i1 %172, label %173, label %177

173:                                              ; preds = %171
  %174 = extractvalue { ptr, ptr } %109, 1
  %175 = ptrtoint ptr %174 to i64
  %176 = icmp ne i64 %175, 0
  br i1 %176, label %179, label %184

177:                                              ; preds = %171
  %178 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %178, align 8
  call void @__cxa_throw(ptr %178, ptr @_ZTIPc, ptr null)
  br label %330

179:                                              ; preds = %173
  %180 = getelementptr %Keeper, ptr %174, i32 0, i32 0
  %181 = load ptr, ptr %180, align 8
  %182 = getelementptr ptr, ptr %181, i32 1
  %183 = load ptr, ptr %182, align 8
  br label %185

184:                                              ; preds = %173
  br label %185

185:                                              ; preds = %179, %184
  %186 = phi ptr [ @Holder.null..vtbl, %184 ], [ %183, %179 ]
  %187 = insertvalue { ptr, ptr } undef, ptr %186, 0
  %188 = insertvalue { ptr, ptr } %187, ptr %174, 1
  store { ptr, ptr } %188, ptr %3, align 8
  br label %328

189:                                              ; preds = %158
  %190 = extractvalue { ptr, ptr } %109, 1
  %191 = ptrtoint ptr %190 to i64
  %192 = icmp ne i64 %191, 0
  br i1 %192, label %193, label %200

193:                                              ; preds = %189
  %194 = extractvalue { ptr, ptr } %109, 0
  %195 = extractvalue { ptr, ptr } %109, 1
  %196 = getelementptr ptr, ptr %194, i32 1
  %197 = load ptr, ptr %196, align 8
  %198 = load ptr, ptr @Item..rtti, align 8
  %199 = call i1 %197(ptr %195, ptr %198)
  br label %200

200:                                              ; preds = %193, %189
  %201 = phi i1 [ %199, %193 ], [ false, %189 ]
  br i1 %201, label %202, label %219

202:                                              ; preds = %200
  %203 = extractvalue { ptr, ptr } %109, 1
  %204 = ptrtoint ptr %203 to i64
  %205 = icmp ne i64 %204, 0
  br i1 %205, label %206, label %213

206:                                              ; preds = %202
  %207 = extractvalue { ptr, ptr } %109, 0
  %208 = extractvalue { ptr, ptr } %109, 1
  %209 = getelementptr ptr, ptr %207, i32 1
  %210 = load ptr, ptr %209, align 8
  %211 = load ptr, ptr @Item..rtti, align 8
  %212 = call i1 %210(ptr %208, ptr %211)
  br label %213

213:                                              ; preds = %206, %202
  %214 = phi i1 [ %212, %206 ], [ false, %202 ]
  br i1 %214, label %215, label %217

215:                                              ; preds = %213
  %216 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_927594501218240418, i64 8), ptr %216, align 8
  call void @__cxa_throw(ptr %216, ptr @_ZTIPc, ptr null)
  br label %330

217:                                              ; preds = %213
  %218 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %218, align 8
  call void @__cxa_throw(ptr %218, ptr @_ZTIPc, ptr null)
  br label %330

219:                                              ; preds = %200
  %220 = extractvalue { ptr, ptr } %109, 1
  %221 = ptrtoint ptr %220 to i64
  %222 = icmp ne i64 %221, 0
  br i1 %222, label %223, label %230

223:                                              ; preds = %219
  %224 = extractvalue { ptr, ptr } %109, 0
  %225 = extractvalue { ptr, ptr } %109, 1
  %226 = getelementptr ptr, ptr %224, i32 1
  %227 = load ptr, ptr %226, align 8
  %228 = load ptr, ptr @Walker..rtti, align 8
  %229 = call i1 %227(ptr %225, ptr %228)
  br label %230

230:                                              ; preds = %223, %219
  %231 = phi i1 [ %229, %223 ], [ false, %219 ]
  br i1 %231, label %232, label %249

232:                                              ; preds = %230
  %233 = extractvalue { ptr, ptr } %109, 1
  %234 = ptrtoint ptr %233 to i64
  %235 = icmp ne i64 %234, 0
  br i1 %235, label %236, label %243

236:                                              ; preds = %232
  %237 = extractvalue { ptr, ptr } %109, 0
  %238 = extractvalue { ptr, ptr } %109, 1
  %239 = getelementptr ptr, ptr %237, i32 1
  %240 = load ptr, ptr %239, align 8
  %241 = load ptr, ptr @Walker..rtti, align 8
  %242 = call i1 %240(ptr %238, ptr %241)
  br label %243

243:                                              ; preds = %236, %232
  %244 = phi i1 [ %242, %236 ], [ false, %232 ]
  br i1 %244, label %245, label %247

245:                                              ; preds = %243
  %246 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_15633735204921351509, i64 8), ptr %246, align 8
  call void @__cxa_throw(ptr %246, ptr @_ZTIPc, ptr null)
  br label %330

247:                                              ; preds = %243
  %248 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %248, align 8
  call void @__cxa_throw(ptr %248, ptr @_ZTIPc, ptr null)
  br label %330

249:                                              ; preds = %230, %106
  %250 = load ptr, ptr %2, align 8
  %251 = getelementptr { i64, ptr, i8 }, ptr %250, i32 0, i32 1
  %252 = load ptr, ptr %251, align 8
  %253 = ptrtoint ptr %252 to i64
  %254 = icmp ne i64 %253, 0
  %255 = zext i1 %254 to i32
  %256 = and i32 %255, 1
  %257 = icmp ne i32 %256, 0
  br i1 %257, label %258, label %261

258:                                              ; preds = %249
  %259 = call i32 @strcmp(ptr %252, ptr getelementptr inbounds nuw (i8, ptr @s_10231808476453998586, i64 8))
  %260 = icmp eq i32 %259, 0
  br label %263

261:                                              ; preds = %249
  %262 = icmp eq i64 %253, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10231808476453998586, i64 8) to i64)
  br label %263

263:                                              ; preds = %258, %261
  %264 = phi i1 [ %262, %261 ], [ %260, %258 ]
  br label %265

265:                                              ; preds = %263
  br i1 %264, label %283, label %266

266:                                              ; preds = %265
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
  %276 = call i32 @strcmp(ptr %269, ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8))
  %277 = icmp eq i32 %276, 0
  br label %280

278:                                              ; preds = %266
  %279 = icmp eq i64 %270, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8) to i64)
  br label %280

280:                                              ; preds = %275, %278
  %281 = phi i1 [ %279, %278 ], [ %277, %275 ]
  br label %282

282:                                              ; preds = %280
  br label %283

283:                                              ; preds = %282, %265
  %284 = phi i1 [ %281, %282 ], [ %264, %265 ]
  br i1 %284, label %285, label %287

285:                                              ; preds = %283
  %286 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_5656299461393339625, i64 8), ptr %286, align 8
  call void @__cxa_throw(ptr %286, ptr @_ZTIPc, ptr null)
  br label %330

287:                                              ; preds = %283
  %288 = load ptr, ptr %2, align 8
  %289 = getelementptr { i64, ptr, i8 }, ptr %288, i32 0, i32 1
  %290 = load ptr, ptr %289, align 8
  %291 = ptrtoint ptr %290 to i64
  %292 = icmp ne i64 %291, 0
  %293 = zext i1 %292 to i32
  %294 = and i32 %293, 1
  %295 = icmp ne i32 %294, 0
  br i1 %295, label %296, label %299

296:                                              ; preds = %287
  %297 = call i32 @strcmp(ptr %290, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8))
  %298 = icmp eq i32 %297, 0
  br label %301

299:                                              ; preds = %287
  %300 = icmp eq i64 %291, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8) to i64)
  br label %301

301:                                              ; preds = %296, %299
  %302 = phi i1 [ %300, %299 ], [ %298, %296 ]
  br label %303

303:                                              ; preds = %301
  br i1 %302, label %321, label %304

304:                                              ; preds = %303
  %305 = load ptr, ptr %2, align 8
  %306 = getelementptr { i64, ptr, i8 }, ptr %305, i32 0, i32 1
  %307 = load ptr, ptr %306, align 8
  %308 = ptrtoint ptr %307 to i64
  %309 = icmp ne i64 %308, 0
  %310 = zext i1 %309 to i32
  %311 = and i32 %310, 1
  %312 = icmp ne i32 %311, 0
  br i1 %312, label %313, label %316

313:                                              ; preds = %304
  %314 = call i32 @strcmp(ptr %307, ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8))
  %315 = icmp eq i32 %314, 0
  br label %318

316:                                              ; preds = %304
  %317 = icmp eq i64 %308, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8) to i64)
  br label %318

318:                                              ; preds = %313, %316
  %319 = phi i1 [ %317, %316 ], [ %315, %313 ]
  br label %320

320:                                              ; preds = %318
  br label %321

321:                                              ; preds = %320, %303
  %322 = phi i1 [ %319, %320 ], [ %302, %303 ]
  br i1 %322, label %323, label %324

323:                                              ; preds = %321
  store { ptr, ptr } zeroinitializer, ptr %3, align 8
  br label %328

324:                                              ; preds = %321
  %325 = load ptr, ptr %2, align 8
  %326 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %325, i32 0, i32 2
  %327 = load { ptr, ptr }, ptr %326, align 8
  store { ptr, ptr } %327, ptr %3, align 8
  br label %328

328:                                              ; preds = %46, %66, %143, %185, %323, %324
  %329 = load { ptr, ptr }, ptr %3, align 8
  ret { ptr, ptr } %329

330:                                              ; preds = %77, %86, %88, %139, %177, %215, %217, %245, %247, %285
  unreachable
}

; Function Attrs: mustprogress
define void @otherInterface() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = call ptr @GC_malloc(i64 40)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Walker, ptr %5, i32 0, i32 0
  store ptr @Walker..vtbl, ptr %6, align 8
  invoke void @Walker.constructor(ptr %5, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
          to label %7 unwind label %57

7:                                                ; preds = %0
  %8 = ptrtoint ptr %5 to i64
  %9 = icmp ne i64 %8, 0
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr %Walker, ptr %5, i32 0, i32 0
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr ptr, ptr %12, i32 1
  %14 = load ptr, ptr %13, align 8
  br label %16

15:                                               ; preds = %7
  br label %16

16:                                               ; preds = %10, %15
  %17 = phi ptr [ @Runner.null..vtbl, %15 ], [ %14, %10 ]
  %18 = insertvalue { ptr, ptr } undef, ptr %17, 0
  %19 = insertvalue { ptr, ptr } %18, ptr %5, 1
  %20 = call ptr @GC_malloc(i64 40)
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %21, i32 0, i32 0
  store i64 16, ptr %22, align 8
  %23 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %21, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_14063698103718993911_interface, i64 40), ptr %23, align 8
  %24 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %21, i32 0, i32 2
  store { ptr, ptr } %19, ptr %24, align 8
  %25 = invoke { ptr, ptr } @"___unbox_interface<!ts.iface<Holder>>"(ptr %21)
          to label %26 unwind label %57

26:                                               ; preds = %16
  %27 = extractvalue { ptr, ptr } %25, 0
  %28 = extractvalue { ptr, ptr } %25, 1
  %29 = getelementptr ptr, ptr %27, i32 2
  %30 = load ptr, ptr %29, align 8
  %31 = ptrtoint ptr %30 to i64
  %32 = and i64 %31, 1
  %33 = icmp ne i64 %32, 0
  br i1 %33, label %34, label %37

34:                                               ; preds = %26
  %35 = sub i64 %31, 1
  %36 = inttoptr i64 %35 to ptr
  br label %42

37:                                               ; preds = %26
  %38 = ptrtoint ptr %28 to i64
  %39 = ptrtoint ptr %30 to i64
  %40 = add i64 %38, %39
  %41 = inttoptr i64 %40 to ptr
  br label %42

42:                                               ; preds = %34, %37
  %43 = phi ptr [ %41, %37 ], [ %36, %34 ]
  br label %44

44:                                               ; preds = %42
  %45 = load ptr, ptr %43, align 8
  %46 = extractvalue { ptr, ptr } %25, 1
  %47 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %45, 0
  %48 = insertvalue { ptr, ptr, ptr } %47, ptr %46, 1
  %49 = insertvalue { ptr, ptr, ptr } %48, ptr null, 2
  %50 = call ptr @GC_malloc(i64 24)
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr %Item, ptr %51, i32 0, i32 0
  store ptr @Item..vtbl, ptr %52, align 8
  invoke void @Item.constructor(ptr %51, double 1.000000e+03)
          to label %53 unwind label %57

53:                                               ; preds = %44
  %54 = extractvalue { ptr, ptr, ptr } %49, 1
  %55 = extractvalue { ptr, ptr, ptr } %49, 0
  invoke void %55(ptr %54, ptr %51)
          to label %56 unwind label %57

56:                                               ; preds = %53
  br label %196

57:                                               ; preds = %53, %44, %16, %0
  %58 = landingpad { ptr, i32 }
          catch ptr null
  %59 = extractvalue { ptr, i32 } %58, 0
  %60 = call ptr @__cxa_begin_catch(ptr %59)
  %61 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %57
  %62 = icmp eq ptr %61, @_ZTIi
  br i1 %62, label %63, label %70

63:                                               ; preds = %.noexc
  %64 = load i32, ptr %60, align 4
  %65 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %63
  %66 = getelementptr i8, ptr %65, i64 8
  %67 = getelementptr { i64, ptr, i32 }, ptr %66, i32 0, i32 0
  store i64 4, ptr %67, align 8
  %68 = getelementptr { i64, ptr, i32 }, ptr %66, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %68, align 8
  %69 = getelementptr { i64, ptr, i32 }, ptr %66, i32 0, i32 2
  store i32 %64, ptr %69, align 4
  br label %193

70:                                               ; preds = %.noexc
  %71 = icmp eq ptr %61, @_ZTIj
  br i1 %71, label %72, label %79

72:                                               ; preds = %70
  %73 = load i32, ptr %60, align 4
  %74 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %72
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = getelementptr { i64, ptr, i32 }, ptr %75, i32 0, i32 0
  store i64 4, ptr %76, align 8
  %77 = getelementptr { i64, ptr, i32 }, ptr %75, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %77, align 8
  %78 = getelementptr { i64, ptr, i32 }, ptr %75, i32 0, i32 2
  store i32 %73, ptr %78, align 4
  br label %190

79:                                               ; preds = %70
  %80 = icmp eq ptr %61, @_ZTId
  br i1 %80, label %81, label %88

81:                                               ; preds = %79
  %82 = load double, ptr %60, align 8
  %83 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %81
  %84 = getelementptr i8, ptr %83, i64 8
  %85 = getelementptr { i64, ptr, double }, ptr %84, i32 0, i32 0
  store i64 8, ptr %85, align 8
  %86 = getelementptr { i64, ptr, double }, ptr %84, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %86, align 8
  %87 = getelementptr { i64, ptr, double }, ptr %84, i32 0, i32 2
  store double %82, ptr %87, align 8
  br label %187

88:                                               ; preds = %79
  %89 = icmp eq ptr %61, @_ZTIf
  br i1 %89, label %90, label %97

90:                                               ; preds = %88
  %91 = load float, ptr %60, align 4
  %92 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %90
  %93 = getelementptr i8, ptr %92, i64 8
  %94 = getelementptr { i64, ptr, float }, ptr %93, i32 0, i32 0
  store i64 4, ptr %94, align 8
  %95 = getelementptr { i64, ptr, float }, ptr %93, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %95, align 8
  %96 = getelementptr { i64, ptr, float }, ptr %93, i32 0, i32 2
  store float %91, ptr %96, align 4
  br label %184

97:                                               ; preds = %88
  %98 = icmp eq ptr %61, @_ZTIb
  br i1 %98, label %99, label %106

99:                                               ; preds = %97
  %100 = load i1, ptr %60, align 1
  %101 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %99
  %102 = getelementptr i8, ptr %101, i64 8
  %103 = getelementptr { i64, ptr, i1 }, ptr %102, i32 0, i32 0
  store i64 1, ptr %103, align 8
  %104 = getelementptr { i64, ptr, i1 }, ptr %102, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %104, align 8
  %105 = getelementptr { i64, ptr, i1 }, ptr %102, i32 0, i32 2
  store i1 %100, ptr %105, align 1
  br label %181

106:                                              ; preds = %97
  %107 = icmp eq ptr %61, @_ZTIx
  br i1 %107, label %108, label %115

108:                                              ; preds = %106
  %109 = load i64, ptr %60, align 8
  %110 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %108
  %111 = getelementptr i8, ptr %110, i64 8
  %112 = getelementptr { i64, ptr, i64 }, ptr %111, i32 0, i32 0
  store i64 8, ptr %112, align 8
  %113 = getelementptr { i64, ptr, i64 }, ptr %111, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %113, align 8
  %114 = getelementptr { i64, ptr, i64 }, ptr %111, i32 0, i32 2
  store i64 %109, ptr %114, align 8
  br label %178

115:                                              ; preds = %106
  %116 = icmp eq ptr %61, @_ZTIPc
  br i1 %116, label %117, label %124

117:                                              ; preds = %115
  %118 = load ptr, ptr %60, align 8
  %119 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %117
  %120 = getelementptr i8, ptr %119, i64 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 0
  store i64 8, ptr %121, align 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %120, i32 0, i32 2
  store ptr %118, ptr %123, align 8
  br label %175

124:                                              ; preds = %115
  %125 = icmp eq ptr %61, @_ZTIPv
  br i1 %125, label %126, label %128

126:                                              ; preds = %124
  %127 = load ptr, ptr %60, align 8
  br label %172

128:                                              ; preds = %124
  %129 = load ptr, ptr %61, align 8
  %130 = icmp eq ptr %129, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %130, label %131, label %162

131:                                              ; preds = %128
  %132 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %61, i32 0, i32 2
  %133 = load i32, ptr %132, align 4
  %134 = and i32 %133, 1073741824
  %135 = icmp ne i32 %134, 0
  br i1 %135, label %136, label %152

136:                                              ; preds = %131
  %137 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %61, i32 0, i32 4
  %138 = load ptr, ptr %137, align 8
  %139 = icmp ne ptr %138, null
  br i1 %139, label %140, label %142

140:                                              ; preds = %136
  invoke void %138(ptr %1, ptr %60)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %140
  %141 = load ptr, ptr %1, align 8
  br label %149

142:                                              ; preds = %136
  %143 = load ptr, ptr %60, align 8
  %144 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %142
  %145 = getelementptr i8, ptr %144, i64 8
  %146 = getelementptr { i64, ptr, ptr }, ptr %145, i32 0, i32 0
  store i64 8, ptr %146, align 8
  %147 = getelementptr { i64, ptr, ptr }, ptr %145, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %147, align 8
  %148 = getelementptr { i64, ptr, ptr }, ptr %145, i32 0, i32 2
  store ptr %143, ptr %148, align 8
  br label %149

149:                                              ; preds = %.noexc3, %.noexc5
  %150 = phi ptr [ %145, %.noexc5 ], [ %141, %.noexc3 ]
  br label %151

151:                                              ; preds = %149
  br label %159

152:                                              ; preds = %131
  %153 = load ptr, ptr %60, align 8
  %154 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %152
  %155 = getelementptr i8, ptr %154, i64 8
  %156 = getelementptr { i64, ptr, ptr }, ptr %155, i32 0, i32 0
  store i64 8, ptr %156, align 8
  %157 = getelementptr { i64, ptr, ptr }, ptr %155, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %157, align 8
  %158 = getelementptr { i64, ptr, ptr }, ptr %155, i32 0, i32 2
  store ptr %153, ptr %158, align 8
  br label %159

159:                                              ; preds = %151, %.noexc9
  %160 = phi ptr [ %155, %.noexc9 ], [ %150, %151 ]
  br label %161

161:                                              ; preds = %159
  br label %169

162:                                              ; preds = %128
  %163 = load ptr, ptr %60, align 8
  %164 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %162
  %165 = getelementptr i8, ptr %164, i64 8
  %166 = getelementptr { i64, ptr, ptr }, ptr %165, i32 0, i32 0
  store i64 8, ptr %166, align 8
  %167 = getelementptr { i64, ptr, ptr }, ptr %165, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %167, align 8
  %168 = getelementptr { i64, ptr, ptr }, ptr %165, i32 0, i32 2
  store ptr %163, ptr %168, align 8
  br label %169

169:                                              ; preds = %161, %.noexc10
  %170 = phi ptr [ %165, %.noexc10 ], [ %160, %161 ]
  br label %171

171:                                              ; preds = %169
  br label %172

172:                                              ; preds = %126, %171
  %173 = phi ptr [ %170, %171 ], [ %127, %126 ]
  br label %174

174:                                              ; preds = %172
  br label %175

175:                                              ; preds = %.noexc4, %174
  %176 = phi ptr [ %173, %174 ], [ %120, %.noexc4 ]
  br label %177

177:                                              ; preds = %175
  br label %178

178:                                              ; preds = %.noexc2, %177
  %179 = phi ptr [ %176, %177 ], [ %111, %.noexc2 ]
  br label %180

180:                                              ; preds = %178
  br label %181

181:                                              ; preds = %.noexc11, %180
  %182 = phi ptr [ %179, %180 ], [ %102, %.noexc11 ]
  br label %183

183:                                              ; preds = %181
  br label %184

184:                                              ; preds = %.noexc1, %183
  %185 = phi ptr [ %182, %183 ], [ %93, %.noexc1 ]
  br label %186

186:                                              ; preds = %184
  br label %187

187:                                              ; preds = %.noexc8, %186
  %188 = phi ptr [ %185, %186 ], [ %84, %.noexc8 ]
  br label %189

189:                                              ; preds = %187
  br label %190

190:                                              ; preds = %.noexc6, %189
  %191 = phi ptr [ %188, %189 ], [ %75, %.noexc6 ]
  br label %192

192:                                              ; preds = %190
  br label %193

193:                                              ; preds = %.noexc7, %192
  %194 = phi ptr [ %191, %192 ], [ %66, %.noexc7 ]
  br label %195

195:                                              ; preds = %193
  store ptr %194, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %196

196:                                              ; preds = %56, %195
  %197 = load i1, ptr %3, align 1
  br i1 %197, label %198, label %200

198:                                              ; preds = %196
  ret void

199:                                              ; preds = %200
  unreachable

200:                                              ; preds = %196
  %201 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10917319126229082950, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12745439577603844570, i64 8), i32 51, ptr null)
  br label %199

endcatch.resume:                                  ; preds = %99, %162, %152, %81, %63, %72, %142, %117, %140, %108, %90, %57
  %202 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %202
}

; Function Attrs: mustprogress
define void @classObject() #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = call ptr @GC_malloc(i64 40)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Walker, ptr %5, i32 0, i32 0
  store ptr @Walker..vtbl, ptr %6, align 8
  invoke void @Walker.constructor(ptr %5, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
          to label %7 unwind label %45

7:                                                ; preds = %0
  %8 = call ptr @GC_malloc(i64 32)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr { i64, ptr, ptr }, ptr %9, i32 0, i32 0
  store i64 8, ptr %10, align 8
  %11 = getelementptr { i64, ptr, ptr }, ptr %9, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_7966825562501435538_class, i64 40), ptr %11, align 8
  %12 = getelementptr { i64, ptr, ptr }, ptr %9, i32 0, i32 2
  store ptr %5, ptr %12, align 8
  %13 = invoke { ptr, ptr } @"___unbox_interface<!ts.iface<Holder>>"(ptr %9)
          to label %14 unwind label %45

14:                                               ; preds = %7
  %15 = extractvalue { ptr, ptr } %13, 0
  %16 = extractvalue { ptr, ptr } %13, 1
  %17 = getelementptr ptr, ptr %15, i32 2
  %18 = load ptr, ptr %17, align 8
  %19 = ptrtoint ptr %18 to i64
  %20 = and i64 %19, 1
  %21 = icmp ne i64 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %14
  %23 = sub i64 %19, 1
  %24 = inttoptr i64 %23 to ptr
  br label %30

25:                                               ; preds = %14
  %26 = ptrtoint ptr %16 to i64
  %27 = ptrtoint ptr %18 to i64
  %28 = add i64 %26, %27
  %29 = inttoptr i64 %28 to ptr
  br label %30

30:                                               ; preds = %22, %25
  %31 = phi ptr [ %29, %25 ], [ %24, %22 ]
  br label %32

32:                                               ; preds = %30
  %33 = load ptr, ptr %31, align 8
  %34 = extractvalue { ptr, ptr } %13, 1
  %35 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %33, 0
  %36 = insertvalue { ptr, ptr, ptr } %35, ptr %34, 1
  %37 = insertvalue { ptr, ptr, ptr } %36, ptr null, 2
  %38 = call ptr @GC_malloc(i64 24)
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr %Item, ptr %39, i32 0, i32 0
  store ptr @Item..vtbl, ptr %40, align 8
  invoke void @Item.constructor(ptr %39, double 1.000000e+03)
          to label %41 unwind label %45

41:                                               ; preds = %32
  %42 = extractvalue { ptr, ptr, ptr } %37, 1
  %43 = extractvalue { ptr, ptr, ptr } %37, 0
  invoke void %43(ptr %42, ptr %39)
          to label %44 unwind label %45

44:                                               ; preds = %41
  br label %184

45:                                               ; preds = %41, %32, %7, %0
  %46 = landingpad { ptr, i32 }
          catch ptr null
  %47 = extractvalue { ptr, i32 } %46, 0
  %48 = call ptr @__cxa_begin_catch(ptr %47)
  %49 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %45
  %50 = icmp eq ptr %49, @_ZTIi
  br i1 %50, label %51, label %58

51:                                               ; preds = %.noexc5
  %52 = load i32, ptr %48, align 4
  %53 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %51
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, i32 }, ptr %54, i32 0, i32 0
  store i64 4, ptr %55, align 8
  %56 = getelementptr { i64, ptr, i32 }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, i32 }, ptr %54, i32 0, i32 2
  store i32 %52, ptr %57, align 4
  br label %181

58:                                               ; preds = %.noexc5
  %59 = icmp eq ptr %49, @_ZTIj
  br i1 %59, label %60, label %67

60:                                               ; preds = %58
  %61 = load i32, ptr %48, align 4
  %62 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %60
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = getelementptr { i64, ptr, i32 }, ptr %63, i32 0, i32 0
  store i64 4, ptr %64, align 8
  %65 = getelementptr { i64, ptr, i32 }, ptr %63, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %65, align 8
  %66 = getelementptr { i64, ptr, i32 }, ptr %63, i32 0, i32 2
  store i32 %61, ptr %66, align 4
  br label %178

67:                                               ; preds = %58
  %68 = icmp eq ptr %49, @_ZTId
  br i1 %68, label %69, label %76

69:                                               ; preds = %67
  %70 = load double, ptr %48, align 8
  %71 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %69
  %72 = getelementptr i8, ptr %71, i64 8
  %73 = getelementptr { i64, ptr, double }, ptr %72, i32 0, i32 0
  store i64 8, ptr %73, align 8
  %74 = getelementptr { i64, ptr, double }, ptr %72, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %74, align 8
  %75 = getelementptr { i64, ptr, double }, ptr %72, i32 0, i32 2
  store double %70, ptr %75, align 8
  br label %175

76:                                               ; preds = %67
  %77 = icmp eq ptr %49, @_ZTIf
  br i1 %77, label %78, label %85

78:                                               ; preds = %76
  %79 = load float, ptr %48, align 4
  %80 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %78
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr { i64, ptr, float }, ptr %81, i32 0, i32 0
  store i64 4, ptr %82, align 8
  %83 = getelementptr { i64, ptr, float }, ptr %81, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %83, align 8
  %84 = getelementptr { i64, ptr, float }, ptr %81, i32 0, i32 2
  store float %79, ptr %84, align 4
  br label %172

85:                                               ; preds = %76
  %86 = icmp eq ptr %49, @_ZTIb
  br i1 %86, label %87, label %94

87:                                               ; preds = %85
  %88 = load i1, ptr %48, align 1
  %89 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %87
  %90 = getelementptr i8, ptr %89, i64 8
  %91 = getelementptr { i64, ptr, i1 }, ptr %90, i32 0, i32 0
  store i64 1, ptr %91, align 8
  %92 = getelementptr { i64, ptr, i1 }, ptr %90, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %92, align 8
  %93 = getelementptr { i64, ptr, i1 }, ptr %90, i32 0, i32 2
  store i1 %88, ptr %93, align 1
  br label %169

94:                                               ; preds = %85
  %95 = icmp eq ptr %49, @_ZTIx
  br i1 %95, label %96, label %103

96:                                               ; preds = %94
  %97 = load i64, ptr %48, align 8
  %98 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %96
  %99 = getelementptr i8, ptr %98, i64 8
  %100 = getelementptr { i64, ptr, i64 }, ptr %99, i32 0, i32 0
  store i64 8, ptr %100, align 8
  %101 = getelementptr { i64, ptr, i64 }, ptr %99, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %101, align 8
  %102 = getelementptr { i64, ptr, i64 }, ptr %99, i32 0, i32 2
  store i64 %97, ptr %102, align 8
  br label %166

103:                                              ; preds = %94
  %104 = icmp eq ptr %49, @_ZTIPc
  br i1 %104, label %105, label %112

105:                                              ; preds = %103
  %106 = load ptr, ptr %48, align 8
  %107 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %105
  %108 = getelementptr i8, ptr %107, i64 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 0
  store i64 8, ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 2
  store ptr %106, ptr %111, align 8
  br label %163

112:                                              ; preds = %103
  %113 = icmp eq ptr %49, @_ZTIPv
  br i1 %113, label %114, label %116

114:                                              ; preds = %112
  %115 = load ptr, ptr %48, align 8
  br label %160

116:                                              ; preds = %112
  %117 = load ptr, ptr %49, align 8
  %118 = icmp eq ptr %117, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %118, label %119, label %150

119:                                              ; preds = %116
  %120 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %49, i32 0, i32 2
  %121 = load i32, ptr %120, align 4
  %122 = and i32 %121, 1073741824
  %123 = icmp ne i32 %122, 0
  br i1 %123, label %124, label %140

124:                                              ; preds = %119
  %125 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %49, i32 0, i32 4
  %126 = load ptr, ptr %125, align 8
  %127 = icmp ne ptr %126, null
  br i1 %127, label %128, label %130

128:                                              ; preds = %124
  invoke void %126(ptr %1, ptr %48)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %128
  %129 = load ptr, ptr %1, align 8
  br label %137

130:                                              ; preds = %124
  %131 = load ptr, ptr %48, align 8
  %132 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %130
  %133 = getelementptr i8, ptr %132, i64 8
  %134 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 0
  store i64 8, ptr %134, align 8
  %135 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %135, align 8
  %136 = getelementptr { i64, ptr, ptr }, ptr %133, i32 0, i32 2
  store ptr %131, ptr %136, align 8
  br label %137

137:                                              ; preds = %.noexc10, %.noexc3
  %138 = phi ptr [ %133, %.noexc3 ], [ %129, %.noexc10 ]
  br label %139

139:                                              ; preds = %137
  br label %147

140:                                              ; preds = %119
  %141 = load ptr, ptr %48, align 8
  %142 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %140
  %143 = getelementptr i8, ptr %142, i64 8
  %144 = getelementptr { i64, ptr, ptr }, ptr %143, i32 0, i32 0
  store i64 8, ptr %144, align 8
  %145 = getelementptr { i64, ptr, ptr }, ptr %143, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %145, align 8
  %146 = getelementptr { i64, ptr, ptr }, ptr %143, i32 0, i32 2
  store ptr %141, ptr %146, align 8
  br label %147

147:                                              ; preds = %139, %.noexc4
  %148 = phi ptr [ %143, %.noexc4 ], [ %138, %139 ]
  br label %149

149:                                              ; preds = %147
  br label %157

150:                                              ; preds = %116
  %151 = load ptr, ptr %48, align 8
  %152 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %150
  %153 = getelementptr i8, ptr %152, i64 8
  %154 = getelementptr { i64, ptr, ptr }, ptr %153, i32 0, i32 0
  store i64 8, ptr %154, align 8
  %155 = getelementptr { i64, ptr, ptr }, ptr %153, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %155, align 8
  %156 = getelementptr { i64, ptr, ptr }, ptr %153, i32 0, i32 2
  store ptr %151, ptr %156, align 8
  br label %157

157:                                              ; preds = %149, %.noexc
  %158 = phi ptr [ %153, %.noexc ], [ %148, %149 ]
  br label %159

159:                                              ; preds = %157
  br label %160

160:                                              ; preds = %114, %159
  %161 = phi ptr [ %158, %159 ], [ %115, %114 ]
  br label %162

162:                                              ; preds = %160
  br label %163

163:                                              ; preds = %.noexc6, %162
  %164 = phi ptr [ %161, %162 ], [ %108, %.noexc6 ]
  br label %165

165:                                              ; preds = %163
  br label %166

166:                                              ; preds = %.noexc2, %165
  %167 = phi ptr [ %164, %165 ], [ %99, %.noexc2 ]
  br label %168

168:                                              ; preds = %166
  br label %169

169:                                              ; preds = %.noexc7, %168
  %170 = phi ptr [ %167, %168 ], [ %90, %.noexc7 ]
  br label %171

171:                                              ; preds = %169
  br label %172

172:                                              ; preds = %.noexc8, %171
  %173 = phi ptr [ %170, %171 ], [ %81, %.noexc8 ]
  br label %174

174:                                              ; preds = %172
  br label %175

175:                                              ; preds = %.noexc11, %174
  %176 = phi ptr [ %173, %174 ], [ %72, %.noexc11 ]
  br label %177

177:                                              ; preds = %175
  br label %178

178:                                              ; preds = %.noexc1, %177
  %179 = phi ptr [ %176, %177 ], [ %63, %.noexc1 ]
  br label %180

180:                                              ; preds = %178
  br label %181

181:                                              ; preds = %.noexc9, %180
  %182 = phi ptr [ %179, %180 ], [ %54, %.noexc9 ]
  br label %183

183:                                              ; preds = %181
  store ptr %182, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %184

184:                                              ; preds = %44, %183
  %185 = load i1, ptr %3, align 1
  br i1 %185, label %186, label %188

186:                                              ; preds = %184
  ret void

187:                                              ; preds = %188
  unreachable

188:                                              ; preds = %184
  %189 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2179998208988451237, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12745439577603844570, i64 8), i32 63, ptr null)
  br label %187

endcatch.resume:                                  ; preds = %69, %128, %51, %78, %87, %105, %45, %140, %130, %96, %60, %150
  %190 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %190
}

; Function Attrs: mustprogress
define void @implementer() #2 {
  %1 = call ptr @GC_malloc(i64 40)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %Keeper, ptr %2, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %3, align 8
  call void @Keeper.constructor(ptr %2, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_12419312426083759605_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %2, ptr %8, align 8
  %9 = call { ptr, ptr } @"___unbox_interface<!ts.iface<Holder>>"(ptr %5)
  %10 = extractvalue { ptr, ptr } %9, 0
  %11 = extractvalue { ptr, ptr } %9, 1
  %12 = getelementptr ptr, ptr %10, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = ptrtoint ptr %13 to i64
  %15 = and i64 %14, 1
  %16 = icmp ne i64 %15, 0
  br i1 %16, label %17, label %20

17:                                               ; preds = %0
  %18 = sub i64 %14, 1
  %19 = inttoptr i64 %18 to ptr
  br label %25

20:                                               ; preds = %0
  %21 = ptrtoint ptr %11 to i64
  %22 = ptrtoint ptr %13 to i64
  %23 = add i64 %21, %22
  %24 = inttoptr i64 %23 to ptr
  br label %25

25:                                               ; preds = %17, %20
  %26 = phi ptr [ %24, %20 ], [ %19, %17 ]
  br label %27

27:                                               ; preds = %25
  %28 = load ptr, ptr %26, align 8
  %29 = extractvalue { ptr, ptr } %9, 1
  %30 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %28, 0
  %31 = insertvalue { ptr, ptr, ptr } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, ptr } %31, ptr null, 2
  %33 = call ptr @GC_malloc(i64 24)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr %Item, ptr %34, i32 0, i32 0
  store ptr @Item..vtbl, ptr %35, align 8
  call void @Item.constructor(ptr %34, double 1.000000e+00)
  %36 = extractvalue { ptr, ptr, ptr } %32, 1
  %37 = extractvalue { ptr, ptr, ptr } %32, 0
  call void %37(ptr %36, ptr %34)
  %38 = call ptr @GC_malloc(i64 40)
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr %Keeper, ptr %39, i32 0, i32 0
  store ptr @Keeper..vtbl, ptr %40, align 8
  call void @Keeper.constructor(ptr %39, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
  %41 = ptrtoint ptr %39 to i64
  %42 = icmp ne i64 %41, 0
  br i1 %42, label %43, label %48

43:                                               ; preds = %27
  %44 = getelementptr %Keeper, ptr %39, i32 0, i32 0
  %45 = load ptr, ptr %44, align 8
  %46 = getelementptr ptr, ptr %45, i32 1
  %47 = load ptr, ptr %46, align 8
  br label %49

48:                                               ; preds = %27
  br label %49

49:                                               ; preds = %43, %48
  %50 = phi ptr [ @Holder.null..vtbl, %48 ], [ %47, %43 ]
  %51 = insertvalue { ptr, ptr } undef, ptr %50, 0
  %52 = insertvalue { ptr, ptr } %51, ptr %39, 1
  %53 = call ptr @GC_malloc(i64 40)
  %54 = getelementptr i8, ptr %53, i64 8
  %55 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %54, i32 0, i32 0
  store i64 16, ptr %55, align 8
  %56 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %54, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9375918230080828099_interface, i64 40), ptr %56, align 8
  %57 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %54, i32 0, i32 2
  store { ptr, ptr } %52, ptr %57, align 8
  %58 = call { ptr, ptr } @"___unbox_interface<!ts.iface<Holder>>"(ptr %54)
  %59 = extractvalue { ptr, ptr } %58, 0
  %60 = extractvalue { ptr, ptr } %58, 1
  %61 = getelementptr ptr, ptr %59, i32 2
  %62 = load ptr, ptr %61, align 8
  %63 = ptrtoint ptr %62 to i64
  %64 = and i64 %63, 1
  %65 = icmp ne i64 %64, 0
  br i1 %65, label %66, label %69

66:                                               ; preds = %49
  %67 = sub i64 %63, 1
  %68 = inttoptr i64 %67 to ptr
  br label %74

69:                                               ; preds = %49
  %70 = ptrtoint ptr %60 to i64
  %71 = ptrtoint ptr %62 to i64
  %72 = add i64 %70, %71
  %73 = inttoptr i64 %72 to ptr
  br label %74

74:                                               ; preds = %66, %69
  %75 = phi ptr [ %73, %69 ], [ %68, %66 ]
  br label %76

76:                                               ; preds = %74
  %77 = load ptr, ptr %75, align 8
  %78 = extractvalue { ptr, ptr } %58, 1
  %79 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %77, 0
  %80 = insertvalue { ptr, ptr, ptr } %79, ptr %78, 1
  %81 = insertvalue { ptr, ptr, ptr } %80, ptr null, 2
  %82 = call ptr @GC_malloc(i64 24)
  %83 = getelementptr i8, ptr %82, i64 8
  %84 = getelementptr %Item, ptr %83, i32 0, i32 0
  store ptr @Item..vtbl, ptr %84, align 8
  call void @Item.constructor(ptr %83, double 2.000000e+00)
  %85 = extractvalue { ptr, ptr, ptr } %81, 1
  %86 = extractvalue { ptr, ptr, ptr } %81, 0
  call void %86(ptr %85, ptr %83)
  ret void
}

; Function Attrs: mustprogress
define void @castBeforeTheVTable() #2 {
  %1 = call ptr @GC_malloc(i64 40)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %Later, ptr %2, i32 0, i32 0
  store ptr @Later..vtbl, ptr %3, align 8
  call void @Later.constructor(ptr %2, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1841389987134824288_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %2, ptr %8, align 8
  %9 = call { ptr, ptr } @"___unbox_interface<!ts.iface<Holder>>"(ptr %5)
  %10 = extractvalue { ptr, ptr } %9, 0
  %11 = extractvalue { ptr, ptr } %9, 1
  %12 = getelementptr ptr, ptr %10, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = ptrtoint ptr %13 to i64
  %15 = and i64 %14, 1
  %16 = icmp ne i64 %15, 0
  br i1 %16, label %17, label %20

17:                                               ; preds = %0
  %18 = sub i64 %14, 1
  %19 = inttoptr i64 %18 to ptr
  br label %25

20:                                               ; preds = %0
  %21 = ptrtoint ptr %11 to i64
  %22 = ptrtoint ptr %13 to i64
  %23 = add i64 %21, %22
  %24 = inttoptr i64 %23 to ptr
  br label %25

25:                                               ; preds = %17, %20
  %26 = phi ptr [ %24, %20 ], [ %19, %17 ]
  br label %27

27:                                               ; preds = %25
  %28 = load ptr, ptr %26, align 8
  %29 = extractvalue { ptr, ptr } %9, 1
  %30 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %28, 0
  %31 = insertvalue { ptr, ptr, ptr } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, ptr } %31, ptr null, 2
  %33 = call ptr @GC_malloc(i64 24)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr %Item, ptr %34, i32 0, i32 0
  store ptr @Item..vtbl, ptr %35, align 8
  call void @Item.constructor(ptr %34, double 4.000000e+00)
  %36 = extractvalue { ptr, ptr, ptr } %32, 1
  %37 = extractvalue { ptr, ptr, ptr } %32, 0
  call void %37(ptr %36, ptr %34)
  ret void
}

; Function Attrs: mustprogress
define linkonce_odr { ptr, ptr } @"___unbox_interface<!ts.iface<Runner>>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, ptr }, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr { i64, ptr, i8 }, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp ne i64 %8, 0
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %1
  %14 = call i32 @strcmp(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %15 = icmp eq i32 %14, 0
  br label %18

16:                                               ; preds = %1
  %17 = icmp eq i64 %8, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %18

18:                                               ; preds = %13, %16
  %19 = phi i1 [ %17, %16 ], [ %15, %13 ]
  br label %20

20:                                               ; preds = %18
  br i1 %19, label %21, label %83

21:                                               ; preds = %20
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 2
  %24 = getelementptr { i64, ptr, ptr }, ptr %22, i32 0, i32 0
  %25 = load i64, ptr %24, align 8
  %26 = load i64, ptr %23, align 8
  %27 = icmp uge i64 %25, 8
  %28 = select i1 %27, i64 0, i64 %25
  %29 = mul i64 %28, 8
  %30 = shl i64 1, %29
  %31 = sub i64 %30, 1
  %32 = and i64 %26, %31
  %33 = select i1 %27, i64 %26, i64 %32
  %34 = inttoptr i64 %33 to ptr
  %35 = getelementptr ptr, ptr %34, i32 0
  %36 = load ptr, ptr %35, align 8
  %37 = getelementptr ptr, ptr %36, i32 0
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr @Walker..rtti, align 8
  %40 = call i1 %38(ptr %34, ptr %39)
  br i1 %40, label %41, label %54

41:                                               ; preds = %21
  %42 = ptrtoint ptr %34 to i64
  %43 = icmp ne i64 %42, 0
  br i1 %43, label %44, label %49

44:                                               ; preds = %41
  %45 = getelementptr %Walker, ptr %34, i32 0, i32 0
  %46 = load ptr, ptr %45, align 8
  %47 = getelementptr ptr, ptr %46, i32 1
  %48 = load ptr, ptr %47, align 8
  br label %50

49:                                               ; preds = %41
  br label %50

50:                                               ; preds = %44, %49
  %51 = phi ptr [ @Runner.null..vtbl, %49 ], [ %48, %44 ]
  %52 = insertvalue { ptr, ptr } undef, ptr %51, 0
  %53 = insertvalue { ptr, ptr } %52, ptr %34, 1
  store { ptr, ptr } %53, ptr %3, align 8
  br label %313

54:                                               ; preds = %21
  %55 = getelementptr ptr, ptr %34, i32 0
  %56 = load ptr, ptr %55, align 8
  %57 = getelementptr ptr, ptr %56, i32 0
  %58 = load ptr, ptr %57, align 8
  %59 = load ptr, ptr @Later..rtti, align 8
  %60 = call i1 %58(ptr %34, ptr %59)
  br i1 %60, label %61, label %63

61:                                               ; preds = %54
  %62 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_7324454017586831841, i64 8), ptr %62, align 8
  call void @__cxa_throw(ptr %62, ptr @_ZTIPc, ptr null)
  br label %315

63:                                               ; preds = %54
  %64 = getelementptr ptr, ptr %34, i32 0
  %65 = load ptr, ptr %64, align 8
  %66 = getelementptr ptr, ptr %65, i32 0
  %67 = load ptr, ptr %66, align 8
  %68 = load ptr, ptr @Keeper..rtti, align 8
  %69 = call i1 %67(ptr %34, ptr %68)
  br i1 %69, label %70, label %72

70:                                               ; preds = %63
  %71 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10753497107086709977, i64 8), ptr %71, align 8
  call void @__cxa_throw(ptr %71, ptr @_ZTIPc, ptr null)
  br label %315

72:                                               ; preds = %63
  %73 = getelementptr ptr, ptr %34, i32 0
  %74 = load ptr, ptr %73, align 8
  %75 = getelementptr ptr, ptr %74, i32 0
  %76 = load ptr, ptr %75, align 8
  %77 = load ptr, ptr @Item..rtti, align 8
  %78 = call i1 %76(ptr %34, ptr %77)
  br i1 %78, label %79, label %81

79:                                               ; preds = %72
  %80 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_6007013098191354486, i64 8), ptr %80, align 8
  call void @__cxa_throw(ptr %80, ptr @_ZTIPc, ptr null)
  br label %315

81:                                               ; preds = %72
  %82 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_1157645605329739794, i64 8), ptr %82, align 8
  call void @__cxa_throw(ptr %82, ptr @_ZTIPc, ptr null)
  br label %315

83:                                               ; preds = %20
  %84 = load ptr, ptr %2, align 8
  %85 = getelementptr { i64, ptr, i8 }, ptr %84, i32 0, i32 1
  %86 = load ptr, ptr %85, align 8
  %87 = ptrtoint ptr %86 to i64
  %88 = icmp ne i64 %87, 0
  %89 = zext i1 %88 to i32
  %90 = and i32 %89, 1
  %91 = icmp ne i32 %90, 0
  br i1 %91, label %92, label %95

92:                                               ; preds = %83
  %93 = call i32 @strcmp(ptr %86, ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8))
  %94 = icmp eq i32 %93, 0
  br label %97

95:                                               ; preds = %83
  %96 = icmp eq i64 %87, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8) to i64)
  br label %97

97:                                               ; preds = %92, %95
  %98 = phi i1 [ %96, %95 ], [ %94, %92 ]
  br label %99

99:                                               ; preds = %97
  br i1 %98, label %100, label %234

100:                                              ; preds = %99
  %101 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %4, i32 0, i32 2
  %102 = load { ptr, ptr }, ptr %101, align 8
  %103 = extractvalue { ptr, ptr } %102, 1
  %104 = ptrtoint ptr %103 to i64
  %105 = icmp ne i64 %104, 0
  br i1 %105, label %106, label %113

106:                                              ; preds = %100
  %107 = extractvalue { ptr, ptr } %102, 0
  %108 = extractvalue { ptr, ptr } %102, 1
  %109 = getelementptr ptr, ptr %107, i32 1
  %110 = load ptr, ptr %109, align 8
  %111 = load ptr, ptr @Walker..rtti, align 8
  %112 = call i1 %110(ptr %108, ptr %111)
  br label %113

113:                                              ; preds = %106, %100
  %114 = phi i1 [ %112, %106 ], [ false, %100 ]
  br i1 %114, label %115, label %144

115:                                              ; preds = %113
  %116 = extractvalue { ptr, ptr } %102, 1
  %117 = ptrtoint ptr %116 to i64
  %118 = icmp ne i64 %117, 0
  br i1 %118, label %119, label %126

119:                                              ; preds = %115
  %120 = extractvalue { ptr, ptr } %102, 0
  %121 = extractvalue { ptr, ptr } %102, 1
  %122 = getelementptr ptr, ptr %120, i32 1
  %123 = load ptr, ptr %122, align 8
  %124 = load ptr, ptr @Walker..rtti, align 8
  %125 = call i1 %123(ptr %121, ptr %124)
  br label %126

126:                                              ; preds = %119, %115
  %127 = phi i1 [ %125, %119 ], [ false, %115 ]
  br i1 %127, label %128, label %132

128:                                              ; preds = %126
  %129 = extractvalue { ptr, ptr } %102, 1
  %130 = ptrtoint ptr %129 to i64
  %131 = icmp ne i64 %130, 0
  br i1 %131, label %134, label %139

132:                                              ; preds = %126
  %133 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %133, align 8
  call void @__cxa_throw(ptr %133, ptr @_ZTIPc, ptr null)
  br label %315

134:                                              ; preds = %128
  %135 = getelementptr %Walker, ptr %129, i32 0, i32 0
  %136 = load ptr, ptr %135, align 8
  %137 = getelementptr ptr, ptr %136, i32 1
  %138 = load ptr, ptr %137, align 8
  br label %140

139:                                              ; preds = %128
  br label %140

140:                                              ; preds = %134, %139
  %141 = phi ptr [ @Runner.null..vtbl, %139 ], [ %138, %134 ]
  %142 = insertvalue { ptr, ptr } undef, ptr %141, 0
  %143 = insertvalue { ptr, ptr } %142, ptr %129, 1
  store { ptr, ptr } %143, ptr %3, align 8
  br label %313

144:                                              ; preds = %113
  %145 = extractvalue { ptr, ptr } %102, 1
  %146 = ptrtoint ptr %145 to i64
  %147 = icmp ne i64 %146, 0
  br i1 %147, label %148, label %155

148:                                              ; preds = %144
  %149 = extractvalue { ptr, ptr } %102, 0
  %150 = extractvalue { ptr, ptr } %102, 1
  %151 = getelementptr ptr, ptr %149, i32 1
  %152 = load ptr, ptr %151, align 8
  %153 = load ptr, ptr @Later..rtti, align 8
  %154 = call i1 %152(ptr %150, ptr %153)
  br label %155

155:                                              ; preds = %148, %144
  %156 = phi i1 [ %154, %148 ], [ false, %144 ]
  br i1 %156, label %157, label %174

157:                                              ; preds = %155
  %158 = extractvalue { ptr, ptr } %102, 1
  %159 = ptrtoint ptr %158 to i64
  %160 = icmp ne i64 %159, 0
  br i1 %160, label %161, label %168

161:                                              ; preds = %157
  %162 = extractvalue { ptr, ptr } %102, 0
  %163 = extractvalue { ptr, ptr } %102, 1
  %164 = getelementptr ptr, ptr %162, i32 1
  %165 = load ptr, ptr %164, align 8
  %166 = load ptr, ptr @Later..rtti, align 8
  %167 = call i1 %165(ptr %163, ptr %166)
  br label %168

168:                                              ; preds = %161, %157
  %169 = phi i1 [ %167, %161 ], [ false, %157 ]
  br i1 %169, label %170, label %172

170:                                              ; preds = %168
  %171 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_7324454017586831841, i64 8), ptr %171, align 8
  call void @__cxa_throw(ptr %171, ptr @_ZTIPc, ptr null)
  br label %315

172:                                              ; preds = %168
  %173 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %173, align 8
  call void @__cxa_throw(ptr %173, ptr @_ZTIPc, ptr null)
  br label %315

174:                                              ; preds = %155
  %175 = extractvalue { ptr, ptr } %102, 1
  %176 = ptrtoint ptr %175 to i64
  %177 = icmp ne i64 %176, 0
  br i1 %177, label %178, label %185

178:                                              ; preds = %174
  %179 = extractvalue { ptr, ptr } %102, 0
  %180 = extractvalue { ptr, ptr } %102, 1
  %181 = getelementptr ptr, ptr %179, i32 1
  %182 = load ptr, ptr %181, align 8
  %183 = load ptr, ptr @Keeper..rtti, align 8
  %184 = call i1 %182(ptr %180, ptr %183)
  br label %185

185:                                              ; preds = %178, %174
  %186 = phi i1 [ %184, %178 ], [ false, %174 ]
  br i1 %186, label %187, label %204

187:                                              ; preds = %185
  %188 = extractvalue { ptr, ptr } %102, 1
  %189 = ptrtoint ptr %188 to i64
  %190 = icmp ne i64 %189, 0
  br i1 %190, label %191, label %198

191:                                              ; preds = %187
  %192 = extractvalue { ptr, ptr } %102, 0
  %193 = extractvalue { ptr, ptr } %102, 1
  %194 = getelementptr ptr, ptr %192, i32 1
  %195 = load ptr, ptr %194, align 8
  %196 = load ptr, ptr @Keeper..rtti, align 8
  %197 = call i1 %195(ptr %193, ptr %196)
  br label %198

198:                                              ; preds = %191, %187
  %199 = phi i1 [ %197, %191 ], [ false, %187 ]
  br i1 %199, label %200, label %202

200:                                              ; preds = %198
  %201 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10753497107086709977, i64 8), ptr %201, align 8
  call void @__cxa_throw(ptr %201, ptr @_ZTIPc, ptr null)
  br label %315

202:                                              ; preds = %198
  %203 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %203, align 8
  call void @__cxa_throw(ptr %203, ptr @_ZTIPc, ptr null)
  br label %315

204:                                              ; preds = %185
  %205 = extractvalue { ptr, ptr } %102, 1
  %206 = ptrtoint ptr %205 to i64
  %207 = icmp ne i64 %206, 0
  br i1 %207, label %208, label %215

208:                                              ; preds = %204
  %209 = extractvalue { ptr, ptr } %102, 0
  %210 = extractvalue { ptr, ptr } %102, 1
  %211 = getelementptr ptr, ptr %209, i32 1
  %212 = load ptr, ptr %211, align 8
  %213 = load ptr, ptr @Item..rtti, align 8
  %214 = call i1 %212(ptr %210, ptr %213)
  br label %215

215:                                              ; preds = %208, %204
  %216 = phi i1 [ %214, %208 ], [ false, %204 ]
  br i1 %216, label %217, label %234

217:                                              ; preds = %215
  %218 = extractvalue { ptr, ptr } %102, 1
  %219 = ptrtoint ptr %218 to i64
  %220 = icmp ne i64 %219, 0
  br i1 %220, label %221, label %228

221:                                              ; preds = %217
  %222 = extractvalue { ptr, ptr } %102, 0
  %223 = extractvalue { ptr, ptr } %102, 1
  %224 = getelementptr ptr, ptr %222, i32 1
  %225 = load ptr, ptr %224, align 8
  %226 = load ptr, ptr @Item..rtti, align 8
  %227 = call i1 %225(ptr %223, ptr %226)
  br label %228

228:                                              ; preds = %221, %217
  %229 = phi i1 [ %227, %221 ], [ false, %217 ]
  br i1 %229, label %230, label %232

230:                                              ; preds = %228
  %231 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_6007013098191354486, i64 8), ptr %231, align 8
  call void @__cxa_throw(ptr %231, ptr @_ZTIPc, ptr null)
  br label %315

232:                                              ; preds = %228
  %233 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %233, align 8
  call void @__cxa_throw(ptr %233, ptr @_ZTIPc, ptr null)
  br label %315

234:                                              ; preds = %215, %99
  %235 = load ptr, ptr %2, align 8
  %236 = getelementptr { i64, ptr, i8 }, ptr %235, i32 0, i32 1
  %237 = load ptr, ptr %236, align 8
  %238 = ptrtoint ptr %237 to i64
  %239 = icmp ne i64 %238, 0
  %240 = zext i1 %239 to i32
  %241 = and i32 %240, 1
  %242 = icmp ne i32 %241, 0
  br i1 %242, label %243, label %246

243:                                              ; preds = %234
  %244 = call i32 @strcmp(ptr %237, ptr getelementptr inbounds nuw (i8, ptr @s_10231808476453998586, i64 8))
  %245 = icmp eq i32 %244, 0
  br label %248

246:                                              ; preds = %234
  %247 = icmp eq i64 %238, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10231808476453998586, i64 8) to i64)
  br label %248

248:                                              ; preds = %243, %246
  %249 = phi i1 [ %247, %246 ], [ %245, %243 ]
  br label %250

250:                                              ; preds = %248
  br i1 %249, label %268, label %251

251:                                              ; preds = %250
  %252 = load ptr, ptr %2, align 8
  %253 = getelementptr { i64, ptr, i8 }, ptr %252, i32 0, i32 1
  %254 = load ptr, ptr %253, align 8
  %255 = ptrtoint ptr %254 to i64
  %256 = icmp ne i64 %255, 0
  %257 = zext i1 %256 to i32
  %258 = and i32 %257, 1
  %259 = icmp ne i32 %258, 0
  br i1 %259, label %260, label %263

260:                                              ; preds = %251
  %261 = call i32 @strcmp(ptr %254, ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8))
  %262 = icmp eq i32 %261, 0
  br label %265

263:                                              ; preds = %251
  %264 = icmp eq i64 %255, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8) to i64)
  br label %265

265:                                              ; preds = %260, %263
  %266 = phi i1 [ %264, %263 ], [ %262, %260 ]
  br label %267

267:                                              ; preds = %265
  br label %268

268:                                              ; preds = %267, %250
  %269 = phi i1 [ %266, %267 ], [ %249, %250 ]
  br i1 %269, label %270, label %272

270:                                              ; preds = %268
  %271 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_14806728300850859021, i64 8), ptr %271, align 8
  call void @__cxa_throw(ptr %271, ptr @_ZTIPc, ptr null)
  br label %315

272:                                              ; preds = %268
  %273 = load ptr, ptr %2, align 8
  %274 = getelementptr { i64, ptr, i8 }, ptr %273, i32 0, i32 1
  %275 = load ptr, ptr %274, align 8
  %276 = ptrtoint ptr %275 to i64
  %277 = icmp ne i64 %276, 0
  %278 = zext i1 %277 to i32
  %279 = and i32 %278, 1
  %280 = icmp ne i32 %279, 0
  br i1 %280, label %281, label %284

281:                                              ; preds = %272
  %282 = call i32 @strcmp(ptr %275, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8))
  %283 = icmp eq i32 %282, 0
  br label %286

284:                                              ; preds = %272
  %285 = icmp eq i64 %276, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8) to i64)
  br label %286

286:                                              ; preds = %281, %284
  %287 = phi i1 [ %285, %284 ], [ %283, %281 ]
  br label %288

288:                                              ; preds = %286
  br i1 %287, label %306, label %289

289:                                              ; preds = %288
  %290 = load ptr, ptr %2, align 8
  %291 = getelementptr { i64, ptr, i8 }, ptr %290, i32 0, i32 1
  %292 = load ptr, ptr %291, align 8
  %293 = ptrtoint ptr %292 to i64
  %294 = icmp ne i64 %293, 0
  %295 = zext i1 %294 to i32
  %296 = and i32 %295, 1
  %297 = icmp ne i32 %296, 0
  br i1 %297, label %298, label %301

298:                                              ; preds = %289
  %299 = call i32 @strcmp(ptr %292, ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8))
  %300 = icmp eq i32 %299, 0
  br label %303

301:                                              ; preds = %289
  %302 = icmp eq i64 %293, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8) to i64)
  br label %303

303:                                              ; preds = %298, %301
  %304 = phi i1 [ %302, %301 ], [ %300, %298 ]
  br label %305

305:                                              ; preds = %303
  br label %306

306:                                              ; preds = %305, %288
  %307 = phi i1 [ %304, %305 ], [ %287, %288 ]
  br i1 %307, label %308, label %309

308:                                              ; preds = %306
  store { ptr, ptr } zeroinitializer, ptr %3, align 8
  br label %313

309:                                              ; preds = %306
  %310 = load ptr, ptr %2, align 8
  %311 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %310, i32 0, i32 2
  %312 = load { ptr, ptr }, ptr %311, align 8
  store { ptr, ptr } %312, ptr %3, align 8
  br label %313

313:                                              ; preds = %50, %140, %308, %309
  %314 = load { ptr, ptr }, ptr %3, align 8
  ret { ptr, ptr } %314

315:                                              ; preds = %61, %70, %79, %81, %132, %170, %172, %200, %202, %230, %232, %270
  unreachable
}

; Function Attrs: mustprogress
define void @roundTrip() #2 {
  %1 = call ptr @GC_malloc(i64 40)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %Walker, ptr %2, i32 0, i32 0
  store ptr @Walker..vtbl, ptr %3, align 8
  call void @Walker.constructor(ptr %2, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
  %4 = ptrtoint ptr %2 to i64
  %5 = icmp ne i64 %4, 0
  br i1 %5, label %6, label %11

6:                                                ; preds = %0
  %7 = getelementptr %Walker, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr ptr, ptr %8, i32 1
  %10 = load ptr, ptr %9, align 8
  br label %12

11:                                               ; preds = %0
  br label %12

12:                                               ; preds = %6, %11
  %13 = phi ptr [ @Runner.null..vtbl, %11 ], [ %10, %6 ]
  %14 = insertvalue { ptr, ptr } undef, ptr %13, 0
  %15 = insertvalue { ptr, ptr } %14, ptr %2, 1
  %16 = call ptr @GC_malloc(i64 40)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %17, i32 0, i32 0
  store i64 16, ptr %18, align 8
  %19 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %17, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_14063698103718993911_interface, i64 40), ptr %19, align 8
  %20 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %17, i32 0, i32 2
  store { ptr, ptr } %15, ptr %20, align 8
  %21 = call { ptr, ptr } @"___unbox_interface<!ts.iface<Runner>>"(ptr %17)
  %22 = extractvalue { ptr, ptr } %21, 0
  %23 = extractvalue { ptr, ptr } %21, 1
  %24 = getelementptr ptr, ptr %22, i32 2
  %25 = load ptr, ptr %24, align 8
  %26 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %25, 0
  %27 = insertvalue { ptr, ptr, ptr } %26, ptr %23, 1
  %28 = insertvalue { ptr, ptr, ptr } %27, ptr null, 2
  %29 = extractvalue { ptr, ptr, ptr } %28, 1
  %30 = extractvalue { ptr, ptr, ptr } %28, 0
  %31 = call ptr @GC_malloc(i64 8)
  %32 = getelementptr i8, ptr %31, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %32, ptr @a_0_3230914496259700488, i64 0, i1 false)
  %33 = call ptr @GC_malloc(i64 32)
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = getelementptr { ptr, i64, i64 }, ptr %34, i32 0, i32 0
  store ptr %32, ptr %35, align 8
  %36 = getelementptr { ptr, i64, i64 }, ptr %34, i32 0, i32 1
  store i64 0, ptr %36, align 8
  %37 = getelementptr { ptr, i64, i64 }, ptr %34, i32 0, i32 2
  store i64 0, ptr %37, align 8
  %38 = call double %30(ptr %29, ptr %34)
  %39 = fcmp oeq double %38, 1.000000e+00
  br i1 %39, label %40, label %42

40:                                               ; preds = %12
  ret void

41:                                               ; preds = %42
  unreachable

42:                                               ; preds = %12
  %43 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13538094400678585102, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12745439577603844570, i64 8), i32 83, ptr null)
  br label %41
}

; Function Attrs: mustprogress
define void @sawLater() #2 {
  %1 = call ptr @GC_malloc(i64 40)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %Later, ptr %2, i32 0, i32 0
  store ptr @Later..vtbl, ptr %3, align 8
  call void @Later.constructor(ptr %2, { ptr, ptr, ptr } { ptr @show, ptr null, ptr null })
  %4 = ptrtoint ptr %2 to i64
  %5 = icmp ne i64 %4, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %0
  br label %8

7:                                                ; preds = %0
  br label %8

8:                                                ; preds = %6, %7
  %9 = phi ptr [ @Holder.null..vtbl, %7 ], [ @Later.Holder..vtbl, %6 ]
  %10 = insertvalue { ptr, ptr } undef, ptr %9, 0
  %11 = insertvalue { ptr, ptr } %10, ptr %2, 1
  %12 = extractvalue { ptr, ptr } %11, 0
  %13 = extractvalue { ptr, ptr } %11, 1
  %14 = getelementptr ptr, ptr %12, i32 2
  %15 = load ptr, ptr %14, align 8
  %16 = ptrtoint ptr %15 to i64
  %17 = and i64 %16, 1
  %18 = icmp ne i64 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %8
  %20 = sub i64 %16, 1
  %21 = inttoptr i64 %20 to ptr
  br label %27

22:                                               ; preds = %8
  %23 = ptrtoint ptr %13 to i64
  %24 = ptrtoint ptr %15 to i64
  %25 = add i64 %23, %24
  %26 = inttoptr i64 %25 to ptr
  br label %27

27:                                               ; preds = %19, %22
  %28 = phi ptr [ %26, %22 ], [ %21, %19 ]
  br label %29

29:                                               ; preds = %27
  %30 = load ptr, ptr %28, align 8
  %31 = extractvalue { ptr, ptr } %11, 1
  %32 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %30, 0
  %33 = insertvalue { ptr, ptr, ptr } %32, ptr %31, 1
  %34 = insertvalue { ptr, ptr, ptr } %33, ptr null, 2
  %35 = call ptr @GC_malloc(i64 24)
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr %Item, ptr %36, i32 0, i32 0
  store ptr @Item..vtbl, ptr %37, align 8
  call void @Item.constructor(ptr %36, double 8.000000e+00)
  %38 = extractvalue { ptr, ptr, ptr } %34, 1
  %39 = extractvalue { ptr, ptr, ptr } %34, 0
  call void %39(ptr %38, ptr %36)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @otherInterface()
  call void @classObject()
  call void @implementer()
  call void @castBeforeTheVTable()
  call void @roundTrip()
  call void @sawLater()
  %1 = load double, ptr @shown, align 8
  %2 = fcmp oeq double %1, 1.500000e+01
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

4:                                                ; preds = %5
  unreachable

5:                                                ; preds = %0
  %6 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3166235262791310685, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12745439577603844570, i64 8), i32 98, ptr null)
  br label %4
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

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "__null_interface_vtable" }
attributes #4 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

