; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$"___unbox_interface<!ts.iface<Adder>>" = comdat any

@s_2495249626774661283 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFundefined\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_4345959843307353387 = internal constant [83 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: an object literal has no vtable for interface 'Adder'\00", align 8
@s_10458455444534216913 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtuple\00", align 8
@s_10231808476453998586 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFobject\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_11809972404538290522 = internal constant [82 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any: the class object has no vtable for interface 'Adder'\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_11039441214965475813 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa literal of fields only\00", align 8
@f_10417848714019435974 = internal constant [116 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00any_literal_to_interface.ts\00", align 8
@m_13436669823751029251 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa literal with a method\00", align 8
@m_8835751322448772705 = internal constant [39 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan interface value round-trips\00", align 8
@m_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@Adder.3680198976672767672..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_3680198976672767672_object, i64 40), ptr @.instanceOf.none, ptr @.f_main..f_main..objL33C22FH2872653265.add }
@td_3680198976672767672_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrel_3680198976672767672, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_14594035851153911124_tuple = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 6, i32 -739280342, ptr null, ptr null, ptr null, i64 -1 }, [6 x i8] c"tuple\00" }, align 8
@td_10237190548065851850_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrel_10237190548065851850, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_3693193992939591961_interface = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [10 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 9, i32 -16547381, ptr @tsrel_3693193992939591961, ptr @tsret_3693193992939591961, ptr null, i64 -1 }, [10 x i8] c"interface\00" }, align 8
@td_14527407094101050897_null = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [5 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 13, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [5 x i8] c"null\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

define internal void @tsret_3693193992939591961(ptr %0) {
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

define internal void @tsrel_3693193992939591961(ptr %0) {
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

define internal void @tsrel_10237190548065851850(ptr %0) {
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

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

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

define internal void @tsrel_3680198976672767672(ptr %0) {
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

; Function Attrs: mustprogress
define linkonce_odr { ptr, ptr } @"___unbox_interface<!ts.iface<Adder>>"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, ptr }, align 8
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
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15066323702654938015, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %22

20:                                               ; preds = %19
  %21 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_11809972404538290522, i64 8), ptr %21, align 8
  call void @__cxa_throw(ptr %21, ptr @_ZTIPc, ptr null)
  br label %119

22:                                               ; preds = %19
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr { i64, ptr, i8 }, ptr %23, i32 0, i32 1
  %25 = load ptr, ptr %24, align 8
  %26 = ptrtoint ptr %25 to i64
  %27 = icmp ne i64 %26, 0
  %28 = zext i1 %27 to i32
  %29 = and i32 %28, 1
  %30 = icmp ne i32 %29, 0
  br i1 %30, label %31, label %34

31:                                               ; preds = %22
  %32 = call i32 @strcmp(ptr %25, ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8))
  %33 = icmp eq i32 %32, 0
  br label %36

34:                                               ; preds = %22
  %35 = icmp eq i64 %26, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8) to i64)
  br label %36

36:                                               ; preds = %31, %34
  %37 = phi i1 [ %35, %34 ], [ %33, %31 ]
  br label %38

38:                                               ; preds = %36
  %39 = load ptr, ptr %2, align 8
  %40 = getelementptr { i64, ptr, i8 }, ptr %39, i32 0, i32 1
  %41 = load ptr, ptr %40, align 8
  %42 = ptrtoint ptr %41 to i64
  %43 = icmp ne i64 %42, 0
  %44 = zext i1 %43 to i32
  %45 = and i32 %44, 1
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %50

47:                                               ; preds = %38
  %48 = call i32 @strcmp(ptr %41, ptr getelementptr inbounds nuw (i8, ptr @s_10231808476453998586, i64 8))
  %49 = icmp eq i32 %48, 0
  br label %52

50:                                               ; preds = %38
  %51 = icmp eq i64 %42, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10231808476453998586, i64 8) to i64)
  br label %52

52:                                               ; preds = %47, %50
  %53 = phi i1 [ %51, %50 ], [ %49, %47 ]
  br label %54

54:                                               ; preds = %52
  br i1 %53, label %72, label %55

55:                                               ; preds = %54
  %56 = load ptr, ptr %2, align 8
  %57 = getelementptr { i64, ptr, i8 }, ptr %56, i32 0, i32 1
  %58 = load ptr, ptr %57, align 8
  %59 = ptrtoint ptr %58 to i64
  %60 = icmp ne i64 %59, 0
  %61 = zext i1 %60 to i32
  %62 = and i32 %61, 1
  %63 = icmp ne i32 %62, 0
  br i1 %63, label %64, label %67

64:                                               ; preds = %55
  %65 = call i32 @strcmp(ptr %58, ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8))
  %66 = icmp eq i32 %65, 0
  br label %69

67:                                               ; preds = %55
  %68 = icmp eq i64 %59, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_10458455444534216913, i64 8) to i64)
  br label %69

69:                                               ; preds = %64, %67
  %70 = phi i1 [ %68, %67 ], [ %66, %64 ]
  br label %71

71:                                               ; preds = %69
  br label %72

72:                                               ; preds = %71, %54
  %73 = phi i1 [ %70, %71 ], [ %53, %54 ]
  br i1 %73, label %74, label %76

74:                                               ; preds = %72
  %75 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4345959843307353387, i64 8), ptr %75, align 8
  call void @__cxa_throw(ptr %75, ptr @_ZTIPc, ptr null)
  br label %119

76:                                               ; preds = %72
  %77 = load ptr, ptr %2, align 8
  %78 = getelementptr { i64, ptr, i8 }, ptr %77, i32 0, i32 1
  %79 = load ptr, ptr %78, align 8
  %80 = ptrtoint ptr %79 to i64
  %81 = icmp ne i64 %80, 0
  %82 = zext i1 %81 to i32
  %83 = and i32 %82, 1
  %84 = icmp ne i32 %83, 0
  br i1 %84, label %85, label %88

85:                                               ; preds = %76
  %86 = call i32 @strcmp(ptr %79, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8))
  %87 = icmp eq i32 %86, 0
  br label %90

88:                                               ; preds = %76
  %89 = icmp eq i64 %80, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8) to i64)
  br label %90

90:                                               ; preds = %85, %88
  %91 = phi i1 [ %89, %88 ], [ %87, %85 ]
  br label %92

92:                                               ; preds = %90
  br i1 %91, label %110, label %93

93:                                               ; preds = %92
  %94 = load ptr, ptr %2, align 8
  %95 = getelementptr { i64, ptr, i8 }, ptr %94, i32 0, i32 1
  %96 = load ptr, ptr %95, align 8
  %97 = ptrtoint ptr %96 to i64
  %98 = icmp ne i64 %97, 0
  %99 = zext i1 %98 to i32
  %100 = and i32 %99, 1
  %101 = icmp ne i32 %100, 0
  br i1 %101, label %102, label %105

102:                                              ; preds = %93
  %103 = call i32 @strcmp(ptr %96, ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8))
  %104 = icmp eq i32 %103, 0
  br label %107

105:                                              ; preds = %93
  %106 = icmp eq i64 %97, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_2495249626774661283, i64 8) to i64)
  br label %107

107:                                              ; preds = %102, %105
  %108 = phi i1 [ %106, %105 ], [ %104, %102 ]
  br label %109

109:                                              ; preds = %107
  br label %110

110:                                              ; preds = %109, %92
  %111 = phi i1 [ %108, %109 ], [ %91, %92 ]
  br i1 %111, label %112, label %113

112:                                              ; preds = %110
  store { ptr, ptr } zeroinitializer, ptr %3, align 8
  br label %117

113:                                              ; preds = %110
  %114 = load ptr, ptr %2, align 8
  %115 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %114, i32 0, i32 2
  %116 = load { ptr, ptr }, ptr %115, align 8
  store { ptr, ptr } %116, ptr %3, align 8
  br label %117

117:                                              ; preds = %112, %113
  %118 = load { ptr, ptr }, ptr %3, align 8
  ret { ptr, ptr } %118

119:                                              ; preds = %20, %74
  unreachable
}

; Function Attrs: mustprogress
define { ptr, ptr } @asAdder(ptr %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, ptr }, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call { ptr, ptr } @"___unbox_interface<!ts.iface<Adder>>"(ptr %4)
  store { ptr, ptr } %5, ptr %3, align 8
  %6 = load { ptr, ptr }, ptr %3, align 8
  ret { ptr, ptr } %6
}

; Function Attrs: mustprogress
define i1 @throws(ptr %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca i1, align 1
  store ptr %0, ptr %5, align 8
  store i1 false, ptr %4, align 1
  %7 = load ptr, ptr %5, align 8
  %8 = invoke { ptr, ptr } @asAdder(ptr %7)
          to label %9 unwind label %10

9:                                                ; preds = %1
  br label %149

10:                                               ; preds = %1
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %10
  %15 = icmp eq ptr %14, @_ZTIi
  br i1 %15, label %16, label %23

16:                                               ; preds = %.noexc2
  %17 = load i32, ptr %13, align 4
  %18 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %16
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 0
  store i64 4, ptr %20, align 8
  %21 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %21, align 8
  %22 = getelementptr { i64, ptr, i32 }, ptr %19, i32 0, i32 2
  store i32 %17, ptr %22, align 4
  br label %146

23:                                               ; preds = %.noexc2
  %24 = icmp eq ptr %14, @_ZTIj
  br i1 %24, label %25, label %32

25:                                               ; preds = %23
  %26 = load i32, ptr %13, align 4
  %27 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %25
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 0
  store i64 4, ptr %29, align 8
  %30 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %30, align 8
  %31 = getelementptr { i64, ptr, i32 }, ptr %28, i32 0, i32 2
  store i32 %26, ptr %31, align 4
  br label %143

32:                                               ; preds = %23
  %33 = icmp eq ptr %14, @_ZTId
  br i1 %33, label %34, label %41

34:                                               ; preds = %32
  %35 = load double, ptr %13, align 8
  %36 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %34
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 0
  store i64 8, ptr %38, align 8
  %39 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %39, align 8
  %40 = getelementptr { i64, ptr, double }, ptr %37, i32 0, i32 2
  store double %35, ptr %40, align 8
  br label %140

41:                                               ; preds = %32
  %42 = icmp eq ptr %14, @_ZTIf
  br i1 %42, label %43, label %50

43:                                               ; preds = %41
  %44 = load float, ptr %13, align 4
  %45 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %43
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 0
  store i64 4, ptr %47, align 8
  %48 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %48, align 8
  %49 = getelementptr { i64, ptr, float }, ptr %46, i32 0, i32 2
  store float %44, ptr %49, align 4
  br label %137

50:                                               ; preds = %41
  %51 = icmp eq ptr %14, @_ZTIb
  br i1 %51, label %52, label %59

52:                                               ; preds = %50
  %53 = load i1, ptr %13, align 1
  %54 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %52
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 0
  store i64 1, ptr %56, align 8
  %57 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %57, align 8
  %58 = getelementptr { i64, ptr, i1 }, ptr %55, i32 0, i32 2
  store i1 %53, ptr %58, align 1
  br label %134

59:                                               ; preds = %50
  %60 = icmp eq ptr %14, @_ZTIx
  br i1 %60, label %61, label %68

61:                                               ; preds = %59
  %62 = load i64, ptr %13, align 8
  %63 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %61
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 0
  store i64 8, ptr %65, align 8
  %66 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %66, align 8
  %67 = getelementptr { i64, ptr, i64 }, ptr %64, i32 0, i32 2
  store i64 %62, ptr %67, align 8
  br label %131

68:                                               ; preds = %59
  %69 = icmp eq ptr %14, @_ZTIPc
  br i1 %69, label %70, label %77

70:                                               ; preds = %68
  %71 = load ptr, ptr %13, align 8
  %72 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %70
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 0
  store i64 8, ptr %74, align 8
  %75 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %75, align 8
  %76 = getelementptr { i64, ptr, ptr }, ptr %73, i32 0, i32 2
  store ptr %71, ptr %76, align 8
  br label %128

77:                                               ; preds = %68
  %78 = icmp eq ptr %14, @_ZTIPv
  br i1 %78, label %79, label %81

79:                                               ; preds = %77
  %80 = load ptr, ptr %13, align 8
  br label %125

81:                                               ; preds = %77
  %82 = load ptr, ptr %14, align 8
  %83 = icmp eq ptr %82, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %83, label %84, label %115

84:                                               ; preds = %81
  %85 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %14, i32 0, i32 2
  %86 = load i32, ptr %85, align 4
  %87 = and i32 %86, 1073741824
  %88 = icmp ne i32 %87, 0
  br i1 %88, label %89, label %105

89:                                               ; preds = %84
  %90 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %14, i32 0, i32 4
  %91 = load ptr, ptr %90, align 8
  %92 = icmp ne ptr %91, null
  br i1 %92, label %93, label %95

93:                                               ; preds = %89
  invoke void %91(ptr %2, ptr %13)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %93
  %94 = load ptr, ptr %2, align 8
  br label %102

95:                                               ; preds = %89
  %96 = load ptr, ptr %13, align 8
  %97 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %95
  %98 = getelementptr i8, ptr %97, i64 8
  %99 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 0
  store i64 8, ptr %99, align 8
  %100 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %100, align 8
  %101 = getelementptr { i64, ptr, ptr }, ptr %98, i32 0, i32 2
  store ptr %96, ptr %101, align 8
  br label %102

102:                                              ; preds = %.noexc3, %.noexc10
  %103 = phi ptr [ %98, %.noexc10 ], [ %94, %.noexc3 ]
  br label %104

104:                                              ; preds = %102
  br label %112

105:                                              ; preds = %84
  %106 = load ptr, ptr %13, align 8
  %107 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %105
  %108 = getelementptr i8, ptr %107, i64 8
  %109 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 0
  store i64 8, ptr %109, align 8
  %110 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %110, align 8
  %111 = getelementptr { i64, ptr, ptr }, ptr %108, i32 0, i32 2
  store ptr %106, ptr %111, align 8
  br label %112

112:                                              ; preds = %104, %.noexc5
  %113 = phi ptr [ %108, %.noexc5 ], [ %103, %104 ]
  br label %114

114:                                              ; preds = %112
  br label %122

115:                                              ; preds = %81
  %116 = load ptr, ptr %13, align 8
  %117 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %115
  %118 = getelementptr i8, ptr %117, i64 8
  %119 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 0
  store i64 8, ptr %119, align 8
  %120 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %120, align 8
  %121 = getelementptr { i64, ptr, ptr }, ptr %118, i32 0, i32 2
  store ptr %116, ptr %121, align 8
  br label %122

122:                                              ; preds = %114, %.noexc7
  %123 = phi ptr [ %118, %.noexc7 ], [ %113, %114 ]
  br label %124

124:                                              ; preds = %122
  br label %125

125:                                              ; preds = %79, %124
  %126 = phi ptr [ %123, %124 ], [ %80, %79 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %.noexc6, %127
  %129 = phi ptr [ %126, %127 ], [ %73, %.noexc6 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc1, %130
  %132 = phi ptr [ %129, %130 ], [ %64, %.noexc1 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc8, %133
  %135 = phi ptr [ %132, %133 ], [ %55, %.noexc8 ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc11, %136
  %138 = phi ptr [ %135, %136 ], [ %46, %.noexc11 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc4, %139
  %141 = phi ptr [ %138, %139 ], [ %37, %.noexc4 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc9, %142
  %144 = phi ptr [ %141, %142 ], [ %28, %.noexc9 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc, %145
  %147 = phi ptr [ %144, %145 ], [ %19, %.noexc ]
  br label %148

148:                                              ; preds = %146
  store ptr %147, ptr %3, align 8
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %149

149:                                              ; preds = %9, %148
  %150 = load i1, ptr %4, align 1
  store i1 %150, ptr %6, align 1
  %151 = load i1, ptr %6, align 1
  ret i1 %151

endcatch.resume:                                  ; preds = %43, %95, %25, %52, %115, %70, %105, %34, %93, %10, %61, %16
  %152 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %152
}

; Function Attrs: mustprogress
define double @.f_main..afL27C27FH2872653265(double %0) #1 {
  %2 = alloca double, align 8
  %3 = alloca double, align 8
  store double %0, ptr %2, align 8
  %4 = load double, ptr %2, align 8
  %5 = fadd double %4, 1.000000e+01
  store double %5, ptr %3, align 8
  %6 = load double, ptr %3, align 8
  ret double %6
}

; Function Attrs: mustprogress
define double @.f_main..f_main..objL30C21FH2872653265.add(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %1, ptr %3, align 8
  %5 = load double, ptr %3, align 8
  %6 = fadd double %5, 2.000000e+01
  store double %6, ptr %4, align 8
  %7 = load double, ptr %4, align 8
  ret double %7
}

; Function Attrs: mustprogress
define double @.f_main..f_main..objL33C22FH2872653265.add(ptr %0, double %1) #1 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %1, ptr %3, align 8
  %5 = load double, ptr %3, align 8
  %6 = fadd double %5, 3.000000e+01
  store double %6, ptr %4, align 8
  %7 = load double, ptr %4, align 8
  ret double %7
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #1 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca { ptr }, align 8
  %2 = alloca { ptr }, align 8
  %3 = call ptr @GC_malloc(i64 32)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr { i64, ptr, { ptr } }, ptr %4, i32 0, i32 0
  store i64 8, ptr %5, align 8
  %6 = getelementptr { i64, ptr, { ptr } }, ptr %4, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_14594035851153911124_tuple, i64 40), ptr %6, align 8
  %7 = getelementptr { i64, ptr, { ptr } }, ptr %4, i32 0, i32 2
  store { ptr } { ptr @.f_main..afL27C27FH2872653265 }, ptr %7, align 8
  %8 = call i1 @throws(ptr %4)
  br i1 %8, label %9, label %53

9:                                                ; preds = %0
  store { ptr } { ptr @.f_main..f_main..objL30C21FH2872653265.add }, ptr %2, align 8
  %10 = load { ptr }, ptr %2, align 8
  %11 = call ptr @GC_malloc(i64 16)
  %12 = getelementptr i8, ptr %11, i64 8
  store { ptr } %10, ptr %12, align 8
  %13 = call ptr @GC_malloc(i64 32)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr { i64, ptr, ptr }, ptr %14, i32 0, i32 0
  store i64 8, ptr %15, align 8
  %16 = getelementptr { i64, ptr, ptr }, ptr %14, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_10237190548065851850_object, i64 40), ptr %16, align 8
  %17 = getelementptr { i64, ptr, ptr }, ptr %14, i32 0, i32 2
  store ptr %12, ptr %17, align 8
  %18 = call i1 @throws(ptr %14)
  br i1 %18, label %19, label %55

19:                                               ; preds = %9
  store { ptr } { ptr @.f_main..f_main..objL33C22FH2872653265.add }, ptr %1, align 8
  %20 = load { ptr }, ptr %1, align 8
  %21 = call ptr @GC_malloc(i64 16)
  %22 = getelementptr i8, ptr %21, i64 8
  store { ptr } %20, ptr %22, align 8
  %23 = insertvalue { ptr, ptr } { ptr @Adder.3680198976672767672..vtbl, ptr undef }, ptr %22, 1
  %24 = call ptr @GC_malloc(i64 40)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %25, i32 0, i32 0
  store i64 16, ptr %26, align 8
  %27 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %25, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_3693193992939591961_interface, i64 40), ptr %27, align 8
  %28 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %25, i32 0, i32 2
  store { ptr, ptr } %23, ptr %28, align 8
  %29 = call { ptr, ptr } @asAdder(ptr %25)
  %30 = extractvalue { ptr, ptr } %29, 0
  %31 = extractvalue { ptr, ptr } %29, 1
  %32 = getelementptr ptr, ptr %30, i32 2
  %33 = load ptr, ptr %32, align 8
  %34 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %33, 0
  %35 = insertvalue { ptr, ptr, ptr } %34, ptr %31, 1
  %36 = insertvalue { ptr, ptr, ptr } %35, ptr null, 2
  %37 = extractvalue { ptr, ptr, ptr } %36, 1
  %38 = extractvalue { ptr, ptr, ptr } %36, 0
  %39 = call double %38(ptr %37, double 1.000000e+00)
  %40 = fcmp oeq double %39, 3.100000e+01
  br i1 %40, label %41, label %57

41:                                               ; preds = %19
  %42 = call ptr @GC_malloc(i64 32)
  %43 = getelementptr i8, ptr %42, i64 8
  %44 = getelementptr { i64, ptr, ptr }, ptr %43, i32 0, i32 0
  store i64 8, ptr %44, align 8
  %45 = getelementptr { i64, ptr, ptr }, ptr %43, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_14527407094101050897_null, i64 40), ptr %45, align 8
  %46 = getelementptr { i64, ptr, ptr }, ptr %43, i32 0, i32 2
  store ptr null, ptr %46, align 8
  %47 = call { ptr, ptr } @asAdder(ptr %43)
  %48 = extractvalue { ptr, ptr } %47, 1
  %49 = ptrtoint ptr %48 to i64
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %51, label %59

51:                                               ; preds = %41
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

52:                                               ; preds = %59, %57, %55, %53
  unreachable

53:                                               ; preds = %0
  %54 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11039441214965475813, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10417848714019435974, i64 8), i32 28, ptr null)
  br label %52

55:                                               ; preds = %9
  %56 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13436669823751029251, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10417848714019435974, i64 8), i32 31, ptr null)
  br label %52

57:                                               ; preds = %19
  %58 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8835751322448772705, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10417848714019435974, i64 8), i32 34, ptr null)
  br label %52

59:                                               ; preds = %41
  %60 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6601085983368743140, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_10417848714019435974, i64 8), i32 36, ptr null)
  br label %52
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

