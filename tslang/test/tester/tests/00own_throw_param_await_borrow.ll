; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%E = type { ptr, ptr }
%async_execute_fn.Frame = type { ptr, ptr, ptr, ptr, i1 }
%async_execute_fn_0.Frame = type { ptr, ptr, ptr, ptr, i1 }
%async_execute_fn_1.Frame = type { ptr, ptr, ptr, ptr, i1 }

$"reject<si32,!ts.class<E, !ts.class_storage<E>>>" = comdat any

$"identity<!ts.string>" = comdat any

$E..size = comdat any

$_ZTSP1E = comdat any

$_ZTIP1E = comdat any

$_ZTS1E = comdat any

$_ZTI1E = comdat any

@assert_msg_1 = private constant [40 x i8] c"Awaited async operand is in error state\00"
@assert_msg_0 = private constant [40 x i8] c"Awaited async operand is in error state\00"
@assert_msg = private constant [40 x i8] c"Awaited async operand is in error state\00"
@s_12638217986881231904 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFE\00", align 8
@E..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638217986881231904, i64 8)
@E..size = global i64 16, comdat
@E..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @E..instanceOf, ptr @E..new, ptr @E..rtti, ptr @E..size }
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP1E = linkonce_odr constant [4 x i8] c"P1E\00", comdat
@_ZTIP1E = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP1E, i32 1073741824, ptr @_ZTI1E, ptr @.eh.copy.box.P1E }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS1E = linkonce_odr constant [3 x i8] c"1E\00", comdat
@s_12230838483901444113 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFhello \00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_12661613066403209478 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnamed\00", align 8
@s_11403822988935609658 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFgeneric\00", align 8
@s_18252272687975150087 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFplain\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@m_10759396514797719557 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa thrown parameter\00", align 8
@f_7330186968285609564 = internal constant [120 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00own_throw_param_await_borrow.ts\00", align 8
@m_5374263015666245876 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa thrown generic parameter\00", align 8
@m_11403645802570620318 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa named local given up to the throw\00", align 8
@m_12631761716629730952 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa throw on one path\00", align 8
@s_1317993492180036150 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFlit\00", align 8
@m_10384492284169401148 = internal constant [43 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan awaited borrowed string, copied\00", align 8
@m_9852187990986148368 = internal constant [36 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan awaited borrowed literal\00", align 8
@s_312668867041707993 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFlast\00", align 8
@m_7405353071346851579 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan await in a try body\00", align 8
@s_13069864042462355804 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFyou\00", align 8
@s_12644979747856758223 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFagain\00", align 8
@_ZTI1E = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1E }, comdat
@td_7632308648875476676_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_7632308648875476676, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_uncollectable(i64) #0

declare void @abort()

declare void @puts(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

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

define internal void @tsrel_7632308648875476676(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %E, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare i32 @strcmp(ptr, ptr)

declare void @GC_free(ptr)

; Function Attrs: mustprogress
define i1 @E..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @E..rtti, align 8
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
define ptr @E..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %E, ptr %3, i32 0, i32 0
  store ptr @E..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @E.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %E, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P1E(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_7632308648875476676_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define i32 @rejectE(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %3, ptr %4, align 8
  call void @__cxa_throw(ptr %4, ptr @_ZTIP1E, ptr null)
  br label %5

5:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i32 @maybeThrow(ptr %0, i1 %1) #2 {
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store i1 %1, ptr %3, align 1
  %6 = load i1, ptr %3, align 1
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = load ptr, ptr %4, align 8
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %8, ptr %9, align 8
  call void @__cxa_throw(ptr %9, ptr @_ZTIP1E, ptr null)
  br label %12

10:                                               ; preds = %2
  store i32 1, ptr %5, align 4
  %11 = load i32, ptr %5, align 4
  ret i32 %11

12:                                               ; preds = %7
  unreachable
}

; Function Attrs: mustprogress
define ptr @greet(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = icmp eq ptr %4, null
  %6 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %4
  %7 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12230838483901444113, i64 8))
  %8 = add i64 1, %7
  %9 = call i64 @strlen(ptr %6)
  %10 = add i64 %8, %9
  %11 = add i64 %10, 8
  %12 = call ptr @GC_malloc(i64 %11)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr i8, ptr %13, i64 0
  %15 = sub i64 %10, 0
  %16 = call ptr @__memcpy_chk(ptr %14, ptr getelementptr inbounds nuw (i8, ptr @s_12230838483901444113, i64 8), i64 %7, i64 %15)
  %17 = add i64 0, %7
  %18 = getelementptr i8, ptr %13, i64 %17
  %19 = sub i64 %10, %17
  %20 = call ptr @__memcpy_chk(ptr %18, ptr %6, i64 %9, i64 %19)
  %21 = add i64 %17, %9
  %22 = getelementptr i8, ptr %13, i64 %21
  store i8 0, ptr %22, align 1
  store ptr %13, ptr %3, align 8
  %23 = load ptr, ptr %3, align 8
  ret ptr %23
}

; Function Attrs: mustprogress
define linkonce_odr i32 @"reject<si32,!ts.class<E, !ts.class_storage<E>>>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %3, ptr %4, align 8
  call void @__cxa_throw(ptr %4, ptr @_ZTIP1E, ptr null)
  br label %5

5:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %7, align 8
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %E, ptr %9, i32 0, i32 0
  store ptr @E..vtbl, ptr %10, align 8
  invoke void @E.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_18252272687975150087, i64 8))
          to label %11 unwind label %14

11:                                               ; preds = %0
  %12 = invoke i32 @rejectE(ptr %9)
          to label %13 unwind label %14

13:                                               ; preds = %11
  br label %28

14:                                               ; preds = %11, %0
  %15 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %16 = extractvalue { ptr, i32 } %15, 1
  %17 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %18 = icmp eq i32 %16, %17
  br i1 %18, label %22, label %19

19:                                               ; preds = %14
  %20 = extractvalue { ptr, i32 } %15, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  invoke void @__cxa_rethrow()
          to label %157 unwind label %endcatch.pad

22:                                               ; preds = %14
  %23 = extractvalue { ptr, i32 } %15, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  store ptr %24, ptr %6, align 8
  %25 = load ptr, ptr %6, align 8
  %26 = getelementptr %E, ptr %25, i32 0, i32 1
  %27 = load ptr, ptr %26, align 8
  store ptr %27, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %28

28:                                               ; preds = %13, %22
  %29 = load ptr, ptr %7, align 8
  %30 = ptrtoint ptr %29 to i64
  %31 = icmp ne i64 %30, 0
  %32 = zext i1 %31 to i32
  %33 = and i32 %32, 1
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %35, label %38

35:                                               ; preds = %28
  %36 = call i32 @strcmp(ptr %29, ptr getelementptr inbounds nuw (i8, ptr @s_18252272687975150087, i64 8))
  %37 = icmp eq i32 %36, 0
  br label %40

38:                                               ; preds = %28
  %39 = icmp eq i64 %30, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_18252272687975150087, i64 8) to i64)
  br label %40

40:                                               ; preds = %35, %38
  %41 = phi i1 [ %39, %38 ], [ %37, %35 ]
  br label %42

42:                                               ; preds = %40
  br i1 %41, label %43, label %158

43:                                               ; preds = %42
  %44 = call ptr @GC_malloc(i64 24)
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr %E, ptr %45, i32 0, i32 0
  store ptr @E..vtbl, ptr %46, align 8
  invoke void @E.constructor(ptr %45, ptr getelementptr inbounds nuw (i8, ptr @s_11403822988935609658, i64 8))
          to label %47 unwind label %50

47:                                               ; preds = %43
  %48 = invoke i32 @"reject<si32,!ts.class<E, !ts.class_storage<E>>>"(ptr %45)
          to label %49 unwind label %50

49:                                               ; preds = %47
  br label %64

50:                                               ; preds = %47, %43
  %51 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %52 = extractvalue { ptr, i32 } %51, 1
  %53 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %54 = icmp eq i32 %52, %53
  br i1 %54, label %58, label %55

55:                                               ; preds = %50
  %56 = extractvalue { ptr, i32 } %51, 0
  %57 = call ptr @__cxa_begin_catch(ptr %56)
  invoke void @__cxa_rethrow()
          to label %157 unwind label %endcatch.pad

58:                                               ; preds = %50
  %59 = extractvalue { ptr, i32 } %51, 0
  %60 = call ptr @__cxa_begin_catch(ptr %59)
  store ptr %60, ptr %5, align 8
  %61 = load ptr, ptr %5, align 8
  %62 = getelementptr %E, ptr %61, i32 0, i32 1
  %63 = load ptr, ptr %62, align 8
  store ptr %63, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %64

64:                                               ; preds = %49, %58
  %65 = load ptr, ptr %7, align 8
  %66 = ptrtoint ptr %65 to i64
  %67 = icmp ne i64 %66, 0
  %68 = zext i1 %67 to i32
  %69 = and i32 %68, 1
  %70 = icmp ne i32 %69, 0
  br i1 %70, label %71, label %74

71:                                               ; preds = %64
  %72 = call i32 @strcmp(ptr %65, ptr getelementptr inbounds nuw (i8, ptr @s_11403822988935609658, i64 8))
  %73 = icmp eq i32 %72, 0
  br label %76

74:                                               ; preds = %64
  %75 = icmp eq i64 %66, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11403822988935609658, i64 8) to i64)
  br label %76

76:                                               ; preds = %71, %74
  %77 = phi i1 [ %75, %74 ], [ %73, %71 ]
  br label %78

78:                                               ; preds = %76
  br i1 %77, label %79, label %160

79:                                               ; preds = %78
  %80 = call ptr @GC_malloc(i64 24)
  %81 = getelementptr i8, ptr %80, i64 8
  %82 = getelementptr %E, ptr %81, i32 0, i32 0
  store ptr @E..vtbl, ptr %82, align 8
  invoke void @E.constructor(ptr %81, ptr getelementptr inbounds nuw (i8, ptr @s_12661613066403209478, i64 8))
          to label %83 unwind label %154

83:                                               ; preds = %79
  %84 = invoke i32 @rejectE(ptr %81)
          to label %85 unwind label %86

85:                                               ; preds = %83
  br label %100

86:                                               ; preds = %83
  %87 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %88 = extractvalue { ptr, i32 } %87, 1
  %89 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %90 = icmp eq i32 %88, %89
  br i1 %90, label %94, label %91

91:                                               ; preds = %86
  %92 = extractvalue { ptr, i32 } %87, 0
  %93 = call ptr @__cxa_begin_catch(ptr %92)
  invoke void @__cxa_rethrow()
          to label %157 unwind label %endcatch.pad

94:                                               ; preds = %86
  %95 = extractvalue { ptr, i32 } %87, 0
  %96 = call ptr @__cxa_begin_catch(ptr %95)
  store ptr %96, ptr %4, align 8
  %97 = load ptr, ptr %4, align 8
  %98 = getelementptr %E, ptr %97, i32 0, i32 1
  %99 = load ptr, ptr %98, align 8
  store ptr %99, ptr %7, align 8
  call void @__cxa_end_catch()
  br label %100

100:                                              ; preds = %85, %94
  %101 = load ptr, ptr %7, align 8
  %102 = ptrtoint ptr %101 to i64
  %103 = icmp ne i64 %102, 0
  %104 = zext i1 %103 to i32
  %105 = and i32 %104, 1
  %106 = icmp ne i32 %105, 0
  br i1 %106, label %107, label %110

107:                                              ; preds = %100
  %108 = call i32 @strcmp(ptr %101, ptr getelementptr inbounds nuw (i8, ptr @s_12661613066403209478, i64 8))
  %109 = icmp eq i32 %108, 0
  br label %112

110:                                              ; preds = %100
  %111 = icmp eq i64 %102, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12661613066403209478, i64 8) to i64)
  br label %112

112:                                              ; preds = %107, %110
  %113 = phi i1 [ %111, %110 ], [ %109, %107 ]
  br label %114

114:                                              ; preds = %112
  br i1 %113, label %115, label %162

115:                                              ; preds = %114
  store i32 0, ptr %3, align 4
  store i32 0, ptr %2, align 4
  br label %116

116:                                              ; preds = %144, %115
  %117 = load i32, ptr %2, align 4
  %118 = icmp slt i32 %117, 100
  br i1 %118, label %119, label %147

119:                                              ; preds = %116
  %120 = load i32, ptr %3, align 4
  %121 = call ptr @GC_malloc(i64 24)
  %122 = getelementptr i8, ptr %121, i64 8
  %123 = getelementptr %E, ptr %122, i32 0, i32 0
  store ptr @E..vtbl, ptr %123, align 8
  invoke void @E.constructor(ptr %122, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
          to label %124 unwind label %131

124:                                              ; preds = %119
  %125 = load i32, ptr %2, align 4
  %126 = srem i32 %125, 2
  %127 = icmp eq i32 %126, 0
  %128 = invoke i32 @maybeThrow(ptr %122, i1 %127)
          to label %129 unwind label %131

129:                                              ; preds = %124
  %130 = add i32 %120, %128
  store i32 %130, ptr %3, align 4
  br label %144

131:                                              ; preds = %124, %119
  %132 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
          catch ptr null
  %133 = extractvalue { ptr, i32 } %132, 1
  %134 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %135 = icmp eq i32 %133, %134
  br i1 %135, label %139, label %136

136:                                              ; preds = %131
  %137 = extractvalue { ptr, i32 } %132, 0
  %138 = call ptr @__cxa_begin_catch(ptr %137)
  invoke void @__cxa_rethrow()
          to label %157 unwind label %endcatch.pad

139:                                              ; preds = %131
  %140 = extractvalue { ptr, i32 } %132, 0
  %141 = call ptr @__cxa_begin_catch(ptr %140)
  store ptr %141, ptr %1, align 8
  %142 = load i32, ptr %3, align 4
  %143 = add i32 %142, 10
  store i32 %143, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %144

144:                                              ; preds = %129, %139
  %145 = load i32, ptr %2, align 4
  %146 = add i32 %145, 1
  store i32 %146, ptr %2, align 4
  br label %116

147:                                              ; preds = %116
  %148 = load i32, ptr %3, align 4
  %149 = icmp eq i32 %148, 550
  br i1 %149, label %150, label %164

150:                                              ; preds = %147
  invoke void @awaitBorrowed()
          to label %151 unwind label %154

151:                                              ; preds = %150
  invoke void @awaitInTry()
          to label %152 unwind label %154

152:                                              ; preds = %151
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

endcatch.pad:                                     ; preds = %91, %136, %55, %19
  %153 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

154:                                              ; preds = %151, %150, %79
  %155 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %154
  %156 = phi { ptr, i32 } [ %155, %154 ], [ %153, %endcatch.pad ]
  resume { ptr, i32 } %156

157:                                              ; preds = %136, %164, %162, %91, %160, %55, %158, %19
  unreachable

158:                                              ; preds = %42
  %159 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10759396514797719557, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 42, ptr null)
  br label %157

160:                                              ; preds = %78
  %161 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5374263015666245876, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 50, ptr null)
  br label %157

162:                                              ; preds = %114
  %163 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11403645802570620318, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 59, ptr null)
  br label %157

164:                                              ; preds = %147
  %165 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12631761716629730952, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 70, ptr null)
  br label %157
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"identity<!ts.string>"(ptr %0) #2 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  store ptr %4, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @awaitBorrowed() #2 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  br label %5

5:                                                ; preds = %12, %0
  %6 = load i32, ptr %3, align 4
  %7 = icmp slt i32 %6, 100
  br i1 %7, label %8, label %21

8:                                                ; preds = %5
  %9 = call ptr @async_execute_fn(ptr %2)
  call void @mlirAsyncRuntimeAwaitToken(ptr %9)
  %10 = call i1 @mlirAsyncRuntimeIsTokenError(ptr %9)
  call void @mlirAsyncRuntimeDropRef(ptr %9, i64 1)
  %11 = xor i1 %10, true
  br i1 %11, label %12, label %44

12:                                               ; preds = %8
  %13 = load ptr, ptr %2, align 8
  %14 = load i32, ptr %4, align 4
  %15 = call i64 @strlen(ptr %13)
  %16 = sext i32 %14 to i64
  %17 = add i64 %16, %15
  %18 = trunc i64 %17 to i32
  store i32 %18, ptr %4, align 4
  %19 = load i32, ptr %3, align 4
  %20 = add i32 %19, 1
  store i32 %20, ptr %3, align 4
  br label %5

21:                                               ; preds = %5
  %22 = load i32, ptr %4, align 4
  %23 = icmp eq i32 %22, 900
  br i1 %23, label %24, label %45

24:                                               ; preds = %21
  %25 = call ptr @async_execute_fn_0(ptr %1)
  call void @mlirAsyncRuntimeAwaitToken(ptr %25)
  %26 = call i1 @mlirAsyncRuntimeIsTokenError(ptr %25)
  call void @mlirAsyncRuntimeDropRef(ptr %25, i64 1)
  %27 = xor i1 %26, true
  br i1 %27, label %28, label %47

28:                                               ; preds = %24
  %29 = load ptr, ptr %1, align 8
  %30 = ptrtoint ptr %29 to i64
  %31 = icmp ne i64 %30, 0
  %32 = zext i1 %31 to i32
  %33 = and i32 %32, 1
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %35, label %38

35:                                               ; preds = %28
  %36 = call i32 @strcmp(ptr %29, ptr getelementptr inbounds nuw (i8, ptr @s_1317993492180036150, i64 8))
  %37 = icmp eq i32 %36, 0
  br label %40

38:                                               ; preds = %28
  %39 = icmp eq i64 %30, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1317993492180036150, i64 8) to i64)
  br label %40

40:                                               ; preds = %35, %38
  %41 = phi i1 [ %39, %38 ], [ %37, %35 ]
  br label %42

42:                                               ; preds = %40
  br i1 %41, label %43, label %48

43:                                               ; preds = %42
  ret void

44:                                               ; preds = %8, %48, %45
  call void @puts(ptr @assert_msg)
  call void @abort()
  unreachable

45:                                               ; preds = %21
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10384492284169401148, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 88, ptr null)
  br label %44

47:                                               ; preds = %24
  call void @puts(ptr @assert_msg_0)
  call void @abort()
  unreachable

48:                                               ; preds = %42
  %49 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9852187990986148368, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 91, ptr null)
  br label %44
}

; Function Attrs: mustprogress
define void @awaitInTry() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  br label %5

5:                                                ; preds = %46, %0
  %6 = load i32, ptr %3, align 4
  %7 = icmp slt i32 %6, 100
  br i1 %7, label %8, label %49

8:                                                ; preds = %5
  %9 = call ptr @async_execute_fn_1(ptr %1)
  call void @mlirAsyncRuntimeAwaitToken(ptr %9)
  %10 = call i1 @mlirAsyncRuntimeIsTokenError(ptr %9)
  call void @mlirAsyncRuntimeDropRef(ptr %9, i64 1)
  %11 = xor i1 %10, true
  br i1 %11, label %12, label %53

12:                                               ; preds = %8
  %13 = load ptr, ptr %1, align 8
  %14 = load i32, ptr %4, align 4
  %15 = call i64 @strlen(ptr %13)
  %16 = sext i32 %14 to i64
  %17 = add i64 %16, %15
  %18 = trunc i64 %17 to i32
  store i32 %18, ptr %4, align 4
  %19 = load i32, ptr %3, align 4
  %20 = icmp eq i32 %19, 99
  br i1 %20, label %21, label %46

21:                                               ; preds = %12
  %22 = call ptr @GC_malloc(i64 24)
  %23 = getelementptr i8, ptr %22, i64 8
  %24 = getelementptr %E, ptr %23, i32 0, i32 0
  store ptr @E..vtbl, ptr %24, align 8
  invoke void @E.constructor(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_312668867041707993, i64 8))
          to label %25 unwind label %27

25:                                               ; preds = %21
  %26 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %23, ptr %26, align 8
  invoke void @__cxa_throw(ptr %26, ptr @_ZTIP1E, ptr null)
          to label %53 unwind label %27

27:                                               ; preds = %25, %21
  %28 = landingpad { ptr, i32 }
          catch ptr @_ZTIP1E
  %29 = extractvalue { ptr, i32 } %28, 1
  %30 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP1E)
  %31 = icmp eq i32 %29, %30
  br i1 %31, label %35, label %32

32:                                               ; preds = %27
  %33 = extractvalue { ptr, i32 } %28, 0
  %34 = call ptr @__cxa_begin_catch(ptr %33)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %32
  br label %53

35:                                               ; preds = %27
  %36 = extractvalue { ptr, i32 } %28, 0
  %37 = call ptr @__cxa_begin_catch(ptr %36)
  store ptr %37, ptr %2, align 8
  %38 = load i32, ptr %4, align 4
  %39 = load ptr, ptr %2, align 8
  %40 = getelementptr %E, ptr %39, i32 0, i32 1
  %41 = load ptr, ptr %40, align 8
  %42 = invoke i64 @strlen(ptr %41)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %35
  %43 = sext i32 %38 to i64
  %44 = add i64 %43, %42
  %45 = trunc i64 %44 to i32
  store i32 %45, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %46

46:                                               ; preds = %.noexc, %12
  %47 = load i32, ptr %3, align 4
  %48 = add i32 %47, 1
  store i32 %48, ptr %3, align 4
  br label %5

49:                                               ; preds = %5
  %50 = load i32, ptr %4, align 4
  %51 = icmp eq i32 %50, 1104
  br i1 %51, label %52, label %54

52:                                               ; preds = %49
  ret void

53:                                               ; preds = %.noexc1, %25, %8, %54
  call void @puts(ptr @assert_msg_1)
  call void @abort()
  unreachable

54:                                               ; preds = %49
  %55 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7405353071346851579, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7330186968285609564, i64 8), i32 110, ptr null)
  br label %53

endcatch.resume:                                  ; preds = %32, %35
  %56 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %56
}

define ptr @async_execute_fn(ptr %0) {
AllocaSpillBB:
  %1 = call ptr @mlirAsyncRuntimeCreateToken()
  %2 = add i64 40, 8
  %3 = sub i64 %2, 1
  %4 = sub i64 0, 8
  %5 = and i64 %3, %4
  %6 = call ptr @GC_malloc_uncollectable(i64 %5)
  %resume.addr = getelementptr inbounds nuw %async_execute_fn.Frame, ptr %6, i32 0, i32 0
  store ptr @async_execute_fn.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %async_execute_fn.Frame, ptr %6, i32 0, i32 1
  store ptr @async_execute_fn.destroy, ptr %destroy.addr, align 8
  %.spill.addr1 = getelementptr inbounds %async_execute_fn.Frame, ptr %6, i32 0, i32 3
  store ptr %1, ptr %.spill.addr1, align 8
  %.spill.addr = getelementptr inbounds %async_execute_fn.Frame, ptr %6, i32 0, i32 2
  store ptr %0, ptr %.spill.addr, align 8
  %index.addr4 = getelementptr inbounds nuw %async_execute_fn.Frame, ptr %6, i32 0, i32 4
  store i1 false, ptr %index.addr4, align 1
  call void @mlirAsyncRuntimeExecute(ptr %6, ptr @__resume)
  ret ptr %1
}

define ptr @async_execute_fn_0(ptr %0) {
AllocaSpillBB:
  %1 = call ptr @mlirAsyncRuntimeCreateToken()
  %2 = add i64 40, 8
  %3 = sub i64 %2, 1
  %4 = sub i64 0, 8
  %5 = and i64 %3, %4
  %6 = call ptr @GC_malloc_uncollectable(i64 %5)
  %resume.addr = getelementptr inbounds nuw %async_execute_fn_0.Frame, ptr %6, i32 0, i32 0
  store ptr @async_execute_fn_0.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %async_execute_fn_0.Frame, ptr %6, i32 0, i32 1
  store ptr @async_execute_fn_0.destroy, ptr %destroy.addr, align 8
  %.spill.addr1 = getelementptr inbounds %async_execute_fn_0.Frame, ptr %6, i32 0, i32 3
  store ptr %1, ptr %.spill.addr1, align 8
  %.spill.addr = getelementptr inbounds %async_execute_fn_0.Frame, ptr %6, i32 0, i32 2
  store ptr %0, ptr %.spill.addr, align 8
  %index.addr4 = getelementptr inbounds nuw %async_execute_fn_0.Frame, ptr %6, i32 0, i32 4
  store i1 false, ptr %index.addr4, align 1
  call void @mlirAsyncRuntimeExecute(ptr %6, ptr @__resume)
  ret ptr %1
}

define ptr @async_execute_fn_1(ptr %0) {
AllocaSpillBB:
  %1 = call ptr @mlirAsyncRuntimeCreateToken()
  %2 = add i64 40, 8
  %3 = sub i64 %2, 1
  %4 = sub i64 0, 8
  %5 = and i64 %3, %4
  %6 = call ptr @GC_malloc_uncollectable(i64 %5)
  %resume.addr = getelementptr inbounds nuw %async_execute_fn_1.Frame, ptr %6, i32 0, i32 0
  store ptr @async_execute_fn_1.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %async_execute_fn_1.Frame, ptr %6, i32 0, i32 1
  store ptr @async_execute_fn_1.destroy, ptr %destroy.addr, align 8
  %.spill.addr1 = getelementptr inbounds %async_execute_fn_1.Frame, ptr %6, i32 0, i32 3
  store ptr %1, ptr %.spill.addr1, align 8
  %.spill.addr = getelementptr inbounds %async_execute_fn_1.Frame, ptr %6, i32 0, i32 2
  store ptr %0, ptr %.spill.addr, align 8
  %index.addr4 = getelementptr inbounds nuw %async_execute_fn_1.Frame, ptr %6, i32 0, i32 4
  store i1 false, ptr %index.addr4, align 1
  call void @mlirAsyncRuntimeExecute(ptr %6, ptr @__resume)
  ret ptr %1
}

declare void @mlirAsyncRuntimeDropRef(ptr, i64)

declare ptr @mlirAsyncRuntimeCreateToken()

declare void @mlirAsyncRuntimeEmplaceToken(ptr)

declare i1 @mlirAsyncRuntimeIsTokenError(ptr)

declare void @mlirAsyncRuntimeAwaitToken(ptr)

declare void @mlirAsyncRuntimeExecute(ptr, ptr)

define void @__resume(ptr %0) {
  %2 = getelementptr inbounds { ptr, ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  call fastcc void %3(ptr %0)
  ret void
}

define internal fastcc void @async_execute_fn.resume(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.resume:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.resume
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 0 to i32
  switch i32 %1, label %6 [
    i32 0, label %2
    i32 1, label %5
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @greet(ptr getelementptr inbounds nuw (i8, ptr @s_13069864042462355804, i64 8))
  %4 = call ptr @"identity<!ts.string>"(ptr %3)
  store ptr %4, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %6

5:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %6

6:                                                ; preds = %5, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %6
  ret void
}

define internal fastcc void @async_execute_fn.destroy(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.destroy:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.destroy
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 1 to i32
  switch i32 %1, label %6 [
    i32 0, label %2
    i32 1, label %5
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @greet(ptr getelementptr inbounds nuw (i8, ptr @s_13069864042462355804, i64 8))
  %4 = call ptr @"identity<!ts.string>"(ptr %3)
  store ptr %4, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %6

5:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %6

6:                                                ; preds = %5, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %6
  ret void
}

define internal fastcc void @async_execute_fn_0.resume(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.resume:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.resume
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 0 to i32
  switch i32 %1, label %5 [
    i32 0, label %2
    i32 1, label %4
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @"identity<!ts.string>"(ptr getelementptr inbounds nuw (i8, ptr @s_1317993492180036150, i64 8))
  store ptr %3, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %5

4:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %5

5:                                                ; preds = %4, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %5
  ret void
}

define internal fastcc void @async_execute_fn_0.destroy(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.destroy:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.destroy
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 1 to i32
  switch i32 %1, label %5 [
    i32 0, label %2
    i32 1, label %4
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn_0.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @"identity<!ts.string>"(ptr getelementptr inbounds nuw (i8, ptr @s_1317993492180036150, i64 8))
  store ptr %3, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %5

4:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %5

5:                                                ; preds = %4, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %5
  ret void
}

define internal fastcc void @async_execute_fn_1.resume(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.resume:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.resume
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 0 to i32
  switch i32 %1, label %6 [
    i32 0, label %2
    i32 1, label %5
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn_1.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn_1.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @greet(ptr getelementptr inbounds nuw (i8, ptr @s_12644979747856758223, i64 8))
  %4 = call ptr @"identity<!ts.string>"(ptr %3)
  store ptr %4, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %6

5:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %6

6:                                                ; preds = %5, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %6
  ret void
}

define internal fastcc void @async_execute_fn_1.destroy(ptr noundef nonnull align 8 dereferenceable(40) %0) {
entry.destroy:
  br label %resume.entry

resume.entry:                                     ; preds = %entry.destroy
  br label %resume.0

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  %1 = sext i8 1 to i32
  switch i32 %1, label %6 [
    i32 0, label %2
    i32 1, label %5
  ]

2:                                                ; preds = %AfterCoroSuspend
  %.reload.addr2 = getelementptr inbounds %async_execute_fn_1.Frame, ptr %0, i32 0, i32 3
  %.reload3 = load ptr, ptr %.reload.addr2, align 8
  %.reload.addr = getelementptr inbounds %async_execute_fn_1.Frame, ptr %0, i32 0, i32 2
  %.reload = load ptr, ptr %.reload.addr, align 8
  %3 = call ptr @greet(ptr getelementptr inbounds nuw (i8, ptr @s_12644979747856758223, i64 8))
  %4 = call ptr @"identity<!ts.string>"(ptr %3)
  store ptr %4, ptr %.reload, align 8
  call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload3)
  call void @GC_free(ptr %0)
  br label %6

5:                                                ; preds = %AfterCoroSuspend
  call void @GC_free(ptr %0)
  br label %6

6:                                                ; preds = %5, %2, %AfterCoroSuspend
  br label %CoroEnd

CoroEnd:                                          ; preds = %6
  ret void
}

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

