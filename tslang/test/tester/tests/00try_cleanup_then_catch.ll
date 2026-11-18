; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Node = type { ptr, i32 }
%Res = type { ptr }

$"___unbox<!ts.class<Node, !ts.class_storage<Node>>>" = comdat any

$Node..size = comdat any

$Res..size = comdat any

@s_7403105002735449741 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNode\00", align 8
@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_16124472420822520262 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdisposed\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_12515944260747508003 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin try\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@s_3019566071980655109 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcaught\00", align 8
@s_9690130352454037036 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnot reached\00", align 8
@s_4346689792895745607 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from any type\00", align 8
@s_10002225064126424008 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCan't cast from interface\00", align 8
@s_11582289249049393296 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface\00", align 8
@s_15066323702654938015 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_7316157474859157787 = internal constant [58 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan owning let, a two-argument print, then a throw\00", align 8
@f_9989496759544712621 = internal constant [114 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_cleanup_then_catch.ts\00", align 8
@m_383063154153552527 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan owning let, then a two-argument print in the catch\00", align 8
@m_5248496707480371780 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using, then a two-argument print in the catch\00", align 8
@m_18314962778420153713 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa let whose initializer throws (#456)\00", align 8
@m_10581538897204661568 = internal constant [39 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa let whose cast throws (#456)\00", align 8
@m_12074018024437361393 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa const needs no cleanup\00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@Node..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7403105002735449741, i64 8)
@Node..size = global i64 16, comdat
@Node..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Node..instanceOf, ptr @Node..new, ptr @Node..rtti, ptr @Node..size }
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @puts(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define void @Node.constructor(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 1
  store i32 0, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @Node..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Node..rtti, align 8
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
define ptr @Node..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Node, ptr %3, i32 0, i32 0
  store ptr @Node..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Res..rtti, align 8
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
define ptr @Res..new() #1 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #1 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_16124472420822520262, i64 8))
  ret void
}

; Function Attrs: mustprogress
define ptr @fail() #1 {
  %1 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), ptr %1, align 8
  call void @__cxa_throw(ptr %1, ptr @_ZTIPc, ptr null)
  br label %2

2:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define i1 @letThenThrow(ptr %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store i1 false, ptr %3, align 1
  %6 = load ptr, ptr %4, align 8
  store ptr %6, ptr %2, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = getelementptr %Node, ptr %7, i32 0, i32 1
  %9 = load i32, ptr %8, align 4
  %10 = call ptr @GC_malloc_atomic(i64 58)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %11, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %9)
  %13 = call ptr @llvm.stacksave.p0()
  %14 = icmp eq ptr %11, null
  %15 = select i1 %14, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %11
  %16 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12515944260747508003, i64 8))
  %17 = add i64 1, %16
  %18 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %19 = add i64 %17, %18
  %20 = call i64 @strlen(ptr %15)
  %21 = add i64 %19, %20
  %22 = alloca i8, i64 %21, align 1
  %23 = getelementptr i8, ptr %22, i64 0
  %24 = sub i64 %21, 0
  %25 = call ptr @__memcpy_chk(ptr %23, ptr getelementptr inbounds nuw (i8, ptr @s_12515944260747508003, i64 8), i64 %16, i64 %24)
  %26 = add i64 0, %16
  %27 = getelementptr i8, ptr %22, i64 %26
  %28 = sub i64 %21, %26
  %29 = call ptr @__memcpy_chk(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %18, i64 %28)
  %30 = add i64 %26, %18
  %31 = getelementptr i8, ptr %22, i64 %30
  %32 = sub i64 %21, %30
  %33 = call ptr @__memcpy_chk(ptr %31, ptr %15, i64 %20, i64 %32)
  %34 = add i64 %30, %20
  %35 = getelementptr i8, ptr %22, i64 %34
  store i8 0, ptr %35, align 1
  call void @puts(ptr %22)
  call void @llvm.stackrestore.p0(ptr %13)
  %36 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %36, align 4
  invoke void @__cxa_throw(ptr %36, ptr @_ZTIi, ptr null)
          to label %43 unwind label %37

37:                                               ; preds = %1
  %38 = landingpad { ptr, i32 }
          catch ptr null
  %39 = extractvalue { ptr, i32 } %38, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  %41 = load i1, ptr %3, align 1
  store i1 %41, ptr %5, align 1
  %42 = load i1, ptr %5, align 1
  ret i1 %42

43:                                               ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i1 @printInCatch() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca i1, align 1
  store i1 false, ptr %2, align 1
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Node, ptr %5, i32 0, i32 0
  store ptr @Node..vtbl, ptr %6, align 8
  invoke void @Node.constructor(ptr %5)
          to label %7 unwind label %9

7:                                                ; preds = %0
  store ptr %5, ptr %1, align 8
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %41 unwind label %9

9:                                                ; preds = %7, %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = invoke ptr @GC_malloc_atomic(i64 58)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %9
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = invoke i32 (ptr, i64, ptr, ...) @snprintf(ptr %14, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 5)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %.noexc
  %16 = call ptr @llvm.stacksave.p0()
  %17 = icmp eq ptr %14, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %19 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8))
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %.noexc1
  %20 = add i64 1, %19
  %21 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  %22 = add i64 %20, %21
  %23 = invoke i64 @strlen(ptr %18)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %24 = add i64 %22, %23
  %25 = alloca i8, i64 %24, align 1
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %24, 0
  %28 = invoke ptr @__memcpy_chk(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8), i64 %19, i64 %27)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %24, %29
  %32 = invoke ptr @__memcpy_chk(ptr %30, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %21, i64 %31)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  %35 = sub i64 %24, %33
  %36 = invoke ptr @__memcpy_chk(ptr %34, ptr %18, i64 %23, i64 %35)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %37 = add i64 %33, %23
  %38 = getelementptr i8, ptr %25, i64 %37
  store i8 0, ptr %38, align 1
  invoke void @puts(ptr %25)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  call void @llvm.stackrestore.p0(ptr %16)
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  %39 = load i1, ptr %2, align 1
  store i1 %39, ptr %3, align 1
  %40 = load i1, ptr %3, align 1
  ret i1 %40

41:                                               ; preds = %7
  unreachable

endcatch.resume:                                  ; preds = %.noexc7, %.noexc6, %.noexc5, %.noexc4, %.noexc3, %.noexc2, %.noexc1, %.noexc, %9
  %42 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %42
}

; Function Attrs: mustprogress
define i1 @usingThenThrow() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %5 = call ptr @GC_malloc(i64 16)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  store ptr %6, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %64 unwind label %9

9:                                                ; preds = %0
  %10 = landingpad { ptr, i32 }
          catch ptr null
  %11 = extractvalue { ptr, i32 } %10, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  %13 = load i1, ptr %1, align 1
  br i1 %13, label %14, label %18

14:                                               ; preds = %9
  store i1 false, ptr %1, align 1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr %Res, ptr %15, i32 0, i32 0
  invoke void @Res.dispose(ptr %15)
          to label %17 unwind label %endcatch.pad2

17:                                               ; preds = %14
  br label %18

18:                                               ; preds = %17, %9
  invoke void @__cxa_rethrow()
          to label %64 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %14
  %19 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

20:                                               ; No predecessors!
  %21 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %20
  %22 = phi { ptr, i32 } [ %21, %20 ], [ %19, %endcatch.pad2 ]
  %23 = extractvalue { ptr, i32 } %22, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  %25 = load i1, ptr %1, align 1
  br i1 %25, label %26, label %29

26:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %27 = load ptr, ptr %2, align 8
  %28 = getelementptr %Res, ptr %27, i32 0, i32 0
  invoke void @Res.dispose(ptr %27)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %26
  br label %29

29:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %64 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %29, %18
  %30 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

31:                                               ; No predecessors!
  %32 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %31
  %33 = phi { ptr, i32 } [ %32, %31 ], [ %30, %endcatch.pad ]
  %34 = extractvalue { ptr, i32 } %33, 0
  %35 = call ptr @__cxa_begin_catch(ptr %34)
  %36 = invoke ptr @GC_malloc_atomic(i64 58)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.caught
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = invoke i32 (ptr, i64, ptr, ...) @snprintf(ptr %37, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 6)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %39 = call ptr @llvm.stacksave.p0()
  %40 = icmp eq ptr %37, null
  %41 = select i1 %40, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %37
  %42 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8))
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %43 = add i64 1, %42
  %44 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %45 = add i64 %43, %44
  %46 = invoke i64 @strlen(ptr %41)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %47 = add i64 %45, %46
  %48 = alloca i8, i64 %47, align 1
  %49 = getelementptr i8, ptr %48, i64 0
  %50 = sub i64 %47, 0
  %51 = invoke ptr @__memcpy_chk(ptr %49, ptr getelementptr inbounds nuw (i8, ptr @s_3019566071980655109, i64 8), i64 %42, i64 %50)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %52 = add i64 0, %42
  %53 = getelementptr i8, ptr %48, i64 %52
  %54 = sub i64 %47, %52
  %55 = invoke ptr @__memcpy_chk(ptr %53, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %44, i64 %54)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %56 = add i64 %52, %44
  %57 = getelementptr i8, ptr %48, i64 %56
  %58 = sub i64 %47, %56
  %59 = invoke ptr @__memcpy_chk(ptr %57, ptr %41, i64 %46, i64 %58)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %.noexc9
  %60 = add i64 %56, %46
  %61 = getelementptr i8, ptr %48, i64 %60
  store i8 0, ptr %61, align 1
  invoke void @puts(ptr %48)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  call void @llvm.stackrestore.p0(ptr %39)
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  %62 = load i1, ptr %3, align 1
  store i1 %62, ptr %4, align 1
  %63 = load i1, ptr %4, align 1
  ret i1 %63

64:                                               ; preds = %18, %29, %0
  unreachable

endcatch.resume:                                  ; preds = %.noexc10, %.noexc9, %.noexc8, %.noexc7, %.noexc6, %.noexc5, %.noexc4, %.noexc3, %.caught, %26
  %65 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %65
}

; Function Attrs: mustprogress
define i1 @initializerThrows() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca i1, align 1
  store i1 false, ptr %2, align 1
  %4 = invoke ptr @fail()
          to label %5 unwind label %35

5:                                                ; preds = %0
  store ptr %4, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  %7 = getelementptr %Node, ptr %6, i32 0, i32 1
  %8 = load i32, ptr %7, align 4
  %9 = call ptr @GC_malloc_atomic(i64 58)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %10, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %8)
  %12 = call ptr @llvm.stacksave.p0()
  %13 = icmp eq ptr %10, null
  %14 = select i1 %13, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %10
  %15 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8))
  %16 = add i64 1, %15
  %17 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %18 = add i64 %16, %17
  %19 = call i64 @strlen(ptr %14)
  %20 = add i64 %18, %19
  %21 = alloca i8, i64 %20, align 1
  %22 = getelementptr i8, ptr %21, i64 0
  %23 = sub i64 %20, 0
  %24 = call ptr @__memcpy_chk(ptr %22, ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8), i64 %15, i64 %23)
  %25 = add i64 0, %15
  %26 = getelementptr i8, ptr %21, i64 %25
  %27 = sub i64 %20, %25
  %28 = call ptr @__memcpy_chk(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %17, i64 %27)
  %29 = add i64 %25, %17
  %30 = getelementptr i8, ptr %21, i64 %29
  %31 = sub i64 %20, %29
  %32 = call ptr @__memcpy_chk(ptr %30, ptr %14, i64 %19, i64 %31)
  %33 = add i64 %29, %19
  %34 = getelementptr i8, ptr %21, i64 %33
  store i8 0, ptr %34, align 1
  call void @puts(ptr %21)
  call void @llvm.stackrestore.p0(ptr %12)
  br label %39

35:                                               ; preds = %0
  %36 = landingpad { ptr, i32 }
          catch ptr null
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  br label %39

39:                                               ; preds = %5, %35
  %40 = load i1, ptr %2, align 1
  store i1 %40, ptr %3, align 1
  %41 = load i1, ptr %3, align 1
  ret i1 %41
}

; Function Attrs: mustprogress
define linkonce_odr ptr @"___unbox<!ts.class<Node, !ts.class_storage<Node>>>"(ptr %0) #1 comdat {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
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
  br i1 %19, label %21, label %42

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
  %39 = load ptr, ptr @Node..rtti, align 8
  %40 = call i1 %38(ptr %34, ptr %39)
  br i1 %40, label %41, label %91

41:                                               ; preds = %21
  store ptr %34, ptr %3, align 8
  br label %93

42:                                               ; preds = %20
  %43 = load ptr, ptr %2, align 8
  %44 = getelementptr { i64, ptr, i8 }, ptr %43, i32 0, i32 1
  %45 = load ptr, ptr %44, align 8
  %46 = ptrtoint ptr %45 to i64
  %47 = icmp ne i64 %46, 0
  %48 = zext i1 %47 to i32
  %49 = and i32 %48, 1
  %50 = icmp ne i32 %49, 0
  br i1 %50, label %51, label %54

51:                                               ; preds = %42
  %52 = call i32 @strcmp(ptr %45, ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8))
  %53 = icmp eq i32 %52, 0
  br label %56

54:                                               ; preds = %42
  %55 = icmp eq i64 %46, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_11582289249049393296, i64 8) to i64)
  br label %56

56:                                               ; preds = %51, %54
  %57 = phi i1 [ %55, %54 ], [ %53, %51 ]
  br label %58

58:                                               ; preds = %56
  br i1 %57, label %59, label %91

59:                                               ; preds = %58
  %60 = getelementptr { i64, ptr, { ptr, ptr } }, ptr %4, i32 0, i32 2
  %61 = load { ptr, ptr }, ptr %60, align 8
  %62 = extractvalue { ptr, ptr } %61, 1
  %63 = ptrtoint ptr %62 to i64
  %64 = icmp ne i64 %63, 0
  br i1 %64, label %65, label %72

65:                                               ; preds = %59
  %66 = extractvalue { ptr, ptr } %61, 0
  %67 = extractvalue { ptr, ptr } %61, 1
  %68 = getelementptr ptr, ptr %66, i32 1
  %69 = load ptr, ptr %68, align 8
  %70 = load ptr, ptr @Node..rtti, align 8
  %71 = call i1 %69(ptr %67, ptr %70)
  br label %72

72:                                               ; preds = %65, %59
  %73 = phi i1 [ %71, %65 ], [ false, %59 ]
  br i1 %73, label %74, label %91

74:                                               ; preds = %72
  %75 = extractvalue { ptr, ptr } %61, 1
  %76 = ptrtoint ptr %75 to i64
  %77 = icmp ne i64 %76, 0
  br i1 %77, label %78, label %85

78:                                               ; preds = %74
  %79 = extractvalue { ptr, ptr } %61, 0
  %80 = extractvalue { ptr, ptr } %61, 1
  %81 = getelementptr ptr, ptr %79, i32 1
  %82 = load ptr, ptr %81, align 8
  %83 = load ptr, ptr @Node..rtti, align 8
  %84 = call i1 %82(ptr %80, ptr %83)
  br label %85

85:                                               ; preds = %78, %74
  %86 = phi i1 [ %84, %78 ], [ false, %74 ]
  br i1 %86, label %87, label %89

87:                                               ; preds = %85
  %88 = extractvalue { ptr, ptr } %61, 1
  store ptr %88, ptr %3, align 8
  br label %93

89:                                               ; preds = %85
  %90 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_10002225064126424008, i64 8), ptr %90, align 8
  call void @__cxa_throw(ptr %90, ptr @_ZTIPc, ptr null)
  br label %95

91:                                               ; preds = %21, %72, %58
  %92 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_4346689792895745607, i64 8), ptr %92, align 8
  call void @__cxa_throw(ptr %92, ptr @_ZTIPc, ptr null)
  br label %95

93:                                               ; preds = %41, %87
  %94 = load ptr, ptr %3, align 8
  ret ptr %94

95:                                               ; preds = %91, %89
  unreachable
}

; Function Attrs: mustprogress
define i1 @castThrows() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca i1, align 1
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, i32 }, ptr %5, i32 0, i32 0
  store i64 4, ptr %6, align 8
  %7 = getelementptr { i64, ptr, i32 }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, i32 }, ptr %5, i32 0, i32 2
  store i32 5, ptr %8, align 4
  store i1 false, ptr %2, align 1
  %9 = invoke ptr @"___unbox<!ts.class<Node, !ts.class_storage<Node>>>"(ptr %5)
          to label %10 unwind label %40

10:                                               ; preds = %0
  store ptr %9, ptr %1, align 8
  %11 = load ptr, ptr %1, align 8
  %12 = getelementptr %Node, ptr %11, i32 0, i32 1
  %13 = load i32, ptr %12, align 4
  %14 = call ptr @GC_malloc_atomic(i64 58)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %15, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %13)
  %17 = call ptr @llvm.stacksave.p0()
  %18 = icmp eq ptr %15, null
  %19 = select i1 %18, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %20 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8))
  %21 = add i64 1, %20
  %22 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %23 = add i64 %21, %22
  %24 = call i64 @strlen(ptr %19)
  %25 = add i64 %23, %24
  %26 = alloca i8, i64 %25, align 1
  %27 = getelementptr i8, ptr %26, i64 0
  %28 = sub i64 %25, 0
  %29 = call ptr @__memcpy_chk(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8), i64 %20, i64 %28)
  %30 = add i64 0, %20
  %31 = getelementptr i8, ptr %26, i64 %30
  %32 = sub i64 %25, %30
  %33 = call ptr @__memcpy_chk(ptr %31, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %22, i64 %32)
  %34 = add i64 %30, %22
  %35 = getelementptr i8, ptr %26, i64 %34
  %36 = sub i64 %25, %34
  %37 = call ptr @__memcpy_chk(ptr %35, ptr %19, i64 %24, i64 %36)
  %38 = add i64 %34, %24
  %39 = getelementptr i8, ptr %26, i64 %38
  store i8 0, ptr %39, align 1
  call void @puts(ptr %26)
  call void @llvm.stackrestore.p0(ptr %17)
  br label %44

40:                                               ; preds = %0
  %41 = landingpad { ptr, i32 }
          catch ptr null
  %42 = extractvalue { ptr, i32 } %41, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  br label %44

44:                                               ; preds = %10, %40
  %45 = load i1, ptr %2, align 1
  store i1 %45, ptr %3, align 1
  %46 = load i1, ptr %3, align 1
  ret i1 %46
}

; Function Attrs: mustprogress
define i1 @constControl(ptr %0) #2 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %0, ptr %3, align 8
  store i1 false, ptr %2, align 1
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr %Node, ptr %5, i32 0, i32 1
  %7 = load i32, ptr %6, align 4
  %8 = call ptr @GC_malloc_atomic(i64 58)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %9, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %7)
  %11 = call ptr @llvm.stacksave.p0()
  %12 = icmp eq ptr %9, null
  %13 = select i1 %12, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %9
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12515944260747508003, i64 8))
  %15 = add i64 1, %14
  %16 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %17 = add i64 %15, %16
  %18 = call i64 @strlen(ptr %13)
  %19 = add i64 %17, %18
  %20 = alloca i8, i64 %19, align 1
  %21 = getelementptr i8, ptr %20, i64 0
  %22 = sub i64 %19, 0
  %23 = call ptr @__memcpy_chk(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_12515944260747508003, i64 8), i64 %14, i64 %22)
  %24 = add i64 0, %14
  %25 = getelementptr i8, ptr %20, i64 %24
  %26 = sub i64 %19, %24
  %27 = call ptr @__memcpy_chk(ptr %25, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %16, i64 %26)
  %28 = add i64 %24, %16
  %29 = getelementptr i8, ptr %20, i64 %28
  %30 = sub i64 %19, %28
  %31 = call ptr @__memcpy_chk(ptr %29, ptr %13, i64 %18, i64 %30)
  %32 = add i64 %28, %18
  %33 = getelementptr i8, ptr %20, i64 %32
  store i8 0, ptr %33, align 1
  call void @puts(ptr %20)
  call void @llvm.stackrestore.p0(ptr %11)
  %34 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %34, align 4
  invoke void @__cxa_throw(ptr %34, ptr @_ZTIi, ptr null)
          to label %41 unwind label %35

35:                                               ; preds = %1
  %36 = landingpad { ptr, i32 }
          catch ptr null
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  %39 = load i1, ptr %2, align 1
  store i1 %39, ptr %4, align 1
  %40 = load i1, ptr %4, align 1
  ret i1 %40

41:                                               ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca <{ i64, %Node }>, align 16
  %3 = alloca <{ i64, %Node }>, align 16
  %4 = alloca <{ i64, %Node }>, align 16
  store i64 -1, ptr %4, align 8
  %5 = getelementptr i8, ptr %4, i64 8
  store %Node zeroinitializer, ptr %5, align 8
  store i64 -1, ptr %3, align 8
  %6 = getelementptr i8, ptr %3, i64 8
  store %Node zeroinitializer, ptr %6, align 8
  store i64 -1, ptr %2, align 8
  %7 = getelementptr i8, ptr %2, i64 8
  store %Node zeroinitializer, ptr %7, align 8
  store %Node zeroinitializer, ptr %7, align 8
  %8 = getelementptr %Node, ptr %7, i32 0, i32 0
  store ptr @Node..vtbl, ptr %8, align 8
  call void @Node.constructor(ptr %7)
  %9 = call i1 @letThenThrow(ptr %7)
  br i1 %9, label %10, label %33

10:                                               ; preds = %0
  %11 = call i1 @printInCatch()
  br i1 %11, label %12, label %35

12:                                               ; preds = %10
  %13 = call i1 @usingThenThrow()
  br i1 %13, label %14, label %37

14:                                               ; preds = %12
  %15 = call i1 @initializerThrows()
  br i1 %15, label %16, label %39

16:                                               ; preds = %14
  %17 = call i1 @castThrows()
  br i1 %17, label %18, label %41

18:                                               ; preds = %16
  store %Node zeroinitializer, ptr %6, align 8
  %19 = getelementptr %Node, ptr %6, i32 0, i32 0
  store ptr @Node..vtbl, ptr %19, align 8
  call void @Node.constructor(ptr %6)
  %20 = call i1 @constControl(ptr %6)
  br i1 %20, label %21, label %43

21:                                               ; preds = %18
  store i32 0, ptr %1, align 4
  br label %22

22:                                               ; preds = %28, %21
  %23 = load i32, ptr %1, align 4
  %24 = icmp slt i32 %23, 100
  br i1 %24, label %25, label %31

25:                                               ; preds = %22
  store %Node zeroinitializer, ptr %5, align 8
  %26 = getelementptr %Node, ptr %5, i32 0, i32 0
  store ptr @Node..vtbl, ptr %26, align 8
  call void @Node.constructor(ptr %5)
  %27 = call i1 @letThenThrow(ptr %5)
  br i1 %27, label %28, label %45

28:                                               ; preds = %25
  %29 = load i32, ptr %1, align 4
  %30 = add i32 %29, 1
  store i32 %30, ptr %1, align 4
  br label %22

31:                                               ; preds = %22
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

32:                                               ; preds = %45, %43, %41, %39, %37, %35, %33
  unreachable

33:                                               ; preds = %0
  %34 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7316157474859157787, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 97, ptr null)
  br label %32

35:                                               ; preds = %10
  %36 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_383063154153552527, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 98, ptr null)
  br label %32

37:                                               ; preds = %12
  %38 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5248496707480371780, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 99, ptr null)
  br label %32

39:                                               ; preds = %14
  %40 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18314962778420153713, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 100, ptr null)
  br label %32

41:                                               ; preds = %16
  %42 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10581538897204661568, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 101, ptr null)
  br label %32

43:                                               ; preds = %18
  %44 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12074018024437361393, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 102, ptr null)
  br label %32

45:                                               ; preds = %25
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_9989496759544712621, i64 8), i32 105, ptr null)
  br label %32
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

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #3

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }
attributes #3 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

