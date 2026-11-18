; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr, ptr }

$Res..size = comdat any

@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_12638116831811436492 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF!\00", align 8
@s_12638208091276578005 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFr\00", align 8
@s_12638190499090526629 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb\00", align 8
@s_12638189399578898418 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_622146593055446729 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFca\00", align 8
@s_623171337892750156 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFb!\00", align 8
@s_638546908498625180 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFr!\00", align 8
@m_2465139471050184658 = internal constant [69 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa using nested in an if inside a try body disposes on unwind\00", align 8
@f_15410362430848042469 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00using_nested_scopes.ts\00", align 8
@m_15716656525873847192 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFand nothing runs when that branch is not taken\00", align 8
@m_16056805844513512544 = internal constant [41 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe same for a bare nested block\00", align 8
@m_15962717650171491568 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe throw must reach the caller\00", align 8
@m_4585685177048270522 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner disposes at its scope exit, outer on the unwind\00", align 8
@m_1091598629432840353 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFboth dispose on the unwind, innermost first\00", align 8
@disposed = private global ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 16, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }
@_ZTIi = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_end_catch()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #1

declare i64 @strlen(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Res..instanceOf(ptr %0, ptr %1) #2 {
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
define ptr @Res..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @disposed, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %3, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  %9 = icmp eq ptr %6, null
  %10 = select i1 %9, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %11 = call i64 @strlen(ptr %8)
  %12 = add i64 1, %11
  %13 = call i64 @strlen(ptr %10)
  %14 = add i64 %12, %13
  %15 = add i64 %14, 8
  %16 = call ptr @GC_malloc(i64 %15)
  %17 = getelementptr i8, ptr %16, i64 8
  %18 = getelementptr i8, ptr %17, i64 0
  %19 = sub i64 %14, 0
  %20 = call ptr @__memcpy_chk(ptr %18, ptr %8, i64 %11, i64 %19)
  %21 = add i64 0, %11
  %22 = getelementptr i8, ptr %17, i64 %21
  %23 = sub i64 %14, %21
  %24 = call ptr @__memcpy_chk(ptr %22, ptr %10, i64 %13, i64 %23)
  %25 = add i64 %21, %13
  %26 = getelementptr i8, ptr %17, i64 %25
  store i8 0, ptr %26, align 1
  store ptr %17, ptr @disposed, align 8
  ret void
}

; Function Attrs: mustprogress
define void @nestedInIf(i1 %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i1, align 1
  store i1 %0, ptr %5, align 1
  %6 = load i1, ptr %5, align 1
  br i1 %6, label %7, label %67

7:                                                ; preds = %1
  store i1 false, ptr %2, align 1
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Res, ptr %9, i32 0, i32 0
  store ptr @Res..vtbl, ptr %10, align 8
  invoke void @Res.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_12638208091276578005, i64 8))
          to label %11 unwind label %13

11:                                               ; preds = %7
  store ptr %9, ptr %3, align 8
  store i1 true, ptr %2, align 1
  %12 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %12, align 4
  invoke void @__cxa_throw(ptr %12, ptr @_ZTIi, ptr null)
          to label %68 unwind label %13

13:                                               ; preds = %11, %7
  %14 = landingpad { ptr, i32 }
          catch ptr null
  %15 = extractvalue { ptr, i32 } %14, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i1, ptr %2, align 1
  br i1 %17, label %18, label %22

18:                                               ; preds = %13
  store i1 false, ptr %2, align 1
  %19 = load ptr, ptr %3, align 8
  %20 = getelementptr %Res, ptr %19, i32 0, i32 0
  invoke void @Res.dispose(ptr %19)
          to label %21 unwind label %endcatch.pad2

21:                                               ; preds = %18
  br label %22

22:                                               ; preds = %21, %13
  invoke void @__cxa_rethrow()
          to label %68 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %18
  %23 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

24:                                               ; No predecessors!
  %25 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %24
  %26 = phi { ptr, i32 } [ %25, %24 ], [ %23, %endcatch.pad2 ]
  %27 = extractvalue { ptr, i32 } %26, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i1, ptr %2, align 1
  br i1 %29, label %30, label %33

30:                                               ; preds = %.caught1
  store i1 false, ptr %2, align 1
  %31 = load ptr, ptr %3, align 8
  %32 = getelementptr %Res, ptr %31, i32 0, i32 0
  invoke void @Res.dispose(ptr %31)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %30
  br label %33

33:                                               ; preds = %.noexc7, %.caught1
  invoke void @__cxa_rethrow()
          to label %68 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %22, %33
  %34 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

35:                                               ; No predecessors!
  %36 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %35
  %37 = phi { ptr, i32 } [ %36, %35 ], [ %34, %endcatch.pad ]
  %38 = extractvalue { ptr, i32 } %37, 1
  %39 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %40 = icmp eq i32 %38, %39
  br i1 %40, label %44, label %41

41:                                               ; preds = %.caught
  %42 = extractvalue { ptr, i32 } %37, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  invoke void @__cxa_rethrow()
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %41
  br label %68

44:                                               ; preds = %.caught
  %45 = extractvalue { ptr, i32 } %37, 0
  %46 = call ptr @__cxa_begin_catch(ptr %45)
  %47 = load i32, ptr %46, align 4
  store i32 %47, ptr %4, align 4
  %48 = load ptr, ptr @disposed, align 8
  %49 = icmp eq ptr %48, null
  %50 = select i1 %49, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %48
  %51 = invoke i64 @strlen(ptr %50)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %44
  %52 = add i64 1, %51
  %53 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8))
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc
  %54 = add i64 %52, %53
  %55 = add i64 %54, 8
  %56 = invoke ptr @GC_malloc(i64 %55)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %57 = getelementptr i8, ptr %56, i64 8
  %58 = getelementptr i8, ptr %57, i64 0
  %59 = sub i64 %54, 0
  %60 = invoke ptr @__memcpy_chk(ptr %58, ptr %50, i64 %51, i64 %59)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %61 = add i64 0, %51
  %62 = getelementptr i8, ptr %57, i64 %61
  %63 = sub i64 %54, %61
  %64 = invoke ptr @__memcpy_chk(ptr %62, ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8), i64 %53, i64 %63)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %65 = add i64 %61, %53
  %66 = getelementptr i8, ptr %57, i64 %65
  store i8 0, ptr %66, align 1
  store ptr %57, ptr @disposed, align 8
  call void @__cxa_end_catch()
  br label %67

67:                                               ; preds = %.noexc6, %1
  ret void

68:                                               ; preds = %.noexc8, %22, %33, %11
  unreachable

endcatch.resume:                                  ; preds = %41, %30, %.noexc5, %.noexc4, %.noexc3, %.noexc, %44
  %69 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %69
}

; Function Attrs: mustprogress
define void @nestedInBlock() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i1 false, ptr %1, align 1
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr %Res, ptr %5, i32 0, i32 0
  store ptr @Res..vtbl, ptr %6, align 8
  invoke void @Res.constructor(ptr %5, ptr getelementptr inbounds nuw (i8, ptr @s_12638190499090526629, i64 8))
          to label %7 unwind label %9

7:                                                ; preds = %0
  store ptr %5, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %8 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %8, align 4
  invoke void @__cxa_throw(ptr %8, ptr @_ZTIi, ptr null)
          to label %63 unwind label %9

9:                                                ; preds = %7, %0
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
          to label %63 unwind label %endcatch.pad

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
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %26
  br label %29

29:                                               ; preds = %.noexc8, %.caught1
  invoke void @__cxa_rethrow()
          to label %63 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %29, %18
  %30 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

31:                                               ; No predecessors!
  %32 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %31
  %33 = phi { ptr, i32 } [ %32, %31 ], [ %30, %endcatch.pad ]
  %34 = extractvalue { ptr, i32 } %33, 1
  %35 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %36 = icmp eq i32 %34, %35
  br i1 %36, label %40, label %37

37:                                               ; preds = %.caught
  %38 = extractvalue { ptr, i32 } %33, 0
  %39 = call ptr @__cxa_begin_catch(ptr %38)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %37
  br label %63

40:                                               ; preds = %.caught
  %41 = extractvalue { ptr, i32 } %33, 0
  %42 = call ptr @__cxa_begin_catch(ptr %41)
  %43 = load i32, ptr %42, align 4
  store i32 %43, ptr %3, align 4
  %44 = load ptr, ptr @disposed, align 8
  %45 = icmp eq ptr %44, null
  %46 = select i1 %45, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %44
  %47 = invoke i64 @strlen(ptr %46)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %40
  %48 = add i64 1, %47
  %49 = invoke i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8))
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %.noexc3
  %50 = add i64 %48, %49
  %51 = add i64 %50, 8
  %52 = invoke ptr @GC_malloc(i64 %51)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %.noexc4
  %53 = getelementptr i8, ptr %52, i64 8
  %54 = getelementptr i8, ptr %53, i64 0
  %55 = sub i64 %50, 0
  %56 = invoke ptr @__memcpy_chk(ptr %54, ptr %46, i64 %47, i64 %55)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %57 = add i64 0, %47
  %58 = getelementptr i8, ptr %53, i64 %57
  %59 = sub i64 %50, %57
  %60 = invoke ptr @__memcpy_chk(ptr %58, ptr getelementptr inbounds nuw (i8, ptr @s_12638116831811436492, i64 8), i64 %49, i64 %59)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %61 = add i64 %57, %49
  %62 = getelementptr i8, ptr %53, i64 %61
  store i8 0, ptr %62, align 1
  store ptr %53, ptr @disposed, align 8
  call void @__cxa_end_catch()
  ret void

63:                                               ; preds = %.noexc, %18, %29, %7
  unreachable

endcatch.resume:                                  ; preds = %26, %.noexc6, %.noexc5, %.noexc4, %.noexc3, %40, %37
  %64 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %64
}

; Function Attrs: mustprogress
define void @outerAndInner() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 false, ptr %3, align 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  invoke void @Res.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %8 unwind label %42

8:                                                ; preds = %0
  store ptr %6, ptr %4, align 8
  store i1 true, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %9 = call ptr @GC_malloc(i64 24)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %Res, ptr %10, i32 0, i32 0
  store ptr @Res..vtbl, ptr %11, align 8
  invoke void @Res.constructor(ptr %10, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
          to label %12 unwind label %20

12:                                               ; preds = %8
  store ptr %10, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %13 = load i1, ptr %1, align 1
  br i1 %13, label %14, label %18

14:                                               ; preds = %12
  store i1 false, ptr %1, align 1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr %Res, ptr %15, i32 0, i32 0
  invoke void @Res.dispose(ptr %15)
          to label %17 unwind label %20

17:                                               ; preds = %14
  br label %18

18:                                               ; preds = %17, %12
  %19 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %19, align 4
  invoke void @__cxa_throw(ptr %19, ptr @_ZTIi, ptr null)
          to label %50 unwind label %42

20:                                               ; preds = %14, %8
  %21 = landingpad { ptr, i32 }
          catch ptr null
  %22 = extractvalue { ptr, i32 } %21, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22)
  %24 = load i1, ptr %1, align 1
  br i1 %24, label %25, label %29

25:                                               ; preds = %20
  store i1 false, ptr %1, align 1
  %26 = load ptr, ptr %2, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %28 unwind label %endcatch.pad

28:                                               ; preds = %25
  br label %29

29:                                               ; preds = %28, %20
  invoke void @__cxa_rethrow()
          to label %50 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %25
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
  %36 = load i1, ptr %1, align 1
  br i1 %36, label %37, label %40

37:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %38 = load ptr, ptr %2, align 8
  %39 = getelementptr %Res, ptr %38, i32 0, i32 0
  invoke void @Res.dispose(ptr %38)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %37
  br label %40

40:                                               ; preds = %.noexc, %.caught
  invoke void @__cxa_rethrow()
          to label %50 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %29, %40
  %41 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught1

42:                                               ; preds = %18, %0
  %43 = landingpad { ptr, i32 }
          cleanup
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %42
  %44 = phi { ptr, i32 } [ %43, %42 ], [ %41, %endcatch.pad2 ]
  %45 = load i1, ptr %3, align 1
  br i1 %45, label %46, label %49

46:                                               ; preds = %.caught1
  store i1 false, ptr %3, align 1
  %47 = load ptr, ptr %4, align 8
  %48 = getelementptr %Res, ptr %47, i32 0, i32 0
  call void @Res.dispose(ptr %47)
  br label %49

49:                                               ; preds = %46, %.caught1
  resume { ptr, i32 } %44

50:                                               ; preds = %18, %29, %40
  unreachable

endcatch.resume:                                  ; preds = %37
  %51 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %51
}

; Function Attrs: mustprogress
define void @outerAndInnerBothLive() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 false, ptr %3, align 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  invoke void @Res.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %8 unwind label %36

8:                                                ; preds = %0
  store ptr %6, ptr %4, align 8
  store i1 true, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %9 = call ptr @GC_malloc(i64 24)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %Res, ptr %10, i32 0, i32 0
  store ptr @Res..vtbl, ptr %11, align 8
  invoke void @Res.constructor(ptr %10, ptr getelementptr inbounds nuw (i8, ptr @s_12638189399578898418, i64 8))
          to label %12 unwind label %14

12:                                               ; preds = %8
  store ptr %10, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %13 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %13, align 4
  invoke void @__cxa_throw(ptr %13, ptr @_ZTIi, ptr null)
          to label %44 unwind label %14

14:                                               ; preds = %12, %8
  %15 = landingpad { ptr, i32 }
          catch ptr null
  %16 = extractvalue { ptr, i32 } %15, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = load i1, ptr %1, align 1
  br i1 %18, label %19, label %23

19:                                               ; preds = %14
  store i1 false, ptr %1, align 1
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr %Res, ptr %20, i32 0, i32 0
  invoke void @Res.dispose(ptr %20)
          to label %22 unwind label %endcatch.pad2

22:                                               ; preds = %19
  br label %23

23:                                               ; preds = %22, %14
  invoke void @__cxa_rethrow()
          to label %44 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %19
  %24 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

25:                                               ; No predecessors!
  %26 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %25
  %27 = phi { ptr, i32 } [ %26, %25 ], [ %24, %endcatch.pad2 ]
  %28 = extractvalue { ptr, i32 } %27, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  %30 = load i1, ptr %1, align 1
  br i1 %30, label %31, label %34

31:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %32 = load ptr, ptr %2, align 8
  %33 = getelementptr %Res, ptr %32, i32 0, i32 0
  invoke void @Res.dispose(ptr %32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %31
  br label %34

34:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %44 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %34, %23
  %35 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

36:                                               ; preds = %0
  %37 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %36
  %38 = phi { ptr, i32 } [ %37, %36 ], [ %35, %endcatch.pad ]
  %39 = load i1, ptr %3, align 1
  br i1 %39, label %40, label %43

40:                                               ; preds = %.caught
  store i1 false, ptr %3, align 1
  %41 = load ptr, ptr %4, align 8
  %42 = getelementptr %Res, ptr %41, i32 0, i32 0
  call void @Res.dispose(ptr %41)
  br label %43

43:                                               ; preds = %40, %.caught
  resume { ptr, i32 } %38

44:                                               ; preds = %23, %34, %12
  unreachable

endcatch.resume:                                  ; preds = %31
  %45 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %45
}

; Function Attrs: mustprogress
define i1 @caught({ ptr, ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca i32, align 4
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca i1, align 1
  store { ptr, ptr, ptr } %0, ptr %3, align 8
  %5 = load { ptr, ptr, ptr }, ptr %3, align 8
  %6 = extractvalue { ptr, ptr, ptr } %5, 1
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  br i1 %8, label %9, label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %10(ptr %6)
          to label %11 unwind label %18

11:                                               ; preds = %9
  br label %15

12:                                               ; preds = %1
  %13 = extractvalue { ptr, ptr, ptr } %5, 0
  invoke void %13()
          to label %14 unwind label %18

14:                                               ; preds = %12
  br label %15

15:                                               ; preds = %11, %14
  br label %16

16:                                               ; preds = %15
  br label %17

17:                                               ; preds = %16
  store i1 false, ptr %4, align 1
  br label %30

18:                                               ; preds = %9, %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %23
  br label %32

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %2, align 4
  store i1 true, ptr %4, align 1
  call void @__cxa_end_catch()
  br label %30

30:                                               ; preds = %17, %26
  %31 = load i1, ptr %4, align 1
  ret i1 %31

32:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %23
  %33 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %33
}

; Function Attrs: mustprogress
define void @.f_main..afL104C19FH2011477099() #2 {
  call void @outerAndInner()
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..afL108C19FH2011477099() #2 {
  call void @outerAndInnerBothLive()
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @disposed, align 8
  call void @nestedInIf(i1 true)
  %1 = load ptr, ptr @disposed, align 8
  %2 = ptrtoint ptr %1 to i64
  %3 = icmp ne i64 %2, 0
  %4 = zext i1 %3 to i32
  %5 = and i32 %4, 1
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %0
  %8 = call i32 @strcmp(ptr %1, ptr getelementptr inbounds nuw (i8, ptr @s_638546908498625180, i64 8))
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %0
  %11 = icmp eq i64 %2, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_638546908498625180, i64 8) to i64)
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i1 [ %11, %10 ], [ %9, %7 ]
  br label %14

14:                                               ; preds = %12
  br i1 %13, label %15, label %81

15:                                               ; preds = %14
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @disposed, align 8
  call void @nestedInIf(i1 false)
  %16 = load ptr, ptr @disposed, align 8
  %17 = ptrtoint ptr %16 to i64
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i32
  %20 = and i32 %19, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %15
  %23 = call i32 @strcmp(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %24 = icmp eq i32 %23, 0
  br label %27

25:                                               ; preds = %15
  %26 = icmp eq i64 %17, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8) to i64)
  br label %27

27:                                               ; preds = %22, %25
  %28 = phi i1 [ %26, %25 ], [ %24, %22 ]
  br label %29

29:                                               ; preds = %27
  br i1 %28, label %30, label %83

30:                                               ; preds = %29
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @disposed, align 8
  call void @nestedInBlock()
  %31 = load ptr, ptr @disposed, align 8
  %32 = ptrtoint ptr %31 to i64
  %33 = icmp ne i64 %32, 0
  %34 = zext i1 %33 to i32
  %35 = and i32 %34, 1
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %40

37:                                               ; preds = %30
  %38 = call i32 @strcmp(ptr %31, ptr getelementptr inbounds nuw (i8, ptr @s_623171337892750156, i64 8))
  %39 = icmp eq i32 %38, 0
  br label %42

40:                                               ; preds = %30
  %41 = icmp eq i64 %32, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_623171337892750156, i64 8) to i64)
  br label %42

42:                                               ; preds = %37, %40
  %43 = phi i1 [ %41, %40 ], [ %39, %37 ]
  br label %44

44:                                               ; preds = %42
  br i1 %43, label %45, label %85

45:                                               ; preds = %44
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @disposed, align 8
  %46 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL104C19FH2011477099, ptr null, ptr null })
  br i1 %46, label %47, label %87

47:                                               ; preds = %45
  %48 = load ptr, ptr @disposed, align 8
  %49 = ptrtoint ptr %48 to i64
  %50 = icmp ne i64 %49, 0
  %51 = zext i1 %50 to i32
  %52 = and i32 %51, 1
  %53 = icmp ne i32 %52, 0
  br i1 %53, label %54, label %57

54:                                               ; preds = %47
  %55 = call i32 @strcmp(ptr %48, ptr getelementptr inbounds nuw (i8, ptr @s_622146593055446729, i64 8))
  %56 = icmp eq i32 %55, 0
  br label %59

57:                                               ; preds = %47
  %58 = icmp eq i64 %49, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_622146593055446729, i64 8) to i64)
  br label %59

59:                                               ; preds = %54, %57
  %60 = phi i1 [ %58, %57 ], [ %56, %54 ]
  br label %61

61:                                               ; preds = %59
  br i1 %60, label %62, label %89

62:                                               ; preds = %61
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @disposed, align 8
  %63 = call i1 @caught({ ptr, ptr, ptr } { ptr @.f_main..afL108C19FH2011477099, ptr null, ptr null })
  br i1 %63, label %64, label %91

64:                                               ; preds = %62
  %65 = load ptr, ptr @disposed, align 8
  %66 = ptrtoint ptr %65 to i64
  %67 = icmp ne i64 %66, 0
  %68 = zext i1 %67 to i32
  %69 = and i32 %68, 1
  %70 = icmp ne i32 %69, 0
  br i1 %70, label %71, label %74

71:                                               ; preds = %64
  %72 = call i32 @strcmp(ptr %65, ptr getelementptr inbounds nuw (i8, ptr @s_622146593055446729, i64 8))
  %73 = icmp eq i32 %72, 0
  br label %76

74:                                               ; preds = %64
  %75 = icmp eq i64 %66, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_622146593055446729, i64 8) to i64)
  br label %76

76:                                               ; preds = %71, %74
  %77 = phi i1 [ %75, %74 ], [ %73, %71 ]
  br label %78

78:                                               ; preds = %76
  br i1 %77, label %79, label %93

79:                                               ; preds = %78
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

80:                                               ; preds = %93, %91, %89, %87, %85, %83, %81
  unreachable

81:                                               ; preds = %14
  %82 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2465139471050184658, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 93, ptr null)
  br label %80

83:                                               ; preds = %29
  %84 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15716656525873847192, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 97, ptr null)
  br label %80

85:                                               ; preds = %44
  %86 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16056805844513512544, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 101, ptr null)
  br label %80

87:                                               ; preds = %45
  %88 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15962717650171491568, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 104, ptr null)
  br label %80

89:                                               ; preds = %61
  %90 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4585685177048270522, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 105, ptr null)
  br label %80

91:                                               ; preds = %62
  %92 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15962717650171491568, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 108, ptr null)
  br label %80

93:                                               ; preds = %78
  %94 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1091598629432840353, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15410362430848042469, i64 8), i32 109, ptr null)
  br label %80
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

attributes #0 = { nounwind memory(none) }
attributes #1 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

