; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Thrower = type { ptr }
%Named = type { ptr, ptr }
%NamedThrower = type { ptr, ptr }

$Thrower..size = comdat any

$Named..size = comdat any

$NamedThrower..size = comdat any

@s_16799820757937681572 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFThrower\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_7786720896912775014 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNamed\00", align 8
@s_11028315291352296977 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNamedThrower\00", align 8
@s_17753883700566607152 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfrom dispose\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_6022810607854959919 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfrom \00", align 8
@s_9932668635479743809 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfirst\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_3263991042789642887 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner\00", align 8
@s_12638123428881205758 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF/\00", align 8
@s_12638225683462629381 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFB\00", align 8
@s_12638187200555641996 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa\00", align 8
@s_12638222384927744748 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFA\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_301695387809324849 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfirst/a\00", align 8
@s_5427783902871050770 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfrom A/BA\00", align 8
@s_15973155909069855405 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfrom B/Ba\00", align 8
@s_5655230408898239064 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF012\00", align 8
@s_9805880408515592371 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfrom dispose1\00", align 8
@m_11910962264726028670 = internal constant [56 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthis try's catch takes the disposal's exception\00", align 8
@f_4766547723502998293 = internal constant [112 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00using_dispose_throws.ts\00", align 8
@m_568347090281078248 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe finally runs and the disposal's exception goes on\00", align 8
@m_718719723438898844 = internal constant [63 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested block's cleanup hands on to the try around it\00", align 8
@m_13865098954200123741 = internal constant [52 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe caller catches the disposal's exception\00", align 8
@m_115703159137015044 = internal constant [45 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa catch of another type passes it on\00", align 8
@m_3960696774835903884 = internal constant [45 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFevery iteration's disposal is caught\00", align 8
@m_9465974079335641511 = internal constant [78 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe disposal's exception replaces the first, and the rest is disposed\00", align 8
@m_6742539089612662085 = internal constant [72 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa second disposal that throws arrives, with everything disposed\00", align 8
@m_7386768615681113615 = internal constant [57 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFon the normal path the cleanup disposes the rest\00", align 8
@m_8345799095647167594 = internal constant [65 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa disposal that does not throw keeps the first exception\00", align 8
@Thrower..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_16799820757937681572, i64 8)
@Thrower..size = global i64 8, comdat
@_ZTIPc = external constant ptr
@Thrower..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Thrower..instanceOf, ptr @Thrower..new, ptr @Thrower.dispose, ptr @Thrower..rtti, ptr @Thrower..size }
@trail = private global ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
@Named..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7786720896912775014, i64 8)
@Named..size = global i64 16, comdat
@Named..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Named..instanceOf, ptr @Named..new, ptr @Named.dispose, ptr @Named..rtti, ptr @Named..size }
@NamedThrower..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11028315291352296977, i64 8)
@NamedThrower..size = global i64 16, comdat
@NamedThrower..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @NamedThrower..instanceOf, ptr @NamedThrower..new, ptr @NamedThrower.dispose, ptr @NamedThrower..rtti, ptr @NamedThrower..size }
@_ZTIi = external constant ptr
@_ZTId = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare ptr @__cxa_current_exception_type()

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @__cxa_end_catch()

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define i1 @Thrower..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Thrower..rtti, align 8
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
define ptr @Thrower..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Thrower, ptr %3, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Thrower.dispose(ptr %0) #2 {
  %2 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8), ptr %2, align 8
  call void @__cxa_throw(ptr %2, ptr @_ZTIPc, ptr null)
  br label %3

3:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i1 @Named..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Named..rtti, align 8
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
define ptr @Named..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Named, ptr %3, i32 0, i32 0
  store ptr @Named..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Named.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Named, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @Named.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @trail, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Named, ptr %4, i32 0, i32 1
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
  store ptr %17, ptr @trail, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @NamedThrower..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @NamedThrower..rtti, align 8
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
define ptr @NamedThrower..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %NamedThrower, ptr %3, i32 0, i32 0
  store ptr @NamedThrower..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @NamedThrower.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %NamedThrower, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @NamedThrower.dispose(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @trail, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %NamedThrower, ptr %4, i32 0, i32 1
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
  store ptr %17, ptr @trail, align 8
  %27 = load ptr, ptr %2, align 8
  %28 = getelementptr %NamedThrower, ptr %27, i32 0, i32 1
  %29 = load ptr, ptr %28, align 8
  %30 = icmp eq ptr %29, null
  %31 = select i1 %30, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %29
  %32 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_6022810607854959919, i64 8))
  %33 = add i64 1, %32
  %34 = call i64 @strlen(ptr %31)
  %35 = add i64 %33, %34
  %36 = add i64 %35, 8
  %37 = call ptr @GC_malloc(i64 %36)
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr i8, ptr %38, i64 0
  %40 = sub i64 %35, 0
  %41 = call ptr @__memcpy_chk(ptr %39, ptr getelementptr inbounds nuw (i8, ptr @s_6022810607854959919, i64 8), i64 %32, i64 %40)
  %42 = add i64 0, %32
  %43 = getelementptr i8, ptr %38, i64 %42
  %44 = sub i64 %35, %42
  %45 = call ptr @__memcpy_chk(ptr %43, ptr %31, i64 %34, i64 %44)
  %46 = add i64 %42, %34
  %47 = getelementptr i8, ptr %38, i64 %46
  store i8 0, ptr %47, align 1
  %48 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %38, ptr %48, align 8
  call void @__cxa_throw(ptr %48, ptr @_ZTIPc, ptr null)
  br label %49

49:                                               ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define ptr @inTryCatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %4, align 8
  store i1 false, ptr %1, align 1
  %6 = call ptr @GC_malloc(i64 16)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Thrower, ptr %7, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %8, align 8
  store ptr %7, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %9, align 8
  invoke void @__cxa_throw(ptr %9, ptr @_ZTIPc, ptr null)
          to label %51 unwind label %10

10:                                               ; preds = %0
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i1, ptr %1, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %10
  store i1 false, ptr %1, align 1
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Thrower, ptr %16, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %16)
          to label %18 unwind label %endcatch.pad

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %10
  invoke void @__cxa_rethrow()
          to label %51 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %15
  %20 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

21:                                               ; No predecessors!
  %22 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %21
  %23 = phi { ptr, i32 } [ %22, %21 ], [ %20, %endcatch.pad ]
  %24 = extractvalue { ptr, i32 } %23, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  %26 = load i1, ptr %1, align 1
  br i1 %26, label %27, label %30

27:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %28 = load ptr, ptr %2, align 8
  %29 = getelementptr %Thrower, ptr %28, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %28)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %27
  br label %30

30:                                               ; preds = %.noexc, %.caught
  invoke void @__cxa_rethrow()
          to label %51 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %30, %19
  %31 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

32:                                               ; No predecessors!
  %33 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %32
  %34 = phi { ptr, i32 } [ %33, %32 ], [ %31, %endcatch.pad2 ]
  %35 = extractvalue { ptr, i32 } %34, 1
  %36 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %37 = icmp eq i32 %35, %36
  br i1 %37, label %41, label %38

38:                                               ; preds = %.caught1
  %39 = extractvalue { ptr, i32 } %34, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  invoke void @__cxa_rethrow()
          to label %51 unwind label %endcatch.pad4

41:                                               ; preds = %.caught1
  %42 = extractvalue { ptr, i32 } %34, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  store ptr %43, ptr %3, align 8
  %44 = load ptr, ptr %3, align 8
  store ptr %44, ptr %4, align 8
  call void @__cxa_end_catch()
  %45 = load ptr, ptr %4, align 8
  store ptr %45, ptr %5, align 8
  %46 = load ptr, ptr %5, align 8
  ret ptr %46

endcatch.pad4:                                    ; preds = %38
  %47 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

48:                                               ; No predecessors!
  %49 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %48
  %50 = phi { ptr, i32 } [ %49, %48 ], [ %47, %endcatch.pad4 ]
  resume { ptr, i32 } %50

51:                                               ; preds = %38, %19, %30, %0
  unreachable

endcatch.resume:                                  ; preds = %27
  %52 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %52
}

; Function Attrs: mustprogress
define ptr @withFinallyOnly() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %5, align 8
  store i32 0, ptr %4, align 4
  store i1 false, ptr %1, align 1
  %7 = call ptr @GC_malloc(i64 16)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Thrower, ptr %8, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %9, align 8
  store ptr %8, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %10, align 8
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIPc, ptr null)
          to label %82 unwind label %11

11:                                               ; preds = %0
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i1, ptr %1, align 1
  br i1 %15, label %16, label %20

16:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr %Thrower, ptr %17, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %17)
          to label %19 unwind label %endcatch.pad

19:                                               ; preds = %16
  br label %20

20:                                               ; preds = %19, %11
  invoke void @__cxa_rethrow()
          to label %82 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %16
  %21 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

22:                                               ; No predecessors!
  %23 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad ]
  %25 = extractvalue { ptr, i32 } %24, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = load i1, ptr %1, align 1
  br i1 %27, label %28, label %31

28:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %29 = load ptr, ptr %2, align 8
  %30 = getelementptr %Thrower, ptr %29, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %29)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  br label %31

31:                                               ; preds = %.noexc, %.caught
  invoke void @__cxa_rethrow()
          to label %82 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %31, %20
  %32 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

33:                                               ; No predecessors!
  %34 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %33
  %35 = phi { ptr, i32 } [ %34, %33 ], [ %32, %endcatch.pad2 ]
  %36 = extractvalue { ptr, i32 } %35, 0
  %37 = call ptr @__cxa_begin_catch(ptr %36)
  store i32 1, ptr %4, align 4
  invoke void @__cxa_rethrow()
          to label %82 unwind label %endcatch.pad4

endcatch.pad4:                                    ; preds = %.caught1
  %38 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

39:                                               ; No predecessors!
  %40 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %39
  %41 = phi { ptr, i32 } [ %40, %39 ], [ %38, %endcatch.pad4 ]
  %42 = extractvalue { ptr, i32 } %41, 1
  %43 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %44 = icmp eq i32 %42, %43
  br i1 %44, label %48, label %45

45:                                               ; preds = %.caught3
  %46 = extractvalue { ptr, i32 } %41, 0
  %47 = call ptr @__cxa_begin_catch(ptr %46)
  invoke void @__cxa_rethrow()
          to label %82 unwind label %endcatch.pad6

48:                                               ; preds = %.caught3
  %49 = extractvalue { ptr, i32 } %41, 0
  %50 = call ptr @__cxa_begin_catch(ptr %49)
  store ptr %50, ptr %3, align 8
  %51 = load ptr, ptr %3, align 8
  store ptr %51, ptr %5, align 8
  call void @__cxa_end_catch()
  %52 = load ptr, ptr %5, align 8
  %53 = load i32, ptr %4, align 4
  %54 = call ptr @GC_malloc_atomic(i64 58)
  %55 = getelementptr i8, ptr %54, i64 8
  %56 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %55, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %53)
  %57 = icmp eq ptr %52, null
  %58 = select i1 %57, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %52
  %59 = icmp eq ptr %55, null
  %60 = select i1 %59, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %55
  %61 = call i64 @strlen(ptr %58)
  %62 = add i64 1, %61
  %63 = call i64 @strlen(ptr %60)
  %64 = add i64 %62, %63
  %65 = add i64 %64, 8
  %66 = call ptr @GC_malloc(i64 %65)
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr i8, ptr %67, i64 0
  %69 = sub i64 %64, 0
  %70 = call ptr @__memcpy_chk(ptr %68, ptr %58, i64 %61, i64 %69)
  %71 = add i64 0, %61
  %72 = getelementptr i8, ptr %67, i64 %71
  %73 = sub i64 %64, %71
  %74 = call ptr @__memcpy_chk(ptr %72, ptr %60, i64 %63, i64 %73)
  %75 = add i64 %71, %63
  %76 = getelementptr i8, ptr %67, i64 %75
  store i8 0, ptr %76, align 1
  store ptr %67, ptr %6, align 8
  %77 = load ptr, ptr %6, align 8
  ret ptr %77

endcatch.pad6:                                    ; preds = %45
  %78 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught5

79:                                               ; No predecessors!
  %80 = landingpad { ptr, i32 }
          cleanup
  br label %.caught5

.caught5:                                         ; preds = %endcatch.pad6, %79
  %81 = phi { ptr, i32 } [ %80, %79 ], [ %78, %endcatch.pad6 ]
  resume { ptr, i32 } %81

82:                                               ; preds = %45, %.caught1, %20, %31, %0
  unreachable

endcatch.resume:                                  ; preds = %28
  %83 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %83
}

; Function Attrs: mustprogress
define ptr @inNestedBlock() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %4, align 8
  store i1 false, ptr %1, align 1
  %6 = call ptr @GC_malloc(i64 16)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Thrower, ptr %7, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %8, align 8
  store ptr %7, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %9 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %9, align 8
  invoke void @__cxa_throw(ptr %9, ptr @_ZTIPc, ptr null)
          to label %51 unwind label %10

10:                                               ; preds = %0
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  %14 = load i1, ptr %1, align 1
  br i1 %14, label %15, label %19

15:                                               ; preds = %10
  store i1 false, ptr %1, align 1
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr %Thrower, ptr %16, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %16)
          to label %18 unwind label %endcatch.pad2

18:                                               ; preds = %15
  br label %19

19:                                               ; preds = %18, %10
  invoke void @__cxa_rethrow()
          to label %51 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %15
  %20 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

21:                                               ; No predecessors!
  %22 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %21
  %23 = phi { ptr, i32 } [ %22, %21 ], [ %20, %endcatch.pad2 ]
  %24 = extractvalue { ptr, i32 } %23, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  %26 = load i1, ptr %1, align 1
  br i1 %26, label %27, label %30

27:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %28 = load ptr, ptr %2, align 8
  %29 = getelementptr %Thrower, ptr %28, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %28)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %27
  br label %30

30:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %51 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %19, %30
  %31 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

32:                                               ; No predecessors!
  %33 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %32
  %34 = phi { ptr, i32 } [ %33, %32 ], [ %31, %endcatch.pad ]
  %35 = extractvalue { ptr, i32 } %34, 1
  %36 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %37 = icmp eq i32 %35, %36
  br i1 %37, label %41, label %38

38:                                               ; preds = %.caught
  %39 = extractvalue { ptr, i32 } %34, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  invoke void @__cxa_rethrow()
          to label %51 unwind label %endcatch.pad4

41:                                               ; preds = %.caught
  %42 = extractvalue { ptr, i32 } %34, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  store ptr %43, ptr %3, align 8
  %44 = load ptr, ptr %3, align 8
  store ptr %44, ptr %4, align 8
  call void @__cxa_end_catch()
  %45 = load ptr, ptr %4, align 8
  store ptr %45, ptr %5, align 8
  %46 = load ptr, ptr %5, align 8
  ret ptr %46

endcatch.pad4:                                    ; preds = %38
  %47 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

48:                                               ; No predecessors!
  %49 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %48
  %50 = phi { ptr, i32 } [ %49, %48 ], [ %47, %endcatch.pad4 ]
  resume { ptr, i32 } %50

51:                                               ; preds = %38, %19, %30, %0
  unreachable

endcatch.resume:                                  ; preds = %27
  %52 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %52
}

; Function Attrs: mustprogress
define void @cleanupOnly() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca <{ i64, %Thrower }>, align 16
  store i64 -1, ptr %3, align 8
  %4 = getelementptr i8, ptr %3, i64 8
  store %Thrower zeroinitializer, ptr %4, align 8
  store i1 false, ptr %1, align 1
  store %Thrower zeroinitializer, ptr %4, align 8
  %5 = getelementptr %Thrower, ptr %4, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %5, align 8
  store ptr %4, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %6 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %6, align 8
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIPc, ptr null)
          to label %14 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          cleanup
  %9 = load i1, ptr %1, align 1
  br i1 %9, label %10, label %13

10:                                               ; preds = %7
  store i1 false, ptr %1, align 1
  %11 = load ptr, ptr %2, align 8
  %12 = getelementptr %Thrower, ptr %11, i32 0, i32 0
  call void @Thrower.dispose(ptr %11)
  br label %13

13:                                               ; preds = %10, %7
  resume { ptr, i32 } %8

14:                                               ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define ptr @callerCatches() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %2, align 8
  invoke void @cleanupOnly()
          to label %4 unwind label %5

4:                                                ; preds = %0
  br label %17

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %24 unwind label %endcatch.pad

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  store ptr %15, ptr %1, align 8
  %16 = load ptr, ptr %1, align 8
  store ptr %16, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %17

17:                                               ; preds = %4, %13
  %18 = load ptr, ptr %2, align 8
  store ptr %18, ptr %3, align 8
  %19 = load ptr, ptr %3, align 8
  ret ptr %19

endcatch.pad:                                     ; preds = %10
  %20 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

21:                                               ; No predecessors!
  %22 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %21
  %23 = phi { ptr, i32 } [ %22, %21 ], [ %20, %endcatch.pad ]
  resume { ptr, i32 } %23

24:                                               ; preds = %10
  unreachable
}

; Function Attrs: mustprogress
define ptr @innerCatchMismatch() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %5, align 8
  store i1 false, ptr %1, align 1
  %7 = call ptr @GC_malloc(i64 16)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Thrower, ptr %8, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %9, align 8
  store ptr %8, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %10 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %10, align 4
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIi, ptr null)
          to label %104 unwind label %11

11:                                               ; preds = %0
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i1, ptr %1, align 1
  br i1 %15, label %16, label %20

16:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr %Thrower, ptr %17, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %17)
          to label %19 unwind label %endcatch.pad6

19:                                               ; preds = %16
  br label %20

20:                                               ; preds = %19, %11
  invoke void @__cxa_rethrow()
          to label %104 unwind label %endcatch.pad2

endcatch.pad6:                                    ; preds = %16
  %21 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught5

22:                                               ; No predecessors!
  %23 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught5

.caught5:                                         ; preds = %endcatch.pad6, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad6 ]
  %25 = extractvalue { ptr, i32 } %24, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = load i1, ptr %1, align 1
  br i1 %27, label %28, label %31

28:                                               ; preds = %.caught5
  store i1 false, ptr %1, align 1
  %29 = load ptr, ptr %2, align 8
  %30 = getelementptr %Thrower, ptr %29, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %29)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %28
  br label %31

31:                                               ; preds = %.noexc7, %.caught5
  invoke void @__cxa_rethrow()
          to label %104 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %31, %20
  %32 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

33:                                               ; No predecessors!
  %34 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %33
  %35 = phi { ptr, i32 } [ %34, %33 ], [ %32, %endcatch.pad2 ]
  %36 = extractvalue { ptr, i32 } %35, 1
  %37 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %38 = icmp eq i32 %36, %37
  %39 = extractvalue { ptr, i32 } %35, 1
  %40 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %41 = icmp eq i32 %39, %40
  %42 = extractvalue { ptr, i32 } %35, 1
  %43 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %44 = icmp eq i32 %42, %43
  %45 = extractvalue { ptr, i32 } %35, 1
  %46 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %47 = icmp eq i32 %45, %46
  %48 = or i1 %38, %41
  %49 = or i1 %48, %44
  %50 = or i1 %49, %47
  br i1 %50, label %54, label %51

51:                                               ; preds = %.caught1
  %52 = extractvalue { ptr, i32 } %35, 0
  %53 = call ptr @__cxa_begin_catch(ptr %52)
  invoke void @__cxa_rethrow()
          to label %104 unwind label %endcatch.pad4

54:                                               ; preds = %.caught1
  %55 = extractvalue { ptr, i32 } %35, 0
  %56 = call ptr @__cxa_begin_catch(ptr %55)
  %57 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %54
  %58 = icmp eq ptr %57, @_ZTIi
  br i1 %58, label %59, label %62

59:                                               ; preds = %.noexc
  %60 = load i32, ptr %56, align 4
  %61 = sitofp i32 %60 to double
  br label %80

62:                                               ; preds = %.noexc
  %63 = icmp eq ptr %57, @_ZTIj
  br i1 %63, label %64, label %67

64:                                               ; preds = %62
  %65 = load i32, ptr %56, align 4
  %66 = uitofp i32 %65 to double
  br label %77

67:                                               ; preds = %62
  %68 = icmp eq ptr %57, @_ZTIf
  br i1 %68, label %69, label %72

69:                                               ; preds = %67
  %70 = load float, ptr %56, align 4
  %71 = fpext float %70 to double
  br label %74

72:                                               ; preds = %67
  %73 = load double, ptr %56, align 8
  br label %74

74:                                               ; preds = %69, %72
  %75 = phi double [ %73, %72 ], [ %71, %69 ]
  br label %76

76:                                               ; preds = %74
  br label %77

77:                                               ; preds = %64, %76
  %78 = phi double [ %75, %76 ], [ %66, %64 ]
  br label %79

79:                                               ; preds = %77
  br label %80

80:                                               ; preds = %59, %79
  %81 = phi double [ %78, %79 ], [ %61, %59 ]
  br label %82

82:                                               ; preds = %80
  store double %81, ptr %3, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8), ptr %5, align 8
  call void @__cxa_end_catch()
  br label %97

endcatch.pad4:                                    ; preds = %51
  %83 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

84:                                               ; No predecessors!
  %85 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %84
  %86 = phi { ptr, i32 } [ %85, %84 ], [ %83, %endcatch.pad4 ]
  %87 = extractvalue { ptr, i32 } %86, 1
  %88 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %89 = icmp eq i32 %87, %88
  br i1 %89, label %93, label %90

90:                                               ; preds = %.caught3
  %91 = extractvalue { ptr, i32 } %86, 0
  %92 = call ptr @__cxa_begin_catch(ptr %91)
  invoke void @__cxa_rethrow()
          to label %104 unwind label %endcatch.pad

93:                                               ; preds = %.caught3
  %94 = extractvalue { ptr, i32 } %86, 0
  %95 = call ptr @__cxa_begin_catch(ptr %94)
  store ptr %95, ptr %4, align 8
  %96 = load ptr, ptr %4, align 8
  store ptr %96, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %97

97:                                               ; preds = %82, %93
  %98 = load ptr, ptr %5, align 8
  store ptr %98, ptr %6, align 8
  %99 = load ptr, ptr %6, align 8
  ret ptr %99

endcatch.pad:                                     ; preds = %90
  %100 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

101:                                              ; No predecessors!
  %102 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %101
  %103 = phi { ptr, i32 } [ %102, %101 ], [ %100, %endcatch.pad ]
  resume { ptr, i32 } %103

104:                                              ; preds = %90, %51, %20, %31, %0
  unreachable

endcatch.resume:                                  ; preds = %28, %54
  %105 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %105
}

; Function Attrs: mustprogress
define ptr @inLoop() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %5, align 8
  store i32 0, ptr %4, align 4
  br label %7

7:                                                ; preds = %84, %0
  %8 = load i32, ptr %4, align 4
  %9 = icmp slt i32 %8, 3
  br i1 %9, label %10, label %87

10:                                               ; preds = %7
  store i1 false, ptr %1, align 1
  %11 = call ptr @GC_malloc(i64 16)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr %Thrower, ptr %12, i32 0, i32 0
  store ptr @Thrower..vtbl, ptr %13, align 8
  store ptr %12, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %14 = load i32, ptr %4, align 4
  %15 = icmp eq i32 %14, 1
  br i1 %15, label %16, label %18

16:                                               ; preds = %10
  %17 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %17, align 8
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIPc, ptr null)
          to label %94 unwind label %25

18:                                               ; preds = %10
  %19 = load i1, ptr %1, align 1
  br i1 %19, label %20, label %24

20:                                               ; preds = %18
  store i1 false, ptr %1, align 1
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr %Thrower, ptr %21, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %21)
          to label %23 unwind label %25

23:                                               ; preds = %20
  br label %24

24:                                               ; preds = %23, %18
  br label %84

25:                                               ; preds = %16, %20
  %26 = landingpad { ptr, i32 }
          catch ptr null
  %27 = extractvalue { ptr, i32 } %26, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i1, ptr %1, align 1
  br i1 %29, label %30, label %34

30:                                               ; preds = %25
  store i1 false, ptr %1, align 1
  %31 = load ptr, ptr %2, align 8
  %32 = getelementptr %Thrower, ptr %31, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %31)
          to label %33 unwind label %endcatch.pad2

33:                                               ; preds = %30
  br label %34

34:                                               ; preds = %33, %25
  invoke void @__cxa_rethrow()
          to label %94 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %30
  %35 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

36:                                               ; No predecessors!
  %37 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %36
  %38 = phi { ptr, i32 } [ %37, %36 ], [ %35, %endcatch.pad2 ]
  %39 = extractvalue { ptr, i32 } %38, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  %41 = load i1, ptr %1, align 1
  br i1 %41, label %42, label %45

42:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %43 = load ptr, ptr %2, align 8
  %44 = getelementptr %Thrower, ptr %43, i32 0, i32 0
  invoke void @Thrower.dispose(ptr %43)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %42
  br label %45

45:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %94 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %45, %34
  %46 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

47:                                               ; No predecessors!
  %48 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %47
  %49 = phi { ptr, i32 } [ %48, %47 ], [ %46, %endcatch.pad ]
  %50 = extractvalue { ptr, i32 } %49, 1
  %51 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %52 = icmp eq i32 %50, %51
  br i1 %52, label %56, label %53

53:                                               ; preds = %.caught
  %54 = extractvalue { ptr, i32 } %49, 0
  %55 = call ptr @__cxa_begin_catch(ptr %54)
  invoke void @__cxa_rethrow()
          to label %94 unwind label %endcatch.pad4

56:                                               ; preds = %.caught
  %57 = extractvalue { ptr, i32 } %49, 0
  %58 = call ptr @__cxa_begin_catch(ptr %57)
  store ptr %58, ptr %3, align 8
  %59 = load ptr, ptr %5, align 8
  %60 = load i32, ptr %4, align 4
  %61 = invoke ptr @GC_malloc_atomic(i64 58)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %56
  %62 = getelementptr i8, ptr %61, i64 8
  %63 = invoke i32 (ptr, i64, ptr, ...) @snprintf(ptr %62, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %60)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %.noexc5
  %64 = icmp eq ptr %59, null
  %65 = select i1 %64, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %59
  %66 = icmp eq ptr %62, null
  %67 = select i1 %66, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %62
  %68 = invoke i64 @strlen(ptr %65)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %.noexc6
  %69 = add i64 1, %68
  %70 = invoke i64 @strlen(ptr %67)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %.noexc7
  %71 = add i64 %69, %70
  %72 = add i64 %71, 8
  %73 = invoke ptr @GC_malloc(i64 %72)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %.noexc8
  %74 = getelementptr i8, ptr %73, i64 8
  %75 = getelementptr i8, ptr %74, i64 0
  %76 = sub i64 %71, 0
  %77 = invoke ptr @__memcpy_chk(ptr %75, ptr %65, i64 %68, i64 %76)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %.noexc9
  %78 = add i64 0, %68
  %79 = getelementptr i8, ptr %74, i64 %78
  %80 = sub i64 %71, %78
  %81 = invoke ptr @__memcpy_chk(ptr %79, ptr %67, i64 %70, i64 %80)
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %.noexc10
  %82 = add i64 %78, %70
  %83 = getelementptr i8, ptr %74, i64 %82
  store i8 0, ptr %83, align 1
  store ptr %74, ptr %5, align 8
  call void @__cxa_end_catch()
  br label %84

84:                                               ; preds = %.noexc11, %24
  %85 = load i32, ptr %4, align 4
  %86 = add i32 %85, 1
  store i32 %86, ptr %4, align 4
  br label %7

87:                                               ; preds = %7
  %88 = load ptr, ptr %5, align 8
  store ptr %88, ptr %6, align 8
  %89 = load ptr, ptr %6, align 8
  ret ptr %89

endcatch.pad4:                                    ; preds = %53
  %90 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

91:                                               ; No predecessors!
  %92 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %91
  %93 = phi { ptr, i32 } [ %92, %91 ], [ %90, %endcatch.pad4 ]
  resume { ptr, i32 } %93

94:                                               ; preds = %53, %34, %45, %16
  unreachable

endcatch.resume:                                  ; preds = %.noexc10, %.noexc9, %.noexc8, %.noexc7, %.noexc6, %.noexc5, %56, %42
  %95 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %95
}

; Function Attrs: mustprogress
define ptr @twoUsings() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Named, ptr %9, i32 0, i32 0
  store ptr @Named..vtbl, ptr %10, align 8
  invoke void @Named.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %11 unwind label %17

11:                                               ; preds = %0
  store ptr %9, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %NamedThrower, ptr %13, i32 0, i32 0
  store ptr @NamedThrower..vtbl, ptr %14, align 8
  invoke void @NamedThrower.constructor(ptr %13, ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8))
          to label %15 unwind label %17

15:                                               ; preds = %11
  store ptr %13, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %16 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %16, align 8
  invoke void @__cxa_throw(ptr %16, ptr @_ZTIPc, ptr null)
          to label %96 unwind label %17

17:                                               ; preds = %15, %11, %0
  %18 = landingpad { ptr, i32 }
          catch ptr null
  %19 = extractvalue { ptr, i32 } %18, 0
  %20 = call ptr @__cxa_begin_catch(ptr %19)
  %21 = load i1, ptr %1, align 1
  br i1 %21, label %22, label %26

22:                                               ; preds = %17
  store i1 false, ptr %1, align 1
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %NamedThrower, ptr %23, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %23)
          to label %25 unwind label %endcatch.pad2

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %25, %17
  %27 = load i1, ptr %3, align 1
  br i1 %27, label %28, label %32

28:                                               ; preds = %26
  store i1 false, ptr %3, align 1
  %29 = load ptr, ptr %4, align 8
  %30 = getelementptr %Named, ptr %29, i32 0, i32 0
  invoke void @Named.dispose(ptr %29)
          to label %31 unwind label %endcatch.pad2

31:                                               ; preds = %28
  br label %32

32:                                               ; preds = %31, %26
  invoke void @__cxa_rethrow()
          to label %96 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %22, %28
  %33 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %33, %endcatch.pad2 ]
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  %39 = load i1, ptr %1, align 1
  br i1 %39, label %40, label %43

40:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %41 = load ptr, ptr %2, align 8
  %42 = getelementptr %NamedThrower, ptr %41, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %41)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %40
  br label %43

43:                                               ; preds = %.noexc, %.caught1
  %44 = load i1, ptr %3, align 1
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  store i1 false, ptr %3, align 1
  %46 = load ptr, ptr %4, align 8
  %47 = getelementptr %Named, ptr %46, i32 0, i32 0
  invoke void @Named.dispose(ptr %46)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %45
  br label %48

48:                                               ; preds = %.noexc5, %43
  invoke void @__cxa_rethrow()
          to label %96 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %32, %48
  %49 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

50:                                               ; No predecessors!
  %51 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %50
  %52 = phi { ptr, i32 } [ %51, %50 ], [ %49, %endcatch.pad ]
  %53 = extractvalue { ptr, i32 } %52, 1
  %54 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %55 = icmp eq i32 %53, %54
  br i1 %55, label %59, label %56

56:                                               ; preds = %.caught
  %57 = extractvalue { ptr, i32 } %52, 0
  %58 = call ptr @__cxa_begin_catch(ptr %57)
  invoke void @__cxa_rethrow()
          to label %96 unwind label %endcatch.pad4

59:                                               ; preds = %.caught
  %60 = extractvalue { ptr, i32 } %52, 0
  %61 = call ptr @__cxa_begin_catch(ptr %60)
  store ptr %61, ptr %5, align 8
  %62 = load ptr, ptr %5, align 8
  store ptr %62, ptr %6, align 8
  call void @__cxa_end_catch()
  %63 = load ptr, ptr %6, align 8
  %64 = load ptr, ptr @trail, align 8
  %65 = icmp eq ptr %63, null
  %66 = select i1 %65, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %63
  %67 = icmp eq ptr %64, null
  %68 = select i1 %67, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %64
  %69 = call i64 @strlen(ptr %66)
  %70 = add i64 1, %69
  %71 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8))
  %72 = add i64 %70, %71
  %73 = call i64 @strlen(ptr %68)
  %74 = add i64 %72, %73
  %75 = add i64 %74, 8
  %76 = call ptr @GC_malloc(i64 %75)
  %77 = getelementptr i8, ptr %76, i64 8
  %78 = getelementptr i8, ptr %77, i64 0
  %79 = sub i64 %74, 0
  %80 = call ptr @__memcpy_chk(ptr %78, ptr %66, i64 %69, i64 %79)
  %81 = add i64 0, %69
  %82 = getelementptr i8, ptr %77, i64 %81
  %83 = sub i64 %74, %81
  %84 = call ptr @__memcpy_chk(ptr %82, ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8), i64 %71, i64 %83)
  %85 = add i64 %81, %71
  %86 = getelementptr i8, ptr %77, i64 %85
  %87 = sub i64 %74, %85
  %88 = call ptr @__memcpy_chk(ptr %86, ptr %68, i64 %73, i64 %87)
  %89 = add i64 %85, %73
  %90 = getelementptr i8, ptr %77, i64 %89
  store i8 0, ptr %90, align 1
  store ptr %77, ptr %7, align 8
  %91 = load ptr, ptr %7, align 8
  ret ptr %91

endcatch.pad4:                                    ; preds = %56
  %92 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

93:                                               ; No predecessors!
  %94 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %93
  %95 = phi { ptr, i32 } [ %94, %93 ], [ %92, %endcatch.pad4 ]
  resume { ptr, i32 } %95

96:                                               ; preds = %56, %32, %48, %15
  unreachable

endcatch.resume:                                  ; preds = %45, %40
  %97 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %97
}

; Function Attrs: mustprogress
define ptr @bothThrowInner() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %NamedThrower, ptr %9, i32 0, i32 0
  store ptr @NamedThrower..vtbl, ptr %10, align 8
  invoke void @NamedThrower.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8))
          to label %11 unwind label %17

11:                                               ; preds = %0
  store ptr %9, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %NamedThrower, ptr %13, i32 0, i32 0
  store ptr @NamedThrower..vtbl, ptr %14, align 8
  invoke void @NamedThrower.constructor(ptr %13, ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8))
          to label %15 unwind label %17

15:                                               ; preds = %11
  store ptr %13, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %16 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %16, align 8
  invoke void @__cxa_throw(ptr %16, ptr @_ZTIPc, ptr null)
          to label %69 unwind label %17

17:                                               ; preds = %15, %11, %0
  %18 = landingpad { ptr, i32 }
          catch ptr null
  %19 = extractvalue { ptr, i32 } %18, 0
  %20 = call ptr @__cxa_begin_catch(ptr %19)
  %21 = load i1, ptr %1, align 1
  br i1 %21, label %22, label %26

22:                                               ; preds = %17
  store i1 false, ptr %1, align 1
  %23 = load ptr, ptr %2, align 8
  %24 = getelementptr %NamedThrower, ptr %23, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %23)
          to label %25 unwind label %endcatch.pad

25:                                               ; preds = %22
  br label %26

26:                                               ; preds = %25, %17
  %27 = load i1, ptr %3, align 1
  br i1 %27, label %28, label %32

28:                                               ; preds = %26
  store i1 false, ptr %3, align 1
  %29 = load ptr, ptr %4, align 8
  %30 = getelementptr %NamedThrower, ptr %29, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %29)
          to label %31 unwind label %endcatch.pad

31:                                               ; preds = %28
  br label %32

32:                                               ; preds = %31, %26
  invoke void @__cxa_rethrow()
          to label %69 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %28, %22
  %33 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

34:                                               ; No predecessors!
  %35 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %34
  %36 = phi { ptr, i32 } [ %35, %34 ], [ %33, %endcatch.pad ]
  %37 = extractvalue { ptr, i32 } %36, 0
  %38 = call ptr @__cxa_begin_catch(ptr %37)
  %39 = load i1, ptr %1, align 1
  br i1 %39, label %40, label %43

40:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %41 = load ptr, ptr %2, align 8
  %42 = getelementptr %NamedThrower, ptr %41, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %41)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %40
  br label %43

43:                                               ; preds = %.noexc, %.caught
  %44 = load i1, ptr %3, align 1
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  store i1 false, ptr %3, align 1
  %46 = load ptr, ptr %4, align 8
  %47 = getelementptr %NamedThrower, ptr %46, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %46)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %45
  br label %48

48:                                               ; preds = %.noexc5, %43
  invoke void @__cxa_rethrow()
          to label %69 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %32, %48
  %49 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

50:                                               ; No predecessors!
  %51 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %50
  %52 = phi { ptr, i32 } [ %51, %50 ], [ %49, %endcatch.pad2 ]
  %53 = extractvalue { ptr, i32 } %52, 1
  %54 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %55 = icmp eq i32 %53, %54
  br i1 %55, label %59, label %56

56:                                               ; preds = %.caught1
  %57 = extractvalue { ptr, i32 } %52, 0
  %58 = call ptr @__cxa_begin_catch(ptr %57)
  invoke void @__cxa_rethrow()
          to label %69 unwind label %endcatch.pad4

59:                                               ; preds = %.caught1
  %60 = extractvalue { ptr, i32 } %52, 0
  %61 = call ptr @__cxa_begin_catch(ptr %60)
  store ptr %61, ptr %5, align 8
  %62 = load ptr, ptr %5, align 8
  store ptr %62, ptr %6, align 8
  call void @__cxa_end_catch()
  %63 = load ptr, ptr %6, align 8
  store ptr %63, ptr %7, align 8
  %64 = load ptr, ptr %7, align 8
  ret ptr %64

endcatch.pad4:                                    ; preds = %56
  %65 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

66:                                               ; No predecessors!
  %67 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %66
  %68 = phi { ptr, i32 } [ %67, %66 ], [ %65, %endcatch.pad4 ]
  resume { ptr, i32 } %68

69:                                               ; preds = %56, %32, %48, %15
  unreachable

endcatch.resume:                                  ; preds = %45, %40
  %70 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %70
}

; Function Attrs: mustprogress
define ptr @bothThrow() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %2, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %4 = invoke ptr @bothThrowInner()
          to label %5 unwind label %6

5:                                                ; preds = %0
  store ptr %4, ptr %2, align 8
  br label %18

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %52 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  store ptr %16, ptr %1, align 8
  %17 = load ptr, ptr %1, align 8
  store ptr %17, ptr %2, align 8
  call void @__cxa_end_catch()
  br label %18

18:                                               ; preds = %5, %14
  %19 = load ptr, ptr %2, align 8
  %20 = load ptr, ptr @trail, align 8
  %21 = icmp eq ptr %19, null
  %22 = select i1 %21, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %19
  %23 = icmp eq ptr %20, null
  %24 = select i1 %23, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %20
  %25 = call i64 @strlen(ptr %22)
  %26 = add i64 1, %25
  %27 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8))
  %28 = add i64 %26, %27
  %29 = call i64 @strlen(ptr %24)
  %30 = add i64 %28, %29
  %31 = add i64 %30, 8
  %32 = call ptr @GC_malloc(i64 %31)
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = getelementptr i8, ptr %33, i64 0
  %35 = sub i64 %30, 0
  %36 = call ptr @__memcpy_chk(ptr %34, ptr %22, i64 %25, i64 %35)
  %37 = add i64 0, %25
  %38 = getelementptr i8, ptr %33, i64 %37
  %39 = sub i64 %30, %37
  %40 = call ptr @__memcpy_chk(ptr %38, ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8), i64 %27, i64 %39)
  %41 = add i64 %37, %27
  %42 = getelementptr i8, ptr %33, i64 %41
  %43 = sub i64 %30, %41
  %44 = call ptr @__memcpy_chk(ptr %42, ptr %24, i64 %29, i64 %43)
  %45 = add i64 %41, %29
  %46 = getelementptr i8, ptr %33, i64 %45
  store i8 0, ptr %46, align 1
  store ptr %33, ptr %3, align 8
  %47 = load ptr, ptr %3, align 8
  ret ptr %47

endcatch.pad:                                     ; preds = %11
  %48 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

49:                                               ; No predecessors!
  %50 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %49
  %51 = phi { ptr, i32 } [ %50, %49 ], [ %48, %endcatch.pad ]
  resume { ptr, i32 } %51

52:                                               ; preds = %11
  unreachable
}

; Function Attrs: mustprogress
define ptr @twoUsingsNormalPath() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %6, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  store i1 false, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Named, ptr %9, i32 0, i32 0
  store ptr @Named..vtbl, ptr %10, align 8
  invoke void @Named.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %11 unwind label %28

11:                                               ; preds = %0
  store ptr %9, ptr %4, align 8
  store i1 true, ptr %3, align 1
  %12 = call ptr @GC_malloc(i64 24)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr %NamedThrower, ptr %13, i32 0, i32 0
  store ptr @NamedThrower..vtbl, ptr %14, align 8
  invoke void @NamedThrower.constructor(ptr %13, ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8))
          to label %15 unwind label %28

15:                                               ; preds = %11
  store ptr %13, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %16 = load i1, ptr %1, align 1
  br i1 %16, label %17, label %21

17:                                               ; preds = %15
  store i1 false, ptr %1, align 1
  %18 = load ptr, ptr %2, align 8
  %19 = getelementptr %NamedThrower, ptr %18, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %18)
          to label %20 unwind label %28

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %15
  %22 = load i1, ptr %3, align 1
  br i1 %22, label %23, label %27

23:                                               ; preds = %21
  store i1 false, ptr %3, align 1
  %24 = load ptr, ptr %4, align 8
  %25 = getelementptr %Named, ptr %24, i32 0, i32 0
  invoke void @Named.dispose(ptr %24)
          to label %26 unwind label %28

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %26, %21
  br label %74

28:                                               ; preds = %23, %17, %11, %0
  %29 = landingpad { ptr, i32 }
          catch ptr null
  %30 = extractvalue { ptr, i32 } %29, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  %32 = load i1, ptr %1, align 1
  br i1 %32, label %33, label %37

33:                                               ; preds = %28
  store i1 false, ptr %1, align 1
  %34 = load ptr, ptr %2, align 8
  %35 = getelementptr %NamedThrower, ptr %34, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %34)
          to label %36 unwind label %endcatch.pad2

36:                                               ; preds = %33
  br label %37

37:                                               ; preds = %36, %28
  %38 = load i1, ptr %3, align 1
  br i1 %38, label %39, label %43

39:                                               ; preds = %37
  store i1 false, ptr %3, align 1
  %40 = load ptr, ptr %4, align 8
  %41 = getelementptr %Named, ptr %40, i32 0, i32 0
  invoke void @Named.dispose(ptr %40)
          to label %42 unwind label %endcatch.pad2

42:                                               ; preds = %39
  br label %43

43:                                               ; preds = %42, %37
  invoke void @__cxa_rethrow()
          to label %108 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %39, %33
  %44 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

45:                                               ; No predecessors!
  %46 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %45
  %47 = phi { ptr, i32 } [ %46, %45 ], [ %44, %endcatch.pad2 ]
  %48 = extractvalue { ptr, i32 } %47, 0
  %49 = call ptr @__cxa_begin_catch(ptr %48)
  %50 = load i1, ptr %1, align 1
  br i1 %50, label %51, label %54

51:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %52 = load ptr, ptr %2, align 8
  %53 = getelementptr %NamedThrower, ptr %52, i32 0, i32 0
  invoke void @NamedThrower.dispose(ptr %52)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %51
  br label %54

54:                                               ; preds = %.noexc, %.caught1
  %55 = load i1, ptr %3, align 1
  br i1 %55, label %56, label %59

56:                                               ; preds = %54
  store i1 false, ptr %3, align 1
  %57 = load ptr, ptr %4, align 8
  %58 = getelementptr %Named, ptr %57, i32 0, i32 0
  invoke void @Named.dispose(ptr %57)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %56
  br label %59

59:                                               ; preds = %.noexc5, %54
  invoke void @__cxa_rethrow()
          to label %108 unwind label %endcatch.pad

endcatch.pad:                                     ; preds = %59, %43
  %60 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

61:                                               ; No predecessors!
  %62 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %61
  %63 = phi { ptr, i32 } [ %62, %61 ], [ %60, %endcatch.pad ]
  %64 = extractvalue { ptr, i32 } %63, 1
  %65 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %66 = icmp eq i32 %64, %65
  br i1 %66, label %70, label %67

67:                                               ; preds = %.caught
  %68 = extractvalue { ptr, i32 } %63, 0
  %69 = call ptr @__cxa_begin_catch(ptr %68)
  invoke void @__cxa_rethrow()
          to label %108 unwind label %endcatch.pad4

70:                                               ; preds = %.caught
  %71 = extractvalue { ptr, i32 } %63, 0
  %72 = call ptr @__cxa_begin_catch(ptr %71)
  store ptr %72, ptr %5, align 8
  %73 = load ptr, ptr %5, align 8
  store ptr %73, ptr %6, align 8
  call void @__cxa_end_catch()
  br label %74

74:                                               ; preds = %27, %70
  %75 = load ptr, ptr %6, align 8
  %76 = load ptr, ptr @trail, align 8
  %77 = icmp eq ptr %75, null
  %78 = select i1 %77, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %75
  %79 = icmp eq ptr %76, null
  %80 = select i1 %79, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %76
  %81 = call i64 @strlen(ptr %78)
  %82 = add i64 1, %81
  %83 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8))
  %84 = add i64 %82, %83
  %85 = call i64 @strlen(ptr %80)
  %86 = add i64 %84, %85
  %87 = add i64 %86, 8
  %88 = call ptr @GC_malloc(i64 %87)
  %89 = getelementptr i8, ptr %88, i64 8
  %90 = getelementptr i8, ptr %89, i64 0
  %91 = sub i64 %86, 0
  %92 = call ptr @__memcpy_chk(ptr %90, ptr %78, i64 %81, i64 %91)
  %93 = add i64 0, %81
  %94 = getelementptr i8, ptr %89, i64 %93
  %95 = sub i64 %86, %93
  %96 = call ptr @__memcpy_chk(ptr %94, ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8), i64 %83, i64 %95)
  %97 = add i64 %93, %83
  %98 = getelementptr i8, ptr %89, i64 %97
  %99 = sub i64 %86, %97
  %100 = call ptr @__memcpy_chk(ptr %98, ptr %80, i64 %85, i64 %99)
  %101 = add i64 %97, %85
  %102 = getelementptr i8, ptr %89, i64 %101
  store i8 0, ptr %102, align 1
  store ptr %89, ptr %7, align 8
  %103 = load ptr, ptr %7, align 8
  ret ptr %103

endcatch.pad4:                                    ; preds = %67
  %104 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

105:                                              ; No predecessors!
  %106 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %105
  %107 = phi { ptr, i32 } [ %106, %105 ], [ %104, %endcatch.pad4 ]
  resume { ptr, i32 } %107

108:                                              ; preds = %67, %43, %59
  unreachable

endcatch.resume:                                  ; preds = %56, %51
  %109 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %109
}

; Function Attrs: mustprogress
define ptr @disposalDoesNotThrow() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %4, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  store i1 false, ptr %1, align 1
  %6 = call ptr @GC_malloc(i64 24)
  %7 = getelementptr i8, ptr %6, i64 8
  %8 = getelementptr %Named, ptr %7, i32 0, i32 0
  store ptr @Named..vtbl, ptr %8, align 8
  invoke void @Named.constructor(ptr %7, ptr getelementptr inbounds nuw (i8, ptr @s_12638187200555641996, i64 8))
          to label %9 unwind label %11

9:                                                ; preds = %0
  store ptr %7, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %10 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_9932668635479743809, i64 8), ptr %10, align 8
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIPc, ptr null)
          to label %79 unwind label %11

11:                                               ; preds = %9, %0
  %12 = landingpad { ptr, i32 }
          catch ptr null
  %13 = extractvalue { ptr, i32 } %12, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i1, ptr %1, align 1
  br i1 %15, label %16, label %20

16:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr %Named, ptr %17, i32 0, i32 0
  invoke void @Named.dispose(ptr %17)
          to label %19 unwind label %endcatch.pad

19:                                               ; preds = %16
  br label %20

20:                                               ; preds = %19, %11
  invoke void @__cxa_rethrow()
          to label %79 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %16
  %21 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

22:                                               ; No predecessors!
  %23 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %22
  %24 = phi { ptr, i32 } [ %23, %22 ], [ %21, %endcatch.pad ]
  %25 = extractvalue { ptr, i32 } %24, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  %27 = load i1, ptr %1, align 1
  br i1 %27, label %28, label %31

28:                                               ; preds = %.caught
  store i1 false, ptr %1, align 1
  %29 = load ptr, ptr %2, align 8
  %30 = getelementptr %Named, ptr %29, i32 0, i32 0
  invoke void @Named.dispose(ptr %29)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  br label %31

31:                                               ; preds = %.noexc, %.caught
  invoke void @__cxa_rethrow()
          to label %79 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %31, %20
  %32 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

33:                                               ; No predecessors!
  %34 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %33
  %35 = phi { ptr, i32 } [ %34, %33 ], [ %32, %endcatch.pad2 ]
  %36 = extractvalue { ptr, i32 } %35, 1
  %37 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %38 = icmp eq i32 %36, %37
  br i1 %38, label %42, label %39

39:                                               ; preds = %.caught1
  %40 = extractvalue { ptr, i32 } %35, 0
  %41 = call ptr @__cxa_begin_catch(ptr %40)
  invoke void @__cxa_rethrow()
          to label %79 unwind label %endcatch.pad4

42:                                               ; preds = %.caught1
  %43 = extractvalue { ptr, i32 } %35, 0
  %44 = call ptr @__cxa_begin_catch(ptr %43)
  store ptr %44, ptr %3, align 8
  %45 = load ptr, ptr %3, align 8
  store ptr %45, ptr %4, align 8
  call void @__cxa_end_catch()
  %46 = load ptr, ptr %4, align 8
  %47 = load ptr, ptr @trail, align 8
  %48 = icmp eq ptr %46, null
  %49 = select i1 %48, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %46
  %50 = icmp eq ptr %47, null
  %51 = select i1 %50, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %47
  %52 = call i64 @strlen(ptr %49)
  %53 = add i64 1, %52
  %54 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8))
  %55 = add i64 %53, %54
  %56 = call i64 @strlen(ptr %51)
  %57 = add i64 %55, %56
  %58 = add i64 %57, 8
  %59 = call ptr @GC_malloc(i64 %58)
  %60 = getelementptr i8, ptr %59, i64 8
  %61 = getelementptr i8, ptr %60, i64 0
  %62 = sub i64 %57, 0
  %63 = call ptr @__memcpy_chk(ptr %61, ptr %49, i64 %52, i64 %62)
  %64 = add i64 0, %52
  %65 = getelementptr i8, ptr %60, i64 %64
  %66 = sub i64 %57, %64
  %67 = call ptr @__memcpy_chk(ptr %65, ptr getelementptr inbounds nuw (i8, ptr @s_12638123428881205758, i64 8), i64 %54, i64 %66)
  %68 = add i64 %64, %54
  %69 = getelementptr i8, ptr %60, i64 %68
  %70 = sub i64 %57, %68
  %71 = call ptr @__memcpy_chk(ptr %69, ptr %51, i64 %56, i64 %70)
  %72 = add i64 %68, %56
  %73 = getelementptr i8, ptr %60, i64 %72
  store i8 0, ptr %73, align 1
  store ptr %60, ptr %5, align 8
  %74 = load ptr, ptr %5, align 8
  ret ptr %74

endcatch.pad4:                                    ; preds = %39
  %75 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught3

76:                                               ; No predecessors!
  %77 = landingpad { ptr, i32 }
          cleanup
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %76
  %78 = phi { ptr, i32 } [ %77, %76 ], [ %75, %endcatch.pad4 ]
  resume { ptr, i32 } %78

79:                                               ; preds = %39, %20, %31, %9
  unreachable

endcatch.resume:                                  ; preds = %28
  %80 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %80
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = call ptr @inTryCatch()
  %2 = ptrtoint ptr %1 to i64
  %3 = icmp ne i64 %2, 0
  %4 = zext i1 %3 to i32
  %5 = and i32 %4, 1
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %0
  %8 = call i32 @strcmp(ptr %1, ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8))
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %0
  %11 = icmp eq i64 %2, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8) to i64)
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i1 [ %11, %10 ], [ %9, %7 ]
  br label %14

14:                                               ; preds = %12
  br i1 %13, label %15, label %152

15:                                               ; preds = %14
  %16 = call ptr @withFinallyOnly()
  %17 = ptrtoint ptr %16 to i64
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i32
  %20 = and i32 %19, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %15
  %23 = call i32 @strcmp(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_9805880408515592371, i64 8))
  %24 = icmp eq i32 %23, 0
  br label %27

25:                                               ; preds = %15
  %26 = icmp eq i64 %17, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9805880408515592371, i64 8) to i64)
  br label %27

27:                                               ; preds = %22, %25
  %28 = phi i1 [ %26, %25 ], [ %24, %22 ]
  br label %29

29:                                               ; preds = %27
  br i1 %28, label %30, label %154

30:                                               ; preds = %29
  %31 = call ptr @inNestedBlock()
  %32 = ptrtoint ptr %31 to i64
  %33 = icmp ne i64 %32, 0
  %34 = zext i1 %33 to i32
  %35 = and i32 %34, 1
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %40

37:                                               ; preds = %30
  %38 = call i32 @strcmp(ptr %31, ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8))
  %39 = icmp eq i32 %38, 0
  br label %42

40:                                               ; preds = %30
  %41 = icmp eq i64 %32, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8) to i64)
  br label %42

42:                                               ; preds = %37, %40
  %43 = phi i1 [ %41, %40 ], [ %39, %37 ]
  br label %44

44:                                               ; preds = %42
  br i1 %43, label %45, label %156

45:                                               ; preds = %44
  %46 = call ptr @callerCatches()
  %47 = ptrtoint ptr %46 to i64
  %48 = icmp ne i64 %47, 0
  %49 = zext i1 %48 to i32
  %50 = and i32 %49, 1
  %51 = icmp ne i32 %50, 0
  br i1 %51, label %52, label %55

52:                                               ; preds = %45
  %53 = call i32 @strcmp(ptr %46, ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8))
  %54 = icmp eq i32 %53, 0
  br label %57

55:                                               ; preds = %45
  %56 = icmp eq i64 %47, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8) to i64)
  br label %57

57:                                               ; preds = %52, %55
  %58 = phi i1 [ %56, %55 ], [ %54, %52 ]
  br label %59

59:                                               ; preds = %57
  br i1 %58, label %60, label %158

60:                                               ; preds = %59
  %61 = call ptr @innerCatchMismatch()
  %62 = ptrtoint ptr %61 to i64
  %63 = icmp ne i64 %62, 0
  %64 = zext i1 %63 to i32
  %65 = and i32 %64, 1
  %66 = icmp ne i32 %65, 0
  br i1 %66, label %67, label %70

67:                                               ; preds = %60
  %68 = call i32 @strcmp(ptr %61, ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8))
  %69 = icmp eq i32 %68, 0
  br label %72

70:                                               ; preds = %60
  %71 = icmp eq i64 %62, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17753883700566607152, i64 8) to i64)
  br label %72

72:                                               ; preds = %67, %70
  %73 = phi i1 [ %71, %70 ], [ %69, %67 ]
  br label %74

74:                                               ; preds = %72
  br i1 %73, label %75, label %160

75:                                               ; preds = %74
  %76 = call ptr @inLoop()
  %77 = ptrtoint ptr %76 to i64
  %78 = icmp ne i64 %77, 0
  %79 = zext i1 %78 to i32
  %80 = and i32 %79, 1
  %81 = icmp ne i32 %80, 0
  br i1 %81, label %82, label %85

82:                                               ; preds = %75
  %83 = call i32 @strcmp(ptr %76, ptr getelementptr inbounds nuw (i8, ptr @s_5655230408898239064, i64 8))
  %84 = icmp eq i32 %83, 0
  br label %87

85:                                               ; preds = %75
  %86 = icmp eq i64 %77, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5655230408898239064, i64 8) to i64)
  br label %87

87:                                               ; preds = %82, %85
  %88 = phi i1 [ %86, %85 ], [ %84, %82 ]
  br label %89

89:                                               ; preds = %87
  br i1 %88, label %90, label %162

90:                                               ; preds = %89
  %91 = call ptr @twoUsings()
  %92 = ptrtoint ptr %91 to i64
  %93 = icmp ne i64 %92, 0
  %94 = zext i1 %93 to i32
  %95 = and i32 %94, 1
  %96 = icmp ne i32 %95, 0
  br i1 %96, label %97, label %100

97:                                               ; preds = %90
  %98 = call i32 @strcmp(ptr %91, ptr getelementptr inbounds nuw (i8, ptr @s_15973155909069855405, i64 8))
  %99 = icmp eq i32 %98, 0
  br label %102

100:                                              ; preds = %90
  %101 = icmp eq i64 %92, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15973155909069855405, i64 8) to i64)
  br label %102

102:                                              ; preds = %97, %100
  %103 = phi i1 [ %101, %100 ], [ %99, %97 ]
  br label %104

104:                                              ; preds = %102
  br i1 %103, label %105, label %164

105:                                              ; preds = %104
  %106 = call ptr @bothThrow()
  %107 = ptrtoint ptr %106 to i64
  %108 = icmp ne i64 %107, 0
  %109 = zext i1 %108 to i32
  %110 = and i32 %109, 1
  %111 = icmp ne i32 %110, 0
  br i1 %111, label %112, label %115

112:                                              ; preds = %105
  %113 = call i32 @strcmp(ptr %106, ptr getelementptr inbounds nuw (i8, ptr @s_5427783902871050770, i64 8))
  %114 = icmp eq i32 %113, 0
  br label %117

115:                                              ; preds = %105
  %116 = icmp eq i64 %107, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5427783902871050770, i64 8) to i64)
  br label %117

117:                                              ; preds = %112, %115
  %118 = phi i1 [ %116, %115 ], [ %114, %112 ]
  br label %119

119:                                              ; preds = %117
  br i1 %118, label %120, label %166

120:                                              ; preds = %119
  %121 = call ptr @twoUsingsNormalPath()
  %122 = ptrtoint ptr %121 to i64
  %123 = icmp ne i64 %122, 0
  %124 = zext i1 %123 to i32
  %125 = and i32 %124, 1
  %126 = icmp ne i32 %125, 0
  br i1 %126, label %127, label %130

127:                                              ; preds = %120
  %128 = call i32 @strcmp(ptr %121, ptr getelementptr inbounds nuw (i8, ptr @s_15973155909069855405, i64 8))
  %129 = icmp eq i32 %128, 0
  br label %132

130:                                              ; preds = %120
  %131 = icmp eq i64 %122, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_15973155909069855405, i64 8) to i64)
  br label %132

132:                                              ; preds = %127, %130
  %133 = phi i1 [ %131, %130 ], [ %129, %127 ]
  br label %134

134:                                              ; preds = %132
  br i1 %133, label %135, label %168

135:                                              ; preds = %134
  %136 = call ptr @disposalDoesNotThrow()
  %137 = ptrtoint ptr %136 to i64
  %138 = icmp ne i64 %137, 0
  %139 = zext i1 %138 to i32
  %140 = and i32 %139, 1
  %141 = icmp ne i32 %140, 0
  br i1 %141, label %142, label %145

142:                                              ; preds = %135
  %143 = call i32 @strcmp(ptr %136, ptr getelementptr inbounds nuw (i8, ptr @s_301695387809324849, i64 8))
  %144 = icmp eq i32 %143, 0
  br label %147

145:                                              ; preds = %135
  %146 = icmp eq i64 %137, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_301695387809324849, i64 8) to i64)
  br label %147

147:                                              ; preds = %142, %145
  %148 = phi i1 [ %146, %145 ], [ %144, %142 ]
  br label %149

149:                                              ; preds = %147
  br i1 %148, label %150, label %170

150:                                              ; preds = %149
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

151:                                              ; preds = %170, %168, %166, %164, %162, %160, %158, %156, %154, %152
  unreachable

152:                                              ; preds = %14
  %153 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11910962264726028670, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 215, ptr null)
  br label %151

154:                                              ; preds = %29
  %155 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_568347090281078248, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 216, ptr null)
  br label %151

156:                                              ; preds = %44
  %157 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_718719723438898844, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 217, ptr null)
  br label %151

158:                                              ; preds = %59
  %159 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13865098954200123741, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 218, ptr null)
  br label %151

160:                                              ; preds = %74
  %161 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_115703159137015044, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 219, ptr null)
  br label %151

162:                                              ; preds = %89
  %163 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3960696774835903884, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 220, ptr null)
  br label %151

164:                                              ; preds = %104
  %165 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9465974079335641511, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 221, ptr null)
  br label %151

166:                                              ; preds = %119
  %167 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6742539089612662085, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 222, ptr null)
  br label %151

168:                                              ; preds = %134
  %169 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7386768615681113615, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 223, ptr null)
  br label %151

170:                                              ; preds = %149
  %171 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8345799095647167594, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_4766547723502998293, i64 8), i32 224, ptr null)
  br label %151
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
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

