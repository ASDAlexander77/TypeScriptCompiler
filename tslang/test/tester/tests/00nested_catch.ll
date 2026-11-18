; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_3523764787506715604 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFouter\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_13873912742287319313 = internal constant [36 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa catch inside a catch runs\00", align 8
@f_15587211099634377936 = internal constant [104 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00nested_catch.ts\00", align 8
@m_15152614229900045830 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe outer clause keeps its own type filter\00", align 8
@m_8565417460156110380 = internal constant [46 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthree levels of catch nesting all run\00", align 8
@m_2201400739337696627 = internal constant [66 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested try whose catch is not taken still runs its body\00", align 8
@m_2258669546068292500 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested try in a catch runs its own finally\00", align 8
@m_8399722909468139115 = internal constant [55 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFentering the nested regions repeatedly is fine\00", align 8
@m_16211019258082653797 = internal constant [59 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa nested catch clause can throw past the outer one\00", align 8
@m_16257237270521442278 = internal constant [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa try in a try body still works\00", align 8
@m_18182207650679986752 = internal constant [39 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa try in a finally still works\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@_ZTIi = external constant ptr
@_ZTIPc = external constant ptr

declare void @GC_enable_threads()

declare void @GC_init()

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #0

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress
define i32 @catchInCatch() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %33 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %11
  br label %33

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %2, align 4
  %18 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %14
  store i32 2, ptr %18, align 4
  invoke void @__cxa_throw(ptr %18, ptr @_ZTIi, ptr null)
          to label %33 unwind label %19

19:                                               ; preds = %.noexc
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %21 = extractvalue { ptr, i32 } %20, 1
  %22 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %27, label %24

24:                                               ; preds = %19
  %25 = extractvalue { ptr, i32 } %20, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  invoke void @__cxa_rethrow()
          to label %.noexc3 unwind label %endcatch.resume2

.noexc3:                                          ; preds = %24
  br label %33

27:                                               ; preds = %19
  %28 = extractvalue { ptr, i32 } %20, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  %30 = load i32, ptr %29, align 4
  store i32 %30, ptr %1, align 4
  store i32 11, ptr %3, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  %31 = load i32, ptr %3, align 4
  store i32 %31, ptr %4, align 4
  %32 = load i32, ptr %4, align 4
  ret i32 %32

33:                                               ; preds = %.noexc3, %.noexc, %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %11, %14
  %34 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %34

endcatch.resume2:                                 ; preds = %24
  %35 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %35
}

; Function Attrs: mustprogress
define i32 @outerFilterIsItsOwn() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_3523764787506715604, i64 8), ptr %5, align 8
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIPc, ptr null)
          to label %34 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIPc
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPc)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %11
  br label %34

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  store ptr %16, ptr %2, align 8
  store i32 1, ptr %3, align 4
  %17 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %14
  store i32 2, ptr %17, align 4
  invoke void @__cxa_throw(ptr %17, ptr @_ZTIi, ptr null)
          to label %34 unwind label %18

18:                                               ; preds = %.noexc
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
          to label %.noexc3 unwind label %endcatch.resume2

.noexc3:                                          ; preds = %23
  br label %34

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  %29 = load i32, ptr %28, align 4
  store i32 %29, ptr %1, align 4
  %30 = load i32, ptr %3, align 4
  %31 = add i32 %30, 10
  store i32 %31, ptr %3, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  %32 = load i32, ptr %3, align 4
  store i32 %32, ptr %4, align 4
  %33 = load i32, ptr %4, align 4
  ret i32 %33

34:                                               ; preds = %.noexc3, %.noexc, %.noexc1, %0
  unreachable

endcatch.resume:                                  ; preds = %11, %14
  %35 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %35

endcatch.resume2:                                 ; preds = %23
  %36 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %36
}

; Function Attrs: mustprogress
define i32 @threeDeep() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  %6 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %6, align 4
  invoke void @__cxa_throw(ptr %6, ptr @_ZTIi, ptr null)
          to label %51 unwind label %7

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = extractvalue { ptr, i32 } %8, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %12
  br label %51

15:                                               ; preds = %7
  %16 = extractvalue { ptr, i32 } %8, 0
  %17 = call ptr @__cxa_begin_catch(ptr %16)
  %18 = load i32, ptr %17, align 4
  store i32 %18, ptr %3, align 4
  %19 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %15
  store i32 2, ptr %19, align 4
  invoke void @__cxa_throw(ptr %19, ptr @_ZTIi, ptr null)
          to label %51 unwind label %20

20:                                               ; preds = %.noexc3
  %21 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %22 = extractvalue { ptr, i32 } %21, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %24 = icmp eq i32 %22, %23
  br i1 %24, label %28, label %25

25:                                               ; preds = %20
  %26 = extractvalue { ptr, i32 } %21, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  invoke void @__cxa_rethrow()
          to label %.noexc5 unwind label %endcatch.resume4

.noexc5:                                          ; preds = %25
  br label %51

28:                                               ; preds = %20
  %29 = extractvalue { ptr, i32 } %21, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  %31 = load i32, ptr %30, align 4
  store i32 %31, ptr %2, align 4
  %32 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc6 unwind label %endcatch.resume4

.noexc6:                                          ; preds = %28
  store i32 3, ptr %32, align 4
  invoke void @__cxa_throw(ptr %32, ptr @_ZTIi, ptr null)
          to label %51 unwind label %33

33:                                               ; preds = %.noexc6
  %34 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %35 = extractvalue { ptr, i32 } %34, 1
  %36 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %37 = icmp eq i32 %35, %36
  br i1 %37, label %41, label %38

38:                                               ; preds = %33
  %39 = extractvalue { ptr, i32 } %34, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  invoke void @__cxa_rethrow()
          to label %.noexc2 unwind label %endcatch.resume1

.noexc2:                                          ; preds = %38
  br label %51

41:                                               ; preds = %33
  %42 = extractvalue { ptr, i32 } %34, 0
  %43 = call ptr @__cxa_begin_catch(ptr %42)
  %44 = load i32, ptr %43, align 4
  store i32 %44, ptr %1, align 4
  store i32 100, ptr %4, align 4
  call void @__cxa_end_catch()
  %45 = load i32, ptr %4, align 4
  %46 = add i32 %45, 10
  store i32 %46, ptr %4, align 4
  call void @__cxa_end_catch()
  %47 = load i32, ptr %4, align 4
  %48 = add i32 %47, 1
  store i32 %48, ptr %4, align 4
  call void @__cxa_end_catch()
  %49 = load i32, ptr %4, align 4
  store i32 %49, ptr %5, align 4
  %50 = load i32, ptr %5, align 4
  ret i32 %50

51:                                               ; preds = %.noexc2, %.noexc6, %.noexc5, %.noexc3, %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %15, %12
  %52 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %52

endcatch.resume1:                                 ; preds = %38
  %53 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %53

endcatch.resume4:                                 ; preds = %28, %25
  %54 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %54
}

; Function Attrs: mustprogress
define i32 @innerCatchNotTaken() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %19 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %10
  br label %19

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i32, ptr %15, align 4
  store i32 %16, ptr %1, align 4
  store i32 5, ptr %2, align 4
  call void @__cxa_end_catch()
  %17 = load i32, ptr %2, align 4
  store i32 %17, ptr %3, align 4
  %18 = load i32, ptr %3, align 4
  ret i32 %18

19:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %20 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %20
}

; Function Attrs: mustprogress
define i32 @nestedTryFinallyInCatch() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %41 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %11
  br label %41

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %2, align 4
  %18 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %14
  store i32 2, ptr %18, align 4
  invoke void @__cxa_throw(ptr %18, ptr @_ZTIi, ptr null)
          to label %41 unwind label %19

19:                                               ; preds = %.noexc1
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %21 = extractvalue { ptr, i32 } %20, 1
  %22 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %27, label %24

24:                                               ; preds = %19
  %25 = extractvalue { ptr, i32 } %20, 0
  %26 = call ptr @__cxa_begin_catch(ptr %25)
  invoke void @__cxa_rethrow()
          to label %41 unwind label %endcatch.pad

27:                                               ; preds = %19
  %28 = extractvalue { ptr, i32 } %20, 0
  %29 = call ptr @__cxa_begin_catch(ptr %28)
  %30 = load i32, ptr %29, align 4
  store i32 %30, ptr %1, align 4
  store i32 7, ptr %3, align 4
  call void @__cxa_end_catch()
  %31 = load i32, ptr %3, align 4
  %32 = add i32 %31, 1
  store i32 %32, ptr %3, align 4
  call void @__cxa_end_catch()
  %33 = load i32, ptr %3, align 4
  store i32 %33, ptr %4, align 4
  %34 = load i32, ptr %4, align 4
  ret i32 %34

endcatch.pad:                                     ; preds = %24
  %35 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

36:                                               ; No predecessors!
  %37 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %36
  %38 = phi { ptr, i32 } [ %37, %36 ], [ %35, %endcatch.pad ]
  %39 = load i32, ptr %3, align 4
  %40 = add i32 %39, 1
  store i32 %40, ptr %3, align 4
  call void @__cxa_end_catch()
  resume { ptr, i32 } %38

41:                                               ; preds = %24, %.noexc1, %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %14, %11
  %42 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %42
}

; Function Attrs: mustprogress
define i32 @catchInCatchInLoop() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  store i32 0, ptr %3, align 4
  br label %6

6:                                                ; preds = %32, %0
  %7 = load i32, ptr %3, align 4
  %8 = icmp slt i32 %7, 3
  br i1 %8, label %9, label %40

9:                                                ; preds = %6
  %10 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %10, align 4
  invoke void @__cxa_throw(ptr %10, ptr @_ZTIi, ptr null)
          to label %43 unwind label %11

11:                                               ; preds = %9
  %12 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %13 = extractvalue { ptr, i32 } %12, 1
  %14 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %15 = icmp eq i32 %13, %14
  br i1 %15, label %19, label %16

16:                                               ; preds = %11
  %17 = extractvalue { ptr, i32 } %12, 0
  %18 = call ptr @__cxa_begin_catch(ptr %17)
  invoke void @__cxa_rethrow()
          to label %.noexc3 unwind label %endcatch.resume1

.noexc3:                                          ; preds = %16
  br label %43

19:                                               ; preds = %11
  %20 = extractvalue { ptr, i32 } %12, 0
  %21 = call ptr @__cxa_begin_catch(ptr %20)
  %22 = load i32, ptr %21, align 4
  store i32 %22, ptr %2, align 4
  %23 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc2 unwind label %endcatch.resume1

.noexc2:                                          ; preds = %19
  store i32 2, ptr %23, align 4
  invoke void @__cxa_throw(ptr %23, ptr @_ZTIi, ptr null)
          to label %43 unwind label %24

24:                                               ; preds = %.noexc2
  %25 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %26 = extractvalue { ptr, i32 } %25, 1
  %27 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %28 = icmp eq i32 %26, %27
  br i1 %28, label %32, label %29

29:                                               ; preds = %24
  %30 = extractvalue { ptr, i32 } %25, 0
  %31 = call ptr @__cxa_begin_catch(ptr %30)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %29
  br label %43

32:                                               ; preds = %24
  %33 = extractvalue { ptr, i32 } %25, 0
  %34 = call ptr @__cxa_begin_catch(ptr %33)
  %35 = load i32, ptr %34, align 4
  store i32 %35, ptr %1, align 4
  %36 = load i32, ptr %4, align 4
  %37 = add i32 %36, 2
  store i32 %37, ptr %4, align 4
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  %38 = load i32, ptr %3, align 4
  %39 = add i32 %38, 1
  store i32 %39, ptr %3, align 4
  br label %6

40:                                               ; preds = %6
  %41 = load i32, ptr %4, align 4
  store i32 %41, ptr %5, align 4
  %42 = load i32, ptr %5, align 4
  ret i32 %42

43:                                               ; preds = %.noexc, %.noexc2, %.noexc3, %9
  unreachable

endcatch.resume:                                  ; preds = %29
  %44 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %44

endcatch.resume1:                                 ; preds = %16, %19
  %45 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %45
}

; Function Attrs: mustprogress
define i32 @nestedCatchThrowsOn() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %3, align 4
  invoke void @__cxa_throw(ptr %3, ptr @_ZTIi, ptr null)
          to label %30 unwind label %4

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %6 = extractvalue { ptr, i32 } %5, 1
  %7 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %8 = icmp eq i32 %6, %7
  br i1 %8, label %12, label %9

9:                                                ; preds = %4
  %10 = extractvalue { ptr, i32 } %5, 0
  %11 = call ptr @__cxa_begin_catch(ptr %10)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %9
  br label %30

12:                                               ; preds = %4
  %13 = extractvalue { ptr, i32 } %5, 0
  %14 = call ptr @__cxa_begin_catch(ptr %13)
  %15 = load i32, ptr %14, align 4
  store i32 %15, ptr %2, align 4
  %16 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %12
  store i32 2, ptr %16, align 4
  invoke void @__cxa_throw(ptr %16, ptr @_ZTIi, ptr null)
          to label %30 unwind label %17

17:                                               ; preds = %.noexc1
  %18 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %19 = extractvalue { ptr, i32 } %18, 1
  %20 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %21 = icmp eq i32 %19, %20
  br i1 %21, label %25, label %22

22:                                               ; preds = %17
  %23 = extractvalue { ptr, i32 } %18, 0
  %24 = call ptr @__cxa_begin_catch(ptr %23)
  invoke void @__cxa_rethrow()
          to label %.noexc5 unwind label %endcatch.resume4

.noexc5:                                          ; preds = %22
  br label %30

25:                                               ; preds = %17
  %26 = extractvalue { ptr, i32 } %18, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  %28 = load i32, ptr %27, align 4
  store i32 %28, ptr %1, align 4
  call void @__cxa_end_catch()
  %29 = invoke ptr @__cxa_allocate_exception(i64 4)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %25
  store i32 3, ptr %29, align 4
  invoke void @__cxa_throw(ptr %29, ptr @_ZTIi, ptr null)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %.noexc2
  br label %30

30:                                               ; preds = %.noexc3, %.noexc5, %.noexc1, %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %.noexc2, %25, %12, %9
  %31 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %31

endcatch.resume4:                                 ; preds = %22
  %32 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  call void @__cxa_end_catch()
  resume { ptr, i32 } %32
}

; Function Attrs: mustprogress
define i32 @tryInTryBody() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  %5 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %5, align 4
  invoke void @__cxa_throw(ptr %5, ptr @_ZTIi, ptr null)
          to label %35 unwind label %6

6:                                                ; preds = %0
  %7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %8 = extractvalue { ptr, i32 } %7, 1
  %9 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %14, label %11

11:                                               ; preds = %6
  %12 = extractvalue { ptr, i32 } %7, 0
  %13 = call ptr @__cxa_begin_catch(ptr %12)
  invoke void @__cxa_rethrow()
          to label %35 unwind label %endcatch.pad

14:                                               ; preds = %6
  %15 = extractvalue { ptr, i32 } %7, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = load i32, ptr %16, align 4
  store i32 %17, ptr %1, align 4
  store i32 11, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %32

endcatch.pad:                                     ; preds = %11
  %18 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  call void @__cxa_end_catch()
  br label %.caught

19:                                               ; No predecessors!
  %20 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %19
  %21 = phi { ptr, i32 } [ %20, %19 ], [ %18, %endcatch.pad ]
  %22 = extractvalue { ptr, i32 } %21, 1
  %23 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %24 = icmp eq i32 %22, %23
  br i1 %24, label %28, label %25

25:                                               ; preds = %.caught
  %26 = extractvalue { ptr, i32 } %21, 0
  %27 = call ptr @__cxa_begin_catch(ptr %26)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %25
  br label %35

28:                                               ; preds = %.caught
  %29 = extractvalue { ptr, i32 } %21, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  %31 = load i32, ptr %30, align 4
  store i32 %31, ptr %2, align 4
  store i32 1, ptr %3, align 4
  call void @__cxa_end_catch()
  br label %32

32:                                               ; preds = %14, %28
  %33 = load i32, ptr %3, align 4
  store i32 %33, ptr %4, align 4
  %34 = load i32, ptr %4, align 4
  ret i32 %34

35:                                               ; preds = %.noexc, %11, %0
  unreachable

endcatch.resume:                                  ; preds = %25
  %36 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %36
}

; Function Attrs: mustprogress
define i32 @tryInFinally() #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  store i32 1, ptr %2, align 4
  %4 = call ptr @__cxa_allocate_exception(i64 4)
  store i32 2, ptr %4, align 4
  invoke void @__cxa_throw(ptr %4, ptr @_ZTIi, ptr null)
          to label %21 unwind label %5

5:                                                ; preds = %0
  %6 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %7 = extractvalue { ptr, i32 } %6, 1
  %8 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %5
  %11 = extractvalue { ptr, i32 } %6, 0
  %12 = call ptr @__cxa_begin_catch(ptr %11)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %10
  br label %21

13:                                               ; preds = %5
  %14 = extractvalue { ptr, i32 } %6, 0
  %15 = call ptr @__cxa_begin_catch(ptr %14)
  %16 = load i32, ptr %15, align 4
  store i32 %16, ptr %1, align 4
  %17 = load i32, ptr %2, align 4
  %18 = add i32 %17, 10
  store i32 %18, ptr %2, align 4
  call void @__cxa_end_catch()
  %19 = load i32, ptr %2, align 4
  store i32 %19, ptr %3, align 4
  %20 = load i32, ptr %3, align 4
  ret i32 %20

21:                                               ; preds = %.noexc, %0
  unreachable

endcatch.resume:                                  ; preds = %10
  %22 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %22
}

; Function Attrs: mustprogress
define i32 @main() #1 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca i32, align 4
  %2 = alloca i1, align 1
  %3 = call i32 @catchInCatch()
  %4 = icmp eq i32 %3, 11
  br i1 %4, label %5, label %45

5:                                                ; preds = %0
  %6 = call i32 @outerFilterIsItsOwn()
  %7 = icmp eq i32 %6, 11
  br i1 %7, label %8, label %47

8:                                                ; preds = %5
  %9 = call i32 @threeDeep()
  %10 = icmp eq i32 %9, 111
  br i1 %10, label %11, label %49

11:                                               ; preds = %8
  %12 = call i32 @innerCatchNotTaken()
  %13 = icmp eq i32 %12, 5
  br i1 %13, label %14, label %51

14:                                               ; preds = %11
  %15 = call i32 @nestedTryFinallyInCatch()
  %16 = icmp eq i32 %15, 8
  br i1 %16, label %17, label %53

17:                                               ; preds = %14
  %18 = call i32 @catchInCatchInLoop()
  %19 = icmp eq i32 %18, 6
  br i1 %19, label %20, label %55

20:                                               ; preds = %17
  store i1 false, ptr %2, align 1
  %21 = invoke i32 @nestedCatchThrowsOn()
          to label %22 unwind label %23

22:                                               ; preds = %20
  br label %35

23:                                               ; preds = %20
  %24 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %25 = extractvalue { ptr, i32 } %24, 1
  %26 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %27 = icmp eq i32 %25, %26
  br i1 %27, label %31, label %28

28:                                               ; preds = %23
  %29 = extractvalue { ptr, i32 } %24, 0
  %30 = call ptr @__cxa_begin_catch(ptr %29)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %28
  br label %44

31:                                               ; preds = %23
  %32 = extractvalue { ptr, i32 } %24, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  %34 = load i32, ptr %33, align 4
  store i32 %34, ptr %1, align 4
  store i1 true, ptr %2, align 1
  call void @__cxa_end_catch()
  br label %35

35:                                               ; preds = %22, %31
  %36 = load i1, ptr %2, align 1
  br i1 %36, label %37, label %57

37:                                               ; preds = %35
  %38 = call i32 @tryInTryBody()
  %39 = icmp eq i32 %38, 11
  br i1 %39, label %40, label %59

40:                                               ; preds = %37
  %41 = call i32 @tryInFinally()
  %42 = icmp eq i32 %41, 11
  br i1 %42, label %43, label %61

43:                                               ; preds = %40
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

44:                                               ; preds = %61, %59, %57, %.noexc, %55, %53, %51, %49, %47, %45
  unreachable

45:                                               ; preds = %0
  %46 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13873912742287319313, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 203, ptr null)
  br label %44

47:                                               ; preds = %5
  %48 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15152614229900045830, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 204, ptr null)
  br label %44

49:                                               ; preds = %8
  %50 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8565417460156110380, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 205, ptr null)
  br label %44

51:                                               ; preds = %11
  %52 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2201400739337696627, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 206, ptr null)
  br label %44

53:                                               ; preds = %14
  %54 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2258669546068292500, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 207, ptr null)
  br label %44

55:                                               ; preds = %17
  %56 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8399722909468139115, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 208, ptr null)
  br label %44

57:                                               ; preds = %35
  %58 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16211019258082653797, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 218, ptr null)
  br label %44

59:                                               ; preds = %37
  %60 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16257237270521442278, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 220, ptr null)
  br label %44

61:                                               ; preds = %40
  %62 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18182207650679986752, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_15587211099634377936, i64 8), i32 221, ptr null)
  br label %44

endcatch.resume:                                  ; preds = %28
  %63 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %63
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
attributes #1 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

