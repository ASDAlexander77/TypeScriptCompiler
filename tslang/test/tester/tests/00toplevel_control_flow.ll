; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Counter = type { ptr, double }

$Counter..size = comdat any

@s_11320984242218861763 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCounter\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_730899674924722773 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFother\00", align 8
@s_6517818557876028835 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthree\00", align 8
@s_1875936269717626031 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFone\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@m_4001038610594322416 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfor-if-continue-break\00", align 8
@f_7388770853745162733 = internal constant [113 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00toplevel_control_flow.ts\00", align 8
@m_14882043299657492846 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwhile\00", align 8
@m_11936925801837197745 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFswitch\00", align 8
@m_14600317178434072784 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtry-catch\00", align 8
@m_795331291742652644 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally-ran\00", align 8
@m_735719861432885908 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclosure-capture-root-let\00", align 8
@m_15721269842407136078 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclass-at-root\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@Counter..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11320984242218861763, i64 8)
@Counter..size = global i64 16, comdat
@Counter..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Counter..instanceOf, ptr @Counter..new, ptr @Counter.increment, ptr @Counter..rtti, ptr @Counter..size }
@sum = private global i32 0
@count = private global i32 0
@label = private global ptr null
@caught = private global i1 false
@_ZTId = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTIf = external constant ptr
@counter = private global i32 0
@c = private global ptr null
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1000, ptr @.mainL1C1FH620438381, ptr null }]

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @__cxa_rethrow()

declare void @puts(ptr)

declare void @__cxa_end_catch()

declare ptr @__cxa_current_exception_type()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @ts.internal.eh_try(i32)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i32 @__gxx_personality_v0(...)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define void @Counter.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Counter, ptr %3, i32 0, i32 1
  store double 0.000000e+00, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Counter..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Counter..rtti, align 8
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
define ptr @Counter..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Counter, ptr %3, i32 0, i32 0
  store ptr @Counter..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Counter.increment(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Counter, ptr %3, i32 0, i32 1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr %Counter, ptr %5, i32 0, i32 1
  %7 = load double, ptr %6, align 8
  %8 = fadd double %7, 1.000000e+00
  store double %8, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define void @.afL51C19FH620438381() #2 {
  %1 = load i32, ptr @counter, align 4
  %2 = add i32 %1, 1
  store i32 %2, ptr @counter, align 4
  ret void
}

; Function Attrs: mustprogress
define void @.mainL1C1FH620438381() #3 personality ptr @__gxx_personality_v0 {
  %1 = alloca double, align 8
  %2 = alloca i32, align 4
  store i32 0, ptr @sum, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %16, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 5
  br i1 %5, label %6, label %19

6:                                                ; preds = %3
  %7 = load i32, ptr %2, align 4
  %8 = icmp eq i32 %7, 2
  br i1 %8, label %16, label %9

9:                                                ; preds = %6
  %10 = load i32, ptr %2, align 4
  %11 = icmp eq i32 %10, 4
  br i1 %11, label %19, label %12

12:                                               ; preds = %9
  %13 = load i32, ptr @sum, align 4
  %14 = load i32, ptr %2, align 4
  %15 = add i32 %13, %14
  store i32 %15, ptr @sum, align 4
  br label %16

16:                                               ; preds = %12, %6
  %17 = load i32, ptr %2, align 4
  %18 = add i32 %17, 1
  store i32 %18, ptr %2, align 4
  br label %3

19:                                               ; preds = %9, %3
  %20 = load i32, ptr @sum, align 4
  %21 = icmp eq i32 %20, 4
  br i1 %21, label %22, label %139

22:                                               ; preds = %19
  store i32 0, ptr @count, align 4
  br label %23

23:                                               ; preds = %26, %22
  %24 = load i32, ptr @count, align 4
  %25 = icmp slt i32 %24, 3
  br i1 %25, label %26, label %29

26:                                               ; preds = %23
  %27 = load i32, ptr @count, align 4
  %28 = add i32 %27, 1
  store i32 %28, ptr @count, align 4
  br label %23

29:                                               ; preds = %23
  %30 = load i32, ptr @count, align 4
  %31 = icmp eq i32 %30, 3
  br i1 %31, label %32, label %141

32:                                               ; preds = %29
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @label, align 8
  %33 = load i32, ptr @count, align 4
  %34 = icmp eq i32 %33, 1
  br i1 %34, label %35, label %36

35:                                               ; preds = %32
  store ptr getelementptr inbounds nuw (i8, ptr @s_1875936269717626031, i64 8), ptr @label, align 8
  br label %40

36:                                               ; preds = %32
  %37 = icmp eq i32 %33, 3
  br i1 %37, label %38, label %39

38:                                               ; preds = %36
  store ptr getelementptr inbounds nuw (i8, ptr @s_6517818557876028835, i64 8), ptr @label, align 8
  br label %40

39:                                               ; preds = %36
  store ptr getelementptr inbounds nuw (i8, ptr @s_730899674924722773, i64 8), ptr @label, align 8
  br label %40

40:                                               ; preds = %35, %38, %39
  %41 = load ptr, ptr @label, align 8
  %42 = ptrtoint ptr %41 to i64
  %43 = icmp ne i64 %42, 0
  %44 = zext i1 %43 to i32
  %45 = and i32 %44, 1
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %50

47:                                               ; preds = %40
  %48 = call i32 @strcmp(ptr %41, ptr getelementptr inbounds nuw (i8, ptr @s_6517818557876028835, i64 8))
  %49 = icmp eq i32 %48, 0
  br label %52

50:                                               ; preds = %40
  %51 = icmp eq i64 %42, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6517818557876028835, i64 8) to i64)
  br label %52

52:                                               ; preds = %47, %50
  %53 = phi i1 [ %51, %50 ], [ %49, %47 ]
  br label %54

54:                                               ; preds = %52
  br i1 %53, label %55, label %143

55:                                               ; preds = %54
  store i1 false, ptr @caught, align 1
  %56 = call ptr @__cxa_allocate_exception(i64 8)
  store double 4.200000e+01, ptr %56, align 8
  invoke void @__cxa_throw(ptr %56, ptr @_ZTId, ptr null)
          to label %138 unwind label %57

57:                                               ; preds = %55
  %58 = landingpad { ptr, i32 }
          catch ptr @_ZTId
          catch ptr @_ZTIi
          catch ptr @_ZTIj
          catch ptr @_ZTIf
          catch ptr null
  %59 = extractvalue { ptr, i32 } %58, 1
  %60 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTId)
  %61 = icmp eq i32 %59, %60
  %62 = extractvalue { ptr, i32 } %58, 1
  %63 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %64 = icmp eq i32 %62, %63
  %65 = extractvalue { ptr, i32 } %58, 1
  %66 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIj)
  %67 = icmp eq i32 %65, %66
  %68 = extractvalue { ptr, i32 } %58, 1
  %69 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIf)
  %70 = icmp eq i32 %68, %69
  %71 = or i1 %61, %64
  %72 = or i1 %71, %67
  %73 = or i1 %72, %70
  br i1 %73, label %77, label %74

74:                                               ; preds = %57
  %75 = extractvalue { ptr, i32 } %58, 0
  %76 = call ptr @__cxa_begin_catch(ptr %75)
  invoke void @__cxa_rethrow()
          to label %138 unwind label %endcatch.pad

77:                                               ; preds = %57
  %78 = extractvalue { ptr, i32 } %58, 0
  %79 = call ptr @__cxa_begin_catch(ptr %78)
  %80 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %77
  %81 = icmp eq ptr %80, @_ZTIi
  br i1 %81, label %82, label %85

82:                                               ; preds = %.noexc
  %83 = load i32, ptr %79, align 4
  %84 = sitofp i32 %83 to double
  br label %103

85:                                               ; preds = %.noexc
  %86 = icmp eq ptr %80, @_ZTIj
  br i1 %86, label %87, label %90

87:                                               ; preds = %85
  %88 = load i32, ptr %79, align 4
  %89 = uitofp i32 %88 to double
  br label %100

90:                                               ; preds = %85
  %91 = icmp eq ptr %80, @_ZTIf
  br i1 %91, label %92, label %95

92:                                               ; preds = %90
  %93 = load float, ptr %79, align 4
  %94 = fpext float %93 to double
  br label %97

95:                                               ; preds = %90
  %96 = load double, ptr %79, align 8
  br label %97

97:                                               ; preds = %92, %95
  %98 = phi double [ %96, %95 ], [ %94, %92 ]
  br label %99

99:                                               ; preds = %97
  br label %100

100:                                              ; preds = %87, %99
  %101 = phi double [ %98, %99 ], [ %89, %87 ]
  br label %102

102:                                              ; preds = %100
  br label %103

103:                                              ; preds = %82, %102
  %104 = phi double [ %101, %102 ], [ %84, %82 ]
  br label %105

105:                                              ; preds = %103
  store double %104, ptr %1, align 8
  %106 = load double, ptr %1, align 8
  %107 = fcmp oeq double %106, 4.200000e+01
  store i1 %107, ptr @caught, align 1
  call void @__cxa_end_catch()
  %108 = load i32, ptr @sum, align 4
  %109 = add i32 %108, 100
  store i32 %109, ptr @sum, align 4
  %110 = load i1, ptr @caught, align 1
  br i1 %110, label %111, label %145

111:                                              ; preds = %105
  %112 = load i32, ptr @sum, align 4
  %113 = icmp eq i32 %112, 104
  br i1 %113, label %114, label %147

114:                                              ; preds = %111
  store i32 0, ptr @counter, align 4
  call void @.afL51C19FH620438381()
  call void @.afL51C19FH620438381()
  %115 = load i32, ptr @counter, align 4
  %116 = icmp eq i32 %115, 2
  br i1 %116, label %117, label %149

117:                                              ; preds = %114
  %118 = call ptr @GC_malloc(i64 24)
  %119 = getelementptr i8, ptr %118, i64 8
  %120 = getelementptr %Counter, ptr %119, i32 0, i32 0
  store ptr @Counter..vtbl, ptr %120, align 8
  call void @Counter.constructor(ptr %119)
  store ptr %119, ptr @c, align 8
  %121 = load ptr, ptr @c, align 8
  %122 = getelementptr %Counter, ptr %121, i32 0, i32 0
  call void @Counter.increment(ptr %121)
  %123 = load ptr, ptr @c, align 8
  %124 = getelementptr %Counter, ptr %123, i32 0, i32 0
  call void @Counter.increment(ptr %123)
  %125 = load ptr, ptr @c, align 8
  %126 = getelementptr %Counter, ptr %125, i32 0, i32 0
  call void @Counter.increment(ptr %125)
  %127 = load ptr, ptr @c, align 8
  %128 = getelementptr %Counter, ptr %127, i32 0, i32 1
  %129 = load double, ptr %128, align 8
  %130 = fcmp oeq double %129, 3.000000e+00
  br i1 %130, label %131, label %151

131:                                              ; preds = %117
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret void

endcatch.pad:                                     ; preds = %74
  %132 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

133:                                              ; No predecessors!
  %134 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %133
  %135 = phi { ptr, i32 } [ %134, %133 ], [ %132, %endcatch.pad ]
  %136 = load i32, ptr @sum, align 4
  %137 = add i32 %136, 100
  store i32 %137, ptr @sum, align 4
  resume { ptr, i32 } %135

138:                                              ; preds = %151, %149, %147, %145, %74, %55, %143, %141, %139
  unreachable

139:                                              ; preds = %19
  %140 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4001038610594322416, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 17, ptr null)
  br label %138

141:                                              ; preds = %29
  %142 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14882043299657492846, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 23, ptr null)
  br label %138

143:                                              ; preds = %54
  %144 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11936925801837197745, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 37, ptr null)
  br label %138

145:                                              ; preds = %105
  %146 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_14600317178434072784, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 47, ptr null)
  br label %138

147:                                              ; preds = %111
  %148 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_795331291742652644, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 48, ptr null)
  br label %138

149:                                              ; preds = %114
  %150 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_735719861432885908, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 56, ptr null)
  br label %138

151:                                              ; preds = %117
  %152 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_15721269842407136078, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_7388770853745162733, i64 8), i32 69, ptr null)
  br label %138

endcatch.resume:                                  ; preds = %77
  %153 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %153
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

