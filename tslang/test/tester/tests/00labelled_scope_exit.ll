; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Res = type { ptr, ptr }

$Res..size = comdat any

@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_12638145419113769978 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF;\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_3263991042789642887 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner\00", align 8
@s_3523764787506715604 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFouter\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_15839897912459002582 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFafter;\00", align 8
@s_3446774449893561193 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunreached;\00", align 8
@s_20852345718408352 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFblk\00", align 8
@s_619343937915704565 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfn\00", align 8
@s_12638194897137039473 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFn\00", align 8
@s_14793735007222968981 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbody\00", align 8
@s_12638214688346347271 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFx\00", align 8
@s_12638213588834719060 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFy\00", align 8
@s_584909432750204567 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF: \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_1741907030045678937 = internal constant [111 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00labelled_scope_exit.ts\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_643251718754793924 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFyx\00", align 8
@s_13832720603085879183 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFxxx\00", align 8
@s_4272736271669636967 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFloop bodies without braces\00", align 8
@s_15402960112721072915 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFafter;fn;\00", align 8
@s_9005461004495928640 = internal constant [45 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFunlabelled break in a labelled block\00", align 8
@s_4705311095735501259 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFblk0;body0;after;\00", align 8
@s_7174868576751045779 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcontinue a labelled block\00", align 8
@s_6833090846104757368 = internal constant [27 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFblk0;blk1;blk2;n3;\00", align 8
@s_15799391499023530575 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak a labelled block\00", align 8
@s_13533501543527038415 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFblk;after;fn;\00", align 8
@s_3961742956203555479 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbreak outer from an if\00", align 8
@s_1762663484488150012 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner;outer;\00", align 8
@s_8219663663114267555 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFcontinue outer from an if\00", align 8
@s_1297198650753530112 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinner0;outer0;inner0;outer1;\00", align 8
@m_8261687135864320793 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring, break a block\00", align 8
@m_10284318878980036539 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring, continue a block\00", align 8
@m_9482693678377368718 = internal constant [36 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstring, break a labelled if\00", align 8
@trail = private global ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 16, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare void @__cxa_rethrow()

declare ptr @__cxa_begin_catch(ptr)

declare void @ts.internal.eh_try_pad(i32)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare i32 @strcmp(ptr, ptr)

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
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.constructor(ptr %0, ptr %1) #1 {
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
define void @Res.dispose(ptr %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr @trail, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Res, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = icmp eq ptr %6, null
  %8 = select i1 %7, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %9 = call i64 @strlen(ptr %8)
  %10 = add i64 1, %9
  %11 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %12 = add i64 %10, %11
  %13 = add i64 %12, 8
  %14 = call ptr @GC_malloc(i64 %13)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = getelementptr i8, ptr %15, i64 0
  %17 = sub i64 %12, 0
  %18 = call ptr @__memcpy_chk(ptr %16, ptr %8, i64 %9, i64 %17)
  %19 = add i64 0, %9
  %20 = getelementptr i8, ptr %15, i64 %19
  %21 = sub i64 %12, %19
  %22 = call ptr @__memcpy_chk(ptr %20, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %11, i64 %21)
  %23 = add i64 %19, %11
  %24 = getelementptr i8, ptr %15, i64 %23
  store i8 0, ptr %24, align 1
  %25 = icmp eq ptr %3, null
  %26 = select i1 %25, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  %27 = icmp eq ptr %15, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %29 = call i64 @strlen(ptr %26)
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr %28)
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %26, i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %28, i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  store ptr %35, ptr @trail, align 8
  ret void
}

; Function Attrs: mustprogress
define void @continueOuterFromIf() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i1, align 1
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store i32 0, ptr %6, align 4
  br label %7

7:                                                ; preds = %128, %0
  %8 = load i32, ptr %6, align 4
  %9 = icmp slt i32 %8, 2
  br i1 %9, label %10, label %131

10:                                               ; preds = %7
  store i1 false, ptr %4, align 1
  %11 = call ptr @GC_malloc(i64 24)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr %Res, ptr %12, i32 0, i32 0
  store ptr @Res..vtbl, ptr %13, align 8
  %14 = load i32, ptr %6, align 4
  %15 = call ptr @GC_malloc_atomic(i64 58)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %16, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %14)
  %18 = icmp eq ptr %16, null
  %19 = select i1 %18, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %16
  %20 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3523764787506715604, i64 8))
  %21 = add i64 1, %20
  %22 = call i64 @strlen(ptr %19)
  %23 = add i64 %21, %22
  %24 = add i64 %23, 8
  %25 = call ptr @GC_malloc(i64 %24)
  %26 = getelementptr i8, ptr %25, i64 8
  %27 = getelementptr i8, ptr %26, i64 0
  %28 = sub i64 %23, 0
  %29 = call ptr @__memcpy_chk(ptr %27, ptr getelementptr inbounds nuw (i8, ptr @s_3523764787506715604, i64 8), i64 %20, i64 %28)
  %30 = add i64 0, %20
  %31 = getelementptr i8, ptr %26, i64 %30
  %32 = sub i64 %23, %30
  %33 = call ptr @__memcpy_chk(ptr %31, ptr %19, i64 %22, i64 %32)
  %34 = add i64 %30, %22
  %35 = getelementptr i8, ptr %26, i64 %34
  store i8 0, ptr %35, align 1
  invoke void @Res.constructor(ptr %12, ptr %26)
          to label %36 unwind label %120

36:                                               ; preds = %10
  store ptr %12, ptr %5, align 8
  store i1 true, ptr %4, align 1
  store i32 0, ptr %3, align 4
  br label %37

37:                                               ; preds = %88, %36
  %38 = load i32, ptr %3, align 4
  %39 = icmp slt i32 %38, 2
  br i1 %39, label %40, label %112

40:                                               ; preds = %37
  store i1 false, ptr %1, align 1
  %41 = call ptr @GC_malloc(i64 24)
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = getelementptr %Res, ptr %42, i32 0, i32 0
  store ptr @Res..vtbl, ptr %43, align 8
  %44 = load i32, ptr %3, align 4
  %45 = call ptr @GC_malloc_atomic(i64 58)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %46, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %44)
  %48 = icmp eq ptr %46, null
  %49 = select i1 %48, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %46
  %50 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8))
  %51 = add i64 1, %50
  %52 = call i64 @strlen(ptr %49)
  %53 = add i64 %51, %52
  %54 = add i64 %53, 8
  %55 = call ptr @GC_malloc(i64 %54)
  %56 = getelementptr i8, ptr %55, i64 8
  %57 = getelementptr i8, ptr %56, i64 0
  %58 = sub i64 %53, 0
  %59 = call ptr @__memcpy_chk(ptr %57, ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8), i64 %50, i64 %58)
  %60 = add i64 0, %50
  %61 = getelementptr i8, ptr %56, i64 %60
  %62 = sub i64 %53, %60
  %63 = call ptr @__memcpy_chk(ptr %61, ptr %49, i64 %52, i64 %62)
  %64 = add i64 %60, %52
  %65 = getelementptr i8, ptr %56, i64 %64
  store i8 0, ptr %65, align 1
  invoke void @Res.constructor(ptr %42, ptr %56)
          to label %66 unwind label %91

66:                                               ; preds = %40
  store ptr %42, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %67 = load i32, ptr %3, align 4
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %69, label %82

69:                                               ; preds = %66
  %70 = load i1, ptr %1, align 1
  br i1 %70, label %71, label %75

71:                                               ; preds = %69
  store i1 false, ptr %1, align 1
  %72 = load ptr, ptr %2, align 8
  %73 = getelementptr %Res, ptr %72, i32 0, i32 0
  invoke void @Res.dispose(ptr %72)
          to label %74 unwind label %91

74:                                               ; preds = %71
  br label %75

75:                                               ; preds = %74, %69
  %76 = load i1, ptr %4, align 1
  br i1 %76, label %77, label %81

77:                                               ; preds = %75
  store i1 false, ptr %4, align 1
  %78 = load ptr, ptr %5, align 8
  %79 = getelementptr %Res, ptr %78, i32 0, i32 0
  invoke void @Res.dispose(ptr %78)
          to label %80 unwind label %91

80:                                               ; preds = %77
  br label %81

81:                                               ; preds = %80, %75
  br label %128

82:                                               ; preds = %66
  %83 = load i1, ptr %1, align 1
  br i1 %83, label %84, label %88

84:                                               ; preds = %82
  store i1 false, ptr %1, align 1
  %85 = load ptr, ptr %2, align 8
  %86 = getelementptr %Res, ptr %85, i32 0, i32 0
  invoke void @Res.dispose(ptr %85)
          to label %87 unwind label %91

87:                                               ; preds = %84
  br label %88

88:                                               ; preds = %87, %82
  %89 = load i32, ptr %3, align 4
  %90 = add i32 %89, 1
  store i32 %90, ptr %3, align 4
  br label %37

91:                                               ; preds = %77, %71, %84, %40
  %92 = landingpad { ptr, i32 }
          catch ptr null
  %93 = extractvalue { ptr, i32 } %92, 0
  %94 = call ptr @__cxa_begin_catch(ptr %93)
  %95 = load i1, ptr %1, align 1
  br i1 %95, label %96, label %100

96:                                               ; preds = %91
  store i1 false, ptr %1, align 1
  %97 = load ptr, ptr %2, align 8
  %98 = getelementptr %Res, ptr %97, i32 0, i32 0
  invoke void @Res.dispose(ptr %97)
          to label %99 unwind label %endcatch.pad2

99:                                               ; preds = %96
  br label %100

100:                                              ; preds = %99, %91
  invoke void @__cxa_rethrow()
          to label %132 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %96
  %101 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

102:                                              ; No predecessors!
  %103 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %102
  %104 = phi { ptr, i32 } [ %103, %102 ], [ %101, %endcatch.pad2 ]
  %105 = extractvalue { ptr, i32 } %104, 0
  %106 = call ptr @__cxa_begin_catch(ptr %105)
  %107 = load i1, ptr %1, align 1
  br i1 %107, label %108, label %111

108:                                              ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %109 = load ptr, ptr %2, align 8
  %110 = getelementptr %Res, ptr %109, i32 0, i32 0
  invoke void @Res.dispose(ptr %109)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %108
  br label %111

111:                                              ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %132 unwind label %endcatch.pad

112:                                              ; preds = %37
  %113 = load i1, ptr %4, align 1
  br i1 %113, label %114, label %118

114:                                              ; preds = %112
  store i1 false, ptr %4, align 1
  %115 = load ptr, ptr %5, align 8
  %116 = getelementptr %Res, ptr %115, i32 0, i32 0
  invoke void @Res.dispose(ptr %115)
          to label %117 unwind label %120

117:                                              ; preds = %114
  br label %118

118:                                              ; preds = %117, %112
  br label %128

endcatch.pad:                                     ; preds = %111, %100
  %119 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

120:                                              ; preds = %114, %10
  %121 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %120
  %122 = phi { ptr, i32 } [ %121, %120 ], [ %119, %endcatch.pad ]
  %123 = load i1, ptr %4, align 1
  br i1 %123, label %124, label %127

124:                                              ; preds = %.caught
  store i1 false, ptr %4, align 1
  %125 = load ptr, ptr %5, align 8
  %126 = getelementptr %Res, ptr %125, i32 0, i32 0
  call void @Res.dispose(ptr %125)
  br label %127

127:                                              ; preds = %124, %.caught
  resume { ptr, i32 } %122

128:                                              ; preds = %81, %118
  %129 = load i32, ptr %6, align 4
  %130 = add i32 %129, 1
  store i32 %130, ptr %6, align 4
  br label %7

131:                                              ; preds = %7
  ret void

132:                                              ; preds = %100, %111
  unreachable

endcatch.resume:                                  ; preds = %108
  %133 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %133
}

; Function Attrs: mustprogress
define void @breakOuterFromIf() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  br label %5

5:                                                ; preds = %64, %0
  br i1 true, label %6, label %74

6:                                                ; preds = %5
  store i1 false, ptr %3, align 1
  %7 = call ptr @GC_malloc(i64 24)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr %Res, ptr %8, i32 0, i32 0
  store ptr @Res..vtbl, ptr %9, align 8
  invoke void @Res.constructor(ptr %8, ptr getelementptr inbounds nuw (i8, ptr @s_3523764787506715604, i64 8))
          to label %10 unwind label %66

10:                                               ; preds = %6
  store ptr %8, ptr %4, align 8
  store i1 true, ptr %3, align 1
  br label %11

11:                                               ; preds = %36, %10
  br i1 true, label %12, label %58

12:                                               ; preds = %11
  store i1 false, ptr %1, align 1
  %13 = call ptr @GC_malloc(i64 24)
  %14 = getelementptr i8, ptr %13, i64 8
  %15 = getelementptr %Res, ptr %14, i32 0, i32 0
  store ptr @Res..vtbl, ptr %15, align 8
  invoke void @Res.constructor(ptr %14, ptr getelementptr inbounds nuw (i8, ptr @s_3263991042789642887, i64 8))
          to label %16 unwind label %37

16:                                               ; preds = %12
  store ptr %14, ptr %2, align 8
  store i1 true, ptr %1, align 1
  br i1 true, label %17, label %30

17:                                               ; preds = %16
  %18 = load i1, ptr %1, align 1
  br i1 %18, label %19, label %23

19:                                               ; preds = %17
  store i1 false, ptr %1, align 1
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr %Res, ptr %20, i32 0, i32 0
  invoke void @Res.dispose(ptr %20)
          to label %22 unwind label %37

22:                                               ; preds = %19
  br label %23

23:                                               ; preds = %22, %17
  %24 = load i1, ptr %3, align 1
  br i1 %24, label %25, label %29

25:                                               ; preds = %23
  store i1 false, ptr %3, align 1
  %26 = load ptr, ptr %4, align 8
  %27 = getelementptr %Res, ptr %26, i32 0, i32 0
  invoke void @Res.dispose(ptr %26)
          to label %28 unwind label %37

28:                                               ; preds = %25
  br label %29

29:                                               ; preds = %28, %23
  br label %74

30:                                               ; preds = %16
  %31 = load i1, ptr %1, align 1
  br i1 %31, label %32, label %36

32:                                               ; preds = %30
  store i1 false, ptr %1, align 1
  %33 = load ptr, ptr %2, align 8
  %34 = getelementptr %Res, ptr %33, i32 0, i32 0
  invoke void @Res.dispose(ptr %33)
          to label %35 unwind label %37

35:                                               ; preds = %32
  br label %36

36:                                               ; preds = %35, %30
  br label %11

37:                                               ; preds = %25, %19, %32, %12
  %38 = landingpad { ptr, i32 }
          catch ptr null
  %39 = extractvalue { ptr, i32 } %38, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39)
  %41 = load i1, ptr %1, align 1
  br i1 %41, label %42, label %46

42:                                               ; preds = %37
  store i1 false, ptr %1, align 1
  %43 = load ptr, ptr %2, align 8
  %44 = getelementptr %Res, ptr %43, i32 0, i32 0
  invoke void @Res.dispose(ptr %43)
          to label %45 unwind label %endcatch.pad2

45:                                               ; preds = %42
  br label %46

46:                                               ; preds = %45, %37
  invoke void @__cxa_rethrow()
          to label %75 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %42
  %47 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

48:                                               ; No predecessors!
  %49 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %48
  %50 = phi { ptr, i32 } [ %49, %48 ], [ %47, %endcatch.pad2 ]
  %51 = extractvalue { ptr, i32 } %50, 0
  %52 = call ptr @__cxa_begin_catch(ptr %51)
  %53 = load i1, ptr %1, align 1
  br i1 %53, label %54, label %57

54:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %55 = load ptr, ptr %2, align 8
  %56 = getelementptr %Res, ptr %55, i32 0, i32 0
  invoke void @Res.dispose(ptr %55)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %54
  br label %57

57:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %75 unwind label %endcatch.pad

58:                                               ; preds = %11
  %59 = load i1, ptr %3, align 1
  br i1 %59, label %60, label %64

60:                                               ; preds = %58
  store i1 false, ptr %3, align 1
  %61 = load ptr, ptr %4, align 8
  %62 = getelementptr %Res, ptr %61, i32 0, i32 0
  invoke void @Res.dispose(ptr %61)
          to label %63 unwind label %66

63:                                               ; preds = %60
  br label %64

64:                                               ; preds = %63, %58
  br label %5

endcatch.pad:                                     ; preds = %57, %46
  %65 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

66:                                               ; preds = %60, %6
  %67 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %66
  %68 = phi { ptr, i32 } [ %67, %66 ], [ %65, %endcatch.pad ]
  %69 = load i1, ptr %3, align 1
  br i1 %69, label %70, label %73

70:                                               ; preds = %.caught
  store i1 false, ptr %3, align 1
  %71 = load ptr, ptr %4, align 8
  %72 = getelementptr %Res, ptr %71, i32 0, i32 0
  call void @Res.dispose(ptr %71)
  br label %73

73:                                               ; preds = %70, %.caught
  resume { ptr, i32 } %68

74:                                               ; preds = %29, %5
  ret void

75:                                               ; preds = %46, %57
  unreachable

endcatch.resume:                                  ; preds = %54
  %76 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %76
}

; Function Attrs: mustprogress
define void @breakLabelledBlock() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 false, ptr %3, align 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  invoke void @Res.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_619343937915704565, i64 8))
          to label %8 unwind label %110

8:                                                ; preds = %0
  store ptr %6, ptr %4, align 8
  store i1 true, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %9 = call ptr @GC_malloc(i64 24)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %Res, ptr %10, i32 0, i32 0
  store ptr @Res..vtbl, ptr %11, align 8
  invoke void @Res.constructor(ptr %10, ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8))
          to label %12 unwind label %62

12:                                               ; preds = %8
  store ptr %10, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %13 = load ptr, ptr %2, align 8
  %14 = getelementptr %Res, ptr %13, i32 0, i32 1
  %15 = load ptr, ptr %14, align 8
  %16 = ptrtoint ptr %15 to i64
  %17 = icmp ne i64 %16, 0
  %18 = zext i1 %17 to i32
  %19 = and i32 %18, 1
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %24

21:                                               ; preds = %12
  %22 = call i32 @strcmp(ptr %15, ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8))
  %23 = icmp eq i32 %22, 0
  br label %26

24:                                               ; preds = %12
  %25 = icmp eq i64 %16, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8) to i64)
  br label %26

26:                                               ; preds = %21, %24
  %27 = phi i1 [ %25, %24 ], [ %23, %21 ]
  br label %28

28:                                               ; preds = %26
  br i1 %27, label %29, label %36

29:                                               ; preds = %28
  %30 = load i1, ptr %1, align 1
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  store i1 false, ptr %1, align 1
  %32 = load ptr, ptr %2, align 8
  %33 = getelementptr %Res, ptr %32, i32 0, i32 0
  invoke void @Res.dispose(ptr %32)
          to label %34 unwind label %62

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %34, %29
  br label %83

36:                                               ; preds = %28
  %37 = load ptr, ptr @trail, align 8
  %38 = icmp eq ptr %37, null
  %39 = select i1 %38, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %37
  %40 = call i64 @strlen(ptr %39)
  %41 = add i64 1, %40
  %42 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8))
  %43 = add i64 %41, %42
  %44 = add i64 %43, 8
  %45 = call ptr @GC_malloc(i64 %44)
  %46 = getelementptr i8, ptr %45, i64 8
  %47 = getelementptr i8, ptr %46, i64 0
  %48 = sub i64 %43, 0
  %49 = call ptr @__memcpy_chk(ptr %47, ptr %39, i64 %40, i64 %48)
  %50 = add i64 0, %40
  %51 = getelementptr i8, ptr %46, i64 %50
  %52 = sub i64 %43, %50
  %53 = call ptr @__memcpy_chk(ptr %51, ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8), i64 %42, i64 %52)
  %54 = add i64 %50, %42
  %55 = getelementptr i8, ptr %46, i64 %54
  store i8 0, ptr %55, align 1
  store ptr %46, ptr @trail, align 8
  %56 = load i1, ptr %1, align 1
  br i1 %56, label %57, label %61

57:                                               ; preds = %36
  store i1 false, ptr %1, align 1
  %58 = load ptr, ptr %2, align 8
  %59 = getelementptr %Res, ptr %58, i32 0, i32 0
  invoke void @Res.dispose(ptr %58)
          to label %60 unwind label %62

60:                                               ; preds = %57
  br label %61

61:                                               ; preds = %60, %36
  br label %83

62:                                               ; preds = %31, %57, %8
  %63 = landingpad { ptr, i32 }
          catch ptr null
  %64 = extractvalue { ptr, i32 } %63, 0
  %65 = call ptr @__cxa_begin_catch(ptr %64)
  %66 = load i1, ptr %1, align 1
  br i1 %66, label %67, label %71

67:                                               ; preds = %62
  store i1 false, ptr %1, align 1
  %68 = load ptr, ptr %2, align 8
  %69 = getelementptr %Res, ptr %68, i32 0, i32 0
  invoke void @Res.dispose(ptr %68)
          to label %70 unwind label %endcatch.pad2

70:                                               ; preds = %67
  br label %71

71:                                               ; preds = %70, %62
  invoke void @__cxa_rethrow()
          to label %118 unwind label %endcatch.pad

endcatch.pad2:                                    ; preds = %67
  %72 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

73:                                               ; No predecessors!
  %74 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %73
  %75 = phi { ptr, i32 } [ %74, %73 ], [ %72, %endcatch.pad2 ]
  %76 = extractvalue { ptr, i32 } %75, 0
  %77 = call ptr @__cxa_begin_catch(ptr %76)
  %78 = load i1, ptr %1, align 1
  br i1 %78, label %79, label %82

79:                                               ; preds = %.caught1
  store i1 false, ptr %1, align 1
  %80 = load ptr, ptr %2, align 8
  %81 = getelementptr %Res, ptr %80, i32 0, i32 0
  invoke void @Res.dispose(ptr %80)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %79
  br label %82

82:                                               ; preds = %.noexc, %.caught1
  invoke void @__cxa_rethrow()
          to label %118 unwind label %endcatch.pad

83:                                               ; preds = %35, %61
  %84 = load ptr, ptr @trail, align 8
  %85 = icmp eq ptr %84, null
  %86 = select i1 %85, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %84
  %87 = call i64 @strlen(ptr %86)
  %88 = add i64 1, %87
  %89 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %90 = add i64 %88, %89
  %91 = add i64 %90, 8
  %92 = call ptr @GC_malloc(i64 %91)
  %93 = getelementptr i8, ptr %92, i64 8
  %94 = getelementptr i8, ptr %93, i64 0
  %95 = sub i64 %90, 0
  %96 = call ptr @__memcpy_chk(ptr %94, ptr %86, i64 %87, i64 %95)
  %97 = add i64 0, %87
  %98 = getelementptr i8, ptr %93, i64 %97
  %99 = sub i64 %90, %97
  %100 = call ptr @__memcpy_chk(ptr %98, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %89, i64 %99)
  %101 = add i64 %97, %89
  %102 = getelementptr i8, ptr %93, i64 %101
  store i8 0, ptr %102, align 1
  store ptr %93, ptr @trail, align 8
  %103 = load i1, ptr %3, align 1
  br i1 %103, label %104, label %108

104:                                              ; preds = %83
  store i1 false, ptr %3, align 1
  %105 = load ptr, ptr %4, align 8
  %106 = getelementptr %Res, ptr %105, i32 0, i32 0
  invoke void @Res.dispose(ptr %105)
          to label %107 unwind label %110

107:                                              ; preds = %104
  br label %108

108:                                              ; preds = %107, %83
  ret void

endcatch.pad:                                     ; preds = %71, %82
  %109 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

110:                                              ; preds = %104, %0
  %111 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %110
  %112 = phi { ptr, i32 } [ %111, %110 ], [ %109, %endcatch.pad ]
  %113 = load i1, ptr %3, align 1
  br i1 %113, label %114, label %117

114:                                              ; preds = %.caught
  store i1 false, ptr %3, align 1
  %115 = load ptr, ptr %4, align 8
  %116 = getelementptr %Res, ptr %115, i32 0, i32 0
  call void @Res.dispose(ptr %115)
  br label %117

117:                                              ; preds = %114, %.caught
  resume { ptr, i32 } %112

118:                                              ; preds = %71, %82
  unreachable

endcatch.resume:                                  ; preds = %79
  %119 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %119
}

; Function Attrs: mustprogress
define void @continueLabelledBlock() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %41, %0
  store i1 false, ptr %1, align 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  %8 = load i32, ptr %3, align 4
  %9 = call ptr @GC_malloc_atomic(i64 58)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %10, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %8)
  %12 = icmp eq ptr %10, null
  %13 = select i1 %12, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %10
  %14 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8))
  %15 = add i64 1, %14
  %16 = call i64 @strlen(ptr %13)
  %17 = add i64 %15, %16
  %18 = add i64 %17, 8
  %19 = call ptr @GC_malloc(i64 %18)
  %20 = getelementptr i8, ptr %19, i64 8
  %21 = getelementptr i8, ptr %20, i64 0
  %22 = sub i64 %17, 0
  %23 = call ptr @__memcpy_chk(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8), i64 %14, i64 %22)
  %24 = add i64 0, %14
  %25 = getelementptr i8, ptr %20, i64 %24
  %26 = sub i64 %17, %24
  %27 = call ptr @__memcpy_chk(ptr %25, ptr %13, i64 %16, i64 %26)
  %28 = add i64 %24, %16
  %29 = getelementptr i8, ptr %20, i64 %28
  store i8 0, ptr %29, align 1
  invoke void @Res.constructor(ptr %6, ptr %20)
          to label %30 unwind label %98

30:                                               ; preds = %4
  store ptr %6, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %31 = load i32, ptr %3, align 4
  %32 = add i32 %31, 1
  store i32 %32, ptr %3, align 4
  %33 = load i32, ptr %3, align 4
  %34 = icmp slt i32 %33, 3
  br i1 %34, label %35, label %42

35:                                               ; preds = %30
  %36 = load i1, ptr %1, align 1
  br i1 %36, label %37, label %41

37:                                               ; preds = %35
  store i1 false, ptr %1, align 1
  %38 = load ptr, ptr %2, align 8
  %39 = getelementptr %Res, ptr %38, i32 0, i32 0
  invoke void @Res.dispose(ptr %38)
          to label %40 unwind label %98

40:                                               ; preds = %37
  br label %41

41:                                               ; preds = %40, %35
  br label %4

42:                                               ; preds = %30
  %43 = load i1, ptr %1, align 1
  br i1 %43, label %44, label %48

44:                                               ; preds = %42
  store i1 false, ptr %1, align 1
  %45 = load ptr, ptr %2, align 8
  %46 = getelementptr %Res, ptr %45, i32 0, i32 0
  invoke void @Res.dispose(ptr %45)
          to label %47 unwind label %98

47:                                               ; preds = %44
  br label %48

48:                                               ; preds = %47, %42
  %49 = load ptr, ptr @trail, align 8
  %50 = load i32, ptr %3, align 4
  %51 = call ptr @GC_malloc_atomic(i64 58)
  %52 = getelementptr i8, ptr %51, i64 8
  %53 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %52, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %50)
  %54 = icmp eq ptr %52, null
  %55 = select i1 %54, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %52
  %56 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638194897137039473, i64 8))
  %57 = add i64 1, %56
  %58 = call i64 @strlen(ptr %55)
  %59 = add i64 %57, %58
  %60 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8))
  %61 = add i64 %59, %60
  %62 = add i64 %61, 8
  %63 = call ptr @GC_malloc(i64 %62)
  %64 = getelementptr i8, ptr %63, i64 8
  %65 = getelementptr i8, ptr %64, i64 0
  %66 = sub i64 %61, 0
  %67 = call ptr @__memcpy_chk(ptr %65, ptr getelementptr inbounds nuw (i8, ptr @s_12638194897137039473, i64 8), i64 %56, i64 %66)
  %68 = add i64 0, %56
  %69 = getelementptr i8, ptr %64, i64 %68
  %70 = sub i64 %61, %68
  %71 = call ptr @__memcpy_chk(ptr %69, ptr %55, i64 %58, i64 %70)
  %72 = add i64 %68, %58
  %73 = getelementptr i8, ptr %64, i64 %72
  %74 = sub i64 %61, %72
  %75 = call ptr @__memcpy_chk(ptr %73, ptr getelementptr inbounds nuw (i8, ptr @s_12638145419113769978, i64 8), i64 %60, i64 %74)
  %76 = add i64 %72, %60
  %77 = getelementptr i8, ptr %64, i64 %76
  store i8 0, ptr %77, align 1
  %78 = icmp eq ptr %49, null
  %79 = select i1 %78, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %49
  %80 = icmp eq ptr %64, null
  %81 = select i1 %80, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %64
  %82 = call i64 @strlen(ptr %79)
  %83 = add i64 1, %82
  %84 = call i64 @strlen(ptr %81)
  %85 = add i64 %83, %84
  %86 = add i64 %85, 8
  %87 = call ptr @GC_malloc(i64 %86)
  %88 = getelementptr i8, ptr %87, i64 8
  %89 = getelementptr i8, ptr %88, i64 0
  %90 = sub i64 %85, 0
  %91 = call ptr @__memcpy_chk(ptr %89, ptr %79, i64 %82, i64 %90)
  %92 = add i64 0, %82
  %93 = getelementptr i8, ptr %88, i64 %92
  %94 = sub i64 %85, %92
  %95 = call ptr @__memcpy_chk(ptr %93, ptr %81, i64 %84, i64 %94)
  %96 = add i64 %92, %84
  %97 = getelementptr i8, ptr %88, i64 %96
  store i8 0, ptr %97, align 1
  store ptr %88, ptr @trail, align 8
  ret void

98:                                               ; preds = %37, %44, %4
  %99 = landingpad { ptr, i32 }
          cleanup
  %100 = load i1, ptr %1, align 1
  br i1 %100, label %101, label %104

101:                                              ; preds = %98
  store i1 false, ptr %1, align 1
  %102 = load ptr, ptr %2, align 8
  %103 = getelementptr %Res, ptr %102, i32 0, i32 0
  call void @Res.dispose(ptr %102)
  br label %104

104:                                              ; preds = %101, %98
  resume { ptr, i32 } %99
}

; Function Attrs: mustprogress
define void @unlabelledBreakInLabelledBlock() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i1, align 1
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 0, ptr %5, align 4
  br label %6

6:                                                ; preds = %148, %0
  %7 = load i32, ptr %5, align 4
  %8 = icmp slt i32 %7, 2
  br i1 %8, label %9, label %160

9:                                                ; preds = %6
  store i1 false, ptr %3, align 1
  %10 = call ptr @GC_malloc(i64 24)
  %11 = getelementptr i8, ptr %10, i64 8
  %12 = getelementptr %Res, ptr %11, i32 0, i32 0
  store ptr @Res..vtbl, ptr %12, align 8
  %13 = load i32, ptr %5, align 4
  %14 = call ptr @GC_malloc_atomic(i64 58)
  %15 = getelementptr i8, ptr %14, i64 8
  %16 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %15, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %13)
  %17 = icmp eq ptr %15, null
  %18 = select i1 %17, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %19 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14793735007222968981, i64 8))
  %20 = add i64 1, %19
  %21 = call i64 @strlen(ptr %18)
  %22 = add i64 %20, %21
  %23 = add i64 %22, 8
  %24 = call ptr @GC_malloc(i64 %23)
  %25 = getelementptr i8, ptr %24, i64 8
  %26 = getelementptr i8, ptr %25, i64 0
  %27 = sub i64 %22, 0
  %28 = call ptr @__memcpy_chk(ptr %26, ptr getelementptr inbounds nuw (i8, ptr @s_14793735007222968981, i64 8), i64 %19, i64 %27)
  %29 = add i64 0, %19
  %30 = getelementptr i8, ptr %25, i64 %29
  %31 = sub i64 %22, %29
  %32 = call ptr @__memcpy_chk(ptr %30, ptr %18, i64 %21, i64 %31)
  %33 = add i64 %29, %21
  %34 = getelementptr i8, ptr %25, i64 %33
  store i8 0, ptr %34, align 1
  invoke void @Res.constructor(ptr %11, ptr %25)
          to label %35 unwind label %152

35:                                               ; preds = %9
  store ptr %11, ptr %4, align 8
  store i1 true, ptr %3, align 1
  store i1 false, ptr %1, align 1
  %36 = call ptr @GC_malloc(i64 24)
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr %Res, ptr %37, i32 0, i32 0
  store ptr @Res..vtbl, ptr %38, align 8
  %39 = load i32, ptr %5, align 4
  %40 = call ptr @GC_malloc_atomic(i64 58)
  %41 = getelementptr i8, ptr %40, i64 8
  %42 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %41, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %39)
  %43 = icmp eq ptr %41, null
  %44 = select i1 %43, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %41
  %45 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8))
  %46 = add i64 1, %45
  %47 = call i64 @strlen(ptr %44)
  %48 = add i64 %46, %47
  %49 = add i64 %48, 8
  %50 = call ptr @GC_malloc(i64 %49)
  %51 = getelementptr i8, ptr %50, i64 8
  %52 = getelementptr i8, ptr %51, i64 0
  %53 = sub i64 %48, 0
  %54 = call ptr @__memcpy_chk(ptr %52, ptr getelementptr inbounds nuw (i8, ptr @s_20852345718408352, i64 8), i64 %45, i64 %53)
  %55 = add i64 0, %45
  %56 = getelementptr i8, ptr %51, i64 %55
  %57 = sub i64 %48, %55
  %58 = call ptr @__memcpy_chk(ptr %56, ptr %44, i64 %47, i64 %57)
  %59 = add i64 %55, %47
  %60 = getelementptr i8, ptr %51, i64 %59
  store i8 0, ptr %60, align 1
  invoke void @Res.constructor(ptr %37, ptr %51)
          to label %61 unwind label %123

61:                                               ; preds = %35
  store ptr %37, ptr %2, align 8
  store i1 true, ptr %1, align 1
  %62 = load i32, ptr %5, align 4
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %77

64:                                               ; preds = %61
  %65 = load i1, ptr %1, align 1
  br i1 %65, label %66, label %70

66:                                               ; preds = %64
  store i1 false, ptr %1, align 1
  %67 = load ptr, ptr %2, align 8
  %68 = getelementptr %Res, ptr %67, i32 0, i32 0
  invoke void @Res.dispose(ptr %67)
          to label %69 unwind label %123

69:                                               ; preds = %66
  br label %70

70:                                               ; preds = %69, %64
  %71 = load i1, ptr %3, align 1
  br i1 %71, label %72, label %76

72:                                               ; preds = %70
  store i1 false, ptr %3, align 1
  %73 = load ptr, ptr %4, align 8
  %74 = getelementptr %Res, ptr %73, i32 0, i32 0
  invoke void @Res.dispose(ptr %73)
          to label %75 unwind label %123

75:                                               ; preds = %72
  br label %76

76:                                               ; preds = %75, %70
  br label %160

77:                                               ; preds = %61
  %78 = load ptr, ptr @trail, align 8
  %79 = icmp eq ptr %78, null
  %80 = select i1 %79, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %78
  %81 = call i64 @strlen(ptr %80)
  %82 = add i64 1, %81
  %83 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8))
  %84 = add i64 %82, %83
  %85 = add i64 %84, 8
  %86 = call ptr @GC_malloc(i64 %85)
  %87 = getelementptr i8, ptr %86, i64 8
  %88 = getelementptr i8, ptr %87, i64 0
  %89 = sub i64 %84, 0
  %90 = call ptr @__memcpy_chk(ptr %88, ptr %80, i64 %81, i64 %89)
  %91 = add i64 0, %81
  %92 = getelementptr i8, ptr %87, i64 %91
  %93 = sub i64 %84, %91
  %94 = call ptr @__memcpy_chk(ptr %92, ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8), i64 %83, i64 %93)
  %95 = add i64 %91, %83
  %96 = getelementptr i8, ptr %87, i64 %95
  store i8 0, ptr %96, align 1
  store ptr %87, ptr @trail, align 8
  %97 = load i1, ptr %1, align 1
  br i1 %97, label %98, label %102

98:                                               ; preds = %77
  store i1 false, ptr %1, align 1
  %99 = load ptr, ptr %2, align 8
  %100 = getelementptr %Res, ptr %99, i32 0, i32 0
  invoke void @Res.dispose(ptr %99)
          to label %101 unwind label %123

101:                                              ; preds = %98
  br label %102

102:                                              ; preds = %101, %77
  %103 = load ptr, ptr @trail, align 8
  %104 = icmp eq ptr %103, null
  %105 = select i1 %104, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %103
  %106 = call i64 @strlen(ptr %105)
  %107 = add i64 1, %106
  %108 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8))
  %109 = add i64 %107, %108
  %110 = add i64 %109, 8
  %111 = call ptr @GC_malloc(i64 %110)
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr i8, ptr %112, i64 0
  %114 = sub i64 %109, 0
  %115 = call ptr @__memcpy_chk(ptr %113, ptr %105, i64 %106, i64 %114)
  %116 = add i64 0, %106
  %117 = getelementptr i8, ptr %112, i64 %116
  %118 = sub i64 %109, %116
  %119 = call ptr @__memcpy_chk(ptr %117, ptr getelementptr inbounds nuw (i8, ptr @s_3446774449893561193, i64 8), i64 %108, i64 %118)
  %120 = add i64 %116, %108
  %121 = getelementptr i8, ptr %112, i64 %120
  store i8 0, ptr %121, align 1
  store ptr %112, ptr @trail, align 8
  %122 = load i1, ptr %3, align 1
  br i1 %122, label %144, label %148

123:                                              ; preds = %72, %66, %98, %35
  %124 = landingpad { ptr, i32 }
          catch ptr null
  %125 = extractvalue { ptr, i32 } %124, 0
  %126 = call ptr @__cxa_begin_catch(ptr %125)
  %127 = load i1, ptr %1, align 1
  br i1 %127, label %128, label %132

128:                                              ; preds = %123
  store i1 false, ptr %1, align 1
  %129 = load ptr, ptr %2, align 8
  %130 = getelementptr %Res, ptr %129, i32 0, i32 0
  invoke void @Res.dispose(ptr %129)
          to label %131 unwind label %endcatch.pad

131:                                              ; preds = %128
  br label %132

132:                                              ; preds = %131, %123
  invoke void @__cxa_rethrow()
          to label %180 unwind label %endcatch.pad2

endcatch.pad:                                     ; preds = %128
  %133 = landingpad { ptr, i32 }
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught

134:                                              ; No predecessors!
  %135 = landingpad { ptr, i32 }
          catch ptr null
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %134
  %136 = phi { ptr, i32 } [ %135, %134 ], [ %133, %endcatch.pad ]
  %137 = extractvalue { ptr, i32 } %136, 0
  %138 = call ptr @__cxa_begin_catch(ptr %137)
  %139 = load i1, ptr %1, align 1
  br i1 %139, label %140, label %143

140:                                              ; preds = %.caught
  store i1 false, ptr %1, align 1
  %141 = load ptr, ptr %2, align 8
  %142 = getelementptr %Res, ptr %141, i32 0, i32 0
  invoke void @Res.dispose(ptr %141)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %140
  br label %143

143:                                              ; preds = %.noexc, %.caught
  invoke void @__cxa_rethrow()
          to label %180 unwind label %endcatch.pad2

144:                                              ; preds = %102
  store i1 false, ptr %3, align 1
  %145 = load ptr, ptr %4, align 8
  %146 = getelementptr %Res, ptr %145, i32 0, i32 0
  invoke void @Res.dispose(ptr %145)
          to label %147 unwind label %152

147:                                              ; preds = %144
  br label %148

148:                                              ; preds = %147, %102
  %149 = load i32, ptr %5, align 4
  %150 = add i32 %149, 1
  store i32 %150, ptr %5, align 4
  br label %6

endcatch.pad2:                                    ; preds = %143, %132
  %151 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught1

152:                                              ; preds = %144, %9
  %153 = landingpad { ptr, i32 }
          cleanup
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %152
  %154 = phi { ptr, i32 } [ %153, %152 ], [ %151, %endcatch.pad2 ]
  %155 = load i1, ptr %3, align 1
  br i1 %155, label %156, label %159

156:                                              ; preds = %.caught1
  store i1 false, ptr %3, align 1
  %157 = load ptr, ptr %4, align 8
  %158 = getelementptr %Res, ptr %157, i32 0, i32 0
  call void @Res.dispose(ptr %157)
  br label %159

159:                                              ; preds = %156, %.caught1
  resume { ptr, i32 } %154

160:                                              ; preds = %76, %6
  %161 = load ptr, ptr @trail, align 8
  %162 = icmp eq ptr %161, null
  %163 = select i1 %162, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %161
  %164 = call i64 @strlen(ptr %163)
  %165 = add i64 1, %164
  %166 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %167 = add i64 %165, %166
  %168 = add i64 %167, 8
  %169 = call ptr @GC_malloc(i64 %168)
  %170 = getelementptr i8, ptr %169, i64 8
  %171 = getelementptr i8, ptr %170, i64 0
  %172 = sub i64 %167, 0
  %173 = call ptr @__memcpy_chk(ptr %171, ptr %163, i64 %164, i64 %172)
  %174 = add i64 0, %164
  %175 = getelementptr i8, ptr %170, i64 %174
  %176 = sub i64 %167, %174
  %177 = call ptr @__memcpy_chk(ptr %175, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %166, i64 %176)
  %178 = add i64 %174, %166
  %179 = getelementptr i8, ptr %170, i64 %178
  store i8 0, ptr %179, align 1
  store ptr %170, ptr @trail, align 8
  ret void

180:                                              ; preds = %132, %143
  unreachable

endcatch.resume:                                  ; preds = %140
  %181 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %181
}

; Function Attrs: mustprogress
define void @bracelessBody() #2 personality ptr @__gxx_personality_v0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 false, ptr %3, align 1
  %5 = call ptr @GC_malloc(i64 24)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = getelementptr %Res, ptr %6, i32 0, i32 0
  store ptr @Res..vtbl, ptr %7, align 8
  invoke void @Res.constructor(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_619343937915704565, i64 8))
          to label %8 unwind label %51

8:                                                ; preds = %0
  store ptr %6, ptr %4, align 8
  store i1 true, ptr %3, align 1
  store i32 0, ptr %2, align 4
  br label %9

9:                                                ; preds = %12, %8
  %10 = load i32, ptr %2, align 4
  %11 = icmp slt i32 %10, 2
  br i1 %11, label %12, label %17

12:                                               ; preds = %9
  %13 = load i32, ptr %2, align 4
  %14 = icmp eq i32 %13, 0
  %15 = load i32, ptr %2, align 4
  %16 = add i32 %15, 1
  store i32 %16, ptr %2, align 4
  br label %9

17:                                               ; preds = %9
  store i32 0, ptr %1, align 4
  br label %18

18:                                               ; preds = %22, %17
  %19 = load i32, ptr %1, align 4
  %20 = add i32 %19, 1
  store i32 %20, ptr %1, align 4
  %21 = icmp slt i32 %19, 3
  br i1 %21, label %22, label %25

22:                                               ; preds = %18
  %23 = load i32, ptr %1, align 4
  %24 = icmp eq i32 %23, 2
  br i1 %24, label %25, label %18

25:                                               ; preds = %22, %18
  %26 = load ptr, ptr @trail, align 8
  %27 = icmp eq ptr %26, null
  %28 = select i1 %27, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %26
  %29 = call i64 @strlen(ptr %28)
  %30 = add i64 1, %29
  %31 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8))
  %32 = add i64 %30, %31
  %33 = add i64 %32, 8
  %34 = call ptr @GC_malloc(i64 %33)
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = getelementptr i8, ptr %35, i64 0
  %37 = sub i64 %32, 0
  %38 = call ptr @__memcpy_chk(ptr %36, ptr %28, i64 %29, i64 %37)
  %39 = add i64 0, %29
  %40 = getelementptr i8, ptr %35, i64 %39
  %41 = sub i64 %32, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr getelementptr inbounds nuw (i8, ptr @s_15839897912459002582, i64 8), i64 %31, i64 %41)
  %43 = add i64 %39, %31
  %44 = getelementptr i8, ptr %35, i64 %43
  store i8 0, ptr %44, align 1
  store ptr %35, ptr @trail, align 8
  %45 = load i1, ptr %3, align 1
  br i1 %45, label %46, label %50

46:                                               ; preds = %25
  store i1 false, ptr %3, align 1
  %47 = load ptr, ptr %4, align 8
  %48 = getelementptr %Res, ptr %47, i32 0, i32 0
  invoke void @Res.dispose(ptr %47)
          to label %49 unwind label %51

49:                                               ; preds = %46
  br label %50

50:                                               ; preds = %49, %25
  ret void

51:                                               ; preds = %46, %0
  %52 = landingpad { ptr, i32 }
          cleanup
  %53 = load i1, ptr %3, align 1
  br i1 %53, label %54, label %57

54:                                               ; preds = %51
  store i1 false, ptr %3, align 1
  %55 = load ptr, ptr %4, align 8
  %56 = getelementptr %Res, ptr %55, i32 0, i32 0
  call void @Res.dispose(ptr %55)
  br label %57

57:                                               ; preds = %54, %51
  resume { ptr, i32 } %52
}

; Function Attrs: mustprogress
define ptr @stringInBreakBlock() #1 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = icmp eq ptr %3, null
  %5 = select i1 %4, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %3
  %6 = call i64 @strlen(ptr %5)
  %7 = add i64 1, %6
  %8 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %9 = add i64 %7, %8
  %10 = add i64 %9, 8
  %11 = call ptr @GC_malloc(i64 %10)
  %12 = getelementptr i8, ptr %11, i64 8
  %13 = getelementptr i8, ptr %12, i64 0
  %14 = sub i64 %9, 0
  %15 = call ptr @__memcpy_chk(ptr %13, ptr %5, i64 %6, i64 %14)
  %16 = add i64 0, %6
  %17 = getelementptr i8, ptr %12, i64 %16
  %18 = sub i64 %9, %16
  %19 = call ptr @__memcpy_chk(ptr %17, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), i64 %8, i64 %18)
  %20 = add i64 %16, %8
  %21 = getelementptr i8, ptr %12, i64 %20
  store i8 0, ptr %21, align 1
  store ptr %12, ptr %1, align 8
  %22 = load ptr, ptr %1, align 8
  store ptr %22, ptr %2, align 8
  %23 = load ptr, ptr %2, align 8
  ret ptr %23
}

; Function Attrs: mustprogress
define ptr @stringInContinueBlock() #1 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %1, align 8
  br label %3

3:                                                ; preds = %3, %0
  %4 = load ptr, ptr %1, align 8
  %5 = icmp eq ptr %4, null
  %6 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %4
  %7 = call i64 @strlen(ptr %6)
  %8 = add i64 1, %7
  %9 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %10 = add i64 %8, %9
  %11 = add i64 %10, 8
  %12 = call ptr @GC_malloc(i64 %11)
  %13 = getelementptr i8, ptr %12, i64 8
  %14 = getelementptr i8, ptr %13, i64 0
  %15 = sub i64 %10, 0
  %16 = call ptr @__memcpy_chk(ptr %14, ptr %6, i64 %7, i64 %15)
  %17 = add i64 0, %7
  %18 = getelementptr i8, ptr %13, i64 %17
  %19 = sub i64 %10, %17
  %20 = call ptr @__memcpy_chk(ptr %18, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), i64 %9, i64 %19)
  %21 = add i64 %17, %9
  %22 = getelementptr i8, ptr %13, i64 %21
  store i8 0, ptr %22, align 1
  store ptr %13, ptr %1, align 8
  %23 = load ptr, ptr %1, align 8
  %24 = call i64 @strlen(ptr %23)
  %25 = icmp slt i64 %24, 3
  br i1 %25, label %3, label %26

26:                                               ; preds = %3
  %27 = load ptr, ptr %1, align 8
  store ptr %27, ptr %2, align 8
  %28 = load ptr, ptr %2, align 8
  ret ptr %28
}

; Function Attrs: mustprogress
define ptr @stringInLabelledIf(i1 %0) #1 {
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = alloca ptr, align 8
  store i1 %0, ptr %3, align 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_12638213588834719060, i64 8), ptr %2, align 8
  %5 = load i1, ptr %3, align 1
  br i1 %5, label %6, label %26

6:                                                ; preds = %1
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, null
  %9 = select i1 %8, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %7
  %10 = call i64 @strlen(ptr %9)
  %11 = add i64 1, %10
  %12 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %13 = add i64 %11, %12
  %14 = add i64 %13, 8
  %15 = call ptr @GC_malloc(i64 %14)
  %16 = getelementptr i8, ptr %15, i64 8
  %17 = getelementptr i8, ptr %16, i64 0
  %18 = sub i64 %13, 0
  %19 = call ptr @__memcpy_chk(ptr %17, ptr %9, i64 %10, i64 %18)
  %20 = add i64 0, %10
  %21 = getelementptr i8, ptr %16, i64 %20
  %22 = sub i64 %13, %20
  %23 = call ptr @__memcpy_chk(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8), i64 %12, i64 %22)
  %24 = add i64 %20, %12
  %25 = getelementptr i8, ptr %16, i64 %24
  store i8 0, ptr %25, align 1
  store ptr %16, ptr %2, align 8
  br label %26

26:                                               ; preds = %6, %1
  %27 = load ptr, ptr %2, align 8
  store ptr %27, ptr %4, align 8
  %28 = load ptr, ptr %4, align 8
  ret ptr %28
}

; Function Attrs: mustprogress
define void @check({ ptr, ptr, ptr } %0, ptr %1, ptr %2) #1 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca { ptr, ptr, ptr }, align 8
  store { ptr, ptr, ptr } %0, ptr %6, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %4, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr @trail, align 8
  %7 = load { ptr, ptr, ptr }, ptr %6, align 8
  %8 = extractvalue { ptr, ptr, ptr } %7, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = icmp ne i64 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %3
  %12 = extractvalue { ptr, ptr, ptr } %7, 0
  call void %12(ptr %8)
  br label %15

13:                                               ; preds = %3
  %14 = extractvalue { ptr, ptr, ptr } %7, 0
  call void %14()
  br label %15

15:                                               ; preds = %11, %13
  br label %16

16:                                               ; preds = %15
  %17 = load ptr, ptr %4, align 8
  %18 = load ptr, ptr @trail, align 8
  %19 = icmp eq ptr %17, null
  %20 = select i1 %19, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %17
  %21 = icmp eq ptr %18, null
  %22 = select i1 %21, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %18
  %23 = call i64 @strlen(ptr %20)
  %24 = add i64 1, %23
  %25 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_584909432750204567, i64 8))
  %26 = add i64 %24, %25
  %27 = call i64 @strlen(ptr %22)
  %28 = add i64 %26, %27
  %29 = add i64 %28, 8
  %30 = call ptr @GC_malloc(i64 %29)
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr i8, ptr %31, i64 0
  %33 = sub i64 %28, 0
  %34 = call ptr @__memcpy_chk(ptr %32, ptr %20, i64 %23, i64 %33)
  %35 = add i64 0, %23
  %36 = getelementptr i8, ptr %31, i64 %35
  %37 = sub i64 %28, %35
  %38 = call ptr @__memcpy_chk(ptr %36, ptr getelementptr inbounds nuw (i8, ptr @s_584909432750204567, i64 8), i64 %25, i64 %37)
  %39 = add i64 %35, %25
  %40 = getelementptr i8, ptr %31, i64 %39
  %41 = sub i64 %28, %39
  %42 = call ptr @__memcpy_chk(ptr %40, ptr %22, i64 %27, i64 %41)
  %43 = add i64 %39, %27
  %44 = getelementptr i8, ptr %31, i64 %43
  store i8 0, ptr %44, align 1
  %45 = icmp eq ptr %31, null
  %46 = select i1 %45, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %31
  call void @puts(ptr %46)
  %47 = load ptr, ptr @trail, align 8
  %48 = load ptr, ptr %5, align 8
  %49 = ptrtoint ptr %47 to i64
  %50 = ptrtoint ptr %48 to i64
  %51 = icmp ne i64 %49, 0
  %52 = icmp ne i64 %50, 0
  %53 = zext i1 %51 to i32
  %54 = zext i1 %52 to i32
  %55 = and i32 %53, %54
  %56 = icmp ne i32 %55, 0
  br i1 %56, label %57, label %60

57:                                               ; preds = %16
  %58 = call i32 @strcmp(ptr %47, ptr %48)
  %59 = icmp eq i32 %58, 0
  br label %62

60:                                               ; preds = %16
  %61 = icmp eq i64 %49, %50
  br label %62

62:                                               ; preds = %57, %60
  %63 = phi i1 [ %61, %60 ], [ %59, %57 ]
  br label %64

64:                                               ; preds = %62
  %65 = load ptr, ptr %4, align 8
  br i1 %63, label %66, label %68

66:                                               ; preds = %64
  ret void

67:                                               ; preds = %68
  unreachable

68:                                               ; preds = %64
  %69 = call i32 @fflush(ptr null)
  %70 = icmp eq ptr %65, null
  %71 = select i1 %70, ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr %65
  call void @__assert_fail(ptr %71, ptr getelementptr inbounds nuw (i8, ptr @f_1741907030045678937, i64 8), i32 126, ptr null)
  br label %67
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  call void @check({ ptr, ptr, ptr } { ptr @continueOuterFromIf, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_1297198650753530112, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_8219663663114267555, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @breakOuterFromIf, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_1762663484488150012, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_3961742956203555479, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @breakLabelledBlock, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_13533501543527038415, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_15799391499023530575, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @continueLabelledBlock, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_6833090846104757368, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_7174868576751045779, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @unlabelledBreakInLabelledBlock, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_4705311095735501259, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_9005461004495928640, i64 8))
  call void @check({ ptr, ptr, ptr } { ptr @bracelessBody, ptr null, ptr null }, ptr getelementptr inbounds nuw (i8, ptr @s_15402960112721072915, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_4272736271669636967, i64 8))
  %1 = call ptr @stringInBreakBlock()
  %2 = ptrtoint ptr %1 to i64
  %3 = icmp ne i64 %2, 0
  %4 = zext i1 %3 to i32
  %5 = and i32 %4, 1
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %0
  %8 = call i32 @strcmp(ptr %1, ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8))
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %0
  %11 = icmp eq i64 %2, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_12638214688346347271, i64 8) to i64)
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i1 [ %11, %10 ], [ %9, %7 ]
  br label %14

14:                                               ; preds = %12
  br i1 %13, label %15, label %47

15:                                               ; preds = %14
  %16 = call ptr @stringInContinueBlock()
  %17 = ptrtoint ptr %16 to i64
  %18 = icmp ne i64 %17, 0
  %19 = zext i1 %18 to i32
  %20 = and i32 %19, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %15
  %23 = call i32 @strcmp(ptr %16, ptr getelementptr inbounds nuw (i8, ptr @s_13832720603085879183, i64 8))
  %24 = icmp eq i32 %23, 0
  br label %27

25:                                               ; preds = %15
  %26 = icmp eq i64 %17, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_13832720603085879183, i64 8) to i64)
  br label %27

27:                                               ; preds = %22, %25
  %28 = phi i1 [ %26, %25 ], [ %24, %22 ]
  br label %29

29:                                               ; preds = %27
  br i1 %28, label %30, label %49

30:                                               ; preds = %29
  %31 = call ptr @stringInLabelledIf(i1 true)
  %32 = ptrtoint ptr %31 to i64
  %33 = icmp ne i64 %32, 0
  %34 = zext i1 %33 to i32
  %35 = and i32 %34, 1
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %40

37:                                               ; preds = %30
  %38 = call i32 @strcmp(ptr %31, ptr getelementptr inbounds nuw (i8, ptr @s_643251718754793924, i64 8))
  %39 = icmp eq i32 %38, 0
  br label %42

40:                                               ; preds = %30
  %41 = icmp eq i64 %32, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_643251718754793924, i64 8) to i64)
  br label %42

42:                                               ; preds = %37, %40
  %43 = phi i1 [ %41, %40 ], [ %39, %37 ]
  br label %44

44:                                               ; preds = %42
  br i1 %43, label %45, label %51

45:                                               ; preds = %44
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

46:                                               ; preds = %51, %49, %47
  unreachable

47:                                               ; preds = %14
  %48 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8261687135864320793, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1741907030045678937, i64 8), i32 137, ptr null)
  br label %46

49:                                               ; preds = %29
  %50 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10284318878980036539, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1741907030045678937, i64 8), i32 138, ptr null)
  br label %46

51:                                               ; preds = %44
  %52 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9482693678377368718, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_1741907030045678937, i64 8), i32 139, ptr null)
  br label %46
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

declare void @__cxa_end_catch()

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress }
attributes #2 = { mustprogress "personality" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

