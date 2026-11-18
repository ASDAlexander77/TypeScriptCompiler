; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s_2906374625889817672 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFplain at exit\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_3585166406363658812 = internal constant [135 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa closure or a bound method can't be used as a plain function pointer: the pointer carries no captured variables and no 'this'\00", align 8
@s_14883366158344380534 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFclosure sees\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_4371274181479235895 = internal constant [49 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa closure as a C function pointer throws\00", align 8
@f_12645122197652191209 = internal constant [122 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00tslang_issue_29_function_value.ts\00", align 8
@_ZTIPc = external constant ptr
@_ZTIi = external constant ptr
@_ZTIj = external constant ptr
@_ZTId = external constant ptr
@_ZTIf = external constant ptr
@_ZTIb = external constant ptr
@_ZTIx = external constant ptr
@_ZTIPv = external constant ptr
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@td_10598442300733552669_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_10598442300733552669, ptr @tsretcb_10598442300733552669, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8
@td_15270880585852054582_s32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"s32\00" }, align 8
@td_11955861690735328174_u32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"u32\00" }, align 8
@td_9918219527506168722_number = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"number\00" }, align 8
@td_211483071870485656_f32 = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [4 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 1, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [4 x i8] c"f32\00" }, align 8
@td_4164229472042861253_boolean = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [8 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 3, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [8 x i8] c"boolean\00" }, align 8
@td_6780496704984787879_bigint = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 0, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"bigint\00" }, align 8
@td_1752011451674293319_string = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 2, i32 0, ptr @tsrel_1752011451674293319, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [7 x i8] c"string\00" }, align 8
@td_9449143269271153370_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr null, ptr null, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__cxa_end_catch()

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

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

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

define internal void @tsretcb_10598442300733552669(ptr %0) {
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

define internal void @tsrelcb_10598442300733552669(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %14

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %14

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  %8 = load ptr, ptr %7, align 8
  %9 = icmp ne ptr %8, null
  br i1 %9, label %10, label %13

10:                                               ; preds = %6
  %11 = call i1 @__tslang_dec_ref(ptr %8)
  br i1 %11, label %12, label %13

12:                                               ; preds = %10
  call void @__tslang_free_block(ptr %8)
  br label %13

13:                                               ; preds = %12, %10, %6
  call void @__tslang_free_block(ptr %2)
  br label %14

14:                                               ; preds = %13, %4, %1
  ret void
}

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

declare i64 @strlen(ptr)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

declare void @puts(ptr)

; Function Attrs: mustprogress
declare i32 @atexit(ptr) #1

; Function Attrs: mustprogress
define void @plain() #1 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_2906374625889817672, i64 8))
  ret void
}

; Function Attrs: mustprogress
define void @register({ ptr, ptr, ptr } %0) #1 {
  %2 = alloca { ptr, ptr, ptr }, align 8
  store { ptr, ptr, ptr } %0, ptr %2, align 8
  %3 = load { ptr, ptr, ptr }, ptr %2, align 8
  %4 = extractvalue { ptr, ptr, ptr } %3, 1
  %5 = ptrtoint ptr %4 to i64
  %6 = icmp ne i64 %5, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %1
  %8 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr getelementptr inbounds nuw (i8, ptr @s_3585166406363658812, i64 8), ptr %8, align 8
  call void @__cxa_throw(ptr %8, ptr @_ZTIPc, ptr null)
  br label %12

9:                                                ; preds = %1
  %10 = extractvalue { ptr, ptr, ptr } %3, 0
  %11 = call i32 @atexit(ptr %10)
  ret void

12:                                               ; preds = %7
  unreachable
}

; Function Attrs: mustprogress
define void @.f_main..afL22C40FH3704470766(ptr %0) #1 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = load i32, ptr %3, align 4
  %5 = call ptr @GC_malloc_atomic(i64 58)
  %6 = getelementptr i8, ptr %5, i64 8
  %7 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %6, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %4)
  %8 = call ptr @llvm.stacksave.p0()
  %9 = icmp eq ptr %6, null
  %10 = select i1 %9, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %6
  %11 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14883366158344380534, i64 8))
  %12 = add i64 1, %11
  %13 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %14 = add i64 %12, %13
  %15 = call i64 @strlen(ptr %10)
  %16 = add i64 %14, %15
  %17 = alloca i8, i64 %16, align 1
  %18 = getelementptr i8, ptr %17, i64 0
  %19 = sub i64 %16, 0
  %20 = call ptr @__memcpy_chk(ptr %18, ptr getelementptr inbounds nuw (i8, ptr @s_14883366158344380534, i64 8), i64 %11, i64 %19)
  %21 = add i64 0, %11
  %22 = getelementptr i8, ptr %17, i64 %21
  %23 = sub i64 %16, %21
  %24 = call ptr @__memcpy_chk(ptr %22, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %13, i64 %23)
  %25 = add i64 %21, %13
  %26 = getelementptr i8, ptr %17, i64 %25
  %27 = sub i64 %16, %25
  %28 = call ptr @__memcpy_chk(ptr %26, ptr %10, i64 %15, i64 %27)
  %29 = add i64 %25, %15
  %30 = getelementptr i8, ptr %17, i64 %29
  store i8 0, ptr %30, align 1
  call void @puts(ptr %17)
  call void @llvm.stackrestore.p0(ptr %8)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #2 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i1, align 1
  %4 = call ptr @GC_malloc(i64 12)
  %5 = getelementptr i8, ptr %4, i64 8
  store i32 42, ptr %5, align 4
  %6 = call i32 @atexit(ptr @plain)
  call void @register({ ptr, ptr, ptr } { ptr @plain, ptr null, ptr null })
  store i1 false, ptr %3, align 1
  %7 = call ptr @GC_malloc(i64 16)
  %8 = getelementptr i8, ptr %7, i64 8
  %9 = getelementptr { ptr }, ptr %8, i32 0, i32 0
  store ptr %5, ptr %9, align 8
  %10 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL22C40FH3704470766, ptr null, ptr null }, ptr %8, 1
  %11 = insertvalue { ptr, ptr, ptr } %10, ptr getelementptr inbounds nuw (i8, ptr @td_10598442300733552669_object, i64 40), 2
  invoke void @register({ ptr, ptr, ptr } %11)
          to label %12 unwind label %13

12:                                               ; preds = %0
  br label %152

13:                                               ; preds = %0
  %14 = landingpad { ptr, i32 }
          catch ptr null
  %15 = extractvalue { ptr, i32 } %14, 0
  %16 = call ptr @__cxa_begin_catch(ptr %15)
  %17 = invoke ptr @__cxa_current_exception_type()
          to label %.noexc11 unwind label %endcatch.resume

.noexc11:                                         ; preds = %13
  %18 = icmp eq ptr %17, @_ZTIi
  br i1 %18, label %19, label %26

19:                                               ; preds = %.noexc11
  %20 = load i32, ptr %16, align 4
  %21 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc7 unwind label %endcatch.resume

.noexc7:                                          ; preds = %19
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 0
  store i64 4, ptr %23, align 8
  %24 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_15270880585852054582_s32, i64 40), ptr %24, align 8
  %25 = getelementptr { i64, ptr, i32 }, ptr %22, i32 0, i32 2
  store i32 %20, ptr %25, align 4
  br label %149

26:                                               ; preds = %.noexc11
  %27 = icmp eq ptr %17, @_ZTIj
  br i1 %27, label %28, label %35

28:                                               ; preds = %26
  %29 = load i32, ptr %16, align 4
  %30 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc9 unwind label %endcatch.resume

.noexc9:                                          ; preds = %28
  %31 = getelementptr i8, ptr %30, i64 8
  %32 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 0
  store i64 4, ptr %32, align 8
  %33 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_11955861690735328174_u32, i64 40), ptr %33, align 8
  %34 = getelementptr { i64, ptr, i32 }, ptr %31, i32 0, i32 2
  store i32 %29, ptr %34, align 4
  br label %146

35:                                               ; preds = %26
  %36 = icmp eq ptr %17, @_ZTId
  br i1 %36, label %37, label %44

37:                                               ; preds = %35
  %38 = load double, ptr %16, align 8
  %39 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc8 unwind label %endcatch.resume

.noexc8:                                          ; preds = %37
  %40 = getelementptr i8, ptr %39, i64 8
  %41 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 0
  store i64 8, ptr %41, align 8
  %42 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9918219527506168722_number, i64 40), ptr %42, align 8
  %43 = getelementptr { i64, ptr, double }, ptr %40, i32 0, i32 2
  store double %38, ptr %43, align 8
  br label %143

44:                                               ; preds = %35
  %45 = icmp eq ptr %17, @_ZTIf
  br i1 %45, label %46, label %53

46:                                               ; preds = %44
  %47 = load float, ptr %16, align 4
  %48 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc2 unwind label %endcatch.resume

.noexc2:                                          ; preds = %46
  %49 = getelementptr i8, ptr %48, i64 8
  %50 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 0
  store i64 4, ptr %50, align 8
  %51 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_211483071870485656_f32, i64 40), ptr %51, align 8
  %52 = getelementptr { i64, ptr, float }, ptr %49, i32 0, i32 2
  store float %47, ptr %52, align 4
  br label %140

53:                                               ; preds = %44
  %54 = icmp eq ptr %17, @_ZTIb
  br i1 %54, label %55, label %62

55:                                               ; preds = %53
  %56 = load i1, ptr %16, align 1
  %57 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc3 unwind label %endcatch.resume

.noexc3:                                          ; preds = %55
  %58 = getelementptr i8, ptr %57, i64 8
  %59 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 0
  store i64 1, ptr %59, align 8
  %60 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_4164229472042861253_boolean, i64 40), ptr %60, align 8
  %61 = getelementptr { i64, ptr, i1 }, ptr %58, i32 0, i32 2
  store i1 %56, ptr %61, align 1
  br label %137

62:                                               ; preds = %53
  %63 = icmp eq ptr %17, @_ZTIx
  br i1 %63, label %64, label %71

64:                                               ; preds = %62
  %65 = load i64, ptr %16, align 8
  %66 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %64
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 0
  store i64 8, ptr %68, align 8
  %69 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_6780496704984787879_bigint, i64 40), ptr %69, align 8
  %70 = getelementptr { i64, ptr, i64 }, ptr %67, i32 0, i32 2
  store i64 %65, ptr %70, align 8
  br label %134

71:                                               ; preds = %62
  %72 = icmp eq ptr %17, @_ZTIPc
  br i1 %72, label %73, label %80

73:                                               ; preds = %71
  %74 = load ptr, ptr %16, align 8
  %75 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc1 unwind label %endcatch.resume

.noexc1:                                          ; preds = %73
  %76 = getelementptr i8, ptr %75, i64 8
  %77 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 0
  store i64 8, ptr %77, align 8
  %78 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_1752011451674293319_string, i64 40), ptr %78, align 8
  %79 = getelementptr { i64, ptr, ptr }, ptr %76, i32 0, i32 2
  store ptr %74, ptr %79, align 8
  br label %131

80:                                               ; preds = %71
  %81 = icmp eq ptr %17, @_ZTIPv
  br i1 %81, label %82, label %84

82:                                               ; preds = %80
  %83 = load ptr, ptr %16, align 8
  br label %128

84:                                               ; preds = %80
  %85 = load ptr, ptr %17, align 8
  %86 = icmp eq ptr %85, getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16)
  br i1 %86, label %87, label %118

87:                                               ; preds = %84
  %88 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %17, i32 0, i32 2
  %89 = load i32, ptr %88, align 4
  %90 = and i32 %89, 1073741824
  %91 = icmp ne i32 %90, 0
  br i1 %91, label %92, label %108

92:                                               ; preds = %87
  %93 = getelementptr { ptr, ptr, i32, ptr, ptr }, ptr %17, i32 0, i32 4
  %94 = load ptr, ptr %93, align 8
  %95 = icmp ne ptr %94, null
  br i1 %95, label %96, label %98

96:                                               ; preds = %92
  invoke void %94(ptr %1, ptr %16)
          to label %.noexc5 unwind label %endcatch.resume

.noexc5:                                          ; preds = %96
  %97 = load ptr, ptr %1, align 8
  br label %105

98:                                               ; preds = %92
  %99 = load ptr, ptr %16, align 8
  %100 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc10 unwind label %endcatch.resume

.noexc10:                                         ; preds = %98
  %101 = getelementptr i8, ptr %100, i64 8
  %102 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 0
  store i64 8, ptr %102, align 8
  %103 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %103, align 8
  %104 = getelementptr { i64, ptr, ptr }, ptr %101, i32 0, i32 2
  store ptr %99, ptr %104, align 8
  br label %105

105:                                              ; preds = %.noexc5, %.noexc10
  %106 = phi ptr [ %101, %.noexc10 ], [ %97, %.noexc5 ]
  br label %107

107:                                              ; preds = %105
  br label %115

108:                                              ; preds = %87
  %109 = load ptr, ptr %16, align 8
  %110 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc4 unwind label %endcatch.resume

.noexc4:                                          ; preds = %108
  %111 = getelementptr i8, ptr %110, i64 8
  %112 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 0
  store i64 8, ptr %112, align 8
  %113 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %113, align 8
  %114 = getelementptr { i64, ptr, ptr }, ptr %111, i32 0, i32 2
  store ptr %109, ptr %114, align 8
  br label %115

115:                                              ; preds = %107, %.noexc4
  %116 = phi ptr [ %111, %.noexc4 ], [ %106, %107 ]
  br label %117

117:                                              ; preds = %115
  br label %125

118:                                              ; preds = %84
  %119 = load ptr, ptr %16, align 8
  %120 = invoke ptr @GC_malloc(i64 32)
          to label %.noexc6 unwind label %endcatch.resume

.noexc6:                                          ; preds = %118
  %121 = getelementptr i8, ptr %120, i64 8
  %122 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 0
  store i64 8, ptr %122, align 8
  %123 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_9449143269271153370_object, i64 40), ptr %123, align 8
  %124 = getelementptr { i64, ptr, ptr }, ptr %121, i32 0, i32 2
  store ptr %119, ptr %124, align 8
  br label %125

125:                                              ; preds = %117, %.noexc6
  %126 = phi ptr [ %121, %.noexc6 ], [ %116, %117 ]
  br label %127

127:                                              ; preds = %125
  br label %128

128:                                              ; preds = %82, %127
  %129 = phi ptr [ %126, %127 ], [ %83, %82 ]
  br label %130

130:                                              ; preds = %128
  br label %131

131:                                              ; preds = %.noexc1, %130
  %132 = phi ptr [ %129, %130 ], [ %76, %.noexc1 ]
  br label %133

133:                                              ; preds = %131
  br label %134

134:                                              ; preds = %.noexc, %133
  %135 = phi ptr [ %132, %133 ], [ %67, %.noexc ]
  br label %136

136:                                              ; preds = %134
  br label %137

137:                                              ; preds = %.noexc3, %136
  %138 = phi ptr [ %135, %136 ], [ %58, %.noexc3 ]
  br label %139

139:                                              ; preds = %137
  br label %140

140:                                              ; preds = %.noexc2, %139
  %141 = phi ptr [ %138, %139 ], [ %49, %.noexc2 ]
  br label %142

142:                                              ; preds = %140
  br label %143

143:                                              ; preds = %.noexc8, %142
  %144 = phi ptr [ %141, %142 ], [ %40, %.noexc8 ]
  br label %145

145:                                              ; preds = %143
  br label %146

146:                                              ; preds = %.noexc9, %145
  %147 = phi ptr [ %144, %145 ], [ %31, %.noexc9 ]
  br label %148

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %.noexc7, %148
  %150 = phi ptr [ %147, %148 ], [ %22, %.noexc7 ]
  br label %151

151:                                              ; preds = %149
  store ptr %150, ptr %2, align 8
  store i1 true, ptr %3, align 1
  call void @__cxa_end_catch()
  br label %152

152:                                              ; preds = %12, %151
  %153 = load i1, ptr %3, align 1
  br i1 %153, label %154, label %156

154:                                              ; preds = %152
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

155:                                              ; preds = %156
  unreachable

156:                                              ; preds = %152
  %157 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4371274181479235895, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_12645122197652191209, i64 8), i32 30, ptr null)
  br label %155

endcatch.resume:                                  ; preds = %13, %98, %28, %37, %19, %118, %96, %108, %55, %46, %73, %64
  %158 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %158
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

