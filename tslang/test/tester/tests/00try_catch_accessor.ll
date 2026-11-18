; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Fail = type { ptr, ptr }
%Plain = type { ptr, ptr }
%Box = type { ptr, ptr }
%A = type { ptr, i32 }
%B = type { %A }
%Square = type { %Shape }
%Shape = type { ptr }

$Fail..size = comdat any

$A..size = comdat any

$_ZTSP4Fail = comdat any

$_ZTIP4Fail = comdat any

$_ZTS4Fail = comdat any

$_ZTI4Fail = comdat any

$B..size = comdat any

$Box..size = comdat any

$Plain..size = comdat any

$Shape..size = comdat any

$Square..size = comdat any

@s_1841804593947372813 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFail\00", align 8
@s_12638222384927744748 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFA\00", align 8
@Fail..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1841804593947372813, i64 8)
@Fail..size = global i64 16, comdat
@Fail..vtbl = private global { ptr, ptr, ptr, ptr } { ptr @Fail..instanceOf, ptr @Fail..new, ptr @Fail..rtti, ptr @Fail..size }
@A..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638222384927744748, i64 8)
@A..size = global i64 16, comdat
@_ZTVN10__cxxabiv119__pointer_type_infoE = external constant ptr
@_ZTSP4Fail = linkonce_odr constant [7 x i8] c"P4Fail\00", comdat
@_ZTIP4Fail = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 16), ptr @_ZTSP4Fail, i32 1073741824, ptr @_ZTI4Fail, ptr @.eh.copy.box.P4Fail }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external constant ptr
@_ZTS4Fail = linkonce_odr constant [6 x i8] c"4Fail\00", comdat
@s_12638225683462629381 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFB\00", align 8
@s_1609559789082440018 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBox\00", align 8
@s_1410190387630922087 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFPlain\00", align 8
@s_3308291604520544166 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFShape\00", align 8
@s_7368384151532411686 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFSquare\00", align 8
@s_15339932239713171959 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFget\00", align 8
@s_9384242791558811171 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFset\00", align 8
@s_16759446751862886611 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstatic get\00", align 8
@s_5856227656950745351 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstatic set\00", align 8
@s_21724258439390450 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbox\00", align 8
@s_1241800711098863692 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface get\00", align 8
@s_16077166718034104 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbad\00", align 8
@s_7212953690789356384 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFinterface set\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
@s_18252272687975150087 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFplain\00", align 8
@s_17157602999399885131 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFabstract get\00", align 8
@s_6254630195092463135 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFabstract set\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_13597551105163972623 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFplain2\00", align 8
@s_12054661417104037398 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFwrong\00", align 8
@s_626102635892993007 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFok\00", align 8
@s_9690130352454037036 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnot reached\00", align 8
@m_9384242791558811171 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFset\00", align 8
@f_8781166903375489039 = internal constant [110 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00try_catch_accessor.ts\00", align 8
@m_8058474561178550819 = internal constant [34 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa set that does not throw\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@m_11021489436266481586 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFset, the statement after it skipped\00", align 8
@m_5856227656950745351 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstatic set\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@m_16759446751862886611 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFstatic get\00", align 8
@m_6236424509574238428 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFvirtual set\00", align 8
@m_16583843082110648990 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFset through an interface\00", align 8
@m_12515707135700146330 = internal constant [33 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFget through an interface\00", align 8
@m_1910839327606885957 = internal constant [53 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFget through an interface that does not throw\00", align 8
@m_4236572222742585917 = internal constant [42 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFset through an interface, finally\00", align 8
@m_1979075331638643099 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFa field through an interface\00", align 8
@s_3978046374080941754 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNaN\00", align 8
@s_865306855574198937 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFInfinity\00", align 8
@s_5514277459305441004 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-Infinity\00", align 8
@s_555417232352566889 = internal constant [11 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s\00", align 8
@s_12638135523509116079 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF0\00", align 8
@s_12638121229857949336 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF-\00", align 8
@s_13533418361890864287 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%.*e\00", align 8
@s_565227158805101677 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%c%.*s\00", align 8
@s_3707881812392243519 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF000000000000000000000\00", align 8
@s_14597275362824401209 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%s%.*s\00", align 8
@s_2324455066069071285 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%.*s.%s\00", align 8
@s_3663401783476796495 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s0.%.*s%s\00", align 8
@s_12638124528392833969 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.\00", align 8
@s_17791929768168046296 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%c%s%se%+d\00", align 8
@m_13269171204112287757 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFget through an abstract field\00", align 8
@m_7018100576640315633 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFset through an abstract field\00", align 8
@m_18175057562723240456 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfinally\00", align 8
@m_12895414482988252627 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFin a function\00", align 8
@_ZTI4Fail = linkonce_odr constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS4Fail }, comdat
@A..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @A..instanceOf, ptr @A..new, ptr @A.get_v, ptr @A.set_v, ptr @A.get_s, ptr @A.set_s, ptr @A..rtti, ptr @A..size }
@B..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_12638225683462629381, i64 8)
@B..size = global i64 16, comdat
@B..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @B..instanceOf, ptr @B..new, ptr @B.get_v, ptr @B.set_v, ptr @A.get_s, ptr @A.set_s, ptr @A..rtti, ptr @A..size, ptr @B..rtti, ptr @B..size }
@Box..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1609559789082440018, i64 8)
@Box..size = global i64 16, comdat
@Box.Named..vtbl.name.accessor = private global { ptr, ptr } { ptr @Box.get_name, ptr @Box.set_name }
@Box.Named..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_17239760223775721853_class, i64 40), ptr @Box..instanceOf, ptr inttoptr (i64 add (i64 ptrtoint (ptr @Box.Named..vtbl.name.accessor to i64), i64 -9223372036854775808) to ptr) }
@td_17239760223775721853_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_17239760223775721853, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Box..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Box..instanceOf, ptr @Box.Named..vtbl, ptr @Box..new, ptr @Box.get_name, ptr @Box.set_name, ptr @Box..rtti, ptr @Box..size }
@Plain..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_1410190387630922087, i64 8)
@Plain..size = global i64 16, comdat
@Plain.Named..vtbl = private global { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @td_12622666682273857524_class, i64 40), ptr @Plain..instanceOf, ptr inttoptr (i64 8 to ptr) }
@td_12622666682273857524_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_12622666682273857524, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@Plain..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Plain..instanceOf, ptr @Plain.Named..vtbl, ptr @Plain..new, ptr @Plain..rtti, ptr @Plain..size }
@Shape..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_3308291604520544166, i64 8)
@Shape..size = global i64 8, comdat
@Shape..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Shape..instanceOf, ptr null, ptr null, ptr @Shape..rtti, ptr @Shape..size }
@Square..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_7368384151532411686, i64 8)
@Square..size = global i64 8, comdat
@Square..vtbl = private global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Square..instanceOf, ptr @Square.get_size, ptr @Square.set_size, ptr @Shape..rtti, ptr @Shape..size, ptr @Square..new, ptr @Square..rtti, ptr @Square..size }
@Named.null..vtbl = private global { ptr, ptr, ptr } { ptr null, ptr @.instanceOf.none, ptr null }
@td_16282675435063983936_class = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [6 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 8, i32 0, ptr @tsrel_16282675435063983936, ptr @__tslang_retain_block_slot, ptr null, i64 -1 }, [6 x i8] c"class\00" }, align 8
@td_7761478249627842675_object = internal constant { { i32, i32, ptr, ptr, ptr, i64 }, [7 x i8] } { { i32, i32, ptr, ptr, ptr, i64 } { i32 7, i32 0, ptr @tsrelcb_7761478249627842675, ptr @tsretcb_7761478249627842675, ptr null, i64 -1 }, [7 x i8] c"object\00" }, align 8

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

define internal void @tsretcb_7761478249627842675(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  call void @__tslang_inc_ref(ptr %2)
  ret void
}

define internal void @tsrel_4117155613450509001(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %12

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %12

6:                                                ; preds = %4
  %7 = load ptr, ptr %2, align 8
  %8 = icmp eq ptr %7, @B..vtbl
  br i1 %8, label %9, label %10

9:                                                ; preds = %6
  br label %11

10:                                               ; preds = %6
  br label %11

11:                                               ; preds = %9, %10
  call void @__tslang_free_block(ptr %2)
  br label %12

12:                                               ; preds = %11, %4, %1
  ret void
}

define internal void @tsrelcb_7761478249627842675(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr { ptr }, ptr %2, i32 0, i32 0
  call void @tsrel_4117155613450509001(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

define internal void @__ts_number_to_string(ptr %0, double %1) {
  %3 = alloca i8, i32 32, align 1
  %4 = alloca i8, i32 32, align 1
  %5 = fcmp olt double %1, 0.000000e+00
  %6 = fcmp uno double %1, %1
  br i1 %6, label %7, label %9

7:                                                ; preds = %2
  %8 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_3978046374080941754, i64 8))
  ret void

9:                                                ; preds = %2
  %10 = call double @llvm.fabs.f64(double %1)
  %11 = fcmp oeq double %10, 0x7FF0000000000000
  br i1 %11, label %12, label %15

12:                                               ; preds = %9
  %13 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_5514277459305441004, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_865306855574198937, i64 8)
  %14 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_555417232352566889, i64 8), ptr %13)
  ret void

15:                                               ; preds = %9
  %16 = fcmp oeq double %1, 0.000000e+00
  br i1 %16, label %17, label %19

17:                                               ; preds = %15
  %18 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_12638135523509116079, i64 8))
  ret void

19:                                               ; preds = %15
  %20 = select i1 %5, ptr getelementptr inbounds nuw (i8, ptr @s_12638121229857949336, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
  br label %21

21:                                               ; preds = %21, %19
  %22 = phi i32 [ %28, %21 ], [ 0, %19 ]
  %23 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %3, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_13533418361890864287, i64 8), i32 %22, double %10)
  %24 = call double @strtod(ptr %3, ptr null)
  %25 = fcmp oeq double %24, %10
  %26 = icmp sge i32 %22, 16
  %27 = or i1 %25, %26
  %28 = add i32 %22, 1
  br i1 %27, label %29, label %21

29:                                               ; preds = %21
  %30 = add i32 %22, 1
  %31 = icmp eq i32 %22, 0
  %32 = add i32 %22, 3
  %33 = select i1 %31, i32 2, i32 %32
  %34 = getelementptr i8, ptr %3, i32 %33
  %35 = call double @strtod(ptr %34, ptr null)
  %36 = fptosi double %35 to i32
  %37 = add i32 %36, 1
  %38 = load i8, ptr %3, align 1
  %39 = sext i8 %38 to i32
  %40 = getelementptr i8, ptr %3, i32 2
  %41 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %4, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_565227158805101677, i64 8), i32 %39, i32 %22, ptr %40)
  %42 = icmp sle i32 %37, 21
  %43 = icmp sle i32 %30, %37
  %44 = and i1 %43, %42
  br i1 %44, label %45, label %48

45:                                               ; preds = %29
  %46 = sub i32 %37, %30
  %47 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_14597275362824401209, i64 8), ptr %20, ptr %4, i32 %46, ptr getelementptr inbounds nuw (i8, ptr @s_3707881812392243519, i64 8))
  ret void

48:                                               ; preds = %29
  %49 = icmp sgt i32 %37, 0
  %50 = and i1 %49, %42
  br i1 %50, label %51, label %54

51:                                               ; preds = %48
  %52 = getelementptr i8, ptr %4, i32 %37
  %53 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_2324455066069071285, i64 8), ptr %20, i32 %37, ptr %4, ptr %52)
  ret void

54:                                               ; preds = %48
  %55 = icmp sle i32 %37, 0
  %56 = icmp sgt i32 %37, -6
  %57 = and i1 %56, %55
  br i1 %57, label %58, label %61

58:                                               ; preds = %54
  %59 = sub i32 0, %37
  %60 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_3663401783476796495, i64 8), ptr %20, i32 %59, ptr getelementptr inbounds nuw (i8, ptr @s_3707881812392243519, i64 8), ptr %4)
  ret void

61:                                               ; preds = %54
  %62 = icmp sgt i32 %30, 1
  %63 = select i1 %62, ptr getelementptr inbounds nuw (i8, ptr @s_12638124528392833969, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8)
  %64 = getelementptr i8, ptr %4, i32 1
  %65 = sub i32 %37, 1
  %66 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %0, i64 32, ptr getelementptr inbounds nuw (i8, ptr @s_17791929768168046296, i64 8), ptr %20, i32 %39, ptr %63, ptr %64, i32 %65)
  ret void
}

declare double @strtod(ptr, ptr)

declare i32 @snprintf(ptr, i64, ptr, ...)

declare void @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

declare ptr @__cxa_begin_catch(ptr)

; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #1

declare void @ts.internal.eh_try_pad(i32)

declare void @ts.internal.eh_try(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_allocate_exception(i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

define internal void @tsrel_16282675435063983936(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Fail, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

declare i32 @strcmp(ptr, ptr)

define internal void @tsrel_12622666682273857524(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Plain, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
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

define internal void @tsrel_17239760223775721853(ptr %0) {
  %2 = load ptr, ptr %0, align 8
  %3 = icmp ne ptr %2, null
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = call i1 @__tslang_dec_ref(ptr %2)
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = getelementptr %Box, ptr %2, i32 0, i32 1
  call void @tsrel_1752011451674293319(ptr %7)
  call void @__tslang_free_block(ptr %2)
  br label %8

8:                                                ; preds = %6, %4, %1
  ret void
}

; Function Attrs: mustprogress
define i1 @Fail..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Fail..rtti, align 8
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
define ptr @Fail..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Fail, ptr %3, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Fail.constructor(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %Fail, ptr %5, i32 0, i32 1
  %7 = load ptr, ptr %3, align 8
  store ptr %7, ptr %6, align 8
  ret void
}

; Function Attrs: mustprogress
define void @A.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %A, ptr %3, i32 0, i32 1
  store i32 1, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress
define i1 @A..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @A..rtti, align 8
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
define ptr @A..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %A, ptr %3, i32 0, i32 0
  store ptr @A..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define internal void @.eh.copy.box.P4Fail(ptr %0, ptr %1) #2 {
  %3 = load ptr, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 32)
  %5 = getelementptr i8, ptr %4, i64 8
  %6 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 0
  store i64 8, ptr %6, align 8
  %7 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @td_16282675435063983936_class, i64 40), ptr %7, align 8
  %8 = getelementptr { i64, ptr, ptr }, ptr %5, i32 0, i32 2
  store ptr %3, ptr %8, align 8
  store ptr %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress
define i32 @A.get_v(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %A, ptr %4, i32 0, i32 1
  %6 = load i32, ptr %5, align 4
  %7 = icmp slt i32 %6, 0
  br i1 %7, label %8, label %13

8:                                                ; preds = %1
  %9 = call ptr @GC_malloc(i64 24)
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr %Fail, ptr %10, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %11, align 8
  call void @Fail.constructor(ptr %10, ptr getelementptr inbounds nuw (i8, ptr @s_15339932239713171959, i64 8))
  %12 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %10, ptr %12, align 8
  call void @__cxa_throw(ptr %12, ptr @_ZTIP4Fail, ptr null)
  br label %18

13:                                               ; preds = %1
  %14 = load ptr, ptr %2, align 8
  %15 = getelementptr %A, ptr %14, i32 0, i32 1
  %16 = load i32, ptr %15, align 4
  store i32 %16, ptr %3, align 4
  %17 = load i32, ptr %3, align 4
  ret i32 %17

18:                                               ; preds = %8
  unreachable
}

; Function Attrs: mustprogress
define void @A.set_v(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load double, ptr %3, align 8
  %6 = fcmp olt double %5, 0.000000e+00
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = call ptr @GC_malloc(i64 24)
  %9 = getelementptr i8, ptr %8, i64 8
  %10 = getelementptr %Fail, ptr %9, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %10, align 8
  call void @Fail.constructor(ptr %9, ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8))
  %11 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %9, ptr %11, align 8
  call void @__cxa_throw(ptr %11, ptr @_ZTIP4Fail, ptr null)
  br label %18

12:                                               ; preds = %2
  %13 = load ptr, ptr %4, align 8
  %14 = getelementptr %A, ptr %13, i32 0, i32 1
  %15 = load double, ptr %3, align 8
  %16 = fptosi double %15 to i64
  %17 = trunc i64 %16 to i32
  store i32 %17, ptr %14, align 4
  ret void

18:                                               ; preds = %7
  unreachable
}

; Function Attrs: mustprogress
define i32 @A.get_s() #2 {
  %1 = call ptr @GC_malloc(i64 24)
  %2 = getelementptr i8, ptr %1, i64 8
  %3 = getelementptr %Fail, ptr %2, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %3, align 8
  call void @Fail.constructor(ptr %2, ptr getelementptr inbounds nuw (i8, ptr @s_16759446751862886611, i64 8))
  %4 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %2, ptr %4, align 8
  call void @__cxa_throw(ptr %4, ptr @_ZTIP4Fail, ptr null)
  br label %5

5:                                                ; preds = %0
  unreachable
}

; Function Attrs: mustprogress
define void @A.set_s(double %0) #2 {
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Fail, ptr %3, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %4, align 8
  call void @Fail.constructor(ptr %3, ptr getelementptr inbounds nuw (i8, ptr @s_5856227656950745351, i64 8))
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %3, ptr %5, align 8
  call void @__cxa_throw(ptr %5, ptr @_ZTIP4Fail, ptr null)
  br label %6

6:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define i1 @B..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @B..rtti, align 8
  %8 = ptrtoint ptr %6 to i64
  %9 = ptrtoint ptr %7 to i64
  %10 = icmp ne i64 %8, 0
  %11 = icmp ne i64 %9, 0
  %12 = zext i1 %10 to i32
  %13 = zext i1 %11 to i32
  %14 = and i32 %12, %13
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %2
  %17 = call i32 @strcmp(ptr %6, ptr %7)
  %18 = icmp eq i32 %17, 0
  br label %21

19:                                               ; preds = %2
  %20 = icmp eq i64 %8, %9
  br label %21

21:                                               ; preds = %16, %19
  %22 = phi i1 [ %20, %19 ], [ %18, %16 ]
  br label %23

23:                                               ; preds = %21
  br i1 %22, label %29, label %24

24:                                               ; preds = %23
  %25 = load ptr, ptr %4, align 8
  %26 = getelementptr %B, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %3, align 8
  %28 = call i1 @A..instanceOf(ptr %26, ptr %27)
  br label %29

29:                                               ; preds = %24, %23
  %30 = phi i1 [ %28, %24 ], [ %22, %23 ]
  store i1 %30, ptr %5, align 1
  %31 = load i1, ptr %5, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define ptr @B..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %B, ptr %3, i32 0, i32 0
  %5 = getelementptr %A, ptr %4, i32 0, i32 0
  store ptr @B..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define i32 @B.get_v(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %B, ptr %4, i32 0, i32 0
  %6 = call i32 @A.get_v(ptr %5)
  %7 = add i32 %6, 100
  store i32 %7, ptr %3, align 4
  %8 = load i32, ptr %3, align 4
  ret i32 %8
}

; Function Attrs: mustprogress
define void @B.set_v(ptr %0, double %1) #2 {
  %3 = alloca double, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store double %1, ptr %3, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %B, ptr %5, i32 0, i32 0
  %7 = load double, ptr %3, align 8
  call void @A.set_v(ptr %6, double %7)
  ret void
}

; Function Attrs: mustprogress
define void @Box.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Box, ptr %3, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_21724258439390450, i64 8), ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Box..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Box..rtti, align 8
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
define ptr @Box..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Box, ptr %3, i32 0, i32 0
  store ptr @Box..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define ptr @Box.get_name(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr %Box, ptr %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp ne i64 %7, 0
  %9 = zext i1 %8 to i32
  %10 = and i32 %9, 1
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %1
  %13 = call i32 @strcmp(ptr %6, ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8))
  %14 = icmp eq i32 %13, 0
  br label %17

15:                                               ; preds = %1
  %16 = icmp eq i64 %7, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8) to i64)
  br label %17

17:                                               ; preds = %12, %15
  %18 = phi i1 [ %16, %15 ], [ %14, %12 ]
  br label %19

19:                                               ; preds = %17
  br i1 %18, label %20, label %25

20:                                               ; preds = %19
  %21 = call ptr @GC_malloc(i64 24)
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = getelementptr %Fail, ptr %22, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %23, align 8
  call void @Fail.constructor(ptr %22, ptr getelementptr inbounds nuw (i8, ptr @s_1241800711098863692, i64 8))
  %24 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %22, ptr %24, align 8
  call void @__cxa_throw(ptr %24, ptr @_ZTIP4Fail, ptr null)
  br label %30

25:                                               ; preds = %19
  %26 = load ptr, ptr %2, align 8
  %27 = getelementptr %Box, ptr %26, i32 0, i32 1
  %28 = load ptr, ptr %27, align 8
  store ptr %28, ptr %3, align 8
  %29 = load ptr, ptr %3, align 8
  ret ptr %29

30:                                               ; preds = %20
  unreachable
}

; Function Attrs: mustprogress
define void @Box.set_name(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp ne i64 %6, 0
  %8 = zext i1 %7 to i32
  %9 = and i32 %8, 1
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %14

11:                                               ; preds = %2
  %12 = call i32 @strcmp(ptr %5, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %13 = icmp eq i32 %12, 0
  br label %16

14:                                               ; preds = %2
  %15 = icmp eq i64 %6, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8) to i64)
  br label %16

16:                                               ; preds = %11, %14
  %17 = phi i1 [ %15, %14 ], [ %13, %11 ]
  br label %18

18:                                               ; preds = %16
  br i1 %17, label %19, label %24

19:                                               ; preds = %18
  %20 = call ptr @GC_malloc(i64 24)
  %21 = getelementptr i8, ptr %20, i64 8
  %22 = getelementptr %Fail, ptr %21, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %22, align 8
  call void @Fail.constructor(ptr %21, ptr getelementptr inbounds nuw (i8, ptr @s_7212953690789356384, i64 8))
  %23 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %21, ptr %23, align 8
  call void @__cxa_throw(ptr %23, ptr @_ZTIP4Fail, ptr null)
  br label %28

24:                                               ; preds = %18
  %25 = load ptr, ptr %4, align 8
  %26 = getelementptr %Box, ptr %25, i32 0, i32 1
  %27 = load ptr, ptr %3, align 8
  store ptr %27, ptr %26, align 8
  ret void

28:                                               ; preds = %19
  unreachable
}

; Function Attrs: mustprogress
define void @Plain.constructor(ptr %0) #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %Plain, ptr %3, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_18252272687975150087, i64 8), ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress
define i1 @Plain..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Plain..rtti, align 8
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
define ptr @Plain..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Plain, ptr %3, i32 0, i32 0
  store ptr @Plain..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define i1 @Shape..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i1, align 1
  store ptr %1, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr @Shape..rtti, align 8
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
declare double @Shape.get_size(ptr) #2

; Function Attrs: mustprogress
declare void @Shape.set_size(ptr, double) #2

; Function Attrs: mustprogress
define i1 @Square..instanceOf(ptr %0, ptr %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i1, align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @Square..rtti, align 8
  %8 = ptrtoint ptr %6 to i64
  %9 = ptrtoint ptr %7 to i64
  %10 = icmp ne i64 %8, 0
  %11 = icmp ne i64 %9, 0
  %12 = zext i1 %10 to i32
  %13 = zext i1 %11 to i32
  %14 = and i32 %12, %13
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %2
  %17 = call i32 @strcmp(ptr %6, ptr %7)
  %18 = icmp eq i32 %17, 0
  br label %21

19:                                               ; preds = %2
  %20 = icmp eq i64 %8, %9
  br label %21

21:                                               ; preds = %16, %19
  %22 = phi i1 [ %20, %19 ], [ %18, %16 ]
  br label %23

23:                                               ; preds = %21
  br i1 %22, label %29, label %24

24:                                               ; preds = %23
  %25 = load ptr, ptr %4, align 8
  %26 = getelementptr %Square, ptr %25, i32 0, i32 0
  %27 = load ptr, ptr %3, align 8
  %28 = call i1 @Shape..instanceOf(ptr %26, ptr %27)
  br label %29

29:                                               ; preds = %24, %23
  %30 = phi i1 [ %28, %24 ], [ %22, %23 ]
  store i1 %30, ptr %5, align 1
  %31 = load i1, ptr %5, align 1
  ret i1 %31
}

; Function Attrs: mustprogress
define ptr @Square..new() #2 {
  %1 = alloca ptr, align 8
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Square, ptr %3, i32 0, i32 0
  %5 = getelementptr %Shape, ptr %4, i32 0, i32 0
  store ptr @Square..vtbl, ptr %5, align 8
  store ptr %3, ptr %1, align 8
  %6 = load ptr, ptr %1, align 8
  ret ptr %6
}

; Function Attrs: mustprogress
define double @Square.get_size(ptr %0) #2 {
  %2 = call ptr @GC_malloc(i64 24)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Fail, ptr %3, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %4, align 8
  call void @Fail.constructor(ptr %3, ptr getelementptr inbounds nuw (i8, ptr @s_17157602999399885131, i64 8))
  %5 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %3, ptr %5, align 8
  call void @__cxa_throw(ptr %5, ptr @_ZTIP4Fail, ptr null)
  br label %6

6:                                                ; preds = %1
  unreachable
}

; Function Attrs: mustprogress
define void @Square.set_size(ptr %0, double %1) #2 {
  %3 = call ptr @GC_malloc(i64 24)
  %4 = getelementptr i8, ptr %3, i64 8
  %5 = getelementptr %Fail, ptr %4, i32 0, i32 0
  store ptr @Fail..vtbl, ptr %5, align 8
  call void @Fail.constructor(ptr %4, ptr getelementptr inbounds nuw (i8, ptr @s_6254630195092463135, i64 8))
  %6 = call ptr @__cxa_allocate_exception(i64 8)
  store ptr %4, ptr %6, align 8
  call void @__cxa_throw(ptr %6, ptr @_ZTIP4Fail, ptr null)
  br label %7

7:                                                ; preds = %2
  unreachable
}

; Function Attrs: mustprogress
define ptr @caught({ ptr, ptr, ptr } %0) #3 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca { ptr, ptr, ptr }, align 8
  %4 = alloca ptr, align 8
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
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %4, align 8
  br label %32

18:                                               ; preds = %9, %12
  %19 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
  %20 = extractvalue { ptr, i32 } %19, 1
  %21 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %22 = icmp eq i32 %20, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %18
  %24 = extractvalue { ptr, i32 } %19, 0
  %25 = call ptr @__cxa_begin_catch(ptr %24)
  invoke void @__cxa_rethrow()
          to label %.noexc unwind label %endcatch.resume

.noexc:                                           ; preds = %23
  br label %34

26:                                               ; preds = %18
  %27 = extractvalue { ptr, i32 } %19, 0
  %28 = call ptr @__cxa_begin_catch(ptr %27)
  store ptr %28, ptr %2, align 8
  %29 = load ptr, ptr %2, align 8
  %30 = getelementptr %Fail, ptr %29, i32 0, i32 1
  %31 = load ptr, ptr %30, align 8
  store ptr %31, ptr %4, align 8
  call void @__cxa_end_catch()
  br label %32

32:                                               ; preds = %17, %26
  %33 = load ptr, ptr %4, align 8
  ret ptr %33

34:                                               ; preds = %.noexc
  unreachable

endcatch.resume:                                  ; preds = %23
  %35 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  resume { ptr, i32 } %35
}

; Function Attrs: mustprogress
define internal i1 @.instanceOf.none(ptr %0, ptr %1) #2 {
  %3 = alloca i1, align 1
  store i1 false, ptr %3, align 1
  %4 = load i1, ptr %3, align 1
  ret i1 %4
}

; Function Attrs: mustprogress
define void @.f_main..afL129C19FH2287545390(ptr %0) #2 {
  %2 = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %A, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr ptr, ptr %5, i32 3
  %7 = load ptr, ptr %6, align 8
  call void %7(ptr %3, double -4.000000e+00)
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #3 personality ptr @__gxx_personality_v0 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i1, align 1
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = call ptr @GC_malloc(i64 24)
  %18 = getelementptr i8, ptr %17, i64 8
  %19 = getelementptr %A, ptr %18, i32 0, i32 0
  store ptr @A..vtbl, ptr %19, align 8
  invoke void @A.constructor(ptr %18)
          to label %20 unwind label %790

20:                                               ; preds = %0
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %21 = getelementptr %A, ptr %18, i32 0, i32 0
  %22 = load ptr, ptr %21, align 8
  %23 = getelementptr ptr, ptr %22, i32 3
  %24 = load ptr, ptr %23, align 8
  invoke void %24(ptr %18, double -1.000000e+00)
          to label %25 unwind label %26

25:                                               ; preds = %20
  br label %40

26:                                               ; preds = %20
  %27 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %28 = extractvalue { ptr, i32 } %27, 1
  %29 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %30 = icmp eq i32 %28, %29
  br i1 %30, label %34, label %31

31:                                               ; preds = %26
  %32 = extractvalue { ptr, i32 } %27, 0
  %33 = call ptr @__cxa_begin_catch(ptr %32)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

34:                                               ; preds = %26
  %35 = extractvalue { ptr, i32 } %27, 0
  %36 = call ptr @__cxa_begin_catch(ptr %35)
  store ptr %36, ptr %15, align 8
  %37 = load ptr, ptr %15, align 8
  %38 = getelementptr %Fail, ptr %37, i32 0, i32 1
  %39 = load ptr, ptr %38, align 8
  store ptr %39, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %40

40:                                               ; preds = %25, %34
  %41 = load ptr, ptr %16, align 8
  %42 = ptrtoint ptr %41 to i64
  %43 = icmp ne i64 %42, 0
  %44 = zext i1 %43 to i32
  %45 = and i32 %44, 1
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %50

47:                                               ; preds = %40
  %48 = call i32 @strcmp(ptr %41, ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8))
  %49 = icmp eq i32 %48, 0
  br label %52

50:                                               ; preds = %40
  %51 = icmp eq i64 %42, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8) to i64)
  br label %52

52:                                               ; preds = %47, %50
  %53 = phi i1 [ %51, %50 ], [ %49, %47 ]
  br label %54

54:                                               ; preds = %52
  br i1 %53, label %55, label %794

55:                                               ; preds = %54
  %56 = getelementptr %A, ptr %18, i32 0, i32 0
  %57 = load ptr, ptr %56, align 8
  %58 = getelementptr ptr, ptr %57, i32 3
  %59 = load ptr, ptr %58, align 8
  invoke void %59(ptr %18, double 5.000000e+00)
          to label %60 unwind label %790

60:                                               ; preds = %55
  %61 = getelementptr %A, ptr %18, i32 0, i32 0
  %62 = load ptr, ptr %61, align 8
  %63 = getelementptr ptr, ptr %62, i32 2
  %64 = load ptr, ptr %63, align 8
  %65 = invoke i32 %64(ptr %18)
          to label %66 unwind label %790

66:                                               ; preds = %60
  %67 = icmp eq i32 %65, 5
  br i1 %67, label %68, label %796

68:                                               ; preds = %66
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %69 = getelementptr %A, ptr %18, i32 0, i32 0
  %70 = load ptr, ptr %69, align 8
  %71 = getelementptr ptr, ptr %70, i32 3
  %72 = load ptr, ptr %71, align 8
  invoke void %72(ptr %18, double -2.000000e+00)
          to label %73 unwind label %74

73:                                               ; preds = %68
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_9690130352454037036, i64 8))
  br label %88

74:                                               ; preds = %68
  %75 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %76 = extractvalue { ptr, i32 } %75, 1
  %77 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %78 = icmp eq i32 %76, %77
  br i1 %78, label %82, label %79

79:                                               ; preds = %74
  %80 = extractvalue { ptr, i32 } %75, 0
  %81 = call ptr @__cxa_begin_catch(ptr %80)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

82:                                               ; preds = %74
  %83 = extractvalue { ptr, i32 } %75, 0
  %84 = call ptr @__cxa_begin_catch(ptr %83)
  store ptr %84, ptr %14, align 8
  %85 = load ptr, ptr %14, align 8
  %86 = getelementptr %Fail, ptr %85, i32 0, i32 1
  %87 = load ptr, ptr %86, align 8
  store ptr %87, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %88

88:                                               ; preds = %73, %82
  %89 = load ptr, ptr %16, align 8
  %90 = ptrtoint ptr %89 to i64
  %91 = icmp ne i64 %90, 0
  %92 = zext i1 %91 to i32
  %93 = and i32 %92, 1
  %94 = icmp ne i32 %93, 0
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = call i32 @strcmp(ptr %89, ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8))
  %97 = icmp eq i32 %96, 0
  br label %100

98:                                               ; preds = %88
  %99 = icmp eq i64 %90, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8) to i64)
  br label %100

100:                                              ; preds = %95, %98
  %101 = phi i1 [ %99, %98 ], [ %97, %95 ]
  br label %102

102:                                              ; preds = %100
  br i1 %101, label %103, label %111

103:                                              ; preds = %102
  %104 = getelementptr %A, ptr %18, i32 0, i32 0
  %105 = load ptr, ptr %104, align 8
  %106 = getelementptr ptr, ptr %105, i32 2
  %107 = load ptr, ptr %106, align 8
  %108 = invoke i32 %107(ptr %18)
          to label %109 unwind label %790

109:                                              ; preds = %103
  %110 = icmp eq i32 %108, 5
  br label %111

111:                                              ; preds = %109, %102
  %112 = phi i1 [ %110, %109 ], [ %101, %102 ]
  br i1 %112, label %113, label %798

113:                                              ; preds = %111
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  invoke void @A.set_s(double 1.000000e+00)
          to label %114 unwind label %115

114:                                              ; preds = %113
  br label %129

115:                                              ; preds = %113
  %116 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %117 = extractvalue { ptr, i32 } %116, 1
  %118 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %119 = icmp eq i32 %117, %118
  br i1 %119, label %123, label %120

120:                                              ; preds = %115
  %121 = extractvalue { ptr, i32 } %116, 0
  %122 = call ptr @__cxa_begin_catch(ptr %121)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

123:                                              ; preds = %115
  %124 = extractvalue { ptr, i32 } %116, 0
  %125 = call ptr @__cxa_begin_catch(ptr %124)
  store ptr %125, ptr %13, align 8
  %126 = load ptr, ptr %13, align 8
  %127 = getelementptr %Fail, ptr %126, i32 0, i32 1
  %128 = load ptr, ptr %127, align 8
  store ptr %128, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %129

129:                                              ; preds = %114, %123
  %130 = load ptr, ptr %16, align 8
  %131 = ptrtoint ptr %130 to i64
  %132 = icmp ne i64 %131, 0
  %133 = zext i1 %132 to i32
  %134 = and i32 %133, 1
  %135 = icmp ne i32 %134, 0
  br i1 %135, label %136, label %139

136:                                              ; preds = %129
  %137 = call i32 @strcmp(ptr %130, ptr getelementptr inbounds nuw (i8, ptr @s_5856227656950745351, i64 8))
  %138 = icmp eq i32 %137, 0
  br label %141

139:                                              ; preds = %129
  %140 = icmp eq i64 %131, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_5856227656950745351, i64 8) to i64)
  br label %141

141:                                              ; preds = %136, %139
  %142 = phi i1 [ %140, %139 ], [ %138, %136 ]
  br label %143

143:                                              ; preds = %141
  br i1 %142, label %144, label %800

144:                                              ; preds = %143
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %145 = invoke i32 @A.get_s()
          to label %146 unwind label %152

146:                                              ; preds = %144
  %147 = call ptr @GC_malloc_atomic(i64 58)
  %148 = getelementptr i8, ptr %147, i64 8
  %149 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %148, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %145)
  %150 = icmp eq ptr %148, null
  %151 = select i1 %150, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %148
  call void @puts(ptr %151)
  br label %166

152:                                              ; preds = %144
  %153 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %154 = extractvalue { ptr, i32 } %153, 1
  %155 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %156 = icmp eq i32 %154, %155
  br i1 %156, label %160, label %157

157:                                              ; preds = %152
  %158 = extractvalue { ptr, i32 } %153, 0
  %159 = call ptr @__cxa_begin_catch(ptr %158)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

160:                                              ; preds = %152
  %161 = extractvalue { ptr, i32 } %153, 0
  %162 = call ptr @__cxa_begin_catch(ptr %161)
  store ptr %162, ptr %12, align 8
  %163 = load ptr, ptr %12, align 8
  %164 = getelementptr %Fail, ptr %163, i32 0, i32 1
  %165 = load ptr, ptr %164, align 8
  store ptr %165, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %166

166:                                              ; preds = %146, %160
  %167 = load ptr, ptr %16, align 8
  %168 = ptrtoint ptr %167 to i64
  %169 = icmp ne i64 %168, 0
  %170 = zext i1 %169 to i32
  %171 = and i32 %170, 1
  %172 = icmp ne i32 %171, 0
  br i1 %172, label %173, label %176

173:                                              ; preds = %166
  %174 = call i32 @strcmp(ptr %167, ptr getelementptr inbounds nuw (i8, ptr @s_16759446751862886611, i64 8))
  %175 = icmp eq i32 %174, 0
  br label %178

176:                                              ; preds = %166
  %177 = icmp eq i64 %168, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_16759446751862886611, i64 8) to i64)
  br label %178

178:                                              ; preds = %173, %176
  %179 = phi i1 [ %177, %176 ], [ %175, %173 ]
  br label %180

180:                                              ; preds = %178
  br i1 %179, label %181, label %802

181:                                              ; preds = %180
  %182 = call ptr @GC_malloc(i64 24)
  %183 = getelementptr i8, ptr %182, i64 8
  %184 = getelementptr %B, ptr %183, i32 0, i32 0
  %185 = getelementptr %A, ptr %184, i32 0, i32 0
  store ptr @B..vtbl, ptr %185, align 8
  invoke void @A.constructor(ptr %183)
          to label %186 unwind label %790

186:                                              ; preds = %181
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %187 = getelementptr %A, ptr %183, i32 0, i32 0
  %188 = load ptr, ptr %187, align 8
  %189 = getelementptr ptr, ptr %188, i32 3
  %190 = load ptr, ptr %189, align 8
  invoke void %190(ptr %183, double -1.000000e+00)
          to label %191 unwind label %192

191:                                              ; preds = %186
  br label %206

192:                                              ; preds = %186
  %193 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %194 = extractvalue { ptr, i32 } %193, 1
  %195 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %196 = icmp eq i32 %194, %195
  br i1 %196, label %200, label %197

197:                                              ; preds = %192
  %198 = extractvalue { ptr, i32 } %193, 0
  %199 = call ptr @__cxa_begin_catch(ptr %198)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

200:                                              ; preds = %192
  %201 = extractvalue { ptr, i32 } %193, 0
  %202 = call ptr @__cxa_begin_catch(ptr %201)
  store ptr %202, ptr %11, align 8
  %203 = load ptr, ptr %11, align 8
  %204 = getelementptr %Fail, ptr %203, i32 0, i32 1
  %205 = load ptr, ptr %204, align 8
  store ptr %205, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %206

206:                                              ; preds = %191, %200
  %207 = load ptr, ptr %16, align 8
  %208 = ptrtoint ptr %207 to i64
  %209 = icmp ne i64 %208, 0
  %210 = zext i1 %209 to i32
  %211 = and i32 %210, 1
  %212 = icmp ne i32 %211, 0
  br i1 %212, label %213, label %216

213:                                              ; preds = %206
  %214 = call i32 @strcmp(ptr %207, ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8))
  %215 = icmp eq i32 %214, 0
  br label %218

216:                                              ; preds = %206
  %217 = icmp eq i64 %208, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8) to i64)
  br label %218

218:                                              ; preds = %213, %216
  %219 = phi i1 [ %217, %216 ], [ %215, %213 ]
  br label %220

220:                                              ; preds = %218
  br i1 %219, label %221, label %804

221:                                              ; preds = %220
  %222 = call ptr @GC_malloc(i64 24)
  %223 = getelementptr i8, ptr %222, i64 8
  %224 = getelementptr %Box, ptr %223, i32 0, i32 0
  store ptr @Box..vtbl, ptr %224, align 8
  invoke void @Box.constructor(ptr %223)
          to label %225 unwind label %790

225:                                              ; preds = %221
  %226 = ptrtoint ptr %223 to i64
  %227 = icmp ne i64 %226, 0
  br i1 %227, label %228, label %233

228:                                              ; preds = %225
  %229 = getelementptr %Box, ptr %223, i32 0, i32 0
  %230 = load ptr, ptr %229, align 8
  %231 = getelementptr ptr, ptr %230, i32 1
  %232 = load ptr, ptr %231, align 8
  br label %234

233:                                              ; preds = %225
  br label %234

234:                                              ; preds = %228, %233
  %235 = phi ptr [ @Named.null..vtbl, %233 ], [ %232, %228 ]
  %236 = insertvalue { ptr, ptr } undef, ptr %235, 0
  %237 = insertvalue { ptr, ptr } %236, ptr %223, 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %238 = extractvalue { ptr, ptr } %237, 0
  %239 = extractvalue { ptr, ptr } %237, 1
  %240 = getelementptr ptr, ptr %238, i32 2
  %241 = load ptr, ptr %240, align 8
  %242 = ptrtoint ptr %239 to i64
  %243 = ptrtoint ptr %241 to i64
  %244 = add i64 %242, %243
  %245 = inttoptr i64 %244 to ptr
  %246 = extractvalue { ptr, ptr } %237, 0
  %247 = extractvalue { ptr, ptr } %237, 1
  %248 = getelementptr ptr, ptr %246, i32 2
  %249 = load ptr, ptr %248, align 8
  %250 = ptrtoint ptr %249 to i64
  %251 = and i64 %250, -9223372036854775808
  %252 = icmp ne i64 %251, 0
  br i1 %252, label %253, label %264

253:                                              ; preds = %234
  %254 = sub i64 %250, -9223372036854775808
  %255 = inttoptr i64 %254 to ptr
  %256 = getelementptr { ptr, ptr }, ptr %255, i32 0, i32 1
  %257 = load ptr, ptr %256, align 8
  %258 = icmp ne ptr %257, null
  br i1 %258, label %259, label %261

259:                                              ; preds = %253
  invoke void %257(ptr %247, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
          to label %260 unwind label %268

260:                                              ; preds = %259
  br label %262

261:                                              ; preds = %253
  br label %262

262:                                              ; preds = %260, %261
  br label %263

263:                                              ; preds = %262
  br label %265

264:                                              ; preds = %234
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %245, align 8
  br label %265

265:                                              ; preds = %263, %264
  br label %266

266:                                              ; preds = %265
  br label %267

267:                                              ; preds = %266
  br label %282

268:                                              ; preds = %259
  %269 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %270 = extractvalue { ptr, i32 } %269, 1
  %271 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %272 = icmp eq i32 %270, %271
  br i1 %272, label %276, label %273

273:                                              ; preds = %268
  %274 = extractvalue { ptr, i32 } %269, 0
  %275 = call ptr @__cxa_begin_catch(ptr %274)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

276:                                              ; preds = %268
  %277 = extractvalue { ptr, i32 } %269, 0
  %278 = call ptr @__cxa_begin_catch(ptr %277)
  store ptr %278, ptr %10, align 8
  %279 = load ptr, ptr %10, align 8
  %280 = getelementptr %Fail, ptr %279, i32 0, i32 1
  %281 = load ptr, ptr %280, align 8
  store ptr %281, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %282

282:                                              ; preds = %267, %276
  %283 = load ptr, ptr %16, align 8
  %284 = ptrtoint ptr %283 to i64
  %285 = icmp ne i64 %284, 0
  %286 = zext i1 %285 to i32
  %287 = and i32 %286, 1
  %288 = icmp ne i32 %287, 0
  br i1 %288, label %289, label %292

289:                                              ; preds = %282
  %290 = call i32 @strcmp(ptr %283, ptr getelementptr inbounds nuw (i8, ptr @s_7212953690789356384, i64 8))
  %291 = icmp eq i32 %290, 0
  br label %294

292:                                              ; preds = %282
  %293 = icmp eq i64 %284, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_7212953690789356384, i64 8) to i64)
  br label %294

294:                                              ; preds = %289, %292
  %295 = phi i1 [ %293, %292 ], [ %291, %289 ]
  br label %296

296:                                              ; preds = %294
  br i1 %295, label %297, label %806

297:                                              ; preds = %296
  %298 = extractvalue { ptr, ptr } %237, 0
  %299 = extractvalue { ptr, ptr } %237, 1
  %300 = getelementptr ptr, ptr %298, i32 2
  %301 = load ptr, ptr %300, align 8
  %302 = ptrtoint ptr %299 to i64
  %303 = ptrtoint ptr %301 to i64
  %304 = add i64 %302, %303
  %305 = inttoptr i64 %304 to ptr
  %306 = extractvalue { ptr, ptr } %237, 0
  %307 = extractvalue { ptr, ptr } %237, 1
  %308 = getelementptr ptr, ptr %306, i32 2
  %309 = load ptr, ptr %308, align 8
  %310 = ptrtoint ptr %309 to i64
  %311 = and i64 %310, -9223372036854775808
  %312 = icmp ne i64 %311, 0
  br i1 %312, label %313, label %324

313:                                              ; preds = %297
  %314 = sub i64 %310, -9223372036854775808
  %315 = inttoptr i64 %314 to ptr
  %316 = getelementptr { ptr, ptr }, ptr %315, i32 0, i32 1
  %317 = load ptr, ptr %316, align 8
  %318 = icmp ne ptr %317, null
  br i1 %318, label %319, label %321

319:                                              ; preds = %313
  invoke void %317(ptr %307, ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8))
          to label %320 unwind label %790

320:                                              ; preds = %319
  br label %322

321:                                              ; preds = %313
  br label %322

322:                                              ; preds = %320, %321
  br label %323

323:                                              ; preds = %322
  br label %325

324:                                              ; preds = %297
  store ptr getelementptr inbounds nuw (i8, ptr @s_16077166718034104, i64 8), ptr %305, align 8
  br label %325

325:                                              ; preds = %323, %324
  br label %326

326:                                              ; preds = %325
  br label %327

327:                                              ; preds = %326
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %328 = extractvalue { ptr, ptr } %237, 0
  %329 = extractvalue { ptr, ptr } %237, 1
  %330 = getelementptr ptr, ptr %328, i32 2
  %331 = load ptr, ptr %330, align 8
  %332 = ptrtoint ptr %329 to i64
  %333 = ptrtoint ptr %331 to i64
  %334 = add i64 %332, %333
  %335 = inttoptr i64 %334 to ptr
  %336 = extractvalue { ptr, ptr } %237, 0
  %337 = extractvalue { ptr, ptr } %237, 1
  %338 = getelementptr ptr, ptr %336, i32 2
  %339 = load ptr, ptr %338, align 8
  %340 = ptrtoint ptr %339 to i64
  %341 = and i64 %340, -9223372036854775808
  %342 = icmp ne i64 %341, 0
  br i1 %342, label %343, label %350

343:                                              ; preds = %327
  %344 = sub i64 %340, -9223372036854775808
  %345 = inttoptr i64 %344 to ptr
  %346 = getelementptr { ptr, ptr }, ptr %345, i32 0, i32 0
  %347 = load ptr, ptr %346, align 8
  %348 = invoke ptr %347(ptr %337)
          to label %349 unwind label %358

349:                                              ; preds = %343
  br label %352

350:                                              ; preds = %327
  %351 = load ptr, ptr %335, align 8
  br label %352

352:                                              ; preds = %349, %350
  %353 = phi ptr [ %351, %350 ], [ %348, %349 ]
  br label %354

354:                                              ; preds = %352
  br label %355

355:                                              ; preds = %354
  %356 = icmp eq ptr %353, null
  %357 = select i1 %356, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %353
  call void @puts(ptr %357)
  br label %372

358:                                              ; preds = %343
  %359 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %360 = extractvalue { ptr, i32 } %359, 1
  %361 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %362 = icmp eq i32 %360, %361
  br i1 %362, label %366, label %363

363:                                              ; preds = %358
  %364 = extractvalue { ptr, i32 } %359, 0
  %365 = call ptr @__cxa_begin_catch(ptr %364)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

366:                                              ; preds = %358
  %367 = extractvalue { ptr, i32 } %359, 0
  %368 = call ptr @__cxa_begin_catch(ptr %367)
  store ptr %368, ptr %9, align 8
  %369 = load ptr, ptr %9, align 8
  %370 = getelementptr %Fail, ptr %369, i32 0, i32 1
  %371 = load ptr, ptr %370, align 8
  store ptr %371, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %372

372:                                              ; preds = %355, %366
  %373 = load ptr, ptr %16, align 8
  %374 = ptrtoint ptr %373 to i64
  %375 = icmp ne i64 %374, 0
  %376 = zext i1 %375 to i32
  %377 = and i32 %376, 1
  %378 = icmp ne i32 %377, 0
  br i1 %378, label %379, label %382

379:                                              ; preds = %372
  %380 = call i32 @strcmp(ptr %373, ptr getelementptr inbounds nuw (i8, ptr @s_1241800711098863692, i64 8))
  %381 = icmp eq i32 %380, 0
  br label %384

382:                                              ; preds = %372
  %383 = icmp eq i64 %374, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_1241800711098863692, i64 8) to i64)
  br label %384

384:                                              ; preds = %379, %382
  %385 = phi i1 [ %383, %382 ], [ %381, %379 ]
  br label %386

386:                                              ; preds = %384
  br i1 %385, label %387, label %808

387:                                              ; preds = %386
  %388 = extractvalue { ptr, ptr } %237, 0
  %389 = extractvalue { ptr, ptr } %237, 1
  %390 = getelementptr ptr, ptr %388, i32 2
  %391 = load ptr, ptr %390, align 8
  %392 = ptrtoint ptr %389 to i64
  %393 = ptrtoint ptr %391 to i64
  %394 = add i64 %392, %393
  %395 = inttoptr i64 %394 to ptr
  %396 = extractvalue { ptr, ptr } %237, 0
  %397 = extractvalue { ptr, ptr } %237, 1
  %398 = getelementptr ptr, ptr %396, i32 2
  %399 = load ptr, ptr %398, align 8
  %400 = ptrtoint ptr %399 to i64
  %401 = and i64 %400, -9223372036854775808
  %402 = icmp ne i64 %401, 0
  br i1 %402, label %403, label %414

403:                                              ; preds = %387
  %404 = sub i64 %400, -9223372036854775808
  %405 = inttoptr i64 %404 to ptr
  %406 = getelementptr { ptr, ptr }, ptr %405, i32 0, i32 1
  %407 = load ptr, ptr %406, align 8
  %408 = icmp ne ptr %407, null
  br i1 %408, label %409, label %411

409:                                              ; preds = %403
  invoke void %407(ptr %397, ptr getelementptr inbounds nuw (i8, ptr @s_626102635892993007, i64 8))
          to label %410 unwind label %790

410:                                              ; preds = %409
  br label %412

411:                                              ; preds = %403
  br label %412

412:                                              ; preds = %410, %411
  br label %413

413:                                              ; preds = %412
  br label %415

414:                                              ; preds = %387
  store ptr getelementptr inbounds nuw (i8, ptr @s_626102635892993007, i64 8), ptr %395, align 8
  br label %415

415:                                              ; preds = %413, %414
  br label %416

416:                                              ; preds = %415
  br label %417

417:                                              ; preds = %416
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %418 = extractvalue { ptr, ptr } %237, 0
  %419 = extractvalue { ptr, ptr } %237, 1
  %420 = getelementptr ptr, ptr %418, i32 2
  %421 = load ptr, ptr %420, align 8
  %422 = ptrtoint ptr %419 to i64
  %423 = ptrtoint ptr %421 to i64
  %424 = add i64 %422, %423
  %425 = inttoptr i64 %424 to ptr
  %426 = extractvalue { ptr, ptr } %237, 0
  %427 = extractvalue { ptr, ptr } %237, 1
  %428 = getelementptr ptr, ptr %426, i32 2
  %429 = load ptr, ptr %428, align 8
  %430 = ptrtoint ptr %429 to i64
  %431 = and i64 %430, -9223372036854775808
  %432 = icmp ne i64 %431, 0
  br i1 %432, label %433, label %440

433:                                              ; preds = %417
  %434 = sub i64 %430, -9223372036854775808
  %435 = inttoptr i64 %434 to ptr
  %436 = getelementptr { ptr, ptr }, ptr %435, i32 0, i32 0
  %437 = load ptr, ptr %436, align 8
  %438 = invoke ptr %437(ptr %427)
          to label %439 unwind label %446

439:                                              ; preds = %433
  br label %442

440:                                              ; preds = %417
  %441 = load ptr, ptr %425, align 8
  br label %442

442:                                              ; preds = %439, %440
  %443 = phi ptr [ %441, %440 ], [ %438, %439 ]
  br label %444

444:                                              ; preds = %442
  br label %445

445:                                              ; preds = %444
  store ptr %443, ptr %16, align 8
  br label %457

446:                                              ; preds = %433
  %447 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %448 = extractvalue { ptr, i32 } %447, 1
  %449 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %450 = icmp eq i32 %448, %449
  br i1 %450, label %454, label %451

451:                                              ; preds = %446
  %452 = extractvalue { ptr, i32 } %447, 0
  %453 = call ptr @__cxa_begin_catch(ptr %452)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

454:                                              ; preds = %446
  %455 = extractvalue { ptr, i32 } %447, 0
  %456 = call ptr @__cxa_begin_catch(ptr %455)
  store ptr %456, ptr %8, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_12054661417104037398, i64 8), ptr %16, align 8
  call void @__cxa_end_catch()
  br label %457

457:                                              ; preds = %445, %454
  %458 = load ptr, ptr %16, align 8
  %459 = ptrtoint ptr %458 to i64
  %460 = icmp ne i64 %459, 0
  %461 = zext i1 %460 to i32
  %462 = and i32 %461, 1
  %463 = icmp ne i32 %462, 0
  br i1 %463, label %464, label %467

464:                                              ; preds = %457
  %465 = call i32 @strcmp(ptr %458, ptr getelementptr inbounds nuw (i8, ptr @s_626102635892993007, i64 8))
  %466 = icmp eq i32 %465, 0
  br label %469

467:                                              ; preds = %457
  %468 = icmp eq i64 %459, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_626102635892993007, i64 8) to i64)
  br label %469

469:                                              ; preds = %464, %467
  %470 = phi i1 [ %468, %467 ], [ %466, %464 ]
  br label %471

471:                                              ; preds = %469
  br i1 %470, label %472, label %810

472:                                              ; preds = %471
  store i1 false, ptr %7, align 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %473 = extractvalue { ptr, ptr } %237, 0
  %474 = extractvalue { ptr, ptr } %237, 1
  %475 = getelementptr ptr, ptr %473, i32 2
  %476 = load ptr, ptr %475, align 8
  %477 = ptrtoint ptr %474 to i64
  %478 = ptrtoint ptr %476 to i64
  %479 = add i64 %477, %478
  %480 = inttoptr i64 %479 to ptr
  %481 = extractvalue { ptr, ptr } %237, 0
  %482 = extractvalue { ptr, ptr } %237, 1
  %483 = getelementptr ptr, ptr %481, i32 2
  %484 = load ptr, ptr %483, align 8
  %485 = ptrtoint ptr %484 to i64
  %486 = and i64 %485, -9223372036854775808
  %487 = icmp ne i64 %486, 0
  br i1 %487, label %488, label %499

488:                                              ; preds = %472
  %489 = sub i64 %485, -9223372036854775808
  %490 = inttoptr i64 %489 to ptr
  %491 = getelementptr { ptr, ptr }, ptr %490, i32 0, i32 1
  %492 = load ptr, ptr %491, align 8
  %493 = icmp ne ptr %492, null
  br i1 %493, label %494, label %496

494:                                              ; preds = %488
  invoke void %492(ptr %482, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
          to label %495 unwind label %503

495:                                              ; preds = %494
  br label %497

496:                                              ; preds = %488
  br label %497

497:                                              ; preds = %495, %496
  br label %498

498:                                              ; preds = %497
  br label %500

499:                                              ; preds = %472
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %480, align 8
  br label %500

500:                                              ; preds = %498, %499
  br label %501

501:                                              ; preds = %500
  br label %502

502:                                              ; preds = %501
  store i1 true, ptr %7, align 1
  br label %523

503:                                              ; preds = %494
  %504 = landingpad { ptr, i32 }
          catch ptr null
  %505 = extractvalue { ptr, i32 } %504, 0
  %506 = call ptr @__cxa_begin_catch(ptr %505)
  store i1 true, ptr %7, align 1
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad4

endcatch.pad4:                                    ; preds = %503
  %507 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught3

508:                                              ; No predecessors!
  %509 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  br label %.caught3

.caught3:                                         ; preds = %endcatch.pad4, %508
  %510 = phi { ptr, i32 } [ %509, %508 ], [ %507, %endcatch.pad4 ]
  %511 = extractvalue { ptr, i32 } %510, 1
  %512 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %513 = icmp eq i32 %511, %512
  br i1 %513, label %517, label %514

514:                                              ; preds = %.caught3
  %515 = extractvalue { ptr, i32 } %510, 0
  %516 = call ptr @__cxa_begin_catch(ptr %515)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

517:                                              ; preds = %.caught3
  %518 = extractvalue { ptr, i32 } %510, 0
  %519 = call ptr @__cxa_begin_catch(ptr %518)
  store ptr %519, ptr %6, align 8
  %520 = load ptr, ptr %6, align 8
  %521 = getelementptr %Fail, ptr %520, i32 0, i32 1
  %522 = load ptr, ptr %521, align 8
  store ptr %522, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %523

523:                                              ; preds = %502, %517
  %524 = load i1, ptr %7, align 1
  br i1 %524, label %525, label %540

525:                                              ; preds = %523
  %526 = load ptr, ptr %16, align 8
  %527 = ptrtoint ptr %526 to i64
  %528 = icmp ne i64 %527, 0
  %529 = zext i1 %528 to i32
  %530 = and i32 %529, 1
  %531 = icmp ne i32 %530, 0
  br i1 %531, label %532, label %535

532:                                              ; preds = %525
  %533 = call i32 @strcmp(ptr %526, ptr getelementptr inbounds nuw (i8, ptr @s_7212953690789356384, i64 8))
  %534 = icmp eq i32 %533, 0
  br label %537

535:                                              ; preds = %525
  %536 = icmp eq i64 %527, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_7212953690789356384, i64 8) to i64)
  br label %537

537:                                              ; preds = %532, %535
  %538 = phi i1 [ %536, %535 ], [ %534, %532 ]
  br label %539

539:                                              ; preds = %537
  br label %540

540:                                              ; preds = %539, %523
  %541 = phi i1 [ %538, %539 ], [ %524, %523 ]
  br i1 %541, label %542, label %812

542:                                              ; preds = %540
  %543 = call ptr @GC_malloc(i64 24)
  %544 = getelementptr i8, ptr %543, i64 8
  %545 = getelementptr %Plain, ptr %544, i32 0, i32 0
  store ptr @Plain..vtbl, ptr %545, align 8
  invoke void @Plain.constructor(ptr %544)
          to label %546 unwind label %790

546:                                              ; preds = %542
  %547 = ptrtoint ptr %544 to i64
  %548 = icmp ne i64 %547, 0
  br i1 %548, label %549, label %554

549:                                              ; preds = %546
  %550 = getelementptr %Plain, ptr %544, i32 0, i32 0
  %551 = load ptr, ptr %550, align 8
  %552 = getelementptr ptr, ptr %551, i32 1
  %553 = load ptr, ptr %552, align 8
  br label %555

554:                                              ; preds = %546
  br label %555

555:                                              ; preds = %549, %554
  %556 = phi ptr [ @Named.null..vtbl, %554 ], [ %553, %549 ]
  %557 = insertvalue { ptr, ptr } undef, ptr %556, 0
  %558 = insertvalue { ptr, ptr } %557, ptr %544, 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %559 = extractvalue { ptr, ptr } %558, 0
  %560 = extractvalue { ptr, ptr } %558, 1
  %561 = getelementptr ptr, ptr %559, i32 2
  %562 = load ptr, ptr %561, align 8
  %563 = ptrtoint ptr %560 to i64
  %564 = ptrtoint ptr %562 to i64
  %565 = add i64 %563, %564
  %566 = inttoptr i64 %565 to ptr
  %567 = extractvalue { ptr, ptr } %558, 0
  %568 = extractvalue { ptr, ptr } %558, 1
  %569 = getelementptr ptr, ptr %567, i32 2
  %570 = load ptr, ptr %569, align 8
  %571 = ptrtoint ptr %570 to i64
  %572 = and i64 %571, -9223372036854775808
  %573 = icmp ne i64 %572, 0
  br i1 %573, label %574, label %585

574:                                              ; preds = %555
  %575 = sub i64 %571, -9223372036854775808
  %576 = inttoptr i64 %575 to ptr
  %577 = getelementptr { ptr, ptr }, ptr %576, i32 0, i32 1
  %578 = load ptr, ptr %577, align 8
  %579 = icmp ne ptr %578, null
  br i1 %579, label %580, label %582

580:                                              ; preds = %574
  invoke void %578(ptr %568, ptr getelementptr inbounds nuw (i8, ptr @s_13597551105163972623, i64 8))
          to label %581 unwind label %617

581:                                              ; preds = %580
  br label %583

582:                                              ; preds = %574
  br label %583

583:                                              ; preds = %581, %582
  br label %584

584:                                              ; preds = %583
  br label %586

585:                                              ; preds = %555
  store ptr getelementptr inbounds nuw (i8, ptr @s_13597551105163972623, i64 8), ptr %566, align 8
  br label %586

586:                                              ; preds = %584, %585
  br label %587

587:                                              ; preds = %586
  br label %588

588:                                              ; preds = %587
  %589 = extractvalue { ptr, ptr } %558, 0
  %590 = extractvalue { ptr, ptr } %558, 1
  %591 = getelementptr ptr, ptr %589, i32 2
  %592 = load ptr, ptr %591, align 8
  %593 = ptrtoint ptr %590 to i64
  %594 = ptrtoint ptr %592 to i64
  %595 = add i64 %593, %594
  %596 = inttoptr i64 %595 to ptr
  %597 = extractvalue { ptr, ptr } %558, 0
  %598 = extractvalue { ptr, ptr } %558, 1
  %599 = getelementptr ptr, ptr %597, i32 2
  %600 = load ptr, ptr %599, align 8
  %601 = ptrtoint ptr %600 to i64
  %602 = and i64 %601, -9223372036854775808
  %603 = icmp ne i64 %602, 0
  br i1 %603, label %604, label %611

604:                                              ; preds = %588
  %605 = sub i64 %601, -9223372036854775808
  %606 = inttoptr i64 %605 to ptr
  %607 = getelementptr { ptr, ptr }, ptr %606, i32 0, i32 0
  %608 = load ptr, ptr %607, align 8
  %609 = invoke ptr %608(ptr %598)
          to label %610 unwind label %617

610:                                              ; preds = %604
  br label %613

611:                                              ; preds = %588
  %612 = load ptr, ptr %596, align 8
  br label %613

613:                                              ; preds = %610, %611
  %614 = phi ptr [ %612, %611 ], [ %609, %610 ]
  br label %615

615:                                              ; preds = %613
  br label %616

616:                                              ; preds = %615
  store ptr %614, ptr %16, align 8
  br label %628

617:                                              ; preds = %604, %580
  %618 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %619 = extractvalue { ptr, i32 } %618, 1
  %620 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %621 = icmp eq i32 %619, %620
  br i1 %621, label %625, label %622

622:                                              ; preds = %617
  %623 = extractvalue { ptr, i32 } %618, 0
  %624 = call ptr @__cxa_begin_catch(ptr %623)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

625:                                              ; preds = %617
  %626 = extractvalue { ptr, i32 } %618, 0
  %627 = call ptr @__cxa_begin_catch(ptr %626)
  store ptr %627, ptr %5, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_12054661417104037398, i64 8), ptr %16, align 8
  call void @__cxa_end_catch()
  br label %628

628:                                              ; preds = %616, %625
  %629 = load ptr, ptr %16, align 8
  %630 = ptrtoint ptr %629 to i64
  %631 = icmp ne i64 %630, 0
  %632 = zext i1 %631 to i32
  %633 = and i32 %632, 1
  %634 = icmp ne i32 %633, 0
  br i1 %634, label %635, label %638

635:                                              ; preds = %628
  %636 = call i32 @strcmp(ptr %629, ptr getelementptr inbounds nuw (i8, ptr @s_13597551105163972623, i64 8))
  %637 = icmp eq i32 %636, 0
  br label %640

638:                                              ; preds = %628
  %639 = icmp eq i64 %630, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_13597551105163972623, i64 8) to i64)
  br label %640

640:                                              ; preds = %635, %638
  %641 = phi i1 [ %639, %638 ], [ %637, %635 ]
  br label %642

642:                                              ; preds = %640
  br i1 %641, label %643, label %814

643:                                              ; preds = %642
  %644 = call ptr @GC_malloc(i64 16)
  %645 = getelementptr i8, ptr %644, i64 8
  %646 = getelementptr %Square, ptr %645, i32 0, i32 0
  %647 = getelementptr %Shape, ptr %646, i32 0, i32 0
  store ptr @Square..vtbl, ptr %647, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %648 = getelementptr %Shape, ptr %645, i32 0, i32 0
  %649 = load ptr, ptr %648, align 8
  %650 = getelementptr ptr, ptr %649, i32 1
  %651 = load ptr, ptr %650, align 8
  %652 = invoke double %651(ptr %645)
          to label %653 unwind label %658

653:                                              ; preds = %643
  %654 = call ptr @GC_malloc_atomic(i64 40)
  %655 = getelementptr i8, ptr %654, i64 8
  call void @__ts_number_to_string(ptr %655, double %652)
  %656 = icmp eq ptr %655, null
  %657 = select i1 %656, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %655
  call void @puts(ptr %657)
  br label %672

658:                                              ; preds = %643
  %659 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %660 = extractvalue { ptr, i32 } %659, 1
  %661 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %662 = icmp eq i32 %660, %661
  br i1 %662, label %666, label %663

663:                                              ; preds = %658
  %664 = extractvalue { ptr, i32 } %659, 0
  %665 = call ptr @__cxa_begin_catch(ptr %664)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

666:                                              ; preds = %658
  %667 = extractvalue { ptr, i32 } %659, 0
  %668 = call ptr @__cxa_begin_catch(ptr %667)
  store ptr %668, ptr %4, align 8
  %669 = load ptr, ptr %4, align 8
  %670 = getelementptr %Fail, ptr %669, i32 0, i32 1
  %671 = load ptr, ptr %670, align 8
  store ptr %671, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %672

672:                                              ; preds = %653, %666
  %673 = load ptr, ptr %16, align 8
  %674 = ptrtoint ptr %673 to i64
  %675 = icmp ne i64 %674, 0
  %676 = zext i1 %675 to i32
  %677 = and i32 %676, 1
  %678 = icmp ne i32 %677, 0
  br i1 %678, label %679, label %682

679:                                              ; preds = %672
  %680 = call i32 @strcmp(ptr %673, ptr getelementptr inbounds nuw (i8, ptr @s_17157602999399885131, i64 8))
  %681 = icmp eq i32 %680, 0
  br label %684

682:                                              ; preds = %672
  %683 = icmp eq i64 %674, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_17157602999399885131, i64 8) to i64)
  br label %684

684:                                              ; preds = %679, %682
  %685 = phi i1 [ %683, %682 ], [ %681, %679 ]
  br label %686

686:                                              ; preds = %684
  br i1 %685, label %687, label %816

687:                                              ; preds = %686
  store ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), ptr %16, align 8
  %688 = getelementptr %Shape, ptr %645, i32 0, i32 0
  %689 = load ptr, ptr %688, align 8
  %690 = getelementptr ptr, ptr %689, i32 2
  %691 = load ptr, ptr %690, align 8
  invoke void %691(ptr %645, double 1.000000e+00)
          to label %692 unwind label %693

692:                                              ; preds = %687
  br label %707

693:                                              ; preds = %687
  %694 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  %695 = extractvalue { ptr, i32 } %694, 1
  %696 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %697 = icmp eq i32 %695, %696
  br i1 %697, label %701, label %698

698:                                              ; preds = %693
  %699 = extractvalue { ptr, i32 } %694, 0
  %700 = call ptr @__cxa_begin_catch(ptr %699)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

701:                                              ; preds = %693
  %702 = extractvalue { ptr, i32 } %694, 0
  %703 = call ptr @__cxa_begin_catch(ptr %702)
  store ptr %703, ptr %3, align 8
  %704 = load ptr, ptr %3, align 8
  %705 = getelementptr %Fail, ptr %704, i32 0, i32 1
  %706 = load ptr, ptr %705, align 8
  store ptr %706, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %707

707:                                              ; preds = %692, %701
  %708 = load ptr, ptr %16, align 8
  %709 = ptrtoint ptr %708 to i64
  %710 = icmp ne i64 %709, 0
  %711 = zext i1 %710 to i32
  %712 = and i32 %711, 1
  %713 = icmp ne i32 %712, 0
  br i1 %713, label %714, label %717

714:                                              ; preds = %707
  %715 = call i32 @strcmp(ptr %708, ptr getelementptr inbounds nuw (i8, ptr @s_6254630195092463135, i64 8))
  %716 = icmp eq i32 %715, 0
  br label %719

717:                                              ; preds = %707
  %718 = icmp eq i64 %709, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_6254630195092463135, i64 8) to i64)
  br label %719

719:                                              ; preds = %714, %717
  %720 = phi i1 [ %718, %717 ], [ %716, %714 ]
  br label %721

721:                                              ; preds = %719
  br i1 %720, label %722, label %818

722:                                              ; preds = %721
  store i1 false, ptr %2, align 1
  %723 = getelementptr %A, ptr %18, i32 0, i32 0
  %724 = load ptr, ptr %723, align 8
  %725 = getelementptr ptr, ptr %724, i32 3
  %726 = load ptr, ptr %725, align 8
  invoke void %726(ptr %18, double -3.000000e+00)
          to label %727 unwind label %728

727:                                              ; preds = %722
  store i1 true, ptr %2, align 1
  br label %748

728:                                              ; preds = %722
  %729 = landingpad { ptr, i32 }
          catch ptr null
  %730 = extractvalue { ptr, i32 } %729, 0
  %731 = call ptr @__cxa_begin_catch(ptr %730)
  store i1 true, ptr %2, align 1
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad2

endcatch.pad2:                                    ; preds = %728
  %732 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  call void @__cxa_end_catch()
  br label %.caught1

733:                                              ; No predecessors!
  %734 = landingpad { ptr, i32 }
          catch ptr @_ZTIP4Fail
          catch ptr null
  br label %.caught1

.caught1:                                         ; preds = %endcatch.pad2, %733
  %735 = phi { ptr, i32 } [ %734, %733 ], [ %732, %endcatch.pad2 ]
  %736 = extractvalue { ptr, i32 } %735, 1
  %737 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP4Fail)
  %738 = icmp eq i32 %736, %737
  br i1 %738, label %742, label %739

739:                                              ; preds = %.caught1
  %740 = extractvalue { ptr, i32 } %735, 0
  %741 = call ptr @__cxa_begin_catch(ptr %740)
  invoke void @__cxa_rethrow()
          to label %793 unwind label %endcatch.pad

742:                                              ; preds = %.caught1
  %743 = extractvalue { ptr, i32 } %735, 0
  %744 = call ptr @__cxa_begin_catch(ptr %743)
  store ptr %744, ptr %1, align 8
  %745 = load ptr, ptr %1, align 8
  %746 = getelementptr %Fail, ptr %745, i32 0, i32 1
  %747 = load ptr, ptr %746, align 8
  store ptr %747, ptr %16, align 8
  call void @__cxa_end_catch()
  br label %748

748:                                              ; preds = %727, %742
  %749 = load i1, ptr %2, align 1
  br i1 %749, label %750, label %765

750:                                              ; preds = %748
  %751 = load ptr, ptr %16, align 8
  %752 = ptrtoint ptr %751 to i64
  %753 = icmp ne i64 %752, 0
  %754 = zext i1 %753 to i32
  %755 = and i32 %754, 1
  %756 = icmp ne i32 %755, 0
  br i1 %756, label %757, label %760

757:                                              ; preds = %750
  %758 = call i32 @strcmp(ptr %751, ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8))
  %759 = icmp eq i32 %758, 0
  br label %762

760:                                              ; preds = %750
  %761 = icmp eq i64 %752, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8) to i64)
  br label %762

762:                                              ; preds = %757, %760
  %763 = phi i1 [ %761, %760 ], [ %759, %757 ]
  br label %764

764:                                              ; preds = %762
  br label %765

765:                                              ; preds = %764, %748
  %766 = phi i1 [ %763, %764 ], [ %749, %748 ]
  br i1 %766, label %767, label %820

767:                                              ; preds = %765
  %768 = call ptr @GC_malloc(i64 16)
  %769 = getelementptr i8, ptr %768, i64 8
  %770 = getelementptr { ptr }, ptr %769, i32 0, i32 0
  store ptr %18, ptr %770, align 8
  %771 = insertvalue { ptr, ptr, ptr } { ptr @.f_main..afL129C19FH2287545390, ptr null, ptr null }, ptr %769, 1
  %772 = insertvalue { ptr, ptr, ptr } %771, ptr getelementptr inbounds nuw (i8, ptr @td_7761478249627842675_object, i64 40), 2
  %773 = invoke ptr @caught({ ptr, ptr, ptr } %772)
          to label %774 unwind label %790

774:                                              ; preds = %767
  %775 = ptrtoint ptr %773 to i64
  %776 = icmp ne i64 %775, 0
  %777 = zext i1 %776 to i32
  %778 = and i32 %777, 1
  %779 = icmp ne i32 %778, 0
  br i1 %779, label %780, label %783

780:                                              ; preds = %774
  %781 = call i32 @strcmp(ptr %773, ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8))
  %782 = icmp eq i32 %781, 0
  br label %785

783:                                              ; preds = %774
  %784 = icmp eq i64 %775, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_9384242791558811171, i64 8) to i64)
  br label %785

785:                                              ; preds = %780, %783
  %786 = phi i1 [ %784, %783 ], [ %782, %780 ]
  br label %787

787:                                              ; preds = %785
  br i1 %786, label %788, label %822

788:                                              ; preds = %787
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

endcatch.pad:                                     ; preds = %197, %31, %273, %79, %363, %157, %451, %120, %622, %514, %663, %739, %698
  %789 = landingpad { ptr, i32 }
          cleanup
  call void @__cxa_end_catch()
  br label %.caught

790:                                              ; preds = %767, %542, %409, %319, %221, %181, %103, %60, %55, %0
  %791 = landingpad { ptr, i32 }
          cleanup
  br label %.caught

.caught:                                          ; preds = %endcatch.pad, %790
  %792 = phi { ptr, i32 } [ %791, %790 ], [ %789, %endcatch.pad ]
  resume { ptr, i32 } %792

793:                                              ; preds = %822, %820, %739, %728, %818, %698, %816, %663, %814, %622, %812, %514, %503, %810, %451, %808, %363, %806, %273, %804, %197, %802, %157, %800, %120, %798, %79, %796, %794, %31
  unreachable

794:                                              ; preds = %54
  %795 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_9384242791558811171, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 60, ptr null)
  br label %793

796:                                              ; preds = %66
  %797 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_8058474561178550819, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 63, ptr null)
  br label %793

798:                                              ; preds = %111
  %799 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_11021489436266481586, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 67, ptr null)
  br label %793

800:                                              ; preds = %143
  %801 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_5856227656950745351, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 71, ptr null)
  br label %793

802:                                              ; preds = %180
  %803 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16759446751862886611, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 75, ptr null)
  br label %793

804:                                              ; preds = %220
  %805 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_6236424509574238428, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 80, ptr null)
  br label %793

806:                                              ; preds = %296
  %807 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_16583843082110648990, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 85, ptr null)
  br label %793

808:                                              ; preds = %386
  %809 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12515707135700146330, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 90, ptr null)
  br label %793

810:                                              ; preds = %471
  %811 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1910839327606885957, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 95, ptr null)
  br label %793

812:                                              ; preds = %540
  %813 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_4236572222742585917, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 104, ptr null)
  br label %793

814:                                              ; preds = %642
  %815 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_1979075331638643099, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 110, ptr null)
  br label %793

816:                                              ; preds = %686
  %817 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_13269171204112287757, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 115, ptr null)
  br label %793

818:                                              ; preds = %721
  %819 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_7018100576640315633, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 118, ptr null)
  br label %793

820:                                              ; preds = %765
  %821 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_18175057562723240456, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 127, ptr null)
  br label %793

822:                                              ; preds = %787
  %823 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_12895414482988252627, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8781166903375489039, i64 8), i32 129, ptr null)
  br label %793
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #4

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nounwind memory(none) }
attributes #2 = { mustprogress }
attributes #3 = { mustprogress "personality" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

