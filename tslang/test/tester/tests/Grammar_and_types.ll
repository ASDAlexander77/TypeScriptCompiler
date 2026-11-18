; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a_1_5762750502333681466 = internal constant <1 x i32> splat (i32 1)
@__ts_empty_array = internal constant { ptr, i64, i64 } zeroinitializer
@s_2125742250238303305 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFlocal value\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_16101355973854746 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbar\00", align 8
@s_11131119937623569912 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFbar2\00", align 8
@s_12638124528392833969 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF.\00", align 8
@s_422219368153924640 = internal constant [30 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFSorry, we don't sell \00", align 8
@s_4834688834925861241 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHonda\00", align 8
@s_14695981039346656037 = internal constant [9 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\00", align 8
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
@s_17791929768168046296 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%s%c%s%se%+d\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@s_2506949098668931267 = internal constant [88 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 2028 2029 0085 t v f b r n\00", align 8
@s_2851890610989147717 = internal constant [84 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRoses are red,\0A    Violets are blue.\0A    Sugar is sweet,\0A    and so is foo.\00", align 8
@s_5476372761493277963 = internal constant [65 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthis string     is broken     across multiple     lines.\00", align 8
@s_5594953940062847258 = internal constant [16 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFc:\\temp\00", align 8
@s_10966898795647604191 = internal constant [62 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHe read \22The Cremation of Sam McGee\22 by R.W. Service.\00", align 8
@s_3432704340853296579 = internal constant [28 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFEscaping characters\00", align 8
@s_6321389093131908277 = internal constant [44 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFUsing special characters in strings\00", align 8
@s_27830736345390744 = internal constant [22 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFParticipant \22\00", align 8
@s_3513325390892711705 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\22 is \00", align 8
@s_13510339310099422788 = internal constant [20 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF years old.\00", align 8
@s_345403531931479259 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFMika\00", align 8
@s_17664890869449432037 = internal constant [35 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFString interpolation - tag\00", align 8
@s_12638141021067257134 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF?\00", align 8
@s_5920926082801741360 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF, how are you \00", align 8
@s_10530820969299836017 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHello \00", align 8
@s_11822328890338333214 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtoday\00", align 8
@s_1609583978338260660 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBob\00", align 8
@s_3617302750436170686 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFString interpolation\00", align 8
@s_2694515762537206396 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFString literals\00", align 8
@s_305597083721566844 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRegExp literals\00", align 8
@s_15379658938586843080 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFAn empty string\00", align 8
@s_17317583992087840406 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBang!\00", align 8
@s_3753534942992164126 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFSaab\00", align 8
@s_11431534339755074821 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFJeep\00", align 8
@s_617707879461881500 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFMazda\00", align 8
@s_2738302745049774710 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFSaturn\00", align 8
@s_17292252969566779935 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFToyota\00", align 8
@s_9255881455004091586 = internal constant [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFObject literals\00", align 8
@s_14669464332616883376 = internal constant [32 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFloating-point literals\00", align 8
@s_2658421951147754508 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNumeric literals\00", align 8
@s_13749028156498010861 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBoolean literals\00", align 8
@s_14717692112504537385 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFschool\00", align 8
@s_4624382957487889774 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFhome\00", align 8
@s_6599060811930723210 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFAngel\00", align 8
@s_4394248473669529567 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFLion\00", align 8
@s_7934265953746294420 = internal constant [21 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFrench Roast\00", align 8
@s_12395442134831387347 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFColombian\00", align 8
@s_9172927397968420710 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFKona\00", align 8
@s_11276182295403084134 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFArray literals\00", align 8
@s_2996941089346072501 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFLiterals\00", align 8
@s_5001449799818188079 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF101\00", align 8
@s_13141189159377525121 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFConverting strings to numbers\00", align 8
@s_9276932448023608736 = internal constant [37 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFNumbers and the '+' operator\00", align 8
@s_10634310313915411333 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFData type conversion\00", align 8
@s_14524023967120200944 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFHTML\00", align 8
@s_811268167711933856 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFCSS\00", align 8
@s_5766643768467844826 = internal constant [19 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFotherValue\00", align 8
@s_8999596768310594794 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFvalue\00", align 8
@s_3974561738185464706 = internal constant [18 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFConstants\00", align 8
@s_8570510144365560058 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFFunction hoisting\00", align 8
@s_452236695178438998 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFmy value\00", align 8
@s_16011791931586107870 = internal constant [26 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFVariable hoisting\00", align 8
@s_17543757828066490973 = internal constant [23 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFVariable scope\00", align 8
@s_15442652226749703132 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFn * 32\00", align 8
@s_16176116648655433384 = internal constant [29 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFEvaluating variables\00", align 8
@s_15857971242039570103 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFComments\00", align 8
@s_9625390261332436968 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfoobar\00", align 8
@s_1564491103094674154 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFBasics\00", align 8
@frmt_11053768727887132666 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF%-i\00", align 8
@__false__ = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFfalse\00", align 8
@__true__ = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFtrue\00", align 8
@s_12638117931323064703 = internal constant [10 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF \00", align 8
@m_17914885825359534683 = internal constant [15 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFassert\00", align 8
@f_2330421925823702781 = internal constant [107 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\Grammar_and_types.ts\00", align 8
@a_1_16629603895491369024 = internal constant [1 x ptr] [ptr getelementptr inbounds nuw (i8, ptr @s_2506949098668931267, i64 8)]
@a_3_2084061994484565833 = internal constant [3 x ptr] [ptr getelementptr inbounds nuw (i8, ptr @s_27830736345390744, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_3513325390892711705, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_13510339310099422788, i64 8)]
@a_3_6335243048188567126 = internal constant [3 x ptr] [ptr getelementptr inbounds nuw (i8, ptr @s_7934265953746294420, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_12395442134831387347, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_9172927397968420710, i64 8)]
@a_2_2540699524713220113 = internal constant [2 x ptr] [ptr getelementptr inbounds nuw (i8, ptr @s_14524023967120200944, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_811268167711933856, i64 8)]

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc_atomic(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare i64 @strtoll(ptr, ptr, i32)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

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

declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare i64 @strlen(ptr)

declare i32 @strcmp(ptr, ptr)

declare void @puts(ptr)

; Function Attrs: mustprogress
define void @.f_main..feL63C6FH1583102() #2 {
  %1 = alloca ptr, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_2125742250238303305, i64 8), ptr %1, align 8
  %2 = load ptr, ptr %1, align 8
  %3 = icmp eq ptr %2, null
  %4 = select i1 %3, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %2
  call void @puts(ptr %4)
  ret void
}

; Function Attrs: mustprogress
define void @.f_main.foo() #2 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_16101355973854746, i64 8))
  ret void
}

; Function Attrs: mustprogress
define void @.f_main..feL77C15FH1583102() #2 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_11131119937623569912, i64 8))
  ret void
}

; Function Attrs: mustprogress
define ptr @.f_main.carTypes(ptr %0) #2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = ptrtoint ptr %4 to i64
  %6 = icmp ne i64 %5, 0
  %7 = zext i1 %6 to i32
  %8 = and i32 %7, 1
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %10, label %13

10:                                               ; preds = %1
  %11 = call i32 @strcmp(ptr %4, ptr getelementptr inbounds nuw (i8, ptr @s_4834688834925861241, i64 8))
  %12 = icmp eq i32 %11, 0
  br label %15

13:                                               ; preds = %1
  %14 = icmp eq i64 %5, ptrtoint (ptr getelementptr inbounds nuw (i8, ptr @s_4834688834925861241, i64 8) to i64)
  br label %15

15:                                               ; preds = %10, %13
  %16 = phi i1 [ %14, %13 ], [ %12, %10 ]
  br label %17

17:                                               ; preds = %15
  br i1 %16, label %18, label %20

18:                                               ; preds = %17
  %19 = load ptr, ptr %2, align 8
  store ptr %19, ptr %3, align 8
  br label %46

20:                                               ; preds = %17
  %21 = load ptr, ptr %2, align 8
  %22 = icmp eq ptr %21, null
  %23 = select i1 %22, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %21
  %24 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_422219368153924640, i64 8))
  %25 = add i64 1, %24
  %26 = call i64 @strlen(ptr %23)
  %27 = add i64 %25, %26
  %28 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638124528392833969, i64 8))
  %29 = add i64 %27, %28
  %30 = add i64 %29, 8
  %31 = call ptr @GC_malloc(i64 %30)
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = getelementptr i8, ptr %32, i64 0
  %34 = sub i64 %29, 0
  %35 = call ptr @__memcpy_chk(ptr %33, ptr getelementptr inbounds nuw (i8, ptr @s_422219368153924640, i64 8), i64 %24, i64 %34)
  %36 = add i64 0, %24
  %37 = getelementptr i8, ptr %32, i64 %36
  %38 = sub i64 %29, %36
  %39 = call ptr @__memcpy_chk(ptr %37, ptr %23, i64 %26, i64 %38)
  %40 = add i64 %36, %26
  %41 = getelementptr i8, ptr %32, i64 %40
  %42 = sub i64 %29, %40
  %43 = call ptr @__memcpy_chk(ptr %41, ptr getelementptr inbounds nuw (i8, ptr @s_12638124528392833969, i64 8), i64 %28, i64 %42)
  %44 = add i64 %40, %28
  %45 = getelementptr i8, ptr %32, i64 %44
  store i8 0, ptr %45, align 1
  store ptr %32, ptr %3, align 8
  br label %46

46:                                               ; preds = %18, %20
  %47 = load ptr, ptr %3, align 8
  ret ptr %47
}

; Function Attrs: mustprogress
define ptr @.f_main..afL217C17FH1583102(ptr %0, ptr %1, double %2) #2 {
  %4 = alloca double, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store ptr %1, ptr %5, align 8
  store double %2, ptr %4, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = icmp eq ptr %8, null
  %10 = select i1 %9, ptr @__ts_empty_array, ptr %8
  %11 = getelementptr { ptr, i64, i64 }, ptr %10, i32 0, i32 0
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr ptr, ptr %12, i32 0
  %14 = load ptr, ptr %13, align 8
  %15 = load ptr, ptr %5, align 8
  %16 = load ptr, ptr %6, align 8
  %17 = icmp eq ptr %16, null
  %18 = select i1 %17, ptr @__ts_empty_array, ptr %16
  %19 = getelementptr { ptr, i64, i64 }, ptr %18, i32 0, i32 0
  %20 = load ptr, ptr %19, align 8
  %21 = getelementptr ptr, ptr %20, i32 1
  %22 = load ptr, ptr %21, align 8
  %23 = load double, ptr %4, align 8
  %24 = call ptr @GC_malloc_atomic(i64 40)
  %25 = getelementptr i8, ptr %24, i64 8
  call void @__ts_number_to_string(ptr %25, double %23)
  %26 = load ptr, ptr %6, align 8
  %27 = icmp eq ptr %26, null
  %28 = select i1 %27, ptr @__ts_empty_array, ptr %26
  %29 = getelementptr { ptr, i64, i64 }, ptr %28, i32 0, i32 0
  %30 = load ptr, ptr %29, align 8
  %31 = getelementptr ptr, ptr %30, i32 2
  %32 = load ptr, ptr %31, align 8
  %33 = icmp eq ptr %14, null
  %34 = select i1 %33, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %14
  %35 = icmp eq ptr %15, null
  %36 = select i1 %35, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %15
  %37 = icmp eq ptr %22, null
  %38 = select i1 %37, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %22
  %39 = icmp eq ptr %25, null
  %40 = select i1 %39, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %25
  %41 = icmp eq ptr %32, null
  %42 = select i1 %41, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %32
  %43 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %44 = add i64 1, %43
  %45 = call i64 @strlen(ptr %34)
  %46 = add i64 %44, %45
  %47 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %48 = add i64 %46, %47
  %49 = call i64 @strlen(ptr %36)
  %50 = add i64 %48, %49
  %51 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %52 = add i64 %50, %51
  %53 = call i64 @strlen(ptr %38)
  %54 = add i64 %52, %53
  %55 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %56 = add i64 %54, %55
  %57 = call i64 @strlen(ptr %40)
  %58 = add i64 %56, %57
  %59 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %60 = add i64 %58, %59
  %61 = call i64 @strlen(ptr %42)
  %62 = add i64 %60, %61
  %63 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8))
  %64 = add i64 %62, %63
  %65 = add i64 %64, 8
  %66 = call ptr @GC_malloc(i64 %65)
  %67 = getelementptr i8, ptr %66, i64 8
  %68 = getelementptr i8, ptr %67, i64 0
  %69 = sub i64 %64, 0
  %70 = call ptr @__memcpy_chk(ptr %68, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %43, i64 %69)
  %71 = add i64 0, %43
  %72 = getelementptr i8, ptr %67, i64 %71
  %73 = sub i64 %64, %71
  %74 = call ptr @__memcpy_chk(ptr %72, ptr %34, i64 %45, i64 %73)
  %75 = add i64 %71, %45
  %76 = getelementptr i8, ptr %67, i64 %75
  %77 = sub i64 %64, %75
  %78 = call ptr @__memcpy_chk(ptr %76, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %47, i64 %77)
  %79 = add i64 %75, %47
  %80 = getelementptr i8, ptr %67, i64 %79
  %81 = sub i64 %64, %79
  %82 = call ptr @__memcpy_chk(ptr %80, ptr %36, i64 %49, i64 %81)
  %83 = add i64 %79, %49
  %84 = getelementptr i8, ptr %67, i64 %83
  %85 = sub i64 %64, %83
  %86 = call ptr @__memcpy_chk(ptr %84, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %51, i64 %85)
  %87 = add i64 %83, %51
  %88 = getelementptr i8, ptr %67, i64 %87
  %89 = sub i64 %64, %87
  %90 = call ptr @__memcpy_chk(ptr %88, ptr %38, i64 %53, i64 %89)
  %91 = add i64 %87, %53
  %92 = getelementptr i8, ptr %67, i64 %91
  %93 = sub i64 %64, %91
  %94 = call ptr @__memcpy_chk(ptr %92, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %55, i64 %93)
  %95 = add i64 %91, %55
  %96 = getelementptr i8, ptr %67, i64 %95
  %97 = sub i64 %64, %95
  %98 = call ptr @__memcpy_chk(ptr %96, ptr %40, i64 %57, i64 %97)
  %99 = add i64 %95, %57
  %100 = getelementptr i8, ptr %67, i64 %99
  %101 = sub i64 %64, %99
  %102 = call ptr @__memcpy_chk(ptr %100, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %59, i64 %101)
  %103 = add i64 %99, %59
  %104 = getelementptr i8, ptr %67, i64 %103
  %105 = sub i64 %64, %103
  %106 = call ptr @__memcpy_chk(ptr %104, ptr %42, i64 %61, i64 %105)
  %107 = add i64 %103, %61
  %108 = getelementptr i8, ptr %67, i64 %107
  %109 = sub i64 %64, %107
  %110 = call ptr @__memcpy_chk(ptr %108, ptr getelementptr inbounds nuw (i8, ptr @s_14695981039346656037, i64 8), i64 %63, i64 %109)
  %111 = add i64 %107, %63
  %112 = getelementptr i8, ptr %67, i64 %111
  store i8 0, ptr %112, align 1
  store ptr %67, ptr %7, align 8
  %113 = load ptr, ptr %7, align 8
  ret ptr %113
}

; Function Attrs: mustprogress
define i32 @main() #2 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  %8 = alloca { ptr, ptr, ptr }, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca { ptr, ptr }, align 8
  %12 = alloca { ptr, ptr }, align 8
  %13 = alloca { { ptr, ptr }, ptr }, align 8
  %14 = alloca { { ptr, ptr }, ptr }, align 8
  %15 = alloca { ptr, ptr, ptr }, align 8
  %16 = alloca { ptr, ptr, ptr }, align 8
  %17 = alloca ptr, align 8
  %18 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %19 = alloca <{ i64, [4 x { ptr, i1 }] }>, align 16
  %20 = alloca ptr, align 8
  %21 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %22 = alloca <{ i64, [4 x { ptr, i1 }] }>, align 16
  %23 = alloca ptr, align 8
  %24 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %25 = alloca <{ i64, [3 x { ptr, i1 }] }>, align 16
  %26 = alloca ptr, align 8
  %27 = alloca <{ i64, { ptr, i64, i64 } }>, align 16
  %28 = alloca <{ i64, [3 x { ptr, i1 }] }>, align 16
  %29 = alloca ptr, align 8
  %30 = alloca ptr, align 8
  %31 = alloca { ptr }, align 8
  %32 = alloca { ptr }, align 8
  %33 = alloca { ptr, ptr, ptr }, align 8
  %34 = alloca ptr, align 8
  %35 = alloca i32, align 4
  %36 = alloca ptr, align 8
  %37 = alloca ptr, align 8
  %38 = alloca ptr, align 8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_1564491103094674154, i64 8))
  store ptr getelementptr inbounds nuw (i8, ptr @s_9625390261332436968, i64 8), ptr %38, align 8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_15857971242039570103, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_16176116648655433384, i64 8))
  br i1 false, label %40, label %39

39:                                               ; preds = %0
  call void @doThat()
  br label %40

40:                                               ; preds = %39, %0
  %41 = call ptr @GC_malloc(i64 12)
  %42 = getelementptr i8, ptr %41, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %42, ptr @a_1_5762750502333681466, i64 4, i1 false)
  %43 = call ptr @GC_malloc(i64 32)
  %44 = getelementptr i8, ptr %43, i64 8
  %45 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 0
  store ptr %42, ptr %45, align 8
  %46 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 1
  store i64 1, ptr %46, align 8
  %47 = getelementptr { ptr, i64, i64 }, ptr %44, i32 0, i32 2
  store i64 1, ptr %47, align 8
  store ptr %44, ptr %37, align 8
  %48 = load ptr, ptr %37, align 8
  %49 = icmp eq ptr %48, null
  %50 = select i1 %49, ptr @__ts_empty_array, ptr %48
  %51 = getelementptr { ptr, i64, i64 }, ptr %50, i32 0, i32 0
  %52 = load ptr, ptr %51, align 8
  %53 = getelementptr i32, ptr %52, i32 0
  %54 = load i32, ptr %53, align 4
  %55 = icmp ne i32 %54, 0
  %56 = xor i1 %55, true
  br i1 %56, label %57, label %58

57:                                               ; preds = %40
  call void @myFunction()
  br label %58

58:                                               ; preds = %57, %40
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_15442652226749703132, i64 8))
  store ptr null, ptr %36, align 8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_17543757828066490973, i64 8))
  store i32 0, ptr %35, align 4
  br i1 true, label %59, label %60

59:                                               ; preds = %58
  store i32 5, ptr %35, align 4
  br label %60

60:                                               ; preds = %59, %58
  %61 = load i32, ptr %35, align 4
  %62 = call ptr @GC_malloc_atomic(i64 58)
  %63 = getelementptr i8, ptr %62, i64 8
  %64 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %63, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 %61)
  %65 = icmp eq ptr %63, null
  %66 = select i1 %65, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %63
  call void @puts(ptr %66)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_16011791931586107870, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @__false__, i64 8))
  store i32 3, ptr %35, align 4
  store ptr getelementptr inbounds nuw (i8, ptr @s_452236695178438998, i64 8), ptr %34, align 8
  call void @.f_main..feL63C6FH1583102()
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_8570510144365560058, i64 8))
  call void @.f_main.foo()
  store { ptr, ptr, ptr } { ptr @.f_main..feL77C15FH1583102, ptr null, ptr null }, ptr %33, align 8
  %67 = load { ptr, ptr, ptr }, ptr %33, align 8
  %68 = extractvalue { ptr, ptr, ptr } %67, 1
  %69 = ptrtoint ptr %68 to i64
  %70 = icmp ne i64 %69, 0
  br i1 %70, label %71, label %73

71:                                               ; preds = %60
  %72 = extractvalue { ptr, ptr, ptr } %67, 0
  call void %72(ptr %68)
  br label %75

73:                                               ; preds = %60
  %74 = extractvalue { ptr, ptr, ptr } %67, 0
  call void %74()
  br label %75

75:                                               ; preds = %71, %73
  br label %76

76:                                               ; preds = %75
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3974561738185464706, i64 8))
  %77 = getelementptr { ptr }, ptr %32, i32 0, i32 0
  store ptr getelementptr inbounds nuw (i8, ptr @s_8999596768310594794, i64 8), ptr %77, align 8
  %78 = load { ptr }, ptr %32, align 8
  store { ptr } %78, ptr %31, align 8
  %79 = getelementptr { ptr }, ptr %31, i32 0, i32 0
  store ptr getelementptr inbounds nuw (i8, ptr @s_5766643768467844826, i64 8), ptr %79, align 8
  %80 = call ptr @GC_malloc(i64 24)
  %81 = getelementptr i8, ptr %80, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %81, ptr @a_2_2540699524713220113, i64 16, i1 false)
  %82 = call ptr @GC_malloc(i64 32)
  %83 = getelementptr i8, ptr %82, i64 8
  %84 = getelementptr { ptr, i64, i64 }, ptr %83, i32 0, i32 0
  store ptr %81, ptr %84, align 8
  %85 = getelementptr { ptr, i64, i64 }, ptr %83, i32 0, i32 1
  store i64 2, ptr %85, align 8
  %86 = getelementptr { ptr, i64, i64 }, ptr %83, i32 0, i32 2
  store i64 2, ptr %86, align 8
  store ptr %83, ptr %30, align 8
  %87 = load ptr, ptr %30, align 8
  %88 = icmp eq ptr %87, null
  %89 = select i1 %88, ptr @__ts_empty_array, ptr %87
  %90 = getelementptr { ptr, i64, i64 }, ptr %89, i32 0, i32 0
  %91 = load ptr, ptr %90, align 8
  %92 = getelementptr ptr, ptr %91, i32 0
  %93 = load ptr, ptr %92, align 8
  %94 = load ptr, ptr %30, align 8
  %95 = icmp eq ptr %94, null
  %96 = select i1 %95, ptr @__ts_empty_array, ptr %94
  %97 = getelementptr { ptr, i64, i64 }, ptr %96, i32 0, i32 0
  %98 = load ptr, ptr %97, align 8
  %99 = getelementptr ptr, ptr %98, i32 1
  %100 = load ptr, ptr %99, align 8
  %101 = call ptr @llvm.stacksave.p0()
  %102 = icmp eq ptr %93, null
  %103 = select i1 %102, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %93
  %104 = icmp eq ptr %100, null
  %105 = select i1 %104, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %100
  %106 = call i64 @strlen(ptr %103)
  %107 = add i64 1, %106
  %108 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8))
  %109 = add i64 %107, %108
  %110 = call i64 @strlen(ptr %105)
  %111 = add i64 %109, %110
  %112 = alloca i8, i64 %111, align 1
  %113 = getelementptr i8, ptr %112, i64 0
  %114 = sub i64 %111, 0
  %115 = call ptr @__memcpy_chk(ptr %113, ptr %103, i64 %106, i64 %114)
  %116 = add i64 0, %106
  %117 = getelementptr i8, ptr %112, i64 %116
  %118 = sub i64 %111, %116
  %119 = call ptr @__memcpy_chk(ptr %117, ptr getelementptr inbounds nuw (i8, ptr @s_12638117931323064703, i64 8), i64 %108, i64 %118)
  %120 = add i64 %116, %108
  %121 = getelementptr i8, ptr %112, i64 %120
  %122 = sub i64 %111, %120
  %123 = call ptr @__memcpy_chk(ptr %121, ptr %105, i64 %110, i64 %122)
  %124 = add i64 %120, %110
  %125 = getelementptr i8, ptr %112, i64 %124
  store i8 0, ptr %125, align 1
  call void @puts(ptr %112)
  call void @llvm.stackrestore.p0(ptr %101)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_10634310313915411333, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_9276932448023608736, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_13141189159377525121, i64 8))
  %126 = call i64 @strtoll(ptr getelementptr inbounds nuw (i8, ptr @s_5001449799818188079, i64 8), ptr null, i32 2)
  %127 = call i64 @llvm.smin.i64(i64 %126, i64 2147483647)
  %128 = call i64 @llvm.smax.i64(i64 %127, i64 -2147483648)
  %129 = trunc i64 %128 to i32
  %130 = icmp eq i32 %129, 5
  br i1 %130, label %131, label %295

131:                                              ; preds = %76
  br i1 true, label %132, label %297

132:                                              ; preds = %131
  br i1 true, label %133, label %299

133:                                              ; preds = %132
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_2996941089346072501, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_11276182295403084134, i64 8))
  %134 = call ptr @GC_malloc(i64 32)
  %135 = getelementptr i8, ptr %134, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %135, ptr @a_3_6335243048188567126, i64 24, i1 false)
  %136 = call ptr @GC_malloc(i64 32)
  %137 = getelementptr i8, ptr %136, i64 8
  %138 = getelementptr { ptr, i64, i64 }, ptr %137, i32 0, i32 0
  store ptr %135, ptr %138, align 8
  %139 = getelementptr { ptr, i64, i64 }, ptr %137, i32 0, i32 1
  store i64 3, ptr %139, align 8
  %140 = getelementptr { ptr, i64, i64 }, ptr %137, i32 0, i32 2
  store i64 3, ptr %140, align 8
  store ptr %137, ptr %29, align 8
  store i64 -1, ptr %28, align 8
  %141 = getelementptr i8, ptr %28, i64 8
  store [3 x { ptr, i1 }] zeroinitializer, ptr %141, align 8
  %142 = getelementptr { ptr, i1 }, ptr %141, i64 0
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_4394248473669529567, i64 8), i1 true }, ptr %142, align 8
  %143 = getelementptr { ptr, i1 }, ptr %141, i64 1
  store { ptr, i1 } zeroinitializer, ptr %143, align 8
  %144 = getelementptr { ptr, i1 }, ptr %141, i64 2
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_6599060811930723210, i64 8), i1 true }, ptr %144, align 8
  store i64 -1, ptr %27, align 8
  %145 = getelementptr i8, ptr %27, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %145, align 8
  %146 = getelementptr { ptr, i64, i64 }, ptr %145, i32 0, i32 0
  store ptr %141, ptr %146, align 8
  %147 = getelementptr { ptr, i64, i64 }, ptr %145, i32 0, i32 1
  store i64 3, ptr %147, align 8
  %148 = getelementptr { ptr, i64, i64 }, ptr %145, i32 0, i32 2
  store i64 3, ptr %148, align 8
  store ptr %145, ptr %26, align 8
  store i64 -1, ptr %25, align 8
  %149 = getelementptr i8, ptr %25, i64 8
  store [3 x { ptr, i1 }] zeroinitializer, ptr %149, align 8
  %150 = getelementptr { ptr, i1 }, ptr %149, i64 0
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_4624382957487889774, i64 8), i1 true }, ptr %150, align 8
  %151 = getelementptr { ptr, i1 }, ptr %149, i64 1
  store { ptr, i1 } zeroinitializer, ptr %151, align 8
  %152 = getelementptr { ptr, i1 }, ptr %149, i64 2
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_14717692112504537385, i64 8), i1 true }, ptr %152, align 8
  store i64 -1, ptr %24, align 8
  %153 = getelementptr i8, ptr %24, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %153, align 8
  %154 = getelementptr { ptr, i64, i64 }, ptr %153, i32 0, i32 0
  store ptr %149, ptr %154, align 8
  %155 = getelementptr { ptr, i64, i64 }, ptr %153, i32 0, i32 1
  store i64 3, ptr %155, align 8
  %156 = getelementptr { ptr, i64, i64 }, ptr %153, i32 0, i32 2
  store i64 3, ptr %156, align 8
  store ptr %153, ptr %23, align 8
  store i64 -1, ptr %22, align 8
  %157 = getelementptr i8, ptr %22, i64 8
  store [4 x { ptr, i1 }] zeroinitializer, ptr %157, align 8
  %158 = getelementptr { ptr, i1 }, ptr %157, i64 0
  store { ptr, i1 } zeroinitializer, ptr %158, align 8
  %159 = getelementptr { ptr, i1 }, ptr %157, i64 1
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_4624382957487889774, i64 8), i1 true }, ptr %159, align 8
  %160 = getelementptr { ptr, i1 }, ptr %157, i64 2
  store { ptr, i1 } zeroinitializer, ptr %160, align 8
  %161 = getelementptr { ptr, i1 }, ptr %157, i64 3
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_14717692112504537385, i64 8), i1 true }, ptr %161, align 8
  store i64 -1, ptr %21, align 8
  %162 = getelementptr i8, ptr %21, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %162, align 8
  %163 = getelementptr { ptr, i64, i64 }, ptr %162, i32 0, i32 0
  store ptr %157, ptr %163, align 8
  %164 = getelementptr { ptr, i64, i64 }, ptr %162, i32 0, i32 1
  store i64 4, ptr %164, align 8
  %165 = getelementptr { ptr, i64, i64 }, ptr %162, i32 0, i32 2
  store i64 4, ptr %165, align 8
  store ptr %162, ptr %20, align 8
  store i64 -1, ptr %19, align 8
  %166 = getelementptr i8, ptr %19, i64 8
  store [4 x { ptr, i1 }] zeroinitializer, ptr %166, align 8
  %167 = getelementptr { ptr, i1 }, ptr %166, i64 0
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_4624382957487889774, i64 8), i1 true }, ptr %167, align 8
  %168 = getelementptr { ptr, i1 }, ptr %166, i64 1
  store { ptr, i1 } zeroinitializer, ptr %168, align 8
  %169 = getelementptr { ptr, i1 }, ptr %166, i64 2
  store { ptr, i1 } { ptr getelementptr inbounds nuw (i8, ptr @s_14717692112504537385, i64 8), i1 true }, ptr %169, align 8
  %170 = getelementptr { ptr, i1 }, ptr %166, i64 3
  store { ptr, i1 } zeroinitializer, ptr %170, align 8
  store i64 -1, ptr %18, align 8
  %171 = getelementptr i8, ptr %18, i64 8
  store { ptr, i64, i64 } zeroinitializer, ptr %171, align 8
  %172 = getelementptr { ptr, i64, i64 }, ptr %171, i32 0, i32 0
  store ptr %166, ptr %172, align 8
  %173 = getelementptr { ptr, i64, i64 }, ptr %171, i32 0, i32 1
  store i64 4, ptr %173, align 8
  %174 = getelementptr { ptr, i64, i64 }, ptr %171, i32 0, i32 2
  store i64 4, ptr %174, align 8
  store ptr %171, ptr %17, align 8
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_13749028156498010861, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_2658421951147754508, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_14669464332616883376, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_9255881455004091586, i64 8))
  %175 = call ptr @.f_main.carTypes(ptr getelementptr inbounds nuw (i8, ptr @s_4834688834925861241, i64 8))
  store { ptr, ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @s_2738302745049774710, i64 8), ptr null, ptr getelementptr inbounds nuw (i8, ptr @s_17292252969566779935, i64 8) }, ptr %16, align 8
  %176 = getelementptr { ptr, ptr, ptr }, ptr %16, i32 0, i32 1
  store ptr %175, ptr %176, align 8
  %177 = load { ptr, ptr, ptr }, ptr %16, align 8
  store { ptr, ptr, ptr } %177, ptr %15, align 8
  %178 = getelementptr { ptr, ptr, ptr }, ptr %15, i32 0, i32 0
  %179 = load ptr, ptr %178, align 8
  %180 = icmp eq ptr %179, null
  %181 = select i1 %180, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %179
  call void @puts(ptr %181)
  %182 = getelementptr { ptr, ptr, ptr }, ptr %15, i32 0, i32 1
  %183 = load ptr, ptr %182, align 8
  %184 = icmp eq ptr %183, null
  %185 = select i1 %184, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %183
  call void @puts(ptr %185)
  %186 = getelementptr { ptr, ptr, ptr }, ptr %15, i32 0, i32 2
  %187 = load ptr, ptr %186, align 8
  %188 = icmp eq ptr %187, null
  %189 = select i1 %188, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %187
  call void @puts(ptr %189)
  %190 = getelementptr { { ptr, ptr }, ptr }, ptr %14, i32 0, i32 0
  store { ptr, ptr } { ptr getelementptr inbounds nuw (i8, ptr @s_3753534942992164126, i64 8), ptr getelementptr inbounds nuw (i8, ptr @s_11431534339755074821, i64 8) }, ptr %190, align 8
  %191 = getelementptr { { ptr, ptr }, ptr }, ptr %14, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_617707879461881500, i64 8), ptr %191, align 8
  %192 = load { { ptr, ptr }, ptr }, ptr %14, align 8
  store { { ptr, ptr }, ptr } %192, ptr %13, align 8
  %193 = getelementptr { { ptr, ptr }, ptr }, ptr %13, i32 0, i32 0
  %194 = getelementptr { ptr, ptr }, ptr %193, i32 0, i32 1
  %195 = load ptr, ptr %194, align 8
  %196 = icmp eq ptr %195, null
  %197 = select i1 %196, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %195
  call void @puts(ptr %197)
  %198 = getelementptr { { ptr, ptr }, ptr }, ptr %13, i32 0, i32 1
  %199 = load ptr, ptr %198, align 8
  %200 = icmp eq ptr %199, null
  %201 = select i1 %200, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %199
  call void @puts(ptr %201)
  %202 = getelementptr { ptr, ptr }, ptr %12, i32 0, i32 0
  store ptr getelementptr inbounds nuw (i8, ptr @s_15379658938586843080, i64 8), ptr %202, align 8
  %203 = getelementptr { ptr, ptr }, ptr %12, i32 0, i32 1
  store ptr getelementptr inbounds nuw (i8, ptr @s_17317583992087840406, i64 8), ptr %203, align 8
  %204 = load { ptr, ptr }, ptr %12, align 8
  store { ptr, ptr } %204, ptr %11, align 8
  %205 = getelementptr { ptr, ptr }, ptr %11, i32 0, i32 0
  %206 = load ptr, ptr %205, align 8
  %207 = icmp eq ptr %206, null
  %208 = select i1 %207, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %206
  call void @puts(ptr %208)
  %209 = getelementptr { ptr, ptr }, ptr %11, i32 0, i32 1
  %210 = load ptr, ptr %209, align 8
  %211 = icmp eq ptr %210, null
  %212 = select i1 %211, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %210
  call void @puts(ptr %212)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_305597083721566844, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_2694515762537206396, i64 8))
  %213 = call ptr @GC_malloc_atomic(i64 58)
  %214 = getelementptr i8, ptr %213, i64 8
  %215 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr %214, i64 50, ptr getelementptr inbounds nuw (i8, ptr @frmt_11053768727887132666, i64 8), i32 10)
  %216 = icmp eq ptr %214, null
  %217 = select i1 %216, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %214
  call void @puts(ptr %217)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3617302750436170686, i64 8))
  store ptr getelementptr inbounds nuw (i8, ptr @s_1609583978338260660, i64 8), ptr %10, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_11822328890338333214, i64 8), ptr %9, align 8
  %218 = load ptr, ptr %10, align 8
  %219 = load ptr, ptr %9, align 8
  %220 = icmp eq ptr %218, null
  %221 = select i1 %220, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %218
  %222 = icmp eq ptr %219, null
  %223 = select i1 %222, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %219
  %224 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_10530820969299836017, i64 8))
  %225 = add i64 1, %224
  %226 = call i64 @strlen(ptr %221)
  %227 = add i64 %225, %226
  %228 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_5920926082801741360, i64 8))
  %229 = add i64 %227, %228
  %230 = call i64 @strlen(ptr %223)
  %231 = add i64 %229, %230
  %232 = call i64 @strlen(ptr getelementptr inbounds nuw (i8, ptr @s_12638141021067257134, i64 8))
  %233 = add i64 %231, %232
  %234 = add i64 %233, 8
  %235 = call ptr @GC_malloc(i64 %234)
  %236 = getelementptr i8, ptr %235, i64 8
  %237 = getelementptr i8, ptr %236, i64 0
  %238 = sub i64 %233, 0
  %239 = call ptr @__memcpy_chk(ptr %237, ptr getelementptr inbounds nuw (i8, ptr @s_10530820969299836017, i64 8), i64 %224, i64 %238)
  %240 = add i64 0, %224
  %241 = getelementptr i8, ptr %236, i64 %240
  %242 = sub i64 %233, %240
  %243 = call ptr @__memcpy_chk(ptr %241, ptr %221, i64 %226, i64 %242)
  %244 = add i64 %240, %226
  %245 = getelementptr i8, ptr %236, i64 %244
  %246 = sub i64 %233, %244
  %247 = call ptr @__memcpy_chk(ptr %245, ptr getelementptr inbounds nuw (i8, ptr @s_5920926082801741360, i64 8), i64 %228, i64 %246)
  %248 = add i64 %244, %228
  %249 = getelementptr i8, ptr %236, i64 %248
  %250 = sub i64 %233, %248
  %251 = call ptr @__memcpy_chk(ptr %249, ptr %223, i64 %230, i64 %250)
  %252 = add i64 %248, %230
  %253 = getelementptr i8, ptr %236, i64 %252
  %254 = sub i64 %233, %252
  %255 = call ptr @__memcpy_chk(ptr %253, ptr getelementptr inbounds nuw (i8, ptr @s_12638141021067257134, i64 8), i64 %232, i64 %254)
  %256 = add i64 %252, %232
  %257 = getelementptr i8, ptr %236, i64 %256
  store i8 0, ptr %257, align 1
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_17664890869449432037, i64 8))
  store { ptr, ptr, ptr } { ptr @.f_main..afL217C17FH1583102, ptr null, ptr null }, ptr %8, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_345403531931479259, i64 8), ptr %7, align 8
  store i32 28, ptr %6, align 4
  %258 = load ptr, ptr %7, align 8
  %259 = load i32, ptr %6, align 4
  %260 = call ptr @GC_malloc(i64 32)
  %261 = getelementptr i8, ptr %260, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %261, ptr @a_3_2084061994484565833, i64 24, i1 false)
  %262 = call ptr @GC_malloc(i64 32)
  %263 = getelementptr i8, ptr %262, i64 8
  %264 = getelementptr { ptr, i64, i64 }, ptr %263, i32 0, i32 0
  store ptr %261, ptr %264, align 8
  %265 = getelementptr { ptr, i64, i64 }, ptr %263, i32 0, i32 1
  store i64 3, ptr %265, align 8
  %266 = getelementptr { ptr, i64, i64 }, ptr %263, i32 0, i32 2
  store i64 3, ptr %266, align 8
  %267 = load { ptr, ptr, ptr }, ptr %8, align 8
  %268 = sitofp i32 %259 to double
  %269 = extractvalue { ptr, ptr, ptr } %267, 1
  %270 = ptrtoint ptr %269 to i64
  %271 = icmp ne i64 %270, 0
  br i1 %271, label %272, label %275

272:                                              ; preds = %133
  %273 = extractvalue { ptr, ptr, ptr } %267, 0
  %274 = call ptr %273(ptr %269, ptr %263, ptr %258, double %268)
  br label %278

275:                                              ; preds = %133
  %276 = extractvalue { ptr, ptr, ptr } %267, 0
  %277 = call ptr %276(ptr %263, ptr %258, double %268)
  br label %278

278:                                              ; preds = %272, %275
  %279 = phi ptr [ %277, %275 ], [ %274, %272 ]
  br label %280

280:                                              ; preds = %278
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6321389093131908277, i64 8))
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_3432704340853296579, i64 8))
  store ptr getelementptr inbounds nuw (i8, ptr @s_10966898795647604191, i64 8), ptr %5, align 8
  %281 = load ptr, ptr %5, align 8
  %282 = icmp eq ptr %281, null
  %283 = select i1 %282, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %281
  call void @puts(ptr %283)
  store ptr getelementptr inbounds nuw (i8, ptr @s_5594953940062847258, i64 8), ptr %4, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_5476372761493277963, i64 8), ptr %3, align 8
  %284 = load ptr, ptr %3, align 8
  %285 = icmp eq ptr %284, null
  %286 = select i1 %285, ptr getelementptr inbounds nuw (i8, ptr @s_6601085983368743140, i64 8), ptr %284
  call void @puts(ptr %286)
  store ptr getelementptr inbounds nuw (i8, ptr @s_2851890610989147717, i64 8), ptr %2, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @s_2851890610989147717, i64 8), ptr %1, align 8
  %287 = call ptr @GC_malloc(i64 16)
  %288 = getelementptr i8, ptr %287, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %288, ptr @a_1_16629603895491369024, i64 8, i1 false)
  %289 = call ptr @GC_malloc(i64 32)
  %290 = getelementptr i8, ptr %289, i64 8
  %291 = getelementptr { ptr, i64, i64 }, ptr %290, i32 0, i32 0
  store ptr %288, ptr %291, align 8
  %292 = getelementptr { ptr, i64, i64 }, ptr %290, i32 0, i32 1
  store i64 1, ptr %292, align 8
  %293 = getelementptr { ptr, i64, i64 }, ptr %290, i32 0, i32 2
  store i64 1, ptr %293, align 8
  call void @f(ptr %290)
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

294:                                              ; preds = %299, %297, %295
  unreachable

295:                                              ; preds = %76
  %296 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2330421925823702781, i64 8), i32 114, ptr null)
  br label %294

297:                                              ; preds = %131
  %298 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2330421925823702781, i64 8), i32 116, ptr null)
  br label %294

299:                                              ; preds = %132
  %300 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_17914885825359534683, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_2330421925823702781, i64 8), i32 117, ptr null)
  br label %294
}

; Function Attrs: mustprogress
define void @f(ptr %0) #2 {
  ret void
}

; Function Attrs: mustprogress
define void @doThis() #2 {
  ret void
}

; Function Attrs: mustprogress
define void @doThat() #2 {
  ret void
}

; Function Attrs: mustprogress
define void @myFunction() #2 {
  ret void
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
declare double @llvm.fabs.f64(double) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #4

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #3

attributes #0 = { allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

