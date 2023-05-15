; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%Sphere = type { ptr, double, ptr, { ptr, ptr } }
%Plane = type { ptr, ptr, double, { ptr, ptr } }
%Vector = type { ptr, double, double, double }
%Color = type { ptr, double, double, double }
%Camera = type { ptr, ptr, ptr, ptr, ptr }
%RayTracer = type { ptr, i32 }

@s_5322524335133436399 = internal constant [5 x i8] c"Math\00"
@s_8888464120224770146 = internal constant [7 x i8] c"Vector\00"
@s_3769135706557701272 = internal constant [6 x i8] c"Color\00"
@s_6111862596534002478 = internal constant [7 x i8] c"Camera\00"
@s_7971022780007706968 = internal constant [7 x i8] c"Sphere\00"
@s_1414967765654552757 = internal constant [6 x i8] c"Plane\00"
@s_15629559017655921514 = internal constant [10 x i8] c"RayTracer\00"
@s_4155466971959723698 = internal unnamed_addr constant [7 x i8] c");\22 />\00"
@s_8482466337658167522 = internal unnamed_addr constant [40 x i8] c"\22 width=\221\22 height=\221\22 style=\22fill:rgb(\00"
@s_11894988290003604653 = internal unnamed_addr constant [6 x i8] c"\22 y=\22\00"
@s_1937339058070584390 = internal unnamed_addr constant [10 x i8] c"<rect x=\22\00"
@frmt_555400739678143724 = internal constant [3 x i8] c"%d\00"
@frmt_555404038213028357 = internal constant [3 x i8] c"%g\00"
@s_6682479467004374669 = internal constant [6 x i8] c"done.\00"
@s_17165743417360025467 = internal constant [9 x i8] c"start...\00"
@Infinity = local_unnamed_addr constant double 0x7FF0000000000000
@Math..rtti = global ptr @s_5322524335133436399
@Math..type_descr = local_unnamed_addr global i64 0
@Math..vtbl = global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Math..instanceOf, ptr @Math..new, ptr @Math.sqrt, ptr @Math.floor, ptr @Math.pow, ptr @Math..rtti }
@Vector..rtti = global ptr @s_8888464120224770146
@Vector..type_descr = local_unnamed_addr global i64 0
@Vector..vtbl = global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Vector..instanceOf, ptr @Vector..new, ptr @Vector.times, ptr @Vector.minus, ptr @Vector.plus, ptr @Vector.dot, ptr @Vector.mag, ptr @Vector.norm, ptr @Vector.cross, ptr @Vector..rtti }
@Color..rtti = global ptr @s_3769135706557701272
@Color..type_descr = local_unnamed_addr global i64 0
@Color.white = global ptr undef
@Color.grey = global ptr undef
@Color.black = global ptr undef
@Color.background = global ptr undef
@Color.defaultColor = global ptr undef
@Color..vtbl = global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Color..instanceOf, ptr @Color..new, ptr @Color.scale, ptr @Color.plus, ptr @Color.times, ptr @Color.toDrawingColor, ptr @Color..rtti, ptr @Color.white, ptr @Color.grey, ptr @Color.black, ptr @Color.background, ptr @Color.defaultColor, ptr @Color.white, ptr @Color.grey, ptr @Color.black, ptr @Color.background, ptr @Color.defaultColor }
@Camera..rtti = global ptr @s_6111862596534002478
@Camera..type_descr = local_unnamed_addr global i64 0
@Camera..vtbl = global { ptr, ptr, ptr } { ptr @Camera..instanceOf, ptr @Camera..new, ptr @Camera..rtti }
@Sphere..rtti = global ptr @s_7971022780007706968
@Sphere..type_descr = local_unnamed_addr global i64 0
@Intersection.14787487..vtbl = global { ptr, ptr, ptr, ptr, ptr } { ptr getelementptr ({ { ptr, ptr }, { ptr, ptr }, double }, ptr null, i32 0, i32 1), ptr getelementptr ({ { ptr, ptr }, { ptr, ptr }, double }, ptr null, i32 0, i32 2), ptr getelementptr ({ { ptr, ptr }, { ptr, ptr }, double }, ptr null, i32 0, i32 1), ptr getelementptr ({ { ptr, ptr }, { ptr, ptr }, double }, ptr null, i32 0, i32 2), ptr null }
@Sphere.Thing..vtbl = global { ptr, ptr, ptr } { ptr @Sphere.intersect, ptr @Sphere.normal, ptr getelementptr (%Sphere, ptr null, i32 0, i32 3) }
@Sphere..vtbl = global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Sphere.Thing..vtbl, ptr @Sphere..instanceOf, ptr @Sphere..new, ptr @Sphere.normal, ptr @Sphere.intersect, ptr @Sphere..rtti }
@Plane..rtti = global ptr @s_1414967765654552757
@Plane..type_descr = local_unnamed_addr global i64 0
@Plane.Thing..vtbl = global { ptr, ptr, ptr } { ptr @Plane.intersect, ptr @Plane.normal, ptr getelementptr (%Plane, ptr null, i32 0, i32 3) }
@Plane..vtbl = global { ptr, ptr, ptr, ptr, ptr, ptr } { ptr @Plane.Thing..vtbl, ptr @Plane..instanceOf, ptr @Plane..new, ptr @Plane.normal, ptr @Plane.intersect, ptr @Plane..rtti }
@Surfaces.shiny = local_unnamed_addr global { ptr, ptr } undef
@Surface.14796823..vtbl = global { ptr, ptr, ptr, ptr } { ptr getelementptr ({ double, ptr, ptr, ptr }, ptr null, i32 0, i32 1), ptr getelementptr ({ double, ptr, ptr, ptr }, ptr null, i32 0, i32 2), ptr getelementptr ({ double, ptr, ptr, ptr }, ptr null, i32 0, i32 3), ptr null }
@Surfaces.checkerboard = local_unnamed_addr global { ptr, ptr } undef
@Surface.14728117..vtbl = global { ptr, ptr, ptr, ptr } { ptr getelementptr ({ double, ptr, ptr, ptr }, ptr null, i32 0, i32 1), ptr getelementptr ({ double, ptr, ptr, ptr }, ptr null, i32 0, i32 2), ptr getelementptr ({ double, ptr, ptr, ptr }, ptr null, i32 0, i32 3), ptr null }
@RayTracer..rtti = global ptr @s_15629559017655921514
@RayTracer..type_descr = local_unnamed_addr global i64 0
@Ray.14722407..vtbl = global { ptr, ptr } { ptr null, ptr getelementptr ({ ptr, ptr }, ptr null, i32 0, i32 1) }
@RayTracer..vtbl = global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { ptr @RayTracer..instanceOf, ptr @RayTracer..new, ptr @RayTracer.intersections, ptr @RayTracer.testRay, ptr @RayTracer.traceRay, ptr @RayTracer.shade, ptr @RayTracer.getReflectionColor, ptr @RayTracer.getNaturalColor, ptr @RayTracer.render, ptr @RayTracer..rtti }
@Light.14713958..vtbl = global { ptr, ptr } { ptr null, ptr getelementptr ({ ptr, ptr }, ptr null, i32 0, i32 1) }
@Scene.14713952..vtbl = global { ptr, ptr, ptr } { ptr null, ptr getelementptr ({ { ptr, i32 }, { ptr, i32 }, ptr }, ptr null, i32 0, i32 1), ptr getelementptr ({ { ptr, i32 }, { ptr, i32 }, ptr }, ptr null, i32 0, i32 2) }
@llvm.global_ctors = appending constant [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__mlir_gctors, ptr null }]

declare void @GC_init() local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #0

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strcat(ptr noalias returned, ptr noalias nocapture readonly) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr nocapture) local_unnamed_addr #2

declare i32 @sprintf_s(ptr, i32, ptr, ...) local_unnamed_addr

declare ptr @GC_malloc_explicitly_typed(i64, i64) local_unnamed_addr

declare i64 @GC_make_descriptor(ptr, i64) local_unnamed_addr

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr nocapture, ptr nocapture) local_unnamed_addr #2

declare ptr @GC_malloc(i64) local_unnamed_addr

define void @Surfaces.shiny__cctor() local_unnamed_addr {
  %1 = tail call ptr @GC_malloc(i64 32)
  store double 2.500000e+02, ptr %1, align 8
  %.repack1 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 1
  store ptr @Surfaces..feL159C18L159C63H14774515, ptr %.repack1, align 8
  %.repack2 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 2
  store ptr @Surfaces..feL160C19L160C63H14867906, ptr %.repack2, align 8
  %.repack3 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 3
  store ptr @Surfaces..feL161C18L161C55H14867908, ptr %.repack3, align 8
  store ptr @Surface.14796823..vtbl, ptr @Surfaces.shiny, align 8
  store ptr %1, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.shiny, i64 0, i32 1), align 8
  ret void
}

define void @Surfaces.checkerboard__cctor() local_unnamed_addr {
  %1 = tail call ptr @GC_malloc(i64 32)
  store double 1.500000e+02, ptr %1, align 8
  %.repack1 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 1
  store ptr @Surfaces..feL165C18L171C9H15086251, ptr %.repack1, align 8
  %.repack2 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 2
  store ptr @Surfaces..feL172C19L172C64H14759908, ptr %.repack2, align 8
  %.repack3 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 3
  store ptr @Surfaces..feL173C18L179C9H14757289, ptr %.repack3, align 8
  store ptr @Surface.14728117..vtbl, ptr @Surfaces.checkerboard, align 8
  store ptr %1, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.checkerboard, i64 0, i32 1), align 8
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
declare double @sqrt(double) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
declare double @pow(double, double) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @Math..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @Math..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @Math..new() {
  %1 = load i64, ptr @Math..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 1)
  store i64 %4, ptr @Math..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 8, i64 %6)
  store ptr @Math..vtbl, ptr %7, align 8
  ret ptr %7
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
define double @Math.sqrt(double %0) #3 {
  %2 = tail call double @sqrt(double %0)
  ret double %2
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
define double @Math.floor(double %0) #5 {
  %2 = tail call double @llvm.floor.f64(double %0)
  ret double %2
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write)
define double @Math.pow(double %0, double %1) #3 {
  %3 = tail call double @pow(double %0, double %1)
  ret double %3
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @Vector..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @Vector..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @Vector..new() {
  %1 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 4)
  store i64 %4, ptr @Vector..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %6)
  store ptr @Vector..vtbl, ptr %7, align 8
  ret ptr %7
}

define ptr @Vector.times(double %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Vector..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Vector..vtbl, ptr %9, align 8
  %10 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = fmul double %11, %0
  %13 = getelementptr %Vector, ptr %1, i64 0, i32 2
  %14 = load double, ptr %13, align 8
  %15 = fmul double %14, %0
  %16 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %17 = load double, ptr %16, align 8
  %18 = fmul double %17, %0
  %19 = getelementptr %Vector, ptr %9, i64 0, i32 1
  store double %12, ptr %19, align 8
  %20 = getelementptr %Vector, ptr %9, i64 0, i32 2
  store double %15, ptr %20, align 8
  %21 = getelementptr %Vector, ptr %9, i64 0, i32 3
  store double %18, ptr %21, align 8
  ret ptr %9
}

define ptr @Vector.minus(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Vector..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Vector..vtbl, ptr %9, align 8
  %10 = getelementptr %Vector, ptr %0, i64 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = fsub double %11, %13
  %15 = getelementptr %Vector, ptr %0, i64 0, i32 2
  %16 = load double, ptr %15, align 8
  %17 = getelementptr %Vector, ptr %1, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = fsub double %16, %18
  %20 = getelementptr %Vector, ptr %0, i64 0, i32 3
  %21 = load double, ptr %20, align 8
  %22 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %23 = load double, ptr %22, align 8
  %24 = fsub double %21, %23
  %25 = getelementptr %Vector, ptr %9, i64 0, i32 1
  store double %14, ptr %25, align 8
  %26 = getelementptr %Vector, ptr %9, i64 0, i32 2
  store double %19, ptr %26, align 8
  %27 = getelementptr %Vector, ptr %9, i64 0, i32 3
  store double %24, ptr %27, align 8
  ret ptr %9
}

define ptr @Vector.plus(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Vector..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Vector..vtbl, ptr %9, align 8
  %10 = getelementptr %Vector, ptr %0, i64 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = fadd double %11, %13
  %15 = getelementptr %Vector, ptr %0, i64 0, i32 2
  %16 = load double, ptr %15, align 8
  %17 = getelementptr %Vector, ptr %1, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = fadd double %16, %18
  %20 = getelementptr %Vector, ptr %0, i64 0, i32 3
  %21 = load double, ptr %20, align 8
  %22 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %23 = load double, ptr %22, align 8
  %24 = fadd double %21, %23
  %25 = getelementptr %Vector, ptr %9, i64 0, i32 1
  store double %14, ptr %25, align 8
  %26 = getelementptr %Vector, ptr %9, i64 0, i32 2
  store double %19, ptr %26, align 8
  %27 = getelementptr %Vector, ptr %9, i64 0, i32 3
  store double %24, ptr %27, align 8
  ret ptr %9
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define double @Vector.dot(ptr nocapture readonly %0, ptr nocapture readonly %1) #6 {
  %3 = getelementptr %Vector, ptr %0, i64 0, i32 1
  %4 = load double, ptr %3, align 8
  %5 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %6 = load double, ptr %5, align 8
  %7 = fmul double %4, %6
  %8 = getelementptr %Vector, ptr %0, i64 0, i32 2
  %9 = load double, ptr %8, align 8
  %10 = getelementptr %Vector, ptr %1, i64 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = fmul double %9, %11
  %13 = fadd double %7, %12
  %14 = getelementptr %Vector, ptr %0, i64 0, i32 3
  %15 = load double, ptr %14, align 8
  %16 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %17 = load double, ptr %16, align 8
  %18 = fmul double %15, %17
  %19 = fadd double %13, %18
  ret double %19
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: readwrite)
define double @Vector.mag(ptr nocapture readonly %0) #7 {
  %2 = getelementptr %Vector, ptr %0, i64 0, i32 1
  %3 = load double, ptr %2, align 8
  %4 = fmul double %3, %3
  %5 = getelementptr %Vector, ptr %0, i64 0, i32 2
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, %6
  %8 = fadd double %4, %7
  %9 = getelementptr %Vector, ptr %0, i64 0, i32 3
  %10 = load double, ptr %9, align 8
  %11 = fmul double %10, %10
  %12 = fadd double %8, %11
  %13 = tail call double @sqrt(double %12)
  ret double %13
}

define ptr @Vector.norm(ptr nocapture readonly %0) {
  %2 = getelementptr %Vector, ptr %0, i64 0, i32 1
  %3 = load double, ptr %2, align 8
  %4 = fmul double %3, %3
  %5 = getelementptr %Vector, ptr %0, i64 0, i32 2
  %6 = load double, ptr %5, align 8
  %7 = fmul double %6, %6
  %8 = fadd double %4, %7
  %9 = getelementptr %Vector, ptr %0, i64 0, i32 3
  %10 = load double, ptr %9, align 8
  %11 = fmul double %10, %10
  %12 = fadd double %8, %11
  %13 = tail call double @sqrt(double %12)
  %14 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %14, 0
  br i1 %.not, label %15, label %18

15:                                               ; preds = %1
  %16 = alloca i64, align 8
  %17 = call i64 @GC_make_descriptor(ptr nonnull %16, i64 4)
  store i64 %17, ptr @Vector..type_descr, align 8
  br label %18

18:                                               ; preds = %15, %1
  %19 = phi i64 [ %17, %15 ], [ %14, %1 ]
  %20 = fcmp oeq double %13, 0.000000e+00
  %21 = fdiv double 1.000000e+00, %13
  %22 = select i1 %20, double 0x7FF0000000000000, double %21
  %23 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %19)
  store ptr @Vector..vtbl, ptr %23, align 8
  %24 = load double, ptr %2, align 8
  %25 = fmul double %22, %24
  %26 = load double, ptr %5, align 8
  %27 = fmul double %22, %26
  %28 = load double, ptr %9, align 8
  %29 = fmul double %22, %28
  %30 = getelementptr %Vector, ptr %23, i64 0, i32 1
  store double %25, ptr %30, align 8
  %31 = getelementptr %Vector, ptr %23, i64 0, i32 2
  store double %27, ptr %31, align 8
  %32 = getelementptr %Vector, ptr %23, i64 0, i32 3
  store double %29, ptr %32, align 8
  ret ptr %23
}

define ptr @Vector.cross(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Vector..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Vector..vtbl, ptr %9, align 8
  %10 = getelementptr %Vector, ptr %0, i64 0, i32 2
  %11 = load double, ptr %10, align 8
  %12 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %13 = load double, ptr %12, align 8
  %14 = fmul double %11, %13
  %15 = getelementptr %Vector, ptr %0, i64 0, i32 3
  %16 = load double, ptr %15, align 8
  %17 = getelementptr %Vector, ptr %1, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = fmul double %16, %18
  %20 = fsub double %14, %19
  %21 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %22 = load double, ptr %21, align 8
  %23 = fmul double %16, %22
  %24 = getelementptr %Vector, ptr %0, i64 0, i32 1
  %25 = load double, ptr %24, align 8
  %26 = fmul double %13, %25
  %27 = fsub double %23, %26
  %28 = fmul double %18, %25
  %29 = fmul double %11, %22
  %30 = fsub double %28, %29
  %31 = getelementptr %Vector, ptr %9, i64 0, i32 1
  store double %20, ptr %31, align 8
  %32 = getelementptr %Vector, ptr %9, i64 0, i32 2
  store double %27, ptr %32, align 8
  %33 = getelementptr %Vector, ptr %9, i64 0, i32 3
  store double %30, ptr %33, align 8
  ret ptr %9
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @Color..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @Color..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @Color..new() {
  %1 = load i64, ptr @Color..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 4)
  store i64 %4, ptr @Color..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %6)
  store ptr @Color..vtbl, ptr %7, align 8
  ret ptr %7
}

define ptr @Color.scale(double %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Color..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Color..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Color..vtbl, ptr %9, align 8
  %10 = getelementptr %Color, ptr %1, i64 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = fmul double %11, %0
  %13 = getelementptr %Color, ptr %1, i64 0, i32 2
  %14 = load double, ptr %13, align 8
  %15 = fmul double %14, %0
  %16 = getelementptr %Color, ptr %1, i64 0, i32 3
  %17 = load double, ptr %16, align 8
  %18 = fmul double %17, %0
  %19 = getelementptr %Color, ptr %9, i64 0, i32 1
  store double %12, ptr %19, align 8
  %20 = getelementptr %Color, ptr %9, i64 0, i32 2
  store double %15, ptr %20, align 8
  %21 = getelementptr %Color, ptr %9, i64 0, i32 3
  store double %18, ptr %21, align 8
  ret ptr %9
}

define ptr @Color.plus(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Color..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Color..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Color..vtbl, ptr %9, align 8
  %10 = getelementptr %Color, ptr %0, i64 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = getelementptr %Color, ptr %1, i64 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = fadd double %11, %13
  %15 = getelementptr %Color, ptr %0, i64 0, i32 2
  %16 = load double, ptr %15, align 8
  %17 = getelementptr %Color, ptr %1, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = fadd double %16, %18
  %20 = getelementptr %Color, ptr %0, i64 0, i32 3
  %21 = load double, ptr %20, align 8
  %22 = getelementptr %Color, ptr %1, i64 0, i32 3
  %23 = load double, ptr %22, align 8
  %24 = fadd double %21, %23
  %25 = getelementptr %Color, ptr %9, i64 0, i32 1
  store double %14, ptr %25, align 8
  %26 = getelementptr %Color, ptr %9, i64 0, i32 2
  store double %19, ptr %26, align 8
  %27 = getelementptr %Color, ptr %9, i64 0, i32 3
  store double %24, ptr %27, align 8
  ret ptr %9
}

define ptr @Color.times(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = load i64, ptr @Color..type_descr, align 8
  %.not = icmp eq i64 %3, 0
  br i1 %.not, label %4, label %7

4:                                                ; preds = %2
  %5 = alloca i64, align 8
  %6 = call i64 @GC_make_descriptor(ptr nonnull %5, i64 4)
  store i64 %6, ptr @Color..type_descr, align 8
  br label %7

7:                                                ; preds = %4, %2
  %8 = phi i64 [ %6, %4 ], [ %3, %2 ]
  %9 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %8)
  store ptr @Color..vtbl, ptr %9, align 8
  %10 = getelementptr %Color, ptr %0, i64 0, i32 1
  %11 = load double, ptr %10, align 8
  %12 = getelementptr %Color, ptr %1, i64 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = fmul double %11, %13
  %15 = getelementptr %Color, ptr %0, i64 0, i32 2
  %16 = load double, ptr %15, align 8
  %17 = getelementptr %Color, ptr %1, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = fmul double %16, %18
  %20 = getelementptr %Color, ptr %0, i64 0, i32 3
  %21 = load double, ptr %20, align 8
  %22 = getelementptr %Color, ptr %1, i64 0, i32 3
  %23 = load double, ptr %22, align 8
  %24 = fmul double %21, %23
  %25 = getelementptr %Color, ptr %9, i64 0, i32 1
  store double %14, ptr %25, align 8
  %26 = getelementptr %Color, ptr %9, i64 0, i32 2
  store double %19, ptr %26, align 8
  %27 = getelementptr %Color, ptr %9, i64 0, i32 3
  store double %24, ptr %27, align 8
  ret ptr %9
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define double @.f_Color.toDrawingColor..afL57C24L57C51H14864701(double %0) local_unnamed_addr #8 {
  %2 = fcmp ogt double %0, 1.000000e+00
  %3 = select i1 %2, double 1.000000e+00, double %0
  ret double %3
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: read)
define { double, double, double } @Color.toDrawingColor(ptr nocapture readonly %0) #9 {
  %2 = getelementptr %Color, ptr %0, i64 0, i32 1
  %3 = load double, ptr %2, align 8
  %4 = fcmp ogt double %3, 1.000000e+00
  %5 = select i1 %4, double 1.000000e+00, double %3
  %6 = fmul double %5, 2.550000e+02
  %7 = tail call double @llvm.floor.f64(double %6)
  %8 = getelementptr %Color, ptr %0, i64 0, i32 2
  %9 = load double, ptr %8, align 8
  %10 = fcmp ogt double %9, 1.000000e+00
  %11 = select i1 %10, double 1.000000e+00, double %9
  %12 = fmul double %11, 2.550000e+02
  %13 = tail call double @llvm.floor.f64(double %12)
  %14 = getelementptr %Color, ptr %0, i64 0, i32 3
  %15 = load double, ptr %14, align 8
  %16 = fcmp ogt double %15, 1.000000e+00
  %17 = select i1 %16, double 1.000000e+00, double %15
  %18 = fmul double %17, 2.550000e+02
  %19 = tail call double @llvm.floor.f64(double %18)
  %.fca.0.insert25 = insertvalue { double, double, double } poison, double %7, 0
  %.fca.1.insert27 = insertvalue { double, double, double } %.fca.0.insert25, double %13, 1
  %.fca.2.insert29 = insertvalue { double, double, double } %.fca.1.insert27, double %19, 2
  ret { double, double, double } %.fca.2.insert29
}

define void @Color.static_constructor() local_unnamed_addr {
  %1 = load i64, ptr @Color..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 4)
  store i64 %4, ptr @Color..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %6)
  store ptr @Color..vtbl, ptr %7, align 8
  %8 = getelementptr %Color, ptr %7, i64 0, i32 1
  store double 1.000000e+00, ptr %8, align 8
  %9 = getelementptr %Color, ptr %7, i64 0, i32 2
  store double 1.000000e+00, ptr %9, align 8
  %10 = getelementptr %Color, ptr %7, i64 0, i32 3
  store double 1.000000e+00, ptr %10, align 8
  store ptr %7, ptr @Color.white, align 8
  %11 = load i64, ptr @Color..type_descr, align 8
  %.not25 = icmp eq i64 %11, 0
  br i1 %.not25, label %12, label %15

12:                                               ; preds = %5
  %13 = alloca i64, align 8
  %14 = call i64 @GC_make_descriptor(ptr nonnull %13, i64 4)
  store i64 %14, ptr @Color..type_descr, align 8
  br label %15

15:                                               ; preds = %12, %5
  %16 = phi i64 [ %14, %12 ], [ %11, %5 ]
  %17 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %16)
  store ptr @Color..vtbl, ptr %17, align 8
  %18 = getelementptr %Color, ptr %17, i64 0, i32 1
  store double 5.000000e-01, ptr %18, align 8
  %19 = getelementptr %Color, ptr %17, i64 0, i32 2
  store double 5.000000e-01, ptr %19, align 8
  %20 = getelementptr %Color, ptr %17, i64 0, i32 3
  store double 5.000000e-01, ptr %20, align 8
  store ptr %17, ptr @Color.grey, align 8
  %21 = load i64, ptr @Color..type_descr, align 8
  %.not26 = icmp eq i64 %21, 0
  br i1 %.not26, label %22, label %25

22:                                               ; preds = %15
  %23 = alloca i64, align 8
  %24 = call i64 @GC_make_descriptor(ptr nonnull %23, i64 4)
  store i64 %24, ptr @Color..type_descr, align 8
  br label %25

25:                                               ; preds = %22, %15
  %26 = phi i64 [ %24, %22 ], [ %21, %15 ]
  %27 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %26)
  store ptr @Color..vtbl, ptr %27, align 8
  %28 = getelementptr %Color, ptr %27, i64 0, i32 1
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %28, i8 0, i64 24, i1 false)
  store ptr %27, ptr @Color.black, align 8
  store ptr %27, ptr @Color.background, align 8
  store ptr %27, ptr @Color.defaultColor, align 8
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @Camera..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @Camera..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @Camera..new() {
  %1 = load i64, ptr @Camera..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 5)
  store i64 %4, ptr @Camera..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %6)
  store ptr @Camera..vtbl, ptr %7, align 8
  ret ptr %7
}

define void @Camera.constructor(ptr nocapture %0, ptr %1, ptr nocapture readonly %2) local_unnamed_addr {
  %4 = getelementptr %Camera, ptr %0, i64 0, i32 4
  store ptr %1, ptr %4, align 8
  %5 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %5, 0
  br i1 %.not, label %6, label %9

6:                                                ; preds = %3
  %7 = alloca i64, align 8
  %8 = call i64 @GC_make_descriptor(ptr nonnull %7, i64 4)
  store i64 %8, ptr @Vector..type_descr, align 8
  br label %9

9:                                                ; preds = %6, %3
  %10 = phi i64 [ %8, %6 ], [ %5, %3 ]
  %11 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %10)
  store ptr @Vector..vtbl, ptr %11, align 8
  %12 = getelementptr %Vector, ptr %11, i64 0, i32 1
  store double 0.000000e+00, ptr %12, align 8
  %13 = getelementptr %Vector, ptr %11, i64 0, i32 2
  store double -1.000000e+00, ptr %13, align 8
  %14 = getelementptr %Vector, ptr %11, i64 0, i32 3
  store double 0.000000e+00, ptr %14, align 8
  %15 = getelementptr %Camera, ptr %0, i64 0, i32 1
  %16 = load ptr, ptr %4, align 8
  %17 = load i64, ptr @Vector..type_descr, align 8
  %.not110 = icmp eq i64 %17, 0
  br i1 %.not110, label %18, label %21

18:                                               ; preds = %9
  %19 = alloca i64, align 8
  %20 = call i64 @GC_make_descriptor(ptr nonnull %19, i64 4)
  store i64 %20, ptr @Vector..type_descr, align 8
  br label %21

21:                                               ; preds = %18, %9
  %22 = phi i64 [ %20, %18 ], [ %17, %9 ]
  %23 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %22)
  store ptr @Vector..vtbl, ptr %23, align 8
  %24 = getelementptr %Vector, ptr %2, i64 0, i32 1
  %25 = load double, ptr %24, align 8
  %26 = getelementptr %Vector, ptr %16, i64 0, i32 1
  %27 = load double, ptr %26, align 8
  %28 = fsub double %25, %27
  %29 = getelementptr %Vector, ptr %2, i64 0, i32 2
  %30 = load double, ptr %29, align 8
  %31 = getelementptr %Vector, ptr %16, i64 0, i32 2
  %32 = load double, ptr %31, align 8
  %33 = fsub double %30, %32
  %34 = getelementptr %Vector, ptr %2, i64 0, i32 3
  %35 = load double, ptr %34, align 8
  %36 = getelementptr %Vector, ptr %16, i64 0, i32 3
  %37 = load double, ptr %36, align 8
  %38 = fsub double %35, %37
  %39 = getelementptr %Vector, ptr %23, i64 0, i32 1
  store double %28, ptr %39, align 8
  %40 = getelementptr %Vector, ptr %23, i64 0, i32 2
  store double %33, ptr %40, align 8
  %41 = getelementptr %Vector, ptr %23, i64 0, i32 3
  store double %38, ptr %41, align 8
  %42 = call ptr @Vector.norm(ptr nonnull %23)
  store ptr %42, ptr %15, align 8
  %43 = getelementptr %Camera, ptr %0, i64 0, i32 2
  %44 = load i64, ptr @Vector..type_descr, align 8
  %.not111 = icmp eq i64 %44, 0
  br i1 %.not111, label %45, label %48

45:                                               ; preds = %21
  %46 = alloca i64, align 8
  %47 = call i64 @GC_make_descriptor(ptr nonnull %46, i64 4)
  store i64 %47, ptr @Vector..type_descr, align 8
  br label %48

48:                                               ; preds = %45, %21
  %49 = phi i64 [ %47, %45 ], [ %44, %21 ]
  %50 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %49)
  store ptr @Vector..vtbl, ptr %50, align 8
  %51 = getelementptr %Vector, ptr %42, i64 0, i32 2
  %52 = load double, ptr %51, align 8
  %53 = load double, ptr %14, align 8
  %54 = fmul double %52, %53
  %55 = getelementptr %Vector, ptr %42, i64 0, i32 3
  %56 = load double, ptr %55, align 8
  %57 = load double, ptr %13, align 8
  %58 = fmul double %56, %57
  %59 = fsub double %54, %58
  %60 = load double, ptr %12, align 8
  %61 = fmul double %56, %60
  %62 = getelementptr %Vector, ptr %42, i64 0, i32 1
  %63 = load double, ptr %62, align 8
  %64 = fmul double %53, %63
  %65 = fsub double %61, %64
  %66 = fmul double %57, %63
  %67 = fmul double %52, %60
  %68 = fsub double %66, %67
  %69 = getelementptr %Vector, ptr %50, i64 0, i32 1
  store double %59, ptr %69, align 8
  %70 = getelementptr %Vector, ptr %50, i64 0, i32 2
  store double %65, ptr %70, align 8
  %71 = getelementptr %Vector, ptr %50, i64 0, i32 3
  store double %68, ptr %71, align 8
  %72 = call ptr @Vector.norm(ptr nonnull %50)
  %73 = load i64, ptr @Vector..type_descr, align 8
  %.not112 = icmp eq i64 %73, 0
  br i1 %.not112, label %74, label %77

74:                                               ; preds = %48
  %75 = alloca i64, align 8
  %76 = call i64 @GC_make_descriptor(ptr nonnull %75, i64 4)
  store i64 %76, ptr @Vector..type_descr, align 8
  br label %77

77:                                               ; preds = %74, %48
  %78 = phi i64 [ %76, %74 ], [ %73, %48 ]
  %79 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %78)
  store ptr @Vector..vtbl, ptr %79, align 8
  %80 = getelementptr %Vector, ptr %72, i64 0, i32 1
  %81 = load double, ptr %80, align 8
  %82 = fmul double %81, 1.500000e+00
  %83 = getelementptr %Vector, ptr %72, i64 0, i32 2
  %84 = load double, ptr %83, align 8
  %85 = fmul double %84, 1.500000e+00
  %86 = getelementptr %Vector, ptr %72, i64 0, i32 3
  %87 = load double, ptr %86, align 8
  %88 = fmul double %87, 1.500000e+00
  %89 = getelementptr %Vector, ptr %79, i64 0, i32 1
  store double %82, ptr %89, align 8
  %90 = getelementptr %Vector, ptr %79, i64 0, i32 2
  store double %85, ptr %90, align 8
  %91 = getelementptr %Vector, ptr %79, i64 0, i32 3
  store double %88, ptr %91, align 8
  store ptr %79, ptr %43, align 8
  %92 = load ptr, ptr %15, align 8
  %93 = load i64, ptr @Vector..type_descr, align 8
  %.not113 = icmp eq i64 %93, 0
  br i1 %.not113, label %94, label %97

94:                                               ; preds = %77
  %95 = alloca i64, align 8
  %96 = call i64 @GC_make_descriptor(ptr nonnull %95, i64 4)
  store i64 %96, ptr @Vector..type_descr, align 8
  br label %97

97:                                               ; preds = %94, %77
  %98 = phi i64 [ %96, %94 ], [ %93, %77 ]
  %99 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %98)
  store ptr @Vector..vtbl, ptr %99, align 8
  %100 = getelementptr %Vector, ptr %92, i64 0, i32 2
  %101 = load double, ptr %100, align 8
  %102 = load double, ptr %91, align 8
  %103 = fmul double %101, %102
  %104 = getelementptr %Vector, ptr %92, i64 0, i32 3
  %105 = load double, ptr %104, align 8
  %106 = load double, ptr %90, align 8
  %107 = fmul double %105, %106
  %108 = fsub double %103, %107
  %109 = load double, ptr %89, align 8
  %110 = fmul double %105, %109
  %111 = getelementptr %Vector, ptr %92, i64 0, i32 1
  %112 = load double, ptr %111, align 8
  %113 = fmul double %102, %112
  %114 = fsub double %110, %113
  %115 = fmul double %106, %112
  %116 = fmul double %101, %109
  %117 = fsub double %115, %116
  %118 = getelementptr %Vector, ptr %99, i64 0, i32 1
  store double %108, ptr %118, align 8
  %119 = getelementptr %Vector, ptr %99, i64 0, i32 2
  store double %114, ptr %119, align 8
  %120 = getelementptr %Vector, ptr %99, i64 0, i32 3
  store double %117, ptr %120, align 8
  %121 = call ptr @Vector.norm(ptr nonnull %99)
  %122 = load i64, ptr @Vector..type_descr, align 8
  %.not114 = icmp eq i64 %122, 0
  br i1 %.not114, label %123, label %126

123:                                              ; preds = %97
  %124 = alloca i64, align 8
  %125 = call i64 @GC_make_descriptor(ptr nonnull %124, i64 4)
  store i64 %125, ptr @Vector..type_descr, align 8
  br label %126

126:                                              ; preds = %123, %97
  %127 = phi i64 [ %125, %123 ], [ %122, %97 ]
  %128 = getelementptr %Camera, ptr %0, i64 0, i32 3
  %129 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %127)
  store ptr @Vector..vtbl, ptr %129, align 8
  %130 = getelementptr %Vector, ptr %121, i64 0, i32 1
  %131 = load double, ptr %130, align 8
  %132 = fmul double %131, 1.500000e+00
  %133 = getelementptr %Vector, ptr %121, i64 0, i32 2
  %134 = load double, ptr %133, align 8
  %135 = fmul double %134, 1.500000e+00
  %136 = getelementptr %Vector, ptr %121, i64 0, i32 3
  %137 = load double, ptr %136, align 8
  %138 = fmul double %137, 1.500000e+00
  %139 = getelementptr %Vector, ptr %129, i64 0, i32 1
  store double %132, ptr %139, align 8
  %140 = getelementptr %Vector, ptr %129, i64 0, i32 2
  store double %135, ptr %140, align 8
  %141 = getelementptr %Vector, ptr %129, i64 0, i32 3
  store double %138, ptr %141, align 8
  store ptr %129, ptr %128, align 8
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @Sphere..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @Sphere..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @Sphere..new() {
  %1 = load i64, ptr @Sphere..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 5)
  store i64 %4, ptr @Sphere..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %6)
  store ptr @Sphere..vtbl, ptr %7, align 8
  ret ptr %7
}

define ptr @Sphere.normal(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = getelementptr %Sphere, ptr %0, i64 0, i32 2
  %4 = load ptr, ptr %3, align 8
  %5 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %5, 0
  br i1 %.not, label %6, label %9

6:                                                ; preds = %2
  %7 = alloca i64, align 8
  %8 = call i64 @GC_make_descriptor(ptr nonnull %7, i64 4)
  store i64 %8, ptr @Vector..type_descr, align 8
  br label %9

9:                                                ; preds = %6, %2
  %10 = phi i64 [ %8, %6 ], [ %5, %2 ]
  %11 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %10)
  store ptr @Vector..vtbl, ptr %11, align 8
  %12 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = getelementptr %Vector, ptr %4, i64 0, i32 1
  %15 = load double, ptr %14, align 8
  %16 = fsub double %13, %15
  %17 = getelementptr %Vector, ptr %1, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = getelementptr %Vector, ptr %4, i64 0, i32 2
  %20 = load double, ptr %19, align 8
  %21 = fsub double %18, %20
  %22 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %23 = load double, ptr %22, align 8
  %24 = getelementptr %Vector, ptr %4, i64 0, i32 3
  %25 = load double, ptr %24, align 8
  %26 = fsub double %23, %25
  %27 = getelementptr %Vector, ptr %11, i64 0, i32 1
  store double %16, ptr %27, align 8
  %28 = getelementptr %Vector, ptr %11, i64 0, i32 2
  store double %21, ptr %28, align 8
  %29 = getelementptr %Vector, ptr %11, i64 0, i32 3
  store double %26, ptr %29, align 8
  %30 = call ptr @Vector.norm(ptr nonnull %11)
  ret ptr %30
}

define { ptr, ptr } @Sphere.intersect(ptr %0, { ptr, ptr } %1) {
  %.fca.0.extract7 = extractvalue { ptr, ptr } %1, 0
  %.fca.1.extract8 = extractvalue { ptr, ptr } %1, 1
  %3 = getelementptr %Sphere, ptr %0, i64 0, i32 2
  %4 = load ptr, ptr %3, align 8
  %5 = load ptr, ptr %.fca.0.extract7, align 8
  %6 = ptrtoint ptr %.fca.1.extract8 to i64
  %7 = ptrtoint ptr %5 to i64
  %8 = add i64 %7, %6
  %9 = inttoptr i64 %8 to ptr
  %10 = load ptr, ptr %9, align 8
  %11 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %11, 0
  br i1 %.not, label %12, label %15

12:                                               ; preds = %2
  %13 = alloca i64, align 8
  %14 = call i64 @GC_make_descriptor(ptr nonnull %13, i64 4)
  store i64 %14, ptr @Vector..type_descr, align 8
  br label %15

15:                                               ; preds = %12, %2
  %16 = phi i64 [ %14, %12 ], [ %11, %2 ]
  %17 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %16)
  store ptr @Vector..vtbl, ptr %17, align 8
  %18 = getelementptr %Vector, ptr %4, i64 0, i32 1
  %19 = load double, ptr %18, align 8
  %20 = getelementptr %Vector, ptr %10, i64 0, i32 1
  %21 = load double, ptr %20, align 8
  %22 = fsub double %19, %21
  %23 = getelementptr %Vector, ptr %4, i64 0, i32 2
  %24 = load double, ptr %23, align 8
  %25 = getelementptr %Vector, ptr %10, i64 0, i32 2
  %26 = load double, ptr %25, align 8
  %27 = fsub double %24, %26
  %28 = getelementptr %Vector, ptr %4, i64 0, i32 3
  %29 = load double, ptr %28, align 8
  %30 = getelementptr %Vector, ptr %10, i64 0, i32 3
  %31 = load double, ptr %30, align 8
  %32 = fsub double %29, %31
  %33 = getelementptr %Vector, ptr %17, i64 0, i32 1
  store double %22, ptr %33, align 8
  %34 = getelementptr %Vector, ptr %17, i64 0, i32 2
  store double %27, ptr %34, align 8
  %35 = getelementptr %Vector, ptr %17, i64 0, i32 3
  store double %32, ptr %35, align 8
  %36 = getelementptr ptr, ptr %.fca.0.extract7, i64 1
  %37 = load ptr, ptr %36, align 8
  %38 = ptrtoint ptr %37 to i64
  %39 = add i64 %38, %6
  %40 = inttoptr i64 %39 to ptr
  %41 = load ptr, ptr %40, align 8
  %42 = getelementptr %Vector, ptr %41, i64 0, i32 1
  %43 = load double, ptr %42, align 8
  %44 = fmul double %22, %43
  %45 = getelementptr %Vector, ptr %41, i64 0, i32 2
  %46 = load double, ptr %45, align 8
  %47 = fmul double %27, %46
  %48 = fadd double %44, %47
  %49 = getelementptr %Vector, ptr %41, i64 0, i32 3
  %50 = load double, ptr %49, align 8
  %51 = fmul double %32, %50
  %52 = fadd double %48, %51
  %53 = fcmp ult double %52, 0.000000e+00
  br i1 %53, label %.thread, label %54

54:                                               ; preds = %15
  %55 = getelementptr %Sphere, ptr %0, i64 0, i32 1
  %56 = load double, ptr %55, align 8
  %57 = fmul double %22, %22
  %58 = fmul double %27, %27
  %59 = fadd double %57, %58
  %60 = fmul double %32, %32
  %61 = fadd double %59, %60
  %62 = fmul double %52, %52
  %63 = fsub double %61, %62
  %64 = fsub double %56, %63
  %65 = fcmp ult double %64, 0.000000e+00
  br i1 %65, label %.thread, label %66

66:                                               ; preds = %54
  %67 = call double @sqrt(double %64)
  %68 = fsub double %52, %67
  %69 = fptosi double %68 to i32
  %70 = icmp eq i32 %69, 0
  br i1 %70, label %.thread, label %71

71:                                               ; preds = %66
  %72 = load ptr, ptr %0, align 8
  %73 = load ptr, ptr %72, align 8
  %74 = sitofp i32 %69 to double
  %75 = call ptr @GC_malloc(i64 40)
  store ptr %73, ptr %75, align 8
  %.repack79 = getelementptr inbounds { ptr, ptr }, ptr %75, i64 0, i32 1
  store ptr %0, ptr %.repack79, align 8
  %.repack75 = getelementptr inbounds { { ptr, ptr }, { ptr, ptr }, double }, ptr %75, i64 0, i32 1
  store ptr %.fca.0.extract7, ptr %.repack75, align 8
  %.repack75.repack81 = getelementptr inbounds { { ptr, ptr }, { ptr, ptr }, double }, ptr %75, i64 0, i32 1, i32 1
  store ptr %.fca.1.extract8, ptr %.repack75.repack81, align 8
  %.repack77 = getelementptr inbounds { { ptr, ptr }, { ptr, ptr }, double }, ptr %75, i64 0, i32 2
  store double %74, ptr %.repack77, align 8
  br label %.thread

.thread:                                          ; preds = %15, %54, %66, %71
  %.sroa.3.0 = phi ptr [ %75, %71 ], [ null, %66 ], [ null, %54 ], [ null, %15 ]
  %.sroa.0.0 = phi ptr [ @Intersection.14787487..vtbl, %71 ], [ null, %66 ], [ null, %54 ], [ null, %15 ]
  %.fca.0.insert = insertvalue { ptr, ptr } poison, ptr %.sroa.0.0, 0
  %.fca.1.insert = insertvalue { ptr, ptr } %.fca.0.insert, ptr %.sroa.3.0, 1
  ret { ptr, ptr } %.fca.1.insert
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @Plane..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @Plane..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @Plane..new() {
  %1 = load i64, ptr @Plane..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 5)
  store i64 %4, ptr @Plane..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %6)
  store ptr @Plane..vtbl, ptr %7, align 8
  ret ptr %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define ptr @Plane.normal(ptr nocapture readonly %0, ptr nocapture readnone %1) #6 {
  %3 = getelementptr %Plane, ptr %0, i64 0, i32 1
  %4 = load ptr, ptr %3, align 8
  ret ptr %4
}

define { ptr, ptr } @Plane.intersect(ptr %0, { ptr, ptr } %1) {
  %.fca.0.extract8 = extractvalue { ptr, ptr } %1, 0
  %.fca.1.extract9 = extractvalue { ptr, ptr } %1, 1
  %3 = getelementptr %Plane, ptr %0, i64 0, i32 1
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr ptr, ptr %.fca.0.extract8, i64 1
  %6 = load ptr, ptr %5, align 8
  %7 = ptrtoint ptr %.fca.1.extract9 to i64
  %8 = ptrtoint ptr %6 to i64
  %9 = add i64 %8, %7
  %10 = inttoptr i64 %9 to ptr
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr %Vector, ptr %4, i64 0, i32 1
  %13 = load double, ptr %12, align 8
  %14 = getelementptr %Vector, ptr %11, i64 0, i32 1
  %15 = load double, ptr %14, align 8
  %16 = fmul double %13, %15
  %17 = getelementptr %Vector, ptr %4, i64 0, i32 2
  %18 = load double, ptr %17, align 8
  %19 = getelementptr %Vector, ptr %11, i64 0, i32 2
  %20 = load double, ptr %19, align 8
  %21 = fmul double %18, %20
  %22 = fadd double %16, %21
  %23 = getelementptr %Vector, ptr %4, i64 0, i32 3
  %24 = load double, ptr %23, align 8
  %25 = getelementptr %Vector, ptr %11, i64 0, i32 3
  %26 = load double, ptr %25, align 8
  %27 = fmul double %24, %26
  %28 = fadd double %22, %27
  %29 = fcmp ogt double %28, 0.000000e+00
  br i1 %29, label %55, label %30

30:                                               ; preds = %2
  %31 = load ptr, ptr %.fca.0.extract8, align 8
  %32 = ptrtoint ptr %31 to i64
  %33 = add i64 %32, %7
  %34 = inttoptr i64 %33 to ptr
  %35 = load ptr, ptr %34, align 8
  %36 = getelementptr %Vector, ptr %35, i64 0, i32 1
  %37 = load double, ptr %36, align 8
  %38 = fmul double %13, %37
  %39 = getelementptr %Vector, ptr %35, i64 0, i32 2
  %40 = load double, ptr %39, align 8
  %41 = fmul double %18, %40
  %42 = fadd double %38, %41
  %43 = getelementptr %Vector, ptr %35, i64 0, i32 3
  %44 = load double, ptr %43, align 8
  %45 = fmul double %24, %44
  %46 = fadd double %42, %45
  %47 = getelementptr %Plane, ptr %0, i64 0, i32 2
  %48 = load double, ptr %47, align 8
  %49 = fadd double %48, %46
  %50 = fsub double 0.000000e+00, %28
  %51 = fdiv double %49, %50
  %52 = load ptr, ptr %0, align 8
  %53 = load ptr, ptr %52, align 8
  %54 = tail call ptr @GC_malloc(i64 40)
  store ptr %53, ptr %54, align 8
  %.repack57 = getelementptr inbounds { ptr, ptr }, ptr %54, i64 0, i32 1
  store ptr %0, ptr %.repack57, align 8
  %.repack53 = getelementptr inbounds { { ptr, ptr }, { ptr, ptr }, double }, ptr %54, i64 0, i32 1
  store ptr %.fca.0.extract8, ptr %.repack53, align 8
  %.repack53.repack59 = getelementptr inbounds { { ptr, ptr }, { ptr, ptr }, double }, ptr %54, i64 0, i32 1, i32 1
  store ptr %.fca.1.extract9, ptr %.repack53.repack59, align 8
  %.repack55 = getelementptr inbounds { { ptr, ptr }, { ptr, ptr }, double }, ptr %54, i64 0, i32 2
  store double %51, ptr %.repack55, align 8
  br label %55

55:                                               ; preds = %2, %30
  %.sroa.3.0 = phi ptr [ %54, %30 ], [ null, %2 ]
  %.sroa.0.0 = phi ptr [ @Intersection.14787487..vtbl, %30 ], [ null, %2 ]
  %.fca.0.insert = insertvalue { ptr, ptr } poison, ptr %.sroa.0.0, 0
  %.fca.1.insert = insertvalue { ptr, ptr } %.fca.0.insert, ptr %.sroa.3.0, 1
  ret { ptr, ptr } %.fca.1.insert
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none)
define ptr @Surfaces..feL159C18L159C63H14774515(ptr nocapture readnone %0, ptr nocapture readnone %1) #10 {
  %3 = load ptr, ptr @Color.white, align 8
  ret ptr %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none)
define ptr @Surfaces..feL160C19L160C63H14867906(ptr nocapture readnone %0, ptr nocapture readnone %1) #10 {
  %3 = load ptr, ptr @Color.grey, align 8
  ret ptr %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define double @Surfaces..feL161C18L161C55H14867908(ptr nocapture readnone %0, ptr nocapture readnone %1) #8 {
  ret double 0x3FE6666666666666
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(read, inaccessiblemem: none)
define ptr @Surfaces..feL165C18L171C9H15086251(ptr nocapture readnone %0, ptr nocapture readonly %1) #11 {
  %3 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %4 = load double, ptr %3, align 8
  %5 = tail call double @llvm.floor.f64(double %4)
  %6 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %7 = load double, ptr %6, align 8
  %8 = tail call double @llvm.floor.f64(double %7)
  %9 = fadd double %5, %8
  %10 = frem double %9, 2.000000e+00
  %11 = fcmp ueq double %10, 0.000000e+00
  %Color.black.val = load ptr, ptr @Color.black, align 8
  %Color.white.val = load ptr, ptr @Color.white, align 8
  %.0 = select i1 %11, ptr %Color.black.val, ptr %Color.white.val
  ret ptr %.0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none)
define ptr @Surfaces..feL172C19L172C64H14759908(ptr nocapture readnone %0, ptr nocapture readnone %1) #10 {
  %3 = load ptr, ptr @Color.white, align 8
  ret ptr %3
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: read)
define double @Surfaces..feL173C18L179C9H14757289(ptr nocapture readnone %0, ptr nocapture readonly %1) #9 {
  %3 = getelementptr %Vector, ptr %1, i64 0, i32 3
  %4 = load double, ptr %3, align 8
  %5 = tail call double @llvm.floor.f64(double %4)
  %6 = getelementptr %Vector, ptr %1, i64 0, i32 1
  %7 = load double, ptr %6, align 8
  %8 = tail call double @llvm.floor.f64(double %7)
  %9 = fadd double %5, %8
  %10 = frem double %9, 2.000000e+00
  %11 = fcmp ueq double %10, 0.000000e+00
  %. = select i1 %11, double 0x3FE6666666666666, double 1.000000e-01
  ret double %.
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none)
define i1 @RayTracer..instanceOf(ptr nocapture readnone %0, ptr readonly %1) #4 {
  %3 = load ptr, ptr @RayTracer..rtti, align 8
  %4 = icmp ne ptr %1, null
  %5 = icmp ne ptr %3, null
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) %3)
  %9 = icmp eq i32 %8, 0
  br label %12

10:                                               ; preds = %2
  %11 = icmp eq ptr %3, %1
  br label %12

12:                                               ; preds = %10, %7
  %13 = phi i1 [ %9, %7 ], [ %11, %10 ]
  ret i1 %13
}

define ptr @RayTracer..new() {
  %1 = load i64, ptr @RayTracer..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 2)
  store i64 %4, ptr @RayTracer..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 16, i64 %6)
  store ptr @RayTracer..vtbl, ptr %7, align 8
  ret ptr %7
}

define { ptr, ptr } @RayTracer.intersections(ptr nocapture readnone %0, { ptr, ptr } %1, { ptr, ptr } %2) {
  %.fca.0.extract11 = extractvalue { ptr, ptr } %2, 0
  %.fca.1.extract12 = extractvalue { ptr, ptr } %2, 1
  %4 = load ptr, ptr %.fca.0.extract11, align 8
  %5 = ptrtoint ptr %.fca.1.extract12 to i64
  %6 = ptrtoint ptr %4 to i64
  %7 = add i64 %6, %5
  %8 = inttoptr i64 %7 to ptr
  %9 = getelementptr inbounds { ptr, i32 }, ptr %8, i64 0, i32 1
  %10 = load i32, ptr %9, align 4
  %11 = icmp sgt i32 %10, 0
  br i1 %11, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %3, %.critedge
  %indvars.iv = phi i64 [ %indvars.iv.next, %.critedge ], [ 0, %3 ]
  %.sroa.333.068 = phi ptr [ %.sroa.333.1, %.critedge ], [ null, %3 ]
  %.sroa.032.067 = phi ptr [ %.sroa.032.1, %.critedge ], [ null, %3 ]
  %.06266 = phi double [ %.1, %.critedge ], [ 0x7FF0000000000000, %3 ]
  %12 = load ptr, ptr %.fca.0.extract11, align 8
  %13 = ptrtoint ptr %12 to i64
  %14 = add i64 %13, %5
  %15 = inttoptr i64 %14 to ptr
  %16 = load ptr, ptr %15, align 8
  %17 = getelementptr { ptr, ptr }, ptr %16, i64 %indvars.iv
  %.unpack = load ptr, ptr %17, align 8
  %.elt64 = getelementptr { ptr, ptr }, ptr %16, i64 %indvars.iv, i32 1
  %.unpack65 = load ptr, ptr %.elt64, align 8
  %18 = load ptr, ptr %.unpack, align 8
  %19 = tail call { ptr, ptr } %18(ptr %.unpack65, { ptr, ptr } %1)
  %.fca.0.extract37 = extractvalue { ptr, ptr } %19, 0
  %.fca.1.extract38 = extractvalue { ptr, ptr } %19, 1
  %.not = icmp eq ptr %.fca.0.extract37, null
  br i1 %.not, label %.critedge, label %20

20:                                               ; preds = %.lr.ph
  %21 = getelementptr ptr, ptr %.fca.0.extract37, i64 1
  %22 = load ptr, ptr %21, align 8
  %23 = ptrtoint ptr %.fca.1.extract38 to i64
  %24 = ptrtoint ptr %22 to i64
  %25 = add i64 %24, %23
  %26 = inttoptr i64 %25 to ptr
  %27 = load double, ptr %26, align 8
  %28 = fcmp olt double %27, %.06266
  br i1 %28, label %29, label %.critedge

29:                                               ; preds = %20
  br label %.critedge

.critedge:                                        ; preds = %.lr.ph, %29, %20
  %.1 = phi double [ %27, %29 ], [ %.06266, %20 ], [ %.06266, %.lr.ph ]
  %.sroa.032.1 = phi ptr [ %.fca.0.extract37, %29 ], [ %.sroa.032.067, %20 ], [ %.sroa.032.067, %.lr.ph ]
  %.sroa.333.1 = phi ptr [ %.fca.1.extract38, %29 ], [ %.sroa.333.068, %20 ], [ %.sroa.333.068, %.lr.ph ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %30 = load i32, ptr %9, align 4
  %31 = sext i32 %30 to i64
  %32 = icmp slt i64 %indvars.iv.next, %31
  br i1 %32, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.critedge, %3
  %.sroa.032.0.lcssa = phi ptr [ null, %3 ], [ %.sroa.032.1, %.critedge ]
  %.sroa.333.0.lcssa = phi ptr [ null, %3 ], [ %.sroa.333.1, %.critedge ]
  %.fca.0.insert25 = insertvalue { ptr, ptr } poison, ptr %.sroa.032.0.lcssa, 0
  %.fca.1.insert27 = insertvalue { ptr, ptr } %.fca.0.insert25, ptr %.sroa.333.0.lcssa, 1
  ret { ptr, ptr } %.fca.1.insert27
}

define double @RayTracer.testRay(ptr %0, { ptr, ptr } %1, { ptr, ptr } %2) {
  %4 = load ptr, ptr %0, align 8
  %5 = getelementptr ptr, ptr %4, i64 2
  %6 = load ptr, ptr %5, align 8
  %7 = tail call { ptr, ptr } %6(ptr nonnull %0, { ptr, ptr } %1, { ptr, ptr } %2)
  %.fca.0.extract12 = extractvalue { ptr, ptr } %7, 0
  %.not = icmp eq ptr %.fca.0.extract12, null
  br i1 %.not, label %16, label %8

8:                                                ; preds = %3
  %.fca.1.extract13 = extractvalue { ptr, ptr } %7, 1
  %9 = getelementptr ptr, ptr %.fca.0.extract12, i64 1
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %.fca.1.extract13 to i64
  %12 = ptrtoint ptr %10 to i64
  %13 = add i64 %12, %11
  %14 = inttoptr i64 %13 to ptr
  %15 = load double, ptr %14, align 8
  br label %16

16:                                               ; preds = %3, %8
  %.0 = phi double [ %15, %8 ], [ undef, %3 ]
  ret double %.0
}

define ptr @RayTracer.traceRay(ptr %0, { ptr, ptr } %1, { ptr, ptr } %2, double %3) {
  %5 = load ptr, ptr %0, align 8
  %6 = getelementptr ptr, ptr %5, i64 2
  %7 = load ptr, ptr %6, align 8
  %8 = tail call { ptr, ptr } %7(ptr nonnull %0, { ptr, ptr } %1, { ptr, ptr } %2)
  %.fca.0.extract19 = extractvalue { ptr, ptr } %8, 0
  %9 = icmp eq ptr %.fca.0.extract19, null
  br i1 %9, label %10, label %12

10:                                               ; preds = %4
  %11 = load ptr, ptr @Color.background, align 8
  br label %17

12:                                               ; preds = %4
  %13 = load ptr, ptr %0, align 8
  %14 = getelementptr ptr, ptr %13, i64 5
  %15 = load ptr, ptr %14, align 8
  %16 = tail call ptr %15(ptr nonnull %0, { ptr, ptr } %8, { ptr, ptr } %2, double %3)
  br label %17

17:                                               ; preds = %10, %12
  %.0 = phi ptr [ %11, %10 ], [ %16, %12 ]
  ret ptr %.0
}

define ptr @RayTracer.shade(ptr %0, { ptr, ptr } %1, { ptr, ptr } %2, double %3) {
  %.fca.0.extract = extractvalue { ptr, ptr } %1, 0
  %.fca.1.extract = extractvalue { ptr, ptr } %1, 1
  %5 = load ptr, ptr %.fca.0.extract, align 8
  %6 = ptrtoint ptr %.fca.1.extract to i64
  %7 = ptrtoint ptr %5 to i64
  %8 = add i64 %7, %6
  %9 = inttoptr i64 %8 to ptr
  %.unpack = load ptr, ptr %9, align 8
  %.elt182 = getelementptr inbounds { ptr, ptr }, ptr %9, i64 0, i32 1
  %.unpack183 = load ptr, ptr %.elt182, align 8
  %10 = getelementptr ptr, ptr %.unpack, i64 1
  %11 = load ptr, ptr %10, align 8
  %12 = ptrtoint ptr %.unpack183 to i64
  %13 = ptrtoint ptr %11 to i64
  %14 = add i64 %13, %12
  %15 = inttoptr i64 %14 to ptr
  %16 = load ptr, ptr %15, align 8
  %17 = getelementptr ptr, ptr %.fca.0.extract, i64 1
  %18 = load ptr, ptr %17, align 8
  %19 = ptrtoint ptr %18 to i64
  %20 = add i64 %19, %6
  %21 = inttoptr i64 %20 to ptr
  %22 = load double, ptr %21, align 8
  %23 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %23, 0
  br i1 %.not, label %24, label %27

24:                                               ; preds = %4
  %25 = alloca i64, align 8
  %26 = call i64 @GC_make_descriptor(ptr nonnull %25, i64 4)
  store i64 %26, ptr @Vector..type_descr, align 8
  br label %27

27:                                               ; preds = %24, %4
  %28 = phi i64 [ %26, %24 ], [ %23, %4 ]
  %29 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %28)
  store ptr @Vector..vtbl, ptr %29, align 8
  %30 = getelementptr %Vector, ptr %16, i64 0, i32 1
  %31 = load double, ptr %30, align 8
  %32 = fmul double %22, %31
  %33 = getelementptr %Vector, ptr %16, i64 0, i32 2
  %34 = load double, ptr %33, align 8
  %35 = fmul double %22, %34
  %36 = getelementptr %Vector, ptr %16, i64 0, i32 3
  %37 = load double, ptr %36, align 8
  %38 = fmul double %22, %37
  %39 = getelementptr %Vector, ptr %29, i64 0, i32 1
  store double %32, ptr %39, align 8
  %40 = getelementptr %Vector, ptr %29, i64 0, i32 2
  store double %35, ptr %40, align 8
  %41 = getelementptr %Vector, ptr %29, i64 0, i32 3
  store double %38, ptr %41, align 8
  %42 = load ptr, ptr %.fca.0.extract, align 8
  %43 = ptrtoint ptr %42 to i64
  %44 = add i64 %43, %6
  %45 = inttoptr i64 %44 to ptr
  %.unpack184 = load ptr, ptr %45, align 8
  %.elt185 = getelementptr inbounds { ptr, ptr }, ptr %45, i64 0, i32 1
  %.unpack186 = load ptr, ptr %.elt185, align 8
  %46 = load ptr, ptr %.unpack184, align 8
  %47 = ptrtoint ptr %.unpack186 to i64
  %48 = ptrtoint ptr %46 to i64
  %49 = add i64 %48, %47
  %50 = inttoptr i64 %49 to ptr
  %51 = load ptr, ptr %50, align 8
  %52 = load i64, ptr @Vector..type_descr, align 8
  %.not187 = icmp eq i64 %52, 0
  br i1 %.not187, label %53, label %56

53:                                               ; preds = %27
  %54 = alloca i64, align 8
  %55 = call i64 @GC_make_descriptor(ptr nonnull %54, i64 4)
  store i64 %55, ptr @Vector..type_descr, align 8
  br label %56

56:                                               ; preds = %53, %27
  %57 = phi i64 [ %55, %53 ], [ %52, %27 ]
  %58 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %57)
  store ptr @Vector..vtbl, ptr %58, align 8
  %59 = load double, ptr %39, align 8
  %60 = getelementptr %Vector, ptr %51, i64 0, i32 1
  %61 = load double, ptr %60, align 8
  %62 = fadd double %59, %61
  %63 = load double, ptr %40, align 8
  %64 = getelementptr %Vector, ptr %51, i64 0, i32 2
  %65 = load double, ptr %64, align 8
  %66 = fadd double %63, %65
  %67 = load double, ptr %41, align 8
  %68 = getelementptr %Vector, ptr %51, i64 0, i32 3
  %69 = load double, ptr %68, align 8
  %70 = fadd double %67, %69
  %71 = getelementptr %Vector, ptr %58, i64 0, i32 1
  store double %62, ptr %71, align 8
  %72 = getelementptr %Vector, ptr %58, i64 0, i32 2
  store double %66, ptr %72, align 8
  %73 = getelementptr %Vector, ptr %58, i64 0, i32 3
  store double %70, ptr %73, align 8
  %74 = getelementptr ptr, ptr %.fca.0.extract, i64 4
  %75 = load ptr, ptr %74, align 8
  %76 = ptrtoint ptr %75 to i64
  %77 = add i64 %76, %6
  %78 = inttoptr i64 %77 to ptr
  %.unpack188 = load ptr, ptr %78, align 8
  %.elt189 = getelementptr inbounds { ptr, ptr }, ptr %78, i64 0, i32 1
  %.unpack190 = load ptr, ptr %.elt189, align 8
  %79 = getelementptr ptr, ptr %.unpack188, i64 1
  %80 = load ptr, ptr %79, align 8
  %81 = call ptr %80(ptr %.unpack190, ptr nonnull %58)
  %82 = getelementptr %Vector, ptr %81, i64 0, i32 1
  %83 = load double, ptr %82, align 8
  %84 = load double, ptr %30, align 8
  %85 = fmul double %83, %84
  %86 = getelementptr %Vector, ptr %81, i64 0, i32 2
  %87 = load double, ptr %86, align 8
  %88 = load double, ptr %33, align 8
  %89 = fmul double %87, %88
  %90 = fadd double %85, %89
  %91 = getelementptr %Vector, ptr %81, i64 0, i32 3
  %92 = load double, ptr %91, align 8
  %93 = load double, ptr %36, align 8
  %94 = fmul double %92, %93
  %95 = fadd double %90, %94
  %96 = load i64, ptr @Vector..type_descr, align 8
  %.not191 = icmp eq i64 %96, 0
  br i1 %.not191, label %97, label %100

97:                                               ; preds = %56
  %98 = alloca i64, align 8
  %99 = call i64 @GC_make_descriptor(ptr nonnull %98, i64 4)
  store i64 %99, ptr @Vector..type_descr, align 8
  br label %100

100:                                              ; preds = %97, %56
  %101 = phi i64 [ %99, %97 ], [ %96, %56 ]
  %102 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %101)
  store ptr @Vector..vtbl, ptr %102, align 8
  %103 = load double, ptr %82, align 8
  %104 = fmul double %95, %103
  %105 = load double, ptr %86, align 8
  %106 = fmul double %95, %105
  %107 = load double, ptr %91, align 8
  %108 = fmul double %95, %107
  %109 = getelementptr %Vector, ptr %102, i64 0, i32 1
  store double %104, ptr %109, align 8
  %110 = getelementptr %Vector, ptr %102, i64 0, i32 2
  store double %106, ptr %110, align 8
  %111 = getelementptr %Vector, ptr %102, i64 0, i32 3
  store double %108, ptr %111, align 8
  %112 = load i64, ptr @Vector..type_descr, align 8
  %.not192 = icmp eq i64 %112, 0
  br i1 %.not192, label %113, label %116

113:                                              ; preds = %100
  %114 = alloca i64, align 8
  %115 = call i64 @GC_make_descriptor(ptr nonnull %114, i64 4)
  store i64 %115, ptr @Vector..type_descr, align 8
  br label %116

116:                                              ; preds = %113, %100
  %117 = phi i64 [ %115, %113 ], [ %112, %100 ]
  %118 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %117)
  store ptr @Vector..vtbl, ptr %118, align 8
  %119 = load double, ptr %109, align 8
  %120 = fmul double %119, 2.000000e+00
  %121 = load double, ptr %110, align 8
  %122 = fmul double %121, 2.000000e+00
  %123 = load double, ptr %111, align 8
  %124 = fmul double %123, 2.000000e+00
  %125 = getelementptr %Vector, ptr %118, i64 0, i32 1
  store double %120, ptr %125, align 8
  %126 = getelementptr %Vector, ptr %118, i64 0, i32 2
  store double %122, ptr %126, align 8
  %127 = getelementptr %Vector, ptr %118, i64 0, i32 3
  store double %124, ptr %127, align 8
  %128 = load i64, ptr @Vector..type_descr, align 8
  %.not193 = icmp eq i64 %128, 0
  br i1 %.not193, label %129, label %132

129:                                              ; preds = %116
  %130 = alloca i64, align 8
  %131 = call i64 @GC_make_descriptor(ptr nonnull %130, i64 4)
  store i64 %131, ptr @Vector..type_descr, align 8
  br label %132

132:                                              ; preds = %129, %116
  %133 = phi i64 [ %131, %129 ], [ %128, %116 ]
  %134 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %133)
  store ptr @Vector..vtbl, ptr %134, align 8
  %135 = load double, ptr %30, align 8
  %136 = load double, ptr %125, align 8
  %137 = fsub double %135, %136
  %138 = load double, ptr %33, align 8
  %139 = load double, ptr %126, align 8
  %140 = fsub double %138, %139
  %141 = load double, ptr %36, align 8
  %142 = load double, ptr %127, align 8
  %143 = fsub double %141, %142
  %144 = getelementptr %Vector, ptr %134, i64 0, i32 1
  store double %137, ptr %144, align 8
  %145 = getelementptr %Vector, ptr %134, i64 0, i32 2
  store double %140, ptr %145, align 8
  %146 = getelementptr %Vector, ptr %134, i64 0, i32 3
  store double %143, ptr %146, align 8
  %147 = load ptr, ptr @Color.background, align 8
  %148 = load ptr, ptr %0, align 8
  %149 = getelementptr ptr, ptr %148, i64 7
  %150 = load ptr, ptr %149, align 8
  %151 = load ptr, ptr %74, align 8
  %152 = ptrtoint ptr %151 to i64
  %153 = add i64 %152, %6
  %154 = inttoptr i64 %153 to ptr
  %.unpack194 = load ptr, ptr %154, align 8
  %155 = insertvalue { ptr, ptr } poison, ptr %.unpack194, 0
  %.elt195 = getelementptr inbounds { ptr, ptr }, ptr %154, i64 0, i32 1
  %.unpack196 = load ptr, ptr %.elt195, align 8
  %156 = insertvalue { ptr, ptr } %155, ptr %.unpack196, 1
  %157 = call ptr %150(ptr nonnull %0, { ptr, ptr } %156, ptr nonnull %58, ptr nonnull %81, ptr nonnull %134, { ptr, ptr } %2)
  %158 = load i64, ptr @Color..type_descr, align 8
  %.not197 = icmp eq i64 %158, 0
  br i1 %.not197, label %159, label %162

159:                                              ; preds = %132
  %160 = alloca i64, align 8
  %161 = call i64 @GC_make_descriptor(ptr nonnull %160, i64 4)
  store i64 %161, ptr @Color..type_descr, align 8
  br label %162

162:                                              ; preds = %159, %132
  %163 = phi i64 [ %161, %159 ], [ %158, %132 ]
  %164 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %163)
  store ptr @Color..vtbl, ptr %164, align 8
  %165 = getelementptr %Color, ptr %147, i64 0, i32 1
  %166 = load double, ptr %165, align 8
  %167 = getelementptr %Color, ptr %157, i64 0, i32 1
  %168 = load double, ptr %167, align 8
  %169 = fadd double %166, %168
  %170 = getelementptr %Color, ptr %147, i64 0, i32 2
  %171 = load double, ptr %170, align 8
  %172 = getelementptr %Color, ptr %157, i64 0, i32 2
  %173 = load double, ptr %172, align 8
  %174 = fadd double %171, %173
  %175 = getelementptr %Color, ptr %147, i64 0, i32 3
  %176 = load double, ptr %175, align 8
  %177 = getelementptr %Color, ptr %157, i64 0, i32 3
  %178 = load double, ptr %177, align 8
  %179 = fadd double %176, %178
  %180 = getelementptr %Color, ptr %164, i64 0, i32 1
  store double %169, ptr %180, align 8
  %181 = getelementptr %Color, ptr %164, i64 0, i32 2
  store double %174, ptr %181, align 8
  %182 = getelementptr %Color, ptr %164, i64 0, i32 3
  store double %179, ptr %182, align 8
  %183 = getelementptr %RayTracer, ptr %0, i64 0, i32 1
  %184 = load i32, ptr %183, align 4
  %185 = sitofp i32 %184 to double
  %186 = fcmp ugt double %185, %3
  br i1 %186, label %189, label %187

187:                                              ; preds = %162
  %188 = load ptr, ptr @Color.grey, align 8
  br label %200

189:                                              ; preds = %162
  %190 = load ptr, ptr %0, align 8
  %191 = getelementptr ptr, ptr %190, i64 6
  %192 = load ptr, ptr %191, align 8
  %193 = load ptr, ptr %74, align 8
  %194 = ptrtoint ptr %193 to i64
  %195 = add i64 %194, %6
  %196 = inttoptr i64 %195 to ptr
  %.unpack198 = load ptr, ptr %196, align 8
  %197 = insertvalue { ptr, ptr } poison, ptr %.unpack198, 0
  %.elt199 = getelementptr inbounds { ptr, ptr }, ptr %196, i64 0, i32 1
  %.unpack200 = load ptr, ptr %.elt199, align 8
  %198 = insertvalue { ptr, ptr } %197, ptr %.unpack200, 1
  %199 = call ptr %192(ptr nonnull %0, { ptr, ptr } %198, ptr nonnull %58, ptr nonnull %81, ptr nonnull %134, { ptr, ptr } %2, double %3)
  br label %200

200:                                              ; preds = %187, %189
  %201 = phi ptr [ %188, %187 ], [ %199, %189 ]
  %202 = load i64, ptr @Color..type_descr, align 8
  %.not201 = icmp eq i64 %202, 0
  br i1 %.not201, label %203, label %206

203:                                              ; preds = %200
  %204 = alloca i64, align 8
  %205 = call i64 @GC_make_descriptor(ptr nonnull %204, i64 4)
  store i64 %205, ptr @Color..type_descr, align 8
  br label %206

206:                                              ; preds = %203, %200
  %207 = phi i64 [ %205, %203 ], [ %202, %200 ]
  %208 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %207)
  store ptr @Color..vtbl, ptr %208, align 8
  %209 = load double, ptr %180, align 8
  %210 = getelementptr %Color, ptr %201, i64 0, i32 1
  %211 = load double, ptr %210, align 8
  %212 = fadd double %209, %211
  %213 = load double, ptr %181, align 8
  %214 = getelementptr %Color, ptr %201, i64 0, i32 2
  %215 = load double, ptr %214, align 8
  %216 = fadd double %213, %215
  %217 = load double, ptr %182, align 8
  %218 = getelementptr %Color, ptr %201, i64 0, i32 3
  %219 = load double, ptr %218, align 8
  %220 = fadd double %217, %219
  %221 = getelementptr %Color, ptr %208, i64 0, i32 1
  store double %212, ptr %221, align 8
  %222 = getelementptr %Color, ptr %208, i64 0, i32 2
  store double %216, ptr %222, align 8
  %223 = getelementptr %Color, ptr %208, i64 0, i32 3
  store double %220, ptr %223, align 8
  ret ptr %208
}

define ptr @RayTracer.getReflectionColor(ptr %0, { ptr, ptr } %1, ptr %2, ptr nocapture readnone %3, ptr %4, { ptr, ptr } %5, double %6) {
  %.fca.0.extract = extractvalue { ptr, ptr } %1, 0
  %.fca.1.extract = extractvalue { ptr, ptr } %1, 1
  %8 = getelementptr ptr, ptr %.fca.0.extract, i64 2
  %9 = load ptr, ptr %8, align 8
  %10 = ptrtoint ptr %.fca.1.extract to i64
  %11 = ptrtoint ptr %9 to i64
  %12 = add i64 %11, %10
  %13 = inttoptr i64 %12 to ptr
  %.unpack = load ptr, ptr %13, align 8
  %.elt35 = getelementptr inbounds { ptr, ptr }, ptr %13, i64 0, i32 1
  %.unpack36 = load ptr, ptr %.elt35, align 8
  %14 = getelementptr ptr, ptr %.unpack, i64 2
  %15 = load ptr, ptr %14, align 8
  %16 = ptrtoint ptr %.unpack36 to i64
  %17 = ptrtoint ptr %15 to i64
  %18 = add i64 %17, %16
  %19 = inttoptr i64 %18 to ptr
  %20 = load ptr, ptr %19, align 8
  %21 = tail call double %20(ptr %.unpack36, ptr %2)
  %22 = load ptr, ptr %0, align 8
  %23 = getelementptr ptr, ptr %22, i64 4
  %24 = load ptr, ptr %23, align 8
  %25 = fadd double %6, 1.000000e+00
  %26 = tail call ptr @GC_malloc(i64 16)
  store ptr %2, ptr %26, align 8
  %.repack37 = getelementptr inbounds { ptr, ptr }, ptr %26, i64 0, i32 1
  store ptr %4, ptr %.repack37, align 8
  %27 = insertvalue { ptr, ptr } { ptr @Ray.14722407..vtbl, ptr undef }, ptr %26, 1
  %28 = tail call ptr %24(ptr nonnull %0, { ptr, ptr } %27, { ptr, ptr } %5, double %25)
  %29 = load i64, ptr @Color..type_descr, align 8
  %.not = icmp eq i64 %29, 0
  br i1 %.not, label %30, label %33

30:                                               ; preds = %7
  %31 = alloca i64, align 8
  %32 = call i64 @GC_make_descriptor(ptr nonnull %31, i64 4)
  store i64 %32, ptr @Color..type_descr, align 8
  br label %33

33:                                               ; preds = %30, %7
  %34 = phi i64 [ %32, %30 ], [ %29, %7 ]
  %35 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %34)
  store ptr @Color..vtbl, ptr %35, align 8
  %36 = getelementptr %Color, ptr %28, i64 0, i32 1
  %37 = load double, ptr %36, align 8
  %38 = fmul double %21, %37
  %39 = getelementptr %Color, ptr %28, i64 0, i32 2
  %40 = load double, ptr %39, align 8
  %41 = fmul double %21, %40
  %42 = getelementptr %Color, ptr %28, i64 0, i32 3
  %43 = load double, ptr %42, align 8
  %44 = fmul double %21, %43
  %45 = getelementptr %Color, ptr %35, i64 0, i32 1
  store double %38, ptr %45, align 8
  %46 = getelementptr %Color, ptr %35, i64 0, i32 2
  store double %41, ptr %46, align 8
  %47 = getelementptr %Color, ptr %35, i64 0, i32 3
  store double %44, ptr %47, align 8
  ret ptr %35
}

define ptr @.f_RayTracer.getNaturalColor..afL237C24L254C9H14864498(ptr nocapture readonly %0, ptr readonly %1, { ptr, ptr } %2) local_unnamed_addr {
  %.fca.0.extract = extractvalue { ptr, ptr } %2, 0
  %.fca.1.extract = extractvalue { ptr, ptr } %2, 1
  %4 = load ptr, ptr %0, align 8
  %5 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i64 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i64 0, i32 2
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i64 0, i32 3
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i64 0, i32 4
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i64 0, i32 5
  %14 = load ptr, ptr %13, align 8
  %15 = load ptr, ptr %.fca.0.extract, align 8
  %16 = ptrtoint ptr %.fca.1.extract to i64
  %17 = ptrtoint ptr %15 to i64
  %18 = add i64 %17, %16
  %19 = inttoptr i64 %18 to ptr
  %20 = load ptr, ptr %19, align 8
  %21 = load ptr, ptr %4, align 8
  %22 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %22, 0
  br i1 %.not, label %23, label %26

23:                                               ; preds = %3
  %24 = alloca i64, align 8
  %25 = call i64 @GC_make_descriptor(ptr nonnull %24, i64 4)
  store i64 %25, ptr @Vector..type_descr, align 8
  br label %26

26:                                               ; preds = %23, %3
  %27 = phi i64 [ %25, %23 ], [ %22, %3 ]
  %28 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %27)
  store ptr @Vector..vtbl, ptr %28, align 8
  %29 = getelementptr %Vector, ptr %20, i64 0, i32 1
  %30 = load double, ptr %29, align 8
  %31 = getelementptr %Vector, ptr %21, i64 0, i32 1
  %32 = load double, ptr %31, align 8
  %33 = fsub double %30, %32
  %34 = getelementptr %Vector, ptr %20, i64 0, i32 2
  %35 = load double, ptr %34, align 8
  %36 = getelementptr %Vector, ptr %21, i64 0, i32 2
  %37 = load double, ptr %36, align 8
  %38 = fsub double %35, %37
  %39 = getelementptr %Vector, ptr %20, i64 0, i32 3
  %40 = load double, ptr %39, align 8
  %41 = getelementptr %Vector, ptr %21, i64 0, i32 3
  %42 = load double, ptr %41, align 8
  %43 = fsub double %40, %42
  %44 = getelementptr %Vector, ptr %28, i64 0, i32 1
  store double %33, ptr %44, align 8
  %45 = getelementptr %Vector, ptr %28, i64 0, i32 2
  store double %38, ptr %45, align 8
  %46 = getelementptr %Vector, ptr %28, i64 0, i32 3
  store double %43, ptr %46, align 8
  %47 = call ptr @Vector.norm(ptr nonnull %28)
  %48 = load ptr, ptr %8, align 8
  %49 = load ptr, ptr %48, align 8
  %50 = getelementptr ptr, ptr %49, i64 3
  %51 = load ptr, ptr %50, align 8
  %52 = load ptr, ptr %4, align 8
  %.unpack = load ptr, ptr %14, align 8
  %53 = insertvalue { ptr, ptr } poison, ptr %.unpack, 0
  %.elt163 = getelementptr inbounds { ptr, ptr }, ptr %14, i64 0, i32 1
  %.unpack164 = load ptr, ptr %.elt163, align 8
  %54 = insertvalue { ptr, ptr } %53, ptr %.unpack164, 1
  %55 = call ptr @GC_malloc(i64 16)
  store ptr %52, ptr %55, align 8
  %.repack165 = getelementptr inbounds { ptr, ptr }, ptr %55, i64 0, i32 1
  store ptr %47, ptr %.repack165, align 8
  %56 = insertvalue { ptr, ptr } { ptr @Ray.14722407..vtbl, ptr undef }, ptr %55, 1
  %57 = call double %51(ptr nonnull %48, { ptr, ptr } %56, { ptr, ptr } %54)
  %58 = load double, ptr %44, align 8
  %59 = fmul double %58, %58
  %60 = load double, ptr %45, align 8
  %61 = fmul double %60, %60
  %62 = fadd double %59, %61
  %63 = load double, ptr %46, align 8
  %64 = fmul double %63, %63
  %65 = fadd double %62, %64
  %66 = call double @sqrt(double %65)
  %67 = fcmp ugt double %57, %66
  br i1 %67, label %68, label %299

68:                                               ; preds = %26
  %69 = load ptr, ptr %12, align 8
  %70 = getelementptr %Vector, ptr %47, i64 0, i32 1
  %71 = load double, ptr %70, align 8
  %72 = getelementptr %Vector, ptr %69, i64 0, i32 1
  %73 = load double, ptr %72, align 8
  %74 = fmul double %71, %73
  %75 = getelementptr %Vector, ptr %47, i64 0, i32 2
  %76 = load double, ptr %75, align 8
  %77 = getelementptr %Vector, ptr %69, i64 0, i32 2
  %78 = load double, ptr %77, align 8
  %79 = fmul double %76, %78
  %80 = fadd double %74, %79
  %81 = getelementptr %Vector, ptr %47, i64 0, i32 3
  %82 = load double, ptr %81, align 8
  %83 = getelementptr %Vector, ptr %69, i64 0, i32 3
  %84 = load double, ptr %83, align 8
  %85 = fmul double %82, %84
  %86 = fadd double %80, %85
  %87 = fcmp ogt double %86, 0.000000e+00
  br i1 %87, label %88, label %114

88:                                               ; preds = %68
  %89 = getelementptr ptr, ptr %.fca.0.extract, i64 1
  %90 = load ptr, ptr %89, align 8
  %91 = ptrtoint ptr %90 to i64
  %92 = add i64 %91, %16
  %93 = inttoptr i64 %92 to ptr
  %94 = load ptr, ptr %93, align 8
  %95 = load i64, ptr @Color..type_descr, align 8
  %.not190 = icmp eq i64 %95, 0
  br i1 %.not190, label %96, label %99

96:                                               ; preds = %88
  %97 = alloca i64, align 8
  %98 = call i64 @GC_make_descriptor(ptr nonnull %97, i64 4)
  store i64 %98, ptr @Color..type_descr, align 8
  br label %99

99:                                               ; preds = %96, %88
  %100 = phi i64 [ %98, %96 ], [ %95, %88 ]
  %101 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %100)
  store ptr @Color..vtbl, ptr %101, align 8
  %102 = getelementptr %Color, ptr %94, i64 0, i32 1
  %103 = load double, ptr %102, align 8
  %104 = fmul double %86, %103
  %105 = getelementptr %Color, ptr %94, i64 0, i32 2
  %106 = load double, ptr %105, align 8
  %107 = fmul double %86, %106
  %108 = getelementptr %Color, ptr %94, i64 0, i32 3
  %109 = load double, ptr %108, align 8
  %110 = fmul double %86, %109
  %111 = getelementptr %Color, ptr %101, i64 0, i32 1
  store double %104, ptr %111, align 8
  %112 = getelementptr %Color, ptr %101, i64 0, i32 2
  store double %107, ptr %112, align 8
  %113 = getelementptr %Color, ptr %101, i64 0, i32 3
  store double %110, ptr %113, align 8
  br label %116

114:                                              ; preds = %68
  %115 = load ptr, ptr @Color.defaultColor, align 8
  br label %116

116:                                              ; preds = %99, %114
  %117 = phi ptr [ %101, %99 ], [ %115, %114 ]
  %118 = load ptr, ptr %6, align 8
  %119 = call ptr @Vector.norm(ptr %118)
  %120 = load double, ptr %70, align 8
  %121 = getelementptr %Vector, ptr %119, i64 0, i32 1
  %122 = load double, ptr %121, align 8
  %123 = fmul double %120, %122
  %124 = load double, ptr %75, align 8
  %125 = getelementptr %Vector, ptr %119, i64 0, i32 2
  %126 = load double, ptr %125, align 8
  %127 = fmul double %124, %126
  %128 = fadd double %123, %127
  %129 = load double, ptr %81, align 8
  %130 = getelementptr %Vector, ptr %119, i64 0, i32 3
  %131 = load double, ptr %130, align 8
  %132 = fmul double %129, %131
  %133 = fadd double %128, %132
  %134 = fcmp ogt double %133, 0.000000e+00
  br i1 %134, label %135, label %175

135:                                              ; preds = %116
  %.unpack183 = load ptr, ptr %10, align 8
  %.elt184 = getelementptr inbounds { ptr, ptr }, ptr %10, i64 0, i32 1
  %.unpack185 = load ptr, ptr %.elt184, align 8
  %136 = getelementptr ptr, ptr %.unpack183, i64 2
  %137 = load ptr, ptr %136, align 8
  %138 = ptrtoint ptr %.unpack185 to i64
  %139 = ptrtoint ptr %137 to i64
  %140 = add i64 %139, %138
  %141 = inttoptr i64 %140 to ptr
  %.unpack186 = load ptr, ptr %141, align 8
  %.elt187 = getelementptr inbounds { ptr, ptr }, ptr %141, i64 0, i32 1
  %.unpack188 = load ptr, ptr %.elt187, align 8
  %142 = getelementptr ptr, ptr %.unpack186, i64 3
  %143 = load ptr, ptr %142, align 8
  %144 = ptrtoint ptr %.unpack188 to i64
  %145 = ptrtoint ptr %143 to i64
  %146 = add i64 %145, %144
  %147 = inttoptr i64 %146 to ptr
  %148 = load double, ptr %147, align 8
  %149 = call double @pow(double %133, double %148)
  %150 = getelementptr ptr, ptr %.fca.0.extract, i64 1
  %151 = load ptr, ptr %150, align 8
  %152 = ptrtoint ptr %151 to i64
  %153 = add i64 %152, %16
  %154 = inttoptr i64 %153 to ptr
  %155 = load ptr, ptr %154, align 8
  %156 = load i64, ptr @Color..type_descr, align 8
  %.not189 = icmp eq i64 %156, 0
  br i1 %.not189, label %157, label %160

157:                                              ; preds = %135
  %158 = alloca i64, align 8
  %159 = call i64 @GC_make_descriptor(ptr nonnull %158, i64 4)
  store i64 %159, ptr @Color..type_descr, align 8
  br label %160

160:                                              ; preds = %157, %135
  %161 = phi i64 [ %159, %157 ], [ %156, %135 ]
  %162 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %161)
  store ptr @Color..vtbl, ptr %162, align 8
  %163 = getelementptr %Color, ptr %155, i64 0, i32 1
  %164 = load double, ptr %163, align 8
  %165 = fmul double %149, %164
  %166 = getelementptr %Color, ptr %155, i64 0, i32 2
  %167 = load double, ptr %166, align 8
  %168 = fmul double %149, %167
  %169 = getelementptr %Color, ptr %155, i64 0, i32 3
  %170 = load double, ptr %169, align 8
  %171 = fmul double %149, %170
  %172 = getelementptr %Color, ptr %162, i64 0, i32 1
  store double %165, ptr %172, align 8
  %173 = getelementptr %Color, ptr %162, i64 0, i32 2
  store double %168, ptr %173, align 8
  %174 = getelementptr %Color, ptr %162, i64 0, i32 3
  store double %171, ptr %174, align 8
  br label %177

175:                                              ; preds = %116
  %176 = load ptr, ptr @Color.defaultColor, align 8
  br label %177

177:                                              ; preds = %160, %175
  %178 = phi ptr [ %162, %160 ], [ %176, %175 ]
  %.unpack167 = load ptr, ptr %10, align 8
  %.elt168 = getelementptr inbounds { ptr, ptr }, ptr %10, i64 0, i32 1
  %.unpack169 = load ptr, ptr %.elt168, align 8
  %179 = getelementptr ptr, ptr %.unpack167, i64 2
  %180 = load ptr, ptr %179, align 8
  %181 = ptrtoint ptr %.unpack169 to i64
  %182 = ptrtoint ptr %180 to i64
  %183 = add i64 %182, %181
  %184 = inttoptr i64 %183 to ptr
  %.unpack170 = load ptr, ptr %184, align 8
  %.elt171 = getelementptr inbounds { ptr, ptr }, ptr %184, i64 0, i32 1
  %.unpack172 = load ptr, ptr %.elt171, align 8
  %185 = load ptr, ptr %.unpack170, align 8
  %186 = ptrtoint ptr %.unpack172 to i64
  %187 = ptrtoint ptr %185 to i64
  %188 = add i64 %187, %186
  %189 = inttoptr i64 %188 to ptr
  %190 = load ptr, ptr %189, align 8
  %191 = load ptr, ptr %4, align 8
  %192 = call ptr %190(ptr %.unpack172, ptr %191)
  %193 = load i64, ptr @Color..type_descr, align 8
  %.not173 = icmp eq i64 %193, 0
  br i1 %.not173, label %194, label %197

194:                                              ; preds = %177
  %195 = alloca i64, align 8
  %196 = call i64 @GC_make_descriptor(ptr nonnull %195, i64 4)
  store i64 %196, ptr @Color..type_descr, align 8
  br label %197

197:                                              ; preds = %194, %177
  %198 = phi i64 [ %196, %194 ], [ %193, %177 ]
  %199 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %198)
  store ptr @Color..vtbl, ptr %199, align 8
  %200 = getelementptr %Color, ptr %192, i64 0, i32 1
  %201 = load double, ptr %200, align 8
  %202 = getelementptr %Color, ptr %117, i64 0, i32 1
  %203 = load double, ptr %202, align 8
  %204 = fmul double %201, %203
  %205 = getelementptr %Color, ptr %192, i64 0, i32 2
  %206 = load double, ptr %205, align 8
  %207 = getelementptr %Color, ptr %117, i64 0, i32 2
  %208 = load double, ptr %207, align 8
  %209 = fmul double %206, %208
  %210 = getelementptr %Color, ptr %192, i64 0, i32 3
  %211 = load double, ptr %210, align 8
  %212 = getelementptr %Color, ptr %117, i64 0, i32 3
  %213 = load double, ptr %212, align 8
  %214 = fmul double %211, %213
  %215 = getelementptr %Color, ptr %199, i64 0, i32 1
  store double %204, ptr %215, align 8
  %216 = getelementptr %Color, ptr %199, i64 0, i32 2
  store double %209, ptr %216, align 8
  %217 = getelementptr %Color, ptr %199, i64 0, i32 3
  store double %214, ptr %217, align 8
  %.unpack174 = load ptr, ptr %10, align 8
  %.unpack176 = load ptr, ptr %.elt168, align 8
  %218 = getelementptr ptr, ptr %.unpack174, i64 2
  %219 = load ptr, ptr %218, align 8
  %220 = ptrtoint ptr %.unpack176 to i64
  %221 = ptrtoint ptr %219 to i64
  %222 = add i64 %221, %220
  %223 = inttoptr i64 %222 to ptr
  %.unpack177 = load ptr, ptr %223, align 8
  %.elt178 = getelementptr inbounds { ptr, ptr }, ptr %223, i64 0, i32 1
  %.unpack179 = load ptr, ptr %.elt178, align 8
  %224 = getelementptr ptr, ptr %.unpack177, i64 1
  %225 = load ptr, ptr %224, align 8
  %226 = ptrtoint ptr %.unpack179 to i64
  %227 = ptrtoint ptr %225 to i64
  %228 = add i64 %227, %226
  %229 = inttoptr i64 %228 to ptr
  %230 = load ptr, ptr %229, align 8
  %231 = load ptr, ptr %4, align 8
  %232 = call ptr %230(ptr %.unpack179, ptr %231)
  %233 = load i64, ptr @Color..type_descr, align 8
  %.not180 = icmp eq i64 %233, 0
  br i1 %.not180, label %234, label %237

234:                                              ; preds = %197
  %235 = alloca i64, align 8
  %236 = call i64 @GC_make_descriptor(ptr nonnull %235, i64 4)
  store i64 %236, ptr @Color..type_descr, align 8
  br label %237

237:                                              ; preds = %234, %197
  %238 = phi i64 [ %236, %234 ], [ %233, %197 ]
  %239 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %238)
  store ptr @Color..vtbl, ptr %239, align 8
  %240 = getelementptr %Color, ptr %232, i64 0, i32 1
  %241 = load double, ptr %240, align 8
  %242 = getelementptr %Color, ptr %178, i64 0, i32 1
  %243 = load double, ptr %242, align 8
  %244 = fmul double %241, %243
  %245 = getelementptr %Color, ptr %232, i64 0, i32 2
  %246 = load double, ptr %245, align 8
  %247 = getelementptr %Color, ptr %178, i64 0, i32 2
  %248 = load double, ptr %247, align 8
  %249 = fmul double %246, %248
  %250 = getelementptr %Color, ptr %232, i64 0, i32 3
  %251 = load double, ptr %250, align 8
  %252 = getelementptr %Color, ptr %178, i64 0, i32 3
  %253 = load double, ptr %252, align 8
  %254 = fmul double %251, %253
  %255 = getelementptr %Color, ptr %239, i64 0, i32 1
  store double %244, ptr %255, align 8
  %256 = getelementptr %Color, ptr %239, i64 0, i32 2
  store double %249, ptr %256, align 8
  %257 = getelementptr %Color, ptr %239, i64 0, i32 3
  store double %254, ptr %257, align 8
  %258 = load i64, ptr @Color..type_descr, align 8
  %.not181 = icmp eq i64 %258, 0
  br i1 %.not181, label %259, label %262

259:                                              ; preds = %237
  %260 = alloca i64, align 8
  %261 = call i64 @GC_make_descriptor(ptr nonnull %260, i64 4)
  store i64 %261, ptr @Color..type_descr, align 8
  br label %262

262:                                              ; preds = %259, %237
  %263 = phi i64 [ %261, %259 ], [ %258, %237 ]
  %264 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %263)
  store ptr @Color..vtbl, ptr %264, align 8
  %265 = load double, ptr %215, align 8
  %266 = load double, ptr %255, align 8
  %267 = fadd double %265, %266
  %268 = load double, ptr %216, align 8
  %269 = load double, ptr %256, align 8
  %270 = fadd double %268, %269
  %271 = load double, ptr %217, align 8
  %272 = load double, ptr %257, align 8
  %273 = fadd double %271, %272
  %274 = getelementptr %Color, ptr %264, i64 0, i32 1
  store double %267, ptr %274, align 8
  %275 = getelementptr %Color, ptr %264, i64 0, i32 2
  store double %270, ptr %275, align 8
  %276 = getelementptr %Color, ptr %264, i64 0, i32 3
  store double %273, ptr %276, align 8
  %277 = load i64, ptr @Color..type_descr, align 8
  %.not182 = icmp eq i64 %277, 0
  br i1 %.not182, label %278, label %281

278:                                              ; preds = %262
  %279 = alloca i64, align 8
  %280 = call i64 @GC_make_descriptor(ptr nonnull %279, i64 4)
  store i64 %280, ptr @Color..type_descr, align 8
  br label %281

281:                                              ; preds = %278, %262
  %282 = phi i64 [ %280, %278 ], [ %277, %262 ]
  %283 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %282)
  store ptr @Color..vtbl, ptr %283, align 8
  %284 = getelementptr %Color, ptr %1, i64 0, i32 1
  %285 = load double, ptr %284, align 8
  %286 = load double, ptr %274, align 8
  %287 = fadd double %285, %286
  %288 = getelementptr %Color, ptr %1, i64 0, i32 2
  %289 = load double, ptr %288, align 8
  %290 = load double, ptr %275, align 8
  %291 = fadd double %289, %290
  %292 = getelementptr %Color, ptr %1, i64 0, i32 3
  %293 = load double, ptr %292, align 8
  %294 = load double, ptr %276, align 8
  %295 = fadd double %293, %294
  %296 = getelementptr %Color, ptr %283, i64 0, i32 1
  store double %287, ptr %296, align 8
  %297 = getelementptr %Color, ptr %283, i64 0, i32 2
  store double %291, ptr %297, align 8
  %298 = getelementptr %Color, ptr %283, i64 0, i32 3
  store double %295, ptr %298, align 8
  br label %299

299:                                              ; preds = %26, %281
  %.0 = phi ptr [ %283, %281 ], [ %1, %26 ]
  ret ptr %.0
}

define ptr @RayTracer.getNaturalColor(ptr %0, { ptr, ptr } %1, ptr %2, ptr %3, ptr %4, { ptr, ptr } %5) {
  %7 = tail call ptr @GC_malloc(i64 16)
  store ptr %0, ptr %7, align 8
  %8 = tail call ptr @GC_malloc(i64 16)
  %.elt = extractvalue { ptr, ptr } %1, 0
  store ptr %.elt, ptr %8, align 8
  %.repack8 = getelementptr inbounds { ptr, ptr }, ptr %8, i64 0, i32 1
  %.elt9 = extractvalue { ptr, ptr } %1, 1
  store ptr %.elt9, ptr %.repack8, align 8
  %9 = tail call ptr @GC_malloc(i64 32)
  store ptr %2, ptr %9, align 8
  %10 = tail call ptr @GC_malloc(i64 32)
  store ptr %3, ptr %10, align 8
  %11 = tail call ptr @GC_malloc(i64 32)
  store ptr %4, ptr %11, align 8
  %12 = tail call ptr @GC_malloc(i64 16)
  %.elt10 = extractvalue { ptr, ptr } %5, 0
  store ptr %.elt10, ptr %12, align 8
  %.repack11 = getelementptr inbounds { ptr, ptr }, ptr %12, i64 0, i32 1
  %.elt12 = extractvalue { ptr, ptr } %5, 1
  store ptr %.elt12, ptr %.repack11, align 8
  %13 = tail call ptr @GC_malloc(i64 48)
  store ptr %9, ptr %13, align 8
  %14 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %13, i64 0, i32 1
  store ptr %11, ptr %14, align 8
  %15 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %13, i64 0, i32 2
  store ptr %7, ptr %15, align 8
  %16 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %13, i64 0, i32 3
  store ptr %8, ptr %16, align 8
  %17 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %13, i64 0, i32 4
  store ptr %10, ptr %17, align 8
  %18 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %13, i64 0, i32 5
  store ptr %12, ptr %18, align 8
  %19 = load ptr, ptr @Color.defaultColor, align 8
  %.unpack = load ptr, ptr %12, align 8
  %.unpack15 = load ptr, ptr %.repack11, align 8
  %20 = getelementptr ptr, ptr %.unpack, i64 1
  %21 = load ptr, ptr %20, align 8
  %22 = ptrtoint ptr %.unpack15 to i64
  %23 = ptrtoint ptr %21 to i64
  %24 = add i64 %23, %22
  %25 = inttoptr i64 %24 to ptr
  %26 = load { ptr, i32 }, ptr %25, align 8
  %27 = extractvalue { ptr, i32 } %26, 1
  %28 = icmp sgt i32 %27, 0
  br i1 %28, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %6, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %6 ]
  %29 = phi { ptr, i32 } [ %35, %.lr.ph ], [ %26, %6 ]
  %.0720 = phi ptr [ %34, %.lr.ph ], [ %19, %6 ]
  %30 = extractvalue { ptr, i32 } %29, 0
  %31 = getelementptr { ptr, ptr }, ptr %30, i64 %indvars.iv
  %.unpack17 = load ptr, ptr %31, align 8
  %32 = insertvalue { ptr, ptr } poison, ptr %.unpack17, 0
  %.elt18 = getelementptr { ptr, ptr }, ptr %30, i64 %indvars.iv, i32 1
  %.unpack19 = load ptr, ptr %.elt18, align 8
  %33 = insertvalue { ptr, ptr } %32, ptr %.unpack19, 1
  %34 = tail call ptr @.f_RayTracer.getNaturalColor..afL237C24L254C9H14864498(ptr nonnull %13, ptr %.0720, { ptr, ptr } %33)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %35 = load { ptr, i32 }, ptr %25, align 8
  %36 = extractvalue { ptr, i32 } %35, 1
  %37 = sext i32 %36 to i64
  %38 = icmp slt i64 %indvars.iv.next, %37
  br i1 %38, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %6
  %.07.lcssa = phi ptr [ %19, %6 ], [ %34, %.lr.ph ]
  ret ptr %.07.lcssa
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none)
define double @.f_RayTracer.render..f_.afL265C24L276C9H14728317..afL266C29L266C88H14723962(ptr nocapture readonly %0, double %1) local_unnamed_addr #12 {
  %3 = load ptr, ptr %0, align 8
  %4 = load double, ptr %3, align 8
  %5 = fmul double %4, 5.000000e-01
  %6 = fsub double %1, %5
  %7 = fmul double %6, 5.000000e-01
  %8 = fdiv double %7, %4
  ret double %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none)
define double @.f_RayTracer.render..f_.afL265C24L276C9H14728317..afL267C29L267C92H14721256(ptr nocapture readonly %0, double %1) local_unnamed_addr #12 {
  %3 = load ptr, ptr %0, align 8
  %4 = load double, ptr %3, align 8
  %5 = fmul double %4, 5.000000e-01
  %6 = fsub double %5, %1
  %7 = fadd double %6, 0.000000e+00
  %8 = fmul double %7, 5.000000e-01
  %9 = fdiv double %8, %4
  ret double %9
}

define ptr @.f_RayTracer.render..afL265C24L276C9H14728317(ptr nocapture readonly %0, double %1, double %2, ptr nocapture readonly %3) local_unnamed_addr {
  %5 = load ptr, ptr %0, align 8
  %6 = getelementptr { ptr, ptr }, ptr %0, i64 0, i32 1
  %7 = load ptr, ptr %6, align 8
  %8 = tail call ptr @GC_malloc(i64 8)
  store ptr %5, ptr %8, align 8
  %9 = tail call ptr @GC_malloc(i64 8)
  store ptr %7, ptr %9, align 8
  %10 = getelementptr %Camera, ptr %3, i64 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = load ptr, ptr %8, align 8
  %13 = load double, ptr %12, align 8
  %14 = fmul double %13, 5.000000e-01
  %15 = fsub double %1, %14
  %16 = fmul double %15, 5.000000e-01
  %17 = fdiv double %16, %13
  %18 = getelementptr %Camera, ptr %3, i64 0, i32 2
  %19 = load ptr, ptr %18, align 8
  %20 = load i64, ptr @Vector..type_descr, align 8
  %.not = icmp eq i64 %20, 0
  br i1 %.not, label %21, label %24

21:                                               ; preds = %4
  %22 = alloca i64, align 8
  %23 = call i64 @GC_make_descriptor(ptr nonnull %22, i64 4)
  store i64 %23, ptr @Vector..type_descr, align 8
  br label %24

24:                                               ; preds = %21, %4
  %25 = phi i64 [ %23, %21 ], [ %20, %4 ]
  %26 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %25)
  store ptr @Vector..vtbl, ptr %26, align 8
  %27 = getelementptr %Vector, ptr %19, i64 0, i32 1
  %28 = load double, ptr %27, align 8
  %29 = fmul double %17, %28
  %30 = getelementptr %Vector, ptr %19, i64 0, i32 2
  %31 = load double, ptr %30, align 8
  %32 = fmul double %17, %31
  %33 = getelementptr %Vector, ptr %19, i64 0, i32 3
  %34 = load double, ptr %33, align 8
  %35 = fmul double %17, %34
  %36 = getelementptr %Vector, ptr %26, i64 0, i32 1
  store double %29, ptr %36, align 8
  %37 = getelementptr %Vector, ptr %26, i64 0, i32 2
  store double %32, ptr %37, align 8
  %38 = getelementptr %Vector, ptr %26, i64 0, i32 3
  store double %35, ptr %38, align 8
  %39 = load ptr, ptr %9, align 8
  %40 = load double, ptr %39, align 8
  %41 = fmul double %40, 5.000000e-01
  %42 = fsub double %41, %2
  %43 = fadd double %42, 0.000000e+00
  %44 = fmul double %43, 5.000000e-01
  %45 = fdiv double %44, %40
  %46 = getelementptr %Camera, ptr %3, i64 0, i32 3
  %47 = load ptr, ptr %46, align 8
  %48 = load i64, ptr @Vector..type_descr, align 8
  %.not78 = icmp eq i64 %48, 0
  br i1 %.not78, label %49, label %52

49:                                               ; preds = %24
  %50 = alloca i64, align 8
  %51 = call i64 @GC_make_descriptor(ptr nonnull %50, i64 4)
  store i64 %51, ptr @Vector..type_descr, align 8
  br label %52

52:                                               ; preds = %49, %24
  %53 = phi i64 [ %51, %49 ], [ %48, %24 ]
  %54 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %53)
  store ptr @Vector..vtbl, ptr %54, align 8
  %55 = getelementptr %Vector, ptr %47, i64 0, i32 1
  %56 = load double, ptr %55, align 8
  %57 = fmul double %45, %56
  %58 = getelementptr %Vector, ptr %47, i64 0, i32 2
  %59 = load double, ptr %58, align 8
  %60 = fmul double %45, %59
  %61 = getelementptr %Vector, ptr %47, i64 0, i32 3
  %62 = load double, ptr %61, align 8
  %63 = fmul double %45, %62
  %64 = getelementptr %Vector, ptr %54, i64 0, i32 1
  store double %57, ptr %64, align 8
  %65 = getelementptr %Vector, ptr %54, i64 0, i32 2
  store double %60, ptr %65, align 8
  %66 = getelementptr %Vector, ptr %54, i64 0, i32 3
  store double %63, ptr %66, align 8
  %67 = load i64, ptr @Vector..type_descr, align 8
  %.not79 = icmp eq i64 %67, 0
  br i1 %.not79, label %68, label %71

68:                                               ; preds = %52
  %69 = alloca i64, align 8
  %70 = call i64 @GC_make_descriptor(ptr nonnull %69, i64 4)
  store i64 %70, ptr @Vector..type_descr, align 8
  br label %71

71:                                               ; preds = %68, %52
  %72 = phi i64 [ %70, %68 ], [ %67, %52 ]
  %73 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %72)
  store ptr @Vector..vtbl, ptr %73, align 8
  %74 = load double, ptr %36, align 8
  %75 = load double, ptr %64, align 8
  %76 = fadd double %74, %75
  %77 = load double, ptr %37, align 8
  %78 = load double, ptr %65, align 8
  %79 = fadd double %77, %78
  %80 = load double, ptr %38, align 8
  %81 = load double, ptr %66, align 8
  %82 = fadd double %80, %81
  %83 = getelementptr %Vector, ptr %73, i64 0, i32 1
  store double %76, ptr %83, align 8
  %84 = getelementptr %Vector, ptr %73, i64 0, i32 2
  store double %79, ptr %84, align 8
  %85 = getelementptr %Vector, ptr %73, i64 0, i32 3
  store double %82, ptr %85, align 8
  %86 = load i64, ptr @Vector..type_descr, align 8
  %.not80 = icmp eq i64 %86, 0
  br i1 %.not80, label %87, label %90

87:                                               ; preds = %71
  %88 = alloca i64, align 8
  %89 = call i64 @GC_make_descriptor(ptr nonnull %88, i64 4)
  store i64 %89, ptr @Vector..type_descr, align 8
  br label %90

90:                                               ; preds = %87, %71
  %91 = phi i64 [ %89, %87 ], [ %86, %71 ]
  %92 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %91)
  store ptr @Vector..vtbl, ptr %92, align 8
  %93 = getelementptr %Vector, ptr %11, i64 0, i32 1
  %94 = load double, ptr %93, align 8
  %95 = load double, ptr %83, align 8
  %96 = fadd double %94, %95
  %97 = getelementptr %Vector, ptr %11, i64 0, i32 2
  %98 = load double, ptr %97, align 8
  %99 = load double, ptr %84, align 8
  %100 = fadd double %98, %99
  %101 = getelementptr %Vector, ptr %11, i64 0, i32 3
  %102 = load double, ptr %101, align 8
  %103 = load double, ptr %85, align 8
  %104 = fadd double %102, %103
  %105 = getelementptr %Vector, ptr %92, i64 0, i32 1
  store double %96, ptr %105, align 8
  %106 = getelementptr %Vector, ptr %92, i64 0, i32 2
  store double %100, ptr %106, align 8
  %107 = getelementptr %Vector, ptr %92, i64 0, i32 3
  store double %104, ptr %107, align 8
  %108 = call ptr @Vector.norm(ptr nonnull %92)
  ret ptr %108
}

define void @RayTracer.render(ptr %0, { ptr, ptr } %1, double %2, double %3) {
  %5 = tail call ptr @GC_malloc(i64 8)
  store double %2, ptr %5, align 8
  %6 = tail call ptr @GC_malloc(i64 8)
  store double %3, ptr %6, align 8
  %7 = tail call ptr @GC_malloc(i64 16)
  store ptr %5, ptr %7, align 8
  %8 = getelementptr { ptr, ptr }, ptr %7, i64 0, i32 1
  store ptr %6, ptr %8, align 8
  %9 = load double, ptr %6, align 8
  %10 = fcmp ogt double %9, 0.000000e+00
  br i1 %10, label %.preheader.lr.ph, label %._crit_edge55

.preheader.lr.ph:                                 ; preds = %4
  %.fca.1.extract = extractvalue { ptr, ptr } %1, 1
  %.fca.0.extract = extractvalue { ptr, ptr } %1, 0
  %11 = getelementptr ptr, ptr %.fca.0.extract, i64 2
  %12 = ptrtoint ptr %.fca.1.extract to i64
  %13 = load double, ptr %5, align 8
  %14 = fcmp ogt double %13, 0.000000e+00
  br i1 %14, label %.preheader, label %.preheader.us

.preheader.us:                                    ; preds = %.preheader.lr.ph, %.preheader.us
  %.04054.us = phi i32 [ %15, %.preheader.us ], [ 0, %.preheader.lr.ph ]
  %15 = add i32 %.04054.us, 1
  %16 = sitofp i32 %15 to double
  %17 = fcmp ogt double %9, %16
  br i1 %17, label %.preheader.us, label %._crit_edge55

.preheader:                                       ; preds = %.preheader.lr.ph, %._crit_edge
  %18 = phi double [ %86, %._crit_edge ], [ %9, %.preheader.lr.ph ]
  %19 = phi double [ %87, %._crit_edge ], [ %13, %.preheader.lr.ph ]
  %20 = phi double [ %89, %._crit_edge ], [ 0.000000e+00, %.preheader.lr.ph ]
  %.04054 = phi i32 [ %88, %._crit_edge ], [ 0, %.preheader.lr.ph ]
  %21 = fcmp ogt double %19, 0.000000e+00
  br i1 %21, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.preheader, %.lr.ph
  %22 = phi double [ %84, %.lr.ph ], [ 0.000000e+00, %.preheader ]
  %.053 = phi i32 [ %82, %.lr.ph ], [ 0, %.preheader ]
  %23 = load ptr, ptr %0, align 8
  %24 = getelementptr ptr, ptr %23, i64 4
  %25 = load ptr, ptr %24, align 8
  %26 = load ptr, ptr %11, align 8
  %27 = ptrtoint ptr %26 to i64
  %28 = add i64 %27, %12
  %29 = inttoptr i64 %28 to ptr
  %30 = load ptr, ptr %29, align 8
  %31 = getelementptr %Camera, ptr %30, i64 0, i32 4
  %32 = load ptr, ptr %31, align 8
  %33 = tail call ptr @.f_RayTracer.render..afL265C24L276C9H14728317(ptr nonnull %7, double %22, double %20, ptr %30)
  %34 = tail call ptr @GC_malloc(i64 16)
  store ptr %32, ptr %34, align 8
  %.repack43 = getelementptr inbounds { ptr, ptr }, ptr %34, i64 0, i32 1
  store ptr %33, ptr %.repack43, align 8
  %35 = insertvalue { ptr, ptr } { ptr @Ray.14722407..vtbl, ptr undef }, ptr %34, 1
  %36 = tail call ptr %25(ptr nonnull %0, { ptr, ptr } %35, { ptr, ptr } %1, double 0.000000e+00)
  %37 = getelementptr %Color, ptr %36, i64 0, i32 1
  %38 = load double, ptr %37, align 8
  %39 = fcmp ogt double %38, 1.000000e+00
  %40 = select i1 %39, double 1.000000e+00, double %38
  %41 = fmul double %40, 2.550000e+02
  %42 = tail call double @llvm.floor.f64(double %41)
  %43 = getelementptr %Color, ptr %36, i64 0, i32 2
  %44 = load double, ptr %43, align 8
  %45 = fcmp ogt double %44, 1.000000e+00
  %46 = select i1 %45, double 1.000000e+00, double %44
  %47 = fmul double %46, 2.550000e+02
  %48 = tail call double @llvm.floor.f64(double %47)
  %49 = getelementptr %Color, ptr %36, i64 0, i32 3
  %50 = load double, ptr %49, align 8
  %51 = fcmp ogt double %50, 1.000000e+00
  %52 = select i1 %51, double 1.000000e+00, double %50
  %53 = fmul double %52, 2.550000e+02
  %54 = tail call double @llvm.floor.f64(double %53)
  %55 = tail call ptr @GC_malloc(i64 50)
  %56 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %55, i32 50, ptr nonnull @frmt_555400739678143724, i32 %.053)
  %57 = tail call ptr @GC_malloc(i64 50)
  %58 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %57, i32 50, ptr nonnull @frmt_555400739678143724, i32 %.04054)
  %59 = tail call ptr @GC_malloc(i64 50)
  %60 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %59, i32 50, ptr nonnull @frmt_555404038213028357, double %42)
  %61 = tail call ptr @GC_malloc(i64 50)
  %62 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %61, i32 50, ptr nonnull @frmt_555404038213028357, double %48)
  %63 = tail call ptr @GC_malloc(i64 50)
  %64 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %63, i32 50, ptr nonnull @frmt_555404038213028357, double %54)
  %65 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %55)
  %66 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %57)
  %67 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %59)
  %68 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %61)
  %69 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %63)
  %70 = add i64 %65, 62
  %71 = add i64 %70, %66
  %72 = add i64 %71, %67
  %73 = add i64 %72, %68
  %74 = add i64 %73, %69
  %75 = tail call ptr @GC_malloc(i64 %74)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %75, ptr noundef nonnull align 1 dereferenceable(10) @s_1937339058070584390, i64 10, i1 false)
  %76 = tail call ptr @strcat(ptr noundef nonnull dereferenceable(1) %75, ptr noundef nonnull dereferenceable(1) %55)
  %strlen = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %75)
  %endptr = getelementptr inbounds i8, ptr %75, i64 %strlen
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %endptr, ptr noundef nonnull align 1 dereferenceable(6) @s_11894988290003604653, i64 6, i1 false)
  %77 = tail call ptr @strcat(ptr noundef nonnull dereferenceable(1) %75, ptr noundef nonnull dereferenceable(1) %57)
  %strlen45 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %75)
  %endptr46 = getelementptr inbounds i8, ptr %75, i64 %strlen45
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(40) %endptr46, ptr noundef nonnull align 16 dereferenceable(40) @s_8482466337658167522, i64 40, i1 false)
  %78 = tail call ptr @strcat(ptr noundef nonnull dereferenceable(1) %75, ptr noundef nonnull dereferenceable(1) %59)
  %strlen47 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %75)
  %endptr48 = getelementptr inbounds i8, ptr %75, i64 %strlen47
  store i16 44, ptr %endptr48, align 1
  %79 = tail call ptr @strcat(ptr noundef nonnull dereferenceable(1) %75, ptr noundef nonnull dereferenceable(1) %61)
  %strlen49 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %75)
  %endptr50 = getelementptr inbounds i8, ptr %75, i64 %strlen49
  store i16 44, ptr %endptr50, align 1
  %80 = tail call ptr @strcat(ptr noundef nonnull dereferenceable(1) %75, ptr noundef nonnull dereferenceable(1) %63)
  %strlen51 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %75)
  %endptr52 = getelementptr inbounds i8, ptr %75, i64 %strlen51
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %endptr52, ptr noundef nonnull align 1 dereferenceable(7) @s_4155466971959723698, i64 7, i1 false)
  %81 = tail call i32 @puts(ptr nonnull dereferenceable(1) %75)
  %82 = add i32 %.053, 1
  %83 = load double, ptr %5, align 8
  %84 = sitofp i32 %82 to double
  %85 = fcmp ogt double %83, %84
  br i1 %85, label %.lr.ph, label %._crit_edge.loopexit

._crit_edge.loopexit:                             ; preds = %.lr.ph
  %.pre = load double, ptr %6, align 8
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %.preheader
  %86 = phi double [ %.pre, %._crit_edge.loopexit ], [ %18, %.preheader ]
  %87 = phi double [ %83, %._crit_edge.loopexit ], [ %19, %.preheader ]
  %88 = add i32 %.04054, 1
  %89 = sitofp i32 %88 to double
  %90 = fcmp ogt double %86, %89
  br i1 %90, label %.preheader, label %._crit_edge55, !llvm.loop !0

._crit_edge55:                                    ; preds = %.preheader.us, %._crit_edge, %4
  ret void
}

define { ptr, ptr } @defaultScene() local_unnamed_addr {
  %1 = load i64, ptr @Plane..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 5)
  store i64 %4, ptr @Plane..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %6)
  store ptr @Plane..vtbl, ptr %7, align 8
  %8 = load i64, ptr @Vector..type_descr, align 8
  %.not229 = icmp eq i64 %8, 0
  br i1 %.not229, label %9, label %12

9:                                                ; preds = %5
  %10 = alloca i64, align 8
  %11 = call i64 @GC_make_descriptor(ptr nonnull %10, i64 4)
  store i64 %11, ptr @Vector..type_descr, align 8
  br label %12

12:                                               ; preds = %9, %5
  %13 = phi i64 [ %11, %9 ], [ %8, %5 ]
  %14 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %13)
  store ptr @Vector..vtbl, ptr %14, align 8
  %15 = getelementptr %Vector, ptr %14, i64 0, i32 1
  store double 0.000000e+00, ptr %15, align 8
  %16 = getelementptr %Vector, ptr %14, i64 0, i32 2
  store double 1.000000e+00, ptr %16, align 8
  %17 = getelementptr %Vector, ptr %14, i64 0, i32 3
  store double 0.000000e+00, ptr %17, align 8
  %.unpack = load ptr, ptr @Surfaces.checkerboard, align 8
  %.unpack230 = load ptr, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.checkerboard, i64 0, i32 1), align 8
  %18 = getelementptr %Plane, ptr %7, i64 0, i32 1
  store ptr %14, ptr %18, align 8
  %19 = getelementptr %Plane, ptr %7, i64 0, i32 2
  store double 0.000000e+00, ptr %19, align 8
  %20 = getelementptr %Plane, ptr %7, i64 0, i32 3
  store ptr %.unpack, ptr %20, align 8
  %.repack231 = getelementptr %Plane, ptr %7, i64 0, i32 3, i32 1
  store ptr %.unpack230, ptr %.repack231, align 8
  %21 = load ptr, ptr %7, align 8
  %22 = load ptr, ptr %21, align 8
  %23 = load i64, ptr @Sphere..type_descr, align 8
  %.not233 = icmp eq i64 %23, 0
  br i1 %.not233, label %24, label %27

24:                                               ; preds = %12
  %25 = alloca i64, align 8
  %26 = call i64 @GC_make_descriptor(ptr nonnull %25, i64 5)
  store i64 %26, ptr @Sphere..type_descr, align 8
  br label %27

27:                                               ; preds = %24, %12
  %28 = phi i64 [ %26, %24 ], [ %23, %12 ]
  %29 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %28)
  store ptr @Sphere..vtbl, ptr %29, align 8
  %30 = load i64, ptr @Vector..type_descr, align 8
  %.not234 = icmp eq i64 %30, 0
  br i1 %.not234, label %31, label %34

31:                                               ; preds = %27
  %32 = alloca i64, align 8
  %33 = call i64 @GC_make_descriptor(ptr nonnull %32, i64 4)
  store i64 %33, ptr @Vector..type_descr, align 8
  br label %34

34:                                               ; preds = %31, %27
  %35 = phi i64 [ %33, %31 ], [ %30, %27 ]
  %36 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %35)
  store ptr @Vector..vtbl, ptr %36, align 8
  %37 = getelementptr %Vector, ptr %36, i64 0, i32 1
  store double 0.000000e+00, ptr %37, align 8
  %38 = getelementptr %Vector, ptr %36, i64 0, i32 2
  store double 1.000000e+00, ptr %38, align 8
  %39 = getelementptr %Vector, ptr %36, i64 0, i32 3
  store double -2.500000e-01, ptr %39, align 8
  %.unpack235 = load ptr, ptr @Surfaces.shiny, align 8
  %.unpack236 = load ptr, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.shiny, i64 0, i32 1), align 8
  %40 = getelementptr %Sphere, ptr %29, i64 0, i32 2
  store ptr %36, ptr %40, align 8
  %41 = getelementptr %Sphere, ptr %29, i64 0, i32 3
  store ptr %.unpack235, ptr %41, align 8
  %.repack237 = getelementptr %Sphere, ptr %29, i64 0, i32 3, i32 1
  store ptr %.unpack236, ptr %.repack237, align 8
  %42 = getelementptr %Sphere, ptr %29, i64 0, i32 1
  store double 1.000000e+00, ptr %42, align 8
  %43 = load ptr, ptr %29, align 8
  %44 = load ptr, ptr %43, align 8
  %45 = load i64, ptr @Sphere..type_descr, align 8
  %.not239 = icmp eq i64 %45, 0
  br i1 %.not239, label %46, label %49

46:                                               ; preds = %34
  %47 = alloca i64, align 8
  %48 = call i64 @GC_make_descriptor(ptr nonnull %47, i64 5)
  store i64 %48, ptr @Sphere..type_descr, align 8
  br label %49

49:                                               ; preds = %46, %34
  %50 = phi i64 [ %48, %46 ], [ %45, %34 ]
  %51 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %50)
  store ptr @Sphere..vtbl, ptr %51, align 8
  %52 = load i64, ptr @Vector..type_descr, align 8
  %.not240 = icmp eq i64 %52, 0
  br i1 %.not240, label %53, label %56

53:                                               ; preds = %49
  %54 = alloca i64, align 8
  %55 = call i64 @GC_make_descriptor(ptr nonnull %54, i64 4)
  store i64 %55, ptr @Vector..type_descr, align 8
  br label %56

56:                                               ; preds = %53, %49
  %57 = phi i64 [ %55, %53 ], [ %52, %49 ]
  %58 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %57)
  store ptr @Vector..vtbl, ptr %58, align 8
  %59 = getelementptr %Vector, ptr %58, i64 0, i32 1
  store double -6.000000e+00, ptr %59, align 8
  %60 = getelementptr %Vector, ptr %58, i64 0, i32 2
  store double 3.000000e+00, ptr %60, align 8
  %61 = getelementptr %Vector, ptr %58, i64 0, i32 3
  store double -5.000000e+00, ptr %61, align 8
  %.unpack241 = load ptr, ptr @Surfaces.shiny, align 8
  %.unpack242 = load ptr, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.shiny, i64 0, i32 1), align 8
  %62 = getelementptr %Sphere, ptr %51, i64 0, i32 2
  store ptr %58, ptr %62, align 8
  %63 = getelementptr %Sphere, ptr %51, i64 0, i32 3
  store ptr %.unpack241, ptr %63, align 8
  %.repack243 = getelementptr %Sphere, ptr %51, i64 0, i32 3, i32 1
  store ptr %.unpack242, ptr %.repack243, align 8
  %64 = getelementptr %Sphere, ptr %51, i64 0, i32 1
  store double 1.000000e+00, ptr %64, align 8
  %65 = load ptr, ptr %51, align 8
  %66 = load ptr, ptr %65, align 8
  %67 = load i64, ptr @Sphere..type_descr, align 8
  %.not245 = icmp eq i64 %67, 0
  br i1 %.not245, label %68, label %71

68:                                               ; preds = %56
  %69 = alloca i64, align 8
  %70 = call i64 @GC_make_descriptor(ptr nonnull %69, i64 5)
  store i64 %70, ptr @Sphere..type_descr, align 8
  br label %71

71:                                               ; preds = %68, %56
  %72 = phi i64 [ %70, %68 ], [ %67, %56 ]
  %73 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %72)
  store ptr @Sphere..vtbl, ptr %73, align 8
  %74 = load i64, ptr @Vector..type_descr, align 8
  %.not246 = icmp eq i64 %74, 0
  br i1 %.not246, label %75, label %78

75:                                               ; preds = %71
  %76 = alloca i64, align 8
  %77 = call i64 @GC_make_descriptor(ptr nonnull %76, i64 4)
  store i64 %77, ptr @Vector..type_descr, align 8
  br label %78

78:                                               ; preds = %75, %71
  %79 = phi i64 [ %77, %75 ], [ %74, %71 ]
  %80 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %79)
  store ptr @Vector..vtbl, ptr %80, align 8
  %81 = getelementptr %Vector, ptr %80, i64 0, i32 1
  store double -1.000000e+00, ptr %81, align 8
  %82 = getelementptr %Vector, ptr %80, i64 0, i32 2
  store double 5.000000e-01, ptr %82, align 8
  %83 = getelementptr %Vector, ptr %80, i64 0, i32 3
  store double 1.500000e+00, ptr %83, align 8
  %.unpack247 = load ptr, ptr @Surfaces.shiny, align 8
  %.unpack248 = load ptr, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.shiny, i64 0, i32 1), align 8
  %84 = getelementptr %Sphere, ptr %73, i64 0, i32 2
  store ptr %80, ptr %84, align 8
  %85 = getelementptr %Sphere, ptr %73, i64 0, i32 3
  store ptr %.unpack247, ptr %85, align 8
  %.repack249 = getelementptr %Sphere, ptr %73, i64 0, i32 3, i32 1
  store ptr %.unpack248, ptr %.repack249, align 8
  %86 = getelementptr %Sphere, ptr %73, i64 0, i32 1
  store double 2.500000e-01, ptr %86, align 8
  %87 = load ptr, ptr %73, align 8
  %88 = load ptr, ptr %87, align 8
  %89 = call ptr @GC_malloc(i64 64)
  store ptr %22, ptr %89, align 8
  %.repack251 = getelementptr inbounds { ptr, ptr }, ptr %89, i64 0, i32 1
  store ptr %7, ptr %.repack251, align 8
  %90 = getelementptr { ptr, ptr }, ptr %89, i64 1
  store ptr %44, ptr %90, align 8
  %.repack253 = getelementptr { ptr, ptr }, ptr %89, i64 1, i32 1
  store ptr %29, ptr %.repack253, align 8
  %91 = getelementptr { ptr, ptr }, ptr %89, i64 2
  store ptr %66, ptr %91, align 8
  %.repack255 = getelementptr { ptr, ptr }, ptr %89, i64 2, i32 1
  store ptr %51, ptr %.repack255, align 8
  %92 = getelementptr { ptr, ptr }, ptr %89, i64 3
  store ptr %88, ptr %92, align 8
  %.repack257 = getelementptr { ptr, ptr }, ptr %89, i64 3, i32 1
  store ptr %73, ptr %.repack257, align 8
  %93 = load i64, ptr @Vector..type_descr, align 8
  %.not259 = icmp eq i64 %93, 0
  br i1 %.not259, label %94, label %97

94:                                               ; preds = %78
  %95 = alloca i64, align 8
  %96 = call i64 @GC_make_descriptor(ptr nonnull %95, i64 4)
  store i64 %96, ptr @Vector..type_descr, align 8
  br label %97

97:                                               ; preds = %94, %78
  %98 = phi i64 [ %96, %94 ], [ %93, %78 ]
  %99 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %98)
  store ptr @Vector..vtbl, ptr %99, align 8
  %100 = getelementptr %Vector, ptr %99, i64 0, i32 1
  store double -2.000000e+00, ptr %100, align 8
  %101 = getelementptr %Vector, ptr %99, i64 0, i32 2
  store double 2.500000e+00, ptr %101, align 8
  %102 = getelementptr %Vector, ptr %99, i64 0, i32 3
  store double 0.000000e+00, ptr %102, align 8
  %103 = load i64, ptr @Color..type_descr, align 8
  %.not260 = icmp eq i64 %103, 0
  br i1 %.not260, label %104, label %107

104:                                              ; preds = %97
  %105 = alloca i64, align 8
  %106 = call i64 @GC_make_descriptor(ptr nonnull %105, i64 4)
  store i64 %106, ptr @Color..type_descr, align 8
  br label %107

107:                                              ; preds = %104, %97
  %108 = phi i64 [ %106, %104 ], [ %103, %97 ]
  %109 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %108)
  store ptr @Color..vtbl, ptr %109, align 8
  %110 = getelementptr %Color, ptr %109, i64 0, i32 1
  store double 4.900000e-01, ptr %110, align 8
  %111 = getelementptr %Color, ptr %109, i64 0, i32 2
  store double 7.000000e-02, ptr %111, align 8
  %112 = getelementptr %Color, ptr %109, i64 0, i32 3
  store double 7.000000e-02, ptr %112, align 8
  %113 = call ptr @GC_malloc(i64 16)
  store ptr %99, ptr %113, align 8
  %.repack261 = getelementptr inbounds { ptr, ptr }, ptr %113, i64 0, i32 1
  store ptr %109, ptr %.repack261, align 8
  %114 = load i64, ptr @Vector..type_descr, align 8
  %.not263 = icmp eq i64 %114, 0
  br i1 %.not263, label %115, label %118

115:                                              ; preds = %107
  %116 = alloca i64, align 8
  %117 = call i64 @GC_make_descriptor(ptr nonnull %116, i64 4)
  store i64 %117, ptr @Vector..type_descr, align 8
  br label %118

118:                                              ; preds = %115, %107
  %119 = phi i64 [ %117, %115 ], [ %114, %107 ]
  %120 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %119)
  store ptr @Vector..vtbl, ptr %120, align 8
  %121 = getelementptr %Vector, ptr %120, i64 0, i32 1
  store double 1.500000e+00, ptr %121, align 8
  %122 = getelementptr %Vector, ptr %120, i64 0, i32 2
  store double 2.500000e+00, ptr %122, align 8
  %123 = getelementptr %Vector, ptr %120, i64 0, i32 3
  store double 1.500000e+00, ptr %123, align 8
  %124 = load i64, ptr @Color..type_descr, align 8
  %.not264 = icmp eq i64 %124, 0
  br i1 %.not264, label %125, label %128

125:                                              ; preds = %118
  %126 = alloca i64, align 8
  %127 = call i64 @GC_make_descriptor(ptr nonnull %126, i64 4)
  store i64 %127, ptr @Color..type_descr, align 8
  br label %128

128:                                              ; preds = %125, %118
  %129 = phi i64 [ %127, %125 ], [ %124, %118 ]
  %130 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %129)
  store ptr @Color..vtbl, ptr %130, align 8
  %131 = getelementptr %Color, ptr %130, i64 0, i32 1
  store double 7.000000e-02, ptr %131, align 8
  %132 = getelementptr %Color, ptr %130, i64 0, i32 2
  store double 7.000000e-02, ptr %132, align 8
  %133 = getelementptr %Color, ptr %130, i64 0, i32 3
  store double 4.900000e-01, ptr %133, align 8
  %134 = call ptr @GC_malloc(i64 16)
  store ptr %120, ptr %134, align 8
  %.repack265 = getelementptr inbounds { ptr, ptr }, ptr %134, i64 0, i32 1
  store ptr %130, ptr %.repack265, align 8
  %135 = load i64, ptr @Vector..type_descr, align 8
  %.not267 = icmp eq i64 %135, 0
  br i1 %.not267, label %136, label %139

136:                                              ; preds = %128
  %137 = alloca i64, align 8
  %138 = call i64 @GC_make_descriptor(ptr nonnull %137, i64 4)
  store i64 %138, ptr @Vector..type_descr, align 8
  br label %139

139:                                              ; preds = %136, %128
  %140 = phi i64 [ %138, %136 ], [ %135, %128 ]
  %141 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %140)
  store ptr @Vector..vtbl, ptr %141, align 8
  %142 = getelementptr %Vector, ptr %141, i64 0, i32 1
  store double 1.500000e+00, ptr %142, align 8
  %143 = getelementptr %Vector, ptr %141, i64 0, i32 2
  store double 2.500000e+00, ptr %143, align 8
  %144 = getelementptr %Vector, ptr %141, i64 0, i32 3
  store double -1.500000e+00, ptr %144, align 8
  %145 = load i64, ptr @Color..type_descr, align 8
  %.not268 = icmp eq i64 %145, 0
  br i1 %.not268, label %146, label %149

146:                                              ; preds = %139
  %147 = alloca i64, align 8
  %148 = call i64 @GC_make_descriptor(ptr nonnull %147, i64 4)
  store i64 %148, ptr @Color..type_descr, align 8
  br label %149

149:                                              ; preds = %146, %139
  %150 = phi i64 [ %148, %146 ], [ %145, %139 ]
  %151 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %150)
  store ptr @Color..vtbl, ptr %151, align 8
  %152 = getelementptr %Color, ptr %151, i64 0, i32 1
  store double 7.000000e-02, ptr %152, align 8
  %153 = getelementptr %Color, ptr %151, i64 0, i32 2
  store double 4.900000e-01, ptr %153, align 8
  %154 = getelementptr %Color, ptr %151, i64 0, i32 3
  store double 0x3FB22D0E56041893, ptr %154, align 8
  %155 = call ptr @GC_malloc(i64 16)
  store ptr %141, ptr %155, align 8
  %.repack269 = getelementptr inbounds { ptr, ptr }, ptr %155, i64 0, i32 1
  store ptr %151, ptr %.repack269, align 8
  %156 = load i64, ptr @Vector..type_descr, align 8
  %.not271 = icmp eq i64 %156, 0
  br i1 %.not271, label %157, label %160

157:                                              ; preds = %149
  %158 = alloca i64, align 8
  %159 = call i64 @GC_make_descriptor(ptr nonnull %158, i64 4)
  store i64 %159, ptr @Vector..type_descr, align 8
  br label %160

160:                                              ; preds = %157, %149
  %161 = phi i64 [ %159, %157 ], [ %156, %149 ]
  %162 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %161)
  store ptr @Vector..vtbl, ptr %162, align 8
  %163 = getelementptr %Vector, ptr %162, i64 0, i32 1
  store double 0.000000e+00, ptr %163, align 8
  %164 = getelementptr %Vector, ptr %162, i64 0, i32 2
  store double 3.500000e+00, ptr %164, align 8
  %165 = getelementptr %Vector, ptr %162, i64 0, i32 3
  store double 0.000000e+00, ptr %165, align 8
  %166 = load i64, ptr @Color..type_descr, align 8
  %.not272 = icmp eq i64 %166, 0
  br i1 %.not272, label %167, label %170

167:                                              ; preds = %160
  %168 = alloca i64, align 8
  %169 = call i64 @GC_make_descriptor(ptr nonnull %168, i64 4)
  store i64 %169, ptr @Color..type_descr, align 8
  br label %170

170:                                              ; preds = %167, %160
  %171 = phi i64 [ %169, %167 ], [ %166, %160 ]
  %172 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %171)
  store ptr @Color..vtbl, ptr %172, align 8
  %173 = getelementptr %Color, ptr %172, i64 0, i32 1
  store double 2.100000e-01, ptr %173, align 8
  %174 = getelementptr %Color, ptr %172, i64 0, i32 2
  store double 2.100000e-01, ptr %174, align 8
  %175 = getelementptr %Color, ptr %172, i64 0, i32 3
  store double 3.500000e-01, ptr %175, align 8
  %176 = call ptr @GC_malloc(i64 16)
  store ptr %162, ptr %176, align 8
  %.repack273 = getelementptr inbounds { ptr, ptr }, ptr %176, i64 0, i32 1
  store ptr %172, ptr %.repack273, align 8
  %177 = call ptr @GC_malloc(i64 64)
  store ptr @Light.14713958..vtbl, ptr %177, align 8
  %.repack275 = getelementptr inbounds { ptr, ptr }, ptr %177, i64 0, i32 1
  store ptr %113, ptr %.repack275, align 8
  %178 = getelementptr { ptr, ptr }, ptr %177, i64 1
  store ptr @Light.14713958..vtbl, ptr %178, align 8
  %.repack277 = getelementptr { ptr, ptr }, ptr %177, i64 1, i32 1
  store ptr %134, ptr %.repack277, align 8
  %179 = getelementptr { ptr, ptr }, ptr %177, i64 2
  store ptr @Light.14713958..vtbl, ptr %179, align 8
  %.repack279 = getelementptr { ptr, ptr }, ptr %177, i64 2, i32 1
  store ptr %155, ptr %.repack279, align 8
  %180 = getelementptr { ptr, ptr }, ptr %177, i64 3
  store ptr @Light.14713958..vtbl, ptr %180, align 8
  %.repack281 = getelementptr { ptr, ptr }, ptr %177, i64 3, i32 1
  store ptr %176, ptr %.repack281, align 8
  %181 = load i64, ptr @Camera..type_descr, align 8
  %.not283 = icmp eq i64 %181, 0
  br i1 %.not283, label %182, label %185

182:                                              ; preds = %170
  %183 = alloca i64, align 8
  %184 = call i64 @GC_make_descriptor(ptr nonnull %183, i64 5)
  store i64 %184, ptr @Camera..type_descr, align 8
  br label %185

185:                                              ; preds = %182, %170
  %186 = phi i64 [ %184, %182 ], [ %181, %170 ]
  %187 = call ptr @GC_malloc_explicitly_typed(i64 40, i64 %186)
  store ptr @Camera..vtbl, ptr %187, align 8
  %188 = load i64, ptr @Vector..type_descr, align 8
  %.not284 = icmp eq i64 %188, 0
  br i1 %.not284, label %189, label %192

189:                                              ; preds = %185
  %190 = alloca i64, align 8
  %191 = call i64 @GC_make_descriptor(ptr nonnull %190, i64 4)
  store i64 %191, ptr @Vector..type_descr, align 8
  br label %192

192:                                              ; preds = %189, %185
  %193 = phi i64 [ %191, %189 ], [ %188, %185 ]
  %194 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %193)
  store ptr @Vector..vtbl, ptr %194, align 8
  %195 = getelementptr %Vector, ptr %194, i64 0, i32 1
  store double 3.000000e+00, ptr %195, align 8
  %196 = getelementptr %Vector, ptr %194, i64 0, i32 2
  store double 2.000000e+00, ptr %196, align 8
  %197 = getelementptr %Vector, ptr %194, i64 0, i32 3
  store double 4.000000e+00, ptr %197, align 8
  %198 = load i64, ptr @Vector..type_descr, align 8
  %.not285 = icmp eq i64 %198, 0
  br i1 %.not285, label %199, label %202

199:                                              ; preds = %192
  %200 = alloca i64, align 8
  %201 = call i64 @GC_make_descriptor(ptr nonnull %200, i64 4)
  store i64 %201, ptr @Vector..type_descr, align 8
  br label %202

202:                                              ; preds = %199, %192
  %203 = phi i64 [ %201, %199 ], [ %198, %192 ]
  %204 = call ptr @GC_malloc_explicitly_typed(i64 32, i64 %203)
  store ptr @Vector..vtbl, ptr %204, align 8
  %205 = getelementptr %Vector, ptr %204, i64 0, i32 1
  store double -1.000000e+00, ptr %205, align 8
  %206 = getelementptr %Vector, ptr %204, i64 0, i32 2
  store double 5.000000e-01, ptr %206, align 8
  %207 = getelementptr %Vector, ptr %204, i64 0, i32 3
  store double 0.000000e+00, ptr %207, align 8
  call void @Camera.constructor(ptr nonnull %187, ptr nonnull %194, ptr nonnull %204)
  %208 = call ptr @GC_malloc(i64 40)
  %209 = insertvalue { ptr, i32 } poison, ptr %89, 0
  %.fca.2.insert.elt = insertvalue { ptr, i32 } %209, i32 4, 1
  store { ptr, i32 } %.fca.2.insert.elt, ptr %208, align 8
  %.repack286 = getelementptr inbounds { { ptr, i32 }, { ptr, i32 }, ptr }, ptr %208, i64 0, i32 1
  %210 = insertvalue { ptr, i32 } poison, ptr %177, 0
  %.fca.2.insert.elt287 = insertvalue { ptr, i32 } %210, i32 4, 1
  store { ptr, i32 } %.fca.2.insert.elt287, ptr %.repack286, align 8
  %.repack288 = getelementptr inbounds { { ptr, i32 }, { ptr, i32 }, ptr }, ptr %208, i64 0, i32 2
  store ptr %187, ptr %.repack288, align 8
  %.fca.1.insert = insertvalue { ptr, ptr } { ptr @Scene.14713952..vtbl, ptr undef }, ptr %208, 1
  ret { ptr, ptr } %.fca.1.insert
}

define void @exec() local_unnamed_addr {
  %1 = load i64, ptr @RayTracer..type_descr, align 8
  %.not = icmp eq i64 %1, 0
  br i1 %.not, label %2, label %5

2:                                                ; preds = %0
  %3 = alloca i64, align 8
  %4 = call i64 @GC_make_descriptor(ptr nonnull %3, i64 2)
  store i64 %4, ptr @RayTracer..type_descr, align 8
  br label %5

5:                                                ; preds = %2, %0
  %6 = phi i64 [ %4, %2 ], [ %1, %0 ]
  %7 = call ptr @GC_malloc_explicitly_typed(i64 16, i64 %6)
  store ptr @RayTracer..vtbl, ptr %7, align 8
  %8 = getelementptr %RayTracer, ptr %7, i64 0, i32 1
  store i32 5, ptr %8, align 4
  %9 = load ptr, ptr getelementptr inbounds ({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr @RayTracer..vtbl, i64 0, i32 8), align 16
  %10 = call { ptr, ptr } @defaultScene()
  call void %9(ptr nonnull %7, { ptr, ptr } %10, double 2.560000e+02, double 2.560000e+02)
  ret void
}

define void @main() local_unnamed_addr {
  tail call void @GC_init()
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @s_17165743417360025467)
  tail call void @exec()
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @s_6682479467004374669)
  ret void
}

define void @__mlir_gctors() {
  tail call void @Color.static_constructor()
  %1 = tail call ptr @GC_malloc(i64 32)
  store double 1.500000e+02, ptr %1, align 8
  %.repack1.i = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 1
  store ptr @Surfaces..feL165C18L171C9H15086251, ptr %.repack1.i, align 8
  %.repack2.i = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 2
  store ptr @Surfaces..feL172C19L172C64H14759908, ptr %.repack2.i, align 8
  %.repack3.i = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %1, i64 0, i32 3
  store ptr @Surfaces..feL173C18L179C9H14757289, ptr %.repack3.i, align 8
  store ptr @Surface.14728117..vtbl, ptr @Surfaces.checkerboard, align 8
  store ptr %1, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.checkerboard, i64 0, i32 1), align 8
  %2 = tail call ptr @GC_malloc(i64 32)
  store double 2.500000e+02, ptr %2, align 8
  %.repack1.i1 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %2, i64 0, i32 1
  store ptr @Surfaces..feL159C18L159C63H14774515, ptr %.repack1.i1, align 8
  %.repack2.i2 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %2, i64 0, i32 2
  store ptr @Surfaces..feL160C19L160C63H14867906, ptr %.repack2.i2, align 8
  %.repack3.i3 = getelementptr inbounds { double, ptr, ptr, ptr }, ptr %2, i64 0, i32 3
  store ptr @Surfaces..feL161C18L161C55H14867908, ptr %.repack3.i3, align 8
  store ptr @Surface.14796823..vtbl, ptr @Surfaces.shiny, align 8
  store ptr %2, ptr getelementptr inbounds ({ ptr, ptr }, ptr @Surfaces.shiny, i64 0, i32 1), align 8
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.floor.f64(double) #14

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #15

attributes #0 = { nofree nounwind }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(write) }
attributes #4 = { mustprogress nofree nounwind willreturn memory(read, inaccessiblemem: none) }
attributes #5 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) }
attributes #7 = { mustprogress nofree nounwind willreturn memory(write, argmem: readwrite) }
attributes #8 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #9 = { mustprogress nofree nosync nounwind willreturn memory(argmem: read) }
attributes #10 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) }
attributes #11 = { mustprogress nofree nosync nounwind willreturn memory(read, inaccessiblemem: none) }
attributes #12 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) }
attributes #13 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #14 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #15 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unswitch.partial.disable"}

