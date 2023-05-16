; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@frmt_555400739678143724 = internal constant [3 x i8] c"%d\00"

declare void @GC_init() local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #0

declare ptr @GC_malloc(i64) local_unnamed_addr

declare i32 @sprintf_s(ptr, i32, ptr, ...) local_unnamed_addr

define void @.f_main..afL8C23L8C50H261924014(i32 %0, { double, i1 } %1, { { ptr, i32 }, i1 } %2) local_unnamed_addr {
  %4 = tail call ptr @GC_malloc(i64 50)
  %5 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %4, i32 50, ptr nonnull @frmt_555400739678143724, i32 %0)
  %6 = tail call i32 @puts(ptr nonnull dereferenceable(1) %4)
  ret void
}

define void @main() local_unnamed_addr {
  tail call void @GC_init()
  %1 = tail call ptr @GC_malloc(i64 12)
  store i32 1, ptr %1, align 4
  %.repack29 = getelementptr [3 x i32], ptr %1, i64 0, i64 1
  store i32 2, ptr %.repack29, align 4
  %.repack30 = getelementptr [3 x i32], ptr %1, i64 0, i64 2
  store i32 3, ptr %.repack30, align 4
  %2 = tail call ptr @GC_malloc(i64 50)
  %3 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %2, i32 50, ptr nonnull @frmt_555400739678143724, i32 1)
  %4 = tail call i32 @puts(ptr nonnull dereferenceable(1) %2)
  %5 = load i32, ptr %.repack29, align 4
  %6 = tail call ptr @GC_malloc(i64 50)
  %7 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %6, i32 50, ptr nonnull @frmt_555400739678143724, i32 %5)
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) %6)
  %9 = load i32, ptr %.repack30, align 4
  %10 = tail call ptr @GC_malloc(i64 50)
  %11 = tail call i32 (ptr, i32, ptr, ...) @sprintf_s(ptr %10, i32 50, ptr nonnull @frmt_555400739678143724, i32 %9)
  %12 = tail call i32 @puts(ptr nonnull dereferenceable(1) %10)
  ret void
}

attributes #0 = { nofree nounwind }

