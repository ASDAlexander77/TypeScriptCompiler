; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%.f_gen..objL5C1FH3041951825 = type { i32, ptr }
%Res = type { ptr }

$Res..size = comdat any

@s_11525218432948371097 = internal constant [12 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFRes\00", align 8
@s_16124472420822520262 = internal constant [17 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdisposed\00", align 8
@s_6601085983368743140 = internal constant [13 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFnull\00", align 8
@s_6682479467004374669 = internal constant [14 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFdone.\00", align 8
@m_3276509499301812911 = internal constant [51 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan unbraced else that iterates a generator\00", align 8
@f_8697359278555959997 = internal constant [113 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFI:\\TypeScriptCompiler\\.claude\\worktrees\\own-items-60\\tslang\\test\\tester\\tests\\00unbraced_branch_scope.ts\00", align 8
@m_10050024527826428597 = internal constant [31 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFthe same inside a loop\00", align 8
@m_2241045997123555583 = internal constant [25 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan unbraced then\00", align 8
@m_886492191233840080 = internal constant [38 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FFan unbraced if as a loop body\00", align 8
@Res..rtti = private global ptr getelementptr inbounds nuw (i8, ptr @s_11525218432948371097, i64 8)
@Res..size = global i64 8, comdat
@Res..vtbl = private global { ptr, ptr, ptr, ptr, ptr } { ptr @Res..instanceOf, ptr @Res..new, ptr @Res.dispose, ptr @Res..rtti, ptr @Res..size }

declare void @GC_enable_threads()

declare void @GC_init()

; Function Attrs: allockind("alloc,zeroed") memory(write, argmem: none, inaccessiblemem: readwrite, errnomem: none, target_mem0: none, target_mem1: none)
declare noalias ptr @GC_malloc(i64) #0

declare void @__assert_fail(ptr, ptr, i32, ptr)

declare i32 @fflush(ptr)

declare void @puts(ptr)

declare i32 @strcmp(ptr, ptr)

; Function Attrs: mustprogress
define { i32, i1 } @.f_gen..f_gen..objL5C1FH3041951825.next(ptr %0) #1 {
  %2 = alloca { i32, i1 }, align 8
  %3 = alloca { i32, i1 }, align 8
  %4 = alloca { i32, i1 }, align 8
  %5 = alloca { i32, i1 }, align 8
  %6 = alloca ptr, align 8
  %7 = alloca { i32, i1 }, align 8
  store ptr %0, ptr %6, align 8
  %8 = load ptr, ptr %6, align 8
  %9 = getelementptr %.f_gen..objL5C1FH3041951825, ptr %8, i32 0, i32 0
  %10 = load i32, ptr %9, align 4
  %11 = icmp slt i32 %10, 0
  br i1 %11, label %12, label %16

12:                                               ; preds = %1
  %13 = getelementptr { i32, i1 }, ptr %5, i32 0, i32 0
  store i32 0, ptr %13, align 4
  %14 = getelementptr { i32, i1 }, ptr %5, i32 0, i32 1
  store i1 true, ptr %14, align 1
  %15 = load { i32, i1 }, ptr %5, align 4
  store { i32, i1 } %15, ptr %7, align 4
  br label %36

16:                                               ; preds = %1
  %17 = load ptr, ptr %6, align 8
  %18 = getelementptr %.f_gen..objL5C1FH3041951825, ptr %17, i32 0, i32 0
  %19 = load i32, ptr %18, align 4
  switch i32 %19, label %36 [
    i32 0, label %20
    i32 1, label %26
    i32 2, label %32
  ]

20:                                               ; preds = %16
  %21 = load ptr, ptr %6, align 8
  %22 = getelementptr %.f_gen..objL5C1FH3041951825, ptr %21, i32 0, i32 0
  store i32 1, ptr %22, align 4
  %23 = getelementptr { i32, i1 }, ptr %4, i32 0, i32 0
  store i32 1, ptr %23, align 4
  %24 = getelementptr { i32, i1 }, ptr %4, i32 0, i32 1
  store i1 false, ptr %24, align 1
  %25 = load { i32, i1 }, ptr %4, align 4
  store { i32, i1 } %25, ptr %7, align 4
  br label %36

26:                                               ; preds = %16
  %27 = load ptr, ptr %6, align 8
  %28 = getelementptr %.f_gen..objL5C1FH3041951825, ptr %27, i32 0, i32 0
  store i32 2, ptr %28, align 4
  %29 = getelementptr { i32, i1 }, ptr %3, i32 0, i32 0
  store i32 2, ptr %29, align 4
  %30 = getelementptr { i32, i1 }, ptr %3, i32 0, i32 1
  store i1 false, ptr %30, align 1
  %31 = load { i32, i1 }, ptr %3, align 4
  store { i32, i1 } %31, ptr %7, align 4
  br label %36

32:                                               ; preds = %16
  %33 = getelementptr { i32, i1 }, ptr %2, i32 0, i32 0
  store i32 0, ptr %33, align 4
  %34 = getelementptr { i32, i1 }, ptr %2, i32 0, i32 1
  store i1 true, ptr %34, align 1
  %35 = load { i32, i1 }, ptr %2, align 4
  store { i32, i1 } %35, ptr %7, align 4
  br label %36

36:                                               ; preds = %12, %20, %26, %32, %16
  %37 = load { i32, i1 }, ptr %7, align 4
  ret { i32, i1 } %37
}

; Function Attrs: mustprogress
define ptr @gen() #1 {
  %1 = alloca { i32, ptr }, align 8
  %2 = alloca ptr, align 8
  store { i32, ptr } { i32 0, ptr @.f_gen..f_gen..objL5C1FH3041951825.next }, ptr %1, align 8
  %3 = load { i32, ptr }, ptr %1, align 8
  %4 = call ptr @GC_malloc(i64 24)
  %5 = getelementptr i8, ptr %4, i64 8
  store { i32, ptr } %3, ptr %5, align 8
  store ptr %5, ptr %2, align 8
  %6 = load ptr, ptr %2, align 8
  ret ptr %6
}

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
  %2 = call ptr @GC_malloc(i64 16)
  %3 = getelementptr i8, ptr %2, i64 8
  %4 = getelementptr %Res, ptr %3, i32 0, i32 0
  store ptr @Res..vtbl, ptr %4, align 8
  store ptr %3, ptr %1, align 8
  %5 = load ptr, ptr %1, align 8
  ret ptr %5
}

; Function Attrs: mustprogress
define void @Res.dispose(ptr %0) #1 {
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_16124472420822520262, i64 8))
  ret void
}

; Function Attrs: mustprogress
define i32 @main() #1 {
  call void @GC_init()
  call void @GC_enable_threads()
  %1 = alloca { i32, i1 }, align 8
  %2 = alloca ptr, align 8
  %3 = alloca { i32, i1 }, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca { i32, i1 }, align 8
  %8 = alloca ptr, align 8
  %9 = alloca { i32, i1 }, align 8
  %10 = alloca ptr, align 8
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  store i32 0, ptr %12, align 4
  %13 = call ptr @gen()
  br i1 false, label %51, label %14

14:                                               ; preds = %0
  store ptr %13, ptr %2, align 8
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr { i32, ptr }, ptr %15, i32 0, i32 1
  %17 = insertvalue { ptr, ptr } zeroinitializer, ptr %16, 0
  %18 = insertvalue { ptr, ptr } %17, ptr %15, 1
  %19 = extractvalue { ptr, ptr } %18, 1
  %20 = extractvalue { ptr, ptr } %18, 0
  %21 = load ptr, ptr %20, align 8
  %22 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %21, 0
  %23 = insertvalue { ptr, ptr, ptr } %22, ptr %19, 1
  %24 = insertvalue { ptr, ptr, ptr } %23, ptr null, 2
  %25 = extractvalue { ptr, ptr, ptr } %24, 1
  %26 = extractvalue { ptr, ptr, ptr } %24, 0
  %27 = call { i32, i1 } %26(ptr %25)
  store { i32, i1 } %27, ptr %1, align 4
  br label %28

28:                                               ; preds = %32, %14
  %29 = getelementptr { i32, i1 }, ptr %1, i32 0, i32 1
  %30 = load i1, ptr %29, align 1
  %31 = xor i1 %30, true
  br i1 %31, label %32, label %50

32:                                               ; preds = %28
  %33 = getelementptr { i32, i1 }, ptr %1, i32 0, i32 0
  %34 = load i32, ptr %33, align 4
  %35 = load i32, ptr %12, align 4
  %36 = add i32 %35, %34
  store i32 %36, ptr %12, align 4
  %37 = load ptr, ptr %2, align 8
  %38 = getelementptr { i32, ptr }, ptr %37, i32 0, i32 1
  %39 = insertvalue { ptr, ptr } zeroinitializer, ptr %38, 0
  %40 = insertvalue { ptr, ptr } %39, ptr %37, 1
  %41 = extractvalue { ptr, ptr } %40, 1
  %42 = extractvalue { ptr, ptr } %40, 0
  %43 = load ptr, ptr %42, align 8
  %44 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %43, 0
  %45 = insertvalue { ptr, ptr, ptr } %44, ptr %41, 1
  %46 = insertvalue { ptr, ptr, ptr } %45, ptr null, 2
  %47 = extractvalue { ptr, ptr, ptr } %46, 1
  %48 = extractvalue { ptr, ptr, ptr } %46, 0
  %49 = call { i32, i1 } %48(ptr %47)
  store { i32, i1 } %49, ptr %1, align 4
  br label %28

50:                                               ; preds = %28
  br label %51

51:                                               ; preds = %50, %0
  %52 = load i32, ptr %12, align 4
  %53 = icmp eq i32 %52, 3
  br i1 %53, label %54, label %205

54:                                               ; preds = %51
  store i32 0, ptr %11, align 4
  br label %55

55:                                               ; preds = %102, %54
  %56 = load i32, ptr %11, align 4
  %57 = icmp slt i32 %56, 2
  br i1 %57, label %58, label %105

58:                                               ; preds = %55
  %59 = load i32, ptr %11, align 4
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %61, label %64

61:                                               ; preds = %58
  %62 = load i32, ptr %12, align 4
  %63 = add i32 %62, 10
  store i32 %63, ptr %12, align 4
  br label %102

64:                                               ; preds = %58
  %65 = call ptr @gen()
  store ptr %65, ptr %10, align 8
  %66 = load ptr, ptr %10, align 8
  %67 = getelementptr { i32, ptr }, ptr %66, i32 0, i32 1
  %68 = insertvalue { ptr, ptr } zeroinitializer, ptr %67, 0
  %69 = insertvalue { ptr, ptr } %68, ptr %66, 1
  %70 = extractvalue { ptr, ptr } %69, 1
  %71 = extractvalue { ptr, ptr } %69, 0
  %72 = load ptr, ptr %71, align 8
  %73 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %72, 0
  %74 = insertvalue { ptr, ptr, ptr } %73, ptr %70, 1
  %75 = insertvalue { ptr, ptr, ptr } %74, ptr null, 2
  %76 = extractvalue { ptr, ptr, ptr } %75, 1
  %77 = extractvalue { ptr, ptr, ptr } %75, 0
  %78 = call { i32, i1 } %77(ptr %76)
  store { i32, i1 } %78, ptr %9, align 4
  br label %79

79:                                               ; preds = %83, %64
  %80 = getelementptr { i32, i1 }, ptr %9, i32 0, i32 1
  %81 = load i1, ptr %80, align 1
  %82 = xor i1 %81, true
  br i1 %82, label %83, label %101

83:                                               ; preds = %79
  %84 = getelementptr { i32, i1 }, ptr %9, i32 0, i32 0
  %85 = load i32, ptr %84, align 4
  %86 = load i32, ptr %12, align 4
  %87 = add i32 %86, %85
  store i32 %87, ptr %12, align 4
  %88 = load ptr, ptr %10, align 8
  %89 = getelementptr { i32, ptr }, ptr %88, i32 0, i32 1
  %90 = insertvalue { ptr, ptr } zeroinitializer, ptr %89, 0
  %91 = insertvalue { ptr, ptr } %90, ptr %88, 1
  %92 = extractvalue { ptr, ptr } %91, 1
  %93 = extractvalue { ptr, ptr } %91, 0
  %94 = load ptr, ptr %93, align 8
  %95 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %94, 0
  %96 = insertvalue { ptr, ptr, ptr } %95, ptr %92, 1
  %97 = insertvalue { ptr, ptr, ptr } %96, ptr null, 2
  %98 = extractvalue { ptr, ptr, ptr } %97, 1
  %99 = extractvalue { ptr, ptr, ptr } %97, 0
  %100 = call { i32, i1 } %99(ptr %98)
  store { i32, i1 } %100, ptr %9, align 4
  br label %79

101:                                              ; preds = %79
  br label %102

102:                                              ; preds = %61, %101
  %103 = load i32, ptr %11, align 4
  %104 = add i32 %103, 1
  store i32 %104, ptr %11, align 4
  br label %55

105:                                              ; preds = %55
  %106 = load i32, ptr %12, align 4
  %107 = icmp eq i32 %106, 16
  br i1 %107, label %108, label %207

108:                                              ; preds = %105
  %109 = load i32, ptr %12, align 4
  %110 = icmp sgt i32 %109, 0
  br i1 %110, label %111, label %149

111:                                              ; preds = %108
  %112 = call ptr @gen()
  store ptr %112, ptr %8, align 8
  %113 = load ptr, ptr %8, align 8
  %114 = getelementptr { i32, ptr }, ptr %113, i32 0, i32 1
  %115 = insertvalue { ptr, ptr } zeroinitializer, ptr %114, 0
  %116 = insertvalue { ptr, ptr } %115, ptr %113, 1
  %117 = extractvalue { ptr, ptr } %116, 1
  %118 = extractvalue { ptr, ptr } %116, 0
  %119 = load ptr, ptr %118, align 8
  %120 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %119, 0
  %121 = insertvalue { ptr, ptr, ptr } %120, ptr %117, 1
  %122 = insertvalue { ptr, ptr, ptr } %121, ptr null, 2
  %123 = extractvalue { ptr, ptr, ptr } %122, 1
  %124 = extractvalue { ptr, ptr, ptr } %122, 0
  %125 = call { i32, i1 } %124(ptr %123)
  store { i32, i1 } %125, ptr %7, align 4
  br label %126

126:                                              ; preds = %130, %111
  %127 = getelementptr { i32, i1 }, ptr %7, i32 0, i32 1
  %128 = load i1, ptr %127, align 1
  %129 = xor i1 %128, true
  br i1 %129, label %130, label %148

130:                                              ; preds = %126
  %131 = getelementptr { i32, i1 }, ptr %7, i32 0, i32 0
  %132 = load i32, ptr %131, align 4
  %133 = load i32, ptr %12, align 4
  %134 = add i32 %133, %132
  store i32 %134, ptr %12, align 4
  %135 = load ptr, ptr %8, align 8
  %136 = getelementptr { i32, ptr }, ptr %135, i32 0, i32 1
  %137 = insertvalue { ptr, ptr } zeroinitializer, ptr %136, 0
  %138 = insertvalue { ptr, ptr } %137, ptr %135, 1
  %139 = extractvalue { ptr, ptr } %138, 1
  %140 = extractvalue { ptr, ptr } %138, 0
  %141 = load ptr, ptr %140, align 8
  %142 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %141, 0
  %143 = insertvalue { ptr, ptr, ptr } %142, ptr %139, 1
  %144 = insertvalue { ptr, ptr, ptr } %143, ptr null, 2
  %145 = extractvalue { ptr, ptr, ptr } %144, 1
  %146 = extractvalue { ptr, ptr, ptr } %144, 0
  %147 = call { i32, i1 } %146(ptr %145)
  store { i32, i1 } %147, ptr %7, align 4
  br label %126

148:                                              ; preds = %126
  br label %149

149:                                              ; preds = %148, %108
  %150 = load i32, ptr %12, align 4
  %151 = icmp eq i32 %150, 19
  br i1 %151, label %152, label %209

152:                                              ; preds = %149
  store i32 0, ptr %6, align 4
  store i32 0, ptr %5, align 4
  br label %153

153:                                              ; preds = %197, %152
  %154 = load i32, ptr %5, align 4
  %155 = icmp slt i32 %154, 3
  br i1 %155, label %156, label %200

156:                                              ; preds = %153
  %157 = load i32, ptr %5, align 4
  %158 = icmp ne i32 %157, 1
  br i1 %158, label %159, label %197

159:                                              ; preds = %156
  %160 = call ptr @gen()
  store ptr %160, ptr %4, align 8
  %161 = load ptr, ptr %4, align 8
  %162 = getelementptr { i32, ptr }, ptr %161, i32 0, i32 1
  %163 = insertvalue { ptr, ptr } zeroinitializer, ptr %162, 0
  %164 = insertvalue { ptr, ptr } %163, ptr %161, 1
  %165 = extractvalue { ptr, ptr } %164, 1
  %166 = extractvalue { ptr, ptr } %164, 0
  %167 = load ptr, ptr %166, align 8
  %168 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %167, 0
  %169 = insertvalue { ptr, ptr, ptr } %168, ptr %165, 1
  %170 = insertvalue { ptr, ptr, ptr } %169, ptr null, 2
  %171 = extractvalue { ptr, ptr, ptr } %170, 1
  %172 = extractvalue { ptr, ptr, ptr } %170, 0
  %173 = call { i32, i1 } %172(ptr %171)
  store { i32, i1 } %173, ptr %3, align 4
  br label %174

174:                                              ; preds = %178, %159
  %175 = getelementptr { i32, i1 }, ptr %3, i32 0, i32 1
  %176 = load i1, ptr %175, align 1
  %177 = xor i1 %176, true
  br i1 %177, label %178, label %196

178:                                              ; preds = %174
  %179 = getelementptr { i32, i1 }, ptr %3, i32 0, i32 0
  %180 = load i32, ptr %179, align 4
  %181 = load i32, ptr %6, align 4
  %182 = add i32 %181, %180
  store i32 %182, ptr %6, align 4
  %183 = load ptr, ptr %4, align 8
  %184 = getelementptr { i32, ptr }, ptr %183, i32 0, i32 1
  %185 = insertvalue { ptr, ptr } zeroinitializer, ptr %184, 0
  %186 = insertvalue { ptr, ptr } %185, ptr %183, 1
  %187 = extractvalue { ptr, ptr } %186, 1
  %188 = extractvalue { ptr, ptr } %186, 0
  %189 = load ptr, ptr %188, align 8
  %190 = insertvalue { ptr, ptr, ptr } zeroinitializer, ptr %189, 0
  %191 = insertvalue { ptr, ptr, ptr } %190, ptr %187, 1
  %192 = insertvalue { ptr, ptr, ptr } %191, ptr null, 2
  %193 = extractvalue { ptr, ptr, ptr } %192, 1
  %194 = extractvalue { ptr, ptr, ptr } %192, 0
  %195 = call { i32, i1 } %194(ptr %193)
  store { i32, i1 } %195, ptr %3, align 4
  br label %174

196:                                              ; preds = %174
  br label %197

197:                                              ; preds = %196, %156
  %198 = load i32, ptr %5, align 4
  %199 = add i32 %198, 1
  store i32 %199, ptr %5, align 4
  br label %153

200:                                              ; preds = %153
  %201 = load i32, ptr %6, align 4
  %202 = icmp eq i32 %201, 6
  br i1 %202, label %203, label %211

203:                                              ; preds = %200
  call void @puts(ptr getelementptr inbounds nuw (i8, ptr @s_6682479467004374669, i64 8))
  ret i32 0

204:                                              ; preds = %211, %209, %207, %205
  unreachable

205:                                              ; preds = %51
  %206 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_3276509499301812911, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8697359278555959997, i64 8), i32 20, ptr null)
  br label %204

207:                                              ; preds = %105
  %208 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_10050024527826428597, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8697359278555959997, i64 8), i32 26, ptr null)
  br label %204

209:                                              ; preds = %149
  %210 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_2241045997123555583, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8697359278555959997, i64 8), i32 29, ptr null)
  br label %204

211:                                              ; preds = %200
  %212 = call i32 @fflush(ptr null)
  call void @__assert_fail(ptr getelementptr inbounds nuw (i8, ptr @m_886492191233840080, i64 8), ptr getelementptr inbounds nuw (i8, ptr @f_8697359278555959997, i64 8), i32 33, ptr null)
  br label %204
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
attributes #1 = { mustprogress }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

